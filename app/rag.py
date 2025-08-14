"""
Retrieval-Augmented Generation (RAG) engine.

Coordinates vector retrieval, agent memory recall, prompt assembly,
and LLM generation for construction claims analysis.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import re

from .logging_conf import get_logger
from .vectors import VectorStore, SearchResult
from .matters import Matter
from .models import KnowledgeItem
from .llm.ollama_provider import OllamaProvider
from .llm.base import LLMProvider
from .prompts import (
    SYSTEM_PROMPT, 
    assemble_rag_prompt,
    extract_citations_from_answer,
    validate_citations,
    INFORMATION_EXTRACTION_PROMPT,
    FOLLOWUP_SUGGESTIONS_PROMPT
)

logger = get_logger(__name__)


@dataclass
class SourceChunk:
    """Source chunk for UI display with citation information."""
    doc: str
    page_start: int
    page_end: int
    text: str
    score: float



@dataclass
class RAGResponse:
    """Complete response from RAG pipeline."""
    answer: str
    sources: List[SourceChunk]
    followups: List[str]
    used_memory: List[KnowledgeItem]


class RAGEngine:
    """Main RAG engine coordinating retrieval and generation."""
    
    def __init__(
        self, 
        matter: Matter, 
        llm_provider: Optional[LLMProvider] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.matter = matter
        
        # Initialize vector store for the matter
        if vector_store is None:
            self.vector_store = VectorStore(matter.paths.root)
        else:
            self.vector_store = vector_store
        
        # Initialize LLM provider (default to Ollama)
        if llm_provider is None:
            self.llm_provider = OllamaProvider(model=matter.generation_model)
        else:
            self.llm_provider = llm_provider
        
        # TODO: Initialize Letta adapter when Sprint 5 is implemented
        self.letta_adapter = None
        
        logger.info(
            "RAG engine initialized",
            matter_id=matter.id,
            matter_name=matter.name,
            generation_model=matter.generation_model,
            embedding_model=matter.embedding_model
        )
    
    async def generate_answer(
        self,
        query: str,
        k: int = 8,
        k_memory: int = 6,
        max_tokens: int = 900,
        temperature: float = 0.2
    ) -> RAGResponse:
        """
        Generate answer using RAG pipeline.
        
        Args:
            query: User query
            k: Number of vector search results to retrieve
            k_memory: Number of memory items to recall
            max_tokens: Maximum tokens for generation
            temperature: Generation temperature
            
        Returns:
            RAGResponse with answer, sources, and follow-ups
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(
            "Starting RAG answer generation",
            matter_id=self.matter.id,
            query_preview=query[:100],
            k=k,
            k_memory=k_memory
        )
        
        try:
            # Step 1: Vector retrieval
            logger.debug("Performing vector search")
            search_results = await self.vector_store.search(query, k=k)
            
            logger.debug(
                "Vector search completed",
                results_found=len(search_results),
                avg_similarity=sum(r.similarity_score for r in search_results) / len(search_results) if search_results else 0
            )
            
            # Step 2: Memory recall (placeholder for Sprint 5)
            memory_items = []
            if self.letta_adapter:
                # TODO: Implement when Letta adapter is ready
                memory_items = await self.letta_adapter.recall(query, top_k=k_memory)
            
            # Step 3: Assemble prompt
            logger.debug("Assembling RAG prompt")
            messages = assemble_rag_prompt(query, search_results, memory_items)
            
            # Step 4: Generate answer
            logger.debug("Generating answer with LLM")
            answer = await self.llm_provider.generate(
                system=SYSTEM_PROMPT,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if not answer:
                raise RuntimeError("LLM generated empty response")
            
            logger.debug(
                "Answer generation completed",
                answer_length=len(answer),
                matter_id=self.matter.id
            )
            
            # Step 5: Extract and validate citations
            citations = extract_citations_from_answer(answer)
            citation_validity = validate_citations(citations, search_results)
            
            # Log citation validation results
            valid_citations = sum(1 for valid in citation_validity.values() if valid)
            if citations:
                logger.debug(
                    "Citation validation completed",
                    total_citations=len(citations),
                    valid_citations=valid_citations,
                    validity_rate=valid_citations / len(citations)
                )
            
            # Step 6: Convert search results to source chunks
            sources = self._format_sources(search_results)
            
            # Step 7: Generate follow-up suggestions
            followups = await self._generate_followups(query, answer, memory_items)
            
            # Step 8: Extract knowledge items for future Letta integration
            extracted_facts = await self._extract_knowledge_items(answer, search_results)
            
            # TODO: Step 9: Update Letta memory (Sprint 5)
            if self.letta_adapter:
                await self.letta_adapter.upsert_interaction(
                    query, answer, sources, extracted_facts
                )
            
            response = RAGResponse(
                answer=answer,
                sources=sources,
                followups=followups,
                used_memory=memory_items
            )
            
            logger.info(
                "RAG pipeline completed successfully",
                matter_id=self.matter.id,
                answer_length=len(answer),
                sources_count=len(sources),
                followups_count=len(followups),
                valid_citations=valid_citations,
                total_citations=len(citations)
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "RAG pipeline failed",
                matter_id=self.matter.id,
                query_preview=query[:100],
                error=str(e)
            )
            raise RuntimeError(f"RAG pipeline failed: {str(e)}")
    
    def _format_sources(self, search_results: List[SearchResult]) -> List[SourceChunk]:
        """Convert search results to UI-displayable source chunks."""
        sources = []
        for result in search_results:
            source = SourceChunk(
                doc=result.doc_name,
                page_start=result.page_start,
                page_end=result.page_end,
                text=result.text,  # Already truncated to ~600 chars by vector store
                score=result.similarity_score
            )
            sources.append(source)
        return sources
    
    async def _generate_followups(
        self, 
        query: str, 
        answer: str, 
        memory_items: List[KnowledgeItem]
    ) -> List[str]:
        """Generate contextually relevant follow-up questions."""
        try:
            # Build context for follow-up generation
            context_parts = [f"Original Question: {query}", f"Assistant Answer: {answer[:1000]}..."]
            
            if memory_items:
                memory_summary = ", ".join([f"{item.type}: {item.label}" for item in memory_items[:3]])
                context_parts.append(f"Relevant Memory: {memory_summary}")
            
            context = "\n\n".join(context_parts)
            
            # Generate follow-ups using a smaller, focused prompt
            followup_messages = [{"role": "user", "content": f"{context}\n\n{FOLLOWUP_SUGGESTIONS_PROMPT}"}]
            
            followup_response = await self.llm_provider.generate(
                system="You are a construction claims legal assistant. Generate specific, actionable follow-up questions.",
                messages=followup_messages,
                max_tokens=200,  # Keep it concise
                temperature=0.3   # Slightly more creative for varied suggestions
            )
            
            # Parse follow-up suggestions (one per line)
            followups = []
            for line in followup_response.strip().split('\n'):
                line = line.strip()
                # Remove numbering, bullets, or other formatting
                line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                if line and len(line) <= 150:  # Reasonable length limit
                    followups.append(line)
            
            # Limit to 4 suggestions max
            return followups[:4]
            
        except Exception as e:
            logger.warning("Failed to generate follow-ups", error=str(e))
            # Return some generic construction claims follow-ups as fallback
            return [
                "What additional documentation would strengthen this analysis?",
                "Are there any schedule impacts or delay claims related to this issue?",
                "What are the potential damages or cost implications?",
                "Should we engage technical experts for further analysis?"
            ]
    
    async def _extract_knowledge_items(self, answer: str, sources: List[SearchResult]) -> List[KnowledgeItem]:
        """Extract structured knowledge items for agent memory."""
        try:
            # Build context for information extraction
            source_context = ""
            if sources:
                source_refs = []
                for source in sources[:5]:  # Limit context size
                    ref = f"{source.doc_name} p.{source.page_start}-{source.page_end}"
                    source_refs.append(ref)
                source_context = f"\n\nSource Documents: {', '.join(source_refs)}"
            
            extraction_prompt = f"Assistant Answer: {answer}{source_context}\n\n{INFORMATION_EXTRACTION_PROMPT}"
            
            extraction_messages = [{"role": "user", "content": extraction_prompt}]
            
            extraction_response = await self.llm_provider.generate(
                system="You extract structured knowledge from construction claims analysis. Return only valid JSON.",
                messages=extraction_messages,
                max_tokens=500,
                temperature=0.1  # Low temperature for structured output
            )
            
            # Parse JSON response
            import json
            try:
                extracted_items = json.loads(extraction_response.strip())
                if not isinstance(extracted_items, list):
                    extracted_items = []
                
                # Convert to KnowledgeItem objects
                knowledge_items = []
                for item_data in extracted_items[:10]:  # Limit to 10 items
                    try:
                        # Validate required fields
                        if not item_data.get('type') or not item_data.get('label'):
                            continue
                        
                        # Create KnowledgeItem
                        knowledge_item = KnowledgeItem(
                            type=item_data['type'],
                            label=item_data['label'],
                            date=item_data.get('date'),
                            actors=item_data.get('actors', []),
                            doc_refs=item_data.get('doc_refs', []),
                            support_snippet=item_data.get('support_snippet')
                        )
                        knowledge_items.append(knowledge_item)
                        
                    except Exception as item_e:
                        logger.warning("Failed to parse knowledge item", item=item_data, error=str(item_e))
                        continue
                
                logger.debug("Knowledge extraction completed", items_extracted=len(knowledge_items))
                return knowledge_items
                
            except json.JSONDecodeError as json_e:
                logger.warning("Failed to parse knowledge extraction JSON", error=str(json_e), response=extraction_response[:200])
                return []
                
        except Exception as e:
            logger.warning("Knowledge extraction failed", error=str(e))
            return []