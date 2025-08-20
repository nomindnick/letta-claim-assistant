"""
Retrieval-Augmented Generation (RAG) engine.

Coordinates vector retrieval, agent memory recall, prompt assembly,
and LLM generation for construction claims analysis.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import re
import time

from .logging_conf import get_logger
from .vectors import VectorStore, SearchResult
from .matters import Matter
from .models import KnowledgeItem, SourceChunk, ChatMode
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
# Advanced RAG features
from .citation_manager import CitationManager, CitationMapping, CitationMetrics
from .followup_engine import FollowupEngine, FollowupContext
from .hybrid_retrieval import HybridRetrieval, HybridRetrievalContext, create_retrieval_context, EnhancedSearchResult
from .quality_metrics import QualityAnalyzer, ResponseQualityMetrics, QualityThresholds
# California domain features
from .extractors.california_extractor import california_extractor
from .letta_domain_california import california_domain

logger = get_logger(__name__)




@dataclass
class RAGResponse:
    """Enhanced response from RAG pipeline with quality metrics."""
    answer: str
    sources: List[SourceChunk]
    followups: List[str]
    used_memory: List[KnowledgeItem]
    
    # Advanced features
    citation_mappings: List[CitationMapping] = None
    citation_metrics: CitationMetrics = None
    quality_metrics: ResponseQualityMetrics = None
    processing_time: float = None
    
    # Quality indicators for UI
    confidence_score: float = 0.0
    quality_warnings: List[str] = None
    improvement_suggestions: List[str] = None
    
    # Memory command indicator
    is_memory_command: bool = False


class RAGEngine:
    """Main RAG engine coordinating retrieval and generation."""
    
    def __init__(
        self, 
        matter: Matter, 
        llm_provider: Optional[LLMProvider] = None,
        vector_store: Optional[VectorStore] = None,
        quality_thresholds: Optional[QualityThresholds] = None,
        enable_advanced_features: bool = True
    ):
        self.matter = matter
        self.enable_advanced_features = enable_advanced_features
        
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
        
        # Initialize Letta adapter for agent memory
        from .letta_adapter import LettaAdapter
        try:
            self.letta_adapter = LettaAdapter(
                matter_path=matter.paths.root,
                matter_name=matter.name,
                matter_id=matter.id
            )
        except Exception as e:
            logger.warning("Failed to initialize Letta adapter", error=str(e))
            self.letta_adapter = None
        
        # Initialize advanced features if enabled
        if self.enable_advanced_features:
            self.citation_manager = CitationManager()
            self.followup_engine = FollowupEngine(self.llm_provider)
            self.hybrid_retrieval = HybridRetrieval(self.vector_store)
            self.quality_analyzer = QualityAnalyzer(quality_thresholds)
        else:
            self.citation_manager = None
            self.followup_engine = None
            self.hybrid_retrieval = None
            self.quality_analyzer = None
        
        logger.info(
            "RAG engine initialized",
            matter_id=matter.id,
            matter_name=matter.name,
            generation_model=matter.generation_model,
            embedding_model=matter.embedding_model,
            advanced_features_enabled=self.enable_advanced_features
        )
    
    async def generate_answer(
        self,
        query: str,
        k: int = 8,
        k_memory: int = 6,
        mode: ChatMode = ChatMode.COMBINED,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        conversation_history: Optional[List[str]] = None,
        recent_documents: Optional[List[str]] = None,
        enable_mmr: bool = True
    ) -> RAGResponse:
        """
        Generate answer using RAG pipeline.
        
        Args:
            query: User query
            k: Number of vector search results to retrieve
            k_memory: Number of memory items to recall
            mode: Chat mode (RAG_ONLY, MEMORY_ONLY, or COMBINED)
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
            mode=mode.value,
            k=k,
            k_memory=k_memory
        )
        
        start_time = time.time()
        
        try:
            # Check if this is a memory command (unless in RAG_ONLY mode)
            if mode != ChatMode.RAG_ONLY:
                from .memory_commands import parse_memory_command, MemoryCommandParser, MemoryAction
                command = parse_memory_command(query)
                
                if command and self.letta_adapter:
                    logger.info(
                        "Memory command detected",
                        action=command.action.value,
                        confidence=command.confidence
                    )
                    
                    # Process the memory command
                    if command.action == MemoryAction.REMEMBER:
                        # Create new memory
                        from .models import KnowledgeItem
                        knowledge_item = KnowledgeItem(
                            type="Fact",
                            label=command.content[:100],
                            support_snippet=command.content
                        )
                        
                        item_id = await self.letta_adapter.create_memory_item(
                            text=knowledge_item.model_dump_json(),
                            type="Fact"
                        )
                        
                        confirmation = MemoryCommandParser.format_confirmation(command)
                        
                        return RAGResponse(
                            answer=confirmation,
                            sources=[],
                            followups=["What else would you like me to remember?"],
                            used_memory=[],
                            processing_time=time.time() - start_time,
                            confidence_score=command.confidence,
                            is_memory_command=True
                        )
                    
                    elif command.action == MemoryAction.FORGET:
                        # Delete matching memories
                        success = await self.letta_adapter.search_and_delete_memory(command.content)
                        
                        if success:
                            confirmation = MemoryCommandParser.format_confirmation(command)
                        else:
                            confirmation = f"I couldn't find any memories matching: {command.content}"
                        
                        return RAGResponse(
                            answer=confirmation,
                            sources=[],
                            followups=[],
                            used_memory=[],
                            processing_time=time.time() - start_time,
                            confidence_score=command.confidence if success else 0.3,
                            is_memory_command=True
                        )
                    
                    elif command.action == MemoryAction.UPDATE:
                        # Update existing memory
                        success = await self.letta_adapter.search_and_update_memory(
                            target=command.target,
                            new_content=command.content
                        )
                        
                        if success:
                            confirmation = MemoryCommandParser.format_confirmation(command)
                        else:
                            confirmation = f"I couldn't find memories about '{command.target}' to update"
                        
                        return RAGResponse(
                            answer=confirmation,
                            sources=[],
                            followups=[],
                            used_memory=[],
                            processing_time=time.time() - start_time,
                            confidence_score=command.confidence if success else 0.3,
                            is_memory_command=True
                        )
                    
                    elif command.action == MemoryAction.QUERY:
                        # This will fall through to normal processing but in memory-only mode
                        # Switch to memory-only mode for this query
                        mode = ChatMode.MEMORY_ONLY
                        logger.info("Memory query command - switching to memory-only mode")
            
            # Handle MEMORY_ONLY mode - delegate to Letta adapter
            if mode == ChatMode.MEMORY_ONLY:
                if not self.letta_adapter:
                    logger.warning("Memory-only mode requested but Letta adapter not available")
                    return RAGResponse(
                        answer="Memory-only mode is not available for this matter.",
                        sources=[],
                        followups=[],
                        used_memory=[],
                        processing_time=time.time() - start_time
                    )
                
                logger.debug("Processing memory-only chat")
                memory_response = await self.letta_adapter.memory_only_chat(query)
                
                return RAGResponse(
                    answer=memory_response.get("answer", ""),
                    sources=[],  # No document sources in memory-only mode
                    followups=memory_response.get("followups", []),
                    used_memory=[],  # Memory items not individually listed in this mode
                    processing_time=time.time() - start_time,
                    confidence_score=0.8 if memory_response.get("memory_used") else 0.3
                )
            # Step 1: Enhanced retrieval using hybrid system or fallback to basic vector search
            if self.enable_advanced_features and self.hybrid_retrieval:
                logger.debug("Performing hybrid retrieval")
                retrieval_context = create_retrieval_context(
                    query=query,
                    matter_id=self.matter.id,
                    conversation_history=conversation_history or [],
                    recent_documents=recent_documents or []
                )
                
                # Get memory items first for hybrid retrieval (skip if RAG_ONLY mode)
                memory_items = []
                if mode != ChatMode.RAG_ONLY and self.letta_adapter:
                    try:
                        memory_items = await self.letta_adapter.recall(query, top_k=k_memory)
                        logger.debug("Agent memory recall completed", memory_items=len(memory_items))
                    except Exception as e:
                        logger.warning("Agent memory recall failed", error=str(e))
                        memory_items = []
                
                enhanced_results = await self.hybrid_retrieval.hybrid_search(
                    retrieval_context, memory_items, k=k, enable_mmr=enable_mmr
                )
                
                # Convert enhanced results back to standard format for compatibility
                search_results = []
                for enhanced in enhanced_results:
                    result = SearchResult(
                        chunk_id=enhanced.chunk_id,
                        doc_name=enhanced.doc_name,
                        page_start=enhanced.page_start,
                        page_end=enhanced.page_end,
                        text=enhanced.text,
                        similarity_score=enhanced.final_score,
                        metadata={}
                    )
                    search_results.append(result)
                
                logger.debug(
                    "Hybrid retrieval completed",
                    results_found=len(search_results),
                    avg_score=sum(r.similarity_score for r in search_results) / len(search_results) if search_results else 0
                )
            else:
                # Fallback to basic vector search
                logger.debug("Performing basic vector search")
                search_results = await self.vector_store.search(query, k=k)
                
                logger.debug(
                    "Vector search completed",
                    results_found=len(search_results),
                    avg_similarity=sum(r.similarity_score for r in search_results) / len(search_results) if search_results else 0
                )
                
                # Get memory items for basic retrieval (skip if RAG_ONLY mode)
                memory_items = []
                if mode != ChatMode.RAG_ONLY and self.letta_adapter:
                    try:
                        memory_items = await self.letta_adapter.recall(query, top_k=k_memory)
                        logger.debug("Agent memory recall completed", memory_items=len(memory_items))
                    except Exception as e:
                        logger.warning("Agent memory recall failed", error=str(e))
                        memory_items = []
            
            # Memory items are now retrieved in Step 1 for both hybrid and basic modes
            
            # Step 3: Assemble prompt with California domain context
            logger.debug("Assembling RAG prompt with California domain context")
            
            # Extract California entities from query for additional context
            ca_entities = california_extractor.extract_all(query)
            
            # Add California context to prompt if entities found
            california_context = ""
            if ca_entities.get("statutes") or ca_entities.get("agencies") or ca_entities.get("deadlines"):
                california_context = "\n\nCalifornia Context Detected:\n"
                if ca_entities.get("statutes"):
                    california_context += f"- Statutes referenced: {', '.join([s['citation'] for s in ca_entities['statutes'][:3]])}\n"
                if ca_entities.get("agencies"):
                    california_context += f"- Public entities: {', '.join([a['name'] for a in ca_entities['agencies'][:3]])}\n"
                if ca_entities.get("deadlines"):
                    california_context += f"- Deadlines mentioned: {', '.join([d['text'] for d in ca_entities['deadlines'][:3]])}\n"
                california_context += "\nProvide California-specific legal analysis considering statutory requirements and public entity procedures."
            
            messages = assemble_rag_prompt(query + california_context, search_results, memory_items)
            
            # Step 4: Generate answer with enhanced system prompt for California
            logger.debug("Generating answer with LLM")
            
            # Use California-enhanced system prompt if domain detected
            system_prompt = SYSTEM_PROMPT
            if ca_entities and any(ca_entities.values()):
                system_prompt = f"{SYSTEM_PROMPT}\n\nYou have specialized knowledge of California public works construction law, including Public Contract Code, Government Code claims procedures, mechanics liens, prevailing wage requirements, and public entity requirements. Provide precise statutory citations and deadline information when relevant."
            
            answer = await self.llm_provider.generate(
                system=system_prompt,
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
            
            # Step 5: Advanced citation analysis and validation
            citations = extract_citations_from_answer(answer)
            sources = self._format_sources(search_results)
            
            citation_mappings = []
            citation_metrics = None
            
            if self.enable_advanced_features and self.citation_manager:
                # Enhanced citation analysis
                citation_mappings = self.citation_manager.create_citation_mappings(citations, sources)
                citation_metrics = self.citation_manager.calculate_citation_metrics(
                    answer, citation_mappings, sources
                )
                
                logger.debug(
                    "Advanced citation analysis completed",
                    total_citations=citation_metrics.total_citations,
                    valid_citations=citation_metrics.valid_citations,
                    accuracy_score=citation_metrics.accuracy_score,
                    coverage_score=citation_metrics.coverage_score
                )
            else:
                # Fallback to basic citation validation
                citation_validity = validate_citations(citations, search_results)
                valid_citations = sum(1 for valid in citation_validity.values() if valid)
                if citations:
                    logger.debug(
                        "Basic citation validation completed",
                        total_citations=len(citations),
                        valid_citations=valid_citations,
                        validity_rate=valid_citations / len(citations)
                    )
            
            # Step 6: Enhanced follow-up generation
            if self.enable_advanced_features and self.followup_engine:
                followup_context = FollowupContext(
                    user_query=query,
                    assistant_answer=answer,
                    memory_items=memory_items,
                    conversation_history=conversation_history or [],
                    matter_context={'matter_id': self.matter.id, 'matter_name': self.matter.name}
                )
                
                followup_suggestions = await self.followup_engine.generate_followups(followup_context)
                followups = [suggestion.question for suggestion in followup_suggestions]
                
                logger.debug(
                    "Advanced follow-up generation completed",
                    suggestions_count=len(followups),
                    avg_priority=sum(s.priority for s in followup_suggestions) / len(followup_suggestions) if followup_suggestions else 0
                )
            else:
                # Fallback to basic follow-up generation
                followups = await self._generate_followups(query, answer, memory_items)
                logger.debug("Basic follow-up generation completed", suggestions_count=len(followups))
            
            # Step 7: Extract knowledge items for future Letta integration
            extracted_facts = await self._extract_knowledge_items(answer, search_results)
            
            # Step 8: Update Letta memory with interaction and extracted facts
            if self.letta_adapter:
                try:
                    await self.letta_adapter.upsert_interaction(
                        query, answer, sources, extracted_facts
                    )
                    logger.debug("Agent memory updated successfully")
                except Exception as e:
                    logger.warning("Failed to update agent memory", error=str(e))
            
            # Step 9: Quality analysis and metrics
            processing_time = time.time() - start_time
            quality_metrics = None
            quality_warnings = []
            improvement_suggestions = []
            
            if self.enable_advanced_features and self.quality_analyzer:
                quality_metrics = self.quality_analyzer.analyze_response_quality(
                    user_query=query,
                    assistant_answer=answer,
                    source_chunks=sources,
                    citation_mappings=citation_mappings or [],
                    citation_metrics=citation_metrics or CitationMetrics(0, 0, 0.0, 0.0, 0.0, 0.0),
                    followup_suggestions=followup_suggestions if self.followup_engine else [],
                    memory_items=memory_items,
                    processing_time=processing_time,
                    matter_id=self.matter.id
                )
                
                quality_warnings = quality_metrics.quality_warnings
                improvement_suggestions = self.quality_analyzer.suggest_quality_improvements(quality_metrics)
                
                logger.debug(
                    "Quality analysis completed",
                    overall_quality=quality_metrics.overall_quality,
                    confidence_score=quality_metrics.confidence_score,
                    meets_standards=quality_metrics.meets_minimum_standards,
                    warnings_count=len(quality_warnings)
                )
            
            response = RAGResponse(
                answer=answer,
                sources=sources,
                followups=followups,
                used_memory=memory_items,
                citation_mappings=citation_mappings,
                citation_metrics=citation_metrics,
                quality_metrics=quality_metrics,
                processing_time=processing_time,
                confidence_score=quality_metrics.confidence_score if quality_metrics else 0.5,
                quality_warnings=quality_warnings,
                improvement_suggestions=improvement_suggestions
            )
            
            logger.info(
                "RAG pipeline completed successfully",
                matter_id=self.matter.id,
                answer_length=len(answer),
                sources_count=len(sources),
                followups_count=len(followups),
                processing_time_seconds=processing_time,
                quality_score=quality_metrics.overall_quality if quality_metrics else None,
                confidence_score=quality_metrics.confidence_score if quality_metrics else None,
                meets_quality_standards=quality_metrics.meets_minimum_standards if quality_metrics else None
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
        # Try to use Letta for follow-up generation first
        if self.letta_adapter:
            try:
                followups = await self.letta_adapter.suggest_followups(query, answer)
                if followups:
                    return followups
            except Exception as e:
                logger.warning("Letta follow-up generation failed, using fallback", error=str(e))
        
        # Fallback to direct LLM generation
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
                max_tokens=None,  # Let model complete naturally
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
    
    def get_advanced_features_status(self) -> Dict[str, bool]:
        """Get status of all advanced features."""
        return {
            'advanced_features_enabled': self.enable_advanced_features,
            'citation_manager_available': self.citation_manager is not None,
            'followup_engine_available': self.followup_engine is not None,
            'hybrid_retrieval_available': self.hybrid_retrieval is not None,
            'quality_analyzer_available': self.quality_analyzer is not None,
            'letta_adapter_available': self.letta_adapter is not None
        }
    
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
                max_tokens=None,  # Let model complete naturally
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
    
    def get_quality_insights(self, matter_id: str = None) -> Dict[str, Any]:
        """Get quality insights and statistics for the matter."""
        if not self.enable_advanced_features or not self.quality_analyzer:
            return {'message': 'Advanced features not enabled'}
        
        insights = {
            'matter_id': matter_id or self.matter.id,
            'advanced_features_enabled': True
        }
        
        # Get historical quality stats
        if matter_id or self.matter.id:
            historical_stats = self.quality_analyzer.get_historical_stats(matter_id or self.matter.id)
            if historical_stats:
                insights['historical_stats'] = {
                    'total_responses': historical_stats.total_responses,
                    'average_quality': historical_stats.average_quality,
                    'quality_trend': historical_stats.quality_trend,
                    'best_quality_score': historical_stats.best_quality_score,
                    'worst_quality_score': historical_stats.worst_quality_score,
                    'quality_consistency': historical_stats.quality_consistency,
                    'citation_quality_avg': historical_stats.citation_quality_avg,
                    'content_quality_avg': historical_stats.content_quality_avg,
                    'source_quality_avg': historical_stats.source_quality_avg,
                    'followup_quality_avg': historical_stats.followup_quality_avg
                }
            else:
                insights['historical_stats'] = {'message': 'No historical data available'}
        
        # Get retrieval system stats
        if self.hybrid_retrieval:
            insights['retrieval_stats'] = self.hybrid_retrieval.get_retrieval_stats()
        
        # Get quality thresholds
        insights['quality_thresholds'] = {
            'minimum_citation_coverage': self.quality_analyzer.thresholds.minimum_citation_coverage,
            'minimum_source_diversity': self.quality_analyzer.thresholds.minimum_source_diversity,
            'minimum_answer_completeness': self.quality_analyzer.thresholds.minimum_answer_completeness,
            'minimum_confidence_score': self.quality_analyzer.thresholds.minimum_confidence_score
        }
        
        return insights
    
    def update_quality_thresholds(self, new_thresholds: QualityThresholds):
        """Update quality thresholds for response validation."""
        if self.enable_advanced_features and self.quality_analyzer:
            self.quality_analyzer.thresholds = new_thresholds
            logger.info("Quality thresholds updated", thresholds=new_thresholds)
        else:
            logger.warning("Cannot update quality thresholds - advanced features not enabled")
    
    def regenerate_if_needed(self, response: RAGResponse, max_attempts: int = 2) -> bool:
        """Check if response needs regeneration based on quality metrics."""
        if not self.enable_advanced_features or not response.quality_metrics:
            return False
        
        return response.quality_metrics.requires_regeneration and max_attempts > 0
    
    async def generate_answer_with_quality_retry(
        self,
        query: str,
        k: int = 8,
        k_memory: int = 6,
        mode: ChatMode = ChatMode.COMBINED,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        conversation_history: Optional[List[str]] = None,
        recent_documents: Optional[List[str]] = None,
        enable_mmr: bool = True,
        max_retry_attempts: int = 1
    ) -> RAGResponse:
        """Generate answer with automatic quality-based retry."""
        
        attempt = 0
        best_response = None
        best_quality = -1.0
        
        while attempt <= max_retry_attempts:
            try:
                response = await self.generate_answer(
                    query=query,
                    k=k,
                    k_memory=k_memory,
                    mode=mode,
                    max_tokens=max_tokens,
                    temperature=temperature + (attempt * 0.1),  # Slightly increase temp on retry
                    conversation_history=conversation_history,
                    recent_documents=recent_documents,
                    enable_mmr=enable_mmr
                )
                
                # Track best response
                current_quality = response.quality_metrics.overall_quality if response.quality_metrics else 0.5
                if current_quality > best_quality:
                    best_quality = current_quality
                    best_response = response
                
                # Check if quality is acceptable or if this is the last attempt
                if (not self.regenerate_if_needed(response, max_retry_attempts - attempt) or 
                    attempt >= max_retry_attempts):
                    
                    logger.info(
                        "Answer generation completed",
                        attempt=attempt + 1,
                        final_quality=current_quality,
                        retry_triggered=attempt > 0
                    )
                    return response
                
                logger.info(
                    "Answer quality below threshold, retrying",
                    attempt=attempt + 1,
                    quality_score=current_quality,
                    requires_regeneration=response.quality_metrics.requires_regeneration if response.quality_metrics else False
                )
                
                attempt += 1
                
            except Exception as e:
                logger.error(f"Answer generation attempt {attempt + 1} failed", error=str(e))
                attempt += 1
                if attempt > max_retry_attempts:
                    raise
        
        # Return best response if all attempts completed
        return best_response or response