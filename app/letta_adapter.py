"""
Letta integration for persistent agent memory.

Provides matter-specific agent knowledge management with recall,
upsert operations, and follow-up suggestion generation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict, Counter
import pkg_resources

try:
    from letta_client import AsyncLetta, Letta
    from letta_client.types import (
        AgentState,
        LlmConfig,
        EmbeddingConfig,
        Block  # Changed from MemoryBlock to Block
    )
    # Get Letta version for compatibility checking
    try:
        LETTA_VERSION = pkg_resources.get_distribution("letta").version
        LETTA_CLIENT_VERSION = pkg_resources.get_distribution("letta-client").version
    except:
        LETTA_VERSION = "unknown"
        LETTA_CLIENT_VERSION = "unknown"
except ImportError:
    # Fallback for older versions or installation issues
    AsyncLetta = None
    Letta = None
    LETTA_VERSION = None
    LETTA_CLIENT_VERSION = None
    print("Warning: Letta client import failed, using fallback implementation")

from .logging_conf import get_logger
from .models import KnowledgeItem, SourceChunk
from .letta_server import server_manager
from .letta_config import config_manager, LettaAgentConfig
from .letta_connection import connection_manager, ConnectionState

logger = get_logger(__name__)


class LettaAdapter:
    """Adapter for Letta agent memory integration."""
    
    def __init__(self, matter_path: Path, matter_name: str = "", matter_id: str = ""):
        self.matter_path = matter_path
        self.matter_name = matter_name or "Construction Claim"
        self.matter_id = matter_id or str(uuid.uuid4())
        self.letta_path = matter_path / "knowledge" / "letta_state"
        self.letta_path.mkdir(parents=True, exist_ok=True)
        
        # Letta client and agent
        self.client: Optional[AsyncLetta] = None
        self.sync_client: Optional[Letta] = None
        self.agent_id: Optional[str] = None
        self.agent_state: Optional[AgentState] = None
        self.fallback_mode: bool = False
        
        # Check for existing data before initialization
        self._check_existing_data()
        
        # Initialize client and agent asynchronously
        # This will be called when first operation is attempted
        self._initialized = False
        
        logger.info(
            "LettaAdapter created (pending initialization)",
            matter_id=self.matter_id,
            matter_name=self.matter_name,
            letta_path=str(self.letta_path)
        )
    
    async def _ensure_initialized(self) -> bool:
        """
        Ensure the adapter is initialized with server connection.
        
        Returns:
            True if initialized successfully, False if in fallback mode
        """
        if self._initialized:
            return not self.fallback_mode
        
        try:
            # Use connection manager for connection
            if not await connection_manager.connect():
                logger.warning("Letta server not available, using fallback mode")
                self.fallback_mode = True
                self._initialized = True
                return False
            
            # Get client from connection manager
            self.client = connection_manager.client
            self.sync_client = connection_manager.sync_client
            
            # Initialize or load agent
            await self._initialize_or_load_agent_async()
            
            self.fallback_mode = False
            self._initialized = True
            
            logger.info(
                "LettaAdapter initialized successfully",
                agent_id=self.agent_id,
                connection_state=connection_manager.get_state().value
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LettaAdapter: {e}")
            self.fallback_mode = True
            self._initialized = True
            return False
    
    async def recall(self, query: str, top_k: int = 6) -> List[KnowledgeItem]:
        """
        Recall relevant knowledge items from agent memory.
        
        Args:
            query: Query to find relevant memory items for
            top_k: Number of memory items to return
            
        Returns:
            List of relevant KnowledgeItem objects
        """
        # Ensure initialized
        if not await self._ensure_initialized():
            logger.warning("Letta not available, returning empty memory list")
            return []
        
        if not self.agent_id:
            return []
        
        # Use connection manager to execute with retry and metrics
        async def _recall_operation():
            # Use new passages API for searching archival memory
            passages = await self.client.agents.passages.list(
                agent_id=self.agent_id,
                search=query,
                limit=top_k * 2  # Get more than needed for filtering
            )
            return passages
        
        try:
            # Execute with retry and metrics tracking
            memory_results = await connection_manager.execute_with_retry(
                "recall",
                _recall_operation
            )
            
            if not memory_results:
                return []
            
            knowledge_items = []
            for passage in memory_results[:top_k]:
                # Parse passage text as JSON knowledge item
                try:
                    content = passage.text if hasattr(passage, 'text') else str(passage)
                    if content.startswith('{') and content.endswith('}'):
                        # Parse JSON knowledge item
                        item_data = json.loads(content)
                        knowledge_item = KnowledgeItem(
                            type=item_data.get('type', 'Fact'),
                            label=item_data.get('label', 'Unknown'),
                            date=item_data.get('date'),
                            actors=item_data.get('actors', []),
                            doc_refs=item_data.get('doc_refs', []),
                            support_snippet=item_data.get('support_snippet')
                        )
                        knowledge_items.append(knowledge_item)
                    else:
                        # Convert unstructured memory to knowledge item
                        knowledge_item = KnowledgeItem(
                            type="Fact",
                            label=content[:100],  # Use first 100 chars as label
                            support_snippet=content[:300] if len(content) > 100 else None
                        )
                        knowledge_items.append(knowledge_item)
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning("Failed to parse memory item", error=str(e), content=content[:100])
                    continue
            
            logger.debug(
                "Memory recall completed",
                query_preview=query[:50],
                items_found=len(knowledge_items),
                total_memory=len(memory_results)
            )
            
            return knowledge_items
            
        except Exception as e:
            logger.error("Memory recall failed", error=str(e), query=query[:50])
            return []
    
    async def recall_with_context(
        self,
        query: str,
        conversation_history: List[str] = None,
        top_k: int = 8,
        recency_weight: float = 0.2
    ) -> List[KnowledgeItem]:
        """
        Recall memory with conversation context and recency weighting.
        
        Args:
            query: Current query
            conversation_history: Recent conversation turns
            top_k: Number of items to return
            recency_weight: Weight for recent memories (0-1)
            
        Returns:
            Context-aware list of knowledge items
        """
        if not await self._ensure_initialized():
            return []
        
        if not self.agent_id:
            return []
        
        try:
            # Expand query with conversation context
            expanded_query = query
            if conversation_history:
                context_snippet = " ".join(conversation_history[-3:])  # Last 3 turns
                expanded_query = f"{query} Context: {context_snippet}"
            
            # Get base memories
            async def _search_memories():
                return await self.client.agents.passages.list(
                    agent_id=self.agent_id,
                    search=expanded_query,
                    limit=top_k * 3  # Get more for scoring
                )
            
            memories = await connection_manager.execute_with_retry(
                "recall_with_context",
                _search_memories
            )
            
            if not memories:
                return []
            
            # Score and rank memories
            scored_memories = []
            current_time = datetime.now()
            
            for passage in memories:
                try:
                    # Parse memory content
                    content = passage.text if hasattr(passage, 'text') else str(passage)
                    
                    # Calculate recency score if timestamp available
                    recency_score = 1.0
                    if hasattr(passage, 'created_at'):
                        age_hours = (current_time - passage.created_at).total_seconds() / 3600
                        recency_score = max(0.1, 1.0 - (age_hours / 168))  # Decay over a week
                    
                    # Calculate relevance score (would be from vector similarity in real implementation)
                    relevance_score = 0.8  # Placeholder - would use actual similarity
                    
                    # Combined score
                    final_score = (relevance_score * (1 - recency_weight)) + (recency_score * recency_weight)
                    
                    scored_memories.append((final_score, content))
                    
                except Exception as e:
                    logger.debug(f"Error scoring memory: {e}")
                    continue
            
            # Sort by score and convert to knowledge items
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            knowledge_items = []
            
            for score, content in scored_memories[:top_k]:
                try:
                    if content.startswith('{') and content.endswith('}'):
                        item_data = json.loads(content)
                        knowledge_item = KnowledgeItem(
                            type=item_data.get('type', 'Fact'),
                            label=item_data.get('label', 'Unknown'),
                            date=item_data.get('date'),
                            actors=item_data.get('actors', []),
                            doc_refs=item_data.get('doc_refs', []),
                            support_snippet=item_data.get('support_snippet')
                        )
                    else:
                        knowledge_item = KnowledgeItem(
                            type="Fact",
                            label=content[:100],
                            support_snippet=content[:300] if len(content) > 100 else None
                        )
                    knowledge_items.append(knowledge_item)
                except:
                    continue
            
            logger.debug(
                "Context-aware recall completed",
                query_preview=query[:50],
                items_found=len(knowledge_items),
                context_provided=bool(conversation_history)
            )
            
            return knowledge_items
            
        except Exception as e:
            logger.error("Context recall failed", error=str(e))
            return []
    
    async def semantic_memory_search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        top_k: int = 10
    ) -> List[Tuple[KnowledgeItem, float]]:
        """
        Advanced semantic search with filtering and confidence scores.
        
        Args:
            query: Search query (supports AND/OR logic)
            filters: Metadata filters (date_range, doc_sources, types)
            top_k: Number of results
            
        Returns:
            List of (knowledge_item, confidence_score) tuples
        """
        if not await self._ensure_initialized():
            return []
        
        if not self.agent_id:
            return []
        
        try:
            # Parse complex query
            search_terms = self._parse_complex_query(query)
            
            # Perform multiple searches if needed
            all_results = []
            for term in search_terms:
                results = await self.client.agents.passages.list(
                    agent_id=self.agent_id,
                    search=term,
                    limit=top_k * 2
                )
                all_results.extend(results or [])
            
            # Deduplicate and filter
            seen_hashes = set()
            filtered_results = []
            
            for passage in all_results:
                try:
                    content = passage.text if hasattr(passage, 'text') else str(passage)
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    if content_hash in seen_hashes:
                        continue
                    seen_hashes.add(content_hash)
                    
                    # Apply filters if provided
                    if filters and content.startswith('{'):
                        try:
                            item_data = json.loads(content)
                            
                            # Type filter
                            if 'types' in filters and item_data.get('type') not in filters['types']:
                                continue
                            
                            # Date range filter
                            if 'date_range' in filters and item_data.get('date'):
                                item_date = datetime.fromisoformat(item_data['date'])
                                if not (filters['date_range']['start'] <= item_date <= filters['date_range']['end']):
                                    continue
                            
                            # Document source filter
                            if 'doc_sources' in filters:
                                doc_refs = item_data.get('doc_refs', [])
                                if not any(ref['doc'] in filters['doc_sources'] for ref in doc_refs):
                                    continue
                        except:
                            pass
                    
                    filtered_results.append(content)
                    
                except Exception as e:
                    logger.debug(f"Error processing search result: {e}")
                    continue
            
            # Convert to knowledge items with confidence scores
            results_with_scores = []
            for content in filtered_results[:top_k]:
                try:
                    confidence = 0.8  # Placeholder - would calculate based on match quality
                    
                    if content.startswith('{') and content.endswith('}'):
                        item_data = json.loads(content)
                        knowledge_item = KnowledgeItem(
                            type=item_data.get('type', 'Fact'),
                            label=item_data.get('label', 'Unknown'),
                            date=item_data.get('date'),
                            actors=item_data.get('actors', []),
                            doc_refs=item_data.get('doc_refs', []),
                            support_snippet=item_data.get('support_snippet')
                        )
                    else:
                        knowledge_item = KnowledgeItem(
                            type="Fact",
                            label=content[:100],
                            support_snippet=content[:300] if len(content) > 100 else None
                        )
                    
                    results_with_scores.append((knowledge_item, confidence))
                    
                except Exception as e:
                    logger.debug(f"Error converting to knowledge item: {e}")
                    continue
            
            logger.debug(
                "Semantic search completed",
                query=query[:50],
                results=len(results_with_scores),
                filters_applied=bool(filters)
            )
            
            return results_with_scores
            
        except Exception as e:
            logger.error("Semantic search failed", error=str(e))
            return []
    
    async def upsert_interaction(
        self,
        user_query: str,
        llm_answer: str,
        sources: List[SourceChunk],
        extracted_facts: List[KnowledgeItem]
    ) -> None:
        """
        Store interaction and extracted facts in agent memory.
        
        Args:
            user_query: The user's original query
            llm_answer: The generated answer
            sources: Source chunks used in the answer
            extracted_facts: Knowledge items extracted from the interaction
        """
        # Ensure initialized
        if not await self._ensure_initialized():
            logger.warning("Letta not available, skipping interaction storage")
            return
        
        if not self.agent_id:
            return
        
        try:
            # Store the conversation turn in agent memory
            timestamp = datetime.now().isoformat()
            
            # Create interaction summary for archival memory
            interaction_summary = {
                "type": "interaction",
                "timestamp": timestamp,
                "query": user_query[:200],  # Truncate for storage
                "answer_preview": llm_answer[:300],
                "sources_count": len(sources),
                "facts_extracted": len(extracted_facts)
            }
            
            # Store interaction summary with retry
            async def _insert_summary():
                # Use new passages API to insert memory
                passages = await self.client.agents.passages.create(
                    agent_id=self.agent_id,
                    text=json.dumps(interaction_summary)
                )
                return passages
            
            await connection_manager.execute_with_retry(
                "upsert",
                _insert_summary
            )
            
            # Store each extracted knowledge item
            for fact in extracted_facts:
                fact_data = {
                    "type": fact.type,
                    "label": fact.label,
                    "date": fact.date,
                    "actors": fact.actors,
                    "doc_refs": fact.doc_refs,
                    "support_snippet": fact.support_snippet,
                    "extracted_from": timestamp
                }
                
                async def _insert_fact():
                    # Use new passages API to insert fact
                    passages = await self.client.agents.passages.create(
                        agent_id=self.agent_id,
                        text=json.dumps(fact_data)
                    )
                    return passages
                
                await connection_manager.execute_with_retry(
                    "upsert",
                    _insert_fact
                )
            
            # Update core memory with recent activity context
            if extracted_facts:
                recent_facts = ", ".join([f"{fact.type}: {fact.label}" for fact in extracted_facts[:3]])
                core_update = f"Recent discussion: {user_query[:100]}... Key items: {recent_facts}"
                
                # Get current agent state to update memory
                agent = await self.client.agents.retrieve(self.agent_id)
                
                # Update memory blocks
                for block in agent.memory.blocks:
                    if block.label == "human":
                        block.value = f"{block.value}\n\nRecent context: {core_update}"
                        break
                
                # Update agent with new memory
                await self.client.agents.modify(
                    agent_id=self.agent_id,
                    memory_blocks=[{"label": b.label, "value": b.value} for b in agent.memory.blocks]
                )
            
            logger.debug(
                "Interaction stored successfully",
                facts_stored=len(extracted_facts),
                interaction_id=timestamp,
                query_preview=user_query[:50]
            )
            
        except Exception as e:
            logger.error(
                "Failed to store interaction",
                error=str(e),
                query_preview=user_query[:50],
                facts_count=len(extracted_facts)
            )
    
    async def store_knowledge_batch(
        self,
        knowledge_items: List[KnowledgeItem],
        deduplicate: bool = True,
        importance_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Batch insert multiple knowledge items with deduplication.
        
        Args:
            knowledge_items: List of knowledge items to store
            deduplicate: Whether to check for duplicates
            importance_threshold: Minimum importance score to store
            
        Returns:
            Dictionary with storage statistics
        """
        if not await self._ensure_initialized():
            logger.warning("Letta not available, skipping batch storage")
            return {"stored": 0, "duplicates": 0, "skipped": 0}
        
        if not self.agent_id or not knowledge_items:
            return {"stored": 0, "duplicates": 0, "skipped": 0}
        
        stored_count = 0
        duplicate_count = 0
        skipped_count = 0
        
        try:
            # Get existing memories for deduplication if needed
            existing_hashes = set()
            if deduplicate:
                existing_memories = await self._get_all_memories_cached()
                # Parse existing memories and hash their content
                for mem_text in existing_memories:
                    try:
                        if mem_text.startswith('{'):
                            mem_data = json.loads(mem_text)
                            # Create KnowledgeItem from memory for consistent hashing
                            existing_item = KnowledgeItem(
                                type=mem_data.get('type', 'Fact'),
                                label=mem_data.get('label', ''),
                                date=mem_data.get('date'),
                                actors=mem_data.get('actors', [])
                            )
                            existing_hashes.add(self._hash_knowledge_item(existing_item))
                        else:
                            existing_hashes.add(self._hash_memory(mem_text))
                    except:
                        existing_hashes.add(self._hash_memory(mem_text))
            
            # Process each knowledge item
            tasks = []
            for item in knowledge_items:
                # Calculate importance score
                importance = self._calculate_importance_score(item)
                if importance < importance_threshold:
                    skipped_count += 1
                    continue
                
                # Check for duplicates
                item_hash = self._hash_knowledge_item(item)
                if deduplicate and item_hash in existing_hashes:
                    duplicate_count += 1
                    continue
                
                # Prepare memory data with metadata
                memory_data = {
                    "type": item.type,
                    "label": item.label,
                    "date": item.date,
                    "actors": item.actors,
                    "doc_refs": item.doc_refs,
                    "support_snippet": item.support_snippet,
                    "importance_score": importance,
                    "stored_at": datetime.now().isoformat(),
                    "hash": item_hash
                }
                
                # Create async task for insertion
                async def _insert_memory(data):
                    return await self.client.agents.passages.create(
                        agent_id=self.agent_id,
                        text=json.dumps(data)
                    )
                
                tasks.append(_insert_memory(memory_data))
                existing_hashes.add(item_hash)
            
            # Execute batch insertions in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                stored_count = sum(1 for r in results if not isinstance(r, Exception))
                
                # Log any errors
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    logger.warning(f"Some batch insertions failed: {len(errors)} errors")
            
            logger.info(
                "Batch knowledge storage completed",
                total_items=len(knowledge_items),
                stored=stored_count,
                duplicates=duplicate_count,
                skipped=skipped_count
            )
            
            return {
                "stored": stored_count,
                "duplicates": duplicate_count,
                "skipped": skipped_count,
                "total": len(knowledge_items)
            }
            
        except Exception as e:
            logger.error("Batch storage failed", error=str(e))
            return {"stored": stored_count, "duplicates": duplicate_count, "skipped": skipped_count, "error": str(e)}
    
    async def update_core_memory_smart(
        self,
        block_label: str,
        new_content: str,
        mode: str = "append",
        max_size: int = 2000
    ) -> bool:
        """
        Smart update of core memory blocks with size management.
        
        Args:
            block_label: Memory block to update (e.g., "human", "persona")
            new_content: Content to add/update
            mode: Update mode - "append", "replace", "prepend", "smart"
            max_size: Maximum size for memory block
            
        Returns:
            True if update successful
        """
        if not await self._ensure_initialized():
            return False
        
        if not self.agent_id:
            return False
        
        try:
            # Get current agent state
            agent = await self.client.agents.retrieve(self.agent_id)
            
            # Find and update the target block
            updated = False
            for block in agent.memory.blocks:
                if block.label == block_label:
                    current_value = block.value or ""
                    
                    if mode == "replace":
                        new_value = new_content
                    elif mode == "prepend":
                        new_value = f"{new_content}\n\n{current_value}"
                    elif mode == "append":
                        new_value = f"{current_value}\n\n{new_content}"
                    elif mode == "smart":
                        # Smart mode: summarize if too long, keep important parts
                        combined = f"{current_value}\n\n{new_content}"
                        if len(combined) > max_size:
                            new_value = await self._summarize_memory_block(combined, max_size)
                        else:
                            new_value = combined
                    else:
                        new_value = new_content
                    
                    # Truncate if exceeds max size
                    if len(new_value) > max_size:
                        if mode in ["append", "smart"]:
                            # Keep recent content when appending
                            new_value = "..." + new_value[-(max_size - 3):]
                        else:
                            # Keep beginning when prepending or replacing
                            new_value = new_value[:max_size - 3] + "..."
                    
                    block.value = new_value
                    updated = True
                    break
            
            if updated:
                # Update agent with new memory
                await self.client.agents.modify(
                    agent_id=self.agent_id,
                    memory_blocks=[{"label": b.label, "value": b.value} for b in agent.memory.blocks]
                )
                
                logger.debug(
                    "Core memory updated",
                    block_label=block_label,
                    mode=mode,
                    new_size=len(new_value) if updated else 0
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error("Core memory update failed", error=str(e))
            return False
    
    async def suggest_followups(
        self,
        user_query: str,
        llm_answer: str
    ) -> List[str]:
        """
        Generate contextual follow-up suggestions using agent memory.
        
        Args:
            user_query: The user's original query
            llm_answer: The generated answer
            
        Returns:
            List of suggested follow-up questions
        """
        # Ensure initialized
        if not await self._ensure_initialized():
            logger.warning("Letta not available, returning generic follow-ups")
            return self._fallback_followups()
        
        if not self.agent_id:
            return self._fallback_followups()
        
        try:
            # Use agent's memory and context to generate follow-ups
            followup_prompt = f"""
Based on this construction claims discussion:
Q: {user_query}
A: {llm_answer[:500]}...

Generate 3-4 specific follow-up questions that would help a construction attorney analyze this claim further. Focus on:
- Causation and responsibility
- Damages and costs
- Additional evidence needed
- Related issues or claims

Return only the questions, one per line."""
            
            # Send message to agent for follow-up generation with retry
            async def _suggest_operation():
                return await self.client.messages.send_message(
                    agent_id=self.agent_id,
                    role="user",
                    message=followup_prompt,
                    stream=False
                )
            
            response = await connection_manager.execute_with_retry(
                "suggest",
                _suggest_operation
            )
            
            if not response:
                return self._fallback_followups()
            
            # Parse response to extract follow-up questions
            followups = []
            if response and hasattr(response, 'messages'):
                for msg in response.messages:
                    if hasattr(msg, 'text') and msg.text:
                        # Extract questions from response
                        lines = msg.text.strip().split('\n')
                        for line in lines:
                            line = line.strip()
                            # Remove numbering and formatting
                            import re
                            line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                            if line and line.endswith('?') and len(line) <= 150:
                                followups.append(line)
            
            # Limit to 4 follow-ups max
            return followups[:4] if followups else self._fallback_followups()
            
        except Exception as e:
            logger.warning("Follow-up generation failed", error=str(e))
            return self._fallback_followups()
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about agent memory usage."""
        # Ensure initialized
        if not await self._ensure_initialized():
            return {
                "status": "unavailable",
                "memory_items": 0,
                "connection_state": connection_manager.get_state().value,
                "connection_metrics": connection_manager.get_metrics()
            }
        
        if not self.agent_id:
            return {
                "status": "unavailable",
                "memory_items": 0,
                "connection_state": connection_manager.get_state().value
            }
        
        try:
            # Get archival memory count with retry
            async def _get_memory_operation():
                # Use new passages API to get all memories
                passages = await self.client.agents.passages.list(
                    agent_id=self.agent_id,
                    limit=1000  # Get count estimate
                )
                return passages
            
            memory_results = await connection_manager.execute_with_retry(
                "get_memory_stats",
                _get_memory_operation
            )
            
            return {
                "status": "active",
                "memory_items": len(memory_results) if memory_results else 0,
                "agent_id": self.agent_id,
                "matter_name": self.matter_name,
                "connection_state": connection_manager.get_state().value,
                "connection_metrics": connection_manager.get_metrics()
            }
            
        except Exception as e:
            logger.warning("Failed to get memory stats", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def get_memory_summary(self, max_length: int = 500) -> str:
        """
        Generate a concise summary of agent's knowledge.
        
        Args:
            max_length: Maximum length of summary
            
        Returns:
            Summary string
        """
        if not await self._ensure_initialized():
            return "Memory unavailable"
        
        if not self.agent_id:
            return "No agent memory"
        
        try:
            # Get all memories
            memories = await self.client.agents.passages.list(
                agent_id=self.agent_id,
                limit=100
            )
            
            if not memories:
                return "No memories stored yet"
            
            # Categorize memories
            categories = defaultdict(list)
            for passage in memories:
                try:
                    content = passage.text if hasattr(passage, 'text') else str(passage)
                    if content.startswith('{'):
                        data = json.loads(content)
                        mem_type = data.get('type', 'Fact')
                        categories[mem_type].append(data.get('label', content[:50]))
                    else:
                        categories['Fact'].append(content[:50])
                except:
                    continue
            
            # Build summary
            summary_parts = [f"Memory contains {len(memories)} items:"]
            for category, items in categories.items():
                summary_parts.append(f"- {category}: {len(items)} items")
                if items[:3]:  # Show first 3 examples
                    examples = ", ".join(items[:3])
                    summary_parts.append(f"  Examples: {examples[:100]}")
            
            summary = "\n".join(summary_parts)
            
            # Truncate if needed
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + "..."
            
            return summary
            
        except Exception as e:
            logger.error("Failed to generate memory summary", error=str(e))
            return f"Error generating summary: {str(e)}"
    
    async def prune_memory(
        self,
        max_items: int = 1000,
        importance_threshold: float = 0.3,
        age_days: int = 90
    ) -> Dict[str, int]:
        """
        Prune old or low-importance memories.
        
        Args:
            max_items: Maximum items to keep
            importance_threshold: Minimum importance to keep
            age_days: Remove items older than this
            
        Returns:
            Dictionary with pruning statistics
        """
        if not await self._ensure_initialized():
            return {"removed": 0, "kept": 0, "error": "Not initialized"}
        
        if not self.agent_id:
            return {"removed": 0, "kept": 0, "error": "No agent"}
        
        try:
            # Get all memories
            memories = await self.client.agents.passages.list(
                agent_id=self.agent_id,
                limit=10000
            )
            
            if not memories:
                return {"removed": 0, "kept": len(memories)}
            
            # Score and sort memories
            scored_memories = []
            cutoff_date = datetime.now() - timedelta(days=age_days)
            
            for passage in memories:
                try:
                    content = passage.text if hasattr(passage, 'text') else str(passage)
                    
                    # Calculate importance
                    importance = 0.5  # Default
                    created_at = datetime.now()  # Default to now
                    
                    if content.startswith('{'):
                        data = json.loads(content)
                        importance = data.get('importance_score', 0.5)
                        if data.get('stored_at'):
                            created_at = datetime.fromisoformat(data['stored_at'])
                    
                    # Skip if too old or low importance
                    if created_at < cutoff_date or importance < importance_threshold:
                        continue
                    
                    scored_memories.append((importance, passage))
                    
                except:
                    # Keep on error
                    scored_memories.append((0.5, passage))
            
            # Sort by importance and keep top N
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            to_keep = scored_memories[:max_items]
            to_remove = scored_memories[max_items:]
            
            # Delete excess memories
            removed_count = 0
            if to_remove:
                for _, passage in to_remove:
                    try:
                        if hasattr(passage, 'id'):
                            await self.client.agents.passages.delete(
                                agent_id=self.agent_id,
                                passage_id=passage.id
                            )
                            removed_count += 1
                    except:
                        pass
            
            logger.info(
                "Memory pruning completed",
                total=len(memories),
                kept=len(to_keep),
                removed=removed_count
            )
            
            return {
                "total": len(memories),
                "kept": len(to_keep),
                "removed": removed_count,
                "threshold": importance_threshold,
                "age_days": age_days
            }
            
        except Exception as e:
            logger.error("Memory pruning failed", error=str(e))
            return {"removed": 0, "kept": 0, "error": str(e)}
    
    async def export_memory(
        self,
        format: str = "json",
        include_metadata: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """
        Export memory to JSON or CSV format.
        
        Args:
            format: Export format ("json" or "csv")
            include_metadata: Include metadata in export
            
        Returns:
            Exported data as string or dict
        """
        if not await self._ensure_initialized():
            return {"error": "Not initialized"}
        
        if not self.agent_id:
            return {"error": "No agent"}
        
        try:
            # Get all memories
            memories = await self.client.agents.passages.list(
                agent_id=self.agent_id,
                limit=10000
            )
            
            if format == "json":
                export_data = {
                    "agent_id": self.agent_id,
                    "matter_name": self.matter_name,
                    "export_date": datetime.now().isoformat(),
                    "memory_count": len(memories) if memories else 0,
                    "memories": []
                }
                
                for passage in (memories or []):
                    memory_item = {
                        "text": passage.text if hasattr(passage, 'text') else str(passage)
                    }
                    
                    if include_metadata:
                        memory_item["id"] = passage.id if hasattr(passage, 'id') else None
                        memory_item["created_at"] = str(passage.created_at) if hasattr(passage, 'created_at') else None
                    
                    # Parse structured data if available
                    try:
                        if memory_item["text"].startswith('{'):
                            memory_item["data"] = json.loads(memory_item["text"])
                    except:
                        pass
                    
                    export_data["memories"].append(memory_item)
                
                return export_data
                
            elif format == "csv":
                # Build CSV string
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Header
                headers = ["Type", "Label", "Date", "Actors", "Support", "Importance"]
                if include_metadata:
                    headers.extend(["ID", "Created"])
                writer.writerow(headers)
                
                # Rows
                for passage in (memories or []):
                    try:
                        content = passage.text if hasattr(passage, 'text') else str(passage)
                        row = []
                        
                        if content.startswith('{'):
                            data = json.loads(content)
                            row = [
                                data.get('type', ''),
                                data.get('label', '')[:100],
                                data.get('date', ''),
                                ';'.join(data.get('actors', [])),
                                data.get('support_snippet', '')[:200],
                                data.get('importance_score', '')
                            ]
                        else:
                            row = ['Fact', content[:100], '', '', content[:200], '']
                        
                        if include_metadata:
                            row.append(passage.id if hasattr(passage, 'id') else '')
                            row.append(str(passage.created_at) if hasattr(passage, 'created_at') else '')
                        
                        writer.writerow(row)
                    except:
                        continue
                
                return output.getvalue()
            
            else:
                return {"error": f"Unsupported format: {format}"}
                
        except Exception as e:
            logger.error("Memory export failed", error=str(e))
            return {"error": str(e)}
    
    async def import_memory(
        self,
        data: Union[str, Dict[str, Any]],
        format: str = "json",
        deduplicate: bool = True
    ) -> Dict[str, int]:
        """
        Import memory from external source.
        
        Args:
            data: Memory data to import
            format: Import format ("json" or "csv")
            deduplicate: Check for duplicates
            
        Returns:
            Import statistics
        """
        if not await self._ensure_initialized():
            return {"imported": 0, "skipped": 0, "error": "Not initialized"}
        
        if not self.agent_id:
            return {"imported": 0, "skipped": 0, "error": "No agent"}
        
        imported = 0
        skipped = 0
        
        try:
            memories_to_import = []
            
            if format == "json":
                if isinstance(data, str):
                    data = json.loads(data)
                
                if isinstance(data, dict) and 'memories' in data:
                    memories_to_import = data['memories']
                elif isinstance(data, list):
                    memories_to_import = data
                    
            elif format == "csv":
                import csv
                import io
                
                reader = csv.DictReader(io.StringIO(data))
                for row in reader:
                    memory_item = {
                        "type": row.get("Type", "Fact"),
                        "label": row.get("Label", ""),
                        "date": row.get("Date"),
                        "actors": row.get("Actors", "").split(';') if row.get("Actors") else [],
                        "support_snippet": row.get("Support", "")
                    }
                    memories_to_import.append({"data": memory_item})
            
            # Get existing hashes if deduplicating
            existing_hashes = set()
            if deduplicate:
                existing = await self.client.agents.passages.list(
                    agent_id=self.agent_id,
                    limit=10000
                )
                for p in (existing or []):
                    content = p.text if hasattr(p, 'text') else str(p)
                    existing_hashes.add(hashlib.md5(content.encode()).hexdigest())
            
            # Import memories
            for memory in memories_to_import:
                try:
                    # Extract content
                    if isinstance(memory, dict):
                        if 'data' in memory:
                            content = json.dumps(memory['data'])
                        elif 'text' in memory:
                            content = memory['text']
                        else:
                            content = json.dumps(memory)
                    else:
                        content = str(memory)
                    
                    # Check duplicate
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    if deduplicate and content_hash in existing_hashes:
                        skipped += 1
                        continue
                    
                    # Import
                    await self.client.agents.passages.create(
                        agent_id=self.agent_id,
                        text=content
                    )
                    imported += 1
                    existing_hashes.add(content_hash)
                    
                except Exception as e:
                    logger.debug(f"Failed to import memory item: {e}")
                    skipped += 1
            
            logger.info(
                "Memory import completed",
                imported=imported,
                skipped=skipped,
                total=len(memories_to_import)
            )
            
            return {
                "imported": imported,
                "skipped": skipped,
                "total": len(memories_to_import)
            }
            
        except Exception as e:
            logger.error("Memory import failed", error=str(e))
            return {"imported": imported, "skipped": skipped, "error": str(e)}
    
    async def _initialize_or_load_agent_async(self) -> None:
        """Initialize new agent or load existing one for this Matter."""
        if not self.client:
            return
        
        try:
            # Check if agent already exists
            agent_config_path = self.letta_path / "agent_config.json"
            
            if agent_config_path.exists():
                # Load existing agent
                with open(agent_config_path, 'r') as f:
                    config = json.load(f)
                    self.agent_id = config.get('agent_id')
                
                # Verify agent still exists
                try:
                    agent_state = await self.client.agents.retrieve(self.agent_id)
                    self.agent_state = agent_state
                    logger.debug("Loaded existing agent", agent_id=self.agent_id)
                    return
                except Exception:
                    logger.warning("Existing agent not found, creating new one")
                    self.agent_id = None
            
            # Create new agent
            await self._create_new_agent_async()
            
            # Save agent configuration with version info
            with open(agent_config_path, 'w') as f:
                json.dump({
                    'agent_id': self.agent_id,
                    'matter_id': self.matter_id,
                    'matter_name': self.matter_name,
                    'created_at': datetime.now().isoformat(),
                    'letta_version': LETTA_VERSION or 'unknown',
                    'letta_client_version': LETTA_CLIENT_VERSION or 'unknown'
                }, f, indent=2)
            
        except Exception as e:
            logger.error("Failed to initialize agent", error=str(e))
            self.agent_id = None
    
    async def _create_new_agent_async(self) -> None:
        """Create new Letta agent with construction domain configuration."""
        try:
            # Get agent configuration from config manager
            agent_config = config_manager.get_agent_config(
                matter_name=self.matter_name,
                llm_provider="ollama",
                llm_model="gpt-oss:20b"
            )
            
            # Create LLM configuration
            llm_config = LlmConfig(
                model=agent_config.llm_model,
                model_endpoint_type=agent_config.llm_provider,
                model_endpoint=agent_config.llm_endpoint,
                context_window=agent_config.context_window,
                max_tokens=agent_config.max_tokens,
                temperature=agent_config.temperature
            )
            
            # Create embedding configuration
            embedding_config = EmbeddingConfig(
                embedding_model=agent_config.embedding_model,
                embedding_endpoint_type=agent_config.embedding_provider,
                embedding_endpoint=agent_config.embedding_endpoint,
                embedding_dim=agent_config.embedding_dim
            )
            
            # Create agent
            agent_state = await self.client.agents.create(
                name=agent_config.name,
                description=f"Construction claims analyst for {self.matter_name}",
                system=agent_config.system_prompt,
                llm_config=llm_config,
                embedding_config=embedding_config,
                memory_blocks=agent_config.memory_blocks
            )
            
            self.agent_id = agent_state.id
            self.agent_state = agent_state
            
            logger.info(
                "Created new Letta agent",
                agent_id=self.agent_id,
                matter_name=self.matter_name
            )
            
        except Exception as e:
            logger.error("Failed to create new agent", error=str(e))
            raise
    
    async def update_agent(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update agent configuration with new settings.
        
        Args:
            config_updates: Dictionary of configuration updates to apply
            
        Returns:
            True if update successful, False otherwise
        """
        # Ensure initialized
        if not await self._ensure_initialized():
            logger.warning("Letta not available, cannot update agent")
            return False
        
        if not self.agent_id:
            logger.warning("No agent to update")
            return False
        
        try:
            # Build update parameters
            update_params = {}
            
            # Handle LLM config updates
            if 'llm_model' in config_updates or 'llm_provider' in config_updates:
                agent_config = config_manager.get_agent_config(
                    matter_name=self.matter_name,
                    llm_provider=config_updates.get('llm_provider', 'ollama'),
                    llm_model=config_updates.get('llm_model', 'gpt-oss:20b')
                )
                
                update_params['llm_config'] = LlmConfig(
                    model=agent_config.llm_model,
                    model_endpoint_type=agent_config.llm_provider,
                    model_endpoint=agent_config.llm_endpoint,
                    context_window=agent_config.context_window,
                    max_tokens=agent_config.max_tokens,
                    temperature=config_updates.get('temperature', agent_config.temperature)
                )
            
            # Handle system prompt updates
            if 'system_prompt' in config_updates:
                update_params['system'] = config_updates['system_prompt']
            
            # Handle description updates
            if 'description' in config_updates:
                update_params['description'] = config_updates['description']
            
            # Update agent
            if update_params:
                await self.client.agents.modify(
                    agent_id=self.agent_id,
                    **update_params
                )
                
                # Update local config
                agent_config_path = self.letta_path / "agent_config.json"
                if agent_config_path.exists():
                    with open(agent_config_path, 'r') as f:
                        config = json.load(f)
                    
                    config['last_updated'] = datetime.now().isoformat()
                    config.update(config_updates)
                    
                    with open(agent_config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                
                logger.info(
                    "Agent updated successfully",
                    agent_id=self.agent_id,
                    updates=list(update_params.keys())
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to update agent", error=str(e))
            return False
    
    async def delete_agent(self) -> bool:
        """
        Delete the agent and clean up all associated data.
        
        Returns:
            True if deletion successful, False otherwise
        """
        # Ensure initialized
        if not await self._ensure_initialized():
            logger.warning("Letta not available, cleaning up local data only")
            # Still clean up local data
            return self._cleanup_local_data()
        
        if not self.agent_id:
            logger.warning("No agent to delete")
            return self._cleanup_local_data()
        
        try:
            # Delete agent from server
            await self.client.agents.delete(self.agent_id)
            
            logger.info(
                "Agent deleted from server",
                agent_id=self.agent_id,
                matter_name=self.matter_name
            )
            
            # Clean up local data
            self._cleanup_local_data()
            
            # Clear instance variables
            self.agent_id = None
            self.agent_state = None
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete agent from server", error=str(e))
            # Still clean up local data
            return self._cleanup_local_data()
    
    async def reload_agent(self) -> bool:
        """
        Reload agent from server to refresh state.
        
        Returns:
            True if reload successful, False otherwise
        """
        # Ensure initialized
        if not await self._ensure_initialized():
            logger.warning("Letta not available, cannot reload agent")
            return False
        
        if not self.agent_id:
            logger.warning("No agent to reload")
            return False
        
        try:
            # Get fresh agent state from server
            self.agent_state = await self.client.agents.retrieve(self.agent_id)
            
            logger.info(
                "Agent reloaded successfully",
                agent_id=self.agent_id,
                matter_name=self.matter_name
            )
            return True
            
        except Exception as e:
            logger.error("Failed to reload agent", error=str(e))
            return False
    
    async def get_agent_state(self) -> Dict[str, Any]:
        """
        Get current agent state and metadata.
        
        Returns:
            Dictionary containing agent state information
        """
        # Check local config first
        agent_config_path = self.letta_path / "agent_config.json"
        local_config = {}
        
        if agent_config_path.exists():
            with open(agent_config_path, 'r') as f:
                local_config = json.load(f)
        
        # Ensure initialized
        if not await self._ensure_initialized():
            return {
                "status": "offline",
                "agent_id": local_config.get('agent_id'),
                "matter_id": self.matter_id,
                "matter_name": self.matter_name,
                "created_at": local_config.get('created_at'),
                "last_updated": local_config.get('last_updated'),
                "fallback_mode": True,
                "connection_state": connection_manager.get_state().value
            }
        
        if not self.agent_id:
            return {
                "status": "not_created",
                "matter_id": self.matter_id,
                "matter_name": self.matter_name,
                "fallback_mode": self.fallback_mode
            }
        
        try:
            # Get fresh state from server if needed
            if not self.agent_state:
                self.agent_state = await self.client.agents.retrieve(self.agent_id)
            
            # Get memory stats
            memory_stats = await self.get_memory_stats()
            
            return {
                "status": "active",
                "agent_id": self.agent_id,
                "matter_id": self.matter_id,
                "matter_name": self.matter_name,
                "created_at": local_config.get('created_at'),
                "last_updated": local_config.get('last_updated'),
                "memory_items": memory_stats.get('memory_items', 0),
                "llm_model": self.agent_state.llm_config.model if self.agent_state else None,
                "fallback_mode": self.fallback_mode,
                "connection_state": connection_manager.get_state().value,
                "letta_version": local_config.get('letta_version'),
                "letta_client_version": local_config.get('letta_client_version')
            }
            
        except Exception as e:
            logger.error("Failed to get agent state", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "agent_id": self.agent_id,
                "matter_id": self.matter_id,
                "matter_name": self.matter_name
            }
    
    async def detect_old_agents(self) -> bool:
        """
        Detect if there are old LocalClient agents that need migration.
        
        Returns:
            True if old agents detected, False otherwise
        """
        try:
            # Check for SQLite databases from LocalClient
            sqlite_files = list(self.letta_path.glob("*.db")) + list(self.letta_path.glob("*.sqlite"))
            
            # Check for old config directory structure
            old_config_dir = self.letta_path / "config"
            
            # Check for old agent state files
            old_state_files = list(self.letta_path.glob("agent_*.json"))
            
            has_old_data = bool(sqlite_files or (old_config_dir.exists() and old_config_dir.is_dir()) or old_state_files)
            
            if has_old_data:
                logger.info(
                    "Detected old LocalClient agent data",
                    sqlite_files=len(sqlite_files),
                    has_config_dir=old_config_dir.exists(),
                    state_files=len(old_state_files)
                )
            
            return has_old_data
            
        except Exception as e:
            logger.error("Error detecting old agents", error=str(e))
            return False
    
    async def migrate_agent(self, old_data_path: Optional[Path] = None) -> bool:
        """
        Migrate old LocalClient agent data to new server-based format.
        
        Args:
            old_data_path: Path to old agent data (defaults to current letta_path)
            
        Returns:
            True if migration successful, False otherwise
        """
        old_path = old_data_path or self.letta_path
        
        try:
            # Backup existing data
            backup_path = await self.backup_agent()
            logger.info(f"Created backup at {backup_path}")
            
            # Extract memory from old SQLite databases
            memories = []
            sqlite_files = list(old_path.glob("*.db")) + list(old_path.glob("*.sqlite"))
            
            for db_file in sqlite_files:
                try:
                    conn = sqlite3.connect(str(db_file))
                    cursor = conn.cursor()
                    
                    # Try to extract archival memory (common table in LocalClient)
                    try:
                        cursor.execute("SELECT content FROM archival_memory;")
                        rows = cursor.fetchall()
                        memories.extend([row[0] for row in rows if row[0]])
                    except:
                        pass
                    
                    conn.close()
                except Exception as db_error:
                    logger.warning(f"Could not extract from {db_file.name}", error=str(db_error))
            
            # Create new agent if needed
            if not self.agent_id:
                await self._ensure_initialized()
            
            # Import extracted memories
            if memories and self.agent_id:
                logger.info(f"Migrating {len(memories)} memory items")
                
                for memory in memories:
                    try:
                        # Use new passages API to insert migrated memory
                        await self.client.agents.passages.create(
                            agent_id=self.agent_id,
                            text=memory
                        )
                    except Exception as mem_error:
                        logger.warning("Failed to migrate memory item", error=str(mem_error))
            
            logger.info(
                "Agent migration completed",
                memories_migrated=len(memories),
                backup_path=str(backup_path)
            )
            return True
            
        except Exception as e:
            logger.error("Agent migration failed", error=str(e))
            return False
    
    async def backup_agent(self, backup_name: Optional[str] = None) -> Path:
        """
        Create a backup of agent data.
        
        Args:
            backup_name: Optional name for backup (defaults to timestamp)
            
        Returns:
            Path to backup location
        """
        import shutil
        
        # Generate backup name
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create backup directory
        backup_path = self.letta_path.parent / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy all files from letta_path to backup
            if self.letta_path.exists():
                shutil.copytree(
                    self.letta_path,
                    backup_path / "letta_state",
                    dirs_exist_ok=True
                )
            
            # Export current agent memory if available
            if self.agent_id and await self._ensure_initialized():
                try:
                    # Use new passages API to get all memories for backup
                    memories = await self.client.agents.passages.list(
                        agent_id=self.agent_id,
                        limit=10000
                    )
                    
                    # Save memories to JSON
                    memory_export = {
                        "agent_id": self.agent_id,
                        "matter_name": self.matter_name,
                        "export_date": datetime.now().isoformat(),
                        "memory_count": len(memories) if memories else 0,
                        "memories": [
                            {
                                "text": p.text,
                                "id": p.id if hasattr(p, 'id') else None,
                                "created_at": str(p.created_at) if hasattr(p, 'created_at') else None
                            } 
                            for p in (memories or [])
                        ]
                    }
                    
                    with open(backup_path / "memory_export.json", 'w') as f:
                        json.dump(memory_export, f, indent=2, default=str)
                    
                except Exception as export_error:
                    logger.warning("Could not export agent memory", error=str(export_error))
            
            logger.info(f"Agent backup created at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error("Failed to create agent backup", error=str(e))
            raise
    
    async def restore_agent(self, backup_path: Path) -> bool:
        """
        Restore agent from backup.
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            True if restore successful, False otherwise
        """
        try:
            # Check if backup exists
            if not backup_path.exists():
                logger.error(f"Backup path does not exist: {backup_path}")
                return False
            
            # Load memory export if available
            memory_export_path = backup_path / "memory_export.json"
            if memory_export_path.exists():
                with open(memory_export_path, 'r') as f:
                    memory_export = json.load(f)
                
                # Create new agent if needed
                if not self.agent_id:
                    await self._ensure_initialized()
                
                # Restore memories
                if self.agent_id and 'memories' in memory_export:
                    logger.info(f"Restoring {len(memory_export['memories'])} memories")
                    
                    for memory in memory_export['memories']:
                        try:
                            # Extract text from the memory structure
                            if isinstance(memory, dict):
                                memory_text = memory.get('text') or memory.get('memory') or str(memory)
                            else:
                                memory_text = str(memory)
                            
                            # Use new passages API to restore memory
                            await self.client.agents.passages.create(
                                agent_id=self.agent_id,
                                text=memory_text
                            )
                        except Exception as mem_error:
                            logger.warning("Failed to restore memory item", error=str(mem_error))
            
            # Restore local configuration
            backup_letta_state = backup_path / "letta_state"
            if backup_letta_state.exists():
                import shutil
                shutil.copytree(
                    backup_letta_state,
                    self.letta_path,
                    dirs_exist_ok=True
                )
            
            logger.info(f"Agent restored from {backup_path}")
            return True
            
        except Exception as e:
            logger.error("Failed to restore agent", error=str(e))
            return False
    
    def _cleanup_local_data(self) -> bool:
        """
        Clean up local agent data files.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Remove agent config file
            agent_config_path = self.letta_path / "agent_config.json"
            if agent_config_path.exists():
                agent_config_path.unlink()
            
            # Remove any SQLite databases
            for db_file in self.letta_path.glob("*.db"):
                db_file.unlink()
            for db_file in self.letta_path.glob("*.sqlite"):
                db_file.unlink()
            
            # Remove config directory if exists
            config_dir = self.letta_path / "config"
            if config_dir.exists() and config_dir.is_dir():
                import shutil
                shutil.rmtree(config_dir)
            
            logger.info("Local agent data cleaned up", letta_path=str(self.letta_path))
            return True
            
        except Exception as e:
            logger.error("Failed to clean up local data", error=str(e))
            return False
    
    async def analyze_memory_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns and insights in stored memory.
        
        Returns:
            Dictionary with pattern analysis results
        """
        if not await self._ensure_initialized():
            return {"error": "Not initialized"}
        
        if not self.agent_id:
            return {"error": "No agent"}
        
        try:
            # Get all memories
            memories = await self.client.agents.passages.list(
                agent_id=self.agent_id,
                limit=10000
            )
            
            if not memories:
                return {"patterns": [], "insights": [], "total_memories": 0}
            
            # Analyze patterns
            type_counts = Counter()
            actor_counts = Counter()
            date_distribution = defaultdict(int)
            doc_references = Counter()
            co_occurrences = defaultdict(list)
            
            for passage in memories:
                try:
                    content = passage.text if hasattr(passage, 'text') else str(passage)
                    
                    if content.startswith('{'):
                        data = json.loads(content)
                        
                        # Count types
                        type_counts[data.get('type', 'Unknown')] += 1
                        
                        # Count actors
                        for actor in data.get('actors', []):
                            actor_counts[actor] += 1
                        
                        # Date distribution
                        if data.get('date'):
                            try:
                                date = datetime.fromisoformat(data['date'])
                                month_key = date.strftime('%Y-%m')
                                date_distribution[month_key] += 1
                            except:
                                pass
                        
                        # Document references
                        for ref in data.get('doc_refs', []):
                            doc_references[ref.get('doc', 'Unknown')] += 1
                        
                        # Track co-occurrences
                        label = data.get('label', '')
                        for actor in data.get('actors', []):
                            co_occurrences[actor].append(label)
                    
                except:
                    continue
            
            # Identify patterns
            patterns = []
            
            # Most common memory types
            if type_counts:
                most_common_type = type_counts.most_common(1)[0]
                patterns.append({
                    "type": "dominant_memory_type",
                    "value": most_common_type[0],
                    "count": most_common_type[1],
                    "percentage": (most_common_type[1] / len(memories)) * 100
                })
            
            # Key actors
            if actor_counts:
                key_actors = actor_counts.most_common(5)
                patterns.append({
                    "type": "key_actors",
                    "actors": [{"name": a[0], "mentions": a[1]} for a in key_actors]
                })
            
            # Temporal patterns
            if date_distribution:
                peak_month = max(date_distribution.items(), key=lambda x: x[1])
                patterns.append({
                    "type": "peak_activity",
                    "month": peak_month[0],
                    "count": peak_month[1]
                })
            
            # Document focus
            if doc_references:
                top_docs = doc_references.most_common(3)
                patterns.append({
                    "type": "primary_sources",
                    "documents": [{"name": d[0], "references": d[1]} for d in top_docs]
                })
            
            # Generate insights
            insights = []
            
            # Memory diversity
            type_diversity = len(type_counts) / 4  # Assuming 4 main types
            insights.append({
                "insight": "memory_diversity",
                "score": type_diversity,
                "interpretation": "High" if type_diversity > 0.75 else "Medium" if type_diversity > 0.5 else "Low"
            })
            
            # Actor network complexity
            if actor_counts:
                avg_connections = sum(len(v) for v in co_occurrences.values()) / len(actor_counts)
                insights.append({
                    "insight": "network_complexity",
                    "avg_connections": avg_connections,
                    "interpretation": "Complex" if avg_connections > 5 else "Moderate" if avg_connections > 2 else "Simple"
                })
            
            # Memory growth rate (if timestamps available)
            growth_rate = len(memories) / max(1, len(date_distribution))  # Memories per time period
            insights.append({
                "insight": "growth_rate",
                "rate": growth_rate,
                "interpretation": "Rapid" if growth_rate > 50 else "Steady" if growth_rate > 10 else "Slow"
            })
            
            return {
                "total_memories": len(memories),
                "patterns": patterns,
                "insights": insights,
                "type_distribution": dict(type_counts),
                "actor_network": dict(actor_counts.most_common(10)),
                "temporal_distribution": dict(date_distribution)
            }
            
        except Exception as e:
            logger.error("Pattern analysis failed", error=str(e))
            return {"error": str(e)}
    
    async def get_memory_quality_metrics(self) -> Dict[str, float]:
        """
        Calculate quality metrics for stored memory.
        
        Returns:
            Dictionary with quality scores
        """
        if not await self._ensure_initialized():
            return {"quality_score": 0.0}
        
        if not self.agent_id:
            return {"quality_score": 0.0}
        
        try:
            memories = await self.client.agents.passages.list(
                agent_id=self.agent_id,
                limit=1000
            )
            
            if not memories:
                return {"quality_score": 0.0}
            
            # Calculate quality metrics
            structured_count = 0
            has_support_count = 0
            has_references_count = 0
            has_date_count = 0
            avg_length = 0
            total_length = 0
            
            for passage in memories:
                try:
                    content = passage.text if hasattr(passage, 'text') else str(passage)
                    total_length += len(content)
                    
                    if content.startswith('{'):
                        data = json.loads(content)
                        structured_count += 1
                        
                        if data.get('support_snippet'):
                            has_support_count += 1
                        if data.get('doc_refs'):
                            has_references_count += 1
                        if data.get('date'):
                            has_date_count += 1
                except:
                    continue
            
            # Calculate scores
            structure_score = structured_count / len(memories)
            support_score = has_support_count / max(1, structured_count)
            reference_score = has_references_count / max(1, structured_count)
            temporal_score = has_date_count / max(1, structured_count)
            avg_length = total_length / len(memories)
            
            # Length score (optimal between 50-500 chars)
            if avg_length < 50:
                length_score = avg_length / 50
            elif avg_length > 500:
                length_score = max(0.3, 1 - (avg_length - 500) / 1000)
            else:
                length_score = 1.0
            
            # Overall quality score
            quality_score = (
                structure_score * 0.3 +
                support_score * 0.25 +
                reference_score * 0.25 +
                temporal_score * 0.1 +
                length_score * 0.1
            )
            
            return {
                "quality_score": round(quality_score, 3),
                "structure_score": round(structure_score, 3),
                "support_score": round(support_score, 3),
                "reference_score": round(reference_score, 3),
                "temporal_score": round(temporal_score, 3),
                "length_score": round(length_score, 3),
                "avg_memory_length": round(avg_length, 1),
                "total_memories": len(memories)
            }
            
        except Exception as e:
            logger.error("Quality metrics calculation failed", error=str(e))
            return {"quality_score": 0.0, "error": str(e)}
    
    # Helper methods for new features
    def _calculate_importance_score(self, item: KnowledgeItem) -> float:
        """Calculate importance score for a knowledge item."""
        score = 0.5  # Base score
        
        # Boost for having support
        if item.support_snippet:
            score += 0.2
        
        # Boost for having document references
        if item.doc_refs:
            score += 0.15
        
        # Boost for having actors
        if item.actors:
            score += 0.1
        
        # Boost for having date
        if item.date:
            score += 0.05
        
        # Type-based adjustments
        if item.type == "Issue":
            score += 0.1
        elif item.type == "Event":
            score += 0.05
        
        return min(1.0, score)
    
    def _hash_knowledge_item(self, item: KnowledgeItem) -> str:
        """Generate hash for a knowledge item for deduplication."""
        # Create consistent hash by sorting actors and normalizing content
        actors_str = ','.join(sorted(item.actors or []))
        content = f"{item.type}:{item.label}:{item.date or ''}:{actors_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _hash_memory(self, memory_text: str) -> str:
        """Generate hash for memory text."""
        return hashlib.md5(memory_text.encode()).hexdigest()
    
    async def _get_all_memories_cached(self) -> List[str]:
        """Get all memories with caching."""
        # Simple in-memory cache (would use Redis in production)
        cache_key = f"memories_{self.agent_id}"
        
        # For now, just fetch fresh
        memories = await self.client.agents.passages.list(
            agent_id=self.agent_id,
            limit=10000
        )
        
        return [p.text if hasattr(p, 'text') else str(p) for p in (memories or [])]
    
    async def _summarize_memory_block(self, content: str, max_size: int) -> str:
        """Summarize memory block content to fit size limit."""
        # Simple truncation for now - could use LLM summarization
        if len(content) <= max_size:
            return content
        
        # Keep beginning and end
        half_size = (max_size - 20) // 2
        return f"{content[:half_size]}\n...[truncated]...\n{content[-half_size:]}"
    
    def _parse_complex_query(self, query: str) -> List[str]:
        """Parse complex query with AND/OR logic into search terms."""
        # Simple implementation - split on OR
        if " OR " in query:
            return [term.strip() for term in query.split(" OR ")]
        return [query]
    
    def _fallback_followups(self) -> List[str]:
        """Fallback follow-up suggestions when Letta is unavailable."""
        return [
            "What additional documentation would strengthen this analysis?",
            "Are there any schedule impacts or delay claims related to this issue?", 
            "What are the potential damages or cost implications?",
            "Should we engage technical experts for further analysis?"
        ]
    
    def _check_existing_data(self) -> None:
        """
        Check for existing Letta data from previous versions.
        Logs warnings and provides guidance for data backup if needed.
        """
        try:
            # Check for existing agent config
            agent_config_path = self.letta_path / "agent_config.json"
            
            if agent_config_path.exists():
                with open(agent_config_path, 'r') as f:
                    config = json.load(f)
                    
                stored_version = config.get('letta_version', 'unknown')
                agent_id = config.get('agent_id')
                created_at = config.get('created_at', 'unknown')
                
                # Log existing data information
                logger.info(
                    "Found existing Letta agent data",
                    agent_id=agent_id,
                    stored_version=stored_version,
                    created_at=created_at,
                    current_version=LETTA_VERSION
                )
                
                # Check for version compatibility
                if LETTA_VERSION and stored_version != 'unknown':
                    if stored_version != LETTA_VERSION:
                        logger.warning(
                            "Letta version mismatch detected",
                            stored_version=stored_version,
                            current_version=LETTA_VERSION,
                            recommendation="Consider backing up agent data before proceeding"
                        )
                        
                        # Provide user guidance
                        backup_path = self.letta_path.parent / f"letta_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        logger.warning(
                            f"BACKUP RECOMMENDED: To preserve existing data, copy {self.letta_path} to {backup_path}"
                        )
            
            # Check for SQLite databases (common in Letta LocalClient)
            sqlite_files = list(self.letta_path.glob("*.db")) + list(self.letta_path.glob("*.sqlite"))
            if sqlite_files:
                logger.info(
                    "Found existing Letta database files",
                    database_count=len(sqlite_files),
                    databases=[str(f.name) for f in sqlite_files]
                )
                
                # Check if databases are accessible
                for db_file in sqlite_files:
                    try:
                        conn = sqlite3.connect(str(db_file))
                        cursor = conn.cursor()
                        # Try to get table list (non-destructive read)
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        conn.close()
                        
                        logger.debug(
                            f"Database {db_file.name} is accessible",
                            table_count=len(tables)
                        )
                    except Exception as db_error:
                        logger.warning(
                            f"Cannot access database {db_file.name}",
                            error=str(db_error),
                            recommendation="Database may be from incompatible version"
                        )
            
            # Check for config directory (Letta LocalClient uses this)
            config_dir = self.letta_path / "config"
            if config_dir.exists() and config_dir.is_dir():
                config_files = list(config_dir.glob("*"))
                if config_files:
                    logger.info(
                        "Found existing Letta config directory",
                        file_count=len(config_files)
                    )
                    
        except Exception as e:
            # Non-critical error - log but continue
            logger.debug(
                "Error checking existing data (non-critical)",
                error=str(e)
            )