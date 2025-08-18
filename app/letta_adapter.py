"""
Letta integration for persistent agent memory.

Provides matter-specific agent knowledge management with recall,
upsert operations, and follow-up suggestion generation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
import os
import sqlite3
import uuid
from datetime import datetime
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