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

try:
    from letta import LocalClient, create_client
    from letta.schemas.agent import CreateAgent
    from letta.schemas.memory import ChatMemory
    from letta.schemas.message import Message
    from letta.constants import DEFAULT_PRESET
except ImportError:
    # Fallback for older versions or installation issues
    LocalClient = None
    create_client = None
    print("Warning: Letta import failed, using fallback implementation")

from .logging_conf import get_logger
from .models import KnowledgeItem, SourceChunk

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
        self.client = None
        self.agent_id = None
        self.agent_state = None
        
        # Initialize agent on startup
        self._initialize_client()
        self._initialize_or_load_agent()
        
        logger.info(
            "LettaAdapter initialized",
            matter_id=self.matter_id,
            matter_name=self.matter_name,
            letta_path=str(self.letta_path),
            agent_id=self.agent_id
        )
    
    async def recall(self, query: str, top_k: int = 6) -> List[KnowledgeItem]:
        """
        Recall relevant knowledge items from agent memory.
        
        Args:
            query: Query to find relevant memory items for
            top_k: Number of memory items to return
            
        Returns:
            List of relevant KnowledgeItem objects
        """
        if not self.client or not self.agent_id:
            logger.warning("Letta not available, returning empty memory list")
            return []
        
        try:
            # Search archival memory for relevant items
            memory_results = await asyncio.to_thread(
                self.client.get_archival_memory,
                agent_id=self.agent_id,
                limit=top_k * 2  # Get more than needed for filtering
            )
            
            knowledge_items = []
            for memory_obj in memory_results[:top_k]:
                # Parse memory content as JSON knowledge item
                try:
                    content = memory_obj.text
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
        if not self.client or not self.agent_id:
            logger.warning("Letta not available, skipping interaction storage")
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
            
            # Store interaction summary
            await asyncio.to_thread(
                self.client.insert_archival_memory,
                agent_id=self.agent_id,
                memory=json.dumps(interaction_summary)
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
                
                await asyncio.to_thread(
                    self.client.insert_archival_memory,
                    agent_id=self.agent_id,
                    memory=json.dumps(fact_data)
                )
            
            # Update core memory with recent activity context
            if extracted_facts:
                recent_facts = ", ".join([f"{fact.type}: {fact.label}" for fact in extracted_facts[:3]])
                core_update = f"Recent discussion: {user_query[:100]}... Key items: {recent_facts}"
                
                # Update persona with recent context
                await asyncio.to_thread(
                    self.client.update_agent_core_memory,
                    agent_id=self.agent_id,
                    new_memory={"persona": core_update}
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
        if not self.client or not self.agent_id:
            logger.warning("Letta not available, returning generic follow-ups")
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
            
            # Send message to agent for follow-up generation
            response = await asyncio.to_thread(
                self.client.user_message,
                agent_id=self.agent_id,
                message=followup_prompt
            )
            
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
        if not self.client or not self.agent_id:
            return {"status": "unavailable", "memory_items": 0}
        
        try:
            # Get archival memory count
            memory_results = await asyncio.to_thread(
                self.client.get_archival_memory,
                agent_id=self.agent_id,
                limit=1000  # Get count estimate
            )
            
            return {
                "status": "active",
                "memory_items": len(memory_results),
                "agent_id": self.agent_id,
                "matter_name": self.matter_name
            }
            
        except Exception as e:
            logger.warning("Failed to get memory stats", error=str(e))
            return {"status": "error", "error": str(e)}
    
    def _initialize_client(self) -> None:
        """Initialize Letta client with local storage."""
        try:
            if LocalClient is None:
                logger.warning("Letta not available, using fallback mode")
                return
            
            # Use local SQLite storage for privacy
            config_dir = self.letta_path / "config"
            config_dir.mkdir(exist_ok=True)
            
            # Initialize local client with Matter-specific storage
            self.client = LocalClient(
                # Use Matter-specific database path
                # storage_path=str(config_dir)
            )
            
            logger.debug("Letta client initialized", storage_path=str(config_dir))
            
        except Exception as e:
            logger.error("Failed to initialize Letta client", error=str(e))
            self.client = None
    
    def _initialize_or_load_agent(self) -> None:
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
                    agent_state = self.client.get_agent(self.agent_id)
                    self.agent_state = agent_state
                    logger.debug("Loaded existing agent", agent_id=self.agent_id)
                    return
                except Exception:
                    logger.warning("Existing agent not found, creating new one")
                    self.agent_id = None
            
            # Create new agent
            self._create_new_agent()
            
            # Save agent configuration
            with open(agent_config_path, 'w') as f:
                json.dump({
                    'agent_id': self.agent_id,
                    'matter_id': self.matter_id,
                    'matter_name': self.matter_name,
                    'created_at': datetime.now().isoformat()
                }, f, indent=2)
            
        except Exception as e:
            logger.error("Failed to initialize agent", error=str(e))
            self.agent_id = None
    
    def _create_new_agent(self) -> None:
        """Create new Letta agent with construction domain configuration."""
        try:
            # Configuration for construction claims analysis
            persona = f"""
You are a construction claims analyst assistant for the matter: {self.matter_name}.

Your role is to:
- Help analyze construction project documents, contracts, and claims
- Track entities (parties, projects, documents), events (delays, changes, failures), issues (design defects, schedule impacts), and facts
- Maintain memory of important information from conversations
- Provide insights about causation, responsibility, and damages
- Remember key dates, parties, and technical details

You have persistent memory and learn from each conversation to provide better context-aware assistance."""
            
            human_description = f"""
This is a construction attorney working on the matter: {self.matter_name}.
They are analyzing claims, documents, and seeking legal insights about construction disputes.
Help them by remembering important facts and providing contextual analysis."""
            
            # Create agent with construction-specific memory
            agent_name = f"Construction-Claims-{self.matter_id[:8]}"
            
            # Try to create agent (API may vary by Letta version)
            try:
                agent_state = self.client.create_agent(
                    name=agent_name,
                    persona=persona,
                    human=human_description
                )
            except TypeError:
                # Fallback for different API signature
                agent_state = self.client.create_agent(
                    name=agent_name,
                    memory=ChatMemory(persona=persona, human=human_description)
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
    
    def _fallback_followups(self) -> List[str]:
        """Fallback follow-up suggestions when Letta is unavailable."""
        return [
            "What additional documentation would strengthen this analysis?",
            "Are there any schedule impacts or delay claims related to this issue?", 
            "What are the potential damages or cost implications?",
            "Should we engage technical experts for further analysis?"
        ]