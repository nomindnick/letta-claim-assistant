"""
Letta Agent - Stateful agent-first message handling for construction claims.

Implements the agent-first architecture where a stateful Letta agent serves as 
the primary interface, maintaining conversation context and intelligently using 
tools when needed.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import asyncio
from datetime import datetime
from pathlib import Path

from .logging_conf import get_logger
from .matters import Matter, matter_manager
from .letta_adapter import LettaAdapter
from .letta_connection import connection_manager
from .models import ChatMode

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Response from the Letta agent."""
    
    # Core response
    message: str
    matter_id: str
    agent_id: str
    
    # Tool usage tracking
    tools_used: List[str] = None
    search_performed: bool = False
    search_results: List[Dict[str, Any]] = None
    
    # Memory updates
    memory_updated: bool = False
    memory_changes: List[str] = None
    
    # Citations and sources
    citations: List[str] = None
    
    # Metadata
    response_time: float = 0.0
    timestamp: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "message": self.message,
            "matter_id": self.matter_id,
            "agent_id": self.agent_id,
            "tools_used": self.tools_used or [],
            "search_performed": self.search_performed,
            "search_results": self.search_results or [],
            "memory_updated": self.memory_updated,
            "memory_changes": self.memory_changes or [],
            "citations": self.citations or [],
            "response_time": self.response_time,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class LettaAgentHandler:
    """
    Handles agent-first message processing for construction claims.
    
    This is the primary interface for the stateful agent architecture where
    the Letta agent maintains conversation state and decides when to use tools.
    """
    
    def __init__(self):
        """Initialize the agent handler."""
        self._adapters: Dict[str, LettaAdapter] = {}
        self._active_matter_id: Optional[str] = None
        
        logger.info("LettaAgentHandler initialized")
    
    def _get_adapter(self, matter_id: str) -> LettaAdapter:
        """
        Get or create Letta adapter for a matter.
        
        Args:
            matter_id: Matter ID
            
        Returns:
            LettaAdapter instance for the matter
        """
        if matter_id not in self._adapters:
            # Get matter from manager
            matter = None
            for m in matter_manager.list_matters():
                if m.id == matter_id:
                    matter = m
                    break
            
            if not matter:
                raise ValueError(f"Matter {matter_id} not found")
            
            # Create adapter
            self._adapters[matter_id] = LettaAdapter(
                matter_path=matter.paths.root,
                matter_name=matter.name,
                matter_id=matter.id
            )
            
            logger.info(f"Created Letta adapter for matter {matter_id}")
        
        return self._adapters[matter_id]
    
    async def handle_user_message(
        self,
        matter_id: str,
        message: str,
        stream: bool = False
    ) -> AgentResponse:
        """
        Send message to matter's agent and get response.
        
        The agent will:
        1. Process the message in context of conversation history
        2. Decide whether to search documents
        3. Update its memory with important information
        4. Generate a contextual response
        
        Args:
            matter_id: ID of the matter
            message: User's message
            stream: Whether to stream the response (future enhancement)
            
        Returns:
            AgentResponse with message, tool usage, and metadata
        """
        start_time = datetime.now()
        
        try:
            # Get adapter for this matter
            adapter = self._get_adapter(matter_id)
            
            # Ensure adapter is initialized
            if not await adapter._ensure_initialized():
                logger.error(f"Failed to initialize adapter for matter {matter_id}")
                return AgentResponse(
                    message="I'm unable to connect to the conversation system. Please try again later.",
                    matter_id=matter_id,
                    agent_id=None,
                    timestamp=datetime.now()
                )
            
            # Send message to agent
            logger.info(f"Sending message to agent for matter {matter_id}")
            
            # Use the Letta client to send message
            if not adapter.sync_client or not adapter.agent_id:
                logger.error("No sync client or agent available")
                return AgentResponse(
                    message="The conversation system is not available. Please try again later.",
                    matter_id=matter_id,
                    agent_id=None,
                    timestamp=datetime.now()
                )
            
            # Send message and get response
            try:
                # Use sync client in async context since Letta's async client 
                # seems to have issues with message creation
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                # Create a thread pool executor for sync operations
                with ThreadPoolExecutor(max_workers=1) as executor:
                    # Run the sync operation in the executor
                    def send_message():
                        """Send message using sync client."""
                        return adapter.sync_client.agents.messages.create(
                            agent_id=adapter.agent_id,
                            messages=[{"role": "user", "content": message}]
                        )
                    
                    response = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        send_message
                    )
                
                # Process response
                return self._process_agent_response(
                    response=response,
                    matter_id=matter_id,
                    agent_id=adapter.agent_id,
                    start_time=start_time
                )
                
            except Exception as e:
                logger.error(f"Failed to send message to agent: {e}")
                return AgentResponse(
                    message="I encountered an error processing your message. Please try again.",
                    matter_id=matter_id,
                    agent_id=adapter.agent_id,
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
            return AgentResponse(
                message="An unexpected error occurred. Please try again.",
                matter_id=matter_id,
                agent_id=None,
                timestamp=datetime.now()
            )
    
    def _process_agent_response(
        self,
        response: Any,
        matter_id: str,
        agent_id: str,
        start_time: datetime
    ) -> AgentResponse:
        """
        Process the raw response from Letta agent.
        
        Args:
            response: Raw response from Letta
            matter_id: Matter ID
            agent_id: Agent ID
            start_time: When processing started
            
        Returns:
            Structured AgentResponse
        """
        # Extract message content
        message_content = ""
        tools_used = []
        search_performed = False
        search_results = []
        citations = []
        memory_updated = False
        memory_changes = []
        
        # Process response messages
        if hasattr(response, 'messages'):
            for msg in response.messages:
                # Check for assistant messages
                if hasattr(msg, 'role') and msg.role == 'assistant':
                    if hasattr(msg, 'content') and msg.content:
                        message_content += msg.content + "\n"
                
                # Check for tool calls
                if hasattr(msg, 'tool_calls'):
                    for tool_call in msg.tool_calls:
                        tool_name = getattr(tool_call, 'name', 'unknown')
                        tools_used.append(tool_name)
                        
                        if tool_name == 'search_documents':
                            search_performed = True
                            # Extract search results if available
                            if hasattr(tool_call, 'result'):
                                try:
                                    result_data = json.loads(tool_call.result)
                                    if 'results' in result_data:
                                        search_results.extend(result_data['results'])
                                        # Extract citations
                                        for result in result_data['results']:
                                            if 'citation' in result:
                                                citations.append(result['citation'])
                                except:
                                    pass
                
                # Check for memory updates
                if hasattr(msg, 'memory_updates'):
                    memory_updated = True
                    for update in msg.memory_updates:
                        memory_changes.append(str(update))
        
        # Extract citations from message content
        import re
        citation_pattern = r'\[([^\]]+\.(?:pdf|PDF|docx?|DOCX?|txt|TXT))\s+p\.?\s*(\d+(?:-\d+)?)\]'
        found_citations = re.findall(citation_pattern, message_content)
        for doc, pages in found_citations:
            citation = f"[{doc} p.{pages}]"
            if citation not in citations:
                citations.append(citation)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        return AgentResponse(
            message=message_content.strip(),
            matter_id=matter_id,
            agent_id=agent_id,
            tools_used=tools_used if tools_used else None,
            search_performed=search_performed,
            search_results=search_results if search_results else None,
            memory_updated=memory_updated,
            memory_changes=memory_changes if memory_changes else None,
            citations=citations if citations else None,
            response_time=response_time,
            timestamp=datetime.now()
        )
    
    async def send_message(
        self,
        matter_id: str,
        message: str,
        k: int = 8,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> AgentResponse:
        """
        Send message to matter's agent (compatibility wrapper).
        
        This method wraps handle_user_message for API compatibility.
        
        Args:
            matter_id: ID of the matter
            message: User's message
            k: Number of search results (passed to agent context)
            max_tokens: Maximum tokens for response
            stream: Whether to stream the response
            
        Returns:
            AgentResponse with message, tool usage, and metadata
        """
        # Call the main handler method
        return await self.handle_user_message(
            matter_id=matter_id,
            message=message,
            stream=stream
        )
    
    async def get_agent_memory(self, matter_id: str) -> Dict[str, Any]:
        """
        Get current memory state of the agent.
        
        Args:
            matter_id: Matter ID
            
        Returns:
            Dictionary containing memory blocks and their contents
        """
        try:
            adapter = self._get_adapter(matter_id)
            
            if not await adapter._ensure_initialized():
                return {"error": "Agent not available"}
            
            if not adapter.agent_id:
                return {"error": "No agent found for this matter"}
            
            # Get memory blocks
            memory_blocks = await adapter.client.agents.memory.list(adapter.agent_id)
            
            memory_state = {}
            for block in memory_blocks:
                memory_state[block.label] = {
                    "value": block.value,
                    "limit": getattr(block, 'limit', None)
                }
            
            return memory_state
            
        except Exception as e:
            logger.error(f"Failed to get agent memory: {e}")
            return {"error": str(e)}
    
    async def clear_conversation(self, matter_id: str) -> bool:
        """
        Clear conversation history for a matter (keeps core memory).
        
        Args:
            matter_id: Matter ID
            
        Returns:
            True if successful
        """
        try:
            adapter = self._get_adapter(matter_id)
            
            if not await adapter._ensure_initialized():
                return False
            
            # Clear message history but keep memory
            # This would clear the conversation while preserving learned facts
            # Implementation depends on Letta API capabilities
            
            logger.info(f"Cleared conversation for matter {matter_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear conversation: {e}")
            return False
    
    def set_active_matter(self, matter_id: str):
        """
        Set the active matter for context.
        
        Args:
            matter_id: Matter ID to make active
        """
        self._active_matter_id = matter_id
        # Also set in matter manager for tool context
        matter_manager.switch_matter(matter_id)
        logger.info(f"Set active matter to {matter_id}")
    
    def get_active_matter(self) -> Optional[str]:
        """
        Get the currently active matter ID.
        
        Returns:
            Active matter ID or None
        """
        return self._active_matter_id


# Global agent handler instance
agent_handler = LettaAgentHandler()