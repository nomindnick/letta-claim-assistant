"""
Chat history management for matter-specific conversation storage.

Handles persistence and retrieval of chat messages with metadata
for maintaining conversation context across sessions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .logging_conf import get_logger
from .models import Matter

logger = get_logger(__name__)


class ChatMessage:
    """Chat message with metadata for storage."""
    
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        followups: Optional[List[str]] = None,
        used_memory: Optional[List[Dict[str, Any]]] = None,
        query_metadata: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.sources = sources or []
        self.followups = followups or []
        self.used_memory = used_memory or []
        self.query_metadata = query_metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "sources": self.sources,
            "followups": self.followups,
            "used_memory": self.used_memory,
            "query_metadata": self.query_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary loaded from JSON."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sources=data.get("sources", []),
            followups=data.get("followups", []),
            used_memory=data.get("used_memory", []),
            query_metadata=data.get("query_metadata", {})
        )


class ChatHistoryManager:
    """Manages chat history for matters."""
    
    def __init__(self, matter: Matter):
        self.matter = matter
        self.history_file = matter.paths.chat / "history.jsonl"
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure chat directory exists."""
        self.matter.paths.chat.mkdir(parents=True, exist_ok=True)
    
    async def save_interaction(
        self,
        user_query: str,
        assistant_response: str,
        sources: List[Dict[str, Any]],
        followups: List[str],
        used_memory: List[Dict[str, Any]],
        query_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a complete user-assistant interaction.
        
        Args:
            user_query: User's question
            assistant_response: Assistant's answer
            sources: Source chunks used in response
            followups: Generated follow-up suggestions
            used_memory: Memory items used from Letta
            query_metadata: Additional metadata about the query
        """
        timestamp = datetime.now()
        
        # Create user message
        user_message = ChatMessage(
            role="user",
            content=user_query,
            timestamp=timestamp,
            query_metadata=query_metadata
        )
        
        # Create assistant message
        assistant_message = ChatMessage(
            role="assistant",
            content=assistant_response,
            timestamp=timestamp,
            sources=sources,
            followups=followups,
            used_memory=used_memory,
            query_metadata=query_metadata
        )
        
        # Save both messages
        await self._append_message(user_message)
        await self._append_message(assistant_message)
        
        logger.info(
            "Saved chat interaction",
            matter_id=self.matter.id,
            user_query_length=len(user_query),
            response_length=len(assistant_response),
            sources_count=len(sources),
            followups_count=len(followups)
        )
    
    async def _append_message(self, message: ChatMessage) -> None:
        """Append a single message to the history file."""
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                json.dump(message.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            logger.error("Failed to save chat message", error=str(e))
            raise
    
    async def load_history(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """
        Load chat history from file.
        
        Args:
            limit: Maximum number of messages to return (most recent)
            
        Returns:
            List of chat messages in chronological order
        """
        if not self.history_file.exists():
            logger.info("No chat history found", matter_id=self.matter.id)
            return []
        
        try:
            messages = []
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            message = ChatMessage.from_dict(data)
                            messages.append(message)
                        except json.JSONDecodeError as e:
                            logger.warning("Skipping invalid JSON line in history", error=str(e))
                            continue
            
            # Apply limit if specified
            if limit and len(messages) > limit:
                messages = messages[-limit:]
            
            logger.info(
                "Loaded chat history",
                matter_id=self.matter.id,
                message_count=len(messages)
            )
            return messages
            
        except Exception as e:
            logger.error("Failed to load chat history", error=str(e))
            return []
    
    async def clear_history(self) -> None:
        """Clear all chat history for this matter."""
        if self.history_file.exists():
            self.history_file.unlink()
        
        logger.info("Cleared chat history", matter_id=self.matter.id)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the conversation."""
        if not self.history_file.exists():
            return {
                "message_count": 0,
                "last_activity": None,
                "total_queries": 0,
                "avg_response_length": 0
            }
        
        try:
            messages = []
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            messages.append(data)
                        except json.JSONDecodeError:
                            continue
            
            if not messages:
                return {
                    "message_count": 0,
                    "last_activity": None,
                    "total_queries": 0,
                    "avg_response_length": 0
                }
            
            # Calculate statistics
            user_messages = [m for m in messages if m.get("role") == "user"]
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            
            last_message = messages[-1]
            last_activity = datetime.fromisoformat(last_message["timestamp"])
            
            avg_response_length = 0
            if assistant_messages:
                total_length = sum(len(m.get("content", "")) for m in assistant_messages)
                avg_response_length = total_length / len(assistant_messages)
            
            return {
                "message_count": len(messages),
                "last_activity": last_activity.isoformat(),
                "total_queries": len(user_messages),
                "avg_response_length": int(avg_response_length)
            }
            
        except Exception as e:
            logger.error("Failed to get conversation summary", error=str(e))
            return {
                "message_count": 0,
                "last_activity": None,
                "total_queries": 0,
                "avg_response_length": 0
            }


def get_chat_history_manager(matter: Matter) -> ChatHistoryManager:
    """Get chat history manager for a matter."""
    return ChatHistoryManager(matter)