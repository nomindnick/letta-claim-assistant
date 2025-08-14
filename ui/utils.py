"""
Utility functions for the UI layer.

Provides PDF viewer integration, clipboard operations, and other
helper functions for the NiceGUI interface.
"""

import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
from nicegui import ui

import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.logging_conf import get_logger

logger = get_logger(__name__)


class PDFViewerError(Exception):
    """Raised when PDF viewer operations fail."""
    pass


async def open_pdf_at_page(doc_path: Path, page: int) -> bool:
    """
    Open PDF document at specific page using system viewer.
    
    Args:
        doc_path: Path to PDF file
        page: Page number (1-based)
        
    Returns:
        True if opened successfully, False otherwise
    """
    if not doc_path.exists():
        logger.error("PDF file not found", path=str(doc_path))
        ui.notify(f"PDF file not found: {doc_path.name}", type="negative")
        return False
    
    try:
        # Try different PDF viewers in order of preference
        viewers = [
            # Evince with page support
            ["evince", "--page-index", str(page - 1), str(doc_path)],
            # Okular with page support  
            ["okular", "--page", str(page), str(doc_path)],
            # Atril (MATE PDF viewer)
            ["atril", "--page-index", str(page - 1), str(doc_path)],
            # Generic fallback
            ["xdg-open", str(doc_path)]
        ]
        
        for viewer_cmd in viewers:
            viewer_name = viewer_cmd[0]
            
            # Check if viewer is available
            if shutil.which(viewer_name):
                logger.info("Opening PDF", viewer=viewer_name, page=page, path=str(doc_path))
                
                # Launch viewer in background
                process = await asyncio.create_subprocess_exec(
                    *viewer_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                
                # Don't wait for viewer to close
                if viewer_name == "xdg-open":
                    ui.notify(f"Opened {doc_path.name} (page navigation may not be supported)", type="info")
                else:
                    ui.notify(f"Opened {doc_path.name} at page {page}", type="positive")
                
                return True
        
        # No viewers found
        logger.warning("No PDF viewers available")
        ui.notify("No PDF viewer found. Please install evince, okular, or another PDF viewer.", type="negative")
        return False
        
    except Exception as e:
        logger.error("Failed to open PDF", error=str(e), path=str(doc_path))
        ui.notify(f"Failed to open PDF: {str(e)}", type="negative")
        return False


async def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard.
    
    Args:
        text: Text to copy
        
    Returns:
        True if copied successfully, False otherwise
    """
    try:
        # Try xclip first (most common on Linux)
        if shutil.which("xclip"):
            process = await asyncio.create_subprocess_exec(
                "xclip", "-selection", "clipboard",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.communicate(input=text.encode())
            
        # Try xsel as fallback
        elif shutil.which("xsel"):
            process = await asyncio.create_subprocess_exec(
                "xsel", "--clipboard", "--input",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.communicate(input=text.encode())
            
        else:
            logger.warning("No clipboard utilities available")
            ui.notify("Clipboard not available. Please install xclip or xsel.", type="negative")
            return False
        
        ui.notify("Copied to clipboard", type="positive")
        return True
        
    except Exception as e:
        logger.error("Failed to copy to clipboard", error=str(e))
        ui.notify(f"Failed to copy: {str(e)}", type="negative")
        return False


class ChatMessage:
    """Represents a chat message with metadata."""
    
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        followups: Optional[List[str]] = None,
        used_memory: Optional[List[Dict[str, Any]]] = None
    ):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.sources = sources or []
        self.followups = followups or []
        self.used_memory = used_memory or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "sources": self.sources,
            "followups": self.followups,
            "used_memory": self.used_memory
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
            used_memory=data.get("used_memory", [])
        )


class ChatHistory:
    """Manages chat message history for a matter."""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.messages: List[ChatMessage] = []
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure history file directory exists."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def load_history(self) -> List[ChatMessage]:
        """Load chat history from file."""
        if not self.history_file.exists():
            logger.info("No chat history file found", path=str(self.history_file))
            return []
        
        try:
            messages = []
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        message = ChatMessage.from_dict(data)
                        messages.append(message)
            
            self.messages = messages
            logger.info("Loaded chat history", message_count=len(messages))
            return messages
            
        except Exception as e:
            logger.error("Failed to load chat history", error=str(e))
            return []
    
    async def add_message(self, message: ChatMessage) -> None:
        """Add message to history and save to file."""
        self.messages.append(message)
        await self._save_message(message)
    
    async def _save_message(self, message: ChatMessage) -> None:
        """Append message to history file."""
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                json.dump(message.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            logger.error("Failed to save message", error=str(e))
    
    async def clear_history(self) -> None:
        """Clear all chat history."""
        self.messages = []
        if self.history_file.exists():
            self.history_file.unlink()
        logger.info("Cleared chat history", path=str(self.history_file))
    
    def get_recent_messages(self, limit: int = 50) -> List[ChatMessage]:
        """Get recent messages up to limit."""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages


def format_citation(doc_name: str, page_start: int, page_end: int) -> str:
    """Format citation for copy to clipboard."""
    if page_start == page_end:
        return f"[{doc_name} p.{page_start}]"
    else:
        return f"[{doc_name} p.{page_start}-{page_end}]"


def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display in chat."""
    now = datetime.now()
    diff = now - dt
    
    if diff.days > 0:
        return dt.strftime("%m/%d %H:%M")
    elif diff.seconds > 3600:  # More than 1 hour
        return dt.strftime("%H:%M")
    else:  # Less than 1 hour
        return dt.strftime("%H:%M")


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe display."""
    # Remove potentially dangerous characters
    import re
    return re.sub(r'[<>:"/\\|?*]', '_', filename)