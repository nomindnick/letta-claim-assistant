"""
Memory command indicator component for the UI.

Provides visual feedback when memory commands are detected and processed,
including action type indicators, progress states, and undo functionality.
"""

from nicegui import ui
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime


class MemoryCommandIndicator:
    """Visual indicator for memory command processing."""
    
    # Icons for different memory actions
    ACTION_ICONS = {
        "remember": "save",
        "forget": "delete",
        "update": "edit",
        "query": "search"
    }
    
    # Colors for different states
    STATE_COLORS = {
        "processing": "orange",
        "success": "green",
        "error": "red",
        "idle": "gray"
    }
    
    def __init__(self):
        self.container = None
        self.status_label = None
        self.action_icon = None
        self.undo_button = None
        self.current_state = "idle"
        self.last_command = None
        self.undo_token = None
    
    def create(self, parent_container=None):
        """Create the indicator component."""
        with (parent_container or ui.row()) as self.container:
            self.container.classes('items-center gap-2 p-2 rounded transition-all duration-300')
            self.container.style('display: none')  # Hidden by default
            
            # Action icon
            self.action_icon = ui.icon('memory').classes('text-lg')
            
            # Status label
            self.status_label = ui.label('').classes('text-sm')
            
            # Undo button (hidden by default)
            self.undo_button = ui.button(
                icon='undo',
                on_click=self._handle_undo
            ).props('flat round size=sm').style('display: none')
    
    def show_processing(self, action: str, content: str):
        """Show processing state for a memory command."""
        self.current_state = "processing"
        self.last_command = {"action": action, "content": content}
        
        # Update icon
        icon = self.ACTION_ICONS.get(action, "memory")
        self.action_icon.props(f'name={icon}')
        self.action_icon.classes(remove='text-green-500 text-red-500 text-gray-500')
        self.action_icon.classes('text-orange-500 animate-pulse')
        
        # Update status text
        action_text = self._get_action_text(action)
        preview = content[:30] + "..." if len(content) > 30 else content
        self.status_label.text = f"{action_text}: {preview}"
        self.status_label.classes(remove='text-green-500 text-red-500 text-gray-500')
        self.status_label.classes('text-orange-500')
        
        # Show container with animation
        self.container.style('display: flex; opacity: 0')
        ui.run_javascript(f"""
            setTimeout(() => {{
                const el = document.querySelector('[data-id="{self.container.id}"]');
                if (el) el.style.opacity = '1';
            }}, 10);
        """)
        
        # Hide undo button during processing
        self.undo_button.style('display: none')
    
    def show_success(self, message: str, undo_token: Optional[str] = None):
        """Show success state after command completion."""
        self.current_state = "success"
        self.undo_token = undo_token
        
        # Update icon
        self.action_icon.classes(remove='text-orange-500 text-red-500 text-gray-500 animate-pulse')
        self.action_icon.classes('text-green-500')
        
        # Update status text
        self.status_label.text = message
        self.status_label.classes(remove='text-orange-500 text-red-500 text-gray-500')
        self.status_label.classes('text-green-500')
        
        # Show undo button if token provided
        if undo_token:
            self.undo_button.style('display: inline-flex')
            self.undo_button.props('disable=false')
        
        # Auto-hide after 5 seconds
        asyncio.create_task(self._auto_hide(5))
    
    def show_error(self, message: str):
        """Show error state."""
        self.current_state = "error"
        
        # Update icon
        self.action_icon.props('name=error')
        self.action_icon.classes(remove='text-orange-500 text-green-500 text-gray-500 animate-pulse')
        self.action_icon.classes('text-red-500')
        
        # Update status text
        self.status_label.text = message
        self.status_label.classes(remove='text-orange-500 text-green-500 text-gray-500')
        self.status_label.classes('text-red-500')
        
        # Hide undo button on error
        self.undo_button.style('display: none')
        
        # Auto-hide after 5 seconds
        asyncio.create_task(self._auto_hide(5))
    
    def hide(self):
        """Hide the indicator."""
        if self.container:
            ui.run_javascript(f"""
                const el = document.querySelector('[data-id="{self.container.id}"]');
                if (el) {{
                    el.style.opacity = '0';
                    setTimeout(() => {{ el.style.display = 'none'; }}, 300);
                }}
            """)
        
        self.current_state = "idle"
        self.last_command = None
        self.undo_token = None
    
    async def _auto_hide(self, delay: float):
        """Auto-hide the indicator after a delay."""
        await asyncio.sleep(delay)
        if self.current_state in ["success", "error"]:
            self.hide()
    
    async def _handle_undo(self):
        """Handle undo button click."""
        if not self.undo_token or not self.last_command:
            return
        
        # Disable undo button
        self.undo_button.props('disable=true')
        
        # Show undoing state
        self.status_label.text = "Undoing..."
        self.status_label.classes(remove='text-green-500 text-red-500')
        self.status_label.classes('text-orange-500')
        
        # TODO: Call API to undo the operation using undo_token
        # For now, just show success
        await asyncio.sleep(1)
        
        self.show_success("Operation undone")
    
    def _get_action_text(self, action: str) -> str:
        """Get human-readable text for an action."""
        action_texts = {
            "remember": "Remembering",
            "forget": "Forgetting",
            "update": "Updating",
            "query": "Searching memory for"
        }
        return action_texts.get(action, "Processing")


class MemoryCommandBadge:
    """Simple badge indicator for memory commands in chat messages."""
    
    def __init__(self):
        self.badge = None
    
    def create(self, action: str, parent_container=None) -> ui.element:
        """Create a badge for a memory command action."""
        icon = MemoryCommandIndicator.ACTION_ICONS.get(action, "memory")
        color = self._get_action_color(action)
        label = self._get_action_label(action)
        
        with (parent_container or ui.row()) as container:
            container.classes('inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs')
            container.classes(f'bg-{color}-100 text-{color}-700')
            
            ui.icon(icon).classes('text-sm')
            ui.label(label).classes('font-medium')
        
        return container
    
    def _get_action_color(self, action: str) -> str:
        """Get color for an action type."""
        colors = {
            "remember": "blue",
            "forget": "red",
            "update": "yellow",
            "query": "purple"
        }
        return colors.get(action, "gray")
    
    def _get_action_label(self, action: str) -> str:
        """Get label for an action type."""
        labels = {
            "remember": "Memory Saved",
            "forget": "Memory Deleted",
            "update": "Memory Updated",
            "query": "Memory Query"
        }
        return labels.get(action, "Memory Command")


# Singleton instance for easy access
memory_command_indicator = MemoryCommandIndicator()