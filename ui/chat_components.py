"""
Chat-related UI components for the Letta Construction Claim Assistant.

Provides components for chat mode selection and chat interface enhancements.
"""

from nicegui import ui
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ChatMode(str, Enum):
    """Chat mode options matching backend."""
    RAG_ONLY = "rag"
    MEMORY_ONLY = "memory"
    COMBINED = "combined"


@dataclass
class ModeInfo:
    """Information about a chat mode."""
    value: str
    label: str
    icon: str
    color: str
    tooltip: str
    description: str


class ChatModeSelector:
    """
    Component for selecting chat mode (RAG-only, Memory-only, or Combined).
    
    Provides a radio button group with tooltips and visual indicators
    for each mode, allowing users to control how the AI processes queries.
    """
    
    # Mode definitions with all display information
    MODES = {
        ChatMode.RAG_ONLY: ModeInfo(
            value=ChatMode.RAG_ONLY,
            label="Documents Only",
            icon="description",
            color="blue",
            tooltip="Search only your uploaded PDF documents",
            description="Searches through your uploaded PDFs to find relevant passages. Best for finding specific document references."
        ),
        ChatMode.MEMORY_ONLY: ModeInfo(
            value=ChatMode.MEMORY_ONLY,
            label="Memory Only",
            icon="psychology",
            color="purple",
            tooltip="Use only the agent's learned knowledge (faster)",
            description="Uses only the agent's accumulated memory about this matter. Faster responses, best for recalling established facts."
        ),
        ChatMode.COMBINED: ModeInfo(
            value=ChatMode.COMBINED,
            label="Combined",
            icon="merge_type",
            color="green",
            tooltip="Search both documents and memory (recommended)",
            description="Searches both documents and agent memory. Provides the most comprehensive answers by combining all available information."
        )
    }
    
    def __init__(self, 
                 default_mode: str = ChatMode.COMBINED,
                 on_change: Optional[Callable[[str], None]] = None):
        """
        Initialize the chat mode selector.
        
        Args:
            default_mode: Initial mode selection
            on_change: Callback when mode changes
        """
        self.current_mode = default_mode
        self.on_change = on_change
        self.container = None
        self.mode_buttons = {}
        self.mode_description_label = None
        
    def create(self) -> ui.element:
        """Create and return the mode selector UI element."""
        with ui.card().classes('w-full p-3 mb-3') as self.container:
            # Header
            with ui.row().classes('w-full items-center mb-2'):
                ui.icon('tune', size='sm').classes('text-gray-600')
                ui.label('Chat Mode').classes('font-semibold text-sm')
                
                # Help icon with explanation
                with ui.icon('help_outline', size='xs').classes('ml-2 text-gray-400 cursor-help'):
                    ui.tooltip('Control how the AI processes your questions')
            
            # Radio button group
            with ui.row().classes('w-full gap-3 mb-2'):
                for mode, info in self.MODES.items():
                    self._create_mode_button(mode, info)
            
            # Description area
            self.mode_description_label = ui.label(
                self.MODES[self.current_mode].description
            ).classes('text-xs text-gray-600 italic')
            
        return self.container
    
    def _create_mode_button(self, mode: str, info: ModeInfo) -> None:
        """Create a single mode selection button."""
        is_selected = mode == self.current_mode
        
        with ui.button(
            on_click=lambda m=mode: self._on_mode_selected(m)
        ).props(
            f"{'color=primary' if is_selected else 'outline'} dense"
        ).classes(
            f"flex-1 transition-all {'ring-2 ring-primary' if is_selected else ''}"
        ) as btn:
            with ui.row().classes('items-center gap-1 justify-center'):
                ui.icon(info.icon, size='xs')
                ui.label(info.label).classes('text-xs')
                
                # Add a small badge for recommended mode
                if mode == ChatMode.COMBINED:
                    ui.badge('Recommended', color='green').props('floating').classes('text-xs')
        
        # Add tooltip
        with btn:
            ui.tooltip(info.tooltip)
        
        self.mode_buttons[mode] = btn
    
    def _on_mode_selected(self, mode: str) -> None:
        """Handle mode selection."""
        if mode == self.current_mode:
            return
            
        # Update current mode
        old_mode = self.current_mode
        self.current_mode = mode
        
        # Update button styles
        for m, btn in self.mode_buttons.items():
            if m == mode:
                btn.props('color=primary')
                btn.props(remove='outline')
                btn.classes(add='ring-2 ring-primary')
            else:
                btn.props('outline')
                btn.props(remove='color')
                btn.classes(remove='ring-2 ring-primary')
        
        # Update description
        if self.mode_description_label:
            self.mode_description_label.text = self.MODES[mode].description
        
        # Notify callback
        if self.on_change:
            self.on_change(mode)
        
        # Show notification
        ui.notify(
            f"Chat mode changed to: {self.MODES[mode].label}",
            type="info",
            position="top",
            timeout=2000
        )
    
    def get_mode(self) -> str:
        """Get the currently selected mode."""
        return self.current_mode
    
    def set_mode(self, mode: str) -> None:
        """Programmatically set the mode."""
        if mode in self.MODES:
            self._on_mode_selected(mode)
    
    def get_mode_info(self) -> ModeInfo:
        """Get full information about the current mode."""
        return self.MODES[self.current_mode]
    
    def disable(self) -> None:
        """Disable the mode selector."""
        for btn in self.mode_buttons.values():
            btn.props('disable')
    
    def enable(self) -> None:
        """Enable the mode selector."""
        for btn in self.mode_buttons.values():
            btn.props(remove='disable')


class ChatModeIndicator:
    """
    Small indicator showing the current chat mode in messages.
    """
    
    @staticmethod
    def create(mode: str) -> ui.element:
        """Create a mode indicator badge."""
        mode_info = ChatModeSelector.MODES.get(mode)
        if not mode_info:
            mode_info = ChatModeSelector.MODES[ChatMode.COMBINED]
        
        with ui.badge(
            color=mode_info.color
        ).props('outline').classes('text-xs') as badge:
            with ui.row().classes('items-center gap-1'):
                ui.icon(mode_info.icon, size='xs')
                ui.label(mode_info.label).classes('text-xs')
        
        return badge