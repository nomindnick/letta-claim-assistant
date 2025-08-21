"""
Agent tool usage indicators for the Letta Construction Claim Assistant.

Provides visual components to show when the agent uses tools like document search.
"""

from nicegui import ui
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ToolType(str, Enum):
    """Types of tools the agent can use."""
    SEARCH_DOCUMENTS = "search_documents"
    RECALL_MEMORY = "recall_memory"
    STORE_MEMORY = "store_memory"


@dataclass
class ToolUsage:
    """Information about a tool usage."""
    tool_name: str
    tool_type: ToolType
    description: str
    icon: str
    color: str


class AgentToolIndicator:
    """
    Component to show when the agent uses tools.
    
    Displays inline indicators in chat messages showing which tools
    were used to generate the response.
    """
    
    # Tool definitions with display information
    TOOL_INFO = {
        "search_documents": ToolUsage(
            tool_name="search_documents",
            tool_type=ToolType.SEARCH_DOCUMENTS,
            description="Searched case documents",
            icon="search",
            color="blue"
        ),
        "recall_memory": ToolUsage(
            tool_name="recall_memory",
            tool_type=ToolType.RECALL_MEMORY,
            description="Recalled from memory",
            icon="psychology",
            color="purple"
        ),
        "store_memory": ToolUsage(
            tool_name="store_memory",
            tool_type=ToolType.STORE_MEMORY,
            description="Stored to memory",
            icon="save",
            color="green"
        )
    }
    
    @classmethod
    def create_tool_badge(cls, tool_name: str) -> ui.element:
        """Create a badge showing a tool was used."""
        tool_info = cls.TOOL_INFO.get(tool_name)
        if not tool_info:
            # Unknown tool - create generic badge
            return ui.badge(
                tool_name,
                color='gray'
            ).props('outline').classes('text-xs')
        
        with ui.badge(
            color=tool_info.color
        ).props('outline').classes('text-xs') as badge:
            with ui.row().classes('items-center gap-1'):
                ui.icon(tool_info.icon, size='xs')
                ui.label(tool_info.description).classes('text-xs')
        
        return badge
    
    @classmethod
    def create_tools_row(cls, tools_used: List[str]) -> Optional[ui.element]:
        """Create a row showing all tools used."""
        if not tools_used:
            return None
        
        with ui.row().classes('gap-2 items-center mb-2') as row:
            ui.label('Tools used:').classes('text-xs text-gray-600')
            for tool in tools_used:
                cls.create_tool_badge(tool)
        
        return row


class SearchInProgressIndicator:
    """
    Animated indicator showing document search in progress.
    """
    
    @staticmethod
    def create() -> ui.element:
        """Create search in progress indicator."""
        with ui.card().classes('w-full mb-2 bg-blue-50 border-l-4 border-blue-500') as card:
            with ui.row().classes('items-center gap-2 p-2'):
                ui.spinner(type='dots', size='sm').classes('text-blue-500')
                ui.icon('search', size='sm').classes('text-blue-600')
                ui.label('Searching documents...').classes('text-sm text-blue-700')
                
                # Animated pulse effect
                with ui.html().classes('ml-auto'):
                    ui.add_head_html('''
                    <style>
                        @keyframes pulse-blue {
                            0%, 100% { opacity: 1; }
                            50% { opacity: 0.5; }
                        }
                        .pulse-blue {
                            animation: pulse-blue 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
                        }
                    </style>
                    ''')
                    ui.icon('auto_awesome', size='sm').classes('text-blue-500 pulse-blue')
        
        return card


class ToolUsageCard:
    """
    Card showing details of tool usage in a response.
    """
    
    @staticmethod
    def create(tools_used: List[str], search_performed: bool, result_count: int = 0) -> Optional[ui.element]:
        """Create a card showing tool usage details."""
        if not tools_used and not search_performed:
            return None
        
        with ui.card().classes('w-full mb-2 bg-gray-50 border-l-4 border-gray-400') as card:
            with ui.column().classes('p-2 gap-1'):
                # Header
                with ui.row().classes('items-center gap-2'):
                    ui.icon('build', size='sm').classes('text-gray-600')
                    ui.label('Agent Actions').classes('text-sm font-semibold text-gray-700')
                
                # Tool details
                if search_performed:
                    with ui.row().classes('items-center gap-2 ml-6'):
                        ui.icon('search', size='xs').classes('text-blue-600')
                        ui.label(f'Searched documents').classes('text-xs text-gray-600')
                        if result_count > 0:
                            ui.badge(f'{result_count} results', color='blue').props('outline').classes('text-xs')
                
                # Other tools
                for tool in tools_used:
                    if tool != 'search_documents':  # Already handled above
                        tool_info = AgentToolIndicator.TOOL_INFO.get(tool)
                        if tool_info:
                            with ui.row().classes('items-center gap-2 ml-6'):
                                ui.icon(tool_info.icon, size='xs').classes(f'text-{tool_info.color}-600')
                                ui.label(tool_info.description).classes('text-xs text-gray-600')
        
        return card


class MemoryOperationIndicator:
    """
    Indicator for memory operations (recall, store, update).
    """
    
    @staticmethod
    def create_recalling() -> ui.element:
        """Create memory recall indicator."""
        with ui.row().classes('items-center gap-2 p-1 bg-purple-50 rounded') as row:
            ui.spinner(type='dots', size='xs').classes('text-purple-500')
            ui.icon('psychology', size='xs').classes('text-purple-600')
            ui.label('Recalling from memory...').classes('text-xs text-purple-700')
        return row
    
    @staticmethod
    def create_storing() -> ui.element:
        """Create memory storing indicator."""
        with ui.row().classes('items-center gap-2 p-1 bg-green-50 rounded') as row:
            ui.spinner(type='dots', size='xs').classes('text-green-500')
            ui.icon('save', size='xs').classes('text-green-600')
            ui.label('Storing to memory...').classes('text-xs text-green-700')
        return row
    
    @staticmethod
    def create_memory_used_badge() -> ui.element:
        """Create badge showing memory was used."""
        return ui.badge(
            'Memory Enhanced',
            color='purple'
        ).props('outline').classes('text-xs')


class AgentThinkingIndicator:
    """
    Enhanced thinking indicator that shows what the agent is doing.
    """
    
    @staticmethod
    def create(initial_text: str = "Agent analyzing...") -> ui.element:
        """Create an animated thinking indicator."""
        with ui.card().classes('w-full mb-2 bg-gradient-to-r from-blue-50 to-purple-50') as card:
            with ui.row().classes('items-center gap-2 p-3'):
                # Animated spinner
                ui.spinner(type='dots', size='sm').classes('text-blue-600')
                
                # Dynamic text that can be updated
                thinking_label = ui.label(initial_text).classes('text-sm text-gray-700')
                thinking_label.classes('thinking-text')
                
                # Animated brain icon
                with ui.html().classes('ml-auto'):
                    ui.add_head_html('''
                    <style>
                        @keyframes thinking-pulse {
                            0%, 100% { 
                                transform: scale(1);
                                opacity: 0.8;
                            }
                            50% { 
                                transform: scale(1.1);
                                opacity: 1;
                            }
                        }
                        .thinking-icon {
                            animation: thinking-pulse 1.5s ease-in-out infinite;
                        }
                    </style>
                    ''')
                    ui.icon('psychology', size='sm').classes('text-purple-600 thinking-icon')
        
        # Store reference to text element for updates
        card.thinking_label = thinking_label
        return card
    
    @staticmethod
    def update_text(indicator: ui.element, new_text: str):
        """Update the thinking indicator text."""
        if hasattr(indicator, 'thinking_label'):
            indicator.thinking_label.text = new_text