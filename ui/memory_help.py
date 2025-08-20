"""
Memory Help System with tooltips and tutorials.

Provides contextual help, keyboard shortcuts reference,
and interactive tutorials for memory management.
"""

from nicegui import ui
from typing import Optional, Dict, Any
import asyncio

class MemoryHelp:
    """Help system for memory management features."""
    
    def __init__(self):
        self.help_dialog = None
        self.tooltips = self._get_tooltips()
        self.shortcuts = self._get_shortcuts()
    
    def _get_tooltips(self) -> Dict[str, str]:
        """Get tooltip definitions for UI elements."""
        return {
            'search': 'Search through all memories using semantic search, keywords, exact phrases, or regex patterns',
            'search_type': 'Choose how to search: Semantic (AI-powered), Keyword (all words), Exact (phrase), or Regex (patterns)',
            'edit_mode': 'Toggle edit mode to modify or delete memory items',
            'add_memory': 'Add a new memory item manually',
            'type_filter': 'Filter memories by type: Entity, Event, Issue, Fact, Interaction, or Raw',
            'memory_card': 'Click to expand details. In edit mode, use action buttons to modify or delete',
            'pagination': 'Navigate through pages of memory items',
            'analytics': 'View memory patterns and insights',
            'export': 'Export memories to JSON or CSV format',
            'import': 'Import memories from a file'
        }
    
    def _get_shortcuts(self) -> Dict[str, str]:
        """Get keyboard shortcut definitions."""
        return {
            'Ctrl+F / Cmd+F': 'Focus search field',
            'Ctrl+E': 'Toggle edit mode',
            'Ctrl+N': 'Add new memory',
            'Escape': 'Close dialog',
            'Arrow Left/Right': 'Navigate pages',
            'Enter': 'Perform search (when in search field)',
            'Tab': 'Navigate between elements'
        }
    
    def add_tooltip(self, element: ui.element, key: str) -> ui.element:
        """Add a tooltip to an element."""
        if key in self.tooltips:
            element.tooltip(self.tooltips[key])
        return element
    
    async def show_help_dialog(self):
        """Show the help dialog with shortcuts and tips."""
        if self.help_dialog:
            self.help_dialog.open()
            return
        
        with ui.dialog() as self.help_dialog:
            with ui.card().classes('w-full max-w-2xl'):
                # Header
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    ui.label('Memory Management Help').classes('text-xl font-bold')
                    ui.button(icon='close', on_click=self.help_dialog.close).props('flat round')
                
                # Content tabs
                with ui.tabs().classes('w-full') as tabs:
                    shortcuts_tab = ui.tab('Keyboard Shortcuts', icon='keyboard')
                    tips_tab = ui.tab('Tips & Tricks', icon='lightbulb')
                    memory_types_tab = ui.tab('Memory Types', icon='category')
                
                with ui.tab_panels(tabs, value=shortcuts_tab).classes('w-full'):
                    # Keyboard shortcuts panel
                    with ui.tab_panel(shortcuts_tab):
                        with ui.column().classes('w-full gap-2'):
                            ui.label('Keyboard Shortcuts').classes('text-lg font-semibold mb-2')
                            for shortcut, description in self.shortcuts.items():
                                with ui.row().classes('w-full justify-between py-2 border-b'):
                                    ui.label(shortcut).classes('font-mono bg-gray-100 px-2 py-1 rounded')
                                    ui.label(description).classes('text-gray-600')
                    
                    # Tips panel
                    with ui.tab_panel(tips_tab):
                        with ui.column().classes('w-full gap-3'):
                            ui.label('Tips & Tricks').classes('text-lg font-semibold mb-2')
                            
                            tips = [
                                ('Search Effectively', 'Use semantic search for concepts, keyword for specific terms, exact for phrases, and regex for patterns'),
                                ('Manage Memory', 'Toggle edit mode to modify or delete incorrect memories'),
                                ('Natural Language', 'Use commands like "Remember that..." or "Forget about..." in chat'),
                                ('Performance', 'Memory-only mode is faster for questions about established facts'),
                                ('Organization', 'Filter by type to focus on specific kinds of information'),
                                ('Bulk Operations', 'Export memories for backup, then import to another matter')
                            ]
                            
                            for title, description in tips:
                                with ui.card().classes('w-full'):
                                    ui.label(title).classes('font-semibold')
                                    ui.label(description).classes('text-sm text-gray-600')
                    
                    # Memory types panel
                    with ui.tab_panel(memory_types_tab):
                        with ui.column().classes('w-full gap-3'):
                            ui.label('Memory Types').classes('text-lg font-semibold mb-2')
                            
                            types = {
                                'Entity': ('People, companies, or organizations mentioned in documents', 'purple'),
                                'Event': ('Significant occurrences with dates and participants', 'blue'),
                                'Issue': ('Problems, disputes, or concerns that need resolution', 'red'),
                                'Fact': ('Verified information and established truths', 'green'),
                                'Interaction': ('User conversations and agent responses', 'orange'),
                                'Raw': ('Unstructured memories and notes', 'gray')
                            }
                            
                            for type_name, (description, color) in types.items():
                                with ui.row().classes('w-full items-center gap-3 py-2'):
                                    ui.badge(type_name, color=color).classes('px-3 py-1')
                                    ui.label(description).classes('text-sm text-gray-600')
        
        self.help_dialog.open()
    
    def create_help_button(self) -> ui.element:
        """Create a help button that shows the help dialog."""
        return ui.button(
            icon='help_outline',
            on_click=lambda: asyncio.create_task(self.show_help_dialog())
        ).props('flat round').tooltip('Show help and keyboard shortcuts (F1)')
    
    def show_interactive_tutorial(self):
        """Show an interactive tutorial for first-time users."""
        # This could be expanded with a step-by-step guide
        pass


# Global help instance
memory_help = MemoryHelp()