"""
Memory Viewer component for displaying all memory items.

Provides a comprehensive interface to view, search, and filter
all memory items stored by the Letta agent.
"""

from nicegui import ui
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from app.logging_conf import get_logger
from .memory_editor import MemoryEditor

logger = get_logger(__name__)


class MemoryViewer:
    """Component for viewing all memory items with search and filtering."""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
        self.current_matter_id = None
        self.dialog = None
        self.container = None
        
        # State management
        self.current_page = 0
        self.items_per_page = 20
        self.total_items = 0
        self.current_filter = None
        self.current_search = ""
        self.memory_items = []
        self.edit_mode = False  # Track edit mode state
        
        # UI elements
        self.search_input = None
        self.type_tabs = None
        self.items_container = None
        self.pagination_container = None
        self.loading_indicator = None
        self.empty_state = None
        self.edit_mode_button = None
        self.add_memory_button = None
        
        # Memory editor
        self.memory_editor = None
        
        # Type configuration
        self.memory_types = {
            "All": {"icon": "dashboard", "color": "purple"},
            "Entity": {"icon": "person", "color": "blue"},
            "Event": {"icon": "event", "color": "green"},
            "Issue": {"icon": "warning", "color": "orange"},
            "Fact": {"icon": "fact_check", "color": "cyan"},
            "Interaction": {"icon": "chat", "color": "indigo"},
            "Raw": {"icon": "description", "color": "gray"}
        }
    
    async def show(self, matter_id: str):
        """Show the memory viewer dialog."""
        if not self.api_client:
            ui.notify("API client not configured", type="warning")
            return
        
        self.current_matter_id = matter_id
        self.current_page = 0
        self.current_filter = None
        self.current_search = ""
        
        # Create dialog
        with ui.dialog() as self.dialog:
            with ui.card().classes('w-full max-w-6xl h-5/6 overflow-hidden'):
                # Header
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Memory Items').classes('text-xl font-bold')
                        # Add memory button (visible in edit mode)
                        self.add_memory_button = ui.button(
                            'Add Memory',
                            icon='add',
                            on_click=self._show_create_dialog
                        ).props('color=primary').classes('hidden')
                    
                    with ui.row().classes('items-center gap-2'):
                        # Edit mode toggle
                        self.edit_mode_button = ui.button(
                            'Edit Mode',
                            icon='edit',
                            on_click=self._toggle_edit_mode
                        ).props('outline')
                        ui.button(icon='close', on_click=self.dialog.close).props('flat round')
                
                # Search bar
                with ui.row().classes('w-full mb-4'):
                    self.search_input = ui.input(
                        placeholder='Search memories...',
                        on_change=self._on_search_change
                    ).props('outlined clearable').classes('flex-grow')
                    self.search_input.on('keydown.enter', self._perform_search)
                    
                    ui.button(
                        'Search',
                        icon='search',
                        on_click=self._perform_search
                    ).props('color=primary')
                
                # Type filter tabs (just visual indicators, not panels)
                with ui.tabs().classes('w-full') as self.type_tabs:
                    for type_name, config in self.memory_types.items():
                        tab = ui.tab(type_name, icon=config["icon"])
                        # Create a closure to capture the current type_name value
                        def make_filter_handler(filter_type):
                            async def handler():
                                await self._filter_by_type(filter_type)
                            return handler
                        
                        if type_name != "All":
                            tab.on('click', make_filter_handler(type_name))
                        else:
                            tab.on('click', make_filter_handler(None))
                
                # Content area - single container for all filtered results
                with ui.scroll_area().classes('w-full h-96 flex-grow'):
                    # Main container for all items (shared across all tabs)
                    self.items_container = ui.column().classes('w-full gap-3 p-2')
                    
                    # Loading indicator
                    self.loading_indicator = ui.column().classes('w-full items-center py-8 hidden')
                    with self.loading_indicator:
                        ui.spinner(size='lg', color='purple')
                        ui.label('Loading memory items...').classes('mt-4 text-gray-600')
                    
                    # Empty state
                    self.empty_state = ui.column().classes('w-full items-center py-12 hidden')
                    with self.empty_state:
                        ui.icon('psychology', size='4rem').classes('text-gray-300')
                        ui.label('No memory items found').classes('text-lg text-gray-500 mt-4')
                        ui.label('The agent hasn\'t learned anything yet').classes('text-sm text-gray-400')
                
                # Pagination controls
                self.pagination_container = ui.row().classes('w-full justify-between items-center mt-4 pt-4 border-t')
                self._create_pagination_controls()
        
        # Open dialog and load initial data
        self.dialog.open()
        await self._load_memory_items()
    
    def _create_pagination_controls(self):
        """Create pagination controls."""
        with self.pagination_container:
            # Page info
            self.page_info = ui.label('').classes('text-sm text-gray-600')
            
            # Navigation buttons
            with ui.row().classes('gap-2'):
                self.prev_button = ui.button(
                    icon='chevron_left',
                    on_click=self._previous_page
                ).props('flat round')
                
                self.page_number = ui.label('').classes('px-3 py-1')
                
                self.next_button = ui.button(
                    icon='chevron_right',
                    on_click=self._next_page
                ).props('flat round')
    
    async def _load_memory_items(self):
        """Load memory items from the API."""
        if not self.api_client or not self.current_matter_id:
            return
        
        # Show loading state
        self._show_loading()
        
        try:
            # Calculate offset
            offset = self.current_page * self.items_per_page
            
            # Fetch memory items
            logger.info(f"Loading items - filter: {self.current_filter}, search: {self.current_search}, offset: {offset}")
            result = await self.api_client.get_memory_items(
                matter_id=self.current_matter_id,
                limit=self.items_per_page,
                offset=offset,
                type_filter=self.current_filter,
                search_query=self.current_search if self.current_search else None
            )
            
            self.memory_items = result.get('items', [])
            self.total_items = result.get('total', 0)
            
            # Debug log to see what types we're getting
            if self.memory_items:
                types_received = [item.get('type', 'Unknown') for item in self.memory_items]
                logger.info(f"Received {len(self.memory_items)} items with types: {set(types_received)}")
            
            # Update UI
            await self._render_memory_items()
            self._update_pagination()
            
        except Exception as e:
            logger.error(f"Failed to load memory items: {e}")
            # Only show UI notification if we're in a UI context
            try:
                ui.notify(f"Failed to load memory items: {str(e)}", type="negative")
            except:
                pass  # Ignore if no UI context (e.g., in tests)
            self._show_error(str(e))
    
    async def _render_memory_items(self):
        """Render the loaded memory items."""
        # Clear container
        if self.items_container:
            self.items_container.clear()
        
        # Hide loading, show appropriate state
        self._hide_loading()
        
        if not self.memory_items:
            self._show_empty_state()
            return
        
        self._hide_empty_state()
        
        # Render each memory item
        if self.items_container:
            try:
                with self.items_container:
                    for item in self.memory_items:
                        card = self._create_memory_card(item)
                        # Apply edit mode visibility if in edit mode
                        if self.edit_mode and hasattr(card, 'action_container'):
                            card.action_container.classes(remove='hidden')
            except TypeError:
                # Handle case where items_container is mocked in tests
                pass
    
    def _create_memory_card(self, item: Dict[str, Any]) -> ui.element:
        """Create a card for a memory item."""
        memory_type = item.get('type', 'Raw')
        item_id = item.get('id', 'unknown')
        
        # Handle case variations - normalize to our expected format
        if memory_type and isinstance(memory_type, str):
            # First try exact match
            if memory_type not in self.memory_types:
                # Try capitalizing first letter
                memory_type_cap = memory_type.capitalize()
                if memory_type_cap in self.memory_types:
                    memory_type = memory_type_cap
                # Special case for lowercase 'interaction' -> 'Interaction'
                elif memory_type.lower() == 'interaction':
                    memory_type = 'Interaction'
                else:
                    # Default to Raw if unknown
                    memory_type = 'Raw'
        
        type_config = self.memory_types.get(memory_type, self.memory_types['Raw'])
        
        with ui.card().classes('w-full cursor-pointer hover:shadow-lg transition-shadow') as card:
            with ui.column().classes('w-full gap-2'):
                # Header with type badge and timestamp
                with ui.row().classes('w-full justify-between items-start'):
                    # Left side: type badge
                    with ui.row().classes('items-start gap-2 flex-grow'):
                        # Type badge
                        with ui.row().classes(f'items-center gap-1 px-2 py-1 rounded-full bg-{type_config["color"]}-100 text-{type_config["color"]}-700 text-xs'):
                            ui.icon(type_config["icon"]).classes('text-sm')
                            ui.label(memory_type).classes('font-medium')
                    
                    # Right side: timestamp and action buttons
                    with ui.row().classes('items-center gap-2'):
                        # Action buttons (only visible in edit mode)
                        action_container = ui.row().classes('gap-1 hidden')
                        with action_container:
                            # Edit button
                            edit_btn = ui.button(
                                icon='edit',
                                on_click=lambda it=item: asyncio.create_task(self._show_edit_dialog(it))
                            ).props('flat round size=sm color=primary')
                            edit_btn.tooltip('Edit memory')
                            
                            # Delete button
                            delete_btn = ui.button(
                                icon='delete',
                                on_click=lambda it=item: asyncio.create_task(self._show_delete_confirmation(it))
                            ).props('flat round size=sm color=negative')
                            delete_btn.tooltip('Delete memory')
                        
                        # Store reference to action container for edit mode toggling
                        card.action_container = action_container
                        
                        # Timestamp
                        created_at = item.get('created_at')
                        if created_at:
                            timestamp = self._format_timestamp(created_at)
                            ui.label(timestamp).classes('text-xs text-gray-500')
                
                # Content
                text = item.get('text', '')
                metadata = item.get('metadata', {})
                
                # If it's a structured item with metadata, show formatted
                if metadata and isinstance(metadata, dict):
                    # Show label if available
                    label = metadata.get('label', '')
                    if label:
                        ui.label(label).classes('font-medium text-sm')
                    
                    # Show other metadata fields
                    if metadata.get('date'):
                        with ui.row().classes('items-center gap-1 text-xs text-gray-600'):
                            ui.icon('calendar_today').classes('text-sm')
                            ui.label(metadata['date'])
                    
                    if metadata.get('actors'):
                        actors = metadata['actors']
                        if actors and isinstance(actors, list):
                            with ui.row().classes('items-center gap-1 text-xs text-gray-600'):
                                ui.icon('group').classes('text-sm')
                                ui.label(', '.join(actors[:3]))
                                if len(actors) > 3:
                                    ui.label(f'and {len(actors) - 3} more').classes('text-gray-400')
                    
                    # Show snippet if available
                    snippet = metadata.get('support_snippet', '')
                    if snippet:
                        self._create_expandable_text(snippet, max_lines=3)
                else:
                    # Show raw text
                    self._create_expandable_text(text, max_lines=4)
                
                # Source reference if available
                source = item.get('source')
                if source:
                    with ui.row().classes('items-center gap-1 text-xs text-gray-500 mt-2'):
                        ui.icon('source').classes('text-sm')
                        ui.label(f'Source: {source}')
                
                # Item ID (hidden by default, shown on hover)
                with ui.row().classes('items-center gap-1 text-xs text-gray-400 opacity-0 hover:opacity-100 transition-opacity'):
                    ui.icon('fingerprint').classes('text-sm')
                    ui.label(f'ID: {item.get("id", "unknown")[:8]}...')
        
        return card
    
    def _create_expandable_text(self, text: str, max_lines: int = 3):
        """Create expandable text component."""
        if not text:
            return
        
        # Estimate if text needs expansion (rough calculation)
        estimated_lines = len(text) // 100  # Rough estimate
        
        if estimated_lines <= max_lines:
            # Short text, no expansion needed
            ui.label(text).classes('text-sm text-gray-700 whitespace-pre-wrap')
        else:
            # Long text, make expandable
            container = ui.column().classes('w-full')
            with container:
                # Initially show truncated text
                truncated = text[:300] + '...' if len(text) > 300 else text
                text_label = ui.label(truncated).classes('text-sm text-gray-700 whitespace-pre-wrap')
                
                # Toggle button
                expanded = {'value': False}
                
                def toggle_expand():
                    expanded['value'] = not expanded['value']
                    if expanded['value']:
                        text_label.text = text
                        toggle_btn.text = 'Show less'
                        toggle_btn.props('icon=expand_less')
                    else:
                        text_label.text = truncated
                        toggle_btn.text = 'Show more'
                        toggle_btn.props('icon=expand_more')
                
                toggle_btn = ui.button(
                    'Show more',
                    icon='expand_more',
                    on_click=toggle_expand
                ).props('flat size=sm color=primary').classes('mt-1')
    
    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.now(dt.tzinfo)
            diff = now - dt
            
            if diff.days > 7:
                return dt.strftime('%b %d, %Y')
            elif diff.days > 0:
                return f'{diff.days}d ago'
            elif diff.seconds > 3600:
                return f'{diff.seconds // 3600}h ago'
            elif diff.seconds > 60:
                return f'{diff.seconds // 60}m ago'
            else:
                return 'Just now'
        except:
            return timestamp
    
    def _show_loading(self):
        """Show loading indicator."""
        if self.loading_indicator:
            self.loading_indicator.classes(remove='hidden')
        if self.items_container:
            self.items_container.classes(add='hidden')
        if self.empty_state:
            self.empty_state.classes(add='hidden')
    
    def _hide_loading(self):
        """Hide loading indicator."""
        if self.loading_indicator:
            self.loading_indicator.classes(add='hidden')
        if self.items_container:
            self.items_container.classes(remove='hidden')
    
    def _show_empty_state(self):
        """Show empty state."""
        if self.empty_state:
            self.empty_state.classes(remove='hidden')
        if self.items_container:
            self.items_container.classes(add='hidden')
    
    def _hide_empty_state(self):
        """Hide empty state."""
        if self.empty_state:
            self.empty_state.classes(add='hidden')
        if self.items_container:
            self.items_container.classes(remove='hidden')
    
    def _show_error(self, error: str):
        """Show error state."""
        if self.items_container:
            self.items_container.clear()
            with self.items_container:
                with ui.column().classes('w-full items-center py-12'):
                    ui.icon('error', size='4rem').classes('text-red-400')
                    ui.label('Failed to load memory items').classes('text-lg text-red-600 mt-4')
                    ui.label(error).classes('text-sm text-gray-500 mt-2')
    
    def _update_pagination(self):
        """Update pagination controls."""
        if not self.pagination_container:
            return
        
        # Calculate total pages
        total_pages = max(1, (self.total_items + self.items_per_page - 1) // self.items_per_page)
        
        # Update page info
        start = self.current_page * self.items_per_page + 1
        end = min(start + self.items_per_page - 1, self.total_items)
        
        if self.total_items > 0:
            self.page_info.text = f'Showing {start}-{end} of {self.total_items} items'
        else:
            self.page_info.text = 'No items'
        
        # Update page number
        self.page_number.text = f'{self.current_page + 1} / {total_pages}'
        
        # Update button states
        self.prev_button.set_enabled(self.current_page > 0)
        self.next_button.set_enabled(self.current_page < total_pages - 1)
    
    async def _previous_page(self):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            await self._load_memory_items()
    
    async def _next_page(self):
        """Go to next page."""
        total_pages = max(1, (self.total_items + self.items_per_page - 1) // self.items_per_page)
        if self.current_page < total_pages - 1:
            self.current_page += 1
            await self._load_memory_items()
    
    def _on_search_change(self, e):
        """Handle search input change."""
        self.current_search = e.value
    
    async def _perform_search(self):
        """Perform search with current query."""
        self.current_page = 0  # Reset to first page
        await self._load_memory_items()
    
    async def _filter_by_type(self, type_filter: Optional[str]):
        """Filter memories by type."""
        logger.info(f"Filtering by type: {type_filter}")
        self.current_filter = type_filter
        self.current_page = 0  # Reset to first page
        await self._load_memory_items()
    
    def _toggle_edit_mode(self):
        """Toggle edit mode on/off."""
        self.edit_mode = not self.edit_mode
        
        # Update button appearance
        if self.edit_mode_button:
            if self.edit_mode:
                self.edit_mode_button.props('color=primary')
                self.edit_mode_button.text = 'Exit Edit'
            else:
                self.edit_mode_button.props(remove='color=primary')
                self.edit_mode_button.props('outline')
                self.edit_mode_button.text = 'Edit Mode'
        
        # Show/hide add memory button
        if self.add_memory_button:
            if self.edit_mode:
                self.add_memory_button.classes(remove='hidden')
            else:
                self.add_memory_button.classes(add='hidden')
        
        # Show/hide action buttons on all cards
        if self.items_container:
            # Find all cards and toggle their action containers
            for element in self.items_container.children:
                if hasattr(element, 'action_container'):
                    if self.edit_mode:
                        element.action_container.classes(remove='hidden')
                    else:
                        element.action_container.classes(add='hidden')
    
    async def _show_create_dialog(self):
        """Show the memory editor dialog in create mode."""
        if not self.memory_editor:
            self.memory_editor = MemoryEditor(api_client=self.api_client)
        
        await self.memory_editor.show_create(
            matter_id=self.current_matter_id,
            on_save=self._on_memory_saved
        )
    
    async def _show_edit_dialog(self, item: Dict[str, Any]):
        """Show the memory editor dialog in edit mode."""
        if not self.memory_editor:
            self.memory_editor = MemoryEditor(api_client=self.api_client)
        
        await self.memory_editor.show_edit(
            matter_id=self.current_matter_id,
            item=item,
            on_save=self._on_memory_saved
        )
    
    async def _show_delete_confirmation(self, item: Dict[str, Any]):
        """Show confirmation dialog before deleting a memory item."""
        item_id = item.get('id', 'unknown')
        item_type = item.get('type', 'Unknown')
        
        # Get item preview text
        preview_text = ""
        metadata = item.get('metadata', {})
        if metadata and isinstance(metadata, dict):
            preview_text = metadata.get('label', '')
        if not preview_text:
            preview_text = item.get('text', '')[:100]
            if len(item.get('text', '')) > 100:
                preview_text += '...'
        
        # Create confirmation dialog
        confirm_dialog = None
        with ui.dialog() as confirm_dialog:
            with ui.card().classes('w-96'):
                # Header
                with ui.row().classes('w-full items-center gap-2 mb-4'):
                    ui.icon('warning').classes('text-2xl text-orange-500')
                    ui.label('Delete Memory Item?').classes('text-lg font-bold')
                
                # Content
                with ui.column().classes('w-full gap-2 mb-4'):
                    ui.label('This action cannot be undone.').classes('text-sm text-gray-600')
                    
                    # Item preview
                    with ui.card().classes('w-full bg-gray-50 p-3'):
                        with ui.row().classes('items-center gap-2 mb-2'):
                            ui.label(f'Type: {item_type}').classes('text-xs font-medium')
                            ui.label(f'ID: {item_id[:8]}...').classes('text-xs text-gray-500')
                        ui.label(preview_text).classes('text-sm text-gray-700')
                
                # Action buttons
                with ui.row().classes('w-full justify-end gap-2'):
                    ui.button(
                        'Cancel',
                        on_click=confirm_dialog.close
                    ).props('flat')
                    
                    async def confirm_delete():
                        await self._delete_memory_item(item_id)
                        confirm_dialog.close()
                    
                    ui.button(
                        'Delete',
                        icon='delete',
                        on_click=confirm_delete
                    ).props('color=negative')
        
        confirm_dialog.open()
    
    async def _delete_memory_item(self, item_id: str):
        """Delete a memory item."""
        if not self.api_client or not self.current_matter_id:
            ui.notify("Cannot delete: API client or matter not configured", type="negative")
            return
        
        try:
            # Show loading notification
            ui.notify('Deleting memory item...', type='ongoing', position='bottom-right')
            
            # Call API to delete
            result = await self.api_client.delete_memory_item(
                matter_id=self.current_matter_id,
                item_id=item_id
            )
            
            if result.get('success'):
                ui.notify('Memory item deleted successfully', type='positive')
                # Reload the list
                await self._load_memory_items()
            else:
                ui.notify(f"Failed to delete: {result.get('message', 'Unknown error')}", type='negative')
        
        except Exception as e:
            logger.error(f"Failed to delete memory item: {e}")
            ui.notify(f"Error deleting memory: {str(e)}", type='negative')
    
    async def _on_memory_saved(self):
        """Callback when a memory item is saved (created or updated)."""
        # Reload the memory items to show the changes
        await self._load_memory_items()