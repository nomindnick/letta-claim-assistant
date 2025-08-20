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
from .memory_analytics import MemoryAnalyticsDashboard
from .memory_help import memory_help

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
        
        # Lazy loading state
        self.is_loading_more = False
        self.has_more_items = True
        self.all_loaded_items = []  # Store all loaded items for virtual scrolling
        
        # Debounce timer for search
        self.search_timer = None
        self.search_debounce_ms = 300
        
        # UI elements
        self.search_input = None
        self.type_tabs = None
        self.items_container = None
        self.pagination_container = None
        self.loading_indicator = None
        self.empty_state = None
        self.edit_mode_button = None
        self.load_more_indicator = None
        self.scroll_area = None
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
        
        # Create responsive dialog
        with ui.dialog() as self.dialog:
            with ui.card().classes('w-full max-w-6xl h-5/6 overflow-hidden lg:max-w-5xl md:max-w-3xl sm:max-w-full sm:h-full'):
                # Header with responsive layout
                with ui.row().classes('w-full justify-between items-center mb-4 flex-wrap sm:flex-col sm:items-start sm:gap-2'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Memory Items').classes('text-xl font-bold')
                        # Add memory button (visible in edit mode)
                        self.add_memory_button = ui.button(
                            'Add Memory',
                            icon='add',
                            on_click=self._show_create_dialog
                        ).props('color=primary').classes('hidden')
                    
                    with ui.row().classes('items-center gap-2'):
                        # Export button
                        ui.button(
                            'Export',
                            icon='download',
                            on_click=self._show_export_dialog
                        ).props('outline').tooltip('Export memories to file')
                        # Import button
                        ui.button(
                            'Import',
                            icon='upload',
                            on_click=self._show_import_dialog
                        ).props('outline').tooltip('Import memories from file')
                        # Analytics button
                        ui.button(
                            'Analytics',
                            icon='analytics',
                            on_click=self._show_analytics
                        ).props('outline')
                        # Edit mode toggle
                        self.edit_mode_button = ui.button(
                            'Edit Mode',
                            icon='edit',
                            on_click=self._toggle_edit_mode
                        ).props('outline')
                        ui.button(icon='close', on_click=self.dialog.close).props('flat round')
                
                # Search bar with advanced options
                with ui.row().classes('w-full mb-4'):
                    # Search type selector
                    self.search_type = ui.select(
                        ['Semantic', 'Keyword', 'Exact', 'Regex'],
                        value='Semantic',
                        label='Search Type'
                    ).props('outlined dense').classes('w-32')
                    
                    self.search_input = ui.input(
                        placeholder='Search memories...',
                        on_change=self._on_search_change
                    ).props('outlined clearable').classes('flex-grow min-w-0 sm:w-full')
                    self.search_input.on('keydown.enter', self._perform_search)
                    
                    ui.button(
                        'Search',
                        icon='search',
                        on_click=self._perform_search
                    ).props('color=primary')
                    
                    # Advanced search help
                    with ui.button(icon='help_outline').props('flat round size=sm'):
                        ui.tooltip('''Search Types:
• Semantic: AI-powered meaning-based search
• Keyword: All words must be present
• Exact: Exact phrase match
• Regex: Regular expression pattern''')
                
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
                
                # Content area - single container for all filtered results with infinite scroll
                self.scroll_area = ui.scroll_area().classes('w-full h-96 flex-grow')
                with self.scroll_area:
                    # Main container for all items (shared across all tabs)
                    self.items_container = ui.column().classes('w-full gap-3 p-2')
                    
                    # Set up infinite scroll detection
                    self.scroll_area.on('scroll', self._on_scroll)
                    
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
                    
                    # Load more indicator for infinite scroll
                    self.load_more_indicator = ui.column().classes('w-full items-center py-4 hidden')
                    with self.load_more_indicator:
                        ui.spinner(size='md', color='purple')
                        ui.label('Loading more...').classes('text-sm text-gray-600')
                
                # Pagination controls
                self.pagination_container = ui.row().classes('w-full justify-between items-center mt-4 pt-4 border-t')
                self._create_pagination_controls()
        
        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()
        
        # Open dialog and load initial data
        self.dialog.open()
        await self._load_memory_items()
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for the memory viewer."""
        if not self.dialog:
            return
        
        # Ctrl+F / Cmd+F: Focus search
        self.dialog.on('keydown.ctrl.f', lambda: self._focus_search())
        self.dialog.on('keydown.meta.f', lambda: self._focus_search())  # Mac
        
        # Ctrl+E: Toggle edit mode
        self.dialog.on('keydown.ctrl.e', lambda: self._toggle_edit_mode() if self.edit_mode_button else None)
        
        # Ctrl+N: Add new memory
        self.dialog.on('keydown.ctrl.n', lambda: asyncio.create_task(self._add_memory()) if self.add_memory_button else None)
        
        # Escape: Close dialog
        self.dialog.on('keydown.escape', lambda: self.dialog.close())
        
        # Arrow keys for navigation (when not in input)
        self.dialog.on('keydown.arrowleft', lambda: asyncio.create_task(self._previous_page()))
        self.dialog.on('keydown.arrowright', lambda: asyncio.create_task(self._next_page()))
    
    def _focus_search(self):
        """Focus the search input field."""
        if self.search_input:
            self.search_input.focus()
            # Prevent default browser search
            return False
    
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
            
            # Get search type from UI selector
            search_type = "semantic"  # Default
            if hasattr(self, 'search_type') and self.search_type:
                search_type = self.search_type.value.lower()
            
            # Fetch memory items
            logger.info(f"Loading items - filter: {self.current_filter}, search: {self.current_search}, search_type: {search_type}, offset: {offset}")
            result = await self.api_client.get_memory_items(
                matter_id=self.current_matter_id,
                limit=self.items_per_page,
                offset=offset,
                type_filter=self.current_filter,
                search_query=self.current_search if self.current_search else None,
                search_type=search_type
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
        """Create a responsive card for a memory item."""
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
        
        with ui.card().classes('w-full cursor-pointer hover:shadow-lg transition-shadow touch-manipulation') as card:
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
        """Show loading indicator with skeleton loaders."""
        if self.loading_indicator:
            self.loading_indicator.classes(remove='hidden')
        if self.items_container:
            # Show skeleton loaders instead of hiding
            self.items_container.clear()
            with self.items_container:
                from .components import SkeletonLoader
                for _ in range(3):  # Show 3 skeleton cards
                    SkeletonLoader.memory_card_skeleton()
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
        # Trigger debounced search
        asyncio.create_task(self._on_search_change_debounced(e.value))
    
    async def _on_search_change_debounced(self, value: str):
        """Debounced search handler."""
        # Cancel previous timer
        if self.search_timer:
            self.search_timer.cancel()
        
        # Start new timer
        async def perform_search():
            await asyncio.sleep(self.search_debounce_ms / 1000)
            self.current_search = value
            self.current_page = 0  # Reset to first page
            self.all_loaded_items = []  # Clear cache for new search
            await self._load_memory_items()
        
        self.search_timer = asyncio.create_task(perform_search())
    
    async def _perform_search(self):
        """Perform search with current query."""
        self.current_page = 0  # Reset to first page
        await self._load_memory_items()
    
    async def _filter_by_type(self, type_filter: Optional[str]):
        """Filter memories by type."""
        logger.info(f"Filtering by type: {type_filter}")
        self.current_filter = type_filter
        self.current_page = 0  # Reset to first page
        self.all_loaded_items = []  # Clear cache for new filter
        self.has_more_items = True  # Reset infinite scroll state
        await self._load_memory_items()
    
    async def _on_scroll(self, e):
        """Handle scroll event for infinite scrolling."""
        if not self.scroll_area or self.is_loading_more or not self.has_more_items:
            return
        
        # Check if scrolled near bottom (within 100px)
        try:
            scroll_height = e.args.get('scrollHeight', 0)
            scroll_top = e.args.get('scrollTop', 0)
            client_height = e.args.get('clientHeight', 0)
            
            if scroll_height - scroll_top - client_height < 100:
                await self._load_more_items()
        except Exception as e:
            logger.debug(f"Scroll detection error: {e}")
    
    async def _load_more_items(self):
        """Load more items for infinite scroll."""
        if self.is_loading_more or not self.has_more_items:
            return
        
        self.is_loading_more = True
        
        # Show load more indicator
        if self.load_more_indicator:
            self.load_more_indicator.classes(remove='hidden')
        
        try:
            # Increment page for next batch
            self.current_page += 1
            offset = self.current_page * self.items_per_page
            
            # Get search type from UI selector
            search_type = "semantic"
            if hasattr(self, 'search_type') and self.search_type:
                search_type = self.search_type.value.lower()
            
            # Fetch more items
            result = await self.api_client.get_memory_items(
                matter_id=self.current_matter_id,
                limit=self.items_per_page,
                offset=offset,
                type_filter=self.current_filter,
                search_query=self.current_search if self.current_search else None,
                search_type=search_type
            )
            
            new_items = result.get('items', [])
            
            if new_items:
                # Add to all loaded items
                self.all_loaded_items.extend(new_items)
                
                # Render new items
                if self.items_container:
                    with self.items_container:
                        for item in new_items:
                            card = self._create_memory_card(item)
                            if self.edit_mode and hasattr(card, 'action_container'):
                                card.action_container.classes(remove='hidden')
            else:
                # No more items to load
                self.has_more_items = False
            
        except Exception as e:
            logger.error(f"Failed to load more items: {e}")
            self.has_more_items = False
        finally:
            self.is_loading_more = False
            if self.load_more_indicator:
                self.load_more_indicator.classes(add='hidden')
    
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
    
    async def _show_analytics(self):
        """Show the memory analytics dashboard."""
        if not self.api_client or not self.current_matter_id:
            ui.notify("Cannot show analytics: No matter selected", type="warning")
            return
        
        # Create analytics dialog
        with ui.dialog() as analytics_dialog:
            # Create analytics dashboard
            analytics_dashboard = MemoryAnalyticsDashboard(
                api_client=self.api_client,
                matter_id=self.current_matter_id
            )
            
            with ui.card().classes('w-full max-w-7xl h-5/6 overflow-auto'):
                # Add close button in top right
                with ui.row().classes('w-full justify-end mb-2'):
                    ui.button(icon='close', on_click=analytics_dialog.close).props('flat round')
                
                # Create the dashboard
                analytics_dashboard.create()
        
        # Show the dialog
        analytics_dialog.open()
    
    async def _show_export_dialog(self):
        """Show export dialog for memory export options."""
        if not self.api_client or not self.current_matter_id:
            ui.notify("Cannot export: No matter selected", type="warning")
            return
        
        # Create export dialog
        export_dialog = ui.dialog()
        
        with export_dialog, ui.card().classes('w-96'):
            ui.label('Export Memory').classes('text-lg font-bold mb-4')
            
            # Format selector
            format_select = ui.select(
                label='Export Format',
                options={'json': 'JSON', 'csv': 'CSV'},
                value='json'
            ).classes('w-full mb-4')
            
            # Include metadata checkbox
            include_metadata = ui.checkbox(
                'Include metadata (IDs, timestamps)',
                value=True
            ).classes('mb-4')
            
            # Buttons
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=export_dialog.close).props('outline')
                
                async def export():
                    try:
                        export_dialog.close()
                        
                        # Show loading notification
                        loading_notif = ui.notify('Exporting memories...', type='ongoing')
                        
                        # Export memory
                        content = await self.api_client.export_memory(
                            matter_id=self.current_matter_id,
                            format=format_select.value,
                            include_metadata=include_metadata.value
                        )
                        
                        # Generate filename
                        filename = f"memory_export_{self.current_matter_id}.{format_select.value}"
                        
                        # Trigger download using JavaScript
                        from nicegui import app
                        ui.run_javascript(f'''
                            const blob = new Blob([{repr(content.decode('utf-8'))}], 
                                {{type: '{("application/json" if format_select.value == "json" else "text/csv")}'}});
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = '{filename}';
                            document.body.appendChild(a);
                            a.click();
                            window.URL.revokeObjectURL(url);
                            document.body.removeChild(a);
                        ''')
                        
                        loading_notif.dismiss()
                        ui.notify(f'Memory exported to {filename}', type='positive')
                        
                    except Exception as e:
                        logger.error(f"Export failed: {str(e)}")
                        ui.notify(f'Export failed: {str(e)}', type='negative')
                
                ui.button('Export', on_click=export).props('color=primary')
        
        export_dialog.open()
    
    async def _show_import_dialog(self):
        """Show import dialog for memory import."""
        if not self.api_client or not self.current_matter_id:
            ui.notify("Cannot import: No matter selected", type="warning")
            return
        
        # Create import dialog
        import_dialog = ui.dialog()
        uploaded_file = {'content': None, 'name': None}
        
        with import_dialog, ui.card().classes('w-96'):
            ui.label('Import Memory').classes('text-lg font-bold mb-4')
            
            # File upload
            async def handle_upload(e):
                uploaded_file['content'] = e.content.read()
                uploaded_file['name'] = e.name
                file_label.set_text(f'Selected: {e.name}')
                import_button.enable()
            
            ui.upload(
                label='Select memory file',
                on_upload=handle_upload,
                auto_upload=True
            ).classes('w-full mb-4').props('accept=".json,.csv"')
            
            file_label = ui.label('No file selected').classes('text-sm text-gray-600 mb-4')
            
            # Format selector
            format_select = ui.select(
                label='Import Format',
                options={'json': 'JSON', 'csv': 'CSV'},
                value='json'
            ).classes('w-full mb-4')
            
            # Deduplicate checkbox
            deduplicate = ui.checkbox(
                'Skip duplicate memories',
                value=True
            ).classes('mb-4')
            
            # Buttons
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=import_dialog.close).props('outline')
                
                async def import_memory():
                    try:
                        if not uploaded_file['content']:
                            ui.notify('Please select a file', type='warning')
                            return
                        
                        import_dialog.close()
                        
                        # Show loading notification
                        loading_notif = ui.notify('Importing memories...', type='ongoing')
                        
                        # Import memory
                        result = await self.api_client.import_memory(
                            matter_id=self.current_matter_id,
                            file_content=uploaded_file['content'],
                            filename=uploaded_file['name'],
                            format=format_select.value,
                            deduplicate=deduplicate.value
                        )
                        
                        loading_notif.dismiss()
                        
                        # Show results
                        imported = result.get('imported', 0)
                        skipped = result.get('skipped', 0)
                        total = result.get('total', 0)
                        
                        if 'error' in result and imported == 0:
                            ui.notify(f"Import failed: {result['error']}", type='negative')
                        else:
                            message = f"Imported {imported} memories"
                            if skipped > 0:
                                message += f" (skipped {skipped} duplicates)"
                            ui.notify(message, type='positive')
                            
                            # Refresh the memory list
                            await self._load_memories()
                        
                    except Exception as e:
                        logger.error(f"Import failed: {str(e)}")
                        ui.notify(f'Import failed: {str(e)}', type='negative')
                
                import_button = ui.button('Import', on_click=import_memory).props('color=primary')
                import_button.disable()  # Initially disabled until file selected
        
        import_dialog.open()
    
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