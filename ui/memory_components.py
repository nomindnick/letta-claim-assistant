"""
Memory-related UI components for Letta integration visibility.

Provides components for displaying memory status, statistics,
and operation indicators.
"""

from nicegui import ui
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime, timedelta
from .memory_viewer import MemoryViewer


class MemoryStatusBadge:
    """Badge component showing memory status in chat."""
    
    def __init__(self):
        self.badge_container = None
        self.status_text = None
        self.status_icon = None
        self.is_active = False
        
    def create(self) -> ui.element:
        """Create the memory status badge."""
        with ui.row().classes('items-center gap-1 px-2 py-1 rounded-full bg-purple-100 text-purple-700 text-xs') as container:
            self.status_icon = ui.icon('psychology').classes('text-sm')
            self.status_text = ui.label('Memory Ready').classes('font-medium')
        
        self.badge_container = container
        self.badge_container.classes('opacity-0 transition-opacity duration-300')
        return container
    
    def show_active(self, operation: str = "Using Memory"):
        """Show active memory operation."""
        if not self.badge_container:
            return
            
        self.is_active = True
        self.badge_container.classes(remove='opacity-0', add='opacity-100 animate-pulse')
        
        if self.status_text:
            self.status_text.text = operation
        
        if self.status_icon:
            self.status_icon.name = 'sync'
            self.status_icon.classes(add='animate-spin')
    
    def show_enhanced(self):
        """Show that response was memory-enhanced."""
        if not self.badge_container:
            return
            
        self.is_active = False
        self.badge_container.classes(remove='animate-pulse', add='opacity-100 bg-green-100 text-green-700')
        
        if self.status_text:
            self.status_text.text = 'Memory Enhanced'
        
        if self.status_icon:
            self.status_icon.name = 'verified'
            self.status_icon.classes(remove='animate-spin')
    
    def hide(self):
        """Hide the badge."""
        if self.badge_container:
            self.badge_container.classes(add='opacity-0', remove='opacity-100 animate-pulse')
            self.is_active = False


class MemoryStatsDashboard:
    """Dashboard component for memory statistics."""
    
    def __init__(self, api_client=None):
        self.dashboard_container = None
        self.stats_elements = {}
        self.auto_refresh_timer = None
        self.refresh_callback = None  # Callback to trigger refresh from parent
        self.api_client = api_client
        self.current_matter_id = None
        self.memory_viewer = None  # Will be created on demand
        
    def create(self) -> ui.element:
        """Create the memory statistics dashboard."""
        with ui.card().classes('w-full') as container:
            # Header
            with ui.row().classes('w-full justify-between items-center mb-3'):
                ui.label('Agent Memory').classes('font-semibold text-sm')
                self.refresh_button = ui.button(
                    icon='refresh',
                    on_click=self._refresh_stats
                ).props('flat round size=sm')
            
            # Stats grid
            with ui.column().classes('w-full gap-2'):
                # Memory items count
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Memory Items').classes('text-xs text-gray-600')
                    self.stats_elements['items_count'] = ui.label('0').classes('text-sm font-medium')
                
                # Connection status
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Connection').classes('text-xs text-gray-600')
                    with ui.row().classes('items-center gap-1'):
                        self.stats_elements['connection_icon'] = ui.icon('circle').classes('text-xs text-gray-400')
                        self.stats_elements['connection_text'] = ui.label('Disconnected').classes('text-xs')
                
                # Last sync time
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Last Sync').classes('text-xs text-gray-600')
                    self.stats_elements['last_sync'] = ui.label('Never').classes('text-xs')
                
                # Memory usage bar
                ui.label('Memory Usage').classes('text-xs text-gray-600 mt-2')
                self.stats_elements['memory_bar'] = ui.linear_progress(
                    value=0.0,
                    color='purple'
                ).classes('w-full h-2')
                
                # Action buttons
                with ui.row().classes('w-full gap-2 mt-2'):
                    self.summary_button = ui.button(
                        'View Summary',
                        icon='summarize',
                        on_click=self._show_memory_summary
                    ).props('outline size=sm').classes('flex-1')
                    
                    self.view_all_button = ui.button(
                        'View All Memories',
                        icon='visibility',
                        on_click=self._show_memory_viewer
                    ).props('outline size=sm').classes('flex-1')
        
        self.dashboard_container = container
        return container
    
    def set_matter_id(self, matter_id: str):
        """Set the current matter ID for API calls."""
        self.current_matter_id = matter_id
    
    async def update_stats(self, stats: Dict[str, Any]):
        """Update the dashboard with new stats."""
        if not self.stats_elements:
            return
        
        # Update memory items count
        items_count = stats.get('memory_items', 0)
        if 'items_count' in self.stats_elements:
            self.stats_elements['items_count'].text = str(items_count)
        
        # Update connection status
        connection_state = stats.get('connection_state', 'disconnected')
        if 'connection_icon' in self.stats_elements and 'connection_text' in self.stats_elements:
            if connection_state == 'connected':
                self.stats_elements['connection_icon'].classes(
                    remove='text-gray-400 text-yellow-500 text-red-500 animate-pulse',
                    add='text-green-500'
                )
                self.stats_elements['connection_text'].text = 'Connected'
            elif connection_state == 'connecting':
                self.stats_elements['connection_icon'].classes(
                    remove='text-gray-400 text-green-500 text-red-500',
                    add='text-yellow-500 animate-pulse'
                )
                self.stats_elements['connection_text'].text = 'Connecting...'
            else:
                self.stats_elements['connection_icon'].classes(
                    remove='text-green-500 text-yellow-500 animate-pulse',
                    add='text-red-500'
                )
                self.stats_elements['connection_text'].text = 'Disconnected'
        
        # Update last sync time
        if 'last_sync' in self.stats_elements:
            # For now, use current time as placeholder
            self.stats_elements['last_sync'].text = self._format_time_ago(datetime.now())
        
        # Update memory usage bar (estimate based on items)
        if 'memory_bar' in self.stats_elements:
            # Estimate usage (100 items = 50% full, 200 items = 100% full)
            usage = min(items_count / 200.0, 1.0)
            self.stats_elements['memory_bar'].value = usage
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as time ago."""
        now = datetime.now()
        diff = now - timestamp
        
        if diff < timedelta(minutes=1):
            return "Just now"
        elif diff < timedelta(hours=1):
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"
        elif diff < timedelta(days=1):
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            days = diff.days
            return f"{days}d ago"
    
    async def _refresh_stats(self):
        """Manually refresh statistics."""
        if self.refresh_button:
            # Show loading state
            self.refresh_button.props('loading')
            
            # Trigger refresh through parent component callback
            if self.refresh_callback:
                await self.refresh_callback()
            else:
                await asyncio.sleep(0.5)  # Fallback placeholder
            
            self.refresh_button.props(remove='loading')
    
    async def _show_memory_summary(self):
        """Show memory summary dialog."""
        if not self.api_client or not self.current_matter_id:
            ui.notify("No matter selected or API client not available", type="warning")
            return
        
        # Create loading dialog
        with ui.dialog() as dialog, ui.card().classes('w-96 max-w-full'):
            dialog.open()
            
            ui.label('Memory Summary').classes('text-lg font-bold mb-4')
            
            # Loading state
            loading_container = ui.column().classes('w-full items-center py-8')
            with loading_container:
                ui.spinner(size='lg', color='purple')
                ui.label('Loading memory summary...').classes('mt-4 text-gray-600')
            
            close_button = ui.button('Close', on_click=dialog.close).classes('w-full mt-4')
            close_button.visible = False
            
            # Fetch memory summary
            try:
                result = await self.api_client.get_memory_summary(
                    self.current_matter_id,
                    max_length=1000
                )
                
                # Clear loading state
                loading_container.clear()
                
                # Display summary
                with loading_container:
                    # Parse the summary text
                    summary_text = result.get('summary', 'No memory items found')
                    
                    # Create scrollable container for the summary
                    with ui.scroll_area().classes('w-full h-96 p-4 bg-gray-50 rounded'):
                        # Split summary into lines and format
                        lines = summary_text.split('\n')
                        
                        for line in lines:
                            if line.startswith('Memory contains'):
                                # Main header
                                ui.label(line).classes('font-semibold text-base mb-3 text-purple-700')
                            elif line.startswith('- '):
                                # Category headers
                                category_line = line[2:]  # Remove "- "
                                if ':' in category_line:
                                    category, count = category_line.split(':', 1)
                                    with ui.row().classes('items-center gap-2 mb-2'):
                                        # Icon based on category
                                        icon_map = {
                                            'Entity': 'person',
                                            'Event': 'event',
                                            'Issue': 'warning',
                                            'Fact': 'fact_check',
                                            'interaction': 'chat'
                                        }
                                        icon = icon_map.get(category, 'category')
                                        ui.icon(icon).classes('text-purple-600')
                                        ui.label(f"{category}:").classes('font-medium')
                                        ui.label(count.strip()).classes('text-gray-600')
                                else:
                                    ui.label(line).classes('ml-4 text-sm')
                            elif line.startswith('  Examples:'):
                                # Examples line
                                ui.label(line).classes('ml-8 text-xs text-gray-500 italic mb-2')
                            elif line.strip():
                                # Other content
                                ui.label(line).classes('text-sm text-gray-700')
                    
                    # Add matter info
                    ui.separator().classes('my-4')
                    with ui.row().classes('w-full justify-between items-center text-xs text-gray-500'):
                        ui.label(f"Matter: {result.get('matter_name', 'Unknown')}")
                        ui.label(f"ID: {result.get('matter_id', '')}")
                
                close_button.visible = True
                
            except Exception as e:
                # Clear loading state and show error
                loading_container.clear()
                with loading_container:
                    ui.icon('error').classes('text-4xl text-red-500 mb-4')
                    ui.label('Failed to load memory summary').classes('font-medium text-red-700')
                    ui.label(str(e)).classes('text-sm text-gray-600 mt-2')
                
                close_button.visible = True
                ui.notify(f"Failed to load memory summary: {str(e)}", type="negative")
    
    async def _show_memory_viewer(self):
        """Show the memory viewer dialog."""
        if not self.api_client or not self.current_matter_id:
            ui.notify("No matter selected or API client not available", type="warning")
            return
        
        # Create memory viewer if not exists
        if not self.memory_viewer:
            self.memory_viewer = MemoryViewer(api_client=self.api_client)
        
        # Show the viewer
        await self.memory_viewer.show(self.current_matter_id)
    
    def start_auto_refresh(self, interval: float = 30.0):
        """Start auto-refresh timer."""
        if self.auto_refresh_timer:
            self.auto_refresh_timer.cancel()
        
        async def refresh_loop():
            while True:
                await asyncio.sleep(interval)
                await self._refresh_stats()
        
        self.auto_refresh_timer = asyncio.create_task(refresh_loop())
    
    def stop_auto_refresh(self):
        """Stop auto-refresh timer."""
        if self.auto_refresh_timer:
            self.auto_refresh_timer.cancel()
            self.auto_refresh_timer = None


class AgentHealthIndicator:
    """Health indicator for Letta agent connection."""
    
    def __init__(self):
        self.indicator_container = None
        self.status_icon = None
        self.status_text = None
        self.details_button = None
        
    def create(self) -> ui.element:
        """Create the health indicator."""
        with ui.row().classes('items-center gap-2') as container:
            # Status icon with animation
            self.status_icon = ui.icon('circle').classes('text-sm text-gray-400')
            
            # Status text
            self.status_text = ui.label('Agent Status').classes('text-sm')
            
            # Details button
            self.details_button = ui.button(
                icon='info',
                on_click=self._show_health_details
            ).props('flat round size=sm')
        
        self.indicator_container = container
        return container
    
    def update_health(self, health_data: Dict[str, Any]):
        """Update health indicator with new data."""
        if not self.indicator_container:
            return
        
        status = health_data.get('status', 'unknown')
        
        if status == 'healthy':
            self.status_icon.name = 'check_circle'
            self.status_icon.classes(
                remove='text-gray-400 text-yellow-500 text-red-500 animate-pulse',
                add='text-green-500'
            )
            self.status_text.text = 'Agent Healthy'
        elif status == 'degraded':
            self.status_icon.name = 'warning'
            self.status_icon.classes(
                remove='text-gray-400 text-green-500 text-red-500',
                add='text-yellow-500 animate-pulse'
            )
            self.status_text.text = 'Agent Degraded'
        elif status == 'error':
            self.status_icon.name = 'error'
            self.status_icon.classes(
                remove='text-gray-400 text-green-500 text-yellow-500 animate-pulse',
                add='text-red-500'
            )
            self.status_text.text = 'Agent Error'
        else:
            self.status_icon.name = 'help'
            self.status_icon.classes(
                remove='text-green-500 text-yellow-500 text-red-500 animate-pulse',
                add='text-gray-400'
            )
            self.status_text.text = 'Agent Unknown'
    
    async def _show_health_details(self):
        """Show detailed health information dialog."""
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            ui.label('Agent Health Details').classes('text-lg font-bold mb-4')
            
            # Placeholder for health details
            with ui.column().classes('w-full gap-2'):
                ui.label('Connection Status: Connected').classes('text-sm')
                ui.label('Server Status: Running').classes('text-sm')
                ui.label('Provider: Ollama (Local)').classes('text-sm')
                ui.label('Response Time: 245ms').classes('text-sm')
                ui.label('Memory Operations: Available').classes('text-sm')
            
            ui.button('Close', on_click=dialog.close).classes('w-full mt-4')
        
        dialog.open()


class MemoryOperationToast:
    """Toast notification for memory operations."""
    
    @staticmethod
    def show_storing(message: str = "Storing to memory..."):
        """Show memory storing notification."""
        ui.notify(
            message,
            type='ongoing',
            position='bottom-right',
            icon='save',
            spinner=True,
            timeout=3000
        )
    
    @staticmethod
    def show_recalling(message: str = "Recalling from memory..."):
        """Show memory recalling notification."""
        ui.notify(
            message,
            type='ongoing',
            position='bottom-right',
            icon='psychology',
            spinner=True,
            timeout=3000
        )
    
    @staticmethod
    def show_success(message: str = "Memory operation successful"):
        """Show success notification."""
        ui.notify(
            message,
            type='positive',
            position='bottom-right',
            icon='check_circle',
            timeout=2000
        )
    
    @staticmethod
    def show_error(message: str = "Memory operation failed"):
        """Show error notification."""
        ui.notify(
            message,
            type='negative',
            position='bottom-right',
            icon='error',
            timeout=4000
        )
    
    @staticmethod
    def show_info(message: str = "Memory operation"):
        """Show info notification."""
        ui.notify(
            message,
            type='info',
            position='bottom-right',
            icon='info',
            timeout=3000
        )


class MemoryContextTooltip:
    """Tooltip component for showing memory context."""
    
    @staticmethod
    def create(element: ui.element, memory_items: List[str]):
        """Add memory context tooltip to an element."""
        if not memory_items:
            return
        
        # Create tooltip content
        tooltip_text = "Memory Context:\n" + "\n".join(f"• {item}" for item in memory_items[:5])
        if len(memory_items) > 5:
            tooltip_text += f"\n... and {len(memory_items) - 5} more"
        
        # Add tooltip to element
        element.tooltip(tooltip_text).classes('cursor-help')