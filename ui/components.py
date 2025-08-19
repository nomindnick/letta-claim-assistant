"""
Enhanced UI components with loading states, animations, and polish.

Provides reusable components with smooth transitions, loading indicators,
and keyboard shortcut support.
"""

from nicegui import ui, events
from typing import Optional, Callable, List, Dict, Any
import asyncio
from datetime import datetime


class LoadingSpinner:
    """Animated loading spinner component."""
    
    def __init__(self, size: str = "md", color: str = "primary"):
        self.size = size
        self.color = color
        self.container = None
    
    def show(self, message: str = "Loading...") -> ui.element:
        """Show loading spinner with message."""
        with ui.row().classes('items-center gap-3 p-4') as container:
            # Animated spinner
            ui.spinner(size=self.size, color=self.color)
            ui.label(message).classes('text-gray-600')
        
        self.container = container
        return container
    
    def hide(self):
        """Hide the loading spinner."""
        if self.container:
            self.container.delete()
            self.container = None


class SkeletonLoader:
    """Skeleton loading placeholder for content."""
    
    @staticmethod
    def document_list_skeleton() -> ui.element:
        """Skeleton for document list."""
        with ui.column().classes('w-full gap-2 p-2') as skeleton:
            for _ in range(3):
                with ui.card().classes('w-full animate-pulse'):
                    with ui.card_section():
                        ui.skeleton().classes('h-4 w-3/4 mb-2')
                        ui.skeleton().classes('h-3 w-1/2 mb-1')
                        ui.skeleton().classes('h-3 w-1/4')
        return skeleton
    
    @staticmethod
    def chat_message_skeleton() -> ui.element:
        """Skeleton for chat message."""
        with ui.row().classes('w-full animate-pulse gap-3 p-3') as skeleton:
            ui.skeleton().classes('w-8 h-8 rounded-full')  # Avatar
            with ui.column().classes('flex-1'):
                ui.skeleton().classes('h-4 w-1/4 mb-2')  # Timestamp
                ui.skeleton().classes('h-4 w-full mb-1')  # Message line 1
                ui.skeleton().classes('h-4 w-3/4 mb-1')   # Message line 2
                ui.skeleton().classes('h-4 w-1/2')        # Message line 3
        return skeleton
    
    @staticmethod
    def source_panel_skeleton() -> ui.element:
        """Skeleton for source panel."""
        with ui.column().classes('w-full gap-2 p-2') as skeleton:
            for _ in range(4):
                with ui.card().classes('w-full animate-pulse'):
                    with ui.card_section():
                        ui.skeleton().classes('h-3 w-1/3 mb-2')  # Document name
                        ui.skeleton().classes('h-3 w-full mb-1')  # Text line 1
                        ui.skeleton().classes('h-3 w-full mb-1')  # Text line 2
                        ui.skeleton().classes('h-3 w-2/3')       # Text line 3
        return skeleton


class ProgressBar:
    """Enhanced progress bar with smooth animations."""
    
    def __init__(self, color: str = "primary", show_percentage: bool = True):
        self.color = color
        self.show_percentage = show_percentage
        self.progress_element = None
        self.label_element = None
        self.container = None
    
    def create(self, initial_value: float = 0.0, message: str = "Processing...") -> ui.element:
        """Create progress bar."""
        with ui.column().classes('w-full gap-2') as container:
            if message:
                self.label_element = ui.label(message).classes('text-sm text-gray-600')
            
            self.progress_element = ui.linear_progress(
                value=initial_value,
                color=self.color,
                show_value=self.show_percentage
            ).classes('w-full transition-all duration-300')
        
        self.container = container
        return container
    
    def update(self, value: float, message: Optional[str] = None):
        """Update progress value and message."""
        if self.progress_element:
            self.progress_element.value = value
        
        if message and self.label_element:
            self.label_element.text = message
    
    def complete(self, message: str = "Complete!", auto_hide_delay: float = 2.0):
        """Mark as complete and optionally auto-hide."""
        self.update(1.0, message)
        
        if auto_hide_delay > 0:
            asyncio.create_task(self._auto_hide(auto_hide_delay))
    
    async def _auto_hide(self, delay: float):
        """Auto-hide after delay."""
        await asyncio.sleep(delay)
        if self.container:
            self.container.delete()


class AnimatedCard:
    """Card component with hover animations and transitions."""
    
    def __init__(self, elevated: bool = True, hover_lift: bool = True):
        self.elevated = elevated
        self.hover_lift = hover_lift
        self.card = None
    
    def create(self, content_builder: Callable = None) -> ui.element:
        """Create animated card."""
        base_classes = 'transition-all duration-200'
        
        if self.elevated:
            base_classes += ' shadow-md'
        
        if self.hover_lift:
            base_classes += ' hover:shadow-lg hover:-translate-y-1'
        
        with ui.card().classes(base_classes) as card:
            if content_builder:
                content_builder()
        
        self.card = card
        return card


class NotificationManager:
    """Enhanced notification system with different types and auto-dismiss."""
    
    @staticmethod
    def success(message: str, duration: int = 3000, dismissible: bool = True):
        """Show success notification."""
        ui.notify(
            message,
            type="positive",
            timeout=duration,
            close_button=dismissible,
            position="top-right"
        )
    
    @staticmethod
    def error(message: str, duration: int = 5000, dismissible: bool = True):
        """Show error notification."""
        ui.notify(
            message,
            type="negative", 
            timeout=duration,
            close_button=dismissible,
            position="top-right"
        )
    
    @staticmethod
    def warning(message: str, duration: int = 4000, dismissible: bool = True):
        """Show warning notification."""
        ui.notify(
            message,
            type="warning",
            timeout=duration,
            close_button=dismissible,
            position="top-right"
        )
    
    @staticmethod
    def info(message: str, duration: int = 3000, dismissible: bool = True):
        """Show info notification."""
        ui.notify(
            message,
            type="info",
            timeout=duration,
            close_button=dismissible,
            position="top-right"
        )
    
    @staticmethod
    def processing(message: str = "Processing...", spinner: bool = True):
        """Show processing notification that stays until dismissed."""
        return ui.notify(
            message,
            type="ongoing",
            spinner=spinner,
            timeout=0,  # Stay until dismissed
            position="top-right"
        )


class KeyboardShortcuts:
    """Keyboard shortcut manager."""
    
    def __init__(self):
        self.shortcuts = {}
        self.help_dialog = None
    
    def register(self, key: str, callback: Callable, description: str = ""):
        """Register a keyboard shortcut."""
        self.shortcuts[key] = {
            "callback": callback,
            "description": description
        }
        
        # Register with NiceGUI
        ui.keyboard(on_key=callback)
    
    def show_help(self):
        """Show keyboard shortcuts help dialog."""
        if self.help_dialog:
            self.help_dialog.close()
        
        with ui.dialog() as dialog:
            with ui.card().classes('w-96'):
                with ui.card_section():
                    ui.label("Keyboard Shortcuts").classes('text-lg font-semibold')
                
                with ui.card_section():
                    with ui.column().classes('gap-2'):
                        for key, info in self.shortcuts.items():
                            with ui.row().classes('justify-between items-center'):
                                ui.label(key).classes('font-mono bg-gray-100 px-2 py-1 rounded text-sm')
                                ui.label(info["description"]).classes('text-sm text-gray-600')
                
                with ui.card_actions():
                    ui.button("Close", on_click=dialog.close)
        
        self.help_dialog = dialog
        dialog.open()


class SearchInput:
    """Enhanced search input with suggestions and keyboard navigation."""
    
    def __init__(self, placeholder: str = "Search...", on_search: Callable = None):
        self.placeholder = placeholder
        self.on_search = on_search
        self.input_element = None
        self.suggestions = []
        self.suggestion_menu = None
    
    def create(self) -> ui.element:
        """Create search input with enhancements."""
        with ui.row().classes('w-full relative') as container:
            self.input_element = ui.input(
                placeholder=self.placeholder,
                on_change=self._on_input_change
            ).classes('flex-1').props('outlined dense')
            
            # Search button
            ui.button(
                icon="search",
                on_click=self._perform_search
            ).props('flat dense')
        
        return container
    
    async def _on_input_change(self, e):
        """Handle input changes for suggestions."""
        query = e.value
        if len(query) >= 2:
            # Could implement search suggestions here
            pass
    
    async def _perform_search(self):
        """Perform search."""
        if self.on_search and self.input_element:
            await self.on_search(self.input_element.value)


class TabContainer:
    """Enhanced tab container with animations."""
    
    def __init__(self):
        self.tabs = None
        self.panels = {}
    
    def create(self, tab_definitions: List[Dict[str, Any]]) -> ui.element:
        """Create animated tab container."""
        with ui.tabs().classes('w-full') as tabs:
            for tab_def in tab_definitions:
                ui.tab(tab_def["name"], label=tab_def["label"], icon=tab_def.get("icon"))
        
        with ui.tab_panels(tabs, value=tab_definitions[0]["name"]).classes('w-full') as panels:
            for tab_def in tab_definitions:
                with ui.tab_panel(tab_def["name"]).classes('transition-all duration-200'):
                    if "content_builder" in tab_def:
                        tab_def["content_builder"]()
                    self.panels[tab_def["name"]] = panels
        
        self.tabs = tabs
        return panels


class CollapsibleSection:
    """Collapsible section with smooth animations."""
    
    def __init__(self, title: str, expanded: bool = False):
        self.title = title
        self.expanded = expanded
        self.content_container = None
        self.toggle_button = None
    
    def create(self, content_builder: Callable = None) -> ui.element:
        """Create collapsible section."""
        with ui.column().classes('w-full border rounded') as container:
            # Header
            with ui.row().classes('w-full items-center p-3 cursor-pointer hover:bg-gray-50').on('click', self._toggle):
                icon = "expand_less" if self.expanded else "expand_more"
                self.toggle_button = ui.icon(icon).classes('text-gray-600')
                ui.label(self.title).classes('flex-1 font-medium')
            
            # Content
            show_class = '' if self.expanded else 'hidden'
            with ui.column().classes(f'w-full transition-all duration-300 {show_class}') as content:
                if content_builder:
                    content_builder()
            
            self.content_container = content
        
        return container
    
    def _toggle(self):
        """Toggle expanded state."""
        self.expanded = not self.expanded
        
        if self.expanded:
            self.content_container.classes(remove='hidden')
            self.toggle_button.name = "expand_less"
        else:
            self.content_container.classes(add='hidden')
            self.toggle_button.name = "expand_more"


class ThinkingIndicator:
    """Animated thinking/typing indicator."""
    
    def __init__(self):
        self.container = None
        self.animation_timer = None
    
    def show(self, message: str = "Assistant is thinking...") -> ui.element:
        """Show thinking indicator with animation."""
        with ui.row().classes('items-center gap-3 p-3 bg-blue-50 rounded animate-fade-in') as container:
            # Animated dots
            with ui.row().classes('gap-1'):
                for i in range(3):
                    ui.element('div').classes(
                        f'w-2 h-2 bg-blue-500 rounded-full animate-bounce'
                    ).style(f'animation-delay: {i * 0.2}s')
            
            ui.label(message).classes('text-blue-700 text-sm')
        
        self.container = container
        return container
    
    def hide(self):
        """Hide thinking indicator."""
        if self.container:
            self.container.delete()
            self.container = None


class DocumentUploader:
    """Enhanced document uploader with drag-and-drop and progress."""
    
    def __init__(self, on_upload: Callable = None, accepted_types: List[str] = None):
        self.on_upload = on_upload
        self.accepted_types = accepted_types or ['.pdf']
        self.upload_area = None
    
    def create(self) -> ui.element:
        """Create enhanced upload area."""
        with ui.column().classes('w-full') as container:
            # Upload area
            with ui.card().classes(
                'w-full min-h-32 border-2 border-dashed border-gray-300 '
                'hover:border-blue-400 transition-colors duration-200 cursor-pointer'
            ) as upload_area:
                with ui.card_section().classes('text-center p-8'):
                    ui.icon('cloud_upload').classes('text-4xl text-gray-400 mb-2')
                    ui.label('Drag and drop PDF files here').classes('text-lg text-gray-600 mb-1')
                    ui.label('or click to browse').classes('text-sm text-gray-500')
                    
                    # Hidden file input
                    ui.upload(
                        on_upload=self._handle_upload,
                        multiple=True,
                        accept=','.join(self.accepted_types)
                    ).classes('absolute inset-0 opacity-0 cursor-pointer')
            
            self.upload_area = upload_area
        
        return container
    
    async def _handle_upload(self, e):
        """Handle file upload."""
        if self.on_upload:
            await self.on_upload(e)


# Custom CSS for animations and visual polish
ANIMATION_CSS = """
<style>
/* Enhanced animations */
@keyframes fade-in {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slide-up {
    from { transform: translateY(100%); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

@keyframes pulse-soft {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

@keyframes pulse-glow {
    0% { box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.4); }
    50% { box-shadow: 0 0 20px 10px rgba(168, 85, 247, 0.2); }
    100% { box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.4); }
}

@keyframes thinking-dots {
    0%, 20% { content: '.'; }
    40% { content: '..'; }
    60%, 100% { content: '...'; }
}

@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

@keyframes spin-smooth {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Animation classes */
.animate-fade-in {
    animation: fade-in 0.3s ease-out;
}

.animate-slide-up {
    animation: slide-up 0.4s ease-out;
}

.animate-pulse-soft {
    animation: pulse-soft 2s infinite;
}

.animate-pulse-glow {
    animation: pulse-glow 2s infinite;
}

.animate-spin-smooth {
    animation: spin-smooth 1s linear infinite;
}

.animate-thinking::after {
    content: '.';
    animation: thinking-dots 1.5s infinite;
}

/* Smooth transitions */
.transition-all {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.transition-fast {
    transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
}

.transition-colors {
    transition: background-color 0.2s, border-color 0.2s, color 0.2s;
}

/* Hover effects */
.hover-lift:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.hover-glow:hover {
    box-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
}

.hover-scale:hover {
    transform: scale(1.02);
}

/* Skeleton loading */
.skeleton-shimmer {
    background: linear-gradient(
        90deg,
        #f0f0f0 25%,
        #e0e0e0 50%,
        #f0f0f0 75%
    );
    background-size: 1000px 100%;
    animation: shimmer 2s infinite;
}

/* Glass morphism */
.glass-morphism {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Smooth scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 10px;
    transition: background 0.2s;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* Focus indicators */
.focus-ring:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.3);
}

/* Gradient backgrounds */
.gradient-purple {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.gradient-blue {
    background: linear-gradient(135deg, #667eea 0%, #4facfe 100%);
}

.gradient-green {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

/* Card shadows */
.card-shadow {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
               0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.card-shadow-lg {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1),
               0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

/* Status indicators */
.status-online::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10b981;
    margin-right: 4px;
    animation: pulse-soft 2s infinite;
}

.status-offline::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #ef4444;
    margin-right: 4px;
}

/* Memory badge glow */
.memory-active {
    box-shadow: 0 0 15px rgba(168, 85, 247, 0.5);
    border: 1px solid rgba(168, 85, 247, 0.3);
}

/* Smooth appearance */
.smooth-appear {
    opacity: 0;
    animation: fade-in 0.5s ease-out forwards;
}
</style>
"""


def inject_animations():
    """Inject custom CSS animations."""
    ui.add_head_html(ANIMATION_CSS)