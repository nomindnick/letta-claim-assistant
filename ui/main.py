"""
NiceGUI desktop interface for the Letta Construction Claim Assistant.

Provides the main 3-pane desktop application with matter management,
chat interface, and document sources display.
"""

from nicegui import ui, app
from pathlib import Path
import asyncio
from typing import Optional

from .api_client import APIClient
from ..app.logging_conf import get_logger, setup_logging

logger = get_logger(__name__)


class LettaClaimUI:
    """Main UI application class."""
    
    def __init__(self):
        self.api_client = APIClient()
        self.current_matter: Optional[str] = None
        self.matters_list = []
        
        # UI component references
        self.matter_selector = None
        self.document_list = None
        self.chat_messages = None
        self.sources_panel = None
    
    async def create_ui(self):
        """Create the main UI layout."""
        # Set up the page
        ui.page_title("Letta Construction Claim Assistant")
        
        # Create main layout
        with ui.row().classes('w-full h-screen'):
            # Left pane: Matter & Documents
            await self._create_left_pane()
            
            # Center pane: Chat Interface  
            await self._create_center_pane()
            
            # Right pane: Sources
            await self._create_right_pane()
        
        # Settings drawer
        await self._create_settings_drawer()
    
    async def _create_left_pane(self):
        """Create the left pane with matter management and documents."""
        with ui.column().classes('w-1/4 h-full border-r bg-gray-50 p-4'):
            ui.label('Matter & Documents').classes('text-lg font-bold mb-4')
            
            # Matter management
            with ui.card().classes('w-full mb-4'):
                ui.label('Active Matter').classes('font-semibold mb-2')
                
                self.matter_selector = ui.select(
                    [],
                    label='Select Matter',
                    on_change=self._on_matter_changed
                ).classes('w-full mb-2')
                
                ui.button(
                    'Create Matter',
                    on_click=self._show_create_matter_dialog
                ).classes('w-full')
            
            # Document upload
            with ui.card().classes('w-full mb-4'):
                ui.label('Documents').classes('font-semibold mb-2')
                
                ui.upload(
                    multiple=True,
                    max_file_size=100 * 1024 * 1024,  # 100MB
                    on_upload=self._handle_file_upload
                ).props('accept=".pdf"').classes('w-full mb-2')
            
            # Document list
            with ui.card().classes('w-full flex-1'):
                ui.label('Document Status').classes('font-semibold mb-2')
                self.document_list = ui.column().classes('w-full')
                
                # Placeholder content
                ui.label('No documents uploaded').classes('text-gray-500 text-center py-4')
    
    async def _create_center_pane(self):
        """Create the center pane with chat interface."""
        with ui.column().classes('w-1/2 h-full p-4'):
            ui.label('Chat').classes('text-lg font-bold mb-4')
            
            # Chat messages area
            with ui.card().classes('w-full flex-1 mb-4'):
                self.chat_messages = ui.column().classes('w-full h-full overflow-y-auto p-4')
                
                # Welcome message
                ui.markdown("""
                **Welcome to Letta Construction Claim Assistant!**
                
                1. Create or select a Matter
                2. Upload PDF documents for analysis
                3. Ask questions about your construction claim
                
                The assistant will provide answers with precise citations and suggest follow-up questions.
                """).classes('text-gray-600')
            
            # Chat input
            with ui.row().classes('w-full'):
                self.chat_input = ui.input(
                    placeholder='Ask a question about your claim...',
                    on_change=self._on_input_change
                ).classes('flex-1')
                
                self.send_button = ui.button(
                    'Send',
                    on_click=self._send_message
                ).classes('ml-2').props('disable')
    
    async def _create_right_pane(self):
        """Create the right pane with sources display."""
        with ui.column().classes('w-1/4 h-full border-l bg-gray-50 p-4'):
            ui.label('Sources').classes('text-lg font-bold mb-4')
            
            with ui.card().classes('w-full flex-1'):
                self.sources_panel = ui.column().classes('w-full h-full')
                
                # Placeholder content
                ui.label('Sources will appear here after asking questions').classes(
                    'text-gray-500 text-center py-4'
                )
    
    async def _create_settings_drawer(self):
        """Create the settings drawer."""
        # TODO: Implement settings drawer
        pass
    
    # Event handlers
    async def _on_matter_changed(self, e):
        """Handle matter selection change."""
        # TODO: Switch matter context
        pass
    
    async def _show_create_matter_dialog(self):
        """Show create matter dialog."""
        # TODO: Implement matter creation dialog
        pass
    
    async def _handle_file_upload(self, e):
        """Handle PDF file upload."""
        # TODO: Implement file upload processing
        pass
    
    def _on_input_change(self, e):
        """Handle chat input change."""
        # Enable/disable send button based on input
        has_text = bool(e.value.strip())
        self.send_button.props(f'{"" if has_text else "disable"}')
    
    async def _send_message(self):
        """Send chat message."""
        # TODO: Implement message sending
        pass


async def create_app():
    """Create and configure the NiceGUI application."""
    
    # Set up logging
    setup_logging(debug=True)
    logger.info("Starting Letta Construction Claim Assistant UI")
    
    # Create UI instance
    ui_app = LettaClaimUI()
    await ui_app.create_ui()
    
    return ui_app


def main():
    """Main entry point for the UI application."""
    
    # Create the app
    @ui.page('/')
    async def index():
        await create_app()
    
    # Run the application
    try:
        # Try native desktop mode first
        ui.run(
            native=True,
            window_size=(1400, 900),
            title="Letta Construction Claim Assistant"
        )
    except Exception as e:
        logger.warning(f"Native desktop mode failed, falling back to browser: {e}")
        # Fallback to browser mode
        ui.run(
            host='localhost',
            port=8080,
            title="Letta Construction Claim Assistant"
        )


if __name__ == "__main__":
    main()