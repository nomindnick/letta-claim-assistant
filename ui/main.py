"""
NiceGUI desktop interface for the Letta Construction Claim Assistant.

Provides the main 3-pane desktop application with matter management,
chat interface, and document sources display.
"""

from nicegui import ui, app
from pathlib import Path
import asyncio
from typing import Optional, List, Dict, Any
import time
import tempfile

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ui.api_client import APIClient
from app.logging_conf import get_logger, setup_logging

logger = get_logger(__name__)


class LettaClaimUI:
    """Main UI application class."""
    
    def __init__(self):
        self.api_client = APIClient()
        self.current_matter: Optional[Dict[str, Any]] = None
        self.matters_list: List[Dict[str, Any]] = []
        self.upload_jobs: Dict[str, Dict] = {}  # Track upload jobs
        
        # UI component references
        self.matter_selector = None
        self.document_list = None
        self.chat_messages = None
        self.sources_panel = None
        self.chat_input = None
        self.send_button = None
        self.settings_drawer = None
        
    async def create_ui(self):
        """Create the main UI layout."""
        # Set up the page
        ui.page_title("Letta Construction Claim Assistant")
        
        # Check backend connectivity
        await self._check_backend_connection()
        
        # Load initial data
        await self._load_matters()
        
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
        
        # Start background job polling
        ui.timer(2.0, self._poll_job_status)
    
    async def _check_backend_connection(self):
        """Check if backend API is available."""
        try:
            is_healthy = await self.api_client.health_check()
            if not is_healthy:
                ui.notify("Backend API not available. Please check server status.", type="negative")
        except Exception as e:
            ui.notify(f"Cannot connect to backend: {str(e)}", type="negative")
    
    async def _load_matters(self):
        """Load matters from the API."""
        try:
            self.matters_list = await self.api_client.list_matters()
            
            # Try to get active matter
            active_matter = await self.api_client.get_active_matter()
            if active_matter:
                self.current_matter = active_matter
                
        except Exception as e:
            logger.error("Failed to load matters", error=str(e))
            ui.notify(f"Failed to load matters: {str(e)}", type="negative")
    
    async def _create_left_pane(self):
        """Create the left pane with matter management and documents."""
        with ui.column().classes('w-1/4 h-full border-r bg-gray-50 p-4'):
            ui.label('Matter & Documents').classes('text-lg font-bold mb-4')
            
            # Matter management
            with ui.card().classes('w-full mb-4'):
                ui.label('Active Matter').classes('font-semibold mb-2')
                
                # Matter selector options
                matter_options = {}
                for matter in self.matters_list:
                    matter_options[matter['id']] = matter['name']
                
                current_value = self.current_matter['id'] if self.current_matter else None
                
                self.matter_selector = ui.select(
                    matter_options,
                    label='Select Matter',
                    value=current_value,
                    on_change=self._on_matter_changed
                ).classes('w-full mb-2')
                
                ui.button(
                    'Create Matter',
                    on_click=self._show_create_matter_dialog,
                    icon='add'
                ).classes('w-full')
            
            # Document upload
            with ui.card().classes('w-full mb-4'):
                ui.label('Documents').classes('font-semibold mb-2')
                
                self.upload_widget = ui.upload(
                    multiple=True,
                    max_file_size=100 * 1024 * 1024,  # 100MB
                    on_upload=self._handle_file_upload,
                    on_rejected=self._handle_upload_rejected
                ).props('accept=".pdf"').classes('w-full mb-2')
                
                if not self.current_matter:
                    self.upload_widget.props('disable')
                    ui.label('Select a Matter to upload documents').classes('text-xs text-gray-500')
            
            # Document list
            with ui.card().classes('w-full flex-1'):
                ui.label('Document Status').classes('font-semibold mb-2')
                self.document_list = ui.column().classes('w-full')
                await self._refresh_document_list()
    
    async def _create_center_pane(self):
        """Create the center pane with chat interface."""
        with ui.column().classes('w-1/2 h-full p-4'):
            # Header with settings button
            with ui.row().classes('w-full justify-between items-center mb-4'):
                ui.label('Chat').classes('text-lg font-bold')
                ui.button(
                    icon='settings',
                    on_click=self._toggle_settings_drawer
                ).props('flat round')
            
            # Chat messages area
            with ui.card().classes('w-full flex-1 mb-4'):
                self.chat_messages = ui.column().classes('w-full h-full overflow-y-auto p-4')
                
                # Welcome message
                if self.current_matter:
                    welcome_text = f"""
                    **Active Matter:** {self.current_matter['name']}
                    
                    Upload PDF documents and ask questions about your construction claim.
                    The assistant will provide answers with precise citations.
                    """
                else:
                    welcome_text = """
                    **Welcome to Letta Construction Claim Assistant!**
                    
                    1. Create or select a Matter
                    2. Upload PDF documents for analysis
                    3. Ask questions about your construction claim
                    
                    The assistant will provide answers with precise citations and suggest follow-up questions.
                    """
                    
                ui.markdown(welcome_text).classes('text-gray-600')
            
            # Chat input
            with ui.row().classes('w-full'):
                self.chat_input = ui.input(
                    placeholder='Ask a question about your claim...',
                    on_change=self._on_input_change
                ).classes('flex-1').props('clearable')
                
                self.send_button = ui.button(
                    'Send',
                    on_click=self._send_message,
                    icon='send'
                ).classes('ml-2')
                
                # Disable if no matter selected
                if not self.current_matter:
                    self.chat_input.props('disable')
                    self.send_button.props('disable')
                else:
                    self.send_button.props('disable')  # Enabled when text entered
    
    async def _create_right_pane(self):
        """Create the right pane with sources display."""
        with ui.column().classes('w-1/4 h-full border-l bg-gray-50 p-4'):
            ui.label('Sources').classes('text-lg font-bold mb-4')
            
            with ui.card().classes('w-full flex-1'):
                self.sources_panel = ui.column().classes('w-full h-full overflow-y-auto')
                
                # Placeholder content
                ui.label('Sources will appear here after asking questions').classes(
                    'text-gray-500 text-center py-4'
                )
    
    async def _create_settings_drawer(self):
        """Create the settings drawer."""
        with ui.right_drawer(value=False, fixed=False).classes('bg-white') as drawer:
            self.settings_drawer = drawer
            
            with ui.column().classes('w-full p-4'):
                ui.label('Settings').classes('text-lg font-bold mb-4')
                
                # Provider settings
                with ui.card().classes('w-full mb-4'):
                    ui.label('LLM Provider').classes('font-semibold mb-2')
                    
                    ui.select(
                        {'ollama': 'Ollama (Local)', 'gemini': 'Gemini (External)'},
                        label='Provider',
                        value='ollama'
                    ).classes('w-full mb-2')
                    
                    ui.input(
                        label='Generation Model',
                        value='gpt-oss:20b'
                    ).classes('w-full mb-2')
                    
                    ui.input(
                        label='Embedding Model', 
                        value='nomic-embed-text'
                    ).classes('w-full mb-2')
                    
                    ui.button('Test Connection', icon='wifi').classes('w-full')
                
                # OCR settings
                with ui.card().classes('w-full mb-4'):
                    ui.label('OCR Settings').classes('font-semibold mb-2')
                    
                    ui.select(
                        {'eng': 'English', 'spa': 'Spanish', 'fra': 'French'},
                        label='Language',
                        value='eng'
                    ).classes('w-full mb-2')
                    
                    ui.checkbox('Force OCR on all pages').classes('w-full')
                    ui.checkbox('Skip text on born-digital PDFs', value=True).classes('w-full')
    
    async def _refresh_document_list(self):
        """Refresh the document list display."""
        self.document_list.clear()
        
        if not self.current_matter:
            with self.document_list:
                ui.label('Select a Matter to see documents').classes('text-gray-500 text-center py-4')
            return
        
        # Show upload jobs in progress
        for job_id, job_info in self.upload_jobs.items():
            if job_info['matter_id'] == self.current_matter['id']:
                with self.document_list:
                    with ui.card().classes('w-full mb-2 p-2'):
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.label(job_info['filename']).classes('font-medium')
                            if job_info['status'] == 'running':
                                ui.spinner()
                        
                        ui.linear_progress(
                            value=job_info.get('progress', 0),
                            show_value=True
                        ).classes('w-full mt-1')
                        
                        ui.label(job_info.get('message', 'Processing...')).classes('text-xs text-gray-600')
        
        # Show processed documents (placeholder for now)
        if not self.upload_jobs:
            with self.document_list:
                ui.label('No documents uploaded yet').classes('text-gray-500 text-center py-4')
    
    # Event handlers
    async def _on_matter_changed(self, e):
        """Handle matter selection change."""
        if not e.value:
            return
            
        try:
            # Switch matter in backend
            result = await self.api_client.switch_matter(e.value)
            
            # Update current matter
            for matter in self.matters_list:
                if matter['id'] == e.value:
                    self.current_matter = matter
                    break
            
            # Enable upload and chat
            self.upload_widget.props('enable')
            if self.chat_input:
                self.chat_input.props('enable')
            
            # Refresh UI
            await self._refresh_document_list()
            await self._update_chat_welcome()
            
            ui.notify(f"Switched to Matter: {self.current_matter['name']}", type="positive")
            
        except Exception as e:
            logger.error("Failed to switch matter", error=str(e))
            ui.notify(f"Failed to switch matter: {str(e)}", type="negative")
    
    async def _show_create_matter_dialog(self):
        """Show create matter dialog."""
        with ui.dialog() as dialog, ui.card():
            ui.label('Create New Matter').classes('text-lg font-bold mb-4')
            
            matter_name = ui.input(
                label='Matter Name',
                placeholder='e.g., Dry Well Claim - ABC Construction'
            ).classes('w-full mb-4')
            
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                ui.button(
                    'Create',
                    on_click=lambda: self._create_matter(dialog, matter_name.value)
                ).props('color=primary')
        
        dialog.open()
    
    async def _create_matter(self, dialog, name: str):
        """Create a new matter."""
        if not name or not name.strip():
            ui.notify("Please enter a matter name", type="negative")
            return
        
        try:
            # Create matter via API
            result = await self.api_client.create_matter(name.strip())
            
            # Refresh matters list
            await self._load_matters()
            
            # Update matter selector
            matter_options = {}
            for matter in self.matters_list:
                matter_options[matter['id']] = matter['name']
            self.matter_selector.options = matter_options
            
            # Select the new matter
            self.matter_selector.value = result['id']
            await self._on_matter_changed(type('obj', (object,), {'value': result['id']})())
            
            dialog.close()
            ui.notify(f"Created matter: {name}", type="positive")
            
        except Exception as e:
            logger.error("Failed to create matter", error=str(e))
            ui.notify(f"Failed to create matter: {str(e)}", type="negative")
    
    async def _handle_file_upload(self, e):
        """Handle PDF file upload."""
        if not self.current_matter:
            ui.notify("Please select a Matter first", type="negative")
            return
        
        try:
            # Convert uploaded files to temp files
            uploaded_files = []
            for file_info in e.files:
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.pdf',
                    delete=False,
                    prefix=f"upload_{file_info.name}_"
                )
                temp_file.write(file_info.content)
                temp_file.close()
                
                uploaded_files.append(Path(temp_file.name))
            
            # Submit upload job
            job_id = await self.api_client.upload_files(
                self.current_matter['id'],
                uploaded_files
            )
            
            # Track upload job
            for i, file_info in enumerate(e.files):
                self.upload_jobs[f"{job_id}_{i}"] = {
                    'job_id': job_id,
                    'matter_id': self.current_matter['id'],
                    'filename': file_info.name,
                    'status': 'queued',
                    'progress': 0.0,
                    'message': 'Queued for processing'
                }
            
            # Refresh document list
            await self._refresh_document_list()
            
            ui.notify(f"Uploaded {len(e.files)} file(s) for processing", type="positive")
            
        except Exception as e:
            logger.error("Failed to upload files", error=str(e))
            ui.notify(f"Failed to upload files: {str(e)}", type="negative")
    
    async def _handle_upload_rejected(self, e):
        """Handle rejected file uploads."""
        ui.notify("File upload rejected. Please ensure files are PDFs under 100MB.", type="negative")
    
    def _on_input_change(self, e):
        """Handle chat input change."""
        # Enable/disable send button based on input
        has_text = bool(e.value and e.value.strip())
        if has_text:
            self.send_button.props('color=primary')
            self.send_button.props(remove='disable')
        else:
            self.send_button.props('disable')
    
    async def _send_message(self):
        """Send chat message."""
        if not self.current_matter:
            ui.notify("Please select a Matter first", type="negative")
            return
        
        if not self.chat_input.value or not self.chat_input.value.strip():
            return
        
        query = self.chat_input.value.strip()
        self.chat_input.value = ''
        self.send_button.props('disable')
        
        # Add user message to chat
        with self.chat_messages:
            with ui.card().classes('w-full mb-2 bg-blue-50'):
                ui.markdown(f"**You:** {query}")
        
        # Show thinking indicator
        with self.chat_messages:
            thinking_card = ui.card().classes('w-full mb-2 bg-gray-50')
            with thinking_card:
                with ui.row().classes('items-center'):
                    ui.spinner()
                    ui.label('Thinking...').classes('ml-2')
        
        try:
            # Send to backend
            response = await self.api_client.send_chat_message(
                self.current_matter['id'],
                query
            )
            
            # Remove thinking indicator
            thinking_card.delete()
            
            # Add assistant response
            with self.chat_messages:
                with ui.card().classes('w-full mb-2 bg-green-50'):
                    ui.markdown(f"**Assistant:**\n\n{response['answer']}")
                    
                    # Add follow-up suggestions
                    if response.get('followups'):
                        ui.label('Suggested follow-ups:').classes('font-semibold mt-2 mb-1')
                        with ui.row().classes('gap-1'):
                            for suggestion in response['followups'][:3]:  # Limit to 3
                                ui.button(
                                    suggestion,
                                    on_click=lambda s=suggestion: self._use_suggestion(s)
                                ).props('size=sm outline')
            
            # Update sources panel
            await self._update_sources_panel(response.get('sources', []))
            
        except Exception as e:
            thinking_card.delete()
            logger.error("Failed to send message", error=str(e))
            ui.notify(f"Failed to send message: {str(e)}", type="negative")
    
    def _use_suggestion(self, suggestion: str):
        """Use a follow-up suggestion as the next query."""
        self.chat_input.value = suggestion
        self._on_input_change(type('obj', (object,), {'value': suggestion})())
    
    async def _update_sources_panel(self, sources: List[Dict[str, Any]]):
        """Update the sources panel with new sources."""
        self.sources_panel.clear()
        
        if not sources:
            with self.sources_panel:
                ui.label('No sources found').classes('text-gray-500 text-center py-4')
            return
        
        with self.sources_panel:
            for i, source in enumerate(sources):
                with ui.card().classes('w-full mb-2 p-2'):
                    # Header with document and page info
                    with ui.row().classes('w-full justify-between items-center mb-2'):
                        ui.label(f"{source['doc']} p.{source['page_start']}").classes('font-medium')
                        ui.label(f"sim={source['score']:.2f}").classes('text-xs bg-gray-200 px-1 rounded')
                    
                    # Source text snippet
                    ui.label(source['text'][:200] + '...' if len(source['text']) > 200 else source['text']).classes('text-sm text-gray-700 mb-2')
                    
                    # Action buttons
                    with ui.row().classes('gap-1'):
                        ui.button('Open PDF', icon='open_in_new').props('size=sm')
                        ui.button('Copy Citation', icon='copy').props('size=sm outline')
    
    async def _update_chat_welcome(self):
        """Update the chat welcome message."""
        # This would clear and recreate the welcome message
        # For now, we'll handle this in the matter change event
        pass
    
    def _toggle_settings_drawer(self):
        """Toggle the settings drawer."""
        if self.settings_drawer:
            self.settings_drawer.toggle()
    
    async def _poll_job_status(self):
        """Poll job status for uploads in progress."""
        if not self.upload_jobs:
            return
        
        jobs_to_remove = []
        
        for job_key, job_info in self.upload_jobs.items():
            if job_info['status'] in ['completed', 'failed']:
                continue
            
            try:
                status = await self.api_client.get_job_status(job_info['job_id'])
                
                # Update job info
                job_info['status'] = status['status']
                job_info['progress'] = status.get('progress', 0.0)
                job_info['message'] = status.get('message', 'Processing...')
                
                if status['status'] in ['completed', 'failed']:
                    jobs_to_remove.append(job_key)
                    
                    if status['status'] == 'completed':
                        ui.notify(f"Document processed: {job_info['filename']}", type="positive")
                    else:
                        ui.notify(f"Processing failed: {job_info['filename']}", type="negative")
                
            except Exception as e:
                logger.error("Failed to get job status", job_id=job_info['job_id'], error=str(e))
        
        # Remove completed jobs after a delay
        for job_key in jobs_to_remove:
            del self.upload_jobs[job_key]
        
        # Refresh document list if any jobs completed
        if jobs_to_remove:
            await self._refresh_document_list()


async def create_app():
    """Create and configure the NiceGUI application."""
    logger.info("Starting Letta Construction Claim Assistant UI")
    
    # Create UI instance
    ui_app = LettaClaimUI()
    await ui_app.create_ui()
    
    return ui_app


def create_ui_app():
    """Create and run the UI application (called from main.py)."""
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


def main():
    """Legacy main entry point (for backward compatibility)."""
    create_ui_app()


if __name__ == "__main__":
    main()