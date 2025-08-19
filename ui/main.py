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
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ui.api_client import APIClient
from ui.utils import (
    open_pdf_at_page, copy_to_clipboard, ChatMessage, 
    format_citation, format_timestamp, truncate_text
)
from ui.components import (
    LoadingSpinner, SkeletonLoader, ProgressBar, AnimatedCard,
    NotificationManager, KeyboardShortcuts, SearchInput, ThinkingIndicator,
    DocumentUploader, inject_animations
)
from ui.memory_components import (
    MemoryStatusBadge, MemoryStatsDashboard, AgentHealthIndicator,
    MemoryOperationToast, MemoryContextTooltip
)
from ui.performance import (
    chat_debouncer, search_debouncer, response_cache,
    lazy_loader, with_loading_state, measure_performance
)
from ui.error_messages import ErrorMessageHandler, handle_ui_error
from app.logging_conf import get_logger, setup_logging

logger = get_logger(__name__)


class LettaClaimUI:
    """Main UI application class."""
    
    def __init__(self):
        self.api_client = APIClient()
        self.current_matter: Optional[Dict[str, Any]] = None
        self.matters_list: List[Dict[str, Any]] = []
        self.upload_jobs: Dict[str, Dict] = {}  # Track upload jobs
        self.chat_history: List[Dict[str, Any]] = []  # Chat message history
        self.current_sources: List[Dict[str, Any]] = []  # Current sources for display
        
        # UI component references
        self.matter_selector = None
        self.document_list = None
        self.chat_messages = None
        self.sources_panel = None
        self.chat_input = None
        self.send_button = None
        self.settings_drawer = None
        
        # Enhanced UI components
        self.loading_spinner = LoadingSpinner()
        self.notification_manager = NotificationManager()
        self.keyboard_shortcuts = KeyboardShortcuts()
        self.thinking_indicator = ThinkingIndicator()
        self.current_progress_bar = None
        
        # Memory components
        self.memory_badge = MemoryStatusBadge()
        self.memory_dashboard = MemoryStatsDashboard()
        self.agent_health = AgentHealthIndicator()
        self.memory_stats_timer = None
        
        # UI state
        self.is_processing = False
        self.document_skeleton = None
        self.sources_skeleton = None
        
    async def create_ui(self):
        """Create the main UI layout with enhanced components."""
        # Inject custom CSS animations
        inject_animations()
        
        # Set up the page
        ui.page_title("Letta Construction Claim Assistant")
        ui.add_head_html('<meta name="viewport" content="width=device-width, initial-scale=1">')
        
        # Setup keyboard shortcuts
        # self._setup_keyboard_shortcuts()  # Disabled due to NiceGUI API changes
        
        # Show initial loading
        loading_container = self.loading_spinner.show("Initializing application...")
        
        try:
            # Check backend connectivity
            await self._check_backend_connection()
            
            # Load initial data
            await self._load_matters()
            
            # Hide loading
            self.loading_spinner.hide()
            
            # Create main layout with animations
            with ui.row().classes('w-full h-screen animate-fade-in'):
                # Left pane: Matter & Documents
                await self._create_left_pane()
                
                # Center pane: Chat Interface  
                await self._create_center_pane()
                
                # Right pane: Sources
                await self._create_right_pane()
            
            # Settings drawer
            await self._create_settings_drawer()
            
            # Initialize chat display
            await self._update_chat_display()
            
            # Start background job polling
            ui.timer(2.0, self._poll_job_status)
            
            # Start memory stats refresh timer
            ui.timer(30.0, self._refresh_memory_stats)
            
            # Start agent health monitoring
            ui.timer(10.0, self._update_agent_health)
            
            # Initialize consent checking for any existing external providers
            await self._check_provider_consent_status()
            
            # Initial memory stats load
            await self._refresh_memory_stats()
            await self._update_agent_health()
            
            # Show welcome notification
            self.notification_manager.success("Application loaded successfully!")
            
        except Exception as e:
            self.loading_spinner.hide()
            self.notification_manager.error(f"Failed to initialize application: {str(e)}")
            logger.error("UI initialization failed", error=str(e))
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for the application."""
        # Create new matter
        self.keyboard_shortcuts.register(
            "ctrl+n",
            lambda: asyncio.create_task(self._show_create_matter_dialog()),
            "Create new matter"
        )
        
        # Send message
        self.keyboard_shortcuts.register(
            "ctrl+enter",
            lambda: asyncio.create_task(self._send_message()),
            "Send chat message"
        )
        
        # Focus search/input
        self.keyboard_shortcuts.register(
            "ctrl+k",
            lambda: self._focus_chat_input(),
            "Focus chat input"
        )
        
        # Open settings
        self.keyboard_shortcuts.register(
            "ctrl+comma",
            lambda: self._toggle_settings_drawer(),
            "Toggle settings"
        )
        
        # Show help
        self.keyboard_shortcuts.register(
            "F1",
            lambda: self.keyboard_shortcuts.show_help(),
            "Show keyboard shortcuts"
        )
        
        # Escape to clear/cancel
        self.keyboard_shortcuts.register(
            "Escape",
            lambda: self._handle_escape(),
            "Cancel current operation"
        )
    
    def _focus_chat_input(self):
        """Focus the chat input field."""
        if self.chat_input:
            self.chat_input.run_method('focus')
    
    def _toggle_settings_drawer(self):
        """Toggle settings drawer."""
        if self.settings_drawer:
            # NiceGUI drawer toggle logic would go here
            pass
    
    def _handle_escape(self):
        """Handle escape key - cancel operations, close dialogs."""
        if self.is_processing:
            # Could implement cancellation logic here
            pass
    
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
            
            # Memory Statistics Dashboard
            self.memory_dashboard.create()
            
            # Document list
            with ui.card().classes('w-full flex-1 mt-4'):
                ui.label('Document Status').classes('font-semibold mb-2')
                self.document_list = ui.column().classes('w-full')
                await self._refresh_document_list()
    
    async def _create_center_pane(self):
        """Create the center pane with chat interface."""
        with ui.column().classes('w-1/2 h-full p-4'):
            # Header with settings button and agent health
            with ui.row().classes('w-full justify-between items-center mb-4'):
                with ui.row().classes('items-center gap-3'):
                    ui.label('Chat').classes('text-lg font-bold')
                    # Add memory status badge here
                    self.memory_badge.create()
                
                with ui.row().classes('items-center gap-2'):
                    # Agent health indicator
                    self.agent_health.create()
                    ui.button(
                        icon='settings',
                        on_click=self._toggle_settings_drawer
                    ).props('flat round')
            
            # Chat messages area
            with ui.card().classes('w-full flex-1 mb-4'):
                self.chat_messages = ui.column().classes('w-full h-full overflow-y-auto p-4')
                # Chat messages will be populated by _update_chat_display()
            
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
                    
                    self.provider_select = ui.select(
                        {'ollama': 'Ollama (Local)', 'gemini': 'Gemini (External)'},
                        label='Provider',
                        value='ollama',
                        on_change=self._on_provider_changed
                    ).classes('w-full mb-2')
                    
                    self.generation_model_input = ui.input(
                        label='Generation Model',
                        value='gpt-oss:20b'
                    ).classes('w-full mb-2')
                    
                    self.embedding_model_input = ui.input(
                        label='Embedding Model', 
                        value='nomic-embed-text'
                    ).classes('w-full mb-2')
                    
                    # API key input (hidden by default)
                    self.api_key_input = ui.input(
                        label='API Key',
                        password=True,
                        placeholder='Enter API key for external providers'
                    ).classes('w-full mb-2 hidden')
                    
                    with ui.row().classes('gap-2 w-full'):
                        ui.button(
                            'Test Connection',
                            icon='wifi',
                            on_click=self._test_provider_connection
                        ).classes('flex-1')
                        ui.button(
                            'Save Settings',
                            icon='save',
                            on_click=self._save_provider_settings
                        ).classes('flex-1').props('color=primary')
                
                # OCR settings
                with ui.card().classes('w-full mb-4'):
                    ui.label('OCR Settings').classes('font-semibold mb-2')
                    
                    self.ocr_language_select = ui.select(
                        {'eng': 'English', 'spa': 'Spanish', 'fra': 'French'},
                        label='Language',
                        value='eng'
                    ).classes('w-full mb-2')
                    
                    self.force_ocr_checkbox = ui.checkbox('Force OCR on all pages').classes('w-full')
                    self.skip_text_checkbox = ui.checkbox('Skip text on born-digital PDFs', value=True).classes('w-full')
        
        # Load current settings
        await self._load_current_settings()
    
    async def _load_current_settings(self):
        """Load current settings from backend."""
        try:
            settings = await self.api_client.get_model_settings()
            
            # Update provider settings
            active_provider = settings.get('active_provider', 'ollama')
            self.provider_select.value = active_provider
            
            # Show/hide API key field based on provider
            await self._on_provider_changed({'value': active_provider})
            
            logger.info("Loaded current settings", active_provider=active_provider)
            
        except Exception as e:
            logger.error("Failed to load settings", error=str(e))
            ui.notify("Failed to load settings", type="negative")
    
    async def _on_provider_changed(self, e):
        """Handle provider selection change."""
        provider = e.value if hasattr(e, 'value') else 'ollama'
        
        if provider == 'gemini':
            self.api_key_input.classes(remove='hidden')
            self.generation_model_input.value = 'gemini-2.0-flash-exp'
            self.embedding_model_input.props('disable')  # Gemini doesn't do embeddings
        else:
            self.api_key_input.classes(add='hidden')
            self.generation_model_input.value = 'gpt-oss:20b'
            self.embedding_model_input.props(remove='disable')
            self.embedding_model_input.value = 'nomic-embed-text'
    
    async def _test_provider_connection(self):
        """Test connection to selected provider."""
        try:
            provider = self.provider_select.value
            generation_model = self.generation_model_input.value
            api_key = self.api_key_input.value if provider == 'gemini' else None
            
            # Show loading indicator
            with ui.dialog() as test_dialog:
                with ui.card().classes('w-64 p-4'):
                    ui.label('Testing connection...').classes('text-center mb-2')
                    ui.spinner(size='lg').classes('mx-auto')
            
            test_dialog.open()
            
            try:
                # Test with consent checking
                response = await self._test_provider_with_consent(
                    provider_type=provider,
                    generation_model=generation_model,
                    api_key=api_key
                )
                
                test_dialog.close()
                
                if response and response.get("success"):
                    ui.notify(f"✓ {provider.title()} connection successful", type="positive")
                elif response and response.get("error") == "consent_required":
                    # Consent dialog is already shown, just provide feedback
                    ui.notify("Please respond to the consent dialog to continue.", type="info")
                else:
                    error_msg = response.get("message") if response else "Unknown error"
                    ui.notify(f"✗ Connection failed: {error_msg}", type="negative")
                    
            except Exception as e:
                test_dialog.close()
                raise e
                
        except Exception as e:
            logger.error("Failed to test provider", error=str(e))
            ui.notify(f"Test failed: {str(e)}", type="negative")
    
    async def _save_provider_settings(self):
        """Save provider settings."""
        try:
            provider = self.provider_select.value
            generation_model = self.generation_model_input.value
            embedding_model = self.embedding_model_input.value
            api_key = self.api_key_input.value if provider == 'gemini' else None
            
            # Save via API
            result = await self.api_client.update_model_settings(
                provider=provider,
                generation_model=generation_model,
                embedding_model=embedding_model,
                api_key=api_key
            )
            
            if result.get('success', False):
                ui.notify("Settings saved successfully", type="positive")
            else:
                error_msg = result.get('error', 'Unknown error')
                ui.notify(f"Failed to save settings: {error_msg}", type="negative")
                
        except Exception as e:
            logger.error("Failed to save settings", error=str(e))
            ui.notify(f"Failed to save settings: {str(e)}", type="negative")
    
    async def _refresh_document_list(self):
        """Refresh the document list display."""
        self.document_list.clear()
        
        if not self.current_matter:
            with self.document_list:
                ui.label('Select a Matter to see documents').classes('text-gray-500 text-center py-4')
            return
        
        try:
            # Get actual documents from backend
            documents = await self.api_client.get_matter_documents(self.current_matter['id'])
            
            # Show upload jobs in progress first
            active_jobs = 0
            for job_id, job_info in self.upload_jobs.items():
                if job_info['matter_id'] == self.current_matter['id']:
                    active_jobs += 1
                    with self.document_list:
                        await self._create_upload_job_card(job_info)
            
            # Show processed documents
            if documents:
                with self.document_list:
                    for doc in documents:
                        await self._create_document_card(doc)
            elif active_jobs == 0:
                with self.document_list:
                    ui.label('No documents uploaded yet').classes('text-gray-500 text-center py-4')
                    ui.label('Upload PDF files to begin analysis').classes('text-xs text-gray-400 text-center')
        
        except Exception as e:
            logger.error("Failed to refresh document list", error=str(e))
            with self.document_list:
                ui.label('Error loading documents').classes('text-red-500 text-center py-4')
    
    async def _create_upload_job_card(self, job_info: Dict[str, Any]):
        """Create a card for an upload job in progress."""
        with ui.card().classes('w-full mb-2 p-2 border-l-4 border-blue-500'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label(job_info['filename']).classes('font-medium text-sm')
                if job_info['status'] == 'running':
                    ui.spinner().classes('text-blue-500')
                elif job_info['status'] == 'completed':
                    ui.icon('check_circle').classes('text-green-500')
                elif job_info['status'] == 'failed':
                    ui.icon('error').classes('text-red-500')
            
            if job_info['status'] in ['queued', 'running']:
                ui.linear_progress(
                    value=job_info.get('progress', 0),
                    show_value=True
                ).classes('w-full mt-1')
            
            status_message = job_info.get('message', 'Processing...')
            ui.label(status_message).classes('text-xs text-gray-600')
    
    async def _create_document_card(self, doc: Dict[str, Any]):
        """Create a card for a processed document."""
        doc_name = doc.get('name', 'Unknown Document')
        pages = doc.get('pages', 0)
        chunks = doc.get('chunks', 0)
        ocr_status = doc.get('ocr_status', 'none')
        status = doc.get('status', 'pending')
        error_message = doc.get('error_message')
        file_size = doc.get('file_size', 0)
        
        # Status color mapping
        status_colors = {
            'completed': 'border-green-500',
            'processing': 'border-yellow-500',
            'failed': 'border-red-500',
            'pending': 'border-gray-500'
        }
        
        border_class = status_colors.get(status, 'border-gray-500')
        
        with ui.card().classes(f'w-full mb-2 p-3 border-l-4 {border_class}'):
            # Header with document name and status
            with ui.row().classes('w-full justify-between items-start mb-2'):
                with ui.column().classes('flex-1'):
                    ui.label(doc_name).classes('font-medium text-sm')
                    
                    # Document stats
                    stats = []
                    if pages > 0:
                        stats.append(f"{pages} pages")
                    if chunks > 0:
                        stats.append(f"{chunks} chunks")
                    if file_size > 0:
                        size_mb = file_size / (1024 * 1024)
                        stats.append(f"{size_mb:.1f}MB")
                    
                    if stats:
                        ui.label(" • ".join(stats)).classes('text-xs text-gray-600')
                
                # Status indicators
                with ui.column().classes('items-end'):
                    # Processing status
                    if status == 'completed':
                        ui.icon('check_circle').classes('text-green-500')
                    elif status == 'processing':
                        ui.spinner().classes('text-yellow-500')
                    elif status == 'failed':
                        ui.icon('error').classes('text-red-500')
                    else:
                        ui.icon('pending').classes('text-gray-500')
                    
                    # OCR status
                    if ocr_status == 'full':
                        ui.label('OCR: Full').classes('text-xs bg-green-100 text-green-800 px-1 rounded')
                    elif ocr_status == 'partial':
                        ui.label('OCR: Partial').classes('text-xs bg-yellow-100 text-yellow-800 px-1 rounded')
                    else:
                        ui.label('OCR: None').classes('text-xs bg-gray-100 text-gray-800 px-1 rounded')
            
            # Error message if failed
            if error_message:
                ui.label(f"Error: {error_message}").classes('text-xs text-red-600 bg-red-50 p-1 rounded')
            
            # Action buttons for completed documents
            if status == 'completed':
                with ui.row().classes('gap-1 mt-2'):
                    ui.button(
                        'Open PDF',
                        icon='picture_as_pdf',
                        on_click=lambda d=doc: self._open_document_pdf(d)
                    ).props('size=sm outline')
                    
                    if chunks > 0:
                        ui.button(
                            'View Chunks',
                            icon='view_list',
                            on_click=lambda d=doc: self._show_document_chunks(d)
                        ).props('size=sm outline')
            
            elif status == 'failed':
                with ui.row().classes('gap-1 mt-2'):
                    ui.button(
                        'Retry Processing',
                        icon='refresh',
                        on_click=lambda d=doc: self._retry_document_processing(d)
                    ).props('size=sm color=warning')
    
    async def _open_document_pdf(self, doc: Dict[str, Any]):
        """Open the document PDF."""
        if not self.current_matter:
            ui.notify("No active matter", type="negative")
            return
        
        doc_name = doc.get('name', '')
        if not doc_name:
            ui.notify("Document name not available", type="negative")
            return
        
        # Construct path to document
        matter_root = Path.home() / "LettaClaims" / f"Matter_{self.current_matter.get('slug', '')}"
        doc_path = matter_root / "docs" / doc_name
        
        success = await open_pdf_at_page(doc_path, 1)
        if not success:
            ui.notify(f"Could not open {doc_name}", type="negative")
    
    async def _show_document_chunks(self, doc: Dict[str, Any]):
        """Show document chunks in a dialog."""
        ui.notify("Chunk viewer not yet implemented", type="info")
        # TODO: Implement chunk viewer dialog
    
    async def _retry_document_processing(self, doc: Dict[str, Any]):
        """Retry processing a failed document."""
        ui.notify("Document retry not yet implemented", type="info")
        # TODO: Implement document retry functionality
    
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
            await self._load_chat_history()
            
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
    
    @chat_debouncer.debounce
    async def _on_input_change(self, e):
        """Handle chat input change with debouncing."""
        # Enable/disable send button based on input
        has_text = bool(e.value and e.value.strip())
        if has_text:
            self.send_button.props('color=primary')
            self.send_button.props(remove='disable')
        else:
            self.send_button.props('disable')
        
        # Could add input suggestions here in the future
        # if has_text and len(e.value) > 3:
        #     suggestions = await self._get_input_suggestions(e.value)
    
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
        
        # Add user message to chat history and display
        user_message = {
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat(),
            "sources": [],
            "followups": [],
            "used_memory": []
        }
        self.chat_history.append(user_message)
        
        # Add user message to display
        with self.chat_messages:
            await self._add_message_to_display(user_message)
        
        # Show memory status - recalling
        self.memory_badge.show_active("Recalling Memory")
        MemoryOperationToast.show_recalling()
        
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
            
            # Check if memory was used
            used_memory = response.get('used_memory', [])
            if used_memory:
                self.memory_badge.show_enhanced()
                MemoryOperationToast.show_success("Memory enhanced response")
            else:
                self.memory_badge.hide()
            
            # Add assistant message to chat history
            assistant_message = {
                "role": "assistant",
                "content": response['answer'],
                "timestamp": datetime.now().isoformat(),
                "sources": response.get('sources', []),
                "followups": response.get('followups', []),
                "used_memory": used_memory
            }
            self.chat_history.append(assistant_message)
            
            # Add assistant response to display
            with self.chat_messages:
                await self._add_message_to_display(assistant_message)
            
            # Auto-scroll to bottom of chat
            await self._scroll_chat_to_bottom()
            
            # Update sources panel
            await self._update_sources_panel(response.get('sources', []))
            
            # Store interaction in memory (show storing indicator)
            if used_memory:
                self.memory_badge.show_active("Storing to Memory")
                MemoryOperationToast.show_storing()
                await asyncio.sleep(1.5)  # Simulate storing
                self.memory_badge.show_enhanced()
            
            # Refresh memory stats
            await self._refresh_memory_stats()
            
        except Exception as e:
            thinking_card.delete()
            self.memory_badge.hide()
            MemoryOperationToast.show_error(f"Failed: {str(e)}")
            logger.error("Failed to send message", error=str(e))
            ui.notify(f"Failed to send message: {str(e)}", type="negative")
    
    def _use_suggestion(self, suggestion: str):
        """Use a follow-up suggestion as the next query."""
        self.chat_input.value = suggestion
        self._on_input_change(type('obj', (object,), {'value': suggestion})())
    
    async def _update_sources_panel(self, sources: List[Dict[str, Any]]):
        """Update the sources panel with new sources."""
        self.current_sources = sources
        self.sources_panel.clear()
        
        if not sources:
            with self.sources_panel:
                ui.label('No sources found').classes('text-gray-500 text-center py-4')
            return
        
        with self.sources_panel:
            for i, source in enumerate(sources):
                await self._create_source_card(source, i)
    
    async def _create_source_card(self, source: Dict[str, Any], index: int):
        """Create a source card with functional buttons."""
        doc_name = source.get('doc', 'Unknown Document')
        page_start = source.get('page_start', 1)
        page_end = source.get('page_end', page_start)
        text = source.get('text', '')
        score = source.get('score', 0.0)
        
        with ui.card().classes('w-full mb-2 p-2'):
            # Header with document and page info
            with ui.row().classes('w-full justify-between items-center mb-2'):
                doc_label = f"{doc_name} p.{page_start}"
                if page_end != page_start:
                    doc_label = f"{doc_name} p.{page_start}-{page_end}"
                ui.label(doc_label).classes('font-medium text-sm')
                ui.label(f"sim={score:.2f}").classes('text-xs bg-gray-200 px-1 rounded')
            
            # Source text snippet
            display_text = truncate_text(text, 200)
            ui.label(display_text).classes('text-sm text-gray-700 mb-2')
            
            # Action buttons
            with ui.row().classes('gap-1'):
                ui.button(
                    'Open PDF',
                    icon='open_in_new',
                    on_click=lambda s=source: self._open_source_pdf(s)
                ).props('size=sm')
                ui.button(
                    'Copy Citation',
                    icon='copy',
                    on_click=lambda s=source: self._copy_source_citation(s)
                ).props('size=sm outline')
    
    async def _open_source_pdf(self, source: Dict[str, Any]):
        """Open PDF document at the source page."""
        if not self.current_matter:
            ui.notify("No active matter", type="negative")
            return
        
        doc_name = source.get('doc', '')
        page_start = source.get('page_start', 1)
        
        if not doc_name:
            ui.notify("Document name not available", type="negative")
            return
        
        # Construct path to document
        from pathlib import Path
        doc_path = Path(self.current_matter.get('docs_path', '')) / doc_name
        if not doc_path.exists():
            # Try in docs directory based on matter structure
            matter_root = Path.home() / "LettaClaims" / f"Matter_{self.current_matter.get('slug', '')}"
            doc_path = matter_root / "docs" / doc_name
        
        # Open PDF at page
        success = await open_pdf_at_page(doc_path, page_start)
        if not success:
            ui.notify(f"Could not open {doc_name}", type="negative")
    
    async def _copy_source_citation(self, source: Dict[str, Any]):
        """Copy source citation to clipboard."""
        doc_name = source.get('doc', 'Unknown Document')
        page_start = source.get('page_start', 1)
        page_end = source.get('page_end', page_start)
        
        citation = format_citation(doc_name, page_start, page_end)
        success = await copy_to_clipboard(citation)
        
        if success:
            ui.notify(f"Copied: {citation}", type="positive")
    
    async def _load_chat_history(self):
        """Load and display chat history for current matter."""
        if not self.current_matter:
            self.chat_history = []
            await self._update_chat_display()
            return
        
        try:
            # Load chat history from API
            messages = await self.api_client.get_chat_history(
                self.current_matter['id'],
                limit=50
            )
            self.chat_history = messages
            await self._update_chat_display()
            
            logger.info("Loaded chat history", message_count=len(messages))
            
        except Exception as e:
            logger.error("Failed to load chat history", error=str(e))
            self.chat_history = []
            await self._update_chat_display()
    
    async def _update_chat_display(self):
        """Update the chat messages display."""
        if not self.chat_messages:
            return
        
        # Clear current messages
        self.chat_messages.clear()
        
        with self.chat_messages:
            if not self.chat_history:
                # Show welcome message
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
                    
                ui.markdown(welcome_text).classes('text-gray-600 p-4')
            else:
                # Display chat history
                for message in self.chat_history:
                    await self._add_message_to_display(message)
    
    async def _add_message_to_display(self, message: Dict[str, Any]):
        """Add a single message to the chat display."""
        role = message.get("role", "user")
        content = message.get("content", "")
        timestamp = message.get("timestamp", "")
        sources = message.get("sources", [])
        followups = message.get("followups", [])
        
        # Format timestamp
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = format_timestamp(dt)
        except:
            time_str = ""
        
        if role == "user":
            # User message with smooth animation
            with ui.card().classes('w-full mb-2 bg-blue-50 animate-fade-in hover-lift transition-all'):
                with ui.row().classes('w-full justify-between items-start'):
                    ui.markdown(f"**You:** {content}")
                    if time_str:
                        ui.label(time_str).classes('text-xs text-gray-500')
        
        elif role == "assistant":
            # Assistant message with smooth animation and memory indicator
            card_classes = 'w-full mb-2 bg-green-50 animate-slide-up hover-lift transition-all'
            
            # Add memory glow if memory was used
            used_memory = message.get('used_memory', [])
            if used_memory:
                card_classes += ' memory-active'
            
            with ui.card().classes(card_classes):
                with ui.column().classes('w-full'):
                    with ui.row().classes('w-full justify-between items-start'):
                        ui.markdown(f"**Assistant:**\n\n{content}")
                        if time_str:
                            ui.label(time_str).classes('text-xs text-gray-500')
                    
                    # Add follow-up suggestions if available
                    if followups:
                        ui.label('Suggested follow-ups:').classes('font-semibold mt-2 mb-1')
                        with ui.row().classes('gap-1 flex-wrap'):
                            for suggestion in followups[:3]:  # Limit to 3
                                ui.button(
                                    truncate_text(suggestion, 50),
                                    on_click=lambda s=suggestion: self._use_suggestion(s)
                                ).props('size=sm outline')
            
            # Update sources display if this is the last assistant message
            if message == self.chat_history[-1]:
                await self._update_sources_panel(sources)
    
    async def _scroll_chat_to_bottom(self):
        """Scroll chat messages to the bottom."""
        # Use JavaScript to scroll to bottom
        ui.run_javascript('''
            const chatContainer = document.querySelector('[class*="overflow-y-auto"]');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        ''')
    
    def _toggle_settings_drawer(self):
        """Toggle the settings drawer."""
        if self.settings_drawer:
            self.settings_drawer.toggle()
    
    async def _poll_job_status(self):
        """Poll job status for uploads in progress."""
        if not self.upload_jobs:
            return
        
        jobs_to_remove = []
        jobs_updated = False
        
        for job_key, job_info in self.upload_jobs.items():
            if job_info['status'] in ['completed', 'failed']:
                continue
            
            try:
                status = await self.api_client.get_job_status(job_info['job_id'])
                
                # Check if status actually changed
                old_status = job_info['status']
                old_progress = job_info.get('progress', 0.0)
                
                # Update job info
                job_info['status'] = status['status']
                job_info['progress'] = status.get('progress', 0.0)
                job_info['message'] = status.get('message', 'Processing...')
                
                # Mark as updated if status or progress changed significantly
                if (old_status != job_info['status'] or 
                    abs(old_progress - job_info['progress']) > 0.05):
                    jobs_updated = True
                
                if status['status'] in ['completed', 'failed']:
                    jobs_to_remove.append(job_key)
                    
                    if status['status'] == 'completed':
                        ui.notify(f"✓ Document processed: {job_info['filename']}", type="positive")
                    else:
                        error_detail = status.get('error', 'Unknown error')
                        ui.notify(f"✗ Processing failed: {job_info['filename']} - {error_detail}", type="negative")
                
            except Exception as e:
                logger.error("Failed to get job status", job_id=job_info['job_id'], error=str(e))
                # Mark job as failed if we can't get status
                job_info['status'] = 'failed'
                job_info['message'] = 'Status check failed'
                jobs_updated = True
        
        # Remove completed jobs after a delay
        for job_key in jobs_to_remove:
            del self.upload_jobs[job_key]
        
        # Refresh document list if any jobs completed or updated significantly
        if jobs_to_remove or jobs_updated:
            await self._refresh_document_list()
            
        # If jobs are still running, show overall progress in browser title
        if self.upload_jobs:
            active_count = len([j for j in self.upload_jobs.values() if j['status'] in ['queued', 'running']])
            if active_count > 0:
                ui.run_javascript(f'document.title = "Letta Claims ({active_count} processing...)"')
        else:
            ui.run_javascript('document.title = "Letta Construction Claim Assistant"')
    
    async def _check_provider_consent_status(self):
        """Check consent status for external providers."""
        try:
            # Check if Gemini is configured and needs consent
            response = await self.api_client.get("api/consent/gemini")
            if response and response.get("requires_consent") and not response.get("consent_granted"):
                # Show consent reminder in settings
                ui.notify(
                    "External LLM consent required. Please review privacy settings.",
                    type="info",
                    position="top-right",
                    timeout=5000
                )
        except Exception as e:
            logger.debug("Could not check consent status", error=str(e))
    
    async def _show_consent_dialog(self, provider: str, provider_info: dict):
        """Show consent dialog for external provider."""
        with ui.dialog() as dialog, ui.card().classes('w-96 p-6'):
            ui.label(f'Privacy Consent Required - {provider.title()}').classes('text-xl font-bold mb-4')
            
            # Data usage notice
            notice = provider_info.get("data_usage_notice", "")
            ui.label(notice).classes('text-sm mb-4 text-gray-600')
            
            ui.separator()
            
            ui.label('By continuing, you consent to:').classes('font-semibold mt-4 mb-2')
            with ui.column().classes('ml-4'):
                ui.label('• Sending your questions to external servers').classes('text-sm')
                ui.label('• Sending document context for better answers').classes('text-sm')
                ui.label('• Data processing according to provider\'s privacy policy').classes('text-sm')
            
            ui.separator().classes('my-4')
            
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button(
                    'Decline',
                    color='negative',
                    on_click=lambda: self._handle_consent_response(dialog, provider, False)
                ).classes('flex-1')
                ui.button(
                    'Accept and Continue',
                    color='positive',
                    on_click=lambda: self._handle_consent_response(dialog, provider, True)
                ).classes('flex-1')
        
        dialog.open()
        return dialog
    
    async def _handle_consent_response(self, dialog, provider: str, consent_granted: bool):
        """Handle user consent response."""
        try:
            if consent_granted:
                # Grant consent via API
                response = await self.api_client.post("api/consent", {
                    "provider": provider,
                    "consent_granted": True
                })
                
                if response and response.get("success"):
                    ui.notify(f"Consent granted for {provider}. You can now use external LLM services.", type="positive")
                    # Refresh provider settings
                    await self._refresh_provider_settings()
                else:
                    ui.notify("Failed to save consent preferences.", type="negative")
            else:
                # Deny consent
                await self.api_client.post("api/consent", {
                    "provider": provider,
                    "consent_granted": False
                })
                ui.notify(f"Consent declined. {provider} will not be available.", type="info")
        
        except Exception as e:
            logger.error("Failed to handle consent response", error=str(e))
            ui.notify("Failed to save consent preferences.", type="negative")
        
        dialog.close()
    
    async def _refresh_provider_settings(self):
        """Refresh provider settings display after consent changes."""
        try:
            # Get updated provider information
            response = await self.api_client.get("api/providers")
            if response and response.get("success"):
                # Update provider selector options
                providers = response.get("providers", {})
                available_providers = {}
                
                for provider_key, provider_info in providers.items():
                    provider_name = provider_key.split('_')[0]
                    consent_info = provider_info.get("consent", {})
                    
                    # Only show providers that don't require consent or have consent granted
                    if not consent_info.get("requires_consent") or consent_info.get("consent_granted"):
                        display_name = f"{provider_name.title()} ({provider_info.get('model', 'Unknown')})"
                        available_providers[provider_key] = display_name
                
                # Update the selector
                if hasattr(self, 'provider_select'):
                    self.provider_select.options = available_providers
                    if not self.provider_select.value or self.provider_select.value not in available_providers:
                        # Set to first available provider
                        if available_providers:
                            self.provider_select.value = list(available_providers.keys())[0]
        
        except Exception as e:
            logger.error("Failed to refresh provider settings", error=str(e))
    
    @response_cache.cached
    @measure_performance
    async def _refresh_memory_stats(self):
        """Refresh memory statistics for current matter with caching."""
        if not self.current_matter:
            return
        
        try:
            # Get memory stats from API (cached)
            stats = await self.api_client.get(f"api/matters/{self.current_matter['id']}/memory/stats")
            
            # Update dashboard
            if stats and self.memory_dashboard:
                await self.memory_dashboard.update_stats(stats)
            
            logger.debug("Memory stats refreshed", stats=stats)
            
        except Exception as e:
            logger.error("Failed to refresh memory stats", error=str(e))
            ErrorMessageHandler.show_error(
                'memory_unavailable',
                additional_info=str(e)
            )
    
    @response_cache.cached
    @measure_performance
    async def _update_agent_health(self):
        """Update agent health indicator with caching."""
        try:
            # Get health status from API (cached)
            health = await self.api_client.get("api/letta/health")
            
            # Update health indicator
            if health and self.agent_health:
                self.agent_health.update_health(health)
            
            logger.debug("Agent health updated", health=health)
            
        except Exception as e:
            logger.error("Failed to update agent health", error=str(e))
            # Show degraded status on error
            if self.agent_health:
                self.agent_health.update_health({"status": "error"})
    
    async def _test_provider_with_consent(self, provider_type: str, generation_model: str, api_key: str = None):
        """Test provider with consent checking."""
        try:
            # First check if provider requires consent
            if provider_type.lower() != "ollama":
                consent_response = await self.api_client.get(f"api/consent/{provider_type.lower()}")
                
                if (consent_response and 
                    consent_response.get("requires_consent") and 
                    not consent_response.get("consent_granted")):
                    
                    # Show consent dialog
                    await self._show_consent_dialog(provider_type.lower(), consent_response)
                    return {"success": False, "message": "Consent required - please respond to the dialog"}
            
            # Test provider connectivity
            test_data = {
                "provider_type": provider_type,
                "generation_model": generation_model,
                "test_only": True
            }
            
            if api_key:
                test_data["api_key"] = api_key
            
            response = await self.api_client.post("api/providers/register", test_data)
            return response
            
        except Exception as e:
            logger.error("Provider test failed", error=str(e))
            return {"success": False, "message": f"Test failed: {str(e)}"}


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
    
    # Find an available port for UI
    import socket
    def get_free_port(start=8080):
        for port in range(start, start + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return port
                except OSError:
                    continue
        return start
    
    ui_port = get_free_port()
    
    # Check if native mode dependencies are available
    native_mode_available = False
    try:
        import gi
        native_mode_available = True
        logger.info("GTK available for native mode")
    except ImportError:
        try:
            import qtpy
            native_mode_available = True
            logger.info("Qt available for native mode")
        except ImportError:
            logger.info("No GUI libraries found, will use browser mode")
    
    # Run the application
    if native_mode_available:
        try:
            # Try native desktop mode
            ui.run(
                native=True,
                host='127.0.0.1',
                port=ui_port,
                window_size=(1400, 900),
                title="Letta Construction Claim Assistant",
                reload=False,
                show=True,
                on_air=False
            )
        except Exception as e:
            logger.warning(f"Native mode failed despite libraries, falling back to browser: {e}")
            # Fallback to browser mode
            ui.run(
                native=False,
                host='127.0.0.1',
                port=ui_port,
                title="Letta Construction Claim Assistant",
                reload=False
            )
    else:
        # Run directly in browser mode if no GUI libraries
        logger.info(f"Starting in browser mode on http://127.0.0.1:{ui_port}")
        ui.run(
            native=False,
            host='127.0.0.1',
            port=ui_port,
            title="Letta Construction Claim Assistant",
            reload=False,
            show=True  # Auto-open browser
        )


def main():
    """Legacy main entry point (for backward compatibility)."""
    create_ui_app()


if __name__ in {"__main__", "__mp_main__"}:
    main()