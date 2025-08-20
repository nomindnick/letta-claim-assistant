"""
Memory Editor dialog component for creating and editing memory items.

Provides a form interface for creating new memory items or editing
existing ones with validation and type selection.
"""

from nicegui import ui
from typing import Optional, Dict, Any, Callable
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from app.logging_conf import get_logger

logger = get_logger(__name__)


class MemoryEditor:
    """Dialog component for creating or editing memory items."""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
        self.dialog = None
        self.on_save_callback = None
        
        # Form elements
        self.text_input = None
        self.type_select = None
        self.metadata_inputs = {}
        self.save_button = None
        self.cancel_button = None
        
        # State
        self.current_matter_id = None
        self.current_item = None
        self.is_edit_mode = False
        
        # Memory types configuration
        self.memory_types = [
            {"value": "Entity", "label": "Entity", "icon": "person", "description": "Person, organization, or thing"},
            {"value": "Event", "label": "Event", "icon": "event", "description": "Something that happened"},
            {"value": "Issue", "label": "Issue", "icon": "warning", "description": "Problem or concern"},
            {"value": "Fact", "label": "Fact", "icon": "fact_check", "description": "Statement of truth"},
            {"value": "Interaction", "label": "Interaction", "icon": "chat", "description": "User interaction"},
            {"value": "Raw", "label": "Raw", "icon": "description", "description": "Unstructured text"}
        ]
    
    async def show_create(self, matter_id: str, on_save: Optional[Callable] = None):
        """Show the editor in create mode."""
        self.current_matter_id = matter_id
        self.current_item = None
        self.is_edit_mode = False
        self.on_save_callback = on_save
        
        await self._show_dialog()
    
    async def show_edit(self, matter_id: str, item: Dict[str, Any], on_save: Optional[Callable] = None):
        """Show the editor in edit mode with existing item data."""
        self.current_matter_id = matter_id
        self.current_item = item
        self.is_edit_mode = True
        self.on_save_callback = on_save
        
        await self._show_dialog()
    
    async def _show_dialog(self):
        """Display the editor dialog."""
        if not self.api_client:
            ui.notify("API client not configured", type="warning")
            return
        
        # Create dialog
        with ui.dialog() as self.dialog:
            with ui.card().classes('w-full max-w-2xl'):
                # Header
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    title = 'Edit Memory Item' if self.is_edit_mode else 'Create Memory Item'
                    ui.label(title).classes('text-xl font-bold')
                    ui.button(icon='close', on_click=self.dialog.close).props('flat round')
                
                # Form container
                with ui.column().classes('w-full gap-4'):
                    # Type selector
                    with ui.column().classes('w-full'):
                        ui.label('Memory Type').classes('text-sm font-medium text-gray-700')
                        
                        # Get current type or default
                        current_type = "Raw"
                        if self.is_edit_mode and self.current_item:
                            current_type = self.current_item.get('type', 'Raw')
                        
                        # Create select with icons
                        type_options = {t['value']: t['label'] for t in self.memory_types}
                        self.type_select = ui.select(
                            options=type_options,
                            value=current_type,
                            on_change=self._on_type_change
                        ).props('outlined').classes('w-full')
                        
                        # Type description
                        self.type_description = ui.label('').classes('text-xs text-gray-500 mt-1')
                        self._update_type_description(current_type)
                    
                    # Main text input
                    with ui.column().classes('w-full'):
                        ui.label('Content *').classes('text-sm font-medium text-gray-700')
                        
                        # Pre-fill text if editing
                        initial_text = ""
                        if self.is_edit_mode and self.current_item:
                            # Try to get text from metadata first (for structured items)
                            metadata = self.current_item.get('metadata', {})
                            if metadata and isinstance(metadata, dict):
                                initial_text = metadata.get('label', '') or metadata.get('support_snippet', '')
                            
                            # Fall back to raw text if no metadata
                            if not initial_text:
                                initial_text = self.current_item.get('text', '')
                        
                        self.text_input = ui.textarea(
                            value=initial_text,
                            placeholder='Enter memory content...'
                        ).props('outlined autogrow').classes('w-full')
                        self.text_input.on('input', self._validate_form)
                    
                    # Optional metadata fields (shown for certain types)
                    self.metadata_container = ui.column().classes('w-full gap-3')
                    self._create_metadata_fields(current_type)
                    
                    # Form validation message
                    self.validation_message = ui.label('').classes('text-sm text-red-600 hidden')
                
                # Action buttons
                with ui.row().classes('w-full justify-end gap-2 mt-4'):
                    self.cancel_button = ui.button(
                        'Cancel',
                        on_click=self.dialog.close
                    ).props('flat')
                    
                    self.save_button = ui.button(
                        'Save',
                        icon='save',
                        on_click=self._save_memory
                    ).props('color=primary')
                
                # Initial validation
                self._validate_form()
        
        # Open dialog
        self.dialog.open()
    
    def _create_metadata_fields(self, memory_type: str):
        """Create optional metadata input fields based on memory type."""
        self.metadata_container.clear()
        self.metadata_inputs = {}
        
        if memory_type in ['Entity', 'Event', 'Issue', 'Fact']:
            with self.metadata_container:
                # Date field for Events
                if memory_type == 'Event':
                    with ui.column().classes('w-full'):
                        ui.label('Date (Optional)').classes('text-sm font-medium text-gray-700')
                        
                        # Pre-fill date if editing
                        initial_date = ""
                        if self.is_edit_mode and self.current_item:
                            metadata = self.current_item.get('metadata', {})
                            if metadata and isinstance(metadata, dict):
                                initial_date = metadata.get('date', '')
                        
                        self.metadata_inputs['date'] = ui.input(
                            value=initial_date,
                            placeholder='YYYY-MM-DD or descriptive date'
                        ).props('outlined').classes('w-full')
                
                # Actors field for Events and Issues
                if memory_type in ['Event', 'Issue']:
                    with ui.column().classes('w-full'):
                        ui.label('Actors/Parties (Optional)').classes('text-sm font-medium text-gray-700')
                        
                        # Pre-fill actors if editing
                        initial_actors = ""
                        if self.is_edit_mode and self.current_item:
                            metadata = self.current_item.get('metadata', {})
                            if metadata and isinstance(metadata, dict):
                                actors = metadata.get('actors', [])
                                if actors and isinstance(actors, list):
                                    initial_actors = ', '.join(actors)
                        
                        self.metadata_inputs['actors'] = ui.input(
                            value=initial_actors,
                            placeholder='Comma-separated list of people/organizations'
                        ).props('outlined').classes('w-full')
                
                # Document references (all structured types)
                with ui.column().classes('w-full'):
                    ui.label('Document References (Optional)').classes('text-sm font-medium text-gray-700')
                    
                    # Pre-fill doc refs if editing
                    initial_refs = ""
                    if self.is_edit_mode and self.current_item:
                        metadata = self.current_item.get('metadata', {})
                        if metadata and isinstance(metadata, dict):
                            doc_refs = metadata.get('doc_refs', [])
                            if doc_refs and isinstance(doc_refs, list):
                                # Format doc refs for display
                                refs_str = []
                                for ref in doc_refs:
                                    if isinstance(ref, dict):
                                        doc = ref.get('doc', '')
                                        pages = ref.get('pages', '')
                                        if doc:
                                            refs_str.append(f"{doc}:{pages}" if pages else doc)
                                initial_refs = ', '.join(refs_str)
                    
                    self.metadata_inputs['doc_refs'] = ui.input(
                        value=initial_refs,
                        placeholder='e.g., Document.pdf:p.5-7, Contract.pdf:p.12'
                    ).props('outlined').classes('w-full')
    
    def _on_type_change(self, e):
        """Handle memory type selection change."""
        selected_type = e.value
        self._update_type_description(selected_type)
        self._create_metadata_fields(selected_type)
        self._validate_form()
    
    def _update_type_description(self, memory_type: str):
        """Update the type description label."""
        for type_info in self.memory_types:
            if type_info['value'] == memory_type:
                self.type_description.text = type_info['description']
                break
    
    def _validate_form(self, e=None):
        """Validate form inputs and update button state."""
        is_valid = True
        error_message = ""
        
        # Check required text field
        if self.text_input:
            text_value = self.text_input.value.strip()
            if not text_value:
                is_valid = False
                error_message = "Content is required"
            elif len(text_value) < 3:
                is_valid = False
                error_message = "Content must be at least 3 characters"
        
        # Update validation message
        if self.validation_message:
            if error_message:
                self.validation_message.text = error_message
                self.validation_message.classes(remove='hidden')
            else:
                self.validation_message.classes(add='hidden')
        
        # Update save button state
        if self.save_button:
            self.save_button.set_enabled(is_valid)
    
    async def _save_memory(self):
        """Save the memory item (create or update)."""
        if not self.api_client or not self.current_matter_id:
            ui.notify("Cannot save: API client or matter not configured", type="negative")
            return
        
        # Get form values
        text = self.text_input.value.strip()
        memory_type = self.type_select.value
        
        # Build metadata if applicable
        metadata = {}
        if memory_type != 'Raw':
            # Add type-specific metadata
            if 'date' in self.metadata_inputs and self.metadata_inputs['date'].value:
                metadata['date'] = self.metadata_inputs['date'].value.strip()
            
            if 'actors' in self.metadata_inputs and self.metadata_inputs['actors'].value:
                actors_str = self.metadata_inputs['actors'].value.strip()
                metadata['actors'] = [a.strip() for a in actors_str.split(',') if a.strip()]
            
            if 'doc_refs' in self.metadata_inputs and self.metadata_inputs['doc_refs'].value:
                refs_str = self.metadata_inputs['doc_refs'].value.strip()
                # Parse document references
                doc_refs = []
                for ref in refs_str.split(','):
                    ref = ref.strip()
                    if ref:
                        if ':' in ref:
                            doc, pages = ref.split(':', 1)
                            doc_refs.append({'doc': doc.strip(), 'pages': pages.strip()})
                        else:
                            doc_refs.append({'doc': ref, 'pages': ''})
                if doc_refs:
                    metadata['doc_refs'] = doc_refs
            
            # Add label (first 100 chars of text)
            metadata['label'] = text[:100] + ('...' if len(text) > 100 else '')
            
            # For structured types, we might want to store the full text as support_snippet
            if len(text) > 100:
                metadata['support_snippet'] = text
        
        # Show loading state
        self.save_button.props('loading')
        self.save_button.set_enabled(False)
        self.cancel_button.set_enabled(False)
        
        try:
            if self.is_edit_mode:
                # Update existing item
                item_id = self.current_item.get('id')
                logger.info(f"Updating memory item {item_id}")
                
                # For updates, we need to format the text properly
                if memory_type != 'Raw' and metadata:
                    # Create a KnowledgeItem-like structure
                    knowledge_item = {
                        'type': memory_type,
                        **metadata
                    }
                    new_text = json.dumps(knowledge_item)
                else:
                    new_text = text
                
                result = await self.api_client.update_memory_item(
                    matter_id=self.current_matter_id,
                    item_id=item_id,
                    new_text=new_text,
                    preserve_type=False  # We're providing the full new structure
                )
                
                if result.get('success'):
                    ui.notify(f"Memory item updated successfully", type="positive")
                else:
                    ui.notify(f"Failed to update: {result.get('message', 'Unknown error')}", type="negative")
                    return
            else:
                # Create new item
                logger.info(f"Creating new memory item of type {memory_type}")
                
                result = await self.api_client.create_memory_item(
                    matter_id=self.current_matter_id,
                    text=text,
                    type=memory_type,
                    metadata=metadata if memory_type != 'Raw' else None
                )
                
                if result.get('success'):
                    ui.notify(f"Memory item created successfully", type="positive")
                else:
                    ui.notify(f"Failed to create: {result.get('message', 'Unknown error')}", type="negative")
                    return
            
            # Close dialog and trigger callback
            self.dialog.close()
            
            if self.on_save_callback:
                await self.on_save_callback()
            
        except Exception as e:
            logger.error(f"Failed to save memory item: {e}")
            ui.notify(f"Error saving memory: {str(e)}", type="negative")
        finally:
            # Reset button states
            self.save_button.props(remove='loading')
            self.save_button.set_enabled(True)
            self.cancel_button.set_enabled(True)