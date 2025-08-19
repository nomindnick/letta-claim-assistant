"""
User-friendly error messages and handling for the UI.

Provides clear, actionable error messages with recovery suggestions.
"""

from typing import Dict, Optional, Callable
from nicegui import ui


class ErrorMessageHandler:
    """Handler for displaying user-friendly error messages."""
    
    # Error message templates with user-friendly text
    ERROR_TEMPLATES = {
        # Connection errors
        "connection_failed": {
            "title": "Connection Problem",
            "message": "Unable to connect to the server",
            "suggestion": "Please check your internet connection and try again",
            "icon": "wifi_off",
            "type": "warning"
        },
        "server_unavailable": {
            "title": "Server Unavailable",
            "message": "The backend server is not responding",
            "suggestion": "The service may be temporarily down. Please try again in a few moments",
            "icon": "cloud_off",
            "type": "warning"
        },
        
        # Matter errors
        "matter_not_found": {
            "title": "Matter Not Found",
            "message": "The selected matter could not be found",
            "suggestion": "Please select a different matter or create a new one",
            "icon": "folder_off",
            "type": "error"
        },
        "matter_creation_failed": {
            "title": "Could Not Create Matter",
            "message": "Failed to create the new matter",
            "suggestion": "Please check the matter name and try again",
            "icon": "create_new_folder",
            "type": "error"
        },
        
        # Document errors
        "upload_failed": {
            "title": "Upload Failed",
            "message": "Could not upload the document",
            "suggestion": "Please check the file is a valid PDF and try again",
            "icon": "upload_file",
            "type": "error"
        },
        "processing_failed": {
            "title": "Processing Error",
            "message": "Failed to process the document",
            "suggestion": "The document may be corrupted or too large. Try a different file",
            "icon": "description",
            "type": "error"
        },
        "ocr_failed": {
            "title": "OCR Failed",
            "message": "Could not extract text from the document",
            "suggestion": "The document may be a scanned image. OCR processing will be attempted",
            "icon": "text_fields",
            "type": "warning"
        },
        
        # Chat/RAG errors
        "chat_failed": {
            "title": "Message Failed",
            "message": "Could not process your message",
            "suggestion": "Please try rephrasing your question or check your connection",
            "icon": "chat_bubble",
            "type": "error"
        },
        "no_sources": {
            "title": "No Sources Found",
            "message": "Could not find relevant information in the documents",
            "suggestion": "Try uploading more documents or asking a different question",
            "icon": "search_off",
            "type": "info"
        },
        
        # Memory/Letta errors
        "memory_unavailable": {
            "title": "Memory System Offline",
            "message": "The agent memory system is currently unavailable",
            "suggestion": "You can still use the system, but responses won't be enhanced with memory",
            "icon": "psychology",
            "type": "warning"
        },
        "memory_sync_failed": {
            "title": "Memory Sync Failed",
            "message": "Could not synchronize with agent memory",
            "suggestion": "Memory will be synced automatically when connection is restored",
            "icon": "sync_problem",
            "type": "warning"
        },
        
        # Provider errors
        "provider_unavailable": {
            "title": "AI Provider Unavailable",
            "message": "The AI service is not responding",
            "suggestion": "Switching to backup provider. Your experience may vary",
            "icon": "smart_toy",
            "type": "warning"
        },
        "api_key_invalid": {
            "title": "Invalid API Key",
            "message": "The provided API key is not valid",
            "suggestion": "Please check your API key in settings and try again",
            "icon": "key_off",
            "type": "error"
        },
        "consent_required": {
            "title": "Consent Required",
            "message": "External API usage requires your consent",
            "suggestion": "Please review and accept the privacy terms to continue",
            "icon": "privacy_tip",
            "type": "info"
        },
        
        # Generic errors
        "unknown_error": {
            "title": "Something Went Wrong",
            "message": "An unexpected error occurred",
            "suggestion": "Please try again. If the problem persists, contact support",
            "icon": "error_outline",
            "type": "error"
        },
        "timeout": {
            "title": "Request Timeout",
            "message": "The operation took too long to complete",
            "suggestion": "This might be due to heavy processing. Please try again",
            "icon": "schedule",
            "type": "warning"
        },
        "rate_limited": {
            "title": "Too Many Requests",
            "message": "You've made too many requests too quickly",
            "suggestion": "Please wait a moment before trying again",
            "icon": "speed",
            "type": "warning"
        }
    }
    
    @classmethod
    def get_error_info(cls, error_code: str) -> Dict[str, str]:
        """Get error information for a given error code."""
        return cls.ERROR_TEMPLATES.get(error_code, cls.ERROR_TEMPLATES["unknown_error"])
    
    @classmethod
    def show_error(cls, 
                   error_code: str,
                   additional_info: Optional[str] = None,
                   retry_callback: Optional[Callable] = None,
                   dismiss_callback: Optional[Callable] = None):
        """
        Show an error notification with user-friendly message.
        
        Args:
            error_code: The error code to look up
            additional_info: Additional context to append to the message
            retry_callback: Optional callback for retry action
            dismiss_callback: Optional callback for dismiss action
        """
        error_info = cls.get_error_info(error_code)
        
        # Build the full message
        message = f"**{error_info['title']}**\n\n{error_info['message']}"
        if additional_info:
            message += f"\n\nDetails: {additional_info}"
        message += f"\n\n💡 {error_info['suggestion']}"
        
        # Determine notification type
        notification_type = error_info['type']
        if notification_type == 'error':
            color = 'negative'
        elif notification_type == 'warning':
            color = 'warning'
        else:
            color = 'info'
        
        # Show persistent notification for errors
        timeout = 0 if notification_type == 'error' else 5000
        
        # Create notification with actions
        notification = ui.notify(
            message,
            type=color,
            position='top',
            timeout=timeout,
            close_button=True,
            html=True
        )
        
        # Add retry button if callback provided
        if retry_callback and notification_type == 'error':
            with notification:
                ui.button('Retry', on_click=retry_callback).props('flat')
    
    @classmethod
    def parse_backend_error(cls, error: Exception) -> str:
        """
        Parse backend error and return appropriate error code.
        
        Args:
            error: The exception from backend
            
        Returns:
            Appropriate error code for the error
        """
        error_str = str(error).lower()
        
        # Connection errors
        if 'connection' in error_str or 'network' in error_str:
            return 'connection_failed'
        elif 'server' in error_str and 'unavailable' in error_str:
            return 'server_unavailable'
        
        # Matter errors
        elif 'matter' in error_str and 'not found' in error_str:
            return 'matter_not_found'
        elif 'matter' in error_str and 'create' in error_str:
            return 'matter_creation_failed'
        
        # Document errors
        elif 'upload' in error_str:
            return 'upload_failed'
        elif 'process' in error_str and 'document' in error_str:
            return 'processing_failed'
        elif 'ocr' in error_str:
            return 'ocr_failed'
        
        # Memory errors
        elif 'memory' in error_str or 'letta' in error_str:
            if 'sync' in error_str:
                return 'memory_sync_failed'
            else:
                return 'memory_unavailable'
        
        # Provider errors
        elif 'provider' in error_str or 'llm' in error_str:
            return 'provider_unavailable'
        elif 'api' in error_str and 'key' in error_str:
            return 'api_key_invalid'
        elif 'consent' in error_str:
            return 'consent_required'
        
        # Timeout and rate limiting
        elif 'timeout' in error_str:
            return 'timeout'
        elif 'rate' in error_str and 'limit' in error_str:
            return 'rate_limited'
        
        # Default
        else:
            return 'unknown_error'


class ErrorRecoveryDialog:
    """Dialog for error recovery options."""
    
    @staticmethod
    async def show_recovery_options(
        error_code: str,
        retry_callback: Optional[Callable] = None,
        skip_callback: Optional[Callable] = None,
        details: Optional[str] = None
    ):
        """
        Show a dialog with recovery options for an error.
        
        Args:
            error_code: The error code
            retry_callback: Callback to retry the operation
            skip_callback: Callback to skip the operation
            details: Additional error details
        """
        error_info = ErrorMessageHandler.get_error_info(error_code)
        
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            # Header with icon
            with ui.row().classes('items-center gap-3 mb-4'):
                ui.icon(error_info['icon']).classes('text-4xl text-red-500')
                ui.label(error_info['title']).classes('text-xl font-bold')
            
            # Error message
            ui.label(error_info['message']).classes('mb-2')
            
            # Suggestion
            with ui.row().classes('items-start gap-2 mb-4 p-3 bg-blue-50 rounded'):
                ui.icon('lightbulb').classes('text-blue-500')
                ui.label(error_info['suggestion']).classes('text-sm')
            
            # Details (if provided)
            if details:
                with ui.expansion('Technical Details', icon='code').classes('w-full mb-4'):
                    ui.label(details).classes('text-xs font-mono text-gray-600')
            
            # Action buttons
            with ui.row().classes('w-full justify-end gap-2'):
                if skip_callback:
                    ui.button(
                        'Skip',
                        on_click=lambda: [skip_callback(), dialog.close()]
                    ).props('flat')
                
                if retry_callback:
                    ui.button(
                        'Retry',
                        icon='refresh',
                        on_click=lambda: [retry_callback(), dialog.close()]
                    ).props('color=primary')
                else:
                    ui.button(
                        'OK',
                        on_click=dialog.close
                    ).props('color=primary')
        
        dialog.open()


def handle_ui_error(operation: str):
    """
    Decorator for handling errors in UI operations.
    
    Usage:
        @handle_ui_error("upload document")
        async def upload_file(self, file):
            # ... operation code ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Parse error and show user-friendly message
                error_code = ErrorMessageHandler.parse_backend_error(e)
                ErrorMessageHandler.show_error(
                    error_code,
                    additional_info=f"During: {operation}",
                    retry_callback=lambda: asyncio.create_task(func(*args, **kwargs))
                )
                # Log the actual error
                import logging
                logging.error(f"Error in {operation}: {str(e)}", exc_info=True)
        return wrapper
    return decorator