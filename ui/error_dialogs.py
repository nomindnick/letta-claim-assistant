"""
Enhanced error handling UI components for NiceGUI.

Provides user-friendly error dialogs, recovery actions, and error status
displays with actionable recovery options and detailed error information.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import asyncio
from pathlib import Path

from nicegui import ui
from nicegui.events import ValueChangeEventArguments

import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.error_handler import (
    BaseApplicationError, ErrorSeverity, RecoveryStrategy, RecoveryAction,
    error_handler
)
from app.resource_monitor import resource_monitor, ResourceStatus
from app.degradation import degradation_manager


class ErrorNotificationManager:
    """
    Manages error notifications and dialogs in the UI.
    """
    
    def __init__(self):
        self.active_notifications: Dict[str, ui.notification] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history = 50
        
    def show_error(
        self,
        error: BaseApplicationError,
        context: Optional[Dict[str, Any]] = None
    ) -> ui.notification:
        """
        Show error notification with appropriate styling and actions.
        
        Args:
            error: Application error to display
            context: Additional context for the error
            
        Returns:
            Notification object
        """
        # Determine notification type and styling
        if error.severity == ErrorSeverity.CRITICAL:
            notification_type = "negative"
            icon = "error"
            timeout = None  # Don't auto-dismiss critical errors
        elif error.severity == ErrorSeverity.ERROR:
            notification_type = "negative"
            icon = "warning"
            timeout = 10000
        elif error.severity == ErrorSeverity.WARNING:
            notification_type = "warning"
            icon = "warning"
            timeout = 8000
        else:
            notification_type = "info"
            icon = "info"
            timeout = 5000
        
        # Create notification
        with ui.notification(
            message=error.user_message,
            type=notification_type,
            timeout=timeout,
            close_button=True,
            multi_line=True
        ) as notification:
            
            # Add error details section if available
            if error.suggestion:
                ui.html(f'<div style="margin-top: 8px; font-size: 0.9em; opacity: 0.9;">{error.suggestion}</div>')
            
            # Add recovery actions
            if error.recovery_actions:
                with ui.row().classes("gap-2 mt-2"):
                    for action in error.recovery_actions[:3]:  # Limit to 3 actions
                        self._create_action_button(action, error, notification)
            
            # Add "Show Details" button for technical users
            if error.error_code or error.cause:
                ui.button(
                    "Show Details",
                    on_click=lambda: self._show_error_details(error)
                ).props("flat size=sm").classes("mt-2")
        
        # Store in active notifications
        error_key = f"{error.error_code or 'unknown'}_{datetime.utcnow().timestamp()}"
        self.active_notifications[error_key] = notification
        
        # Add to history
        self._add_to_history(error, context)
        
        return notification
    
    def _create_action_button(
        self,
        action: RecoveryAction,
        error: BaseApplicationError,
        notification: ui.notification
    ):
        """Create action button for recovery action."""
        button_props = "size=sm"
        if action.is_primary:
            button_props += " color=primary"
        else:
            button_props += " outline"
        
        async def handle_action():
            try:
                if action.callback:
                    await action.callback()
                
                # Close notification after successful action
                notification.dismiss()
                
                # Show success feedback
                ui.notify(
                    f"Action '{action.label}' completed",
                    type="positive",
                    timeout=3000
                )
                
            except Exception as e:
                ui.notify(
                    f"Action failed: {str(e)}",
                    type="negative",
                    timeout=5000
                )
        
        ui.button(
            action.label,
            on_click=handle_action
        ).props(button_props).tooltip(action.description)
    
    def _show_error_details(self, error: BaseApplicationError):
        """Show detailed error information in a dialog."""
        with ui.dialog() as dialog, ui.card().classes("w-96 max-w-full"):
            ui.label("Error Details").classes("text-h6 mb-4")
            
            # Error code and severity
            with ui.row().classes("w-full justify-between mb-2"):
                ui.badge(error.error_code or "UNKNOWN", color="red")
                ui.badge(error.severity.value.upper(), color="orange")
            
            # Technical message
            ui.label("Technical Details:").classes("font-bold mt-4")
            ui.textarea(
                value=error.message,
                readonly=True
            ).classes("w-full").style("min-height: 100px")
            
            # Context information
            if error.context and any([
                error.context.matter_id,
                error.context.file_path,
                error.context.operation,
                error.context.provider
            ]):
                ui.label("Context:").classes("font-bold mt-4")
                context_info = []
                if error.context.matter_id:
                    context_info.append(f"Matter: {error.context.matter_name or error.context.matter_id}")
                if error.context.operation:
                    context_info.append(f"Operation: {error.context.operation}")
                if error.context.file_path:
                    context_info.append(f"File: {Path(error.context.file_path).name}")
                if error.context.provider:
                    context_info.append(f"Provider: {error.context.provider}")
                
                for info in context_info:
                    ui.label(f"• {info}").classes("ml-4")
            
            # Underlying cause
            if error.cause:
                ui.label("Underlying Cause:").classes("font-bold mt-4")
                ui.label(f"{type(error.cause).__name__}: {str(error.cause)}")
            
            # Timestamp
            ui.label("Timestamp:").classes("font-bold mt-4")
            ui.label(error.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
            
            # Close button
            with ui.row().classes("w-full justify-end mt-4"):
                ui.button("Close", on_click=dialog.close).props("flat")
        
        dialog.open()
    
    def _add_to_history(self, error: BaseApplicationError, context: Optional[Dict[str, Any]]):
        """Add error to history."""
        self.error_history.append({
            "timestamp": error.timestamp,
            "severity": error.severity.value,
            "error_code": error.error_code,
            "user_message": error.user_message,
            "technical_message": error.message,
            "context": context or {}
        })
        
        # Trim history
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def show_error_history(self):
        """Show error history in a dialog."""
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
            ui.label("Error History").classes("text-h6 mb-4")
            
            if not self.error_history:
                ui.label("No errors recorded").classes("text-center py-8 text-grey-6")
            else:
                # Create table of errors
                columns = [
                    {"name": "timestamp", "label": "Time", "field": "timestamp", "align": "left"},
                    {"name": "severity", "label": "Severity", "field": "severity", "align": "left"},
                    {"name": "error_code", "label": "Code", "field": "error_code", "align": "left"},
                    {"name": "message", "label": "Message", "field": "user_message", "align": "left"}
                ]
                
                # Format timestamp for display
                formatted_history = []
                for error in reversed(self.error_history[-20:]):  # Show last 20 errors
                    formatted_error = error.copy()
                    formatted_error["timestamp"] = error["timestamp"].strftime("%H:%M:%S")
                    formatted_history.append(formatted_error)
                
                ui.table(
                    columns=columns,
                    rows=formatted_history,
                    row_key="timestamp"
                ).classes("w-full")
            
            with ui.row().classes("w-full justify-between mt-4"):
                ui.button(
                    "Clear History",
                    on_click=lambda: self._clear_history(dialog)
                ).props("flat color=red")
                ui.button("Close", on_click=dialog.close).props("flat")
        
        dialog.open()
    
    def _clear_history(self, dialog):
        """Clear error history."""
        self.error_history.clear()
        ui.notify("Error history cleared", type="info")
        dialog.close()
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for the current session."""
        if not self.error_history:
            return {"total_errors": 0}
        
        severity_counts = {}
        error_code_counts = {}
        
        for error in self.error_history:
            severity = error["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            error_code = error["error_code"] or "unknown"
            error_code_counts[error_code] = error_code_counts.get(error_code, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "severity_counts": severity_counts,
            "error_code_counts": error_code_counts,
            "most_recent": self.error_history[-1] if self.error_history else None
        }


class SystemStatusWidget:
    """
    Widget displaying system status and resource health.
    """
    
    def __init__(self):
        self.status_container = None
        self.auto_update_task = None
        self.update_interval = 30  # seconds
    
    def create_widget(self) -> ui.element:
        """Create the system status widget."""
        with ui.card().classes("w-full") as container:
            self.status_container = container
            
            ui.label("System Status").classes("text-h6 mb-2")
            
            # Overall status indicator
            with ui.row().classes("items-center gap-2 mb-4"):
                self.overall_status_icon = ui.icon("circle").classes("text-lg")
                self.overall_status_label = ui.label("Checking...")
            
            # Individual service status
            with ui.column().classes("gap-2 w-full"):
                self.service_status_elements = {}
                
                services = [
                    ("disk", "Disk Space", "storage"),
                    ("memory", "Memory", "memory"),
                    ("network", "Network", "wifi"),
                    ("ollama", "Ollama", "smart_toy"),
                    ("chromadb", "ChromaDB", "database")
                ]
                
                for service_id, service_name, icon_name in services:
                    with ui.row().classes("items-center gap-2 w-full"):
                        service_icon = ui.icon(icon_name).classes("text-lg")
                        service_label = ui.label(service_name).classes("flex-grow")
                        status_badge = ui.badge("Checking...", color="grey")
                        
                        self.service_status_elements[service_id] = {
                            "icon": service_icon,
                            "label": service_label,
                            "badge": status_badge
                        }
            
            # Action buttons
            with ui.row().classes("gap-2 mt-4"):
                ui.button(
                    "Refresh",
                    icon="refresh",
                    on_click=self.update_status
                ).props("flat size=sm")
                
                ui.button(
                    "Details",
                    icon="info",
                    on_click=self.show_detailed_status
                ).props("flat size=sm")
        
        # Start auto-updates
        self.start_auto_update()
        
        # Initial status check
        asyncio.create_task(self.update_status())
        
        return container
    
    async def update_status(self):
        """Update system status display."""
        try:
            # Get comprehensive status
            status = await resource_monitor.get_comprehensive_status()
            
            # Update overall status
            overall_status = status["overall_status"]
            if overall_status == "healthy":
                self.overall_status_icon.classes("text-green-500")
                self.overall_status_label.text = "All systems operational"
            elif overall_status == "warning":
                self.overall_status_icon.classes("text-yellow-500")
                self.overall_status_label.text = "Some issues detected"
            else:
                self.overall_status_icon.classes("text-red-500")
                self.overall_status_label.text = "Critical issues detected"
            
            # Update individual services
            self._update_service_status("disk", status["disk"])
            self._update_service_status("memory", status["memory"])
            self._update_service_status("network", status["network"])
            self._update_service_status("ollama", status["services"]["ollama"])
            self._update_service_status("chromadb", status["services"]["chromadb"])
            
        except Exception as e:
            # Handle status check failure
            self.overall_status_icon.classes("text-grey-500")
            self.overall_status_label.text = "Status check failed"
            
            for elements in self.service_status_elements.values():
                elements["badge"].text = "Error"
                elements["badge"].color = "red"
    
    def _update_service_status(self, service_id: str, status_data: Dict[str, Any]):
        """Update individual service status."""
        if service_id not in self.service_status_elements:
            return
        
        elements = self.service_status_elements[service_id]
        status = status_data.get("status", "unavailable")
        
        if status == "healthy":
            badge_color = "green"
            badge_text = "OK"
        elif status == "warning":
            badge_color = "yellow"
            badge_text = "Warning"
        elif status == "critical":
            badge_color = "red"
            badge_text = "Critical"
        else:
            badge_color = "grey"
            badge_text = "Unknown"
        
        elements["badge"].text = badge_text
        elements["badge"].color = badge_color
        
        # Add tooltip with details for disk and memory
        if service_id == "disk" and "percent_used" in status_data:
            percent = status_data["percent_used"] * 100
            free_gb = status_data["free"] / (1024**3)
            elements["label"].tooltip(f"Used: {percent:.1f}%, Free: {free_gb:.1f}GB")
        elif service_id == "memory" and "percent_used" in status_data:
            percent = status_data["percent_used"] * 100
            free_gb = status_data["free"] / (1024**3)
            elements["label"].tooltip(f"Used: {percent:.1f}%, Free: {free_gb:.1f}GB")
    
    def start_auto_update(self):
        """Start automatic status updates."""
        if self.auto_update_task:
            return
        
        async def update_loop():
            while True:
                await asyncio.sleep(self.update_interval)
                await self.update_status()
        
        self.auto_update_task = asyncio.create_task(update_loop())
    
    def stop_auto_update(self):
        """Stop automatic status updates."""
        if self.auto_update_task:
            self.auto_update_task.cancel()
            self.auto_update_task = None
    
    def show_detailed_status(self):
        """Show detailed system status in a dialog."""
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
            ui.label("Detailed System Status").classes("text-h6 mb-4")
            
            async def load_detailed_status():
                try:
                    # Get comprehensive status
                    status = await resource_monitor.get_comprehensive_status()
                    
                    # Clear existing content
                    dialog.clear()
                    
                    with ui.card().classes("w-full max-w-4xl"):
                        ui.label("Detailed System Status").classes("text-h6 mb-4")
                        
                        # Resource status
                        ui.label("Resources").classes("text-h6 mt-4 mb-2")
                        
                        # Disk status
                        disk = status["disk"]
                        with ui.row().classes("items-center gap-4"):
                            ui.icon("storage").classes("text-2xl")
                            ui.label(f"Disk Space: {disk['percent_used']*100:.1f}% used")
                            ui.progress(disk['percent_used']).classes("flex-grow")
                        ui.label(f"Free: {disk['free']/(1024**3):.1f}GB of {disk['total']/(1024**3):.1f}GB")
                        
                        # Memory status
                        memory = status["memory"]
                        with ui.row().classes("items-center gap-4 mt-2"):
                            ui.icon("memory").classes("text-2xl")
                            ui.label(f"Memory: {memory['percent_used']*100:.1f}% used")
                            ui.progress(memory['percent_used']).classes("flex-grow")
                        ui.label(f"Free: {memory['free']/(1024**3):.1f}GB of {memory['total']/(1024**3):.1f}GB")
                        
                        # Services status
                        ui.label("Services").classes("text-h6 mt-4 mb-2")
                        
                        for service_name, service_data in status["services"].items():
                            with ui.row().classes("items-center gap-4"):
                                status_icon = "check_circle" if service_data["status"] == "healthy" else "error"
                                status_color = "green" if service_data["status"] == "healthy" else "red"
                                
                                ui.icon(status_icon).classes(f"text-2xl text-{status_color}-500")
                                ui.label(f"{service_name.title()}: {service_data['status']}")
                                
                                if service_data.get("response_time_ms"):
                                    ui.label(f"({service_data['response_time_ms']:.0f}ms)")
                        
                        # Degradation status
                        degradation_status = degradation_manager.get_degradation_status()
                        if degradation_status["global_level"] != "full":
                            ui.label("Degradation Status").classes("text-h6 mt-4 mb-2")
                            ui.label(f"Operating in {degradation_status['global_level']} mode")
                            
                            # Show affected services
                            for service, service_state in degradation_status["services"].items():
                                if service_state["fallback_active"]:
                                    ui.label(f"• {service}: {service_state['mode']} mode")
                        
                        # Close button
                        with ui.row().classes("w-full justify-end mt-4"):
                            ui.button("Close", on_click=dialog.close).props("flat")
                
                except Exception as e:
                    ui.label(f"Failed to load status: {str(e)}").classes("text-red-500")
            
            # Load status asynchronously
            asyncio.create_task(load_detailed_status())
        
        dialog.open()


# Global instances
error_notification_manager = ErrorNotificationManager()


def show_error(error: BaseApplicationError, context: Optional[Dict[str, Any]] = None):
    """Convenience function to show error notification."""
    return error_notification_manager.show_error(error, context)


def create_system_status_widget() -> ui.element:
    """Convenience function to create system status widget."""
    widget = SystemStatusWidget()
    return widget.create_widget()


def setup_global_error_handler():
    """Setup global error handler for the UI."""
    
    def handle_ui_error(error_type: str, error_data: Dict[str, Any]):
        """Handle error notifications from backend."""
        try:
            # Create error object from data if needed
            if isinstance(error_data, dict) and "user_message" in error_data:
                ui.notify(
                    error_data["user_message"],
                    type="negative" if error_data.get("severity") == "error" else "warning",
                    timeout=8000
                )
            else:
                ui.notify(
                    f"System issue detected: {error_type}",
                    type="warning",
                    timeout=5000
                )
        except Exception as e:
            # Fallback notification
            ui.notify(
                "A system error occurred",
                type="negative",
                timeout=5000
            )
    
    # Register with degradation manager
    degradation_manager.add_notification_callback(handle_ui_error)