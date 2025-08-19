"""
Centralized error handling framework for the Letta Claim Assistant.

Provides comprehensive error classification, recovery strategies, and user-friendly
error reporting with contextual information and recovery suggestions.
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import traceback
import sys
from pathlib import Path

from .logging_conf import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"  # System-level failures, app cannot continue
    ERROR = "error"       # Operation failures, user action needed
    WARNING = "warning"   # Potential issues, degraded functionality
    INFO = "info"         # Informational messages, no action needed


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"           # Automatically retry the operation
    FALLBACK = "fallback"     # Use alternative approach/service
    MANUAL = "manual"         # Requires user intervention
    ABORT = "abort"           # Operation cannot be completed
    DEGRADE = "degrade"       # Continue with reduced functionality


@dataclass
class RecoveryAction:
    """Represents a recovery action available to the user."""
    action_id: str
    label: str
    description: str
    callback: Optional[Callable] = None
    is_primary: bool = False


@dataclass
class ErrorContext:
    """Additional context information for errors."""
    matter_id: Optional[str] = None
    matter_name: Optional[str] = None
    operation: Optional[str] = None
    file_path: Optional[str] = None
    job_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None


class BaseApplicationError(Exception):
    """
    Base class for all application errors with enhanced context and recovery information.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.MANUAL,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        recovery_actions: Optional[List[RecoveryAction]] = None,
        error_code: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize application error.
        
        Args:
            message: Technical error message for logging
            severity: Error severity level
            recovery_strategy: How the error can be recovered from
            user_message: User-friendly error message
            suggestion: Recovery suggestion for the user
            context: Additional context information
            recovery_actions: Available recovery actions
            error_code: Unique error code for documentation lookup
            cause: Underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.user_message = user_message or self._generate_user_message()
        self.suggestion = suggestion
        self.context = context or ErrorContext()
        self.recovery_actions = recovery_actions or []
        self.error_code = error_code
        self.cause = cause
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc() if sys.exc_info()[0] else None
    
    def _generate_user_message(self) -> str:
        """Generate a user-friendly message from the technical message."""
        # Override in subclasses for more specific user messages
        return f"An error occurred: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error_code": self.error_code,
            "severity": self.severity.value,
            "recovery_strategy": self.recovery_strategy.value,
            "user_message": self.user_message,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "matter_id": self.context.matter_id,
                "matter_name": self.context.matter_name,
                "operation": self.context.operation,
                "file_path": self.context.file_path,
                "job_id": self.context.job_id,
                "provider": self.context.provider,
                "model": self.context.model
            },
            "recovery_actions": [
                {
                    "action_id": action.action_id,
                    "label": action.label,
                    "description": action.description,
                    "is_primary": action.is_primary
                }
                for action in self.recovery_actions
            ]
        }


class UserFacingError(BaseApplicationError):
    """Error that should be directly shown to users with helpful guidance."""
    
    def __init__(self, user_message: str, suggestion: str = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.ERROR)
        kwargs.setdefault('recovery_strategy', RecoveryStrategy.MANUAL)
        super().__init__(
            message=user_message,
            user_message=user_message,
            suggestion=suggestion,
            **kwargs
        )


class RetryableError(BaseApplicationError):
    """Error that can be automatically retried."""
    
    def __init__(self, message: str, max_retries: int = 3, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.WARNING)
        kwargs.setdefault('recovery_strategy', RecoveryStrategy.RETRY)
        self.max_retries = max_retries
        super().__init__(message, **kwargs)


class ResourceError(BaseApplicationError):
    """System resource related errors (disk, memory, network)."""
    
    def __init__(self, resource_type: str, message: str, **kwargs):
        self.resource_type = resource_type
        kwargs.setdefault('severity', ErrorSeverity.ERROR)
        kwargs.setdefault('recovery_strategy', RecoveryStrategy.MANUAL)
        kwargs.setdefault('error_code', f"RESOURCE_{resource_type.upper()}")
        super().__init__(message, **kwargs)


class ValidationError(BaseApplicationError):
    """Input validation failures."""
    
    def __init__(self, field: str, value: Any, constraint: str, **kwargs):
        self.field = field
        self.value = value
        self.constraint = constraint
        message = f"Validation failed for {field}: {constraint}"
        user_message = f"Invalid {field}: {constraint}"
        kwargs.setdefault('severity', ErrorSeverity.WARNING)
        kwargs.setdefault('recovery_strategy', RecoveryStrategy.MANUAL)
        kwargs.setdefault('error_code', 'VALIDATION_ERROR')
        # Remove user_message from kwargs if it exists to avoid duplicate
        kwargs.pop('user_message', None)
        super().__init__(
            message=message,
            user_message=user_message,
            **kwargs
        )


class ServiceUnavailableError(BaseApplicationError):
    """External service unavailable."""
    
    def __init__(self, service: str, **kwargs):
        self.service = service
        message = f"Service unavailable: {service}"
        user_message = f"The {service} service is currently unavailable"
        suggestion = "Please check your connection and try again, or switch to an alternative provider"
        kwargs.setdefault('severity', ErrorSeverity.WARNING)
        kwargs.setdefault('recovery_strategy', RecoveryStrategy.FALLBACK)
        kwargs.setdefault('error_code', f'SERVICE_{service.upper()}_UNAVAILABLE')
        # Remove user_message from kwargs if it exists to avoid duplicate
        kwargs.pop('user_message', None)
        super().__init__(
            message=message,
            user_message=user_message,
            suggestion=suggestion,
            **kwargs
        )


class ConfigurationError(BaseApplicationError):
    """Configuration and setup errors."""
    
    def __init__(self, component: str, issue: str, **kwargs):
        self.component = component
        self.issue = issue
        message = f"Configuration error in {component}: {issue}"
        user_message = f"Setup issue with {component}: {issue}"
        kwargs.setdefault('severity', ErrorSeverity.ERROR)
        kwargs.setdefault('recovery_strategy', RecoveryStrategy.MANUAL)
        kwargs.setdefault('error_code', f'CONFIG_{component.upper()}')
        # Remove user_message from kwargs if it exists to avoid duplicate
        kwargs.pop('user_message', None)
        super().__init__(
            message=message,
            user_message=user_message,
            **kwargs
        )


class FileProcessingError(BaseApplicationError):
    """File processing related errors."""
    
    def __init__(self, file_path: Union[str, Path], operation: str, reason: str, **kwargs):
        self.file_path = Path(file_path) if isinstance(file_path, str) else file_path
        self.operation = operation
        self.reason = reason
        
        message = f"Failed to {operation} file {self.file_path}: {reason}"
        user_message = f"Could not {operation} {self.file_path.name}: {reason}"
        
        # Set context
        if 'context' not in kwargs:
            kwargs['context'] = ErrorContext()
        kwargs['context'].file_path = str(self.file_path)
        kwargs['context'].operation = operation
        
        kwargs.setdefault('severity', ErrorSeverity.ERROR)
        kwargs.setdefault('recovery_strategy', RecoveryStrategy.RETRY)
        kwargs.setdefault('error_code', f'FILE_{operation.upper()}')
        
        # Remove user_message from kwargs if it exists to avoid duplicate
        kwargs.pop('user_message', None)
        
        super().__init__(
            message=message,
            user_message=user_message,
            **kwargs
        )


class StartupError(BaseApplicationError):
    """Startup validation and initialization errors."""
    
    def __init__(self, check_name: str, issue: str, **kwargs):
        self.check_name = check_name
        self.issue = issue
        message = f"Startup check failed ({check_name}): {issue}"
        user_message = f"System check failed: {issue}"
        kwargs.setdefault('severity', ErrorSeverity.ERROR)
        kwargs.setdefault('recovery_strategy', RecoveryStrategy.MANUAL)
        kwargs.setdefault('error_code', f'STARTUP_{check_name.upper()}')
        # Remove user_message from kwargs if it exists to avoid duplicate
        kwargs.pop('user_message', None)
        super().__init__(
            message=message,
            user_message=user_message,
            **kwargs
        )


class ErrorHandler:
    """
    Centralized error handler for the application.
    """
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[BaseApplicationError] = []
        self.max_recent_errors = 100
    
    def handle_error(
        self,
        error: Union[BaseApplicationError, Exception],
        context: Optional[ErrorContext] = None,
        notify_user: bool = True
    ) -> BaseApplicationError:
        """
        Handle an error with logging, counting, and optional user notification.
        
        Args:
            error: The error to handle
            context: Additional context information
            notify_user: Whether to trigger user notifications
            
        Returns:
            BaseApplicationError instance (converted if needed)
        """
        # Convert standard exceptions to BaseApplicationError
        if not isinstance(error, BaseApplicationError):
            app_error = self._convert_exception(error, context)
        else:
            app_error = error
            if context and not app_error.context:
                app_error.context = context
        
        # Log the error
        self._log_error(app_error)
        
        # Track error statistics
        self._track_error(app_error)
        
        # Store in recent errors
        self._store_recent_error(app_error)
        
        # Trigger user notification if requested
        if notify_user:
            self._notify_user(app_error)
        
        return app_error
    
    def _convert_exception(
        self,
        exc: Exception,
        context: Optional[ErrorContext] = None
    ) -> BaseApplicationError:
        """Convert a standard exception to BaseApplicationError."""
        
        # Map common exception types
        if isinstance(exc, FileNotFoundError):
            error = FileProcessingError(
                file_path=str(exc.filename) if exc.filename else "unknown",
                operation="access",
                reason="File not found",
                cause=exc
            )
            # Update context if provided
            if context:
                for attr in ['matter_id', 'matter_name', 'operation', 'job_id', 'provider', 'model']:
                    if hasattr(context, attr) and getattr(context, attr):
                        setattr(error.context, attr, getattr(context, attr))
            return error
        elif isinstance(exc, PermissionError):
            error = FileProcessingError(
                file_path=str(exc.filename) if exc.filename else "unknown",
                operation="access",
                reason="Permission denied",
                cause=exc
            )
            # Update context if provided
            if context:
                for attr in ['matter_id', 'matter_name', 'operation', 'job_id', 'provider', 'model']:
                    if hasattr(context, attr) and getattr(context, attr):
                        setattr(error.context, attr, getattr(context, attr))
            return error
        elif isinstance(exc, ConnectionError):
            return ServiceUnavailableError(
                service="network",
                cause=exc,
                context=context
            )
        elif isinstance(exc, TimeoutError):
            return RetryableError(
                message=f"Operation timed out: {str(exc)}",
                cause=exc,
                context=context
            )
        elif isinstance(exc, ValueError):
            return ValidationError(
                field="input",
                value=str(exc),
                constraint="Invalid value format",
                cause=exc,
                context=context
            )
        else:
            # Generic error conversion
            return BaseApplicationError(
                message=str(exc),
                user_message=f"An unexpected error occurred: {type(exc).__name__}",
                suggestion="Please try again or contact support if the problem persists",
                severity=ErrorSeverity.ERROR,
                recovery_strategy=RecoveryStrategy.MANUAL,
                cause=exc,
                context=context
            )
    
    def _log_error(self, error: BaseApplicationError):
        """Log error with appropriate level and context."""
        log_data = {
            "error_code": error.error_code,
            "severity": error.severity.value,
            "recovery_strategy": error.recovery_strategy.value,
            "user_message": error.user_message,
            "suggestion": error.suggestion
        }
        
        # Add context data
        if error.context:
            log_data.update({
                "matter_id": error.context.matter_id,
                "operation": error.context.operation,
                "file_path": error.context.file_path,
                "job_id": error.context.job_id,
                "provider": error.context.provider
            })
        
        # Add cause information
        if error.cause:
            log_data["cause"] = str(error.cause)
            log_data["cause_type"] = type(error.cause).__name__
        
        # Log with appropriate level
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(error.message, **log_data)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(error.message, **log_data)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(error.message, **log_data)
        else:
            logger.info(error.message, **log_data)
    
    def _track_error(self, error: BaseApplicationError):
        """Track error statistics."""
        error_key = error.error_code or type(error).__name__
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def _store_recent_error(self, error: BaseApplicationError):
        """Store error in recent errors list."""
        self.recent_errors.append(error)
        
        # Trim to max size
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
    
    def _notify_user(self, error: BaseApplicationError):
        """Trigger user notification (to be implemented in UI layer)."""
        # This will be called by UI components to show error dialogs
        # For now, just log that notification should be shown
        logger.debug(
            "User notification triggered",
            error_code=error.error_code,
            severity=error.severity.value,
            user_message=error.user_message
        )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "recent_error_count": len(self.recent_errors),
            "recent_errors": [error.to_dict() for error in self.recent_errors[-10:]]
        }
    
    def clear_stats(self):
        """Clear error statistics."""
        self.error_counts.clear()
        self.recent_errors.clear()


# Global error handler instance
error_handler = ErrorHandler()


def handle_error(
    error: Union[BaseApplicationError, Exception],
    context: Optional[ErrorContext] = None,
    notify_user: bool = True
) -> BaseApplicationError:
    """
    Convenience function to handle errors using the global error handler.
    """
    return error_handler.handle_error(error, context, notify_user)


def create_context(
    matter_id: str = None,
    matter_name: str = None,
    operation: str = None,
    file_path: Union[str, Path] = None,
    job_id: str = None,
    provider: str = None,
    model: str = None,
    **kwargs
) -> ErrorContext:
    """
    Convenience function to create error context.
    """
    return ErrorContext(
        matter_id=matter_id,
        matter_name=matter_name,
        operation=operation,
        file_path=str(file_path) if file_path else None,
        job_id=job_id,
        provider=provider,
        model=model,
        user_data=kwargs if kwargs else None
    )