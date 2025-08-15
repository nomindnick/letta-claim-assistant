"""
Unit tests for error handling framework.

Tests the error handler, error types, recovery strategies, and context management.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.error_handler import (
    BaseApplicationError, UserFacingError, RetryableError, ResourceError,
    ValidationError, ServiceUnavailableError, ConfigurationError,
    FileProcessingError, ErrorHandler, ErrorSeverity, RecoveryStrategy,
    RecoveryAction, ErrorContext, handle_error, create_context
)


class TestErrorTypes:
    """Test different error types and their properties."""
    
    def test_base_application_error(self):
        """Test BaseApplicationError creation and properties."""
        error = BaseApplicationError(
            message="Test error",
            severity=ErrorSeverity.ERROR,
            recovery_strategy=RecoveryStrategy.RETRY,
            user_message="User-friendly message",
            suggestion="Try again",
            error_code="TEST_ERROR"
        )
        
        assert error.message == "Test error"
        assert error.severity == ErrorSeverity.ERROR
        assert error.recovery_strategy == RecoveryStrategy.RETRY
        assert error.user_message == "User-friendly message"
        assert error.suggestion == "Try again"
        assert error.error_code == "TEST_ERROR"
        assert isinstance(error.timestamp, datetime)
        assert error.context is not None
    
    def test_user_facing_error(self):
        """Test UserFacingError creation."""
        error = UserFacingError(
            user_message="Something went wrong",
            suggestion="Please try again"
        )
        
        assert error.user_message == "Something went wrong"
        assert error.suggestion == "Please try again"
        assert error.severity == ErrorSeverity.ERROR
        assert error.recovery_strategy == RecoveryStrategy.MANUAL
    
    def test_retryable_error(self):
        """Test RetryableError creation."""
        error = RetryableError(
            message="Network timeout",
            max_retries=5
        )
        
        assert error.message == "Network timeout"
        assert error.max_retries == 5
        assert error.severity == ErrorSeverity.WARNING
        assert error.recovery_strategy == RecoveryStrategy.RETRY
    
    def test_resource_error(self):
        """Test ResourceError creation."""
        error = ResourceError(
            resource_type="disk",
            message="Insufficient disk space"
        )
        
        assert error.resource_type == "disk"
        assert error.message == "Insufficient disk space"
        assert error.error_code == "RESOURCE_DISK"
        assert error.severity == ErrorSeverity.ERROR
    
    def test_validation_error(self):
        """Test ValidationError creation."""
        error = ValidationError(
            field="email",
            value="invalid-email",
            constraint="Must be valid email format"
        )
        
        assert error.field == "email"
        assert error.value == "invalid-email"
        assert error.constraint == "Must be valid email format"
        assert error.error_code == "VALIDATION_ERROR"
    
    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError creation."""
        error = ServiceUnavailableError(service="ollama")
        
        assert error.service == "ollama"
        assert error.error_code == "SERVICE_OLLAMA_UNAVAILABLE"
        assert error.recovery_strategy == RecoveryStrategy.FALLBACK
        assert "ollama" in error.user_message.lower()
    
    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError(
            component="database",
            issue="Missing connection string"
        )
        
        assert error.component == "database"
        assert error.issue == "Missing connection string"
        assert error.error_code == "CONFIG_DATABASE"
    
    def test_file_processing_error(self):
        """Test FileProcessingError creation."""
        file_path = Path("/test/file.pdf")
        error = FileProcessingError(
            file_path=file_path,
            operation="parse",
            reason="Corrupted file"
        )
        
        assert error.file_path == file_path
        assert error.operation == "parse"
        assert error.reason == "Corrupted file"
        assert error.error_code == "FILE_PARSE"
        assert str(file_path) in error.context.file_path
    
    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        context = ErrorContext(
            matter_id="test-matter",
            operation="test-op",
            file_path="/test/file.txt"
        )
        
        recovery_action = RecoveryAction(
            action_id="retry",
            label="Retry",
            description="Try again",
            is_primary=True
        )
        
        error = BaseApplicationError(
            message="Test error",
            severity=ErrorSeverity.WARNING,
            recovery_strategy=RecoveryStrategy.RETRY,
            context=context,
            recovery_actions=[recovery_action],
            error_code="TEST_001"
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["severity"] == "warning"
        assert error_dict["recovery_strategy"] == "retry"
        assert error_dict["context"]["matter_id"] == "test-matter"
        assert error_dict["context"]["operation"] == "test-op"
        assert len(error_dict["recovery_actions"]) == 1
        assert error_dict["recovery_actions"][0]["action_id"] == "retry"
        assert error_dict["recovery_actions"][0]["is_primary"] is True


class TestErrorContext:
    """Test error context management."""
    
    def test_create_context(self):
        """Test context creation with create_context helper."""
        context = create_context(
            matter_id="matter-123",
            matter_name="Test Matter",
            operation="pdf_processing",
            file_path="/test/file.pdf",
            provider="ollama"
        )
        
        assert context.matter_id == "matter-123"
        assert context.matter_name == "Test Matter"
        assert context.operation == "pdf_processing"
        assert context.file_path == "/test/file.pdf"
        assert context.provider == "ollama"
    
    def test_context_with_path_object(self):
        """Test context creation with Path object."""
        file_path = Path("/test/document.pdf")
        context = create_context(file_path=file_path)
        
        assert context.file_path == str(file_path)


class TestErrorHandler:
    """Test ErrorHandler class functionality."""
    
    def setup_method(self):
        """Setup test method."""
        self.handler = ErrorHandler()
    
    def test_handle_application_error(self):
        """Test handling BaseApplicationError."""
        error = UserFacingError(
            user_message="Test error",
            suggestion="Try again"
        )
        
        result = self.handler.handle_error(error, notify_user=False)
        
        assert result is error
        assert len(self.handler.recent_errors) == 1
        assert self.handler.recent_errors[0] is error
    
    def test_handle_standard_exception(self):
        """Test converting standard exceptions to BaseApplicationError."""
        original_error = FileNotFoundError("File not found")
        
        result = self.handler.handle_error(original_error, notify_user=False)
        
        assert isinstance(result, FileProcessingError)
        assert result.cause is original_error
        assert len(self.handler.recent_errors) == 1
    
    def test_handle_connection_error(self):
        """Test converting ConnectionError to ServiceUnavailableError."""
        original_error = ConnectionError("Connection failed")
        
        result = self.handler.handle_error(original_error, notify_user=False)
        
        assert isinstance(result, ServiceUnavailableError)
        assert result.service == "network"
        assert result.cause is original_error
    
    def test_handle_timeout_error(self):
        """Test converting TimeoutError to RetryableError."""
        original_error = TimeoutError("Operation timed out")
        
        result = self.handler.handle_error(original_error, notify_user=False)
        
        assert isinstance(result, RetryableError)
        assert result.cause is original_error
    
    def test_handle_permission_error(self):
        """Test converting PermissionError to FileProcessingError."""
        original_error = PermissionError("Permission denied")
        original_error.filename = "/test/file.txt"
        
        result = self.handler.handle_error(original_error, notify_user=False)
        
        assert isinstance(result, FileProcessingError)
        assert result.operation == "access"
        assert result.reason == "Permission denied"
    
    def test_error_counting(self):
        """Test error counting functionality."""
        error1 = UserFacingError("Error 1", error_code="TEST_001")
        error2 = UserFacingError("Error 2", error_code="TEST_001")
        error3 = UserFacingError("Error 3", error_code="TEST_002")
        
        self.handler.handle_error(error1, notify_user=False)
        self.handler.handle_error(error2, notify_user=False)
        self.handler.handle_error(error3, notify_user=False)
        
        stats = self.handler.get_error_stats()
        assert stats["error_counts"]["TEST_001"] == 2
        assert stats["error_counts"]["TEST_002"] == 1
        assert stats["recent_error_count"] == 3
    
    def test_recent_errors_limit(self):
        """Test recent errors list respects maximum size."""
        # Create more errors than the limit
        for i in range(150):  # More than max_recent_errors (100)
            error = UserFacingError(f"Error {i}", error_code=f"TEST_{i:03d}")
            self.handler.handle_error(error, notify_user=False)
        
        assert len(self.handler.recent_errors) == 100
        # Should contain the last 100 errors
        assert self.handler.recent_errors[-1].user_message == "Error 149"
        assert self.handler.recent_errors[0].user_message == "Error 50"
    
    def test_clear_stats(self):
        """Test clearing error statistics."""
        error = UserFacingError("Test error", error_code="TEST_001")
        self.handler.handle_error(error, notify_user=False)
        
        assert len(self.handler.recent_errors) == 1
        assert len(self.handler.error_counts) == 1
        
        self.handler.clear_stats()
        
        assert len(self.handler.recent_errors) == 0
        assert len(self.handler.error_counts) == 0
    
    def test_handle_error_with_context(self):
        """Test handling error with additional context."""
        context = create_context(
            matter_id="test-matter",
            operation="test-operation"
        )
        
        error = UserFacingError("Test error")
        result = self.handler.handle_error(error, context=context, notify_user=False)
        
        # Context should be set on the error
        assert result.context.matter_id == "test-matter"
        assert result.context.operation == "test-operation"


class TestRecoveryActions:
    """Test recovery actions functionality."""
    
    def test_recovery_action_creation(self):
        """Test RecoveryAction creation."""
        callback = Mock()
        action = RecoveryAction(
            action_id="retry_operation",
            label="Retry",
            description="Retry the failed operation",
            callback=callback,
            is_primary=True
        )
        
        assert action.action_id == "retry_operation"
        assert action.label == "Retry"
        assert action.description == "Retry the failed operation"
        assert action.callback is callback
        assert action.is_primary is True
    
    def test_error_with_recovery_actions(self):
        """Test error with multiple recovery actions."""
        retry_action = RecoveryAction(
            action_id="retry",
            label="Retry",
            description="Try again",
            is_primary=True
        )
        
        skip_action = RecoveryAction(
            action_id="skip",
            label="Skip",
            description="Skip this item"
        )
        
        error = BaseApplicationError(
            message="Test error",
            recovery_actions=[retry_action, skip_action]
        )
        
        assert len(error.recovery_actions) == 2
        assert error.recovery_actions[0].is_primary is True
        assert error.recovery_actions[1].is_primary is False


class TestGlobalErrorHandler:
    """Test global error handler functions."""
    
    def test_handle_error_function(self):
        """Test global handle_error function."""
        error = UserFacingError("Test error")
        context = create_context(operation="test")
        
        result = handle_error(error, context, notify_user=False)
        
        assert isinstance(result, BaseApplicationError)
        assert result.context.operation == "test"
    
    def test_handle_standard_exception_function(self):
        """Test global handle_error function with standard exception."""
        original_error = ValueError("Invalid value")
        context = create_context(operation="validation")
        
        result = handle_error(original_error, context, notify_user=False)
        
        assert isinstance(result, ValidationError)
        assert result.cause is original_error
        assert result.context.operation == "validation"


@pytest.mark.asyncio
class TestAsyncErrorHandling:
    """Test error handling in async contexts."""
    
    async def test_async_error_handling(self):
        """Test that error handling works correctly in async contexts."""
        async def failing_operation():
            raise ConnectionError("Network failed")
        
        try:
            await failing_operation()
        except Exception as e:
            result = handle_error(e, notify_user=False)
            
            assert isinstance(result, ServiceUnavailableError)
            assert result.service == "network"
    
    async def test_async_with_context(self):
        """Test async error handling with context."""
        async def failing_operation():
            raise TimeoutError("Operation timed out")
        
        context = create_context(
            operation="async_test",
            provider="test_provider"
        )
        
        try:
            await failing_operation()
        except Exception as e:
            result = handle_error(e, context, notify_user=False)
            
            assert isinstance(result, RetryableError)
            assert result.context.operation == "async_test"
            assert result.context.provider == "test_provider"


class TestErrorSerialization:
    """Test error serialization and deserialization."""
    
    def test_complex_error_serialization(self):
        """Test serialization of complex error with all fields."""
        context = ErrorContext(
            matter_id="matter-123",
            matter_name="Test Matter",
            operation="complex_operation",
            file_path="/path/to/file.pdf",
            job_id="job-456",
            provider="test_provider",
            model="test_model"
        )
        
        recovery_actions = [
            RecoveryAction("retry", "Retry", "Try again", is_primary=True),
            RecoveryAction("skip", "Skip", "Skip this item")
        ]
        
        original_error = ValueError("Original cause")
        
        error = BaseApplicationError(
            message="Complex test error",
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            user_message="Something went wrong",
            suggestion="Try these recovery options",
            context=context,
            recovery_actions=recovery_actions,
            error_code="COMPLEX_ERROR",
            cause=original_error
        )
        
        error_dict = error.to_dict()
        
        # Verify all fields are present
        assert error_dict["error_code"] == "COMPLEX_ERROR"
        assert error_dict["severity"] == "critical"
        assert error_dict["recovery_strategy"] == "fallback"
        assert error_dict["user_message"] == "Something went wrong"
        assert error_dict["suggestion"] == "Try these recovery options"
        
        # Verify context
        context_dict = error_dict["context"]
        assert context_dict["matter_id"] == "matter-123"
        assert context_dict["matter_name"] == "Test Matter"
        assert context_dict["operation"] == "complex_operation"
        assert context_dict["file_path"] == "/path/to/file.pdf"
        assert context_dict["job_id"] == "job-456"
        assert context_dict["provider"] == "test_provider"
        assert context_dict["model"] == "test_model"
        
        # Verify recovery actions
        actions = error_dict["recovery_actions"]
        assert len(actions) == 2
        assert actions[0]["action_id"] == "retry"
        assert actions[0]["is_primary"] is True
        assert actions[1]["action_id"] == "skip"
        assert actions[1]["is_primary"] is False


if __name__ == "__main__":
    pytest.main([__file__])