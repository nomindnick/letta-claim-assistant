#!/usr/bin/env python3
"""
Sprint 11 verification script: Error Handling & Edge Cases

Tests the comprehensive error handling, retry mechanisms, graceful degradation,
and edge case management implemented in Sprint 11.
"""

import asyncio
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.error_handler import (
    BaseApplicationError, UserFacingError, RetryableError, ResourceError,
    ValidationError, ServiceUnavailableError, FileProcessingError,
    handle_error, create_context, error_handler
)
from app.retry_utils import retry_manager, RetryPolicy, BackoffStrategy
from app.resource_monitor import resource_monitor
from app.degradation import degradation_manager
from app.ocr import OCRProcessor


class TestResults:
    """Track test results."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = []
    
    def add_test(self, test_name: str, success: bool, error: str = None):
        """Add test result."""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            print(f"✅ {test_name}")
        else:
            self.failed_tests.append((test_name, error))
            print(f"❌ {test_name}: {error}")
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {len(self.failed_tests)}")
        
        if self.failed_tests:
            print(f"\nFailed tests:")
            for test_name, error in self.failed_tests:
                print(f"  - {test_name}: {error}")
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")
        
        return len(self.failed_tests) == 0


def test_error_types(results: TestResults):
    """Test different error types and their properties."""
    print(f"\n{'='*60}")
    print("TESTING ERROR TYPES")
    print(f"{'='*60}")
    
    # Test BaseApplicationError
    try:
        error = BaseApplicationError(
            message="Test error",
            user_message="User-friendly message",
            error_code="TEST_001"
        )
        assert error.message == "Test error"
        assert error.user_message == "User-friendly message"
        assert error.error_code == "TEST_001"
        assert error.timestamp is not None
        results.add_test("BaseApplicationError creation", True)
    except Exception as e:
        results.add_test("BaseApplicationError creation", False, str(e))
    
    # Test UserFacingError
    try:
        error = UserFacingError("Something went wrong", suggestion="Try again")
        assert error.user_message == "Something went wrong"
        assert error.suggestion == "Try again"
        results.add_test("UserFacingError creation", True)
    except Exception as e:
        results.add_test("UserFacingError creation", False, str(e))
    
    # Test FileProcessingError
    try:
        error = FileProcessingError(
            file_path="/test/file.pdf",
            operation="parse",
            reason="Corrupted file"
        )
        assert str(error.file_path) == "/test/file.pdf"
        assert error.operation == "parse"
        assert error.reason == "Corrupted file"
        results.add_test("FileProcessingError creation", True)
    except Exception as e:
        results.add_test("FileProcessingError creation", False, str(e))
    
    # Test error serialization
    try:
        error = BaseApplicationError(
            message="Test error",
            error_code="TEST_002"
        )
        error_dict = error.to_dict()
        assert "error_code" in error_dict
        assert "severity" in error_dict
        assert "user_message" in error_dict
        assert error_dict["error_code"] == "TEST_002"
        results.add_test("Error serialization", True)
    except Exception as e:
        results.add_test("Error serialization", False, str(e))


def test_error_handler(results: TestResults):
    """Test error handler functionality."""
    print(f"\n{'='*60}")
    print("TESTING ERROR HANDLER")
    print(f"{'='*60}")
    
    # Test handling application error
    try:
        error = UserFacingError("Test error")
        result = error_handler.handle_error(error, notify_user=False)
        assert result is error
        assert len(error_handler.recent_errors) > 0
        results.add_test("Handle application error", True)
    except Exception as e:
        results.add_test("Handle application error", False, str(e))
    
    # Test converting standard exception
    try:
        original_error = FileNotFoundError("File not found")
        result = error_handler.handle_error(original_error, notify_user=False)
        assert isinstance(result, FileProcessingError)
        assert result.cause is original_error
        results.add_test("Convert standard exception", True)
    except Exception as e:
        results.add_test("Convert standard exception", False, str(e))
    
    # Test error statistics
    try:
        error1 = UserFacingError("Error 1", error_code="TEST_STAT")
        error2 = UserFacingError("Error 2", error_code="TEST_STAT")
        
        error_handler.handle_error(error1, notify_user=False)
        error_handler.handle_error(error2, notify_user=False)
        
        stats = error_handler.get_error_stats()
        assert "error_counts" in stats
        assert stats["error_counts"]["TEST_STAT"] == 2
        results.add_test("Error statistics tracking", True)
    except Exception as e:
        results.add_test("Error statistics tracking", False, str(e))
    
    # Test context creation
    try:
        context = create_context(
            matter_id="test-matter",
            operation="test-operation",
            file_path="/test/file.pdf"
        )
        assert context.matter_id == "test-matter"
        assert context.operation == "test-operation"
        assert context.file_path == "/test/file.pdf"
        results.add_test("Error context creation", True)
    except Exception as e:
        results.add_test("Error context creation", False, str(e))


async def test_retry_mechanisms(results: TestResults):
    """Test retry mechanisms and policies."""
    print(f"\n{'='*60}")
    print("TESTING RETRY MECHANISMS")
    print(f"{'='*60}")
    
    # Test retry policy
    try:
        policy = retry_manager.get_policy("api")
        assert policy.max_attempts > 0
        assert policy.base_delay > 0
        results.add_test("Retry policy retrieval", True)
    except Exception as e:
        results.add_test("Retry policy retrieval", False, str(e))
    
    # Test delay calculation
    try:
        policy = RetryPolicy(
            base_delay=1.0,
            multiplier=2.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        
        delay1 = retry_manager.calculate_delay(1, policy)
        delay2 = retry_manager.calculate_delay(2, policy)
        delay3 = retry_manager.calculate_delay(3, policy)
        
        assert delay1 == 1.0  # 1.0 * 2^0
        assert delay2 == 2.0  # 1.0 * 2^1
        assert delay3 == 4.0  # 1.0 * 2^2
        results.add_test("Delay calculation", True)
    except Exception as e:
        results.add_test("Delay calculation", False, str(e))
    
    # Test circuit breaker
    try:
        circuit = retry_manager.get_circuit_breaker("test_service")
        assert circuit.can_execute() is True
        
        # Record failures
        for _ in range(5):
            circuit.record_failure()
        
        # Should be open now
        assert circuit.can_execute() is False
        results.add_test("Circuit breaker functionality", True)
    except Exception as e:
        results.add_test("Circuit breaker functionality", False, str(e))
    
    # Test retry budget
    try:
        budget = retry_manager.get_retry_budget("test_budget")
        assert budget.can_retry() is True
        
        # Exhaust budget
        for _ in range(budget.budget_per_minute):
            budget.record_attempt()
        
        assert budget.can_retry() is False
        results.add_test("Retry budget functionality", True)
    except Exception as e:
        results.add_test("Retry budget functionality", False, str(e))
    
    # Test successful retry
    try:
        call_count = 0
        
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"Failure {call_count}")
            return f"Success after {call_count} attempts"
        
        result = await retry_manager.execute_with_retry(
            flaky_function,
            operation_type="api"
        )
        
        assert result == "Success after 3 attempts"
        assert call_count == 3
        results.add_test("Successful retry execution", True)
    except Exception as e:
        results.add_test("Successful retry execution", False, str(e))


async def test_resource_monitoring(results: TestResults):
    """Test resource monitoring functionality."""
    print(f"\n{'='*60}")
    print("TESTING RESOURCE MONITORING")
    print(f"{'='*60}")
    
    # Test disk space monitoring
    try:
        disk_info = resource_monitor.check_disk_space()
        assert disk_info.total > 0
        assert disk_info.used >= 0
        assert disk_info.free >= 0
        assert 0 <= disk_info.percent_used <= 1
        results.add_test("Disk space monitoring", True)
    except Exception as e:
        results.add_test("Disk space monitoring", False, str(e))
    
    # Test memory monitoring
    try:
        memory_info = resource_monitor.check_memory_usage()
        assert memory_info.total > 0
        assert memory_info.used >= 0
        assert memory_info.free >= 0
        assert 0 <= memory_info.percent_used <= 1
        results.add_test("Memory monitoring", True)
    except Exception as e:
        results.add_test("Memory monitoring", False, str(e))
    
    # Test network connectivity
    try:
        network_info = await resource_monitor.check_network_connectivity()
        assert network_info.status is not None
        assert network_info.last_check is not None
        results.add_test("Network connectivity check", True)
    except Exception as e:
        results.add_test("Network connectivity check", False, str(e))
    
    # Test comprehensive status
    try:
        status = await resource_monitor.get_comprehensive_status()
        assert "overall_status" in status
        assert "disk" in status
        assert "memory" in status
        assert "network" in status
        assert "services" in status
        results.add_test("Comprehensive status check", True)
    except Exception as e:
        results.add_test("Comprehensive status check", False, str(e))


async def test_degradation_management(results: TestResults):
    """Test graceful degradation functionality."""
    print(f"\n{'='*60}")
    print("TESTING DEGRADATION MANAGEMENT")
    print(f"{'='*60}")
    
    # Test service evaluation
    try:
        service_states = await degradation_manager.evaluate_all_services()
        assert isinstance(service_states, dict)
        assert len(service_states) > 0
        results.add_test("Service evaluation", True)
    except Exception as e:
        results.add_test("Service evaluation", False, str(e))
    
    # Test degradation status
    try:
        status = degradation_manager.get_degradation_status()
        assert "global_level" in status
        assert "services" in status
        results.add_test("Degradation status", True)
    except Exception as e:
        results.add_test("Degradation status", False, str(e))
    
    # Test feature availability
    try:
        features = ["agent_memory", "local_llm", "vector_search", "pdf_upload"]
        for feature in features:
            is_available = degradation_manager.is_feature_available(feature)
            assert isinstance(is_available, bool)
        results.add_test("Feature availability check", True)
    except Exception as e:
        results.add_test("Feature availability check", False, str(e))
    
    # Test user guidance
    try:
        guidance = degradation_manager.get_user_guidance()
        assert isinstance(guidance, list)
        results.add_test("User guidance generation", True)
    except Exception as e:
        results.add_test("User guidance generation", False, str(e))


async def test_file_processing_errors(results: TestResults):
    """Test file processing error handling."""
    print(f"\n{'='*60}")
    print("TESTING FILE PROCESSING ERRORS")
    print(f"{'='*60}")
    
    temp_dir = Path(tempfile.mkdtemp())
    ocr_processor = OCRProcessor(timeout_seconds=10)
    
    try:
        # Test missing file error
        try:
            missing_file = temp_dir / "nonexistent.pdf"
            output_file = temp_dir / "output.pdf"
            
            result = await ocr_processor.process_pdf(missing_file, output_file)
            assert result.success is False
            assert result.error_message is not None
            results.add_test("Missing file error handling", True)
        except Exception as e:
            results.add_test("Missing file error handling", False, str(e))
        
        # Test empty file error
        try:
            empty_file = temp_dir / "empty.pdf"
            empty_file.touch()
            output_file = temp_dir / "output.pdf"
            
            result = await ocr_processor.process_pdf(empty_file, output_file)
            assert result.success is False
            assert result.error_message is not None
            results.add_test("Empty file error handling", True)
        except Exception as e:
            results.add_test("Empty file error handling", False, str(e))
        
        # Test invalid PDF content
        try:
            invalid_pdf = temp_dir / "invalid.pdf"
            invalid_pdf.write_text("This is not a PDF file")
            output_file = temp_dir / "output.pdf"
            
            result = await ocr_processor.process_pdf(invalid_pdf, output_file)
            assert result.success is False
            assert result.error_message is not None
            results.add_test("Invalid PDF error handling", True)
        except Exception as e:
            results.add_test("Invalid PDF error handling", False, str(e))
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_integration_scenarios(results: TestResults):
    """Test integration scenarios."""
    print(f"\n{'='*60}")
    print("TESTING INTEGRATION SCENARIOS")
    print(f"{'='*60}")
    
    # Test error context preservation
    try:
        context = create_context(
            matter_id="test-matter",
            operation="integration-test"
        )
        
        original_error = ValueError("Test error")
        handled_error = handle_error(original_error, context, notify_user=False)
        
        assert handled_error.context.matter_id == "test-matter"
        assert handled_error.context.operation == "integration-test"
        assert handled_error.cause is original_error
        results.add_test("Error context preservation", True)
    except Exception as e:
        results.add_test("Error context preservation", False, str(e))
    
    # Test retry with error handling integration
    try:
        policy = retry_manager.get_policy("api")
        
        # RetryableError should be retryable
        retryable_error = RetryableError("Test error")
        should_retry = retry_manager.should_retry(retryable_error, 1, policy)
        assert should_retry is True
        
        # Non-retryable errors should not be retried beyond attempts
        should_not_retry = retry_manager.should_retry(retryable_error, 10, policy)
        assert should_not_retry is False
        
        results.add_test("Retry integration with error types", True)
    except Exception as e:
        results.add_test("Retry integration with error types", False, str(e))


async def main():
    """Run all Sprint 11 tests."""
    print("🧪 SPRINT 11 VERIFICATION: Error Handling & Edge Cases")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = TestResults()
    
    # Run all test suites
    test_error_types(results)
    test_error_handler(results)
    await test_retry_mechanisms(results)
    await test_resource_monitoring(results)
    await test_degradation_management(results)
    await test_file_processing_errors(results)
    test_integration_scenarios(results)
    
    # Print final summary
    success = results.print_summary()
    
    if success:
        print(f"\n🎉 All Sprint 11 tests passed! Error handling system is working correctly.")
        return 0
    else:
        print(f"\n❌ Some Sprint 11 tests failed. Please review the error handling implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)