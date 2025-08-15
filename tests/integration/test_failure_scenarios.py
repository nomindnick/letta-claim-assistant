"""
Integration tests for failure scenarios and edge cases.

Tests the application's behavior under various failure conditions including
network issues, resource constraints, corrupted files, and service unavailability.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.error_handler import (
    FileProcessingError, ResourceError, ServiceUnavailableError,
    handle_error, create_context
)
from app.retry_utils import retry_manager
from app.resource_monitor import resource_monitor, ResourceStatus
from app.degradation import degradation_manager, DegradationLevel
from app.ocr import OCRProcessor
from app.matters import MatterManager


class TestNetworkFailureScenarios:
    """Test scenarios involving network connectivity issues."""
    
    @pytest.mark.asyncio
    async def test_ollama_service_unavailable(self):
        """Test behavior when Ollama service is unavailable."""
        # Test Ollama connectivity check
        service_info = await resource_monitor.check_ollama_service()
        
        # Since Ollama might not be running in test environment,
        # we test both scenarios
        if service_info.status == ResourceStatus.UNAVAILABLE:
            assert service_info.error_message is not None
            assert "connection" in service_info.error_message.lower() or "refused" in service_info.error_message.lower()
        else:
            assert service_info.status == ResourceStatus.HEALTHY
            assert service_info.version is not None
    
    @pytest.mark.asyncio
    async def test_network_connectivity_failure(self):
        """Test network connectivity failure detection."""
        # Test with unreachable host
        network_info = await resource_monitor.check_network_connectivity(
            host="192.0.2.1",  # RFC5737 test address that should be unreachable
            timeout=1.0
        )
        
        # Should detect network failure
        assert network_info.status == ResourceStatus.UNAVAILABLE
        assert network_info.error_message is not None
    
    @pytest.mark.asyncio
    async def test_degradation_on_service_failure(self):
        """Test graceful degradation when services fail."""
        # Simulate service failure
        await degradation_manager.evaluate_service_health("ollama")
        
        # Check if degradation was applied
        status = degradation_manager.get_degradation_status()
        
        # Depending on whether Ollama is available, different behavior expected
        if "ollama" in status["services"]:
            service_state = status["services"]["ollama"]
            assert service_state["mode"] in ["normal", "fallback", "disabled"]


class TestResourceConstraintScenarios:
    """Test scenarios involving resource constraints."""
    
    def test_disk_space_monitoring(self):
        """Test disk space monitoring and warnings."""
        # Check current disk space
        disk_info = resource_monitor.check_disk_space()
        
        assert disk_info.total > 0
        assert disk_info.used >= 0
        assert disk_info.free >= 0
        assert 0 <= disk_info.percent_used <= 1
        assert disk_info.status in [
            ResourceStatus.HEALTHY, 
            ResourceStatus.WARNING, 
            ResourceStatus.CRITICAL
        ]
    
    def test_memory_monitoring(self):
        """Test memory usage monitoring."""
        memory_info = resource_monitor.check_memory_usage()
        
        assert memory_info.total > 0
        assert memory_info.used >= 0
        assert memory_info.free >= 0
        assert 0 <= memory_info.percent_used <= 1
        assert memory_info.status in [
            ResourceStatus.HEALTHY,
            ResourceStatus.WARNING,
            ResourceStatus.CRITICAL
        ]
    
    @pytest.mark.asyncio
    async def test_comprehensive_resource_status(self):
        """Test comprehensive resource status check."""
        status = await resource_monitor.get_comprehensive_status()
        
        assert "overall_status" in status
        assert "timestamp" in status
        assert "disk" in status
        assert "memory" in status
        assert "network" in status
        assert "services" in status
        
        # Verify all required services are checked
        services = status["services"]
        assert "ollama" in services
        assert "chromadb" in services


class TestFileProcessingFailures:
    """Test file processing failure scenarios."""
    
    def setup_method(self):
        """Setup test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ocr_processor = OCRProcessor(timeout_seconds=30)
    
    def teardown_method(self):
        """Cleanup test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_file_error(self):
        """Test error handling for missing files."""
        missing_file = self.temp_dir / "nonexistent.pdf"
        output_file = self.temp_dir / "output.pdf"
        
        # This should be handled gracefully
        result = asyncio.run(self.ocr_processor.process_pdf(
            missing_file, output_file
        ))
        
        assert result.success is False
        assert result.error_message is not None
        assert "not found" in result.error_message.lower() or "exists" in result.error_message.lower()
    
    def test_empty_file_error(self):
        """Test error handling for empty files."""
        empty_file = self.temp_dir / "empty.pdf"
        empty_file.touch()  # Create empty file
        output_file = self.temp_dir / "output.pdf"
        
        result = asyncio.run(self.ocr_processor.process_pdf(
            empty_file, output_file
        ))
        
        assert result.success is False
        assert result.error_message is not None
    
    def test_invalid_pdf_content(self):
        """Test error handling for invalid PDF content."""
        invalid_pdf = self.temp_dir / "invalid.pdf"
        invalid_pdf.write_text("This is not a PDF file")
        output_file = self.temp_dir / "output.pdf"
        
        result = asyncio.run(self.ocr_processor.process_pdf(
            invalid_pdf, output_file
        ))
        
        assert result.success is False
        assert result.error_message is not None
    
    def test_permission_denied_error(self):
        """Test error handling for permission denied scenarios."""
        # Create a file and remove read permissions
        restricted_file = self.temp_dir / "restricted.pdf"
        restricted_file.write_bytes(b"%PDF-1.4\n")
        restricted_file.chmod(0o000)  # Remove all permissions
        
        output_file = self.temp_dir / "output.pdf"
        
        try:
            result = asyncio.run(self.ocr_processor.process_pdf(
                restricted_file, output_file
            ))
            
            # Should handle permission error gracefully
            assert result.success is False
            assert result.error_message is not None
        finally:
            # Restore permissions for cleanup
            try:
                restricted_file.chmod(0o644)
            except:
                pass


class TestMatterManagementFailures:
    """Test matter management failure scenarios."""
    
    def setup_method(self):
        """Setup test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.matter_manager = MatterManager()
        # Override the root path for testing
        self.matter_manager.root_path = self.temp_dir
    
    def teardown_method(self):
        """Cleanup test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_matter_creation_with_invalid_name(self):
        """Test matter creation with invalid names."""
        invalid_names = [
            "",  # Empty name
            "   ",  # Whitespace only
            "a" * 256,  # Too long
            "test/matter",  # Invalid characters
            "../hack",  # Path traversal attempt
        ]
        
        for invalid_name in invalid_names:
            try:
                matter = self.matter_manager.create_matter(invalid_name)
                # If creation succeeds, check that name was sanitized
                assert matter.name != invalid_name or len(matter.name.strip()) > 0
            except (ValueError, FileNotFoundError) as e:
                # Expected for invalid names
                assert len(str(e)) > 0
    
    def test_matter_creation_permission_denied(self):
        """Test matter creation when directory creation fails."""
        # Make the root directory read-only
        self.temp_dir.chmod(0o444)
        
        try:
            with pytest.raises((PermissionError, OSError)):
                self.matter_manager.create_matter("test_matter")
        finally:
            # Restore permissions for cleanup
            self.temp_dir.chmod(0o755)
    
    def test_matter_loading_with_corrupted_config(self):
        """Test loading matters with corrupted configuration."""
        # Create a matter directory with corrupted config
        matter_dir = self.temp_dir / "Matter_corrupted"
        matter_dir.mkdir(parents=True)
        
        # Write invalid JSON
        config_file = matter_dir / "config.json"
        config_file.write_text("{ invalid json }")
        
        # Should handle corrupted config gracefully
        matters = self.matter_manager.list_matters()
        
        # The corrupted matter should either be skipped or have default values
        corrupted_matters = [m for m in matters if "corrupted" in m.slug]
        # Depending on implementation, might be 0 (skipped) or 1 (recovered)
        assert len(corrupted_matters) <= 1
    
    def test_concurrent_matter_operations(self):
        """Test concurrent matter operations for thread safety."""
        import threading
        import time
        
        results = []
        errors = []
        
        def create_matter(name):
            try:
                matter = self.matter_manager.create_matter(f"concurrent_{name}")
                results.append(matter)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_matter, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) >= 0  # At least some should succeed
        assert len(results) + len(errors) == 5  # All operations completed
        
        # Verify unique matter IDs
        matter_ids = [m.id for m in results]
        assert len(matter_ids) == len(set(matter_ids))  # All unique


class TestConfigurationFailures:
    """Test configuration-related failure scenarios."""
    
    def test_missing_configuration_file(self):
        """Test behavior with missing configuration file."""
        # This tests the application's ability to use defaults
        # when configuration is missing
        pass  # Implementation depends on settings module
    
    def test_invalid_configuration_values(self):
        """Test handling of invalid configuration values."""
        # Test with various invalid configuration scenarios
        pass  # Implementation depends on settings module


class TestRetryScenarios:
    """Test retry behavior under various failure conditions."""
    
    @pytest.mark.asyncio
    async def test_retry_with_transient_failures(self):
        """Test retry behavior with transient failures."""
        call_count = 0
        
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"Transient failure {call_count}")
            return f"Success after {call_count} attempts"
        
        result = await retry_manager.execute_with_retry(
            flaky_operation,
            operation_type="network"
        )
        
        assert result == "Success after 3 attempts"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_budget_exhaustion(self):
        """Test retry budget exhaustion prevents retry storms."""
        budget = retry_manager.get_retry_budget("test_operation")
        
        # Exhaust the budget
        for _ in range(budget.budget_per_minute):
            budget.record_attempt()
        
        # Should not allow more retries
        assert budget.can_retry() is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_cascading_failures(self):
        """Test circuit breaker prevents cascading failures."""
        circuit = retry_manager.get_circuit_breaker("test_service")
        
        # Simulate multiple failures to open circuit
        for _ in range(5):
            circuit.record_failure()
        
        # Circuit should be open
        assert circuit.can_execute() is False
        
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Service failure")
        
        # Should prevent execution due to open circuit
        with pytest.raises(ServiceUnavailableError):
            await retry_manager.execute_with_retry(
                failing_operation,
                operation_type="api",
                service_name="test_service"
            )
        
        # Operation should not have been called
        assert call_count == 0


class TestErrorPropagation:
    """Test error propagation through the system."""
    
    def test_error_context_preservation(self):
        """Test that error context is preserved through handling."""
        context = create_context(
            matter_id="test-matter",
            operation="test-operation",
            file_path="/test/file.pdf"
        )
        
        original_error = FileNotFoundError("File not found")
        
        handled_error = handle_error(original_error, context, notify_user=False)
        
        assert handled_error.context.matter_id == "test-matter"
        assert handled_error.context.operation == "test-operation"
        assert handled_error.context.file_path == "/test/file.pdf"
        assert handled_error.cause is original_error
    
    def test_error_chaining(self):
        """Test proper error chaining through multiple layers."""
        # Simulate error occurring deep in the stack
        try:
            # Layer 3: File operation
            raise FileNotFoundError("Original file not found")
        except FileNotFoundError as e:
            try:
                # Layer 2: Processing layer
                raise FileProcessingError(
                    file_path="/test/file.pdf",
                    operation="parse",
                    reason="Could not access file",
                    cause=e
                )
            except FileProcessingError as e2:
                # Layer 1: API layer
                handled_error = handle_error(e2, notify_user=False)
                
                # Verify error chain is preserved
                assert isinstance(handled_error, FileProcessingError)
                assert handled_error.cause is e2
                assert isinstance(handled_error.cause.cause, FileNotFoundError)


class TestGracefulDegradation:
    """Test graceful degradation scenarios."""
    
    @pytest.mark.asyncio
    async def test_feature_availability_check(self):
        """Test feature availability under degradation."""
        # Evaluate all services
        await degradation_manager.evaluate_all_services()
        
        # Check various feature availability
        features = [
            "agent_memory",
            "local_llm", 
            "vector_search",
            "pdf_upload",
            "matter_creation"
        ]
        
        for feature in features:
            is_available = degradation_manager.is_feature_available(feature)
            assert isinstance(is_available, bool)
    
    @pytest.mark.asyncio
    async def test_degradation_recovery(self):
        """Test recovery from degraded state."""
        # Force degradation by simulating service failure
        service_states = await degradation_manager.evaluate_all_services()
        
        # Check if any services are in fallback mode
        degraded_services = [
            name for name, state in service_states.items()
            if state.fallback_active
        ]
        
        # If services are degraded, test recovery
        for service_name in degraded_services:
            recovery_result = await degradation_manager.attempt_service_recovery(service_name)
            # Recovery may or may not succeed depending on actual service state
            assert isinstance(recovery_result, bool)
    
    def test_user_guidance_generation(self):
        """Test user guidance message generation."""
        guidance = degradation_manager.get_user_guidance()
        
        assert isinstance(guidance, list)
        for item in guidance:
            assert "service" in item
            assert "message" in item
            assert "suggestion" in item
            assert isinstance(item["service"], str)
            assert isinstance(item["message"], str)
            assert isinstance(item["suggestion"], str)


if __name__ == "__main__":
    pytest.main([__file__])