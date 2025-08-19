"""
Unit tests for Letta reliability features including circuit breaker,
request queue, connection management, and error handling.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.letta_circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitBreakerError,
    CircuitBreakerManager, with_circuit_breaker
)
from app.letta_request_queue import (
    RequestQueue, RequestType, RequestPriority,
    QueuedRequest, enqueue_letta_operation
)
from app.letta_connection import (
    ConnectionState, ConnectionMetrics, LettaConnectionManager
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        async def failing_operation():
            raise Exception("Test failure")
        
        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await cb.call(failing_operation)
        
        assert cb.state == CircuitState.OPEN
        assert cb.is_open
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_fails_fast_when_open(self):
        """Test circuit breaker fails immediately when open."""
        cb = CircuitBreaker("test", failure_threshold=2)
        
        async def failing_operation():
            raise Exception("Test failure")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call(failing_operation)
        
        # Should fail fast without calling operation
        with pytest.raises(CircuitBreakerError) as exc_info:
            await cb.call(failing_operation)
        
        assert "Circuit breaker 'test' is OPEN" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker transitions to half-open and recovers."""
        cb = CircuitBreaker("test", failure_threshold=2, success_threshold=2, timeout=0.1)
        
        call_count = 0
        
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Failure")
            return "Success"
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call(operation)
        
        assert cb.is_open
        
        # Wait for timeout
        await asyncio.sleep(0.15)
        
        # Should transition to half-open and succeed
        result = await cb.call(operation)
        assert result == "Success"
        assert cb.is_half_open
        
        # Another success should close the circuit
        result = await cb.call(operation)
        assert result == "Success"
        assert cb.is_closed
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics(self):
        """Test circuit breaker metrics collection."""
        cb = CircuitBreaker("test", failure_threshold=2)
        
        async def operation(should_fail=False):
            if should_fail:
                raise Exception("Failure")
            return "Success"
        
        # Some successful calls
        for _ in range(3):
            await cb.call(operation, should_fail=False)
        
        # Some failed calls
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call(operation, should_fail=True)
        
        metrics = cb.get_metrics()
        assert metrics["name"] == "test"
        assert metrics["state"] == CircuitState.OPEN.value
        assert metrics["metrics"]["successful_calls"] == 3
        assert metrics["metrics"]["failed_calls"] == 2
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_manager(self):
        """Test circuit breaker manager."""
        manager = CircuitBreakerManager()
        
        # Configure defaults
        manager.configure_defaults(failure_threshold=5, timeout=30.0)
        
        # Get or create breakers
        cb1 = manager.get_or_create("service1")
        cb2 = manager.get_or_create("service2", failure_threshold=3)
        
        assert cb1.failure_threshold == 5  # Uses default
        assert cb2.failure_threshold == 3  # Uses override
        
        # Get all metrics
        all_metrics = manager.get_all_metrics()
        assert "service1" in all_metrics
        assert "service2" in all_metrics
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        call_count = 0
        
        @with_circuit_breaker("test_operation", failure_threshold=2)
        async def protected_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Failure")
            return "Success"
        
        # Should fail twice and open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await protected_operation()
        
        # Should fail fast
        with pytest.raises(CircuitBreakerError):
            await protected_operation()


class TestRequestQueue:
    """Test request queue and batching functionality."""
    
    @pytest.mark.asyncio
    async def test_request_queue_enqueue_dequeue(self):
        """Test basic enqueue and processing."""
        queue = RequestQueue(batch_size=1, batch_timeout=0.1)
        await queue.start()
        
        async def operation(value):
            return value * 2
        
        try:
            future = await queue.enqueue(
                operation,
                RequestType.OTHER,
                priority=RequestPriority.NORMAL,
                value=5
            )
            
            result = await asyncio.wait_for(future, timeout=2.0)
            assert result == 10
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_request_queue_priority_ordering(self):
        """Test requests are processed by priority."""
        queue = RequestQueue(batch_size=1, batch_timeout=0.1)
        results = []
        
        async def operation(value):
            results.append(value)
            return value
        
        # Enqueue with different priorities (don't start queue yet)
        futures = []
        futures.append(await queue.enqueue(
            operation, RequestType.OTHER, RequestPriority.LOW, value="low"
        ))
        futures.append(await queue.enqueue(
            operation, RequestType.OTHER, RequestPriority.CRITICAL, value="critical"
        ))
        futures.append(await queue.enqueue(
            operation, RequestType.OTHER, RequestPriority.HIGH, value="high"
        ))
        futures.append(await queue.enqueue(
            operation, RequestType.OTHER, RequestPriority.NORMAL, value="normal"
        ))
        
        # Start processing
        await queue.start()
        
        try:
            # Wait for all to complete
            await asyncio.gather(*futures, return_exceptions=True)
            
            # Check order (critical should be first)
            assert results[0] == "critical"
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_request_queue_batching(self):
        """Test request batching for similar operations."""
        queue = RequestQueue(batch_size=3, batch_timeout=0.5)
        await queue.start()
        
        call_count = 0
        
        async def operation(value):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate work
            return value
        
        try:
            # Enqueue multiple similar requests quickly
            futures = []
            for i in range(3):
                future = await queue.enqueue(
                    operation,
                    RequestType.MEMORY_RECALL,
                    priority=RequestPriority.NORMAL,
                    value=i
                )
                futures.append(future)
            
            # Wait for completion
            results = await asyncio.gather(*futures)
            assert results == [0, 1, 2]
            
            # Should process as batch (metrics will show batching)
            metrics = queue.get_metrics()
            assert metrics["metrics"]["processed_requests"] >= 3
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_request_queue_timeout(self):
        """Test request timeout handling."""
        queue = RequestQueue(batch_size=1, batch_timeout=0.1)
        
        async def slow_operation():
            await asyncio.sleep(5)  # Longer than timeout
            return "Done"
        
        # Enqueue with short timeout (don't start queue)
        future = await queue.enqueue(
            slow_operation,
            RequestType.OTHER,
            timeout=0.1  # Very short timeout
        )
        
        # Start queue and let timeout occur
        await queue.start()
        await asyncio.sleep(0.2)
        
        # Should have timed out
        assert future.done()
        with pytest.raises(asyncio.TimeoutError):
            future.result()
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_request_queue_overflow_drop_oldest(self):
        """Test queue overflow with drop_oldest policy."""
        queue = RequestQueue(max_queue_size=3, overflow_policy="drop_oldest")
        
        async def operation(value):
            return value
        
        # Fill queue beyond capacity
        futures = []
        for i in range(5):
            future = await queue.enqueue(
                operation,
                RequestType.OTHER,
                priority=RequestPriority.NORMAL,
                value=i
            )
            futures.append(future)
        
        # First two should be dropped
        assert futures[0].done()  # Dropped
        assert futures[1].done()  # Dropped
        
        with pytest.raises(Exception):
            futures[0].result()  # Should have exception
    
    @pytest.mark.asyncio
    async def test_request_queue_deduplication(self):
        """Test request deduplication."""
        queue = RequestQueue()
        await queue.start()
        
        async def operation():
            return "Result"
        
        try:
            # Enqueue same request twice with same dedup key
            future1 = await queue.enqueue(
                operation,
                RequestType.OTHER,
                dedup_key="same-key"
            )
            future2 = await queue.enqueue(
                operation,
                RequestType.OTHER,
                dedup_key="same-key"
            )
            
            # Second should be deduplicated
            result2 = await future2
            assert result2 is None  # Deduplicated requests return None
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_request_queue_metrics(self):
        """Test queue metrics collection."""
        queue = RequestQueue()
        await queue.start()
        
        async def operation(value):
            return value
        
        try:
            # Process some requests
            futures = []
            for i in range(5):
                future = await queue.enqueue(
                    operation,
                    RequestType.MEMORY_RECALL,
                    value=i
                )
                futures.append(future)
            
            await asyncio.gather(*futures)
            
            # Check metrics
            metrics = queue.get_metrics()
            assert metrics["metrics"]["total_requests"] == 5
            assert metrics["metrics"]["processed_requests"] == 5
            assert metrics["metrics"]["dropped_requests"] == 0
        finally:
            await queue.stop()


class TestConnectionManagement:
    """Test connection management and retry logic."""
    
    @pytest.mark.asyncio
    async def test_connection_retry_with_backoff(self):
        """Test connection retry with exponential backoff."""
        # Mock the AsyncLetta client
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # First 2 attempts fail, third succeeds
            mock_client.health.check.side_effect = [
                Exception("Connection failed"),
                Exception("Connection failed"),
                None  # Success
            ]
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            
            # Should retry and eventually succeed
            success = await manager.connect()
            assert success
            assert mock_client.health.check.call_count == 3
    
    @pytest.mark.asyncio
    async def test_connection_health_monitoring(self):
        """Test periodic health monitoring."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.health.check.return_value = None
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            
            # Connect and start health monitoring
            await manager.connect()
            manager.start_health_monitoring(interval=0.1)
            
            # Wait for a few health checks
            await asyncio.sleep(0.3)
            
            # Should have performed health checks
            assert mock_client.health.check.call_count >= 3
            
            # Stop monitoring
            manager.stop_health_monitoring()
    
    @pytest.mark.asyncio
    async def test_connection_metrics_collection(self):
        """Test connection metrics collection."""
        metrics = ConnectionMetrics(window_size=10)
        
        # Record some operations
        metrics.record_operation("health_check", 0.01, success=True)
        metrics.record_operation("recall", 0.5, success=True)
        metrics.record_operation("recall", 0.3, success=False)
        
        # Check metrics
        assert metrics.success_count == 2
        assert metrics.failure_count == 1
        assert metrics.get_success_rate() == pytest.approx(66.67, rel=0.01)
        assert metrics.get_average_latency("health_check") == 0.01
        assert metrics.get_average_latency("recall") == 0.4
    
    @pytest.mark.asyncio
    async def test_connection_state_transitions(self):
        """Test connection state transitions."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            
            # Initial state
            assert manager.state == ConnectionState.DISCONNECTED
            
            # Connecting
            mock_client.health.check.return_value = None
            await manager.connect()
            assert manager.state == ConnectionState.CONNECTED
            
            # Simulate failure
            mock_client.health.check.side_effect = Exception("Lost connection")
            await manager._check_health()
            assert manager.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_connection_execute_with_retry(self):
        """Test execute_with_retry wrapper."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            await manager.connect()
            
            # Operation that fails once then succeeds
            operation = AsyncMock()
            operation.side_effect = [
                Exception("Temporary failure"),
                "Success"
            ]
            
            result = await manager.execute_with_retry(
                operation,
                "test_operation",
                max_retries=2
            )
            
            assert result == "Success"
            assert operation.call_count == 2


class TestFailureScenarios:
    """Test various failure scenarios and recovery."""
    
    @pytest.mark.asyncio
    async def test_server_unavailable_fallback(self):
        """Test fallback mode when server is unavailable."""
        with patch('app.letta_connection.LETTA_AVAILABLE', False):
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            
            # Should enter fallback mode
            await manager.connect()
            assert manager.state == ConnectionState.FALLBACK
            
            # Operations should return None in fallback
            result = await manager.execute_with_retry(
                AsyncMock(return_value="data"),
                "test_op"
            )
            assert result is None
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test that resources are properly cleaned up."""
        queue = RequestQueue(max_queue_size=100)
        
        async def operation():
            return "Result"
        
        # Process many requests
        for _ in range(100):
            future = await queue.enqueue(operation, RequestType.OTHER)
            # Don't await, let them accumulate
        
        # Clear queue - should clean up futures
        await queue.clear()
        
        # Metrics should show cleared state
        metrics = queue.get_metrics()
        assert metrics["queue_size"] == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test handling of concurrent operations."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.health.check.return_value = None
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            await manager.connect()
            
            # Simulate concurrent operations
            async def operation(id):
                await asyncio.sleep(0.01)  # Simulate work
                return f"Result {id}"
            
            # Launch multiple concurrent operations
            tasks = []
            for i in range(10):
                task = manager.execute_with_retry(operation, f"op_{i}", i)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All should complete successfully
            assert len(results) == 10
            assert all(f"Result {i}" in results for i in range(10))
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown of all components."""
        queue = RequestQueue()
        await queue.start()
        
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            await manager.connect()
            manager.start_health_monitoring()
            
            # Enqueue some operations
            async def operation():
                await asyncio.sleep(0.1)
                return "Done"
            
            futures = []
            for _ in range(5):
                future = await queue.enqueue(operation, RequestType.OTHER)
                futures.append(future)
            
            # Graceful shutdown
            await queue.stop()
            manager.stop_health_monitoring()
            
            # Check that everything stopped cleanly
            assert not queue._processing
            assert manager._health_task is None or manager._health_task.done()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])