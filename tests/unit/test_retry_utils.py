"""
Unit tests for retry utilities.

Tests retry policies, backoff strategies, circuit breakers, and retry manager functionality.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.retry_utils import (
    RetryPolicy, BackoffStrategy, CircuitBreakerConfig, CircuitBreaker,
    CircuitState, RetryBudget, RetryManager, retry_manager,
    with_retry, retry_with_backoff
)
from app.error_handler import RetryableError, ServiceUnavailableError


class TestRetryPolicy:
    """Test RetryPolicy configuration."""
    
    def test_default_policy(self):
        """Test default retry policy creation."""
        policy = RetryPolicy()
        
        assert policy.max_attempts == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 300.0
        assert policy.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER
        assert policy.multiplier == 2.0
        assert policy.jitter_range == 0.1
        assert policy.timeout is None
        assert policy.retryable_exceptions == []
        assert policy.non_retryable_exceptions == []
    
    def test_custom_policy(self):
        """Test custom retry policy creation."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=2.0,
            max_delay=60.0,
            backoff_strategy=BackoffStrategy.LINEAR,
            multiplier=1.5,
            timeout=30.0,
            retryable_exceptions=[ConnectionError],
            non_retryable_exceptions=[ValueError]
        )
        
        assert policy.max_attempts == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 60.0
        assert policy.backoff_strategy == BackoffStrategy.LINEAR
        assert policy.multiplier == 1.5
        assert policy.timeout == 30.0
        assert ConnectionError in policy.retryable_exceptions
        assert ValueError in policy.non_retryable_exceptions


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def setup_method(self):
        """Setup test method."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            success_threshold=2
        )
        self.circuit = CircuitBreaker("test_service", self.config)
    
    def test_initial_state(self):
        """Test circuit breaker initial state."""
        assert self.circuit.state == CircuitState.CLOSED
        assert self.circuit.failure_count == 0
        assert self.circuit.success_count == 0
        assert self.circuit.can_execute() is True
    
    def test_failure_threshold(self):
        """Test circuit opens after failure threshold."""
        # Record failures up to threshold
        for i in range(self.config.failure_threshold):
            self.circuit.record_failure()
            if i < self.config.failure_threshold - 1:
                assert self.circuit.state == CircuitState.CLOSED
        
        # Should be open after threshold
        assert self.circuit.state == CircuitState.OPEN
        assert self.circuit.can_execute() is False
    
    def test_half_open_transition(self):
        """Test transition to half-open state."""
        # Open the circuit
        for _ in range(self.config.failure_threshold):
            self.circuit.record_failure()
        assert self.circuit.state == CircuitState.OPEN
        
        # Manually set next attempt time to past
        self.circuit.next_attempt_time = datetime.utcnow() - timedelta(seconds=1)
        
        # Should transition to half-open
        assert self.circuit.can_execute() is True
        assert self.circuit.state == CircuitState.HALF_OPEN
    
    def test_recovery_from_half_open(self):
        """Test recovery from half-open to closed."""
        # Set to half-open state
        self.circuit.state = CircuitState.HALF_OPEN
        
        # Record successful executions
        for i in range(self.config.success_threshold):
            self.circuit.record_success()
            if i < self.config.success_threshold - 1:
                assert self.circuit.state == CircuitState.HALF_OPEN
        
        # Should be closed after success threshold
        assert self.circuit.state == CircuitState.CLOSED
        assert self.circuit.failure_count == 0
    
    def test_half_open_failure(self):
        """Test failure in half-open state reopens circuit."""
        # Set to half-open state
        self.circuit.state = CircuitState.HALF_OPEN
        
        # Record failure
        self.circuit.record_failure()
        
        # Should reopen
        assert self.circuit.state == CircuitState.OPEN
        assert self.circuit.can_execute() is False
    
    def test_status_reporting(self):
        """Test circuit breaker status reporting."""
        self.circuit.record_failure()
        status = self.circuit.get_status()
        
        assert status["name"] == "test_service"
        assert status["state"] == "closed"
        assert status["failure_count"] == 1
        assert status["success_count"] == 0


class TestRetryBudget:
    """Test RetryBudget functionality."""
    
    def setup_method(self):
        """Setup test method."""
        self.budget = RetryBudget(budget_per_minute=5)
    
    def test_initial_budget(self):
        """Test initial budget state."""
        assert self.budget.can_retry() is True
        assert len(self.budget.attempts_history) == 0
    
    def test_budget_exhaustion(self):
        """Test budget exhaustion."""
        # Use up the budget
        for _ in range(5):
            assert self.budget.can_retry() is True
            self.budget.record_attempt()
        
        # Should be exhausted
        assert self.budget.can_retry() is False
    
    def test_budget_recovery(self):
        """Test budget recovery after time window."""
        # Exhaust budget
        for _ in range(5):
            self.budget.record_attempt()
        assert self.budget.can_retry() is False
        
        # Manually set old timestamps
        old_time = datetime.utcnow() - timedelta(minutes=2)
        self.budget.attempts_history = [old_time] * 5
        
        # Should recover
        assert self.budget.can_retry() is True


class TestRetryManager:
    """Test RetryManager functionality."""
    
    def setup_method(self):
        """Setup test method."""
        self.manager = RetryManager()
    
    def test_default_policies(self):
        """Test default retry policies are set up."""
        policies = ["network", "api", "file", "database", "llm"]
        
        for policy_name in policies:
            policy = self.manager.get_policy(policy_name)
            assert isinstance(policy, RetryPolicy)
            assert policy.max_attempts > 0
    
    def test_custom_policy_setting(self):
        """Test setting custom retry policy."""
        custom_policy = RetryPolicy(max_attempts=10, base_delay=5.0)
        
        self.manager.set_policy("custom", custom_policy)
        retrieved = self.manager.get_policy("custom")
        
        assert retrieved.max_attempts == 10
        assert retrieved.base_delay == 5.0
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and retrieval."""
        cb1 = self.manager.get_circuit_breaker("service1")
        cb2 = self.manager.get_circuit_breaker("service1")  # Same service
        cb3 = self.manager.get_circuit_breaker("service2")  # Different service
        
        assert cb1 is cb2  # Same instance
        assert cb1 is not cb3  # Different instances
        assert cb1.name == "service1"
        assert cb3.name == "service2"
    
    def test_retry_budget_creation(self):
        """Test retry budget creation and retrieval."""
        budget1 = self.manager.get_retry_budget("operation1")
        budget2 = self.manager.get_retry_budget("operation1")  # Same operation
        budget3 = self.manager.get_retry_budget("operation2")  # Different operation
        
        assert budget1 is budget2  # Same instance
        assert budget1 is not budget3  # Different instances
    
    def test_calculate_delay_fixed(self):
        """Test fixed backoff delay calculation."""
        policy = RetryPolicy(
            base_delay=2.0,
            backoff_strategy=BackoffStrategy.FIXED
        )
        
        delay1 = self.manager.calculate_delay(1, policy)
        delay2 = self.manager.calculate_delay(3, policy)
        delay3 = self.manager.calculate_delay(5, policy)
        
        assert delay1 == 2.0
        assert delay2 == 2.0
        assert delay3 == 2.0
    
    def test_calculate_delay_linear(self):
        """Test linear backoff delay calculation."""
        policy = RetryPolicy(
            base_delay=1.0,
            backoff_strategy=BackoffStrategy.LINEAR
        )
        
        delay1 = self.manager.calculate_delay(1, policy)
        delay2 = self.manager.calculate_delay(2, policy)
        delay3 = self.manager.calculate_delay(3, policy)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 3.0
    
    def test_calculate_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(
            base_delay=1.0,
            multiplier=2.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        
        delay1 = self.manager.calculate_delay(1, policy)
        delay2 = self.manager.calculate_delay(2, policy)
        delay3 = self.manager.calculate_delay(3, policy)
        
        assert delay1 == 1.0  # 1.0 * 2^0
        assert delay2 == 2.0  # 1.0 * 2^1
        assert delay3 == 4.0  # 1.0 * 2^2
    
    def test_calculate_delay_max_limit(self):
        """Test delay calculation respects maximum limit."""
        policy = RetryPolicy(
            base_delay=10.0,
            max_delay=20.0,
            multiplier=3.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        
        delay1 = self.manager.calculate_delay(1, policy)
        delay2 = self.manager.calculate_delay(2, policy)
        delay3 = self.manager.calculate_delay(3, policy)
        
        assert delay1 == 10.0  # 10.0 * 3^0 = 10.0
        assert delay2 == 20.0  # min(30.0, 20.0) = 20.0
        assert delay3 == 20.0  # min(90.0, 20.0) = 20.0
    
    def test_should_retry_retryable_exceptions(self):
        """Test should_retry with retryable exceptions."""
        policy = RetryPolicy(
            max_attempts=3,
            retryable_exceptions=[ConnectionError, TimeoutError]
        )
        
        # Retryable exceptions
        assert self.manager.should_retry(ConnectionError(), 1, policy) is True
        assert self.manager.should_retry(TimeoutError(), 2, policy) is True
        
        # Non-retryable exception
        assert self.manager.should_retry(ValueError(), 1, policy) is False
        
        # Max attempts exceeded
        assert self.manager.should_retry(ConnectionError(), 3, policy) is False
    
    def test_should_retry_non_retryable_exceptions(self):
        """Test should_retry with non-retryable exceptions."""
        policy = RetryPolicy(
            max_attempts=3,
            non_retryable_exceptions=[ValueError]
        )
        
        # Non-retryable exception
        assert self.manager.should_retry(ValueError(), 1, policy) is False
        
        # Other exceptions should be retryable
        assert self.manager.should_retry(ConnectionError(), 1, policy) is True
    
    def test_status_reporting(self):
        """Test retry manager status reporting."""
        # Create some circuit breakers and budgets
        self.manager.get_circuit_breaker("service1")
        self.manager.get_retry_budget("operation1")
        
        status = self.manager.get_status()
        
        assert "circuit_breakers" in status
        assert "retry_policies" in status
        assert "retry_budgets" in status
        assert "service1" in status["circuit_breakers"]
        assert "operation1" in status["retry_budgets"]


@pytest.mark.asyncio
class TestAsyncRetryExecution:
    """Test async retry execution."""
    
    def setup_method(self):
        """Setup test method."""
        self.manager = RetryManager()
        self.call_count = 0
    
    async def failing_function(self, fail_times: int = 2):
        """Test function that fails specified number of times."""
        self.call_count += 1
        if self.call_count <= fail_times:
            raise ConnectionError(f"Failure {self.call_count}")
        return f"Success after {self.call_count} attempts"
    
    async def always_failing_function(self):
        """Test function that always fails."""
        self.call_count += 1
        raise ConnectionError(f"Always fails - attempt {self.call_count}")
    
    async def test_successful_retry(self):
        """Test successful operation after retries."""
        self.call_count = 0
        
        result = await self.manager.execute_with_retry(
            self.failing_function,
            operation_type="api",
            fail_times=2  # Will succeed on 3rd attempt
        )
        
        assert result == "Success after 3 attempts"
        assert self.call_count == 3
    
    async def test_retry_exhaustion(self):
        """Test retry exhaustion with persistent failure."""
        self.call_count = 0
        
        with pytest.raises(ConnectionError):
            await self.manager.execute_with_retry(
                self.always_failing_function,
                operation_type="api"
            )
        
        # Should try max_attempts times
        policy = self.manager.get_policy("api")
        assert self.call_count == policy.max_attempts
    
    async def test_circuit_breaker_blocking(self):
        """Test circuit breaker prevents execution when open."""
        # Open the circuit breaker
        circuit = self.manager.get_circuit_breaker("test_service")
        circuit.state = CircuitState.OPEN
        circuit.next_attempt_time = datetime.utcnow() + timedelta(minutes=1)
        
        self.call_count = 0
        
        with pytest.raises(ServiceUnavailableError):
            await self.manager.execute_with_retry(
                self.failing_function,
                operation_type="api",
                service_name="test_service"
            )
        
        # Function should not have been called
        assert self.call_count == 0
    
    async def test_timeout_handling(self):
        """Test timeout handling during retry execution."""
        async def slow_function():
            await asyncio.sleep(2.0)  # Longer than timeout
            return "Success"
        
        policy = RetryPolicy(timeout=0.1)  # Very short timeout
        self.manager.set_policy("slow", policy)
        
        with pytest.raises(asyncio.TimeoutError):
            await self.manager.execute_with_retry(
                slow_function,
                operation_type="slow"
            )
    
    async def test_non_retryable_error(self):
        """Test non-retryable error stops retry immediately."""
        async def function_with_non_retryable_error():
            self.call_count += 1
            raise ValueError("Non-retryable error")
        
        self.call_count = 0
        
        with pytest.raises(ValueError):
            await self.manager.execute_with_retry(
                function_with_non_retryable_error,
                operation_type="api"
            )
        
        # Should only be called once
        assert self.call_count == 1


@pytest.mark.asyncio
class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    def setup_method(self):
        """Setup test method."""
        self.call_count = 0
    
    async def test_with_retry_decorator(self):
        """Test @with_retry decorator."""
        @with_retry(operation_type="api")
        async def decorated_function(fail_times: int = 2):
            self.call_count += 1
            if self.call_count <= fail_times:
                raise ConnectionError(f"Failure {self.call_count}")
            return f"Success after {self.call_count} attempts"
        
        self.call_count = 0
        result = await decorated_function(fail_times=2)
        
        assert result == "Success after 3 attempts"
        assert self.call_count == 3
    
    async def test_retry_with_backoff_function(self):
        """Test retry_with_backoff utility function."""
        async def failing_function(fail_times: int = 1):
            self.call_count += 1
            if self.call_count <= fail_times:
                raise ConnectionError(f"Failure {self.call_count}")
            return f"Success after {self.call_count} attempts"
        
        self.call_count = 0
        result = await retry_with_backoff(
            failing_function,
            max_attempts=3,
            base_delay=0.1,
            backoff_strategy=BackoffStrategy.FIXED,
            fail_times=1
        )
        
        assert result == "Success after 2 attempts"
        assert self.call_count == 2


class TestRetryIntegration:
    """Test retry system integration with other components."""
    
    def test_global_retry_manager(self):
        """Test global retry manager instance."""
        # Test that global instance is accessible
        assert retry_manager is not None
        assert isinstance(retry_manager, RetryManager)
        
        # Test that it has default policies
        api_policy = retry_manager.get_policy("api")
        assert api_policy.max_attempts > 0
    
    def test_retry_with_error_handling(self):
        """Test retry integration with error handling."""
        # This would test integration with error_handler module
        # but we'll keep it simple for unit tests
        
        policy = RetryPolicy(
            max_attempts=2,
            retryable_exceptions=[RetryableError]
        )
        
        manager = RetryManager()
        
        # RetryableError should be retryable
        retryable_error = RetryableError("Test error")
        assert manager.should_retry(retryable_error, 1, policy) is True
        
        # Other errors should follow default rules
        connection_error = ConnectionError("Connection failed")
        assert manager.should_retry(connection_error, 1, policy) is True


if __name__ == "__main__":
    pytest.main([__file__])