"""
Circuit Breaker Pattern for Letta Operations

Implements a circuit breaker to prevent cascading failures when the Letta server
is experiencing issues. The circuit breaker has three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Failures exceeded threshold, requests fail immediately
- HALF_OPEN: Testing if service has recovered

Features:
- Configurable failure thresholds and timeout periods
- Automatic state transitions based on success/failure patterns
- Metrics collection for monitoring circuit state
- Per-operation type circuit breakers for granular control
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable, TypeVar, Coroutine
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import random

from .logging_conf import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerMetrics:
    """Tracks circuit breaker metrics."""
    
    def __init__(self):
        self.state_changes: list[tuple[CircuitState, datetime]] = []
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.half_open_successes = 0
        self.half_open_failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
    
    def record_state_change(self, new_state: CircuitState):
        """Record a state change."""
        self.state_changes.append((new_state, datetime.now()))
        if len(self.state_changes) > 100:  # Keep last 100 changes
            self.state_changes = self.state_changes[-100:]
    
    def record_success(self):
        """Record a successful call."""
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
    
    def record_failure(self):
        """Record a failed call."""
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now()
    
    def record_rejection(self):
        """Record a rejected call (circuit open)."""
        self.rejected_calls += 1
    
    def get_failure_rate(self) -> float:
        """Calculate failure rate."""
        total = self.successful_calls + self.failed_calls
        if total == 0:
            return 0.0
        return (self.failed_calls / total) * 100
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "failure_rate": self.get_failure_rate(),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success": self.last_success_time.isoformat() if self.last_success_time else None,
            "state_changes": len(self.state_changes)
        }


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against cascading failures.
    
    Args:
        name: Circuit breaker name for logging
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open before closing
        timeout: Seconds to wait before attempting recovery (half-open)
        expected_exception: Exception types to count as failures
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time >= self.timeout
    
    async def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        if self._state != new_state:
            logger.info(
                f"Circuit breaker '{self.name}' transitioning from {self._state.value} to {new_state.value}"
            )
            self._state = new_state
            self._metrics.record_state_change(new_state)
            
            # Reset counters on state change
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._success_count = 0
    
    async def call(self, func: Callable[..., Coroutine[Any, Any, T]], *args, **kwargs) -> T:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If func raises an exception
        """
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self.is_open and self._should_attempt_reset():
                await self._transition_to(CircuitState.HALF_OPEN)
            
            # Fail fast if circuit is open
            if self.is_open:
                self._metrics.record_rejection()
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. Service unavailable."
                )
        
        # Try to execute the function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.expected_exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self._metrics.record_success()
            
            if self.is_half_open:
                self._success_count += 1
                self._metrics.half_open_successes += 1
                
                if self._success_count >= self.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
            elif self.is_closed:
                # Reset failure count on success in closed state
                self._failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self._metrics.record_failure()
            self._last_failure_time = time.time()
            
            if self.is_half_open:
                self._metrics.half_open_failures += 1
                await self._transition_to(CircuitState.OPEN)
            elif self.is_closed:
                self._failure_count += 1
                
                if self._failure_count >= self.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "metrics": self._metrics.get_stats()
        }
    
    async def reset(self):
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            logger.info(f"Manually resetting circuit breaker '{self.name}'")
            await self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different operation types.
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = {
            "failure_threshold": 5,
            "success_threshold": 2,
            "timeout": 60.0
        }
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: Optional[int] = None,
        success_threshold: Optional[int] = None,
        timeout: Optional[float] = None,
        expected_exception: type = Exception
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for an operation type."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold or self._default_config["failure_threshold"],
                success_threshold=success_threshold or self._default_config["success_threshold"],
                timeout=timeout or self._default_config["timeout"],
                expected_exception=expected_exception
            )
        return self._breakers[name]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers."""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
    
    def configure_defaults(
        self,
        failure_threshold: Optional[int] = None,
        success_threshold: Optional[int] = None,
        timeout: Optional[float] = None
    ):
        """Configure default settings for new circuit breakers."""
        if failure_threshold is not None:
            self._default_config["failure_threshold"] = failure_threshold
        if success_threshold is not None:
            self._default_config["success_threshold"] = success_threshold
        if timeout is not None:
            self._default_config["timeout"] = timeout


# Global circuit breaker manager instance
circuit_manager = CircuitBreakerManager()


def with_circuit_breaker(
    name: str,
    failure_threshold: Optional[int] = None,
    success_threshold: Optional[int] = None,
    timeout: Optional[float] = None,
    expected_exception: type = Exception
):
    """
    Decorator to wrap async functions with circuit breaker protection.
    
    Example:
        @with_circuit_breaker("letta_memory_recall", failure_threshold=3)
        async def recall_memory(query: str):
            # ... implementation ...
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            breaker = circuit_manager.get_or_create(
                name=name,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout,
                expected_exception=expected_exception
            )
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator