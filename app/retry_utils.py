"""
Retry utilities with exponential backoff, circuit breaker patterns, and retry policies.

Provides robust retry mechanisms for handling transient failures across different
services and operation types with configurable policies and backoff strategies.
"""

import asyncio
import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import functools

from .logging_conf import get_logger
from .error_handler import BaseApplicationError, RetryableError, ServiceUnavailableError

logger = get_logger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategy types."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 300.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    multiplier: float = 2.0
    jitter_range: float = 0.1
    timeout: Optional[float] = None
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # For half-open state
    timeout: float = 30.0


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt: int
    delay: float
    exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RetryBudget:
    """
    Retry budget to prevent retry storms.
    """
    
    def __init__(self, budget_per_minute: int = 100):
        self.budget_per_minute = budget_per_minute
        self.attempts_history: List[datetime] = []
    
    def can_retry(self) -> bool:
        """Check if retry is within budget."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        
        # Remove old attempts
        self.attempts_history = [
            attempt for attempt in self.attempts_history
            if attempt > cutoff
        ]
        
        return len(self.attempts_history) < self.budget_per_minute
    
    def record_attempt(self):
        """Record a retry attempt."""
        self.attempts_history.append(datetime.utcnow())


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        now = datetime.utcnow()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if (self.next_attempt_time and 
                now >= self.next_attempt_time):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} transitioning to half-open")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """Record a successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} closed (recovered)")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.next_attempt_time = (
                    datetime.utcnow() + 
                    timedelta(seconds=self.config.recovery_timeout)
                )
                logger.warning(
                    f"Circuit breaker {self.name} opened",
                    failure_count=self.failure_count,
                    recovery_timeout=self.config.recovery_timeout
                )
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.next_attempt_time = (
                datetime.utcnow() + 
                timedelta(seconds=self.config.recovery_timeout)
            )
            logger.warning(f"Circuit breaker {self.name} reopened")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "next_attempt_time": self.next_attempt_time.isoformat() if self.next_attempt_time else None
        }


class RetryManager:
    """
    Manages retry policies and circuit breakers for different services.
    """
    
    def __init__(self):
        self.policies: Dict[str, RetryPolicy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_budgets: Dict[str, RetryBudget] = {}
        self._setup_default_policies()
    
    def _setup_default_policies(self):
        """Setup default retry policies for different operation types."""
        
        # Network operations
        self.policies["network"] = RetryPolicy(
            max_attempts=5,
            base_delay=1.0,
            max_delay=60.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            retryable_exceptions=[
                ConnectionError, TimeoutError, OSError
            ]
        )
        
        # API calls
        self.policies["api"] = RetryPolicy(
            max_attempts=3,
            base_delay=2.0,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            retryable_exceptions=[
                ConnectionError, TimeoutError, RetryableError
            ]
        )
        
        # File operations
        self.policies["file"] = RetryPolicy(
            max_attempts=3,
            base_delay=0.5,
            max_delay=10.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            retryable_exceptions=[
                PermissionError, OSError
            ],
            non_retryable_exceptions=[
                FileNotFoundError, IsADirectoryError
            ]
        )
        
        # Database operations
        self.policies["database"] = RetryPolicy(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            retryable_exceptions=[
                ConnectionError, TimeoutError
            ]
        )
        
        # LLM operations
        self.policies["llm"] = RetryPolicy(
            max_attempts=3,
            base_delay=5.0,
            max_delay=120.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            retryable_exceptions=[
                ConnectionError, TimeoutError, RetryableError
            ]
        )
    
    def get_policy(self, operation_type: str) -> RetryPolicy:
        """Get retry policy for operation type."""
        return self.policies.get(operation_type, self.policies["api"])
    
    def set_policy(self, operation_type: str, policy: RetryPolicy):
        """Set retry policy for operation type."""
        self.policies[operation_type] = policy
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            config = CircuitBreakerConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(service_name, config)
        return self.circuit_breakers[service_name]
    
    def get_retry_budget(self, operation_type: str) -> RetryBudget:
        """Get or create retry budget for operation type."""
        if operation_type not in self.retry_budgets:
            self.retry_budgets[operation_type] = RetryBudget()
        return self.retry_budgets[operation_type]
    
    def calculate_delay(
        self,
        attempt: int,
        policy: RetryPolicy
    ) -> float:
        """Calculate delay for retry attempt."""
        if policy.backoff_strategy == BackoffStrategy.FIXED:
            delay = policy.base_delay
        elif policy.backoff_strategy == BackoffStrategy.LINEAR:
            delay = policy.base_delay * attempt
        elif policy.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = policy.base_delay * (policy.multiplier ** (attempt - 1))
        elif policy.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            base_delay = policy.base_delay * (policy.multiplier ** (attempt - 1))
            jitter = base_delay * policy.jitter_range * (random.random() * 2 - 1)
            delay = base_delay + jitter
        else:
            delay = policy.base_delay
        
        return min(delay, policy.max_delay)
    
    def should_retry(
        self,
        exception: Exception,
        attempt: int,
        policy: RetryPolicy
    ) -> bool:
        """Determine if exception should be retried."""
        if attempt >= policy.max_attempts:
            return False
        
        # Check non-retryable exceptions first
        for exc_type in policy.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retryable exceptions
        if policy.retryable_exceptions:
            for exc_type in policy.retryable_exceptions:
                if isinstance(exception, exc_type):
                    return True
            return False
        
        # Default: retry RetryableError and common transient errors
        return isinstance(exception, (
            RetryableError,
            ConnectionError,
            TimeoutError,
            OSError
        ))
    
    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        operation_type: str = "api",
        service_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic and circuit breaker.
        
        Args:
            func: Async function to execute
            operation_type: Type of operation for policy selection
            service_name: Service name for circuit breaker
            context: Additional context for logging
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries exhausted
        """
        policy = self.get_policy(operation_type)
        circuit_breaker = self.get_circuit_breaker(service_name) if service_name else None
        retry_budget = self.get_retry_budget(operation_type)
        
        attempts: List[RetryAttempt] = []
        last_exception: Optional[Exception] = None
        
        for attempt in range(1, policy.max_attempts + 1):
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute():
                raise ServiceUnavailableError(
                    service=service_name,
                    user_message=f"Service {service_name} is temporarily unavailable",
                    suggestion="Please try again later or switch to an alternative service"
                )
            
            # Check retry budget
            if attempt > 1 and not retry_budget.can_retry():
                logger.warning(
                    "Retry budget exhausted",
                    operation_type=operation_type,
                    attempt=attempt
                )
                break
            
            try:
                # Execute function with timeout if specified
                if policy.timeout:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=policy.timeout
                    )
                else:
                    result = await func(*args, **kwargs)
                
                # Record success
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                # Log successful retry
                if attempt > 1:
                    logger.info(
                        "Operation succeeded after retry",
                        operation_type=operation_type,
                        attempt=attempt,
                        total_attempts=len(attempts) + 1,
                        context=context
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                attempts.append(RetryAttempt(
                    attempt=attempt,
                    delay=0,  # Will be calculated below
                    exception=e
                ))
                
                # Record failure in circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_failure()
                
                # Check if we should retry
                if not self.should_retry(e, attempt, policy):
                    logger.warning(
                        "Non-retryable error encountered",
                        operation_type=operation_type,
                        attempt=attempt,
                        error=str(e),
                        error_type=type(e).__name__,
                        context=context
                    )
                    break
                
                # Don't delay after the last attempt
                if attempt < policy.max_attempts:
                    delay = self.calculate_delay(attempt, policy)
                    attempts[-1].delay = delay
                    
                    # Record retry attempt for budget
                    retry_budget.record_attempt()
                    
                    logger.warning(
                        "Operation failed, retrying",
                        operation_type=operation_type,
                        attempt=attempt,
                        max_attempts=policy.max_attempts,
                        delay=delay,
                        error=str(e),
                        error_type=type(e).__name__,
                        context=context
                    )
                    
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        logger.error(
            "All retry attempts exhausted",
            operation_type=operation_type,
            total_attempts=len(attempts),
            last_error=str(last_exception),
            context=context
        )
        
        # Raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry loop completed without result or exception")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers and retry statistics."""
        return {
            "circuit_breakers": {
                name: cb.get_status()
                for name, cb in self.circuit_breakers.items()
            },
            "retry_policies": {
                name: {
                    "max_attempts": policy.max_attempts,
                    "base_delay": policy.base_delay,
                    "max_delay": policy.max_delay,
                    "backoff_strategy": policy.backoff_strategy.value
                }
                for name, policy in self.policies.items()
            },
            "retry_budgets": {
                name: {
                    "budget_per_minute": budget.budget_per_minute,
                    "recent_attempts": len(budget.attempts_history)
                }
                for name, budget in self.retry_budgets.items()
            }
        }


# Global retry manager instance
retry_manager = RetryManager()


def with_retry(
    operation_type: str = "api",
    service_name: Optional[str] = None,
    policy: Optional[RetryPolicy] = None
):
    """
    Decorator for adding retry logic to async functions.
    
    Args:
        operation_type: Type of operation for policy selection
        service_name: Service name for circuit breaker
        policy: Custom retry policy (overrides operation_type)
    """
    def decorator(func: Callable[..., Awaitable[Any]]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if policy:
                # Temporarily set custom policy
                original_policy = retry_manager.get_policy(operation_type)
                retry_manager.set_policy(operation_type, policy)
                try:
                    return await retry_manager.execute_with_retry(
                        func, operation_type, service_name, None, *args, **kwargs
                    )
                finally:
                    retry_manager.set_policy(operation_type, original_policy)
            else:
                return await retry_manager.execute_with_retry(
                    func, operation_type, service_name, None, *args, **kwargs
                )
        return wrapper
    return decorator


async def retry_with_backoff(
    func: Callable[..., Awaitable[Any]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
    *args,
    **kwargs
) -> Any:
    """
    Simple retry utility with customizable backoff.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        backoff_strategy: Backoff strategy to use
        
    Returns:
        Function result
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff_strategy=backoff_strategy
    )
    
    # Temporarily set policy and execute
    original_policy = retry_manager.get_policy("custom")
    retry_manager.set_policy("custom", policy)
    try:
        return await retry_manager.execute_with_retry(
            func, "custom", None, None, *args, **kwargs
        )
    finally:
        retry_manager.set_policy("custom", original_policy)