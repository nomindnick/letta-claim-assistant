"""
Letta Connection Management Module

Provides singleton connection management for Letta client with retry logic,
health monitoring, connection pooling, and metrics collection.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import random
from contextlib import asynccontextmanager

try:
    from letta_client import AsyncLetta, Letta
    from letta_client.errors import (
        BadRequestError,
        NotFoundError,
        InternalServerError,
        UnprocessableEntityError
    )
    # Use base Exception as ClientError
    ClientError = Exception
    LETTA_AVAILABLE = True
except ImportError:
    AsyncLetta = None
    Letta = None
    ClientError = Exception
    BadRequestError = Exception
    NotFoundError = Exception
    InternalServerError = Exception
    UnprocessableEntityError = Exception
    LETTA_AVAILABLE = False

from .logging_conf import get_logger
from .letta_server import server_manager

logger = get_logger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RETRYING = "retrying"
    FAILED = "failed"
    FALLBACK = "fallback"


class ConnectionMetrics:
    """Tracks connection and operation metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.operation_latencies: Dict[str, List[float]] = {
            "health_check": [],
            "recall": [],
            "upsert": [],
            "suggest": [],
            "create_agent": [],
            "get_agent": []
        }
        self.success_count = 0
        self.failure_count = 0
        self.retry_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.connection_established_time: Optional[datetime] = None
    
    def record_operation(self, operation: str, latency: float, success: bool):
        """Record an operation's metrics."""
        if operation in self.operation_latencies:
            latencies = self.operation_latencies[operation]
            latencies.append(latency)
            # Keep only the last window_size entries
            if len(latencies) > self.window_size:
                self.operation_latencies[operation] = latencies[-self.window_size:]
        
        if success:
            self.success_count += 1
            self.last_success_time = datetime.now()
        else:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return (self.success_count / total * 100) if total > 0 else 0.0
    
    def get_average_latency(self, operation: str = None) -> float:
        """Get average latency for an operation or all operations."""
        if operation:
            latencies = self.operation_latencies.get(operation, [])
            return sum(latencies) / len(latencies) if latencies else 0.0
        else:
            all_latencies = []
            for latencies in self.operation_latencies.values():
                all_latencies.extend(latencies)
            return sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    
    def get_uptime(self) -> timedelta:
        """Get connection uptime."""
        if self.connection_established_time:
            return datetime.now() - self.connection_established_time
        return timedelta(0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "success_rate": self.get_success_rate(),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "retry_count": self.retry_count,
            "average_latency_ms": self.get_average_latency() * 1000,
            "uptime_seconds": self.get_uptime().total_seconds(),
            "last_success": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "operation_latencies": {
                op: {
                    "average_ms": self.get_average_latency(op) * 1000,
                    "sample_count": len(latencies)
                }
                for op, latencies in self.operation_latencies.items()
                if latencies
            }
        }


class LettaConnectionManager:
    """
    Singleton connection manager for Letta client.
    
    Provides:
    - Connection pooling with singleton pattern
    - Automatic retry with exponential backoff
    - Health monitoring
    - Metrics collection
    - Graceful fallback handling
    """
    
    _instance: Optional['LettaConnectionManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls) -> 'LettaConnectionManager':
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the connection manager."""
        if self._initialized:
            return
        
        self.client: Optional[AsyncLetta] = None
        self.sync_client: Optional[Letta] = None
        self.base_url: Optional[str] = None
        self.state = ConnectionState.DISCONNECTED
        self.metrics = ConnectionMetrics()
        
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        self.max_delay = 30.0  # seconds
        
        # Health check configuration
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Timeout configuration per operation type
        self.operation_timeouts = {
            "health_check": 5.0,
            "recall": 30.0,
            "upsert": 30.0,
            "suggest": 30.0,
            "create_agent": 60.0,
            "get_agent": 10.0,
            "delete_agent": 30.0,
            "default": 30.0
        }
        
        # Resource management
        self._active_operations = set()
        self._cleanup_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._health_task = None  # Add for compatibility with tests
        
        self._initialized = True
        
        logger.info("LettaConnectionManager initialized")
    
    def configure(self, base_url: str = None, **kwargs):
        """Configure the connection manager."""
        if base_url:
            self.base_url = base_url
        # Store any additional configuration
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    async def connect(
        self,
        base_url: Optional[str] = None,
        force_reconnect: bool = False
    ) -> bool:
        """
        Connect to Letta server with retry logic.
        
        Args:
            base_url: Server URL (uses server_manager if not provided)
            force_reconnect: Force a new connection even if already connected
            
        Returns:
            True if connected successfully, False otherwise
        """
        async with self._lock:
            # Check if already connected
            if self.state == ConnectionState.CONNECTED and not force_reconnect:
                return True
            
            # Check if Letta is available
            if not LETTA_AVAILABLE:
                logger.warning("Letta client library not available")
                self.state = ConnectionState.FALLBACK
                return False
            
            self.state = ConnectionState.CONNECTING
            
            # Get base URL from server manager if not provided
            if not base_url:
                # Ensure server is running
                if not server_manager._is_running:
                    if not server_manager.start():
                        logger.error("Failed to start Letta server")
                        self.state = ConnectionState.FAILED
                        return False
                
                base_url = server_manager.get_base_url()
            
            self.base_url = base_url
            
            # Try to connect with retries
            for attempt in range(self.max_retries):
                try:
                    self.state = ConnectionState.RETRYING if attempt > 0 else ConnectionState.CONNECTING
                    
                    logger.debug(f"Attempting to connect to {base_url} (attempt {attempt + 1}/{self.max_retries})")
                    
                    # Create clients
                    self.client = AsyncLetta(base_url=base_url)
                    self.sync_client = Letta(base_url=base_url)
                    
                    # Test connection with health check
                    start_time = time.time()
                    await self.client.health.check()
                    latency = time.time() - start_time
                    
                    # Connection successful
                    self.state = ConnectionState.CONNECTED
                    self.metrics.connection_established_time = datetime.now()
                    self.metrics.record_operation("health_check", latency, True)
                    
                    # Start health monitoring
                    await self._start_health_monitor()
                    
                    logger.info(f"Connected to Letta server at {base_url} (latency: {latency*1000:.2f}ms)")
                    return True
                    
                except Exception as e:
                    self.metrics.retry_count += 1
                    self.metrics.record_operation("health_check", 0, False)
                    
                    if attempt < self.max_retries - 1:
                        # Calculate backoff delay with jitter
                        delay = min(
                            self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                            self.max_delay
                        )
                        logger.warning(
                            f"Connection attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Failed to connect after {self.max_retries} attempts: {e}")
            
            # All retries failed
            self.state = ConnectionState.FAILED
            self.client = None
            self.sync_client = None
            return False
    
    async def ensure_connected(self) -> bool:
        """
        Ensure client is connected, attempting reconnection if needed.
        
        Returns:
            True if connected, False if in fallback mode
        """
        if self.state == ConnectionState.CONNECTED:
            # Quick health check
            if await self._quick_health_check():
                return True
            # Health check failed, try to reconnect
            logger.warning("Health check failed, attempting reconnection")
        
        if self.state == ConnectionState.FALLBACK:
            return False
        
        # Attempt to connect
        return await self.connect()
    
    async def _check_health(self) -> bool:
        """Perform a health check."""
        return await self._quick_health_check()
    
    async def _quick_health_check(self) -> bool:
        """Perform a quick health check."""
        if not self.client:
            return False
        
        try:
            start_time = time.time()
            await self.client.health.check()
            latency = time.time() - start_time
            self.metrics.record_operation("health_check", latency, True)
            return True
        except Exception:
            self.metrics.record_operation("health_check", 0, False)
            return False
    
    async def _start_health_monitor(self):
        """Start background health monitoring."""
        # Cancel existing task if any
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
        
        # Start new health monitor task
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
    
    def start_health_monitoring(self, interval: float = None, auto_restart: bool = False):
        """Start health monitoring (public method)."""
        if interval:
            self.health_check_interval = interval
        # Store auto_restart flag if needed
        asyncio.create_task(self._start_health_monitor())
    
    def stop_health_monitoring(self):
        """Stop health monitoring (public method)."""
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
            self.health_check_task = None
    
    async def _health_monitor_loop(self):
        """Background loop for health monitoring."""
        consecutive_failures = 0
        max_failures = 3
        
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if await self._quick_health_check():
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning(f"Health check failed ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Too many health check failures, marking connection as failed")
                        self.state = ConnectionState.FAILED
                        # Attempt reconnection
                        asyncio.create_task(self.connect())
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
    
    async def execute_with_retry(
        self,
        operation_name: str,
        operation_func,
        *args,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute an operation with retry logic, timeout, and metrics tracking.
        
        Args:
            operation_name: Name of the operation for metrics
            operation_func: Async function to execute
            *args: Arguments for the function
            timeout: Operation timeout in seconds (uses default if not specified)
            max_retries: Maximum retry attempts (uses default if not specified)
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation or None if failed
        """
        # Ensure connected
        if not await self.ensure_connected():
            logger.warning(f"Cannot execute {operation_name}: not connected")
            return None
        
        # Get timeout for operation type
        if timeout is None:
            timeout = self.operation_timeouts.get(operation_name, self.operation_timeouts["default"])
        
        if max_retries is None:
            max_retries = 2
        
        # Track active operation
        operation_id = f"{operation_name}_{time.time()}"
        self._active_operations.add(operation_id)
        
        try:
            # Try operation with retries
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        operation_func(*args, **kwargs),
                        timeout=timeout
                    )
                    
                    latency = time.time() - start_time
                    self.metrics.record_operation(operation_name, latency, True)
                    
                    if latency > 5.0:  # Warn if operation is slow
                        logger.warning(f"{operation_name} took {latency:.2f}s")
                    
                    return result
                    
                except asyncio.TimeoutError:
                    self.metrics.record_operation(operation_name, timeout, False)
                    logger.error(f"{operation_name} timed out after {timeout}s")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying {operation_name} (attempt {attempt + 2}/{max_retries})")
                        await asyncio.sleep(1)
                    else:
                        return None
                    
                except NotFoundError as e:
                    # Don't retry for not found errors
                    self.metrics.record_operation(operation_name, 0, False)
                    logger.error(f"{operation_name} failed: {e}")
                    return None
                    
                except (BadRequestError, ClientError, InternalServerError) as e:
                    self.metrics.record_operation(operation_name, 0, False)
                    
                    # Log more details about the error
                    error_details = str(e)
                    if hasattr(e, 'response'):
                        if hasattr(e.response, 'text'):
                            error_details = f"{error_details} - Response: {e.response.text}"
                        elif hasattr(e.response, 'content'):
                            error_details = f"{error_details} - Response: {e.response.content}"
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"{operation_name} failed, retrying: {error_details}")
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"{operation_name} failed after {max_retries} attempts: {error_details}")
                        # For InternalServerError, raise it so caller can handle differently
                        if isinstance(e, InternalServerError):
                            raise e
                        return None
                        
                except Exception as e:
                    self.metrics.record_operation(operation_name, 0, False)
                    logger.error(f"Unexpected error in {operation_name}: {e}")
                    return None
            
            return None
            
        finally:
            # Clean up active operation
            self._active_operations.discard(operation_id)
    
    @asynccontextmanager
    async def client_session(self):
        """
        Context manager for client operations.
        
        Yields:
            AsyncLetta client or None if not connected
        """
        if await self.ensure_connected():
            yield self.client
        else:
            yield None
    
    def get_state(self) -> ConnectionState:
        """Get current connection state."""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connection metrics."""
        return {
            "state": self.state.value,
            "base_url": self.base_url,
            "metrics": self.metrics.to_dict()
        }
    
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self.state == ConnectionState.CONNECTED
    
    def is_fallback(self) -> bool:
        """Check if in fallback mode."""
        return self.state == ConnectionState.FALLBACK
    
    def configure_timeout(self, operation_name: str, timeout: float):
        """Configure timeout for a specific operation type."""
        self.operation_timeouts[operation_name] = timeout
        logger.debug(f"Set timeout for {operation_name} to {timeout}s")
    
    def configure_retries(self, max_retries: int, base_delay: float = None):
        """Configure retry behavior."""
        self.max_retries = max_retries
        if base_delay is not None:
            self.base_delay = base_delay
        logger.debug(f"Set max retries to {max_retries}, base delay to {self.base_delay}s")
    
    async def cleanup_resources(self):
        """Clean up all resources and active operations."""
        async with self._cleanup_lock:
            logger.info("Starting resource cleanup")
            
            # Cancel all active operations
            if self._active_operations:
                logger.info(f"Cancelling {len(self._active_operations)} active operations")
                # Note: In a real implementation, would track actual tasks to cancel
                self._active_operations.clear()
            
            # Stop health monitoring
            if self.health_check_task and not self.health_check_task.done():
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
                self.health_check_task = None
            
            # Clear metrics older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            for op_name, latencies in self.metrics.operation_latencies.items():
                # In real implementation, would filter by timestamp
                if len(latencies) > 100:
                    self.metrics.operation_latencies[op_name] = latencies[-100:]
            
            logger.info("Resource cleanup completed")
    
    async def wait_for_operations(self, timeout: float = 30.0) -> bool:
        """
        Wait for all active operations to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all operations completed, False if timeout
        """
        start_time = time.time()
        
        while self._active_operations and (time.time() - start_time) < timeout:
            logger.debug(f"Waiting for {len(self._active_operations)} operations to complete")
            await asyncio.sleep(0.5)
        
        if self._active_operations:
            logger.warning(f"{len(self._active_operations)} operations still active after {timeout}s")
            return False
        
        return True
    
    async def graceful_shutdown(self):
        """Perform graceful shutdown with resource cleanup."""
        logger.info("Starting graceful shutdown")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for active operations
        await self.wait_for_operations(timeout=10.0)
        
        # Clean up resources
        await self.cleanup_resources()
        
        # Disconnect
        await self.disconnect()
        
        logger.info("Graceful shutdown completed")
    
    async def disconnect(self):
        """Disconnect from server and clean up."""
        logger.info("Disconnecting from Letta server")
        
        # Stop health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
        
        # Clean up clients
        self.client = None
        self.sync_client = None
        self.state = ConnectionState.DISCONNECTED
        
        # Clear active operations
        self._active_operations.clear()
        
        logger.info("Disconnected from Letta server")


# Global connection manager instance
connection_manager = LettaConnectionManager()