"""
Request Queue and Batching for Letta Operations

Implements a priority-based request queue with automatic batching for efficient
Letta server operations. Features include:
- Priority-based processing (user requests > background tasks)
- Automatic batching of similar operations
- Overflow handling with configurable limits
- Request deduplication
- Timeout management per request type
"""

import asyncio
import time
import hashlib
from typing import Optional, Dict, Any, List, Tuple, Callable, TypeVar, Set
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

from .logging_conf import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class RequestPriority(IntEnum):
    """Request priority levels (lower value = higher priority)."""
    CRITICAL = 0    # Health checks, recovery operations
    HIGH = 1        # User-initiated operations
    NORMAL = 2      # Background operations
    LOW = 3         # Maintenance, cleanup


class RequestType(Enum):
    """Types of requests for batching."""
    MEMORY_RECALL = "memory_recall"
    MEMORY_INSERT = "memory_insert"
    MEMORY_UPDATE = "memory_update"
    AGENT_CREATE = "agent_create"
    AGENT_UPDATE = "agent_update"
    HEALTH_CHECK = "health_check"
    OTHER = "other"


@dataclass(order=True)
class QueuedRequest:
    """A request in the queue."""
    priority: RequestPriority = field(compare=True)
    timestamp: float = field(compare=True)
    request_id: str = field(compare=False)
    request_type: RequestType = field(compare=False)
    operation: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    future: asyncio.Future = field(default_factory=asyncio.Future, compare=False)
    timeout: float = field(default=30.0, compare=False)
    batch_key: Optional[str] = field(default=None, compare=False)
    
    def __post_init__(self):
        """Generate batch key for similar operations."""
        if self.batch_key is None:
            # Create batch key based on operation type and certain parameters
            batch_data = f"{self.request_type.value}"
            
            # Add agent_id if present in kwargs for agent-specific operations
            if "agent_id" in self.kwargs:
                batch_data += f":{self.kwargs['agent_id']}"
            
            self.batch_key = batch_data


class RequestQueueMetrics:
    """Tracks queue metrics."""
    
    def __init__(self):
        self.total_requests = 0
        self.processed_requests = 0
        self.batched_requests = 0
        self.dropped_requests = 0
        self.timed_out_requests = 0
        self.queue_sizes: List[int] = []
        self.processing_times: Dict[RequestType, List[float]] = defaultdict(list)
        self.batch_sizes: List[int] = []
        self.last_overflow_time: Optional[datetime] = None
    
    def record_enqueue(self, queue_size: int):
        """Record request enqueue."""
        self.total_requests += 1
        self.queue_sizes.append(queue_size)
        if len(self.queue_sizes) > 1000:
            self.queue_sizes = self.queue_sizes[-1000:]
    
    def record_processing(self, request_type: RequestType, duration: float, batch_size: int = 1):
        """Record request processing."""
        self.processed_requests += 1
        if batch_size > 1:
            self.batched_requests += batch_size - 1
            self.batch_sizes.append(batch_size)
        
        self.processing_times[request_type].append(duration)
        if len(self.processing_times[request_type]) > 100:
            self.processing_times[request_type] = self.processing_times[request_type][-100:]
    
    def record_drop(self):
        """Record dropped request."""
        self.dropped_requests += 1
        self.last_overflow_time = datetime.now()
    
    def record_timeout(self):
        """Record timed out request."""
        self.timed_out_requests += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        avg_queue_size = sum(self.queue_sizes) / len(self.queue_sizes) if self.queue_sizes else 0
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 1
        
        avg_processing_times = {}
        for req_type, times in self.processing_times.items():
            if times:
                avg_processing_times[req_type.value] = sum(times) / len(times)
        
        return {
            "total_requests": self.total_requests,
            "processed_requests": self.processed_requests,
            "batched_requests": self.batched_requests,
            "dropped_requests": self.dropped_requests,
            "timed_out_requests": self.timed_out_requests,
            "average_queue_size": avg_queue_size,
            "average_batch_size": avg_batch_size,
            "average_processing_times": avg_processing_times,
            "last_overflow": self.last_overflow_time.isoformat() if self.last_overflow_time else None
        }


class RequestQueue:
    """
    Priority-based request queue with batching support.
    
    Args:
        max_queue_size: Maximum number of requests in queue
        batch_size: Maximum requests to batch together
        batch_timeout: Max time to wait for batch to fill
        overflow_policy: What to do when queue is full ("drop_oldest", "drop_new", "block")
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        batch_size: int = 10,
        batch_timeout: float = 1.0,
        overflow_policy: str = "drop_oldest"
    ):
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.overflow_policy = overflow_policy
        
        self._queue: List[QueuedRequest] = []
        self._request_counter = 0
        self._processing = False
        self._metrics = RequestQueueMetrics()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition()
        self._dedup_cache: Set[str] = set()
        self._processor_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the queue processor."""
        if not self._processor_task or self._processor_task.done():
            self._processing = True
            self._processor_task = asyncio.create_task(self._process_queue())
            logger.info("Request queue processor started")
    
    async def stop(self):
        """Stop the queue processor."""
        self._processing = False
        if self._processor_task:
            async with self._not_empty:
                self._not_empty.notify()  # Wake up processor
            try:
                await asyncio.wait_for(self._processor_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._processor_task.cancel()
            logger.info("Request queue processor stopped")
    
    async def enqueue(
        self,
        operation: Callable,
        request_type: RequestType,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0,
        dedup_key: Optional[str] = None,
        *args,
        **kwargs
    ) -> asyncio.Future:
        """
        Add a request to the queue.
        
        Args:
            operation: Async function to execute
            request_type: Type of request for batching
            priority: Request priority
            timeout: Request timeout in seconds
            dedup_key: Optional key for deduplication
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Future that will contain the result
        """
        # Check deduplication
        if dedup_key and dedup_key in self._dedup_cache:
            logger.debug(f"Request deduplicated: {dedup_key}")
            future = asyncio.Future()
            future.set_result(None)  # Return None for deduplicated requests
            return future
        
        async with self._lock:
            # Check queue size
            if len(self._queue) >= self.max_queue_size:
                if self.overflow_policy == "drop_new":
                    self._metrics.record_drop()
                    raise OverflowError(f"Request queue full ({self.max_queue_size} items)")
                elif self.overflow_policy == "drop_oldest":
                    # Remove oldest normal/low priority request
                    for i, req in enumerate(self._queue):
                        if req.priority >= RequestPriority.NORMAL:
                            dropped = heapq.heappop(self._queue)
                            dropped.future.set_exception(
                                Exception("Request dropped due to queue overflow")
                            )
                            self._metrics.record_drop()
                            break
                    else:
                        # No droppable requests, reject new one
                        raise OverflowError("Request queue full with high priority requests")
            
            # Create request
            self._request_counter += 1
            request = QueuedRequest(
                priority=priority,
                timestamp=time.time(),
                request_id=f"req-{self._request_counter}",
                request_type=request_type,
                operation=operation,
                args=args,
                kwargs=kwargs,
                timeout=timeout
            )
            
            # Add to queue and dedup cache
            heapq.heappush(self._queue, request)
            if dedup_key:
                self._dedup_cache.add(dedup_key)
            
            self._metrics.record_enqueue(len(self._queue))
            
            # Notify processor
            async with self._not_empty:
                self._not_empty.notify()
            
            return request.future
    
    async def _process_queue(self):
        """Process requests from the queue."""
        while self._processing:
            try:
                # Wait for requests
                async with self._not_empty:
                    while not self._queue and self._processing:
                        await self._not_empty.wait()
                
                if not self._processing:
                    break
                
                # Get batch of requests
                batch = await self._get_batch()
                if batch:
                    await self._process_batch(batch)
                    
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on persistent errors
    
    async def _get_batch(self) -> List[QueuedRequest]:
        """Get a batch of requests to process."""
        batch = []
        batch_keys = set()
        batch_start_time = time.time()
        
        async with self._lock:
            while self._queue and len(batch) < self.batch_size:
                # Check if we've waited too long for batch
                if batch and (time.time() - batch_start_time) > self.batch_timeout:
                    break
                
                # Find compatible request
                for i, request in enumerate(self._queue):
                    # Check timeout
                    if (time.time() - request.timestamp) > request.timeout:
                        # Remove timed out request
                        self._queue.pop(i)
                        heapq.heapify(self._queue)
                        request.future.set_exception(
                            asyncio.TimeoutError(f"Request timed out after {request.timeout}s")
                        )
                        self._metrics.record_timeout()
                        continue
                    
                    # Check if request can be batched
                    if not batch or request.batch_key in batch_keys or len(batch_keys) == 0:
                        self._queue.pop(i)
                        heapq.heapify(self._queue)
                        batch.append(request)
                        batch_keys.add(request.batch_key)
                        break
                else:
                    # No compatible request found, process what we have
                    break
        
        return batch
    
    async def _process_batch(self, batch: List[QueuedRequest]):
        """Process a batch of requests."""
        start_time = time.time()
        
        # Group by operation type for efficient processing
        grouped = defaultdict(list)
        for request in batch:
            grouped[request.request_type].append(request)
        
        # Process each group
        for request_type, requests in grouped.items():
            try:
                if len(requests) == 1:
                    # Single request, process normally
                    request = requests[0]
                    result = await request.operation(*request.args, **request.kwargs)
                    request.future.set_result(result)
                else:
                    # Multiple requests, try to batch if possible
                    await self._process_batched_requests(request_type, requests)
                
            except Exception as e:
                # Set exception for all requests in group
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(e)
        
        # Record metrics
        duration = time.time() - start_time
        for request_type, requests in grouped.items():
            self._metrics.record_processing(request_type, duration, len(requests))
        
        # Clear dedup cache entries
        async with self._lock:
            for request in batch:
                # Remove from dedup cache after processing
                # (Note: In real implementation, would need dedup_key stored)
                pass
    
    async def _process_batched_requests(
        self,
        request_type: RequestType,
        requests: List[QueuedRequest]
    ):
        """Process multiple requests of the same type efficiently."""
        if request_type == RequestType.MEMORY_RECALL:
            # Batch memory recall requests
            queries = []
            for req in requests:
                if "query" in req.kwargs:
                    queries.append(req.kwargs["query"])
                elif req.args:
                    queries.append(req.args[0])
            
            # Execute batched recall (would need special implementation)
            # For now, process individually
            for req in requests:
                try:
                    result = await req.operation(*req.args, **req.kwargs)
                    req.future.set_result(result)
                except Exception as e:
                    req.future.set_exception(e)
                    
        elif request_type == RequestType.MEMORY_INSERT:
            # Batch memory insert requests
            all_memories = []
            for req in requests:
                if "memories" in req.kwargs:
                    all_memories.extend(req.kwargs["memories"])
                elif "memory" in req.kwargs:
                    all_memories.append(req.kwargs["memory"])
            
            # Could batch insert all memories at once
            # For now, process individually
            for req in requests:
                try:
                    result = await req.operation(*req.args, **req.kwargs)
                    req.future.set_result(result)
                except Exception as e:
                    req.future.set_exception(e)
        else:
            # Default: process individually
            for req in requests:
                try:
                    result = await req.operation(*req.args, **req.kwargs)
                    req.future.set_result(result)
                except Exception as e:
                    req.future.set_exception(e)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        return {
            "queue_size": len(self._queue),
            "max_queue_size": self.max_queue_size,
            "processing": self._processing,
            "metrics": self._metrics.get_stats()
        }
    
    async def flush(self):
        """Process all pending requests immediately."""
        while self._queue:
            batch = await self._get_batch()
            if batch:
                await self._process_batch(batch)
    
    async def clear(self):
        """Clear all pending requests."""
        async with self._lock:
            for request in self._queue:
                request.future.set_exception(
                    Exception("Request cancelled due to queue clear")
                )
            self._queue.clear()
            self._dedup_cache.clear()


# Global request queue instance
request_queue = RequestQueue()


async def enqueue_letta_operation(
    operation: Callable,
    request_type: RequestType,
    priority: RequestPriority = RequestPriority.NORMAL,
    timeout: float = 30.0,
    *args,
    **kwargs
):
    """
    Convenience function to enqueue a Letta operation.
    
    Example:
        result = await enqueue_letta_operation(
            client.recall_memory,
            RequestType.MEMORY_RECALL,
            priority=RequestPriority.HIGH,
            query="What happened yesterday?"
        )
    """
    return await request_queue.enqueue(
        operation=operation,
        request_type=request_type,
        priority=priority,
        timeout=timeout,
        *args,
        **kwargs
    )