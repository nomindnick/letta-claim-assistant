"""
Performance optimization utilities for the UI.

Provides debouncing, caching, and lazy loading functionality.
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Coroutine
from datetime import datetime, timedelta
import hashlib
import json


T = TypeVar('T')


class Debouncer:
    """Debounce function calls to reduce API requests."""
    
    def __init__(self, delay: float = 0.3):
        """
        Initialize debouncer.
        
        Args:
            delay: Delay in seconds before executing the function
        """
        self.delay = delay
        self.timer: Optional[asyncio.Task] = None
        self.pending_call: Optional[Callable] = None
        
    def debounce(self, func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, Optional[T]]]:
        """
        Decorator to debounce async function calls.
        
        Usage:
            @debouncer.debounce
            async def search(query: str):
                return await api.search(query)
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Cancel previous timer if exists
            if self.timer:
                self.timer.cancel()
            
            # Store the function call
            async def delayed_call():
                await asyncio.sleep(self.delay)
                return await func(*args, **kwargs)
            
            # Create new timer
            self.timer = asyncio.create_task(delayed_call())
            
            try:
                return await self.timer
            except asyncio.CancelledError:
                return None
        
        return wrapper
    
    def cancel(self):
        """Cancel pending debounced call."""
        if self.timer:
            self.timer.cancel()
            self.timer = None


class ResponseCache:
    """Cache for API responses to reduce redundant requests."""
    
    def __init__(self, ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            ttl: Time to live in seconds (default: 5 minutes)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - timestamp > timedelta(seconds=self.ttl)
    
    def cached(self, func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        """
        Decorator to cache async function results.
        
        Usage:
            @cache.cached
            async def get_data(id: str):
                return await api.fetch(id)
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_key(func.__name__, *args, **kwargs)
            
            # Check cache
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if not self._is_expired(entry['timestamp']):
                    return entry['value']
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            self.cache[cache_key] = {
                'value': result,
                'timestamp': datetime.now()
            }
            
            return result
        
        return wrapper
    
    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match keys (if None, clears all)
        """
        if pattern is None:
            self.cache.clear()
        else:
            keys_to_remove = [
                key for key in self.cache.keys()
                if pattern in key
            ]
            for key in keys_to_remove:
                del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values()
            if self._is_expired(entry['timestamp'])
        )
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'cache_size_bytes': sum(
                len(str(entry['value'])) for entry in self.cache.values()
            )
        }


class LazyLoader:
    """Lazy loading for heavy components."""
    
    def __init__(self):
        self.loaded_components: Dict[str, Any] = {}
        self.loading_tasks: Dict[str, asyncio.Task] = {}
    
    async def load_component(
        self,
        component_id: str,
        loader_func: Callable[[], Coroutine[Any, Any, T]]
    ) -> T:
        """
        Load a component lazily.
        
        Args:
            component_id: Unique identifier for the component
            loader_func: Async function to load the component
            
        Returns:
            The loaded component
        """
        # Return if already loaded
        if component_id in self.loaded_components:
            return self.loaded_components[component_id]
        
        # Return existing loading task if in progress
        if component_id in self.loading_tasks:
            return await self.loading_tasks[component_id]
        
        # Start loading
        async def load():
            try:
                component = await loader_func()
                self.loaded_components[component_id] = component
                return component
            finally:
                # Clean up loading task
                if component_id in self.loading_tasks:
                    del self.loading_tasks[component_id]
        
        # Create and store loading task
        task = asyncio.create_task(load())
        self.loading_tasks[component_id] = task
        
        return await task
    
    def is_loaded(self, component_id: str) -> bool:
        """Check if a component is loaded."""
        return component_id in self.loaded_components
    
    def is_loading(self, component_id: str) -> bool:
        """Check if a component is currently loading."""
        return component_id in self.loading_tasks
    
    def unload(self, component_id: str):
        """Unload a component to free memory."""
        if component_id in self.loaded_components:
            del self.loaded_components[component_id]
        
        if component_id in self.loading_tasks:
            self.loading_tasks[component_id].cancel()
            del self.loading_tasks[component_id]


class RequestBatcher:
    """Batch multiple requests into single API calls."""
    
    def __init__(self, batch_size: int = 10, batch_delay: float = 0.1):
        """
        Initialize request batcher.
        
        Args:
            batch_size: Maximum number of requests to batch
            batch_delay: Delay before sending batch (seconds)
        """
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.pending_requests: list = []
        self.batch_timer: Optional[asyncio.Task] = None
        
    async def add_request(
        self,
        request_data: Any,
        batch_processor: Callable[[list], Coroutine[Any, Any, list]]
    ) -> Any:
        """
        Add a request to the batch.
        
        Args:
            request_data: Data for this request
            batch_processor: Function to process the batch
            
        Returns:
            Result for this specific request
        """
        # Create future for this request
        future = asyncio.Future()
        self.pending_requests.append((request_data, future))
        
        # Start batch timer if not running
        if not self.batch_timer:
            self.batch_timer = asyncio.create_task(
                self._process_batch(batch_processor)
            )
        
        # If batch is full, process immediately
        if len(self.pending_requests) >= self.batch_size:
            if self.batch_timer:
                self.batch_timer.cancel()
            await self._execute_batch(batch_processor)
        
        return await future
    
    async def _process_batch(self, batch_processor: Callable):
        """Process batch after delay."""
        await asyncio.sleep(self.batch_delay)
        await self._execute_batch(batch_processor)
    
    async def _execute_batch(self, batch_processor: Callable):
        """Execute the batched requests."""
        if not self.pending_requests:
            return
        
        # Extract requests and futures
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        self.batch_timer = None
        
        try:
            # Process batch
            request_data = [req[0] for req in batch]
            results = await batch_processor(request_data)
            
            # Resolve futures
            for i, (_, future) in enumerate(batch):
                if i < len(results):
                    future.set_result(results[i])
                else:
                    future.set_exception(Exception("No result for request"))
                    
        except Exception as e:
            # Reject all futures on error
            for _, future in batch:
                future.set_exception(e)


# Global instances for easy access
chat_debouncer = Debouncer(delay=0.5)
search_debouncer = Debouncer(delay=0.3)
response_cache = ResponseCache(ttl=300)
lazy_loader = LazyLoader()
request_batcher = RequestBatcher()


def with_loading_state(loading_indicator):
    """
    Decorator to show loading state during async operations.
    
    Args:
        loading_indicator: UI element to show/hide
        
    Usage:
        @with_loading_state(self.loading_spinner)
        async def load_data():
            return await api.fetch()
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Show loading
            loading_indicator.visible = True
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                return result
            finally:
                # Hide loading
                loading_indicator.visible = False
        
        return wrapper
    return decorator


def measure_performance(func):
    """
    Decorator to measure function performance.
    
    Usage:
        @measure_performance
        async def slow_operation():
            await asyncio.sleep(1)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            print(f"Performance: {func.__name__} took {duration:.3f}s")
    
    return wrapper