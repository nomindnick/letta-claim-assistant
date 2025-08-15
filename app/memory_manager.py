"""
Memory management and optimization utilities.

Provides memory monitoring, cleanup, and optimized loading patterns
for large document collections.
"""

import psutil
import gc
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path
import time
from dataclasses import dataclass

from .logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryStats:
    """Current memory usage statistics."""
    total_bytes: int
    available_bytes: int
    used_bytes: int
    usage_percent: float
    process_memory_bytes: int


class MemoryManager:
    """Manages memory usage and optimization for the application."""
    
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.process = psutil.Process()
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        return MemoryStats(
            total_bytes=memory.total,
            available_bytes=memory.available,
            used_bytes=memory.used,
            usage_percent=memory.percent,
            process_memory_bytes=process_memory.rss
        )
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check if system is under memory pressure."""
        stats = self.get_memory_stats()
        
        status = {
            "pressure_level": "normal",
            "should_cleanup": False,
            "should_pause": False,
            "stats": stats
        }
        
        if stats.usage_percent >= self.critical_threshold:
            status["pressure_level"] = "critical"
            status["should_cleanup"] = True
            status["should_pause"] = True
        elif stats.usage_percent >= self.warning_threshold:
            status["pressure_level"] = "warning"
            status["should_cleanup"] = True
        
        return status
    
    async def cleanup_memory(self) -> Dict[str, Any]:
        """Perform memory cleanup operations."""
        cleanup_start = time.time()
        initial_stats = self.get_memory_stats()
        
        logger.info("Starting memory cleanup", 
                   usage_percent=initial_stats.usage_percent,
                   process_memory_mb=initial_stats.process_memory_bytes / 1024 / 1024)
        
        # Force garbage collection
        collected = gc.collect()
        
        # Brief pause to let system stabilize
        await asyncio.sleep(0.1)
        
        final_stats = self.get_memory_stats()
        cleanup_time = time.time() - cleanup_start
        
        memory_freed = initial_stats.process_memory_bytes - final_stats.process_memory_bytes
        
        result = {
            "cleanup_time_seconds": cleanup_time,
            "objects_collected": collected,
            "memory_freed_bytes": memory_freed,
            "memory_freed_mb": memory_freed / 1024 / 1024,
            "initial_usage_percent": initial_stats.usage_percent,
            "final_usage_percent": final_stats.usage_percent
        }
        
        logger.info("Memory cleanup completed", **result)
        return result
    
    def format_bytes(self, bytes_value: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"


class ChunkedProcessor:
    """Processes large datasets in memory-efficient chunks."""
    
    def __init__(self, chunk_size: int = 50, memory_manager: Optional[MemoryManager] = None):
        self.chunk_size = chunk_size
        self.memory_manager = memory_manager or MemoryManager()
    
    async def process_in_chunks(
        self, 
        items: List[Any],
        processor_func: callable,
        progress_callback: Optional[callable] = None
    ) -> List[Any]:
        """Process items in memory-efficient chunks."""
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, self.chunk_size):
            chunk = items[i:i + self.chunk_size]
            chunk_num = i // self.chunk_size + 1
            total_chunks = (total_items + self.chunk_size - 1) // self.chunk_size
            
            logger.debug(f"Processing chunk {chunk_num}/{total_chunks}", 
                        chunk_size=len(chunk))
            
            # Check memory pressure before processing
            memory_status = self.memory_manager.check_memory_pressure()
            
            if memory_status["should_pause"]:
                logger.warning("High memory usage detected, pausing processing",
                              usage_percent=memory_status["stats"].usage_percent)
                await asyncio.sleep(1.0)
            
            if memory_status["should_cleanup"]:
                await self.memory_manager.cleanup_memory()
            
            # Process the chunk
            try:
                chunk_results = await processor_func(chunk)
                if isinstance(chunk_results, list):
                    results.extend(chunk_results)
                else:
                    results.append(chunk_results)
                
                # Update progress
                if progress_callback:
                    progress = min(1.0, (i + len(chunk)) / total_items)
                    await progress_callback(progress, f"Processed {i + len(chunk)}/{total_items} items")
                
                # Small delay between chunks to prevent overwhelming
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num}", error=str(e))
                # Continue with next chunk rather than failing completely
                continue
        
        return results


class LazyLoader:
    """Lazy loading utility for large datasets."""
    
    def __init__(self, loader_func: callable, cache_size: int = 100):
        self.loader_func = loader_func
        self.cache_size = cache_size
        self.cache = {}
        self.access_order = []
    
    async def load(self, key: str) -> Any:
        """Load item lazily with caching."""
        if key in self.cache:
            # Move to end of access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # Load the item
        item = await self.loader_func(key)
        
        # Cache management
        if len(self.cache) >= self.cache_size:
            # Remove least recently used item
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = item
        self.access_order.append(key)
        
        return item
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


class DocumentStreamProcessor:
    """Memory-efficient document processing with streaming."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()
    
    async def process_document_stream(
        self,
        document_paths: List[Path],
        processor_func: callable,
        max_concurrent: int = 3
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process documents as a stream to minimize memory usage."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_document(doc_path: Path) -> Dict[str, Any]:
            async with semaphore:
                # Check memory before processing
                memory_status = self.memory_manager.check_memory_pressure()
                
                if memory_status["should_pause"]:
                    logger.warning("Memory pressure detected, waiting before processing", 
                                  document=doc_path.name)
                    await asyncio.sleep(2.0)
                    await self.memory_manager.cleanup_memory()
                
                try:
                    result = await processor_func(doc_path)
                    return {
                        "success": True,
                        "document": str(doc_path),
                        "result": result
                    }
                except Exception as e:
                    logger.error("Document processing failed", 
                               document=doc_path.name, error=str(e))
                    return {
                        "success": False,
                        "document": str(doc_path),
                        "error": str(e)
                    }
        
        # Process documents and yield results as they complete
        tasks = [process_single_document(doc_path) for doc_path in document_paths]
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result


# Global memory manager instance
memory_manager = MemoryManager()


async def monitor_memory_usage(interval_seconds: int = 30) -> None:
    """Background task to monitor memory usage."""
    while True:
        try:
            stats = memory_manager.get_memory_stats()
            
            if stats.usage_percent > 85:
                logger.warning("High memory usage detected",
                              usage_percent=stats.usage_percent,
                              process_memory_mb=stats.process_memory_bytes / 1024 / 1024)
                
                if stats.usage_percent > 95:
                    logger.critical("Critical memory usage, forcing cleanup")
                    await memory_manager.cleanup_memory()
            
            await asyncio.sleep(interval_seconds)
            
        except Exception as e:
            logger.error("Memory monitoring error", error=str(e))
            await asyncio.sleep(interval_seconds)


def get_memory_stats_dict() -> Dict[str, Any]:
    """Get memory statistics as a dictionary for API responses."""
    stats = memory_manager.get_memory_stats()
    return {
        "total_gb": stats.total_bytes / 1024 / 1024 / 1024,
        "available_gb": stats.available_bytes / 1024 / 1024 / 1024,
        "used_gb": stats.used_bytes / 1024 / 1024 / 1024,
        "usage_percent": stats.usage_percent,
        "process_memory_mb": stats.process_memory_bytes / 1024 / 1024
    }