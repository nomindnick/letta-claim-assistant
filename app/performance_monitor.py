"""
Performance monitoring utilities for memory operations.

Tracks and reports on performance metrics for optimization.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
from pathlib import Path
from functools import wraps

from .logging_conf import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.operation_times = defaultdict(deque)  # Track operation durations
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.slow_operations = deque(maxlen=100)  # Track slow operations
        self.metrics_path = Path.home() / ".letta-claim" / "performance_metrics.json"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    def track_operation(self, operation_name: str):
        """Decorator to track operation performance."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    self._record_operation(operation_name, duration, success=True)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self._record_operation(operation_name, duration, success=False, error=str(e))
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self._record_operation(operation_name, duration, success=True)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self._record_operation(operation_name, duration, success=False, error=str(e))
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    def _record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record an operation's performance."""
        # Store in history
        if len(self.operation_times[operation]) >= self.max_history:
            self.operation_times[operation].popleft()
        
        self.operation_times[operation].append({
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'error': error
        })
        
        # Track slow operations (>2 seconds)
        if duration > 2.0:
            self.slow_operations.append({
                'operation': operation,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'error': error
            })
            logger.warning(
                f"Slow operation detected",
                operation=operation,
                duration_seconds=round(duration, 2)
            )
        
        # Log if operation failed
        if not success:
            logger.error(
                f"Operation failed",
                operation=operation,
                duration_seconds=round(duration, 2),
                error=error
            )
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_stats['hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_stats['misses'] += 1
    
    def record_cache_eviction(self):
        """Record a cache eviction."""
        self.cache_stats['evictions'] += 1
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        if total == 0:
            return 0.0
        return self.cache_stats['hits'] / total
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if operation not in self.operation_times:
            return {}
        
        times = self.operation_times[operation]
        if not times:
            return {}
        
        durations = [t['duration'] for t in times]
        success_count = sum(1 for t in times if t['success'])
        
        return {
            'count': len(times),
            'success_rate': success_count / len(times),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'p50_duration': self._percentile(durations, 50),
            'p95_duration': self._percentile(durations, 95),
            'p99_duration': self._percentile(durations, 99)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'cache_stats': {
                **self.cache_stats,
                'hit_rate': self.get_cache_hit_rate()
            },
            'operations': {},
            'slow_operations': list(self.slow_operations)[-10:],  # Last 10 slow ops
            'recommendations': []
        }
        
        # Add operation statistics
        for operation in self.operation_times:
            report['operations'][operation] = self.get_operation_stats(operation)
        
        # Generate recommendations
        if self.get_cache_hit_rate() < 0.5:
            report['recommendations'].append(
                "Low cache hit rate. Consider increasing cache TTL or size."
            )
        
        # Check for consistently slow operations
        for operation, stats in report['operations'].items():
            if stats.get('avg_duration', 0) > 1.0:
                report['recommendations'].append(
                    f"Operation '{operation}' is slow (avg {stats['avg_duration']:.2f}s). Consider optimization."
                )
            if stats.get('success_rate', 1.0) < 0.95:
                report['recommendations'].append(
                    f"Operation '{operation}' has high failure rate. Review error handling."
                )
        
        return report
    
    def save_metrics(self):
        """Save metrics to disk."""
        try:
            report = self.get_summary_report()
            with open(self.metrics_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Performance metrics saved", path=str(self.metrics_path))
        except Exception as e:
            logger.error("Failed to save metrics", error=str(e))
    
    def load_metrics(self) -> Optional[Dict[str, Any]]:
        """Load metrics from disk."""
        try:
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error("Failed to load metrics", error=str(e))
        return None


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Convenience decorators
def track_performance(operation_name: str):
    """Decorator to track operation performance."""
    return performance_monitor.track_operation(operation_name)


class PerformanceContext:
    """Context manager for tracking performance."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        performance_monitor._record_operation(
            self.operation_name,
            duration,
            success,
            error
        )