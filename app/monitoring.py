"""
Performance monitoring and metrics collection for the Letta Construction Claim Assistant.

Collects and reports application performance metrics, resource usage, and health status
for production monitoring and optimization.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from .logging_conf import get_logger
from .settings import settings

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class SystemHealthStatus:
    """System health status information."""
    status: str  # "healthy", "warning", "critical"
    services: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    version: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "status": self.status,
            "services": self.services,
            "resources": self.resources,
            "last_check": self.last_check.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "version": self.version
        }


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: deque = deque(maxlen=max_history)
        self.aggregated_metrics: Dict[str, List[float]] = defaultdict(list)
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # Track specific metrics
        self.ingestion_stats = {
            "pages_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "ocr_operations": 0,
            "total_processing_time": 0.0
        }
        
        self.rag_stats = {
            "queries_processed": 0,
            "total_response_time": 0.0,
            "vector_searches": 0,
            "llm_calls": 0,
            "memory_recalls": 0
        }
        
        self.system_stats = {
            "memory_usage_mb": 0.0,
            "disk_usage_mb": 0.0,
            "cpu_percent": 0.0,
            "active_matters": 0
        }
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        with self._lock:
            self.metrics.append(metric)
            self.aggregated_metrics[metric.name].append(metric.value)
            
            # Keep only recent values for aggregation
            if len(self.aggregated_metrics[metric.name]) > 100:
                self.aggregated_metrics[metric.name] = self.aggregated_metrics[metric.name][-100:]
        
        logger.debug(
            "Metric recorded",
            metric=metric.name,
            value=metric.value,
            unit=metric.unit,
            tags=metric.tags
        )
    
    def record_timing(self, operation: str, duration_ms: float, **tags) -> None:
        """Record a timing metric."""
        metric = PerformanceMetric(
            name=f"{operation}_duration",
            value=duration_ms,
            unit="ms",
            tags=tags
        )
        self.record_metric(metric)
    
    def record_counter(self, name: str, increment: int = 1, **tags) -> None:
        """Record a counter metric."""
        metric = PerformanceMetric(
            name=name,
            value=increment,
            unit="count",
            tags=tags
        )
        self.record_metric(metric)
    
    def record_gauge(self, name: str, value: float, unit: str = "value", **tags) -> None:
        """Record a gauge metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            tags=tags
        )
        self.record_metric(metric)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary = {
                "total_metrics": len(self.metrics),
                "uptime_seconds": time.time() - self.start_time,
                "ingestion": self.ingestion_stats.copy(),
                "rag": self.rag_stats.copy(),
                "system": self.system_stats.copy(),
                "aggregated": {}
            }
            
            # Calculate aggregated statistics
            for name, values in self.aggregated_metrics.items():
                if values:
                    summary["aggregated"][name] = {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1] if values else 0
                    }
        
        return summary
    
    def get_recent_metrics(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent = [
                m.to_dict() for m in self.metrics 
                if m.timestamp >= cutoff_time
            ]
        
        return recent
    
    def update_ingestion_stats(self, **kwargs) -> None:
        """Update ingestion statistics."""
        with self._lock:
            for key, value in kwargs.items():
                if key in self.ingestion_stats:
                    self.ingestion_stats[key] += value
    
    def update_rag_stats(self, **kwargs) -> None:
        """Update RAG statistics."""
        with self._lock:
            for key, value in kwargs.items():
                if key in self.rag_stats:
                    self.rag_stats[key] += value
    
    def update_system_stats(self, **kwargs) -> None:
        """Update system statistics."""
        with self._lock:
            for key, value in kwargs.items():
                if key in self.system_stats:
                    self.system_stats[key] = value


class SystemMonitor:
    """Monitors system health and resources."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.last_health_check = datetime.now()
        self._monitoring = False
        self._monitor_task = None
        
    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start background system monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(interval_seconds)
        )
        
        logger.info("System monitoring started", interval=interval_seconds)
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self, interval_seconds: int) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(interval_seconds)
    
    async def collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            # Memory usage
            try:
                import psutil
                
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                system_memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                self.metrics.record_gauge("memory_usage", memory_mb, "MB")
                self.metrics.record_gauge("system_memory_percent", system_memory.percent, "percent")
                self.metrics.record_gauge("cpu_percent", cpu_percent, "percent")
                
                self.metrics.update_system_stats(
                    memory_usage_mb=memory_mb,
                    cpu_percent=cpu_percent
                )
                
            except ImportError:
                logger.debug("psutil not available for system monitoring")
            except Exception as e:
                logger.warning("Failed to collect system metrics", error=str(e))
            
            # Disk usage
            try:
                data_root = Path(settings.global_config.data_root)
                if data_root.exists():
                    import shutil
                    
                    total, used, free = shutil.disk_usage(data_root)
                    used_mb = used / (1024 * 1024)
                    free_mb = free / (1024 * 1024)
                    
                    self.metrics.record_gauge("disk_used", used_mb, "MB")
                    self.metrics.record_gauge("disk_free", free_mb, "MB")
                    
                    self.metrics.update_system_stats(disk_usage_mb=used_mb)
                    
            except Exception as e:
                logger.warning("Failed to collect disk metrics", error=str(e))
            
            # Count active matters
            try:
                from .matters import MatterManager
                
                manager = MatterManager()
                matters = await manager.list_matters()
                
                self.metrics.record_gauge("active_matters", len(matters), "count")
                self.metrics.update_system_stats(active_matters=len(matters))
                
            except Exception as e:
                logger.warning("Failed to count matters", error=str(e))
            
        except Exception as e:
            logger.error("System metrics collection failed", error=str(e))
    
    async def get_health_status(self) -> SystemHealthStatus:
        """Get current system health status."""
        try:
            status = SystemHealthStatus(
                status="healthy",
                uptime_seconds=time.time() - self.metrics.start_time,
                version=self._get_application_version()
            )
            
            # Check service health
            await self._check_service_health(status)
            
            # Check resource health
            await self._check_resource_health(status)
            
            # Determine overall status
            if any(s == "critical" for s in status.services.values()):
                status.status = "critical"
            elif any(s == "warning" for s in status.services.values()):
                status.status = "warning"
            
            self.last_health_check = datetime.now()
            status.last_check = self.last_health_check
            
            return status
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return SystemHealthStatus(
                status="critical",
                services={"health_check": "failed"},
                last_check=datetime.now()
            )
    
    async def _check_service_health(self, status: SystemHealthStatus) -> None:
        """Check health of external services."""
        
        # Check Ollama
        try:
            import ollama
            
            models = ollama.list()
            if models and 'models' in models:
                status.services["ollama"] = "healthy"
            else:
                status.services["ollama"] = "warning"
                
        except Exception:
            status.services["ollama"] = "critical"
        
        # Check ChromaDB
        try:
            import chromadb
            
            # Try to create a temporary client
            client = chromadb.Client()
            status.services["chromadb"] = "healthy"
            
        except Exception:
            status.services["chromadb"] = "critical"
        
        # Check Letta
        try:
            from .letta_adapter import LettaAdapter
            
            # This is a basic import check - could be enhanced
            status.services["letta"] = "healthy"
            
        except Exception:
            status.services["letta"] = "warning"
    
    async def _check_resource_health(self, status: SystemHealthStatus) -> None:
        """Check resource health."""
        
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            status.resources["memory"] = {
                "percent_used": memory.percent,
                "available_gb": memory.available / (1024**3),
                "status": "healthy" if memory.percent < 80 else ("warning" if memory.percent < 95 else "critical")
            }
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            status.resources["cpu"] = {
                "percent_used": cpu_percent,
                "status": "healthy" if cpu_percent < 80 else ("warning" if cpu_percent < 95 else "critical")
            }
            
        except ImportError:
            status.resources["monitoring"] = {
                "status": "warning",
                "message": "psutil not available"
            }
        except Exception as e:
            status.resources["monitoring"] = {
                "status": "warning",
                "message": f"Monitoring failed: {e}"
            }
        
        # Disk check
        try:
            import shutil
            
            data_root = Path(settings.global_config.data_root)
            if data_root.exists():
                total, used, free = shutil.disk_usage(data_root)
                free_gb = free / (1024**3)
                percent_used = (used / total) * 100
                
                status.resources["disk"] = {
                    "free_gb": free_gb,
                    "percent_used": percent_used,
                    "status": "healthy" if free_gb > 1.0 else ("warning" if free_gb > 0.5 else "critical")
                }
            
        except Exception as e:
            status.resources["disk"] = {
                "status": "warning",
                "message": f"Disk check failed: {e}"
            }
    
    def _get_application_version(self) -> str:
        """Get application version."""
        try:
            # Try to read version from package
            import pkg_resources
            return pkg_resources.get_distribution("letta-claim-assistant").version
        except:
            # Fallback to git info or default
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "describe", "--tags", "--always"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
            
            return "development"


class PerformanceProfiler:
    """Context manager for profiling operation performance."""
    
    def __init__(self, operation: str, metrics_collector: MetricsCollector, **tags):
        self.operation = operation
        self.metrics = metrics_collector
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.metrics.record_timing(self.operation, duration_ms, **self.tags)
            
            # Record success/failure
            status = "success" if exc_type is None else "error"
            self.metrics.record_counter(f"{self.operation}_result", tags={"status": status, **self.tags})


# Global instances
_metrics_collector = MetricsCollector()
_system_monitor = SystemMonitor(_metrics_collector)


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor."""
    return _system_monitor


def profile_operation(operation: str, **tags) -> PerformanceProfiler:
    """Create a performance profiler context manager."""
    return PerformanceProfiler(operation, _metrics_collector, **tags)


async def start_monitoring(interval_seconds: int = 30) -> None:
    """Start system monitoring."""
    await _system_monitor.start_monitoring(interval_seconds)


async def stop_monitoring() -> None:
    """Stop system monitoring."""
    await _system_monitor.stop_monitoring()


async def get_health_status() -> SystemHealthStatus:
    """Get current system health status."""
    return await _system_monitor.get_health_status()


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    return _metrics_collector.get_metrics_summary()


def get_recent_metrics(minutes: int = 5) -> List[Dict[str, Any]]:
    """Get recent metrics."""
    return _metrics_collector.get_recent_metrics(minutes)