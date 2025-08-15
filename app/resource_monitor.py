"""
System resource monitoring for the Letta Claim Assistant.

Monitors disk space, memory usage, network connectivity, and external service
availability to prevent resource-related failures and provide early warnings.
"""

import asyncio
import psutil
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .logging_conf import get_logger
from .error_handler import ResourceError, ServiceUnavailableError, handle_error, create_context

logger = get_logger(__name__)


class ResourceStatus(Enum):
    """Resource status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


@dataclass
class ResourceThresholds:
    """Thresholds for resource monitoring."""
    warning_level: float = 0.8   # 80%
    critical_level: float = 0.95  # 95%
    min_free_space_gb: float = 1.0  # 1GB minimum free space


@dataclass
class DiskInfo:
    """Disk space information."""
    total: int
    used: int
    free: int
    percent_used: float
    status: ResourceStatus
    path: str


@dataclass
class MemoryInfo:
    """Memory usage information."""
    total: int
    used: int
    free: int
    percent_used: float
    status: ResourceStatus
    swap_total: int = 0
    swap_used: int = 0
    swap_percent: float = 0.0


@dataclass
class NetworkInfo:
    """Network connectivity information."""
    status: ResourceStatus
    latency_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ServiceInfo:
    """External service availability information."""
    name: str
    status: ResourceStatus
    last_check: Optional[datetime] = None
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    version: Optional[str] = None


class ResourceMonitor:
    """
    Monitors system resources and external service availability.
    """
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize resource monitor.
        
        Args:
            check_interval: Seconds between automatic checks
        """
        self.check_interval = check_interval
        self.thresholds = ResourceThresholds()
        self.callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self.last_checks: Dict[str, datetime] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Cache for service status
        self.service_cache: Dict[str, ServiceInfo] = {}
        self.cache_ttl = 60  # Cache service status for 60 seconds
    
    def add_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for resource status changes."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Remove callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self, resource_type: str, info: Dict[str, Any]):
        """Notify all callbacks of resource status change."""
        for callback in self.callbacks:
            try:
                callback(resource_type, info)
            except Exception as e:
                logger.warning(f"Resource monitor callback failed: {e}")
    
    def check_disk_space(self, path: Path = None) -> DiskInfo:
        """
        Check disk space for given path.
        
        Args:
            path: Path to check (defaults to home directory)
            
        Returns:
            DiskInfo with current disk usage
        """
        if path is None:
            path = Path.home()
        
        try:
            stat = shutil.disk_usage(path)
            total = stat.total
            used = stat.used
            free = stat.free
            percent_used = used / total if total > 0 else 0
            
            # Determine status
            if percent_used >= self.thresholds.critical_level:
                status = ResourceStatus.CRITICAL
            elif percent_used >= self.thresholds.warning_level:
                status = ResourceStatus.WARNING
            elif free < (self.thresholds.min_free_space_gb * 1024**3):
                status = ResourceStatus.WARNING
            else:
                status = ResourceStatus.HEALTHY
            
            disk_info = DiskInfo(
                total=total,
                used=used,
                free=free,
                percent_used=percent_used,
                status=status,
                path=str(path)
            )
            
            # Log warnings and errors
            if status == ResourceStatus.CRITICAL:
                logger.error(
                    "Critical disk space shortage",
                    path=str(path),
                    percent_used=f"{percent_used:.1%}",
                    free_gb=free / (1024**3)
                )
            elif status == ResourceStatus.WARNING:
                logger.warning(
                    "Low disk space warning",
                    path=str(path),
                    percent_used=f"{percent_used:.1%}",
                    free_gb=free / (1024**3)
                )
            
            self._notify_callbacks("disk", disk_info.__dict__)
            return disk_info
            
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return DiskInfo(
                total=0, used=0, free=0, percent_used=1.0,
                status=ResourceStatus.UNAVAILABLE,
                path=str(path)
            )
    
    def check_memory_usage(self) -> MemoryInfo:
        """
        Check system memory usage.
        
        Returns:
            MemoryInfo with current memory usage
        """
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Determine status based on memory usage
            if memory.percent >= self.thresholds.critical_level * 100:
                status = ResourceStatus.CRITICAL
            elif memory.percent >= self.thresholds.warning_level * 100:
                status = ResourceStatus.WARNING
            else:
                status = ResourceStatus.HEALTHY
            
            memory_info = MemoryInfo(
                total=memory.total,
                used=memory.used,
                free=memory.available,
                percent_used=memory.percent / 100,
                status=status,
                swap_total=swap.total,
                swap_used=swap.used,
                swap_percent=swap.percent / 100
            )
            
            # Log warnings and errors
            if status == ResourceStatus.CRITICAL:
                logger.error(
                    "Critical memory usage",
                    percent_used=f"{memory.percent:.1f}%",
                    available_gb=memory.available / (1024**3)
                )
            elif status == ResourceStatus.WARNING:
                logger.warning(
                    "High memory usage",
                    percent_used=f"{memory.percent:.1f}%",
                    available_gb=memory.available / (1024**3)
                )
            
            self._notify_callbacks("memory", memory_info.__dict__)
            return memory_info
            
        except Exception as e:
            logger.error(f"Failed to check memory usage: {e}")
            return MemoryInfo(
                total=0, used=0, free=0, percent_used=1.0,
                status=ResourceStatus.UNAVAILABLE
            )
    
    async def check_network_connectivity(self, host: str = "8.8.8.8", timeout: float = 5.0) -> NetworkInfo:
        """
        Check network connectivity by pinging a host.
        
        Args:
            host: Host to ping
            timeout: Timeout in seconds
            
        Returns:
            NetworkInfo with connectivity status
        """
        try:
            start_time = time.time()
            
            # Use ping command for network check
            process = await asyncio.create_subprocess_exec(
                "ping", "-c", "1", "-W", str(int(timeout * 1000)), host,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout + 1
            )
            
            latency = (time.time() - start_time) * 1000
            
            if process.returncode == 0:
                status = ResourceStatus.HEALTHY
                error_message = None
            else:
                status = ResourceStatus.UNAVAILABLE
                error_message = stderr.decode().strip() if stderr else "Network unreachable"
            
            network_info = NetworkInfo(
                status=status,
                latency_ms=latency if status == ResourceStatus.HEALTHY else None,
                last_check=datetime.utcnow(),
                error_message=error_message
            )
            
            if status != ResourceStatus.HEALTHY:
                logger.warning(
                    "Network connectivity issue",
                    host=host,
                    error=error_message
                )
            
            self._notify_callbacks("network", network_info.__dict__)
            return network_info
            
        except asyncio.TimeoutError:
            network_info = NetworkInfo(
                status=ResourceStatus.UNAVAILABLE,
                last_check=datetime.utcnow(),
                error_message=f"Network check timed out after {timeout}s"
            )
            logger.warning("Network check timed out", host=host, timeout=timeout)
            self._notify_callbacks("network", network_info.__dict__)
            return network_info
        except Exception as e:
            network_info = NetworkInfo(
                status=ResourceStatus.UNAVAILABLE,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
            logger.error(f"Network check failed: {e}")
            self._notify_callbacks("network", network_info.__dict__)
            return network_info
    
    async def check_ollama_service(self) -> ServiceInfo:
        """
        Check Ollama service availability.
        
        Returns:
            ServiceInfo with Ollama status
        """
        service_name = "ollama"
        
        # Check cache first
        if service_name in self.service_cache:
            cached = self.service_cache[service_name]
            if (cached.last_check and 
                (datetime.utcnow() - cached.last_check).total_seconds() < self.cache_ttl):
                return cached
        
        try:
            import aiohttp
            
            start_time = time.time()
            timeout = aiohttp.ClientTimeout(total=10.0)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get("http://localhost:11434/api/version") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        version = data.get("version", "unknown")
                        status = ResourceStatus.HEALTHY
                        error_message = None
                    else:
                        status = ResourceStatus.UNAVAILABLE
                        error_message = f"HTTP {response.status}"
                        version = None
                        
        except ImportError:
            # Fallback to requests if aiohttp not available
            try:
                import requests
                start_time = time.time()
                response = requests.get("http://localhost:11434/api/version", timeout=10.0)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    version = data.get("version", "unknown")
                    status = ResourceStatus.HEALTHY
                    error_message = None
                else:
                    status = ResourceStatus.UNAVAILABLE
                    error_message = f"HTTP {response.status_code}"
                    version = None
            except Exception as e:
                status = ResourceStatus.UNAVAILABLE
                error_message = str(e)
                version = None
                response_time = None
        except Exception as e:
            status = ResourceStatus.UNAVAILABLE
            error_message = str(e)
            version = None
            response_time = None
        
        service_info = ServiceInfo(
            name=service_name,
            status=status,
            last_check=datetime.utcnow(),
            response_time_ms=response_time,
            error_message=error_message,
            version=version
        )
        
        # Cache the result
        self.service_cache[service_name] = service_info
        
        if status != ResourceStatus.HEALTHY:
            logger.warning(
                "Ollama service unavailable",
                error=error_message,
                response_time_ms=response_time
            )
        
        self._notify_callbacks("ollama", service_info.__dict__)
        return service_info
    
    async def check_chromadb_health(self, chroma_path: Path = None) -> ServiceInfo:
        """
        Check ChromaDB health by testing collection access.
        
        Args:
            chroma_path: Path to ChromaDB storage
            
        Returns:
            ServiceInfo with ChromaDB status
        """
        service_name = "chromadb"
        
        # Check cache first
        if service_name in self.service_cache:
            cached = self.service_cache[service_name]
            if (cached.last_check and 
                (datetime.utcnow() - cached.last_check).total_seconds() < self.cache_ttl):
                return cached
        
        try:
            import chromadb
            
            start_time = time.time()
            
            if chroma_path:
                client = chromadb.PersistentClient(path=str(chroma_path))
            else:
                client = chromadb.Client()
            
            # Test basic operation
            collections = client.list_collections()
            response_time = (time.time() - start_time) * 1000
            
            status = ResourceStatus.HEALTHY
            error_message = None
            version = chromadb.__version__ if hasattr(chromadb, '__version__') else "unknown"
            
        except Exception as e:
            status = ResourceStatus.UNAVAILABLE
            error_message = str(e)
            version = None
            response_time = None
        
        service_info = ServiceInfo(
            name=service_name,
            status=status,
            last_check=datetime.utcnow(),
            response_time_ms=response_time,
            error_message=error_message,
            version=version
        )
        
        # Cache the result
        self.service_cache[service_name] = service_info
        
        if status != ResourceStatus.HEALTHY:
            logger.warning(
                "ChromaDB service issue",
                error=error_message,
                chroma_path=str(chroma_path) if chroma_path else None
            )
        
        self._notify_callbacks("chromadb", service_info.__dict__)
        return service_info
    
    def cleanup_temp_files(self, temp_dirs: List[Path] = None, max_age_hours: int = 24):
        """
        Clean up old temporary files.
        
        Args:
            temp_dirs: Directories to clean (defaults to system temp)
            max_age_hours: Maximum age of files to keep
        """
        if temp_dirs is None:
            import tempfile
            temp_dirs = [Path(tempfile.gettempdir())]
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        files_removed = 0
        space_freed = 0
        
        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
                
            try:
                for file_path in temp_dir.glob("**/*"):
                    if not file_path.is_file():
                        continue
                    
                    # Check if file is old enough
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        try:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            space_freed += file_size
                        except Exception as e:
                            logger.debug(f"Could not remove temp file {file_path}: {e}")
            
            except Exception as e:
                logger.warning(f"Error cleaning temp directory {temp_dir}: {e}")
        
        if files_removed > 0:
            logger.info(
                "Cleaned up temporary files",
                files_removed=files_removed,
                space_freed_mb=space_freed / (1024**2)
            )
    
    async def get_comprehensive_status(self, matter_root: Path = None) -> Dict[str, Any]:
        """
        Get comprehensive status of all monitored resources.
        
        Args:
            matter_root: Root path for Matter data storage
            
        Returns:
            Dictionary with all resource statuses
        """
        if matter_root is None:
            matter_root = Path.home() / "LettaClaims"
        
        # Run checks concurrently
        tasks = [
            asyncio.create_task(self.check_network_connectivity()),
            asyncio.create_task(self.check_ollama_service()),
            asyncio.create_task(self.check_chromadb_health())
        ]
        
        # Synchronous checks
        disk_info = self.check_disk_space(matter_root)
        memory_info = self.check_memory_usage()
        
        # Wait for async checks
        network_info, ollama_info, chromadb_info = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in async checks
        if isinstance(network_info, Exception):
            network_info = NetworkInfo(status=ResourceStatus.UNAVAILABLE, error_message=str(network_info))
        if isinstance(ollama_info, Exception):
            ollama_info = ServiceInfo(name="ollama", status=ResourceStatus.UNAVAILABLE, error_message=str(ollama_info))
        if isinstance(chromadb_info, Exception):
            chromadb_info = ServiceInfo(name="chromadb", status=ResourceStatus.UNAVAILABLE, error_message=str(chromadb_info))
        
        # Determine overall health
        all_statuses = [
            disk_info.status,
            memory_info.status,
            network_info.status,
            ollama_info.status,
            chromadb_info.status
        ]
        
        if ResourceStatus.CRITICAL in all_statuses:
            overall_status = ResourceStatus.CRITICAL
        elif ResourceStatus.UNAVAILABLE in all_statuses:
            overall_status = ResourceStatus.WARNING
        elif ResourceStatus.WARNING in all_statuses:
            overall_status = ResourceStatus.WARNING
        else:
            overall_status = ResourceStatus.HEALTHY
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "disk": disk_info.__dict__,
            "memory": memory_info.__dict__,
            "network": network_info.__dict__,
            "services": {
                "ollama": ollama_info.__dict__,
                "chromadb": chromadb_info.__dict__
            }
        }
    
    async def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        logger.info("Starting resource monitoring", interval=self.check_interval)
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        logger.info("Stopped resource monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                try:
                    # Get comprehensive status
                    status = await self.get_comprehensive_status()
                    
                    # Log overall status
                    logger.debug(
                        "Resource monitoring check completed",
                        overall_status=status["overall_status"]
                    )
                    
                    # Check for critical conditions
                    if status["overall_status"] == ResourceStatus.CRITICAL.value:
                        logger.error("Critical resource conditions detected")
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
        except asyncio.CancelledError:
            logger.info("Resource monitoring loop cancelled")
            raise


# Global resource monitor instance
resource_monitor = ResourceMonitor()


def check_disk_space_before_operation(required_space_gb: float, path: Path = None) -> bool:
    """
    Check if sufficient disk space is available for an operation.
    
    Args:
        required_space_gb: Required space in GB
        path: Path to check (defaults to home directory)
        
    Returns:
        True if sufficient space available
        
    Raises:
        ResourceError: If insufficient space
    """
    disk_info = resource_monitor.check_disk_space(path)
    available_gb = disk_info.free / (1024**3)
    
    if available_gb < required_space_gb:
        raise ResourceError(
            resource_type="disk",
            message=f"Insufficient disk space: {available_gb:.1f}GB available, {required_space_gb:.1f}GB required",
            user_message=f"Not enough disk space to complete this operation",
            suggestion=f"Please free up at least {required_space_gb - available_gb:.1f}GB of disk space and try again",
            context=create_context(operation="disk_space_check", file_path=str(path) if path else None)
        )
    
    return True


async def ensure_service_available(service_name: str) -> bool:
    """
    Ensure a service is available, raising an error if not.
    
    Args:
        service_name: Name of service to check
        
    Returns:
        True if service is available
        
    Raises:
        ServiceUnavailableError: If service is not available
    """
    if service_name == "ollama":
        service_info = await resource_monitor.check_ollama_service()
    elif service_name == "chromadb":
        service_info = await resource_monitor.check_chromadb_health()
    else:
        raise ValueError(f"Unknown service: {service_name}")
    
    if service_info.status != ResourceStatus.HEALTHY:
        raise ServiceUnavailableError(
            service=service_name,
            context=create_context(operation="service_check", provider=service_name)
        )
    
    return True