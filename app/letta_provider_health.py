"""
Letta Provider Health Monitoring and Fallback System.

Monitors provider health, tracks performance metrics, and manages automatic
fallback when providers become unavailable or perform poorly.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
from pathlib import Path
import json

from .letta_provider_bridge import letta_provider_bridge, ProviderConfiguration
from .logging_conf import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Provider health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """Health metrics for a provider."""
    
    provider_name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    # Performance metrics
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    
    # Thresholds
    max_response_time_ms: float = 5000  # 5 seconds
    max_consecutive_failures: int = 3
    degraded_threshold_ms: float = 3000  # 3 seconds
    
    def add_success(self, response_time_ms: float) -> None:
        """Record a successful health check."""
        self.last_check = datetime.now()
        self.last_success = datetime.now()
        self.success_count += 1
        self.consecutive_failures = 0
        
        # Keep last 100 response times
        self.response_times.append(response_time_ms)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        # Update status based on response time
        if response_time_ms > self.max_response_time_ms:
            self.status = HealthStatus.DEGRADED
        elif response_time_ms > self.degraded_threshold_ms:
            self.status = HealthStatus.DEGRADED
        else:
            self.status = HealthStatus.HEALTHY
    
    def add_failure(self, error: Optional[str] = None) -> None:
        """Record a failed health check."""
        self.last_check = datetime.now()
        self.last_failure = datetime.now()
        self.failure_count += 1
        self.consecutive_failures += 1
        
        # Update status based on consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.status = HealthStatus.UNHEALTHY
        else:
            self.status = HealthStatus.DEGRADED
    
    def get_average_response_time(self) -> float:
        """Get average response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    def get_p95_response_time(self) -> float:
        """Get 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    def get_success_rate(self) -> float:
        """Get success rate as a percentage."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100
    
    def is_healthy(self) -> bool:
        """Check if provider is healthy enough to use."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider_name": self.provider_name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "average_response_time_ms": self.get_average_response_time(),
            "p95_response_time_ms": self.get_p95_response_time(),
            "success_rate": self.get_success_rate(),
            "consecutive_failures": self.consecutive_failures,
            "total_checks": self.success_count + self.failure_count
        }


class LettaProviderHealth:
    """
    Health monitoring and automatic fallback for Letta providers.
    """
    
    def __init__(self):
        """Initialize health monitoring system."""
        self.metrics: Dict[str, HealthMetrics] = {}
        self.health_check_interval = 60  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self.fallback_callbacks: List[Callable] = []
        
        # Persistence
        self.metrics_file = Path.home() / ".letta-claim" / "provider_health.json"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_metrics()
        logger.info("Provider health monitoring initialized")
    
    def _load_metrics(self) -> None:
        """Load saved health metrics from disk."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    
                for provider_name, metrics_data in data.items():
                    metrics = HealthMetrics(provider_name=provider_name)
                    
                    # Restore status
                    if "status" in metrics_data:
                        try:
                            metrics.status = HealthStatus(metrics_data["status"])
                        except ValueError:
                            metrics.status = HealthStatus.UNKNOWN
                    
                    # Restore counts
                    metrics.success_count = metrics_data.get("success_count", 0)
                    metrics.failure_count = metrics_data.get("failure_count", 0)
                    metrics.consecutive_failures = metrics_data.get("consecutive_failures", 0)
                    
                    # Restore timestamps
                    if metrics_data.get("last_check"):
                        metrics.last_check = datetime.fromisoformat(metrics_data["last_check"])
                    if metrics_data.get("last_success"):
                        metrics.last_success = datetime.fromisoformat(metrics_data["last_success"])
                    if metrics_data.get("last_failure"):
                        metrics.last_failure = datetime.fromisoformat(metrics_data["last_failure"])
                    
                    self.metrics[provider_name] = metrics
                    
                logger.debug(f"Loaded health metrics for {len(self.metrics)} providers")
                
            except Exception as e:
                logger.warning(f"Could not load health metrics: {e}")
    
    def _save_metrics(self) -> None:
        """Save health metrics to disk."""
        try:
            data = {}
            for provider_name, metrics in self.metrics.items():
                data[provider_name] = {
                    "status": metrics.status.value,
                    "last_check": metrics.last_check.isoformat() if metrics.last_check else None,
                    "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
                    "last_failure": metrics.last_failure.isoformat() if metrics.last_failure else None,
                    "success_count": metrics.success_count,
                    "failure_count": metrics.failure_count,
                    "consecutive_failures": metrics.consecutive_failures
                }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save health metrics: {e}")
    
    def get_or_create_metrics(self, provider_name: str) -> HealthMetrics:
        """Get or create health metrics for a provider."""
        if provider_name not in self.metrics:
            self.metrics[provider_name] = HealthMetrics(provider_name=provider_name)
        return self.metrics[provider_name]
    
    async def check_provider_health(
        self,
        provider_config: ProviderConfiguration,
        test_func: Optional[Callable] = None
    ) -> bool:
        """
        Check health of a specific provider.
        
        Args:
            provider_config: Provider configuration to test
            test_func: Optional custom test function
            
        Returns:
            True if provider is healthy
        """
        metrics = self.get_or_create_metrics(provider_config.model_name)
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            if test_func:
                # Use custom test function
                success = await test_func(provider_config)
            else:
                # Default test - try to create configs
                llm_config, embed_config = letta_provider_bridge.create_letta_configs(provider_config)
                success = llm_config is not None
            
            end_time = asyncio.get_event_loop().time()
            response_time_ms = (end_time - start_time) * 1000
            
            if success:
                metrics.add_success(response_time_ms)
                logger.debug(
                    f"Health check passed for {provider_config.model_name}",
                    response_time_ms=round(response_time_ms, 2)
                )
            else:
                metrics.add_failure("Test returned False")
                logger.warning(f"Health check failed for {provider_config.model_name}")
            
            self._save_metrics()
            return success
            
        except Exception as e:
            metrics.add_failure(str(e))
            logger.error(f"Health check error for {provider_config.model_name}: {e}")
            self._save_metrics()
            return False
    
    async def monitor_active_provider(self) -> None:
        """Monitor the currently active provider."""
        if not letta_provider_bridge.active_provider:
            return
        
        healthy = await self.check_provider_health(letta_provider_bridge.active_provider)
        
        if not healthy:
            logger.warning(f"Active provider {letta_provider_bridge.active_provider.model_name} is unhealthy")
            await self.trigger_fallback()
    
    async def trigger_fallback(self) -> bool:
        """
        Trigger fallback to next available provider.
        
        Returns:
            True if fallback successful
        """
        next_provider = letta_provider_bridge.get_next_provider()
        
        if not next_provider:
            logger.error("No fallback providers available")
            return False
        
        # Check health of next provider before switching
        healthy = await self.check_provider_health(next_provider)
        
        if not healthy:
            logger.warning(f"Fallback provider {next_provider.model_name} is also unhealthy")
            # Try next in chain
            letta_provider_bridge.active_provider = next_provider
            return await self.trigger_fallback()
        
        # Switch to healthy provider
        success = letta_provider_bridge.switch_to_provider(next_provider)
        
        if success:
            logger.info(f"Successfully failed over to {next_provider.model_name}")
            
            # Notify callbacks
            for callback in self.fallback_callbacks:
                try:
                    await callback(next_provider)
                except Exception as e:
                    logger.error(f"Fallback callback error: {e}")
        
        return success
    
    def register_fallback_callback(self, callback: Callable) -> None:
        """
        Register a callback to be called when fallback occurs.
        
        Args:
            callback: Async function to call on fallback
        """
        self.fallback_callbacks.append(callback)
    
    async def start_monitoring(self, interval: Optional[int] = None) -> None:
        """
        Start background health monitoring.
        
        Args:
            interval: Check interval in seconds (default: 60)
        """
        if self.health_check_task and not self.health_check_task.done():
            logger.warning("Health monitoring already running")
            return
        
        self.health_check_interval = interval or 60
        self.health_check_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started health monitoring with {self.health_check_interval}s interval")
    
    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.monitor_active_provider()
                
                # Also check fallback chain health
                if letta_provider_bridge.fallback_chain:
                    for provider in letta_provider_bridge.fallback_chain:
                        if provider != letta_provider_bridge.active_provider:
                            # Check in background without switching
                            asyncio.create_task(self.check_provider_health(provider))
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get summary of all provider health metrics.
        
        Returns:
            Dictionary with health information for all providers
        """
        summary = {
            "providers": {},
            "active_provider": None,
            "healthy_count": 0,
            "unhealthy_count": 0,
            "last_check": None
        }
        
        for provider_name, metrics in self.metrics.items():
            summary["providers"][provider_name] = metrics.to_dict()
            
            if metrics.is_healthy():
                summary["healthy_count"] += 1
            else:
                summary["unhealthy_count"] += 1
            
            if metrics.last_check:
                if summary["last_check"] is None or metrics.last_check > summary["last_check"]:
                    summary["last_check"] = metrics.last_check
        
        if letta_provider_bridge.active_provider:
            summary["active_provider"] = letta_provider_bridge.active_provider.model_name
        
        if summary["last_check"]:
            summary["last_check"] = summary["last_check"].isoformat()
        
        return summary
    
    def get_provider_health(self, provider_name: str) -> Optional[HealthMetrics]:
        """
        Get health metrics for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Health metrics or None if not found
        """
        return self.metrics.get(provider_name)
    
    def reset_provider_health(self, provider_name: str) -> None:
        """
        Reset health metrics for a provider.
        
        Args:
            provider_name: Name of the provider to reset
        """
        if provider_name in self.metrics:
            self.metrics[provider_name] = HealthMetrics(provider_name=provider_name)
            self._save_metrics()
            logger.info(f"Reset health metrics for {provider_name}")
    
    async def test_all_providers(self) -> Dict[str, bool]:
        """
        Test health of all configured providers.
        
        Returns:
            Dictionary mapping provider names to health status
        """
        results = {}
        
        # Test providers in fallback chain
        if letta_provider_bridge.fallback_chain:
            for provider in letta_provider_bridge.fallback_chain:
                healthy = await self.check_provider_health(provider)
                results[provider.model_name] = healthy
        
        return results


# Global health monitor instance
provider_health_monitor = LettaProviderHealth()