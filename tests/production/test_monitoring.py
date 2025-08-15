"""
Tests for production monitoring and metrics collection.

Verifies that the monitoring system correctly collects metrics,
tracks performance, and reports system health.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock

from app.monitoring import (
    MetricsCollector,
    SystemMonitor,
    PerformanceProfiler,
    PerformanceMetric,
    SystemHealthStatus,
    get_metrics_collector,
    get_system_monitor,
    profile_operation,
    get_health_status,
    get_metrics_summary
)


class TestPerformanceMetric:
    """Test the PerformanceMetric data class."""
    
    def test_performance_metric_creation(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            name="test_operation",
            value=123.45,
            unit="ms",
            tags={"component": "test"}
        )
        
        assert metric.name == "test_operation"
        assert metric.value == 123.45
        assert metric.unit == "ms"
        assert metric.tags == {"component": "test"}
        assert metric.timestamp is not None
    
    def test_performance_metric_to_dict(self):
        """Test converting metric to dictionary."""
        metric = PerformanceMetric(
            name="test_operation",
            value=100.0,
            unit="count"
        )
        
        metric_dict = metric.to_dict()
        
        assert isinstance(metric_dict, dict)
        assert metric_dict["name"] == "test_operation"
        assert metric_dict["value"] == 100.0
        assert metric_dict["unit"] == "count"
        assert "timestamp" in metric_dict


class TestSystemHealthStatus:
    """Test the SystemHealthStatus data class."""
    
    def test_system_health_creation(self):
        """Test creating system health status."""
        status = SystemHealthStatus(
            status="healthy",
            services={"ollama": "healthy"},
            resources={"memory": {"status": "healthy"}},
            uptime_seconds=3600.0,
            version="1.0.0"
        )
        
        assert status.status == "healthy"
        assert status.services == {"ollama": "healthy"}
        assert status.resources == {"memory": {"status": "healthy"}}
        assert status.uptime_seconds == 3600.0
        assert status.version == "1.0.0"
    
    def test_system_health_to_dict(self):
        """Test converting health status to dictionary."""
        status = SystemHealthStatus(status="warning")
        status_dict = status.to_dict()
        
        assert isinstance(status_dict, dict)
        assert status_dict["status"] == "warning"
        assert "last_check" in status_dict
        assert "uptime_seconds" in status_dict


class TestMetricsCollector:
    """Test the metrics collector."""
    
    def setup_method(self):
        """Set up test environment."""
        self.collector = MetricsCollector(max_history=100)
    
    def test_record_metric(self):
        """Test recording a metric."""
        metric = PerformanceMetric(
            name="test_metric",
            value=42.0,
            unit="ms"
        )
        
        self.collector.record_metric(metric)
        
        assert len(self.collector.metrics) == 1
        assert "test_metric" in self.collector.aggregated_metrics
        assert self.collector.aggregated_metrics["test_metric"] == [42.0]
    
    def test_record_timing(self):
        """Test recording a timing metric."""
        self.collector.record_timing("operation", 150.5, component="test")
        
        assert len(self.collector.metrics) == 2  # timing + counter
        timing_metrics = [m for m in self.collector.metrics if "duration" in m.name]
        assert len(timing_metrics) == 1
        assert timing_metrics[0].value == 150.5
        assert timing_metrics[0].unit == "ms"
        assert timing_metrics[0].tags["component"] == "test"
    
    def test_record_counter(self):
        """Test recording a counter metric."""
        self.collector.record_counter("requests", 5, endpoint="/api/test")
        
        assert len(self.collector.metrics) == 1
        counter_metric = self.collector.metrics[0]
        assert counter_metric.name == "requests"
        assert counter_metric.value == 5
        assert counter_metric.unit == "count"
        assert counter_metric.tags["endpoint"] == "/api/test"
    
    def test_record_gauge(self):
        """Test recording a gauge metric."""
        self.collector.record_gauge("memory_usage", 75.5, "percent", service="app")
        
        assert len(self.collector.metrics) == 1
        gauge_metric = self.collector.metrics[0]
        assert gauge_metric.name == "memory_usage"
        assert gauge_metric.value == 75.5
        assert gauge_metric.unit == "percent"
        assert gauge_metric.tags["service"] == "app"
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        # Record some metrics
        self.collector.record_timing("op1", 100.0)
        self.collector.record_timing("op1", 200.0)
        self.collector.record_counter("events", 1)
        
        summary = self.collector.get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert "total_metrics" in summary
        assert "uptime_seconds" in summary
        assert "aggregated" in summary
        
        assert summary["total_metrics"] >= 3
        assert "op1_duration" in summary["aggregated"]
        
        op1_stats = summary["aggregated"]["op1_duration"]
        assert op1_stats["count"] == 2
        assert op1_stats["avg"] == 150.0
        assert op1_stats["min"] == 100.0
        assert op1_stats["max"] == 200.0
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics."""
        # Record a metric
        self.collector.record_gauge("test", 1.0)
        
        # Get recent metrics
        recent = self.collector.get_recent_metrics(minutes=1)
        
        assert isinstance(recent, list)
        assert len(recent) == 1
        assert recent[0]["name"] == "test"
        assert recent[0]["value"] == 1.0
    
    def test_update_ingestion_stats(self):
        """Test updating ingestion statistics."""
        self.collector.update_ingestion_stats(
            pages_processed=10,
            chunks_created=25
        )
        
        assert self.collector.ingestion_stats["pages_processed"] == 10
        assert self.collector.ingestion_stats["chunks_created"] == 25
    
    def test_update_rag_stats(self):
        """Test updating RAG statistics."""
        self.collector.update_rag_stats(
            queries_processed=5,
            total_response_time=250.0
        )
        
        assert self.collector.rag_stats["queries_processed"] == 5
        assert self.collector.rag_stats["total_response_time"] == 250.0
    
    def test_max_history_limit(self):
        """Test that metrics history is limited."""
        collector = MetricsCollector(max_history=3)
        
        # Record more metrics than history limit
        for i in range(5):
            collector.record_gauge(f"metric_{i}", float(i))
        
        # Should only keep last 3
        assert len(collector.metrics) == 3
        assert collector.metrics[-1].name == "metric_4"


class TestSystemMonitor:
    """Test the system monitor."""
    
    def setup_method(self):
        """Set up test environment."""
        self.collector = MetricsCollector()
        self.monitor = SystemMonitor(self.collector)
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        assert not self.monitor._monitoring
        
        # Start monitoring
        await self.monitor.start_monitoring(interval_seconds=0.1)
        assert self.monitor._monitoring
        
        # Let it run briefly
        await asyncio.sleep(0.15)
        
        # Stop monitoring
        await self.monitor.stop_monitoring()
        assert not self.monitor._monitoring
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self):
        """Test collecting system metrics."""
        with patch('psutil.Process') as mock_process:
            # Mock process memory info
            mock_memory = MagicMock()
            mock_memory.rss = 100 * 1024 * 1024  # 100MB
            mock_process.return_value.memory_info.return_value = mock_memory
            
            # Mock system memory
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.percent = 60.0
                
                with patch('psutil.cpu_percent', return_value=25.0):
                    await self.monitor.collect_system_metrics()
        
        # Check that metrics were recorded
        assert len(self.collector.metrics) > 0
        metric_names = [m.name for m in self.collector.metrics]
        assert "memory_usage" in metric_names
    
    @pytest.mark.asyncio
    async def test_get_health_status(self):
        """Test getting health status."""
        with patch.object(self.monitor, '_check_service_health'), \
             patch.object(self.monitor, '_check_resource_health'):
            
            status = await self.monitor.get_health_status()
            
            assert isinstance(status, SystemHealthStatus)
            assert status.status in ["healthy", "warning", "critical"]
            assert isinstance(status.services, dict)
            assert isinstance(status.resources, dict)
    
    @pytest.mark.asyncio
    async def test_check_service_health_ollama(self):
        """Test checking Ollama service health."""
        status = SystemHealthStatus(status="healthy")
        
        with patch('importlib.import_module') as mock_import:
            mock_ollama = MagicMock()
            mock_ollama.list.return_value = {"models": [{"name": "test"}]}
            mock_import.return_value = mock_ollama
            
            await self.monitor._check_service_health(status)
            
            assert "ollama" in status.services
            assert status.services["ollama"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_check_service_health_ollama_failed(self):
        """Test checking Ollama service health when failed."""
        status = SystemHealthStatus(status="healthy")
        
        with patch('importlib.import_module') as mock_import:
            mock_ollama = MagicMock()
            mock_ollama.list.side_effect = Exception("Connection failed")
            mock_import.return_value = mock_ollama
            
            await self.monitor._check_service_health(status)
            
            assert "ollama" in status.services
            assert status.services["ollama"] == "critical"
    
    @pytest.mark.asyncio
    async def test_check_resource_health(self):
        """Test checking resource health."""
        status = SystemHealthStatus(status="healthy")
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 50.0
            mock_memory.return_value.available = 4 * 1024**3  # 4GB
            
            with patch('psutil.cpu_percent', return_value=30.0):
                await self.monitor._check_resource_health(status)
            
            assert "memory" in status.resources
            assert status.resources["memory"]["status"] == "healthy"
            assert "cpu" in status.resources
            assert status.resources["cpu"]["status"] == "healthy"


class TestPerformanceProfiler:
    """Test the performance profiler context manager."""
    
    def test_profiler_context_manager(self):
        """Test profiler as context manager."""
        collector = MetricsCollector()
        
        with profile_operation("test_operation", component="test") as profiler:
            # Simulate some work
            time.sleep(0.01)
        
        # Should have recorded timing metric
        timing_metrics = [m for m in collector.metrics if "test_operation_duration" in m.name]
        assert len(timing_metrics) == 1
        assert timing_metrics[0].value > 0
        assert timing_metrics[0].tags["component"] == "test"
    
    def test_profiler_with_exception(self):
        """Test profiler when exception occurs."""
        collector = MetricsCollector()
        
        try:
            with profile_operation("failing_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should have recorded timing and error metrics
        assert len(collector.metrics) >= 2
        
        error_metrics = [m for m in collector.metrics if "failing_operation_result" in m.name]
        assert len(error_metrics) == 1
        assert error_metrics[0].tags["status"] == "error"


class TestMonitoringGlobalFunctions:
    """Test global monitoring functions."""
    
    def test_get_metrics_collector(self):
        """Test getting global metrics collector."""
        collector = get_metrics_collector()
        assert isinstance(collector, MetricsCollector)
    
    def test_get_system_monitor(self):
        """Test getting global system monitor."""
        monitor = get_system_monitor()
        assert isinstance(monitor, SystemMonitor)
    
    @pytest.mark.asyncio
    async def test_get_health_status_global(self):
        """Test getting health status globally."""
        with patch.object(SystemMonitor, 'get_health_status') as mock_health:
            mock_health.return_value = SystemHealthStatus(status="healthy")
            
            status = await get_health_status()
            assert isinstance(status, SystemHealthStatus)
    
    def test_get_metrics_summary_global(self):
        """Test getting metrics summary globally."""
        summary = get_metrics_summary()
        assert isinstance(summary, dict)
        assert "total_metrics" in summary


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring system."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self):
        """Test complete monitoring cycle."""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)
        
        # Start monitoring briefly
        await monitor.start_monitoring(interval_seconds=0.1)
        await asyncio.sleep(0.15)
        await monitor.stop_monitoring()
        
        # Check that metrics were collected
        summary = collector.get_metrics_summary()
        assert summary["total_metrics"] > 0
    
    @pytest.mark.asyncio
    async def test_real_health_check(self):
        """Test health check with real system."""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)
        
        status = await monitor.get_health_status()
        
        assert isinstance(status, SystemHealthStatus)
        assert status.status in ["healthy", "warning", "critical"]
        assert isinstance(status.uptime_seconds, float)
        assert status.uptime_seconds >= 0
    
    def test_real_metrics_collection(self):
        """Test metrics collection with real operations."""
        collector = MetricsCollector()
        
        # Record various metrics
        collector.record_timing("test_operation", 100.0)
        collector.record_counter("test_events", 5)
        collector.record_gauge("test_value", 75.0)
        
        # Get summary
        summary = collector.get_metrics_summary()
        
        assert summary["total_metrics"] == 3
        assert "test_operation_duration" in summary["aggregated"]
        assert "test_events" in summary["aggregated"]
        assert "test_value" in summary["aggregated"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])