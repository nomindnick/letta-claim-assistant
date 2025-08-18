"""
Unit tests for Letta Connection Manager.

Tests connection management, retry logic, health monitoring, and metrics collection.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from app.letta_connection import (
    LettaConnectionManager,
    ConnectionState,
    ConnectionMetrics
)


class TestConnectionMetrics:
    """Test suite for ConnectionMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ConnectionMetrics(window_size=10)
        
        assert metrics.window_size == 10
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.retry_count == 0
        assert metrics.last_failure_time is None
        assert metrics.last_success_time is None
        assert metrics.connection_established_time is None
    
    def test_record_operation_success(self):
        """Test recording successful operations."""
        metrics = ConnectionMetrics()
        
        metrics.record_operation("recall", 0.5, True)
        metrics.record_operation("upsert", 1.0, True)
        
        assert metrics.success_count == 2
        assert metrics.failure_count == 0
        assert metrics.last_success_time is not None
        assert len(metrics.operation_latencies["recall"]) == 1
        assert metrics.operation_latencies["recall"][0] == 0.5
    
    def test_record_operation_failure(self):
        """Test recording failed operations."""
        metrics = ConnectionMetrics()
        
        metrics.record_operation("health_check", 0.1, False)
        
        assert metrics.success_count == 0
        assert metrics.failure_count == 1
        assert metrics.last_failure_time is not None
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ConnectionMetrics()
        
        # Record mixed results
        metrics.record_operation("recall", 0.5, True)
        metrics.record_operation("recall", 0.5, True)
        metrics.record_operation("recall", 0.5, False)
        
        assert metrics.get_success_rate() == pytest.approx(66.67, rel=0.01)
    
    def test_average_latency_calculation(self):
        """Test average latency calculation."""
        metrics = ConnectionMetrics()
        
        metrics.record_operation("recall", 0.5, True)
        metrics.record_operation("recall", 1.5, True)
        metrics.record_operation("recall", 1.0, True)
        
        assert metrics.get_average_latency("recall") == 1.0
        assert metrics.get_average_latency() == 1.0
    
    def test_window_size_limiting(self):
        """Test that metrics window size is respected."""
        metrics = ConnectionMetrics(window_size=3)
        
        # Record more than window size
        for i in range(5):
            metrics.record_operation("recall", float(i), True)
        
        # Should only keep last 3
        assert len(metrics.operation_latencies["recall"]) == 3
        assert metrics.operation_latencies["recall"] == [2.0, 3.0, 4.0]


class TestLettaConnectionManager:
    """Test suite for LettaConnectionManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh connection manager instance."""
        # Reset singleton
        LettaConnectionManager._instance = None
        manager = LettaConnectionManager()
        return manager
    
    @pytest.mark.asyncio
    async def test_singleton_pattern(self, manager):
        """Test that connection manager follows singleton pattern."""
        manager2 = LettaConnectionManager()
        assert manager is manager2
    
    @pytest.mark.asyncio
    async def test_successful_connection(self, manager):
        """Test successful connection to server."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            with patch('app.letta_connection.Letta') as mock_sync_client_class:
                with patch('app.letta_connection.server_manager') as mock_server:
                    # Setup mocks
                    mock_client = AsyncMock()
                    mock_client.health.health_check = AsyncMock()
                    mock_client_class.return_value = mock_client
                    
                    mock_sync_client = Mock()
                    mock_sync_client_class.return_value = mock_sync_client
                    
                    mock_server._is_running = True
                    mock_server.get_base_url.return_value = "http://localhost:8283"
                    
                    # Test connection
                    result = await manager.connect()
                    
                    assert result is True
                    assert manager.state == ConnectionState.CONNECTED
                    assert manager.client is mock_client
                    assert manager.sync_client is mock_sync_client
                    assert manager.metrics.success_count == 1
    
    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, manager):
        """Test connection retry with exponential backoff."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            with patch('app.letta_connection.server_manager') as mock_server:
                with patch('app.letta_connection.asyncio.sleep') as mock_sleep:
                    # Setup mocks
                    mock_client = AsyncMock()
                    # Fail twice, then succeed
                    mock_client.health.health_check = AsyncMock(
                        side_effect=[Exception("Connection failed"), 
                                   Exception("Connection failed"),
                                   None]
                    )
                    mock_client_class.return_value = mock_client
                    
                    mock_server._is_running = True
                    mock_server.get_base_url.return_value = "http://localhost:8283"
                    
                    # Test connection with retries
                    result = await manager.connect()
                    
                    assert result is True
                    assert manager.state == ConnectionState.CONNECTED
                    assert manager.metrics.retry_count == 2
                    assert mock_sleep.call_count == 2
    
    @pytest.mark.asyncio
    async def test_connection_failure_after_max_retries(self, manager):
        """Test connection failure after maximum retries."""
        manager.max_retries = 2
        
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            with patch('app.letta_connection.server_manager') as mock_server:
                # Setup mocks
                mock_client = AsyncMock()
                mock_client.health.health_check = AsyncMock(
                    side_effect=Exception("Connection failed")
                )
                mock_client_class.return_value = mock_client
                
                mock_server._is_running = True
                mock_server.get_base_url.return_value = "http://localhost:8283"
                
                # Test connection failure
                result = await manager.connect()
                
                assert result is False
                assert manager.state == ConnectionState.FAILED
                assert manager.client is None
                assert manager.metrics.failure_count > 0
    
    @pytest.mark.asyncio
    async def test_fallback_mode_when_letta_unavailable(self, manager):
        """Test fallback mode when Letta client is not available."""
        with patch('app.letta_connection.LETTA_AVAILABLE', False):
            result = await manager.connect()
            
            assert result is False
            assert manager.state == ConnectionState.FALLBACK
    
    @pytest.mark.asyncio
    async def test_ensure_connected_with_healthy_connection(self, manager):
        """Test ensure_connected with already healthy connection."""
        manager.state = ConnectionState.CONNECTED
        manager.client = AsyncMock()
        manager.client.health.health_check = AsyncMock()
        
        result = await manager.ensure_connected()
        
        assert result is True
        assert manager.client.health.health_check.called
    
    @pytest.mark.asyncio
    async def test_ensure_connected_reconnects_on_failure(self, manager):
        """Test ensure_connected attempts reconnection on health check failure."""
        manager.state = ConnectionState.CONNECTED
        manager.client = AsyncMock()
        manager.client.health.health_check = AsyncMock(
            side_effect=Exception("Health check failed")
        )
        
        with patch.object(manager, 'connect', new=AsyncMock(return_value=True)):
            result = await manager.ensure_connected()
            
            assert result is True
            assert manager.connect.called
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, manager):
        """Test execute_with_retry for successful operations."""
        manager.state = ConnectionState.CONNECTED
        manager.client = AsyncMock()
        manager.client.health.health_check = AsyncMock()
        
        # Mock operation
        operation = AsyncMock(return_value="test_result")
        
        result = await manager.execute_with_retry(
            "test_operation",
            operation,
            "arg1",
            kwarg1="value1"
        )
        
        assert result == "test_result"
        assert operation.called_with("arg1", kwarg1="value1")
        assert manager.metrics.success_count == 2  # Health check + operation
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_single_retry(self, manager):
        """Test execute_with_retry retries once on failure."""
        manager.state = ConnectionState.CONNECTED
        manager.client = AsyncMock()
        manager.client.health.health_check = AsyncMock()
        
        # Mock operation that fails once then succeeds
        operation = AsyncMock(
            side_effect=[Exception("First attempt failed"), "test_result"]
        )
        
        with patch('app.letta_connection.asyncio.sleep'):
            result = await manager.execute_with_retry(
                "test_operation",
                operation
            )
        
        assert result == "test_result"
        assert operation.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_not_found_no_retry(self, manager):
        """Test execute_with_retry doesn't retry on NotFoundError."""
        manager.state = ConnectionState.CONNECTED
        manager.client = AsyncMock()
        manager.client.health.health_check = AsyncMock()
        
        with patch('app.letta_connection.NotFoundError', Exception):
            # Mock operation that raises NotFoundError
            operation = AsyncMock(
                side_effect=Exception("Not found")
            )
            
            result = await manager.execute_with_retry(
                "test_operation",
                operation
            )
        
        assert result is None
        assert operation.call_count == 1  # No retry
    
    @pytest.mark.asyncio
    async def test_client_session_context_manager(self, manager):
        """Test client_session context manager."""
        manager.state = ConnectionState.CONNECTED
        manager.client = AsyncMock()
        manager.client.health.health_check = AsyncMock()
        
        async with manager.client_session() as client:
            assert client is manager.client
    
    @pytest.mark.asyncio
    async def test_client_session_fallback(self, manager):
        """Test client_session returns None in fallback mode."""
        manager.state = ConnectionState.FALLBACK
        
        async with manager.client_session() as client:
            assert client is None
    
    def test_get_metrics(self, manager):
        """Test getting metrics dictionary."""
        manager.metrics.record_operation("recall", 0.5, True)
        manager.state = ConnectionState.CONNECTED
        manager.base_url = "http://localhost:8283"
        
        metrics = manager.get_metrics()
        
        assert metrics["state"] == "connected"
        assert metrics["base_url"] == "http://localhost:8283"
        assert "metrics" in metrics
        assert metrics["metrics"]["success_count"] == 1
    
    @pytest.mark.asyncio
    async def test_disconnect(self, manager):
        """Test disconnect functionality."""
        manager.state = ConnectionState.CONNECTED
        manager.client = AsyncMock()
        manager.sync_client = Mock()
        
        # Create a mock health check task
        manager.health_check_task = AsyncMock()
        manager.health_check_task.cancel = Mock()
        
        await manager.disconnect()
        
        assert manager.state == ConnectionState.DISCONNECTED
        assert manager.client is None
        assert manager.sync_client is None
        assert manager.health_check_task.cancel.called


class TestConnectionManagerIntegration:
    """Integration tests for connection manager with LettaAdapter."""
    
    @pytest.mark.asyncio
    async def test_adapter_uses_connection_manager(self):
        """Test that LettaAdapter properly uses connection manager."""
        from app.letta_adapter import LettaAdapter
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "test_matter"
            matter_path.mkdir(parents=True, exist_ok=True)
            
            with patch('app.letta_adapter.connection_manager') as mock_conn_mgr:
                mock_conn_mgr.connect = AsyncMock(return_value=True)
                mock_conn_mgr.client = AsyncMock()
                mock_conn_mgr.sync_client = Mock()
                mock_conn_mgr.get_state = Mock(
                    return_value=Mock(value="connected")
                )
                mock_conn_mgr.execute_with_retry = AsyncMock(
                    return_value=[]
                )
                
                adapter = LettaAdapter(
                    matter_path=matter_path,
                    matter_name="Test Matter"
                )
                
                # Test recall uses connection manager
                result = await adapter.recall("test query")
                
                assert mock_conn_mgr.connect.called
                assert isinstance(result, list)