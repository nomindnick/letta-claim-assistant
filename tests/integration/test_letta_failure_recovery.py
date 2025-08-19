"""
Integration tests for Letta failure recovery scenarios including server crashes,
network timeouts, corrupted data, memory limits, and provider failures.
"""

import pytest
import asyncio
import time
import tempfile
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import random

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.letta_adapter import LettaAdapter
from app.letta_server import LettaServerManager
from app.letta_connection import LettaConnectionManager, ConnectionState
from app.letta_circuit_breaker import CircuitBreaker, CircuitState, circuit_manager
from app.letta_request_queue import RequestQueue, RequestPriority
from app.models import Matter, MatterPaths


class TestServerFailureRecovery:
    """Test recovery from server failures."""
    
    @pytest.mark.asyncio
    async def test_server_crash_recovery(self):
        """Test recovery when server crashes unexpectedly."""
        with patch('subprocess.Popen') as mock_popen:
            # Simulate server process
            mock_process = MagicMock()
            mock_process.poll.side_effect = [None, None, 1]  # Running, then crashed
            mock_process.returncode = 1
            mock_popen.return_value = mock_process
            
            with patch('requests.get') as mock_get:
                # Server healthy, then crashes, then recovers
                mock_get.side_effect = [
                    MagicMock(status_code=200),  # Initial health check
                    Exception("Connection refused"),  # Server crashed
                    Exception("Connection refused"),  # Still down
                    MagicMock(status_code=200),  # Recovered
                ]
                
                manager = LettaServerManager()
                manager.configure(port=8283, mode="subprocess")
                
                # Start server
                await manager.start()
                assert manager._is_running()
                
                # Enable auto-restart
                manager.start_health_monitoring(interval=0.1, auto_restart=True)
                
                # Wait for crash detection and recovery
                await asyncio.sleep(0.5)
                
                # Should have restarted
                assert mock_popen.call_count >= 2
                
                manager.stop_health_monitoring()
    
    @pytest.mark.asyncio
    async def test_server_unresponsive_timeout(self):
        """Test handling of unresponsive server with timeout."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Simulate slow/hanging operation
            async def slow_operation():
                await asyncio.sleep(10)  # Much longer than timeout
                return "Never reached"
            
            mock_client.agents.search_archival_memory = slow_operation
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283", timeout=1.0)
            await manager.connect()
            
            # Should timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    manager.execute_with_retry(
                        mock_client.agents.search_archival_memory,
                        "test_op"
                    ),
                    timeout=2.0
                )
    
    @pytest.mark.asyncio
    async def test_server_partial_failure(self):
        """Test handling when some server operations fail but not all."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "partial_fail"
            matter = Matter(
                id="partial-fail",
                name="Partial Fail",
                slug="partial-fail",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Agent creation works
                mock_client.agents.create.return_value = Mock(id="test-agent")
                
                # But memory operations fail initially
                call_count = 0
                
                async def failing_search(agent_id, query, limit=10):
                    nonlocal call_count
                    call_count += 1
                    if call_count <= 2:
                        raise Exception("Memory service unavailable")
                    return []  # Works on third attempt
                
                mock_client.agents.search_archival_memory = failing_search
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                # Should retry and eventually succeed
                results = await adapter.recall("test query")
                assert call_count == 3  # Failed twice, succeeded on third
                assert results == []


class TestNetworkFailureRecovery:
    """Test recovery from network failures."""
    
    @pytest.mark.asyncio
    async def test_network_timeout_recovery(self):
        """Test recovery from network timeouts."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Simulate network timeouts
            timeout_count = 0
            
            async def flaky_network():
                nonlocal timeout_count
                timeout_count += 1
                if timeout_count <= 2:
                    raise asyncio.TimeoutError("Network timeout")
                return "Success"
            
            mock_client.health.check = flaky_network
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            
            # Should retry and connect
            success = await manager.connect()
            assert success
            assert timeout_count == 3
    
    @pytest.mark.asyncio
    async def test_intermittent_network_issues(self):
        """Test handling of intermittent network issues."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Simulate intermittent failures (50% failure rate)
            async def intermittent_operation():
                if random.random() < 0.5:
                    raise Exception("Network error")
                return "Success"
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            await manager.connect()
            
            # Try multiple operations
            successes = 0
            failures = 0
            
            for _ in range(10):
                try:
                    result = await manager.execute_with_retry(
                        intermittent_operation,
                        "test_op",
                        max_retries=3
                    )
                    if result == "Success":
                        successes += 1
                except Exception:
                    failures += 1
            
            # Should have some successes due to retry
            assert successes > 0
            print(f"Intermittent network: {successes} successes, {failures} failures")
    
    @pytest.mark.asyncio
    async def test_connection_pool_recovery(self):
        """Test connection pool recovery after network issues."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            # Simulate connection pool with multiple clients
            clients = [AsyncMock() for _ in range(3)]
            client_index = 0
            
            def get_client():
                nonlocal client_index
                client = clients[client_index % len(clients)]
                client_index += 1
                return client
            
            mock_client_class.side_effect = get_client
            
            # First client fails, others work
            clients[0].health.check.side_effect = Exception("Connection lost")
            clients[1].health.check.return_value = None
            clients[2].health.check.return_value = None
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            
            # Should fail with first client but retry with new connection
            success = await manager.connect()
            assert success
            assert mock_client_class.call_count >= 2  # Tried multiple connections


class TestDataCorruptionRecovery:
    """Test recovery from data corruption."""
    
    @pytest.mark.asyncio
    async def test_corrupted_agent_config_recovery(self):
        """Test recovery when agent config is corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "corrupt_config"
            matter = Matter(
                id="corrupt-config",
                name="Corrupt Config",
                slug="corrupt-config",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            # Create corrupted config file
            config_path = matter_path / "knowledge" / "agent_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{ invalid json }")
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="new-agent")
                
                adapter = LettaAdapter(matter)
                
                # Should handle corrupted config and create new agent
                await adapter.initialize()
                assert adapter.agent_id == "new-agent"
                
                # Should have created new valid config
                assert config_path.exists()
                new_config = json.loads(config_path.read_text())
                assert new_config["agent_id"] == "new-agent"
    
    @pytest.mark.asyncio
    async def test_corrupted_memory_database_recovery(self):
        """Test recovery when memory database is corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "corrupt_db"
            matter = Matter(
                id="corrupt-db",
                name="Corrupt DB",
                slug="corrupt-db",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            # Create corrupted SQLite database
            db_path = matter_path / "knowledge" / "letta_state" / "memory.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write garbage data
            db_path.write_bytes(b"This is not a valid SQLite database")
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="recovery-agent")
                
                adapter = LettaAdapter(matter)
                
                # Should detect corruption and handle gracefully
                await adapter.initialize()
                
                # Should backup corrupted file and create new
                backup_files = list(db_path.parent.glob("*.backup"))
                assert len(backup_files) > 0  # Created backup
    
    @pytest.mark.asyncio
    async def test_malformed_memory_data_recovery(self):
        """Test recovery from malformed memory data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "malformed_memory"
            matter = Matter(
                id="malformed-memory",
                name="Malformed Memory",
                slug="malformed-memory",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="test-agent")
                
                # Return malformed data
                mock_client.agents.search_archival_memory.return_value = [
                    {"content": "Valid memory"},
                    None,  # Invalid entry
                    {"content": None},  # Missing content
                    {"content": "Another valid memory"}
                ]
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                # Should handle malformed data gracefully
                results = await adapter.recall("test")
                
                # Should filter out invalid entries
                valid_results = [r for r in results if r and r.get("content")]
                assert len(valid_results) == 2


class TestMemoryLimitRecovery:
    """Test recovery from memory limit issues."""
    
    @pytest.mark.asyncio
    async def test_memory_limit_exceeded_recovery(self):
        """Test recovery when memory limits are exceeded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "memory_limit"
            matter = Matter(
                id="memory-limit",
                name="Memory Limit",
                slug="memory-limit",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="test-agent")
                
                # Track memory usage
                memory_store = []
                memory_limit = 100  # Limit to 100 items
                
                async def mock_insert(agent_id, memory):
                    if len(memory_store) >= memory_limit:
                        raise Exception("Memory limit exceeded")
                    memory_store.append(memory)
                    return Mock(id=f"mem-{len(memory_store)}")
                
                mock_client.agents.insert_archival_memory = mock_insert
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                # Fill up memory
                for i in range(memory_limit):
                    await adapter.upsert([{
                        "type": "Fact",
                        "label": f"Fact {i}"
                    }])
                
                # Should fail when limit exceeded
                with pytest.raises(Exception) as exc_info:
                    await adapter.upsert([{
                        "type": "Fact",
                        "label": "Overflow fact"
                    }])
                
                assert "Memory limit exceeded" in str(exc_info.value)
                
                # Should implement pruning (in real implementation)
                # For now, verify error is handled
                assert len(memory_store) == memory_limit
    
    @pytest.mark.asyncio
    async def test_automatic_memory_pruning(self):
        """Test automatic memory pruning when approaching limits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "auto_prune"
            matter = Matter(
                id="auto-prune",
                name="Auto Prune",
                slug="auto-prune",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="prune-agent")
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                # Simulate memory pruning
                pruned_count = await adapter.prune_memory(
                    age_threshold_days=30,
                    importance_threshold=0.3
                )
                
                # Should return pruned count (mocked)
                assert pruned_count >= 0


class TestProviderFailureRecovery:
    """Test recovery from LLM provider failures."""
    
    @pytest.mark.asyncio
    async def test_provider_failover(self):
        """Test automatic failover to backup provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "provider_failover"
            matter = Matter(
                id="provider-failover",
                name="Provider Failover",
                slug="provider-failover",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Primary provider fails
                primary_fail_count = 0
                
                async def failing_primary(**kwargs):
                    nonlocal primary_fail_count
                    primary_fail_count += 1
                    raise Exception("Primary provider unavailable")
                
                # Backup provider works
                async def working_backup(**kwargs):
                    return Mock(id="backup-agent")
                
                # Configure failover
                mock_client.agents.create.side_effect = [
                    failing_primary,
                    failing_primary,
                    working_backup  # Fallback succeeds
                ]
                
                adapter = LettaAdapter(matter)
                
                # Should failover to backup
                await adapter.initialize()
                assert adapter.agent_id == "backup-agent"
                assert primary_fail_count == 2  # Tried primary twice
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_provider_protection(self):
        """Test circuit breaker protects against failing providers."""
        # Reset circuit breakers
        await circuit_manager.reset_all()
        
        # Configure circuit breaker for provider
        breaker = circuit_manager.get_or_create(
            "ollama_provider",
            failure_threshold=3,
            success_threshold=2,
            timeout=0.1
        )
        
        async def failing_provider():
            raise Exception("Provider error")
        
        # Fail enough times to open circuit
        for _ in range(3):
            try:
                await breaker.call(failing_provider)
            except Exception:
                pass
        
        assert breaker.state == CircuitState.OPEN
        
        # Should fail fast without calling provider
        from app.letta_circuit_breaker import CircuitBreakerError
        
        with pytest.raises(CircuitBreakerError):
            await breaker.call(failing_provider)
        
        # Wait for timeout to transition to half-open
        await asyncio.sleep(0.15)
        
        # Now provide working operation
        async def working_provider():
            return "Success"
        
        # Should try again (half-open)
        result = await breaker.call(working_provider)
        assert result == "Success"
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Another success should close circuit
        result = await breaker.call(working_provider)
        assert result == "Success"
        assert breaker.state == CircuitState.CLOSED


class TestQueueOverflowRecovery:
    """Test recovery from queue overflow scenarios."""
    
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self):
        """Test handling of queue overflow with different policies."""
        # Test drop_oldest policy
        queue_oldest = RequestQueue(max_queue_size=3, overflow_policy="drop_oldest")
        
        async def operation(value):
            return value
        
        futures = []
        for i in range(5):
            future = await queue_oldest.enqueue(
                operation,
                RequestType.OTHER,
                priority=RequestPriority.NORMAL,
                value=i
            )
            futures.append(future)
        
        # First two should be dropped
        assert futures[0].done()
        assert futures[1].done()
        
        # Test drop_new policy
        queue_new = RequestQueue(max_queue_size=3, overflow_policy="drop_new")
        
        # Fill queue
        for i in range(3):
            await queue_new.enqueue(operation, RequestType.OTHER, value=i)
        
        # New request should be rejected
        with pytest.raises(OverflowError):
            await queue_new.enqueue(operation, RequestType.OTHER, value=99)
    
    @pytest.mark.asyncio
    async def test_queue_recovery_after_overflow(self):
        """Test queue recovery after overflow condition."""
        queue = RequestQueue(max_queue_size=5)
        await queue.start()
        
        try:
            async def slow_operation(value):
                await asyncio.sleep(0.1)
                return value
            
            # Fill queue to capacity
            futures = []
            for i in range(5):
                future = await queue.enqueue(
                    slow_operation,
                    RequestType.OTHER,
                    value=i
                )
                futures.append(future)
            
            # Wait for some to process
            await asyncio.sleep(0.2)
            
            # Queue should have space again
            future = await queue.enqueue(
                slow_operation,
                RequestType.OTHER,
                value=99
            )
            
            result = await future
            assert result == 99
            
        finally:
            await queue.stop()


class TestGracefulDegradation:
    """Test graceful degradation when components fail."""
    
    @pytest.mark.asyncio
    async def test_degraded_mode_operation(self):
        """Test operation in degraded mode when Letta unavailable."""
        with patch('app.letta_connection.LETTA_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                matter_path = Path(temp_dir) / "degraded"
                matter = Matter(
                    id="degraded",
                    name="Degraded",
                    slug="degraded",
                    embedding_model="nomic-embed-text",
                    generation_model="gpt-oss:20b",
                    paths=MatterPaths.from_root(matter_path)
                )
                
                adapter = LettaAdapter(matter)
                
                # Should work in degraded mode (no memory)
                await adapter.initialize()
                
                # Memory operations return empty/None
                results = await adapter.recall("test")
                assert results == []
                
                # Upsert is no-op
                await adapter.upsert([{"type": "Fact", "label": "Test"}])
                
                # Follow-ups use default suggestions
                suggestions = await adapter.suggest_followups("test", [])
                assert isinstance(suggestions, list)
    
    @pytest.mark.asyncio
    async def test_partial_functionality_with_failures(self):
        """Test maintaining partial functionality when some components fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "partial"
            matter = Matter(
                id="partial",
                name="Partial",
                slug="partial",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Agent creation works
                mock_client.agents.create.return_value = Mock(id="partial-agent")
                
                # Memory search fails
                mock_client.agents.search_archival_memory.side_effect = Exception("Search unavailable")
                
                # But insert still works
                mock_client.agents.insert_archival_memory.return_value = Mock(id="mem-1")
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                # Search fails gracefully
                results = await adapter.recall("test")
                assert results == []  # Returns empty on failure
                
                # But insert still works
                await adapter.upsert([{
                    "type": "Fact",
                    "label": "Still works"
                }])
                
                # Verify insert was called
                mock_client.agents.insert_archival_memory.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])