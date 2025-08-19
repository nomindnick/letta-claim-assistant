"""
Integration tests for Letta server lifecycle management, multi-matter isolation,
concurrent operations, and data persistence.
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.letta_server import LettaServerManager
from app.letta_config import LettaConfigManager
from app.letta_connection import LettaConnectionManager, ConnectionState
from app.letta_adapter import LettaAdapter
from app.models import Matter, MatterPaths


class TestServerLifecycle:
    """Test Letta server lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_server_startup_and_shutdown(self):
        """Test server starts and stops cleanly."""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Process is running
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            with patch('requests.get') as mock_get:
                # Health check succeeds
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"status": "healthy"}
                mock_get.return_value = mock_response
                
                manager = LettaServerManager()
                manager.configure(port=8283, mode="subprocess")
                
                # Start server
                success = await manager.start()
                assert success
                assert manager._is_running()
                
                # Verify subprocess was started with correct args
                mock_popen.assert_called_once()
                call_args = mock_popen.call_args[0][0]
                assert "letta" in call_args
                assert "server" in call_args
                assert "--port" in call_args
                assert "8283" in str(call_args)
                
                # Stop server
                await manager.stop()
                mock_process.terminate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_server_port_conflict_resolution(self):
        """Test server finds alternative port on conflict."""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            
            with patch('requests.get') as mock_get:
                # First port fails, second succeeds
                mock_get.side_effect = [
                    Exception("Port in use"),  # 8283 fails
                    MagicMock(status_code=200)  # 8284 succeeds
                ]
                
                manager = LettaServerManager()
                manager.configure(port=8283, mode="subprocess")
                
                # Should try alternative port
                success = await manager.start()
                assert success
                
                # Should have tried both ports
                assert mock_get.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_server_health_monitoring(self):
        """Test server health monitoring and auto-restart."""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            
            with patch('requests.get') as mock_get:
                # Health check succeeds initially
                mock_get.return_value = MagicMock(status_code=200)
                
                manager = LettaServerManager()
                manager.configure(port=8283, mode="subprocess")
                await manager.start()
                
                # Start health monitoring
                manager.start_health_monitoring(interval=0.1)
                
                # Wait for health checks
                await asyncio.sleep(0.3)
                
                # Should have performed multiple health checks
                assert mock_get.call_count >= 3
                
                # Simulate server failure
                mock_get.side_effect = Exception("Server down")
                await asyncio.sleep(0.2)
                
                # Should attempt restart
                assert mock_popen.call_count >= 2  # Original + restart
                
                manager.stop_health_monitoring()
    
    @pytest.mark.asyncio
    async def test_server_configuration_persistence(self):
        """Test server configuration is persisted and loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "letta_config.json"
            
            config_manager = LettaConfigManager()
            
            # Save configuration
            config = {
                "server": {
                    "host": "localhost",
                    "port": 8283,
                    "mode": "subprocess"
                },
                "client": {
                    "base_url": "http://localhost:8283",
                    "timeout": 30
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # Load configuration
            loaded_config = config_manager.load_config(config_path)
            assert loaded_config["server"]["port"] == 8283
            assert loaded_config["client"]["timeout"] == 30


class TestMultiMatterIsolation:
    """Test multi-matter agent isolation."""
    
    @pytest.mark.asyncio
    async def test_matter_agent_isolation(self):
        """Test each matter has isolated agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create two matters
            matter1_path = Path(temp_dir) / "matter1"
            matter2_path = Path(temp_dir) / "matter2"
            
            matter1 = Matter(
                id="matter-1",
                name="Matter One",
                slug="matter-one",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter1_path)
            )
            
            matter2 = Matter(
                id="matter-2",
                name="Matter Two",
                slug="matter-two",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter2_path)
            )
            
            # Create adapters with mock client
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock agent creation
                mock_client.agents.create.side_effect = [
                    MagicMock(id="agent-1"),
                    MagicMock(id="agent-2")
                ]
                
                adapter1 = LettaAdapter(matter1)
                adapter2 = LettaAdapter(matter2)
                
                # Initialize agents
                await adapter1.initialize()
                await adapter2.initialize()
                
                # Verify different agents were created
                assert adapter1.agent_id == "agent-1"
                assert adapter2.agent_id == "agent-2"
                assert adapter1.agent_id != adapter2.agent_id
                
                # Verify agent configs are stored separately
                config1_path = matter1_path / "knowledge" / "agent_config.json"
                config2_path = matter2_path / "knowledge" / "agent_config.json"
                
                # Simulate saving configs
                adapter1._save_agent_config()
                adapter2._save_agent_config()
                
                assert config1_path.exists()
                assert config2_path.exists()
    
    @pytest.mark.asyncio
    async def test_concurrent_matter_operations(self):
        """Test concurrent operations on multiple matters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matters = []
            for i in range(3):
                matter_path = Path(temp_dir) / f"matter{i}"
                matter = Matter(
                    id=f"matter-{i}",
                    name=f"Matter {i}",
                    slug=f"matter-{i}",
                    embedding_model="nomic-embed-text",
                    generation_model="gpt-oss:20b",
                    paths=MatterPaths.from_root(matter_path)
                )
                matters.append(matter)
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock operations
                mock_client.agents.create.return_value = MagicMock(id="test-agent")
                mock_client.agents.search_archival_memory.return_value = []
                
                adapters = [LettaAdapter(matter) for matter in matters]
                
                # Initialize all adapters concurrently
                init_tasks = [adapter.initialize() for adapter in adapters]
                await asyncio.gather(*init_tasks)
                
                # Perform concurrent recalls
                async def recall_operation(adapter, query):
                    return await adapter.recall(query)
                
                recall_tasks = [
                    recall_operation(adapter, f"Query for matter {i}")
                    for i, adapter in enumerate(adapters)
                ]
                
                results = await asyncio.gather(*recall_tasks)
                
                # All operations should complete
                assert len(results) == 3
                assert all(isinstance(r, list) for r in results)
    
    @pytest.mark.asyncio
    async def test_matter_deletion_cleanup(self):
        """Test agent cleanup on matter deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "test_matter"
            matter = Matter(
                id="test-matter",
                name="Test Matter",
                slug="test-matter",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = MagicMock(id="test-agent")
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                # Save agent config
                adapter._save_agent_config()
                config_path = matter_path / "knowledge" / "agent_config.json"
                assert config_path.exists()
                
                # Delete agent
                await adapter.delete_agent()
                
                # Verify deletion was called
                mock_client.agents.delete.assert_called_once_with("test-agent")
                
                # Verify local data was cleaned up
                assert not config_path.exists()


class TestDataPersistence:
    """Test data persistence across restarts."""
    
    @pytest.mark.asyncio
    async def test_agent_persistence_across_restarts(self):
        """Test agent state persists across adapter restarts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "persistent_matter"
            matter = Matter(
                id="persist-matter",
                name="Persistent Matter",
                slug="persist-matter",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # First session - create agent
                mock_client.agents.create.return_value = MagicMock(id="persist-agent")
                mock_client.agents.retrieve.return_value = MagicMock(
                    id="persist-agent",
                    name="test-agent"
                )
                
                adapter1 = LettaAdapter(matter)
                await adapter1.initialize()
                assert adapter1.agent_id == "persist-agent"
                
                # Store some data
                await adapter1.upsert([{
                    "type": "Fact",
                    "label": "Test fact",
                    "date": "2024-01-01"
                }])
                
                # Second session - load existing agent
                adapter2 = LettaAdapter(matter)
                await adapter2.initialize()
                
                # Should load existing agent, not create new
                assert adapter2.agent_id == "persist-agent"
                
                # Verify only one create call (from first session)
                mock_client.agents.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self):
        """Test memory data persists across sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "memory_matter"
            matter = Matter(
                id="memory-matter",
                name="Memory Matter",
                slug="memory-matter",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Setup mock responses
                mock_client.agents.create.return_value = MagicMock(id="memory-agent")
                stored_memories = []
                
                # Mock memory storage
                async def mock_insert(agent_id, memory):
                    stored_memories.append(memory)
                    return MagicMock(id=f"mem-{len(stored_memories)}")
                
                mock_client.agents.insert_archival_memory = mock_insert
                
                # Mock memory retrieval
                async def mock_search(agent_id, query, limit=10):
                    return stored_memories[:limit]
                
                mock_client.agents.search_archival_memory = mock_search
                
                # Session 1: Store memories
                adapter1 = LettaAdapter(matter)
                await adapter1.initialize()
                
                memories = [
                    {"content": "Fact 1"},
                    {"content": "Fact 2"},
                    {"content": "Fact 3"}
                ]
                
                for memory in memories:
                    await adapter1.upsert([{
                        "type": "Fact",
                        "label": memory["content"]
                    }])
                
                # Session 2: Retrieve memories
                adapter2 = LettaAdapter(matter)
                await adapter2.initialize()
                
                recalled = await adapter2.recall("Fact")
                assert len(recalled) == 3
    
    @pytest.mark.asyncio
    async def test_server_recovery_after_crash(self):
        """Test server recovery after unexpected shutdown."""
        with patch('subprocess.Popen') as mock_popen:
            # Simulate process crash
            mock_process = MagicMock()
            mock_process.poll.side_effect = [None, None, 1]  # Process died
            mock_process.returncode = 1
            mock_popen.return_value = mock_process
            
            with patch('requests.get') as mock_get:
                # Health check fails after crash
                mock_get.side_effect = [
                    MagicMock(status_code=200),  # Initial success
                    Exception("Connection refused"),  # Server crashed
                    MagicMock(status_code=200)  # Recovery success
                ]
                
                manager = LettaServerManager()
                manager.configure(port=8283, mode="subprocess")
                
                # Start server
                await manager.start()
                assert mock_popen.call_count == 1
                
                # Start health monitoring
                manager.start_health_monitoring(interval=0.1, auto_restart=True)
                
                # Wait for crash detection and recovery
                await asyncio.sleep(0.3)
                
                # Should have restarted
                assert mock_popen.call_count >= 2
                
                manager.stop_health_monitoring()


class TestConcurrentOperations:
    """Test handling of concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self):
        """Test concurrent memory insertions and recalls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "concurrent_matter"
            matter = Matter(
                id="concurrent-matter",
                name="Concurrent Matter",
                slug="concurrent-matter",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = MagicMock(id="concurrent-agent")
                
                # Track operation order
                operation_log = []
                
                async def mock_insert(agent_id, memory):
                    operation_log.append(("insert", memory))
                    await asyncio.sleep(0.01)  # Simulate work
                    return MagicMock(id=f"mem-{len(operation_log)}")
                
                async def mock_search(agent_id, query, limit=10):
                    operation_log.append(("search", query))
                    await asyncio.sleep(0.01)  # Simulate work
                    return []
                
                mock_client.agents.insert_archival_memory = mock_insert
                mock_client.agents.search_archival_memory = mock_search
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                # Launch concurrent operations
                tasks = []
                
                # Mix insertions and recalls
                for i in range(10):
                    if i % 2 == 0:
                        task = adapter.upsert([{
                            "type": "Fact",
                            "label": f"Fact {i}"
                        }])
                    else:
                        task = adapter.recall(f"Query {i}")
                    tasks.append(task)
                
                # Execute all concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # All should complete
                assert len(results) == 10
                assert len(operation_log) == 10
                
                # Verify mix of operations
                insert_count = sum(1 for op, _ in operation_log if op == "insert")
                search_count = sum(1 for op, _ in operation_log if op == "search")
                assert insert_count == 5
                assert search_count == 5
    
    @pytest.mark.asyncio
    async def test_connection_pool_under_load(self):
        """Test connection pool handles high load."""
        with patch('app.letta_connection.AsyncLetta') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.health.check.return_value = None
            
            manager = LettaConnectionManager()
            manager.configure(base_url="http://localhost:8283")
            await manager.connect()
            
            # Track concurrent connections
            active_connections = []
            max_concurrent = 0
            
            async def mock_operation(id):
                active_connections.append(id)
                nonlocal max_concurrent
                max_concurrent = max(max_concurrent, len(active_connections))
                await asyncio.sleep(0.01)  # Simulate work
                active_connections.remove(id)
                return f"Result {id}"
            
            # Launch many concurrent operations
            tasks = []
            for i in range(50):
                task = manager.execute_with_retry(
                    mock_operation,
                    f"op_{i}",
                    i
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All should complete
            assert len(results) == 50
            assert all(f"Result {i}" in results for i in range(50))
            
            # Should have handled concurrent load
            assert max_concurrent > 1  # Some concurrency occurred


if __name__ == "__main__":
    pytest.main([__file__, "-v"])