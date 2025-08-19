"""
Performance benchmarks for Letta operations including memory operations,
concurrent agent handling, resource usage, and response times under load.
"""

import pytest
import asyncio
import time
import psutil
import gc
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import tempfile
from pathlib import Path
import random
import string

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.letta_adapter import LettaAdapter
from app.letta_connection import LettaConnectionManager
from app.letta_request_queue import RequestQueue, RequestType, RequestPriority
from app.models import Matter, MatterPaths


class PerformanceMonitor:
    """Monitor performance metrics during tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        self.operations = []
        
    def start(self):
        """Start monitoring."""
        gc.collect()  # Force garbage collection
        self.start_time = time.time()
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def record_operation(self, name: str, duration: float):
        """Record an operation."""
        self.operations.append({
            "name": name,
            "duration": duration,
            "timestamp": time.time()
        })
        
        # Update peak memory
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def stop(self):
        """Stop monitoring."""
        self.end_time = time.time()
        process = psutil.Process()
        self.end_memory = process.memory_info().rss / 1024 / 1024
        gc.collect()
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report."""
        total_duration = self.end_time - self.start_time if self.end_time else 0
        memory_growth = self.end_memory - self.start_memory if self.end_memory else 0
        
        op_stats = {}
        for op in self.operations:
            name = op["name"]
            if name not in op_stats:
                op_stats[name] = {
                    "count": 0,
                    "total_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0
                }
            
            op_stats[name]["count"] += 1
            op_stats[name]["total_time"] += op["duration"]
            op_stats[name]["min_time"] = min(op_stats[name]["min_time"], op["duration"])
            op_stats[name]["max_time"] = max(op_stats[name]["max_time"], op["duration"])
        
        # Calculate averages
        for name, stats in op_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        
        return {
            "total_duration": total_duration,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": self.end_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_growth_mb": memory_growth,
            "operations": op_stats,
            "total_operations": len(self.operations)
        }


class TestMemoryOperationPerformance:
    """Benchmark memory operation performance."""
    
    @pytest.mark.asyncio
    async def test_memory_insert_latency(self):
        """Test memory insertion latency."""
        monitor = PerformanceMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "perf_matter"
            matter = Matter(
                id="perf-matter",
                name="Performance Matter",
                slug="perf-matter",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="perf-agent")
                
                # Mock fast memory operations
                async def mock_insert(agent_id, memory):
                    await asyncio.sleep(0.001)  # Simulate fast operation
                    return Mock(id=f"mem-{random.randint(1000, 9999)}")
                
                mock_client.agents.insert_archival_memory = mock_insert
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                monitor.start()
                
                # Insert 100 memory items
                for i in range(100):
                    start = time.time()
                    await adapter.upsert([{
                        "type": "Fact",
                        "label": f"Test fact {i}",
                        "date": "2024-01-01"
                    }])
                    duration = time.time() - start
                    monitor.record_operation("memory_insert", duration)
                
                monitor.stop()
                report = monitor.get_report()
                
                # Performance assertions
                assert report["operations"]["memory_insert"]["count"] == 100
                assert report["operations"]["memory_insert"]["avg_time"] < 0.5  # < 500ms average
                assert report["total_duration"] < 60  # Complete in < 60 seconds
                
                print(f"Memory Insert Performance: {report}")
    
    @pytest.mark.asyncio
    async def test_memory_recall_latency(self):
        """Test memory recall latency."""
        monitor = PerformanceMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "recall_matter"
            matter = Matter(
                id="recall-matter",
                name="Recall Matter",
                slug="recall-matter",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="recall-agent")
                
                # Mock memory search with realistic response
                async def mock_search(agent_id, query, limit=10):
                    await asyncio.sleep(0.01)  # Simulate search time
                    return [
                        {"content": f"Result {i}", "score": 0.9 - i*0.01}
                        for i in range(min(limit, 5))
                    ]
                
                mock_client.agents.search_archival_memory = mock_search
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                monitor.start()
                
                # Perform 50 recall operations
                queries = [
                    "construction contract",
                    "delay claim",
                    "change order",
                    "payment dispute",
                    "project timeline"
                ] * 10
                
                for query in queries:
                    start = time.time()
                    results = await adapter.recall(query)
                    duration = time.time() - start
                    monitor.record_operation("memory_recall", duration)
                
                monitor.stop()
                report = monitor.get_report()
                
                # Performance assertions
                assert report["operations"]["memory_recall"]["count"] == 50
                assert report["operations"]["memory_recall"]["avg_time"] < 0.5  # < 500ms average
                assert report["operations"]["memory_recall"]["max_time"] < 1.0  # < 1s max
                
                print(f"Memory Recall Performance: {report}")
    
    @pytest.mark.asyncio
    async def test_batch_memory_operations(self):
        """Test batch memory operation performance."""
        monitor = PerformanceMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "batch_matter"
            matter = Matter(
                id="batch-matter",
                name="Batch Matter",
                slug="batch-matter",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="batch-agent")
                
                # Mock batch operations
                async def mock_batch_insert(agent_id, memories):
                    await asyncio.sleep(0.05)  # Batch takes longer but more efficient
                    return [Mock(id=f"mem-{i}") for i in range(len(memories))]
                
                mock_client.agents.insert_archival_memory = mock_batch_insert
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                monitor.start()
                
                # Batch insert 1000 items in batches of 100
                batch_size = 100
                total_items = 1000
                
                for batch_num in range(total_items // batch_size):
                    batch = [
                        {
                            "type": "Fact",
                            "label": f"Batch {batch_num} item {i}",
                            "date": "2024-01-01"
                        }
                        for i in range(batch_size)
                    ]
                    
                    start = time.time()
                    await adapter.store_knowledge_batch(batch)
                    duration = time.time() - start
                    monitor.record_operation("batch_insert", duration)
                
                monitor.stop()
                report = monitor.get_report()
                
                # Performance assertions
                assert report["operations"]["batch_insert"]["count"] == 10
                assert report["total_duration"] < 5  # Complete 1000 items in < 5 seconds
                
                # Calculate throughput
                throughput = total_items / report["total_duration"]
                assert throughput > 100  # > 100 items/second
                
                print(f"Batch Performance: {report}")
                print(f"Throughput: {throughput:.2f} items/second")


class TestConcurrentAgentPerformance:
    """Benchmark concurrent agent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_creation(self):
        """Test concurrent agent creation performance."""
        monitor = PerformanceMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock agent creation
                async def mock_create(**kwargs):
                    await asyncio.sleep(0.1)  # Simulate creation time
                    return Mock(id=f"agent-{random.randint(1000, 9999)}")
                
                mock_client.agents.create = mock_create
                
                monitor.start()
                
                # Create multiple agents concurrently
                tasks = []
                for i in range(10):
                    matter_path = Path(temp_dir) / f"matter_{i}"
                    matter = Matter(
                        id=f"matter-{i}",
                        name=f"Matter {i}",
                        slug=f"matter-{i}",
                        embedding_model="nomic-embed-text",
                        generation_model="gpt-oss:20b",
                        paths=MatterPaths.from_root(matter_path)
                    )
                    
                    adapter = LettaAdapter(matter)
                    task = adapter.initialize()
                    tasks.append(task)
                
                # Time concurrent creation
                start = time.time()
                await asyncio.gather(*tasks)
                duration = time.time() - start
                
                monitor.stop()
                report = monitor.get_report()
                
                # Should complete much faster than sequential (10 * 0.1 = 1 second)
                assert duration < 0.5  # Concurrent should be < 0.5 seconds
                assert report["memory_growth_mb"] < 100  # Memory growth < 100MB
                
                print(f"Concurrent Agent Creation: {duration:.2f}s for 10 agents")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_throughput(self):
        """Test throughput of concurrent operations."""
        monitor = PerformanceMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple matters with agents
            matters = []
            adapters = []
            
            for i in range(5):
                matter_path = Path(temp_dir) / f"concurrent_{i}"
                matter = Matter(
                    id=f"concurrent-{i}",
                    name=f"Concurrent {i}",
                    slug=f"concurrent-{i}",
                    embedding_model="nomic-embed-text",
                    generation_model="gpt-oss:20b",
                    paths=MatterPaths.from_root(matter_path)
                )
                matters.append(matter)
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="test-agent")
                
                # Mock operations
                async def mock_search(agent_id, query, limit=10):
                    await asyncio.sleep(0.01)
                    return []
                
                async def mock_insert(agent_id, memory):
                    await asyncio.sleep(0.005)
                    return Mock(id="mem-id")
                
                mock_client.agents.search_archival_memory = mock_search
                mock_client.agents.insert_archival_memory = mock_insert
                
                # Initialize adapters
                for matter in matters:
                    adapter = LettaAdapter(matter)
                    await adapter.initialize()
                    adapters.append(adapter)
                
                monitor.start()
                
                # Launch mixed concurrent operations
                operations_per_adapter = 20
                tasks = []
                
                for adapter in adapters:
                    for i in range(operations_per_adapter):
                        if i % 2 == 0:
                            # Recall operation
                            task = adapter.recall(f"Query {i}")
                        else:
                            # Insert operation
                            task = adapter.upsert([{
                                "type": "Fact",
                                "label": f"Fact {i}"
                            }])
                        tasks.append(task)
                
                start = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start
                
                monitor.stop()
                
                # Calculate throughput
                total_ops = len(tasks)
                throughput = total_ops / duration
                
                # Performance assertions
                assert all(not isinstance(r, Exception) for r in results)
                assert throughput > 50  # > 50 ops/second
                
                print(f"Concurrent Operations: {total_ops} ops in {duration:.2f}s")
                print(f"Throughput: {throughput:.2f} ops/second")


class TestResourceUsage:
    """Test resource usage under load."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_growth(self):
        """Test memory usage growth with large datasets."""
        monitor = PerformanceMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "memory_test"
            matter = Matter(
                id="memory-test",
                name="Memory Test",
                slug="memory-test",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="memory-agent")
                
                # Store large memories
                stored_memories = []
                
                async def mock_insert(agent_id, memory):
                    # Simulate storing in memory
                    stored_memories.append(memory)
                    return Mock(id=f"mem-{len(stored_memories)}")
                
                mock_client.agents.insert_archival_memory = mock_insert
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                monitor.start()
                
                # Generate large text content
                def generate_large_text(size_kb: int) -> str:
                    chars = string.ascii_letters + string.digits + ' '
                    return ''.join(random.choices(chars, k=size_kb * 1024))
                
                # Insert progressively larger memories
                for i in range(10):
                    size_kb = (i + 1) * 10  # 10KB, 20KB, ..., 100KB
                    large_content = generate_large_text(size_kb)
                    
                    await adapter.upsert([{
                        "type": "Fact",
                        "label": f"Large fact {i}",
                        "support_snippet": large_content
                    }])
                    
                    # Check memory growth
                    process = psutil.Process()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    monitor.peak_memory = max(monitor.peak_memory, current_memory)
                
                monitor.stop()
                report = monitor.get_report()
                
                # Memory assertions
                assert report["memory_growth_mb"] < 500  # < 500MB growth
                assert report["peak_memory_mb"] < 1000  # Peak < 1GB
                
                print(f"Memory Usage Report: {report}")
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self):
        """Test memory cleanup and garbage collection."""
        monitor = PerformanceMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "cleanup_test"
            matter = Matter(
                id="cleanup-test",
                name="Cleanup Test",
                slug="cleanup-test",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=MatterPaths.from_root(matter_path)
            )
            
            with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.agents.create.return_value = Mock(id="cleanup-agent")
                
                adapter = LettaAdapter(matter)
                await adapter.initialize()
                
                monitor.start()
                initial_memory = monitor.start_memory
                
                # Create and discard large objects
                for cycle in range(5):
                    large_data = []
                    
                    # Allocate memory
                    for i in range(100):
                        large_data.append('x' * 10000)  # 10KB strings
                    
                    # Simulate processing
                    await asyncio.sleep(0.1)
                    
                    # Clear references
                    large_data = None
                    gc.collect()  # Force garbage collection
                    
                    # Check memory after cleanup
                    process = psutil.Process()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    
                    # Memory should return close to baseline after GC
                    memory_diff = current_memory - initial_memory
                    assert memory_diff < 50  # Less than 50MB above baseline
                
                monitor.stop()
                
                # Final memory should be close to initial
                final_diff = monitor.end_memory - monitor.start_memory
                assert final_diff < 20  # Less than 20MB growth


class TestQueuePerformance:
    """Test request queue performance."""
    
    @pytest.mark.asyncio
    async def test_queue_throughput(self):
        """Test request queue throughput."""
        monitor = PerformanceMonitor()
        queue = RequestQueue(batch_size=10, batch_timeout=0.1)
        
        await queue.start()
        monitor.start()
        
        try:
            # Fast operation for throughput testing
            async def fast_operation(value):
                await asyncio.sleep(0.001)
                return value * 2
            
            # Enqueue many operations
            futures = []
            for i in range(1000):
                future = await queue.enqueue(
                    fast_operation,
                    RequestType.OTHER,
                    priority=RequestPriority.NORMAL,
                    value=i
                )
                futures.append(future)
            
            # Wait for all to complete
            start = time.time()
            results = await asyncio.gather(*futures)
            duration = time.time() - start
            
            # Calculate throughput
            throughput = len(results) / duration
            
            # Performance assertions
            assert len(results) == 1000
            assert throughput > 100  # > 100 requests/second
            
            monitor.stop()
            report = monitor.get_report()
            
            print(f"Queue Throughput: {throughput:.2f} requests/second")
            print(f"Queue Performance: {report}")
            
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_priority_queue_performance(self):
        """Test priority queue performance under load."""
        queue = RequestQueue(batch_size=5)
        await queue.start()
        
        try:
            completion_order = []
            
            async def track_operation(id, priority):
                await asyncio.sleep(0.01)
                completion_order.append((id, priority))
                return id
            
            # Enqueue with mixed priorities
            futures = []
            
            # Add low priority first
            for i in range(10):
                future = await queue.enqueue(
                    track_operation,
                    RequestType.OTHER,
                    priority=RequestPriority.LOW,
                    id=f"low-{i}"
                )
                futures.append(future)
            
            # Add high priority
            for i in range(5):
                future = await queue.enqueue(
                    track_operation,
                    RequestType.OTHER,
                    priority=RequestPriority.HIGH,
                    id=f"high-{i}",
                    priority=RequestPriority.HIGH
                )
                futures.append(future)
            
            # Add critical
            for i in range(2):
                future = await queue.enqueue(
                    track_operation,
                    RequestType.OTHER,
                    priority=RequestPriority.CRITICAL,
                    id=f"critical-{i}",
                    priority=RequestPriority.CRITICAL
                )
                futures.append(future)
            
            # Wait for completion
            await asyncio.gather(*futures)
            
            # Check that critical completed first
            critical_indices = [
                i for i, (id, _) in enumerate(completion_order)
                if id.startswith("critical")
            ]
            
            # Critical should be in first few positions
            assert all(idx < 5 for idx in critical_indices)
            
            print(f"Priority Queue Order: {completion_order[:10]}")
            
        finally:
            await queue.stop()


if __name__ == "__main__":
    # Run with: python -m pytest tests/benchmarks/test_letta_performance.py -v -s
    pytest.main([__file__, "-v", "-s"])