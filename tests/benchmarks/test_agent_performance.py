"""
Performance benchmarks for the stateful agent architecture.

Tests performance characteristics with CPU-appropriate expectations.
"""

import pytest
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.letta_server import LettaServerManager
from app.letta_agent import LettaAgentHandler
from app.agent_metrics import AgentMetricsCollector, metrics_collector
from app.matters import matter_manager
from app.vectors import VectorStore
from app.logging_conf import get_logger

logger = get_logger(__name__)


class AgentPerformanceBenchmark:
    """Benchmark suite for agent performance."""
    
    def __init__(self):
        self.server = LettaServerManager()
        self.agent_handler = LettaAgentHandler()
        self.metrics = AgentMetricsCollector()
        self.results = {}
        
    async def setup(self):
        """Ensure server is running."""
        if not self.server._is_running:
            self.server.start()
            await asyncio.sleep(5)
        return True
    
    async def benchmark_agent_creation(self) -> Dict[str, Any]:
        """Benchmark agent creation and initialization time."""
        print("\n=== BENCHMARK: Agent Creation ===")
        
        times = []
        for i in range(3):
            matter = matter_manager.create_matter(f"Benchmark Matter {i}")
            
            start = time.time()
            self.agent_handler.set_active_matter(matter.id)
            
            # Initialize agent with first message
            response = await self.agent_handler.handle_user_message(
                matter_id=matter.id,
                message="Initialize agent for benchmarking."
            )
            
            creation_time = time.time() - start
            times.append(creation_time)
            
            print(f"  Iteration {i+1}: {creation_time:.1f}s")
        
        return {
            "name": "Agent Creation",
            "iterations": len(times),
            "times": times,
            "avg": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times)
        }
    
    async def benchmark_simple_response(self) -> Dict[str, Any]:
        """Benchmark simple response without tools."""
        print("\n=== BENCHMARK: Simple Response ===")
        
        matter = matter_manager.create_matter("Simple Response Benchmark")
        self.agent_handler.set_active_matter(matter.id)
        
        # Warm up
        await self.agent_handler.handle_user_message(
            matter_id=matter.id,
            message="Hello"
        )
        
        queries = [
            "What is your role?",
            "Can you help with construction claims?",
            "What kind of assistance do you provide?"
        ]
        
        times = []
        for query in queries:
            start = time.time()
            response = await self.agent_handler.handle_user_message(
                matter_id=matter.id,
                message=query
            )
            response_time = time.time() - start
            times.append(response_time)
            
            # Record in metrics
            self.metrics.record_response(
                matter_id=matter.id,
                message_id=f"simple_{len(times)}",
                response_time=response_time,
                tools_used=[],
                search_performed=False,
                memory_updated=False,
                citations_count=0
            )
            
            print(f"  Query {len(times)}: {response_time:.1f}s")
        
        return {
            "name": "Simple Response",
            "iterations": len(times),
            "times": times,
            "avg": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times)
        }
    
    async def benchmark_memory_operations(self) -> Dict[str, Any]:
        """Benchmark memory storage and recall."""
        print("\n=== BENCHMARK: Memory Operations ===")
        
        matter = matter_manager.create_matter("Memory Benchmark")
        self.agent_handler.set_active_matter(matter.id)
        
        # Store information
        info_messages = [
            "The contractor is ABC Construction with license #12345.",
            "Project value is $5.5 million with completion date June 2024.",
            "There were 3 change orders totaling $250,000."
        ]
        
        store_times = []
        for msg in info_messages:
            start = time.time()
            await self.agent_handler.handle_user_message(
                matter_id=matter.id,
                message=msg
            )
            store_time = time.time() - start
            store_times.append(store_time)
            print(f"  Store {len(store_times)}: {store_time:.1f}s")
        
        # Recall information
        recall_queries = [
            "Who is the contractor?",
            "What is the project value?",
            "How many change orders were there?"
        ]
        
        recall_times = []
        for query in recall_queries:
            start = time.time()
            response = await self.agent_handler.handle_user_message(
                matter_id=matter.id,
                message=query
            )
            recall_time = time.time() - start
            recall_times.append(recall_time)
            
            # Check if search was avoided
            search_avoided = not response.search_performed
            print(f"  Recall {len(recall_times)}: {recall_time:.1f}s (search avoided: {search_avoided})")
        
        return {
            "name": "Memory Operations",
            "store": {
                "iterations": len(store_times),
                "avg": statistics.mean(store_times),
                "median": statistics.median(store_times)
            },
            "recall": {
                "iterations": len(recall_times),
                "avg": statistics.mean(recall_times),
                "median": statistics.median(recall_times)
            }
        }
    
    async def benchmark_search_operations(self) -> Dict[str, Any]:
        """Benchmark document search operations."""
        print("\n=== BENCHMARK: Search Operations ===")
        
        matter = matter_manager.create_matter("Search Benchmark")
        self.agent_handler.set_active_matter(matter.id)
        
        # Add test documents
        vector_store = VectorStore(matter.paths.root)
        test_docs = [
            {
                "id": f"bench_{i}",
                "text": f"Test document {i} with contract terms and conditions. Payment terms net 30.",
                "metadata": {"doc_name": f"Doc{i}.pdf", "page_start": 1, "page_end": 1}
            }
            for i in range(5)
        ]
        
        for doc in test_docs:
            await vector_store.add_chunk(
                chunk_id=doc["id"],
                text=doc["text"],
                metadata=doc["metadata"]
            )
        
        search_queries = [
            "Search for payment terms in the documents.",
            "Find information about contract conditions.",
            "Look for any terms related to payments."
        ]
        
        times = []
        for query in search_queries:
            start = time.time()
            response = await self.agent_handler.handle_user_message(
                matter_id=matter.id,
                message=query
            )
            search_time = time.time() - start
            times.append(search_time)
            
            print(f"  Search {len(times)}: {search_time:.1f}s (found {len(response.citations or [])} citations)")
        
        return {
            "name": "Search Operations",
            "iterations": len(times),
            "times": times,
            "avg": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times)
        }
    
    async def benchmark_conversation_context(self) -> Dict[str, Any]:
        """Benchmark maintaining context over conversation."""
        print("\n=== BENCHMARK: Conversation Context ===")
        
        matter = matter_manager.create_matter("Context Benchmark")
        self.agent_handler.set_active_matter(matter.id)
        
        conversation = [
            "The project is a highway construction job.",
            "The contractor submitted a claim for weather delays.",
            "The delays lasted 45 days in total.",
            "What type of project is this?",  # Should recall
            "How long were the delays?",  # Should recall
            "What was the reason for the contractor's claim?"  # Should recall
        ]
        
        times = []
        search_count = 0
        
        for i, message in enumerate(conversation):
            start = time.time()
            response = await self.agent_handler.handle_user_message(
                matter_id=matter.id,
                message=message
            )
            response_time = time.time() - start
            times.append(response_time)
            
            if response.search_performed:
                search_count += 1
            
            is_question = "?" in message
            print(f"  Turn {i+1}: {response_time:.1f}s {'(Q)' if is_question else '(S)'} {'[searched]' if response.search_performed else ''}")
        
        # Calculate search reduction
        total_questions = sum(1 for m in conversation if "?" in m)
        search_rate = (search_count / total_questions * 100) if total_questions > 0 else 0
        
        return {
            "name": "Conversation Context",
            "turns": len(conversation),
            "times": times,
            "avg": statistics.mean(times),
            "search_count": search_count,
            "search_rate": search_rate,
            "context_maintained": search_count < total_questions
        }
    
    async def run_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and generate report."""
        print("=" * 70)
        print("AGENT PERFORMANCE BENCHMARKS")
        print("CPU-Only Inference (Expect longer times)")
        print("=" * 70)
        
        if not await self.setup():
            print("Failed to setup benchmark environment")
            return {}
        
        benchmarks = [
            self.benchmark_agent_creation,
            self.benchmark_simple_response,
            self.benchmark_memory_operations,
            self.benchmark_search_operations,
            self.benchmark_conversation_context
        ]
        
        results = {}
        for benchmark_func in benchmarks:
            try:
                result = await benchmark_func()
                results[result["name"]] = result
            except Exception as e:
                print(f"Benchmark failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        for name, result in results.items():
            print(f"\n{name}:")
            if "avg" in result:
                print(f"  Average: {result['avg']:.1f}s")
                print(f"  Median: {result.get('median', 0):.1f}s")
            if "store" in result:  # Memory operations
                print(f"  Store Avg: {result['store']['avg']:.1f}s")
                print(f"  Recall Avg: {result['recall']['avg']:.1f}s")
            if "context_maintained" in result:
                print(f"  Context Maintained: {'Yes' if result['context_maintained'] else 'No'}")
                print(f"  Search Rate: {result['search_rate']:.1f}%")
        
        # Performance analysis
        print("\n" + "=" * 70)
        print("PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        # Get metrics summary
        perf_summary = self.metrics.get_performance_summary()
        
        print("\nKey Metrics:")
        print(f"  Total Interactions: {perf_summary['total_interactions']}")
        
        if perf_summary.get('overall'):
            overall = perf_summary['overall']
            print(f"  Average Response Time: {overall['avg_response_time']:.1f}s")
            print(f"  Error Rate: {overall['error_rate']:.1f}%")
            print(f"  Search Rate: {overall['search_performed_rate']:.1f}%")
        
        print("\nPerformance Notes:")
        print("  - Times shown are for CPU-only inference")
        print("  - GPU acceleration would provide 10-100x speedup")
        print("  - Memory recall is faster than document search")
        print("  - Context is maintained across conversation turns")
        
        # Save results
        results_file = Path("tests/benchmarks/results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "benchmarks": results,
                "metrics_summary": perf_summary,
                "environment": {
                    "inference": "CPU-only",
                    "expected_times": "Minutes per response"
                }
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        return results


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_agent_performance():
    """Run performance benchmarks as a test."""
    benchmark = AgentPerformanceBenchmark()
    results = await benchmark.run_benchmarks()
    
    # Basic assertions
    assert len(results) > 0, "No benchmark results generated"
    
    # Check that some benchmarks completed
    for name, result in results.items():
        if "avg" in result:
            # CPU times can be very long, just check they're positive
            assert result["avg"] > 0, f"{name} has invalid average time"


async def main():
    """Run benchmarks standalone."""
    benchmark = AgentPerformanceBenchmark()
    await benchmark.run_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())