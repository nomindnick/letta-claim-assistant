#!/usr/bin/env python3
"""
Test Letta memory operations in detail.

This script validates:
1. Archival memory storage and retrieval
2. Core memory updates
3. Memory search and relevance
4. Memory persistence across sessions
5. Memory size and performance
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from letta_client import AsyncLetta
from letta_client.types import LlmConfig


class LettaMemoryTest:
    def __init__(self, letta_url="http://localhost:8283"):
        self.letta_url = letta_url
        self.client = None
        self.test_agent_id = None
        
    async def setup(self):
        """Setup test environment."""
        print("Setting up test environment...")
        try:
            self.client = AsyncLetta(base_url=self.letta_url)
            await self.client.health.health_check()
            
            # Create test agent
            llm_config = LlmConfig(
                model="gpt-oss:20b",
                model_endpoint_type="ollama",
                model_endpoint="http://localhost:11434"
            )
            
            agent = await self.client.agents.create_agent(
                name="memory-test-agent",
                description="Agent for testing memory operations",
                system="You are a construction claims assistant with perfect memory.",
                llm_config=llm_config,
                memory_blocks=[
                    {"label": "human", "value": "Test user - construction attorney"},
                    {"label": "persona", "value": "Expert construction analyst with detailed memory"}
                ]
            )
            
            self.test_agent_id = agent.id
            print(f"✓ Test agent created: {agent.id}")
            return True
            
        except Exception as e:
            print(f"✗ Setup failed: {e}")
            return False
    
    async def test_archival_memory_storage(self):
        """Test storing various types of information in archival memory."""
        print("\n1. Testing archival memory storage...")
        
        test_data = [
            # Structured JSON data
            {
                "type": "entity",
                "name": "ABC Construction Company",
                "role": "General Contractor",
                "contact": "john@abc-construction.com",
                "license": "CA-123456"
            },
            # Event data
            {
                "type": "event",
                "date": "2024-02-14",
                "description": "Foundation failure discovered",
                "severity": "critical",
                "affected_areas": ["Building A", "Building B"]
            },
            # Plain text
            "The project experienced significant delays due to weather conditions in March 2024.",
            # Technical specification
            "Foundation design: 4000 PSF bearing capacity, 24-inch depth, #5 rebar at 12\" OC",
            # Cost data
            {
                "type": "cost",
                "item": "Foundation repair",
                "amount": 250000,
                "currency": "USD",
                "date": "2024-03-01"
            }
        ]
        
        stored_ids = []
        for i, data in enumerate(test_data, 1):
            try:
                # Convert to string if dict
                content = json.dumps(data) if isinstance(data, dict) else data
                
                passage = await self.client.agents.insert_archival_memory(
                    agent_id=self.test_agent_id,
                    memory=content
                )
                stored_ids.append(passage.id)
                print(f"  ✓ Stored item {i}: {passage.id}")
                
            except Exception as e:
                print(f"  ✗ Failed to store item {i}: {e}")
                return False
        
        print(f"✓ Successfully stored {len(stored_ids)} memory items")
        return True
    
    async def test_memory_retrieval(self):
        """Test retrieving memories with different methods."""
        print("\n2. Testing memory retrieval...")
        
        try:
            # Get all memories
            print("  Retrieving all memories...")
            all_memories = await self.client.agents.get_archival_memory(
                agent_id=self.test_agent_id,
                limit=100
            )
            print(f"  ✓ Retrieved {len(all_memories)} total memories")
            
            # Display first few memories
            for i, memory in enumerate(all_memories[:3], 1):
                content = memory.text[:80] if hasattr(memory, 'text') else str(memory)[:80]
                print(f"    Memory {i}: {content}...")
            
            return True
            
        except Exception as e:
            print(f"✗ Retrieval failed: {e}")
            return False
    
    async def test_semantic_search(self):
        """Test semantic search capabilities."""
        print("\n3. Testing semantic search...")
        
        # Store domain-specific memories
        construction_memories = [
            "The concrete pour for foundation was completed on January 15, 2024",
            "Steel reinforcement installation finished on January 10, 2024",
            "Soil compaction test showed 95% density on January 5, 2024",
            "Foundation excavation started December 20, 2023",
            "Structural engineer approved foundation design on December 1, 2023",
            "Weather delay: Heavy rain stopped work from January 12-14, 2024",
            "Change order #5: Additional waterproofing for foundation approved",
            "RFI #23: Clarification on foundation depth requirements",
            "Safety incident: Near miss during foundation formwork on January 8, 2024",
            "Quality issue: Concrete slump test failed on first batch January 15, 2024"
        ]
        
        print("  Storing construction-specific memories...")
        for memory in construction_memories:
            await self.client.agents.insert_archival_memory(
                agent_id=self.test_agent_id,
                memory=memory
            )
        
        # Test various search queries
        search_tests = [
            ("foundation problems and issues", 5),
            ("weather delays rain", 3),
            ("concrete quality testing", 3),
            ("safety incidents", 2),
            ("January 2024 activities", 5)
        ]
        
        for query, expected_max in search_tests:
            print(f"\n  Searching for: '{query}'")
            results = await self.client.agents.search_archival_memory(
                agent_id=self.test_agent_id,
                query=query,
                limit=expected_max
            )
            
            print(f"  ✓ Found {len(results)} result(s)")
            for i, result in enumerate(results[:2], 1):
                content = result.text[:60] if hasattr(result, 'text') else str(result)[:60]
                print(f"    {i}. {content}...")
        
        return True
    
    async def test_core_memory_updates(self):
        """Test updating core memory blocks."""
        print("\n4. Testing core memory updates...")
        
        try:
            # Get current agent state
            agent = await self.client.agents.get_agent(self.test_agent_id)
            
            print("  Current core memory:")
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'blocks'):
                for block in agent.memory.blocks:
                    if hasattr(block, 'label') and hasattr(block, 'value'):
                        print(f"    {block.label}: {block.value[:50]}...")
            
            # Update core memory with case context
            new_context = """Current case: Foundation Failure at 123 Main St.
Key parties: ABC Construction (GC), XYZ Engineering (Designer)
Status: Discovery phase, investigating causation
Focus areas: Soil conditions, design adequacy, construction methods"""
            
            print("\n  Updating persona with case context...")
            
            # This approach may vary based on the Letta API version
            # We'll try to update the memory blocks
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'blocks'):
                memory_blocks = []
                for block in agent.memory.blocks:
                    if hasattr(block, 'label'):
                        if block.label == "persona":
                            memory_blocks.append({
                                "label": "persona",
                                "value": new_context
                            })
                        else:
                            memory_blocks.append({
                                "label": block.label,
                                "value": block.value if hasattr(block, 'value') else ""
                            })
                
                # Update agent with new memory blocks
                updated_agent = await self.client.agents.update_agent(
                    agent_id=self.test_agent_id,
                    memory_blocks=memory_blocks
                )
                
                print("  ✓ Core memory updated successfully")
            else:
                print("  ⚠ Core memory structure not as expected")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Core memory update failed: {e}")
            return False
    
    async def test_memory_persistence(self):
        """Test that memories persist across client sessions."""
        print("\n5. Testing memory persistence...")
        
        try:
            # Store a unique memory
            timestamp = datetime.now().isoformat()
            unique_memory = f"Persistence test conducted at {timestamp}"
            
            passage = await self.client.agents.insert_archival_memory(
                agent_id=self.test_agent_id,
                memory=unique_memory
            )
            
            print(f"  ✓ Stored memory with ID: {passage.id}")
            
            # Simulate new session by creating new client
            print("  Creating new client session...")
            new_client = AsyncLetta(base_url=self.letta_url)
            
            # Search for the unique memory
            results = await new_client.agents.search_archival_memory(
                agent_id=self.test_agent_id,
                query=f"persistence test {timestamp}",
                limit=1
            )
            
            if results and len(results) > 0:
                print(f"  ✓ Memory persisted across sessions")
                return True
            else:
                print(f"  ✗ Memory not found in new session")
                return False
                
        except Exception as e:
            print(f"  ✗ Persistence test failed: {e}")
            return False
    
    async def test_memory_performance(self):
        """Test memory performance with larger datasets."""
        print("\n6. Testing memory performance...")
        
        try:
            # Measure insertion performance
            print("  Testing bulk insertion...")
            start_time = time.time()
            
            bulk_memories = []
            for i in range(50):
                bulk_memories.append(f"Document {i}: Construction log entry for day {i}")
            
            # Insert memories concurrently
            tasks = []
            for memory in bulk_memories:
                task = self.client.agents.insert_archival_memory(
                    agent_id=self.test_agent_id,
                    memory=memory
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            insert_time = time.time() - start_time
            print(f"  ✓ Inserted {len(results)} memories in {insert_time:.2f} seconds")
            print(f"    Average: {insert_time/len(results)*1000:.1f} ms per memory")
            
            # Measure search performance
            print("\n  Testing search performance...")
            search_queries = [
                "construction log day 25",
                "document entry",
                "day 10 to day 20",
                "log entry activities"
            ]
            
            search_times = []
            for query in search_queries:
                start_time = time.time()
                results = await self.client.agents.search_archival_memory(
                    agent_id=self.test_agent_id,
                    query=query,
                    limit=10
                )
                search_time = time.time() - start_time
                search_times.append(search_time)
                print(f"    Query '{query[:30]}...': {search_time*1000:.1f} ms ({len(results)} results)")
            
            avg_search_time = sum(search_times) / len(search_times)
            print(f"  ✓ Average search time: {avg_search_time*1000:.1f} ms")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Performance test failed: {e}")
            return False
    
    async def test_conversation_memory(self):
        """Test memory through conversation interactions."""
        print("\n7. Testing conversation-based memory...")
        
        try:
            # Have a conversation that builds context
            conversation = [
                "I'm working on the foundation failure case at 123 Main Street.",
                "The failure occurred on February 14, 2024. ABC Construction was the GC.",
                "What do you remember about this case?",
                "The soil report showed clay content of 40%. Is this significant?",
                "Can you summarize what we've discussed about this foundation failure?"
            ]
            
            print("  Having conversation with agent...")
            for i, message in enumerate(conversation, 1):
                print(f"\n  User: {message}")
                
                response = await self.client.messages.send_message(
                    agent_id=self.test_agent_id,
                    role="user",
                    message=message
                )
                
                if response and response.messages:
                    # Get assistant's response
                    for msg in response.messages:
                        if hasattr(msg, 'role') and msg.role == 'assistant':
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            print(f"  Assistant: {content[:200]}...")
                            break
            
            # Check if conversation was stored in memory
            print("\n  Checking if conversation is in memory...")
            memories = await self.client.agents.search_archival_memory(
                agent_id=self.test_agent_id,
                query="123 Main Street foundation failure February",
                limit=5
            )
            
            if memories:
                print(f"  ✓ Conversation context stored ({len(memories)} relevant memories)")
            else:
                print(f"  ⚠ Conversation may not be automatically stored in archival memory")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Conversation test failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test resources."""
        print("\nCleaning up...")
        if self.test_agent_id and self.client:
            try:
                await self.client.agents.delete_agent(self.test_agent_id)
                print("✓ Test agent deleted")
            except Exception as e:
                print(f"✗ Cleanup failed: {e}")
    
    async def run_all_tests(self):
        """Run all memory tests."""
        print("=" * 60)
        print("LETTA MEMORY OPERATIONS TEST")
        print("=" * 60)
        
        results = {
            "setup": False,
            "archival_storage": False,
            "memory_retrieval": False,
            "semantic_search": False,
            "core_memory": False,
            "persistence": False,
            "performance": False,
            "conversation": False
        }
        
        # Setup
        results["setup"] = await self.setup()
        if not results["setup"]:
            print("\n✗ Cannot proceed without setup")
            return results
        
        try:
            # Run tests
            results["archival_storage"] = await self.test_archival_memory_storage()
            results["memory_retrieval"] = await self.test_memory_retrieval()
            results["semantic_search"] = await self.test_semantic_search()
            results["core_memory"] = await self.test_core_memory_updates()
            results["persistence"] = await self.test_memory_persistence()
            results["performance"] = await self.test_memory_performance()
            results["conversation"] = await self.test_conversation_memory()
            
        finally:
            # Always cleanup
            await self.cleanup()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        for test, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test:20} {status}")
        
        all_passed = all(results.values())
        print("\n" + ("✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"))
        
        return results


async def main():
    """Main test execution."""
    tester = LettaMemoryTest()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())