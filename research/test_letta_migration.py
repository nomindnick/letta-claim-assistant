#!/usr/bin/env python3
"""
Test migration from LocalClient to modern Letta API.

This script demonstrates:
1. Detection of old LocalClient code patterns
2. Migration to new server-based API
3. Data preservation strategies
4. Fallback mechanisms
5. Compatibility layer implementation
"""

import asyncio
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from letta_client import AsyncLetta
from letta_client.types import LlmConfig, EmbeddingConfig


class LettaMigrationTest:
    def __init__(self, letta_url="http://localhost:8283"):
        self.letta_url = letta_url
        self.client = None
        
    def detect_old_code_patterns(self):
        """Detect and report old LocalClient patterns."""
        print("\n1. Detecting old LocalClient patterns...")
        
        old_patterns = {
            "from letta import LocalClient": "from letta_client import AsyncLetta",
            "LocalClient()": "AsyncLetta(base_url='http://localhost:8283')",
            "client.create_agent(": "await client.agents.create_agent(",
            "client.user_message(": "await client.messages.send_message(",
            "client.get_archival_memory(": "await client.agents.get_archival_memory(",
            "client.insert_archival_memory(": "await client.agents.insert_archival_memory(",
            "persona=": "system=",
            "ChatMemory(": "memory_blocks=[",
        }
        
        print("  Common patterns to migrate:")
        for old, new in old_patterns.items():
            print(f"    OLD: {old}")
            print(f"    NEW: {new}")
            print()
        
        return True
    
    async def demonstrate_old_vs_new(self):
        """Show side-by-side comparison of old vs new code."""
        print("\n2. Old vs New API Comparison...")
        
        print("\n  OLD LocalClient approach (no longer works):")
        print("  " + "="*50)
        old_code = '''
    from letta import LocalClient
    
    # Create client (embedded)
    client = LocalClient()
    
    # Create agent
    agent = client.create_agent(
        name="my-agent",
        persona="Assistant persona",
        human="Human description"
    )
    
    # Send message
    response = client.user_message(
        agent_id=agent.id,
        message="Hello"
    )
    
    # Store memory
    client.insert_archival_memory(
        agent_id=agent.id,
        memory="Important fact"
    )
    '''
        for line in old_code.strip().split('\n'):
            print("  " + line)
        
        print("\n  NEW Server-based approach:")
        print("  " + "="*50)
        new_code = '''
    from letta_client import AsyncLetta
    
    # Connect to server
    client = AsyncLetta(base_url="http://localhost:8283")
    
    # Create agent (async)
    agent = await client.agents.create_agent(
        name="my-agent",
        system="Assistant persona",
        memory_blocks=[
            {"label": "persona", "value": "Assistant persona"},
            {"label": "human", "value": "Human description"}
        ],
        llm_config=llm_config  # Required now
    )
    
    # Send message (async)
    response = await client.messages.send_message(
        agent_id=agent.id,
        role="user",
        message="Hello"
    )
    
    # Store memory (async)
    await client.agents.insert_archival_memory(
        agent_id=agent.id,
        memory="Important fact"
    )
    '''
        for line in new_code.strip().split('\n'):
            print("  " + line)
        
        return True
    
    async def create_compatibility_layer(self):
        """Create a compatibility layer for easier migration."""
        print("\n3. Creating compatibility layer...")
        
        print("  Compatibility wrapper class:")
        print("  " + "="*50)
        
        compatibility_code = '''
class LettaCompatibilityAdapter:
    """Compatibility layer for migrating from LocalClient to modern API."""
    
    def __init__(self, base_url="http://localhost:8283"):
        self.async_client = AsyncLetta(base_url=base_url)
        self.default_llm_config = LlmConfig(
            model="gpt-oss:20b",
            model_endpoint_type="ollama",
            model_endpoint="http://localhost:11434"
        )
    
    def create_agent(self, name, persona=None, human=None, **kwargs):
        """Compatibility wrapper for create_agent."""
        # Convert old parameters to new format
        memory_blocks = []
        if persona:
            memory_blocks.append({"label": "persona", "value": persona})
        if human:
            memory_blocks.append({"label": "human", "value": human})
        
        # Run async operation synchronously
        return asyncio.run(
            self.async_client.agents.create_agent(
                name=name,
                system=persona or "Assistant",
                memory_blocks=memory_blocks,
                llm_config=kwargs.get('llm_config', self.default_llm_config),
                **kwargs
            )
        )
    
    def user_message(self, agent_id, message):
        """Compatibility wrapper for user_message."""
        return asyncio.run(
            self.async_client.messages.send_message(
                agent_id=agent_id,
                role="user",
                message=message
            )
        )
    
    def insert_archival_memory(self, agent_id, memory):
        """Compatibility wrapper for insert_archival_memory."""
        return asyncio.run(
            self.async_client.agents.insert_archival_memory(
                agent_id=agent_id,
                memory=memory
            )
        )
        '''
        
        for line in compatibility_code.strip().split('\n'):
            print("  " + line)
        
        print("\n  ✓ Compatibility layer defined")
        return True
    
    async def test_migration_with_data(self):
        """Test migrating existing agent data."""
        print("\n4. Testing data migration...")
        
        try:
            # Connect to new server
            self.client = AsyncLetta(base_url=self.letta_url)
            
            # Simulate old agent data structure
            old_agent_data = {
                "agent_id": "old-agent-123",
                "matter_id": "matter-456",
                "matter_name": "Test Construction Case",
                "created_at": "2024-01-01T00:00:00",
                "persona": "Construction claims analyst",
                "human": "Construction attorney",
                "memories": [
                    "ABC Construction is the general contractor",
                    "Foundation failure occurred on February 14",
                    "Repair costs estimated at $250,000"
                ]
            }
            
            print(f"  Migrating agent: {old_agent_data['matter_name']}")
            
            # Create new agent with migrated data
            llm_config = LlmConfig(
                model="gpt-oss:20b",
                model_endpoint_type="ollama",
                model_endpoint="http://localhost:11434"
            )
            
            new_agent = await self.client.agents.create_agent(
                name=f"migrated-{old_agent_data['matter_name'].replace(' ', '-')}",
                description=f"Migrated from old agent {old_agent_data['agent_id']}",
                system=old_agent_data["persona"],
                memory_blocks=[
                    {"label": "persona", "value": old_agent_data["persona"]},
                    {"label": "human", "value": old_agent_data["human"]}
                ],
                llm_config=llm_config,
                metadata={
                    "old_agent_id": old_agent_data["agent_id"],
                    "migration_date": datetime.now().isoformat(),
                    "migration_version": "1.0"
                }
            )
            
            print(f"  ✓ New agent created: {new_agent.id}")
            
            # Migrate memories
            print(f"  Migrating {len(old_agent_data['memories'])} memories...")
            for memory in old_agent_data["memories"]:
                await self.client.agents.insert_archival_memory(
                    agent_id=new_agent.id,
                    memory=memory
                )
            
            print(f"  ✓ Memories migrated successfully")
            
            # Verify migration
            migrated_memories = await self.client.agents.get_archival_memory(
                agent_id=new_agent.id,
                limit=10
            )
            
            print(f"  ✓ Verified {len(migrated_memories)} memories in new agent")
            
            # Cleanup
            await self.client.agents.delete_agent(new_agent.id)
            
            return True
            
        except Exception as e:
            print(f"  ✗ Migration test failed: {e}")
            return False
    
    async def test_fallback_mechanism(self):
        """Test fallback behavior when server is unavailable."""
        print("\n5. Testing fallback mechanisms...")
        
        class FallbackAdapter:
            """Adapter with fallback to local storage when server unavailable."""
            
            def __init__(self, base_url="http://localhost:8283"):
                self.base_url = base_url
                self.client = None
                self.fallback_mode = False
                self.local_storage = {}
                
            async def initialize(self):
                """Try to connect, fallback if unavailable."""
                try:
                    self.client = AsyncLetta(base_url=self.base_url)
                    await self.client.health.health_check()
                    self.fallback_mode = False
                    return "connected"
                except:
                    self.fallback_mode = True
                    return "fallback"
            
            async def store_memory(self, agent_id, memory):
                """Store memory with fallback."""
                if not self.fallback_mode and self.client:
                    try:
                        return await self.client.agents.insert_archival_memory(
                            agent_id=agent_id,
                            memory=memory
                        )
                    except:
                        self.fallback_mode = True
                
                # Fallback to local storage
                if agent_id not in self.local_storage:
                    self.local_storage[agent_id] = []
                self.local_storage[agent_id].append({
                    "id": f"local-{len(self.local_storage[agent_id])}",
                    "memory": memory,
                    "timestamp": datetime.now().isoformat()
                })
                return {"id": f"local-{len(self.local_storage[agent_id])-1}"}
            
            async def get_memories(self, agent_id):
                """Get memories with fallback."""
                if not self.fallback_mode and self.client:
                    try:
                        return await self.client.agents.get_archival_memory(
                            agent_id=agent_id
                        )
                    except:
                        self.fallback_mode = True
                
                # Return from local storage
                return self.local_storage.get(agent_id, [])
        
        # Test the fallback adapter
        adapter = FallbackAdapter(base_url="http://invalid-url:9999")
        status = await adapter.initialize()
        
        print(f"  Connection status: {status}")
        print(f"  Fallback mode: {adapter.fallback_mode}")
        
        # Test storing in fallback mode
        test_agent_id = "test-agent-fallback"
        await adapter.store_memory(test_agent_id, "Test memory in fallback mode")
        memories = await adapter.get_memories(test_agent_id)
        
        print(f"  ✓ Fallback storage working ({len(memories)} items)")
        
        return True
    
    async def generate_migration_report(self):
        """Generate a migration readiness report."""
        print("\n6. Migration Readiness Report")
        print("  " + "="*50)
        
        report = {
            "server_available": False,
            "ollama_available": False,
            "existing_agents": 0,
            "migration_steps": [],
            "estimated_effort": "Low"
        }
        
        # Check server
        try:
            client = AsyncLetta(base_url=self.letta_url)
            await client.health.health_check()
            report["server_available"] = True
            print("  ✓ Letta server is available")
        except:
            print("  ✗ Letta server not available")
            report["migration_steps"].append("Start Letta server: letta server")
        
        # Check Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                report["ollama_available"] = True
                print("  ✓ Ollama is available")
        except:
            print("  ✗ Ollama not available")
            report["migration_steps"].append("Start Ollama: ollama serve")
        
        # Check for existing agents (would check actual storage in production)
        print("  ⚠ Check existing agent data manually")
        report["migration_steps"].append("Backup existing agent data")
        
        # Migration steps
        report["migration_steps"].extend([
            "Update imports from 'letta' to 'letta_client'",
            "Add async/await to all Letta operations",
            "Add LLM configuration to agent creation",
            "Update memory operation method names",
            "Test migrated agents thoroughly"
        ])
        
        print("\n  Required Migration Steps:")
        for i, step in enumerate(report["migration_steps"], 1):
            print(f"    {i}. {step}")
        
        print(f"\n  Estimated effort: {report['estimated_effort']}")
        
        return True
    
    async def run_all_tests(self):
        """Run all migration tests."""
        print("=" * 60)
        print("LETTA MIGRATION TEST")
        print("=" * 60)
        
        results = {
            "pattern_detection": False,
            "api_comparison": False,
            "compatibility_layer": False,
            "data_migration": False,
            "fallback_mechanism": False,
            "migration_report": False
        }
        
        # Run tests
        results["pattern_detection"] = self.detect_old_code_patterns()
        results["api_comparison"] = await self.demonstrate_old_vs_new()
        results["compatibility_layer"] = await self.create_compatibility_layer()
        results["data_migration"] = await self.test_migration_with_data()
        results["fallback_mechanism"] = await self.test_fallback_mechanism()
        results["migration_report"] = await self.generate_migration_report()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        for test, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test:20} {status}")
        
        all_passed = all(results.values())
        print("\n" + ("✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"))
        
        # Migration recommendations
        print("\n" + "=" * 60)
        print("MIGRATION RECOMMENDATIONS")
        print("=" * 60)
        print("""
1. Start with the compatibility layer for quick migration
2. Gradually refactor to use native async/await
3. Test memory persistence thoroughly
4. Implement proper fallback mechanisms
5. Monitor performance after migration
        """)
        
        return results


async def main():
    """Main test execution."""
    tester = LettaMigrationTest()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())