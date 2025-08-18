#!/usr/bin/env python3
"""
Test Letta server connectivity and basic operations.

This script validates:
1. Server startup and health check
2. Basic client connection
3. Agent creation and retrieval
4. Server shutdown
"""

import asyncio
import subprocess
import time
import sys
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from letta_client import AsyncLetta, Letta
from letta_client.types import LlmConfig, EmbeddingConfig


class LettaServerTest:
    def __init__(self, port=8283):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.server_process = None
        
    def start_server(self):
        """Start Letta server as subprocess."""
        print(f"Starting Letta server on port {self.port}...")
        
        cmd = ["letta", "server", "--port", str(self.port)]
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait for server to be ready
        if self._wait_for_server():
            print("✓ Server started successfully")
            return True
        else:
            print("✗ Server failed to start")
            return False
    
    def _wait_for_server(self, timeout=30):
        """Wait for server to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(0.5)
        return False
    
    def stop_server(self):
        """Stop the server gracefully."""
        if self.server_process:
            print("Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("✓ Server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("✓ Server killed")
    
    async def test_health_check(self):
        """Test server health endpoint."""
        print("\n1. Testing health check...")
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print(f"✓ Health check passed: {response.json()}")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check error: {e}")
            return False
    
    async def test_client_connection(self):
        """Test client connection."""
        print("\n2. Testing client connection...")
        try:
            # Test async client
            async_client = AsyncLetta(base_url=self.base_url)
            health = await async_client.health.health_check()
            print(f"✓ Async client connected")
            
            # Test sync client
            sync_client = Letta(base_url=self.base_url)
            print(f"✓ Sync client connected")
            
            return True
        except Exception as e:
            print(f"✗ Client connection failed: {e}")
            return False
    
    async def test_agent_creation(self):
        """Test agent creation and retrieval."""
        print("\n3. Testing agent creation...")
        try:
            client = AsyncLetta(base_url=self.base_url)
            
            # Configure LLM (using a minimal model for testing)
            llm_config = LlmConfig(
                model="gpt-oss:20b",  # Or any available model
                model_endpoint_type="ollama",
                model_endpoint="http://localhost:11434",
                context_window=4096
            )
            
            # Create test agent
            agent = await client.agents.create_agent(
                name="test-server-agent",
                description="Test agent for server validation",
                system="You are a test assistant.",
                llm_config=llm_config
            )
            
            print(f"✓ Agent created: {agent.id}")
            
            # Retrieve agent
            retrieved = await client.agents.get_agent(agent.id)
            print(f"✓ Agent retrieved: {retrieved.name}")
            
            # List agents
            agents = await client.agents.list_agents()
            print(f"✓ Listed {len(agents)} agent(s)")
            
            # Delete test agent
            await client.agents.delete_agent(agent.id)
            print(f"✓ Agent deleted")
            
            return True
            
        except Exception as e:
            print(f"✗ Agent operation failed: {e}")
            return False
    
    async def test_memory_operations(self):
        """Test basic memory operations."""
        print("\n4. Testing memory operations...")
        try:
            client = AsyncLetta(base_url=self.base_url)
            
            # Create agent for memory testing
            llm_config = LlmConfig(
                model="gpt-oss:20b",
                model_endpoint_type="ollama",
                model_endpoint="http://localhost:11434"
            )
            
            agent = await client.agents.create_agent(
                name="memory-test-agent",
                system="Memory test assistant",
                llm_config=llm_config
            )
            
            # Insert archival memory
            memory_content = "Test memory: The project started on January 1, 2024"
            passage = await client.agents.insert_archival_memory(
                agent_id=agent.id,
                memory=memory_content
            )
            print(f"✓ Memory inserted: {passage.id}")
            
            # Search archival memory
            results = await client.agents.search_archival_memory(
                agent_id=agent.id,
                query="project start date",
                limit=5
            )
            print(f"✓ Memory search returned {len(results)} result(s)")
            
            # Get all archival memory
            memories = await client.agents.get_archival_memory(
                agent_id=agent.id,
                limit=10
            )
            print(f"✓ Retrieved {len(memories)} memory item(s)")
            
            # Cleanup
            await client.agents.delete_agent(agent.id)
            
            return True
            
        except Exception as e:
            print(f"✗ Memory operation failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all server tests."""
        print("=" * 60)
        print("LETTA SERVER CONNECTIVITY TEST")
        print("=" * 60)
        
        results = {
            "server_start": False,
            "health_check": False,
            "client_connection": False,
            "agent_creation": False,
            "memory_operations": False
        }
        
        # Start server
        results["server_start"] = self.start_server()
        if not results["server_start"]:
            print("\n✗ Cannot proceed without server")
            return results
        
        try:
            # Run tests
            results["health_check"] = await self.test_health_check()
            results["client_connection"] = await self.test_client_connection()
            results["agent_creation"] = await self.test_agent_creation()
            results["memory_operations"] = await self.test_memory_operations()
            
        finally:
            # Always stop server
            self.stop_server()
        
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
    # Check if server is already running
    try:
        response = requests.get("http://localhost:8283/health")
        if response.status_code == 200:
            print("Warning: Server already running on port 8283")
            print("Please stop it and run this test again for full validation")
            print("Or modify the port in the test")
    except:
        pass
    
    # Run tests
    tester = LettaServerTest(port=8283)
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())