#!/usr/bin/env python3
"""
Test Letta integration with Ollama models.

This script validates:
1. Ollama connectivity
2. Available models detection
3. Agent creation with Ollama models
4. Generation and embedding tests
5. Model switching
"""

import asyncio
import sys
import requests
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from letta_client import AsyncLetta
from letta_client.types import LlmConfig, EmbeddingConfig


class LettaOllamaTest:
    def __init__(self, letta_url="http://localhost:8283", ollama_url="http://localhost:11434"):
        self.letta_url = letta_url
        self.ollama_url = ollama_url
        self.client = None
        self.available_models = []
        self.available_embed_models = []
        
    async def check_ollama_connectivity(self):
        """Check if Ollama is running and accessible."""
        print("\n1. Checking Ollama connectivity...")
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"✓ Ollama is running with {len(models)} model(s)")
                
                # Categorize models
                for model in models:
                    name = model["name"]
                    if "embed" in name.lower() or "nomic" in name.lower():
                        self.available_embed_models.append(name)
                    else:
                        self.available_models.append(name)
                
                print(f"  Generation models: {', '.join(self.available_models[:3])}")
                if len(self.available_models) > 3:
                    print(f"  ... and {len(self.available_models)-3} more")
                    
                print(f"  Embedding models: {', '.join(self.available_embed_models)}")
                return True
            else:
                print(f"✗ Ollama returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to Ollama: {e}")
            print("  Make sure Ollama is running: ollama serve")
            return False
    
    async def check_letta_connectivity(self):
        """Check if Letta server is accessible."""
        print("\n2. Checking Letta server connectivity...")
        try:
            self.client = AsyncLetta(base_url=self.letta_url)
            await self.client.health.health_check()
            print(f"✓ Connected to Letta server")
            return True
        except Exception as e:
            print(f"✗ Cannot connect to Letta: {e}")
            print("  Make sure Letta server is running: letta server")
            return False
    
    async def test_agent_with_ollama_model(self, model_name):
        """Test agent creation with specific Ollama model."""
        print(f"\n3. Testing agent with Ollama model: {model_name}")
        
        try:
            # Configure LLM with Ollama
            llm_config = LlmConfig(
                model=model_name,
                model_endpoint_type="ollama",
                model_endpoint=self.ollama_url,
                context_window=4096,
                temperature=0.7
            )
            
            # Configure embeddings (use first available or default)
            embed_model = self.available_embed_models[0] if self.available_embed_models else "nomic-embed-text"
            embedding_config = EmbeddingConfig(
                embedding_model=embed_model,
                embedding_endpoint_type="ollama",
                embedding_endpoint=self.ollama_url,
                embedding_dim=768  # Adjust based on model
            )
            
            # Create agent
            agent = await self.client.agents.create_agent(
                name=f"ollama-test-{model_name.replace(':', '-')}",
                description=f"Test agent using {model_name}",
                system="You are a helpful assistant for testing Ollama integration.",
                llm_config=llm_config,
                embedding_config=embedding_config
            )
            
            print(f"✓ Agent created with ID: {agent.id}")
            
            # Test message generation
            print(f"  Testing generation...")
            response = await self.client.messages.send_message(
                agent_id=agent.id,
                role="user",
                message="Hello! Please confirm you're working correctly by saying 'System operational'."
            )
            
            if response and response.messages:
                print(f"✓ Generation successful")
                # Print first message content (truncated)
                first_msg = str(response.messages[0])[:100]
                print(f"  Response preview: {first_msg}...")
            else:
                print(f"✗ No response generated")
            
            # Test memory storage
            print(f"  Testing memory storage...")
            memory_content = f"Test fact: This agent uses {model_name} model via Ollama"
            await self.client.agents.insert_archival_memory(
                agent_id=agent.id,
                memory=memory_content
            )
            print(f"✓ Memory stored successfully")
            
            # Test memory recall
            print(f"  Testing memory recall...")
            memories = await self.client.agents.search_archival_memory(
                agent_id=agent.id,
                query="model ollama",
                limit=5
            )
            print(f"✓ Memory recall returned {len(memories)} result(s)")
            
            # Cleanup
            await self.client.agents.delete_agent(agent.id)
            print(f"✓ Agent cleaned up")
            
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            return False
    
    async def test_model_switching(self):
        """Test switching between different Ollama models."""
        print("\n4. Testing model switching...")
        
        if len(self.available_models) < 2:
            print("⚠ Need at least 2 models for switching test")
            return False
        
        try:
            # Create agent with first model
            model1 = self.available_models[0]
            print(f"  Creating agent with {model1}...")
            
            llm_config1 = LlmConfig(
                model=model1,
                model_endpoint_type="ollama",
                model_endpoint=self.ollama_url
            )
            
            agent = await self.client.agents.create_agent(
                name="model-switch-test",
                system="Test assistant",
                llm_config=llm_config1
            )
            
            print(f"✓ Agent created with {model1}")
            
            # Update to second model
            model2 = self.available_models[1]
            print(f"  Switching to {model2}...")
            
            llm_config2 = LlmConfig(
                model=model2,
                model_endpoint_type="ollama",
                model_endpoint=self.ollama_url
            )
            
            updated_agent = await self.client.agents.update_agent(
                agent_id=agent.id,
                llm_config=llm_config2
            )
            
            print(f"✓ Agent updated to use {model2}")
            
            # Verify the switch worked by sending a message
            response = await self.client.messages.send_message(
                agent_id=agent.id,
                role="user",
                message="Test message after model switch"
            )
            
            if response and response.messages:
                print(f"✓ Model switch successful - agent still responsive")
            
            # Cleanup
            await self.client.agents.delete_agent(agent.id)
            
            return True
            
        except Exception as e:
            print(f"✗ Model switching failed: {e}")
            return False
    
    async def test_embedding_operations(self):
        """Test embedding operations with Ollama."""
        print("\n5. Testing embedding operations...")
        
        if not self.available_embed_models:
            print("⚠ No embedding models available")
            print("  Install one with: ollama pull nomic-embed-text")
            return False
        
        try:
            embed_model = self.available_embed_models[0]
            print(f"  Using embedding model: {embed_model}")
            
            # Create agent with embedding configuration
            embedding_config = EmbeddingConfig(
                embedding_model=embed_model,
                embedding_endpoint_type="ollama",
                embedding_endpoint=self.ollama_url,
                embedding_dim=768
            )
            
            llm_config = LlmConfig(
                model=self.available_models[0] if self.available_models else "gpt-oss:20b",
                model_endpoint_type="ollama",
                model_endpoint=self.ollama_url
            )
            
            agent = await self.client.agents.create_agent(
                name="embedding-test",
                system="Embedding test assistant",
                llm_config=llm_config,
                embedding_config=embedding_config
            )
            
            print(f"✓ Agent created with embeddings")
            
            # Store multiple memories for similarity testing
            test_memories = [
                "The foundation failure occurred on February 14, 2024",
                "ABC Construction was the general contractor",
                "The weather was rainy during the foundation pour",
                "Steel reinforcement was installed on February 10, 2024",
                "The concrete mix design was approved by the engineer"
            ]
            
            print(f"  Storing {len(test_memories)} test memories...")
            for memory in test_memories:
                await self.client.agents.insert_archival_memory(
                    agent_id=agent.id,
                    memory=memory
                )
            
            print(f"✓ Memories stored with embeddings")
            
            # Test semantic search
            print(f"  Testing semantic search...")
            
            # Search for foundation-related memories
            results = await self.client.agents.search_archival_memory(
                agent_id=agent.id,
                query="foundation problems and issues",
                limit=3
            )
            
            print(f"✓ Semantic search returned {len(results)} result(s)")
            for i, result in enumerate(results[:2], 1):
                print(f"    Result {i}: {str(result)[:80]}...")
            
            # Cleanup
            await self.client.agents.delete_agent(agent.id)
            
            return True
            
        except Exception as e:
            print(f"✗ Embedding test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Ollama integration tests."""
        print("=" * 60)
        print("LETTA-OLLAMA INTEGRATION TEST")
        print("=" * 60)
        
        results = {
            "ollama_connectivity": False,
            "letta_connectivity": False,
            "agent_creation": False,
            "model_switching": False,
            "embedding_operations": False
        }
        
        # Check prerequisites
        results["ollama_connectivity"] = await self.check_ollama_connectivity()
        if not results["ollama_connectivity"]:
            print("\n✗ Cannot proceed without Ollama")
            return results
        
        results["letta_connectivity"] = await self.check_letta_connectivity()
        if not results["letta_connectivity"]:
            print("\n✗ Cannot proceed without Letta server")
            return results
        
        # Test with first available model
        if self.available_models:
            model_to_test = self.available_models[0]
            results["agent_creation"] = await self.test_agent_with_ollama_model(model_to_test)
        else:
            print("\n✗ No generation models available")
            print("  Install a model with: ollama pull gpt-oss:20b")
        
        # Test model switching
        results["model_switching"] = await self.test_model_switching()
        
        # Test embeddings
        results["embedding_operations"] = await self.test_embedding_operations()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        for test, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test:25} {status}")
        
        all_passed = all(results.values())
        print("\n" + ("✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"))
        
        # Recommendations
        if not all_passed:
            print("\nRecommendations:")
            if not results["ollama_connectivity"]:
                print("  - Start Ollama: ollama serve")
            if not results["letta_connectivity"]:
                print("  - Start Letta: letta server")
            if not self.available_models:
                print("  - Install a model: ollama pull gpt-oss:20b")
            if not self.available_embed_models:
                print("  - Install embeddings: ollama pull nomic-embed-text")
        
        return results


async def main():
    """Main test execution."""
    tester = LettaOllamaTest()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())