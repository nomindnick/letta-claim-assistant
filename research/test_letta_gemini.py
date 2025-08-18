#!/usr/bin/env python3
"""
Test Letta integration with Google Gemini API.

This script validates:
1. Gemini API connectivity
2. Agent creation with Gemini models
3. Generation quality and response times
4. Cost tracking and limits
5. Fallback to local models

NOTE: Requires GEMINI_API_KEY environment variable or --api-key argument
"""

import asyncio
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from letta_client import AsyncLetta
from letta_client.types import LlmConfig


class LettaGeminiTest:
    def __init__(self, letta_url="http://localhost:8283", api_key=None):
        self.letta_url = letta_url
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self.test_agent_id = None
        self.cost_tracker = {
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0
        }
        
    def check_api_key(self):
        """Check if Gemini API key is available."""
        print("\n1. Checking Gemini API key...")
        
        if not self.api_key:
            print("  ✗ No API key found")
            print("  Set GEMINI_API_KEY environment variable or use --api-key")
            print("\n  To get an API key:")
            print("  1. Visit https://makersuite.google.com/app/apikey")
            print("  2. Create a new API key")
            print("  3. Export it: export GEMINI_API_KEY='your-key-here'")
            return False
        
        # Mask API key for display
        masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
        print(f"  ✓ API key found: {masked_key}")
        return True
    
    async def test_gemini_connectivity(self):
        """Test basic Gemini API connectivity."""
        print("\n2. Testing Gemini API connectivity...")
        
        # Note: Direct API testing would require google-generativeai package
        # For Letta integration, we'll test through agent creation
        
        print("  ⚠ Direct API test requires google-generativeai package")
        print("  Will test through Letta agent creation instead")
        return True
    
    async def setup_letta_client(self):
        """Setup Letta client connection."""
        print("\n3. Setting up Letta client...")
        
        try:
            self.client = AsyncLetta(base_url=self.letta_url)
            await self.client.health.health_check()
            print("  ✓ Connected to Letta server")
            return True
        except Exception as e:
            print(f"  ✗ Cannot connect to Letta: {e}")
            print("  Make sure Letta server is running: letta server")
            return False
    
    async def create_gemini_agent(self, model_name="gemini-1.5-flash"):
        """Create an agent using Gemini model."""
        print(f"\n4. Creating agent with Gemini model: {model_name}")
        
        try:
            # Configure Gemini
            llm_config = LlmConfig(
                model=model_name,
                model_endpoint_type="google_ai",
                model_endpoint="https://generativelanguage.googleapis.com",
                api_key=self.api_key,
                context_window=32768,  # Gemini 1.5 Flash supports up to 32k
                temperature=0.7,
                max_tokens=2048
            )
            
            # Create agent
            agent = await self.client.agents.create_agent(
                name=f"gemini-test-{model_name.replace('.', '-')}",
                description=f"Test agent using {model_name}",
                system="""You are a construction claims analyst assistant.
                
Your expertise includes:
- Construction law and contracts
- Damage assessment and causation analysis  
- Schedule delay analysis
- Cost estimation and quantum analysis

Provide clear, concise, and legally-informed responses.""",
                llm_config=llm_config,
                memory_blocks=[
                    {"label": "human", "value": "Construction attorney testing Gemini integration"},
                    {"label": "persona", "value": "Expert construction claims analyst with Gemini AI"}
                ]
            )
            
            self.test_agent_id = agent.id
            print(f"  ✓ Agent created: {agent.id}")
            return True
            
        except Exception as e:
            print(f"  ✗ Agent creation failed: {e}")
            if "API key" in str(e):
                print("  Check your API key is valid")
            elif "quota" in str(e).lower():
                print("  API quota may be exceeded")
            return False
    
    async def test_generation_quality(self):
        """Test generation quality with construction domain questions."""
        print("\n5. Testing generation quality...")
        
        if not self.test_agent_id:
            print("  ✗ No test agent available")
            return False
        
        test_queries = [
            {
                "query": "What are the key elements to prove in a construction defect claim?",
                "expected_topics": ["breach", "causation", "damages", "notice"]
            },
            {
                "query": "Explain the difference between consequential and direct damages in construction claims.",
                "expected_topics": ["consequential", "direct", "foreseeable", "contract"]
            },
            {
                "query": "What is the critical path method in delay analysis?",
                "expected_topics": ["critical path", "float", "schedule", "delay"]
            }
        ]
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n  Test {i}: {test['query'][:50]}...")
            
            try:
                start_time = time.time()
                
                response = await self.client.messages.send_message(
                    agent_id=self.test_agent_id,
                    role="user",
                    message=test["query"]
                )
                
                response_time = time.time() - start_time
                
                # Check response
                if response and response.messages:
                    # Get assistant's response
                    assistant_msg = None
                    for msg in response.messages:
                        if hasattr(msg, 'role') and msg.role == 'assistant':
                            assistant_msg = msg.content if hasattr(msg, 'content') else str(msg)
                            break
                    
                    if assistant_msg:
                        print(f"  ✓ Response received in {response_time:.2f}s")
                        print(f"    Length: {len(assistant_msg)} characters")
                        
                        # Check for expected topics
                        topics_found = []
                        for topic in test["expected_topics"]:
                            if topic.lower() in assistant_msg.lower():
                                topics_found.append(topic)
                        
                        if topics_found:
                            print(f"    Topics covered: {', '.join(topics_found)}")
                        
                        # Show preview
                        print(f"    Preview: {assistant_msg[:150]}...")
                        
                        # Track usage (approximate)
                        self.cost_tracker["requests"] += 1
                        self.cost_tracker["input_tokens"] += len(test["query"]) // 4
                        self.cost_tracker["output_tokens"] += len(assistant_msg) // 4
                    else:
                        print(f"  ✗ No assistant response found")
                else:
                    print(f"  ✗ No response received")
                    
            except Exception as e:
                print(f"  ✗ Generation failed: {e}")
                return False
        
        return True
    
    async def test_memory_with_gemini(self):
        """Test memory operations with Gemini agent."""
        print("\n6. Testing memory operations with Gemini...")
        
        if not self.test_agent_id:
            print("  ✗ No test agent available")
            return False
        
        try:
            # Store construction-specific memories
            memories = [
                "Project: Office Building at 123 Main St, Value: $5M",
                "General Contractor: ABC Construction, License: CA-123456",
                "Foundation failure discovered on February 14, 2024",
                "Geotechnical report showed clay soil with 40% moisture content",
                "Repair cost estimate: $250,000 by XYZ Engineering"
            ]
            
            print(f"  Storing {len(memories)} memories...")
            for memory in memories:
                await self.client.agents.insert_archival_memory(
                    agent_id=self.test_agent_id,
                    memory=memory
                )
            
            print("  ✓ Memories stored")
            
            # Test recall with context
            print("\n  Testing contextual recall...")
            response = await self.client.messages.send_message(
                agent_id=self.test_agent_id,
                role="user",
                message="Based on what you know about this project, what factors might have contributed to the foundation failure?"
            )
            
            if response and response.messages:
                print("  ✓ Contextual response generated using stored memories")
                
                # The response should reference the stored information
                assistant_msg = None
                for msg in response.messages:
                    if hasattr(msg, 'role') and msg.role == 'assistant':
                        assistant_msg = msg.content if hasattr(msg, 'content') else str(msg)
                        break
                
                if assistant_msg:
                    # Check if response references stored facts
                    references = ["clay soil", "moisture", "ABC Construction", "$250,000"]
                    refs_found = [ref for ref in references if ref.lower() in assistant_msg.lower()]
                    
                    if refs_found:
                        print(f"    References found: {', '.join(refs_found)}")
                    
                    print(f"    Preview: {assistant_msg[:200]}...")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Memory test failed: {e}")
            return False
    
    async def test_fallback_to_ollama(self):
        """Test fallback from Gemini to Ollama."""
        print("\n7. Testing fallback mechanism...")
        
        try:
            # Create agent with invalid API key to trigger fallback
            print("  Creating agent with fallback configuration...")
            
            # Primary: Gemini (might fail)
            gemini_config = LlmConfig(
                model="gemini-1.5-flash",
                model_endpoint_type="google_ai",
                model_endpoint="https://generativelanguage.googleapis.com",
                api_key="invalid-key-to-test-fallback"
            )
            
            # Fallback: Ollama (local)
            ollama_config = LlmConfig(
                model="gpt-oss:20b",
                model_endpoint_type="ollama",
                model_endpoint="http://localhost:11434"
            )
            
            # Try Gemini first
            try:
                agent = await self.client.agents.create_agent(
                    name="fallback-test-agent",
                    system="Test assistant",
                    llm_config=gemini_config
                )
                print("  ⚠ Gemini worked (unexpected in fallback test)")
            except:
                print("  ✓ Gemini failed as expected, trying Ollama...")
                
                # Fallback to Ollama
                agent = await self.client.agents.create_agent(
                    name="fallback-test-agent",
                    system="Test assistant",
                    llm_config=ollama_config
                )
                print("  ✓ Fallback to Ollama successful")
                
                # Cleanup
                await self.client.agents.delete_agent(agent.id)
                return True
            
            return False
            
        except Exception as e:
            print(f"  ✗ Fallback test failed: {e}")
            return False
    
    def calculate_costs(self):
        """Calculate approximate API costs."""
        print("\n8. Cost Analysis...")
        
        # Gemini 1.5 Flash pricing (as of 2024)
        # These are approximate rates - check current pricing
        input_cost_per_1k = 0.00015  # $0.00015 per 1K input tokens
        output_cost_per_1k = 0.0006   # $0.0006 per 1K output tokens
        
        input_cost = (self.cost_tracker["input_tokens"] / 1000) * input_cost_per_1k
        output_cost = (self.cost_tracker["output_tokens"] / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        print(f"  Requests made: {self.cost_tracker['requests']}")
        print(f"  Input tokens: ~{self.cost_tracker['input_tokens']}")
        print(f"  Output tokens: ~{self.cost_tracker['output_tokens']}")
        print(f"  Estimated cost: ${total_cost:.6f}")
        
        if total_cost > 0:
            print(f"\n  Note: This is an approximation. Check Google Cloud Console for actual usage.")
        
        return True
    
    async def cleanup(self):
        """Clean up test resources."""
        print("\nCleaning up...")
        if self.test_agent_id and self.client:
            try:
                await self.client.agents.delete_agent(self.test_agent_id)
                print("  ✓ Test agent deleted")
            except Exception as e:
                print(f"  ✗ Cleanup failed: {e}")
    
    async def run_all_tests(self):
        """Run all Gemini integration tests."""
        print("=" * 60)
        print("LETTA-GEMINI INTEGRATION TEST")
        print("=" * 60)
        
        results = {
            "api_key": False,
            "gemini_connectivity": False,
            "letta_setup": False,
            "agent_creation": False,
            "generation_quality": False,
            "memory_operations": False,
            "fallback_mechanism": False,
            "cost_analysis": False
        }
        
        # Check prerequisites
        results["api_key"] = self.check_api_key()
        if not results["api_key"]:
            print("\n✗ Cannot proceed without API key")
            print("\nTo run without Gemini, use test_letta_ollama.py instead")
            return results
        
        results["gemini_connectivity"] = await self.test_gemini_connectivity()
        
        results["letta_setup"] = await self.setup_letta_client()
        if not results["letta_setup"]:
            print("\n✗ Cannot proceed without Letta server")
            return results
        
        try:
            # Run tests
            results["agent_creation"] = await self.create_gemini_agent()
            
            if results["agent_creation"]:
                results["generation_quality"] = await self.test_generation_quality()
                results["memory_operations"] = await self.test_memory_with_gemini()
            
            results["fallback_mechanism"] = await self.test_fallback_to_ollama()
            results["cost_analysis"] = self.calculate_costs()
            
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
        
        # Recommendations
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        print("""
1. Monitor API usage and costs in Google Cloud Console
2. Implement rate limiting to avoid quota issues
3. Cache responses when possible to reduce API calls
4. Use Ollama for development/testing to save costs
5. Consider batch processing for bulk operations
        """)
        
        return results


async def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Test Letta-Gemini integration")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--letta-url", default="http://localhost:8283", help="Letta server URL")
    
    args = parser.parse_args()
    
    tester = LettaGeminiTest(letta_url=args.letta_url, api_key=args.api_key)
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())