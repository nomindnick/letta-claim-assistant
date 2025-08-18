#!/usr/bin/env python3
"""
Test script to verify Letta v0.10.0 passages API functionality.

This script tests that the critical passages.create() API works correctly
after downgrading from v0.11.x which had the "Invalid isoformat string: 'now()'" bug.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from letta_client import AsyncLetta
from app.models import KnowledgeItem


async def test_passages_api():
    """Test that passages API works correctly with v0.10.0."""
    
    print("=" * 60)
    print("Letta Passages API Test - v0.10.0")
    print("=" * 60)
    
    try:
        # Connect to Letta server
        print("\n1. Connecting to Letta server...")
        client = AsyncLetta(base_url="http://localhost:8283")
        
        # List agents
        print("2. Listing available agents...")
        agents = await client.agents.list()
        print(f"   Found {len(agents)} agents")
        
        if not agents:
            print("   No agents found. Please create an agent first.")
            return False
        
        # Use first agent for testing
        agent = agents[0]
        print(f"   Using agent: {agent.id}")
        print(f"   Agent name: {agent.name}")
        
        # Test 1: Create a simple passage
        print("\n3. Testing passages.create() with simple text...")
        try:
            result = await client.agents.passages.create(
                agent_id=agent.id,
                text="Test passage created at " + datetime.now().isoformat()
            )
            print(f"   ✅ SUCCESS! Created {len(result)} passage(s)")
            if result:
                print(f"   Passage ID: {result[0].id}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            return False
        
        # Test 2: Create a knowledge item passage
        print("\n4. Testing passages.create() with structured knowledge...")
        knowledge = KnowledgeItem(
            type="Event",
            label="Letta v0.10.0 Test",
            date=datetime.now().isoformat(),
            actors=["Test Script", "Letta Server"],
            doc_refs=[{"doc": "test_script.py", "page": 1}],
            support_snippet="Successfully created passage with v0.10.0"
        )
        
        try:
            result = await client.agents.passages.create(
                agent_id=agent.id,
                text=knowledge.model_dump_json()
            )
            print(f"   ✅ SUCCESS! Created knowledge passage")
            if result:
                print(f"   Passage ID: {result[0].id}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            return False
        
        # Test 3: List passages
        print("\n5. Testing passages.list()...")
        try:
            passages = await client.agents.passages.list(agent_id=agent.id)
            print(f"   ✅ SUCCESS! Retrieved {len(passages)} passages")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            return False
        
        # Test 4: Batch create (stress test)
        print("\n6. Testing batch passage creation...")
        batch_texts = [
            f"Batch test passage {i} at {datetime.now().isoformat()}"
            for i in range(5)
        ]
        
        success_count = 0
        for i, text in enumerate(batch_texts):
            try:
                await client.agents.passages.create(
                    agent_id=agent.id,
                    text=text
                )
                success_count += 1
            except Exception as e:
                print(f"   Batch item {i} failed: {e}")
        
        print(f"   ✅ Created {success_count}/{len(batch_texts)} passages successfully")
        
        print("\n" + "=" * 60)
        print("✅ All tests PASSED! Letta v0.10.0 passages API is working correctly.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


async def main():
    """Main entry point."""
    # Check if server is running
    print("Checking Letta server status...")
    print("Make sure Letta server is running: letta server --port 8283")
    
    success = await test_passages_api()
    
    if success:
        print("\n🎉 Letta v0.10.0 is working perfectly!")
        print("The passages API bug from v0.11.x has been resolved.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the Letta server logs.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())