#!/usr/bin/env python3
"""
Test script for Sprint 8: Memory Search and Analytics.

Tests the enhanced search capabilities and analytics features.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.letta_adapter import LettaAdapter
from app.models import MemoryAnalytics, MemoryPattern, MemoryInsight
from app.matters import matter_manager
import json


async def test_regex_search():
    """Test regex search functionality."""
    print("\n=== Testing Regex Search ===")
    
    # Get or create a test matter
    matters = matter_manager.list_matters()
    if not matters:
        print("No matters found. Please create a matter first.")
        return False
    
    matter = matters[0]
    print(f"Using matter: {matter.name}")
    
    # Create adapter
    adapter = LettaAdapter(
        matter_path=matter.paths.root,
        matter_name=matter.name,
        matter_id=matter.id
    )
    
    # Test regex search
    regex_patterns = [
        r"\d{4}-\d{2}-\d{2}",  # Date pattern
        r"contractor|subcontractor",  # Either word
        r"delay.*days?",  # Delay followed by day/days
    ]
    
    for pattern in regex_patterns:
        print(f"\nSearching with regex: {pattern}")
        results = await adapter.search_memories(
            query=pattern,
            limit=5,
            search_type="regex"
        )
        print(f"Found {len(results)} matches")
        for i, item in enumerate(results[:3], 1):
            print(f"  {i}. {item.text[:100]}...")
    
    return True


async def test_search_caching():
    """Test search result caching."""
    print("\n=== Testing Search Caching ===")
    
    matters = matter_manager.list_matters()
    if not matters:
        print("No matters found.")
        return False
    
    matter = matters[0]
    adapter = LettaAdapter(
        matter_path=matter.paths.root,
        matter_name=matter.name,
        matter_id=matter.id
    )
    
    # Perform same search twice
    import time
    query = "construction delay"
    
    print(f"First search for: {query}")
    start = time.time()
    results1 = await adapter.search_memories(query, limit=10)
    time1 = time.time() - start
    print(f"First search took: {time1:.3f}s, found {len(results1)} items")
    
    print(f"\nSecond search for: {query} (should be cached)")
    start = time.time()
    results2 = await adapter.search_memories(query, limit=10)
    time2 = time.time() - start
    print(f"Second search took: {time2:.3f}s, found {len(results2)} items")
    
    if time2 < time1:
        print("✓ Caching is working (second search was faster)")
    
    return True


async def test_analytics_endpoint():
    """Test analytics endpoint."""
    print("\n=== Testing Analytics Endpoint ===")
    
    from app.api import app
    from fastapi.testclient import TestClient
    
    matters = matter_manager.list_matters()
    if not matters:
        print("No matters found.")
        return False
    
    matter = matters[0]
    print(f"Testing analytics for matter: {matter.name}")
    
    # Create test client
    client = TestClient(app)
    
    # Call analytics endpoint
    response = client.get(f"/api/matters/{matter.id}/memory/analytics")
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return False
    
    analytics = response.json()
    
    print(f"\nAnalytics Summary:")
    print(f"- Total memories: {analytics.get('total_memories', 0)}")
    
    if analytics.get('type_distribution'):
        print(f"\nType Distribution:")
        for mem_type, count in analytics['type_distribution'].items():
            print(f"  - {mem_type}: {count}")
    
    if analytics.get('patterns'):
        print(f"\nIdentified Patterns: {len(analytics['patterns'])}")
        for pattern in analytics['patterns'][:3]:
            print(f"  - {pattern.get('type', 'Unknown')}: {pattern.get('value', 'N/A')}")
    
    if analytics.get('insights'):
        print(f"\nKey Insights: {len(analytics['insights'])}")
        for insight in analytics['insights']:
            print(f"  - {insight.get('insight', 'Unknown')}: {insight.get('interpretation', 'N/A')}")
    
    if analytics.get('actor_network'):
        print(f"\nTop Actors:")
        for actor, count in list(analytics['actor_network'].items())[:5]:
            print(f"  - {actor}: {count} mentions")
    
    return True


async def test_search_types():
    """Test different search types."""
    print("\n=== Testing Search Types ===")
    
    from app.api import app
    from fastapi.testclient import TestClient
    
    matters = matter_manager.list_matters()
    if not matters:
        print("No matters found.")
        return False
    
    matter = matters[0]
    client = TestClient(app)
    
    search_configs = [
        ("semantic", "construction delays"),
        ("keyword", "delay construction"),
        ("exact", "the contractor"),
        ("regex", r"\d+ days?")
    ]
    
    for search_type, query in search_configs:
        print(f"\n{search_type.upper()} search for: '{query}'")
        
        response = client.get(
            f"/api/matters/{matter.id}/memory/items",
            params={
                "search_query": query,
                "search_type": search_type,
                "limit": 5
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            print(f"  Found {len(items)} results")
            for i, item in enumerate(items[:2], 1):
                text = item.get('text', '')[:80]
                print(f"    {i}. {text}...")
        else:
            print(f"  Error: {response.status_code}")
    
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Sprint 8: Memory Search and Analytics - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Regex Search", test_regex_search),
        ("Search Caching", test_search_caching),
        ("Analytics Endpoint", test_analytics_endpoint),
        ("Search Types", test_search_types)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"ERROR in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)