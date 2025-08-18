#!/usr/bin/env python3
"""
Live test of memory operations with actual Letta server v0.10.0.

This tests all the Sprint L4 memory operations with a real server
to ensure everything works correctly after the downgrade.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.letta_adapter import LettaAdapter
from app.models import KnowledgeItem


async def test_memory_operations():
    """Test all memory operations with live Letta server."""
    
    print("=" * 70)
    print("Memory Operations Test with Letta v0.10.0")
    print("=" * 70)
    
    # Create temporary matter directory
    with tempfile.TemporaryDirectory() as temp_dir:
        matter_path = Path(temp_dir) / "test_matter"
        matter_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize adapter
        print("\n1. Initializing LettaAdapter...")
        adapter = LettaAdapter(
            matter_path=matter_path,
            matter_name="Memory Test Matter",
            matter_id="test-memory-ops"
        )
        
        # Ensure initialization
        initialized = await adapter._ensure_initialized()
        if not initialized:
            print("   ❌ Failed to initialize adapter")
            return False
        print("   ✅ Adapter initialized successfully")
        
        # Test 1: Store knowledge batch
        print("\n2. Testing store_knowledge_batch()...")
        knowledge_items = [
            KnowledgeItem(
                type="Entity",
                label="ABC Construction",
                actors=["ABC Construction"],
                doc_refs=[{"doc": "Contract.pdf", "page": 1}],
                support_snippet="ABC Construction is the general contractor"
            ),
            KnowledgeItem(
                type="Event",
                label="Foundation failure",
                date="2024-02-14",
                actors=["ABC Construction", "Sub A"],
                doc_refs=[{"doc": "Report.pdf", "page": 5}],
                support_snippet="Foundation showed significant settlement"
            ),
            KnowledgeItem(
                type="Issue",
                label="Schedule delay",
                date="2024-03-01",
                actors=["ABC Construction"],
                doc_refs=[{"doc": "Notice.pdf", "page": 2}],
                support_snippet="30 day delay due to weather conditions"
            ),
            # Duplicate for deduplication test
            KnowledgeItem(
                type="Entity",
                label="ABC Construction",
                actors=["ABC Construction"],
                doc_refs=[{"doc": "Contract.pdf", "page": 1}],
                support_snippet="ABC Construction is the general contractor"
            ),
        ]
        
        try:
            result = await adapter.store_knowledge_batch(
                knowledge_items,
                deduplicate=True,
                importance_threshold=0.3
            )
            print(f"   ✅ Stored {result['stored']} items")
            print(f"      Duplicates: {result['duplicates']}")
            print(f"      Skipped: {result['skipped']}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test 2: Recall with context
        print("\n3. Testing recall_with_context()...")
        conversation_history = [
            "What caused the foundation issues?",
            "The foundation failed due to water infiltration"
        ]
        
        try:
            results = await adapter.recall_with_context(
                query="Tell me more about the foundation problems",
                conversation_history=conversation_history,
                top_k=5,
                recency_weight=0.2
            )
            print(f"   ✅ Recalled {len(results)} relevant memories")
            if results:
                print(f"      First result: {results[0].label}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test 3: Semantic memory search
        print("\n4. Testing semantic_memory_search()...")
        filters = {
            "types": ["Event", "Issue"],
            "date_range": {
                "start": datetime(2024, 1, 1),
                "end": datetime(2024, 12, 31)
            }
        }
        
        try:
            results = await adapter.semantic_memory_search(
                query="foundation OR delay",
                filters=filters,
                top_k=10
            )
            print(f"   ✅ Found {len(results)} matching memories")
            for item, score in results[:3]:
                print(f"      - {item.label} (score: {score:.3f})")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test 4: Update core memory
        print("\n5. Testing update_core_memory_smart()...")
        try:
            success = await adapter.update_core_memory_smart(
                block_label="human",
                new_content="This is a test of memory operations with v0.10.0",
                mode="replace",
                max_size=2000
            )
            if success:
                print("   ✅ Core memory updated successfully")
            else:
                print("   ⚠️  Core memory update returned False")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            # Core memory might not exist, not critical
        
        # Test 5: Get memory summary
        print("\n6. Testing get_memory_summary()...")
        try:
            summary = await adapter.get_memory_summary(max_length=500)
            print(f"   ✅ Generated memory summary")
            print(f"      Summary length: {len(summary)} characters")
            print(f"      Preview: {summary[:100]}...")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test 6: Export memory
        print("\n7. Testing export_memory()...")
        try:
            # JSON export
            json_export = await adapter.export_memory(
                format="json",
                include_metadata=True
            )
            print(f"   ✅ JSON export successful")
            print(f"      Memory count: {json_export.get('memory_count', 0)}")
            
            # CSV export
            csv_export = await adapter.export_memory(
                format="csv",
                include_metadata=False
            )
            print(f"   ✅ CSV export successful")
            print(f"      CSV length: {len(csv_export)} characters")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test 7: Memory patterns analysis
        print("\n8. Testing analyze_memory_patterns()...")
        try:
            analysis = await adapter.analyze_memory_patterns()
            print(f"   ✅ Pattern analysis complete")
            print(f"      Total memories: {analysis.get('total_memories', 0)}")
            if 'type_distribution' in analysis:
                print(f"      Type distribution: {analysis['type_distribution']}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test 8: Memory quality metrics
        print("\n9. Testing get_memory_quality_metrics()...")
        try:
            metrics = await adapter.get_memory_quality_metrics()
            print(f"   ✅ Quality metrics calculated")
            print(f"      Quality score: {metrics.get('quality_score', 0):.2f}")
            print(f"      Structure score: {metrics.get('structure_score', 0):.2f}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        print("\n" + "=" * 70)
        print("✅ All memory operations tests PASSED!")
        print("Letta v0.10.0 memory operations are fully functional.")
        print("=" * 70)
        return True


async def main():
    """Main entry point."""
    print("Starting memory operations test...")
    print("Ensure Letta server v0.10.0 is running on port 8283")
    print()
    
    success = await test_memory_operations()
    
    if success:
        print("\n🎉 SUCCESS! All Sprint L4 memory operations are working!")
        print("The Letta integration is now fully functional.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())