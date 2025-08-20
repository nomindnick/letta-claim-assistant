#!/usr/bin/env python3
"""
Debug script to test type filtering issue.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from ui.api_client import APIClient
from app.logging_conf import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def test_type_filtering():
    """Test type filtering to debug the issue."""
    client = APIClient()
    
    try:
        # Get list of matters
        logger.info("Fetching matters...")
        matters = await client.list_matters()
        
        if not matters:
            logger.warning("No matters found.")
            return
        
        matter = matters[0]
        matter_id = matter['id']
        logger.info(f"Using matter: {matter['name']} (ID: {matter_id})")
        
        # First, get all items to see what types exist
        logger.info("\n=== Getting ALL memory items ===")
        all_result = await client.get_memory_items(
            matter_id=matter_id,
            limit=100,
            offset=0
        )
        
        all_items = all_result.get('items', [])
        logger.info(f"Total items: {all_result.get('total', 0)}")
        
        # Count types
        type_counts = {}
        for item in all_items:
            item_type = item.get('type', 'Unknown')
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        logger.info("\nType distribution:")
        for type_name, count in sorted(type_counts.items()):
            logger.info(f"  {type_name}: {count}")
        
        # Now test each filter
        test_types = ["Entity", "Event", "Issue", "Fact", "Interaction", "Raw"]
        
        logger.info("\n=== Testing individual type filters ===")
        for test_type in test_types:
            result = await client.get_memory_items(
                matter_id=matter_id,
                limit=10,
                offset=0,
                type_filter=test_type
            )
            
            items = result.get('items', [])
            total = result.get('total', 0)
            
            logger.info(f"{test_type}: {total} items (returned {len(items)})")
            
            # Verify the returned items have the correct type
            if items:
                actual_types = set(item.get('type') for item in items)
                if len(actual_types) == 1 and test_type in actual_types:
                    logger.info(f"  ✓ Filter working correctly")
                else:
                    logger.warning(f"  ✗ Unexpected types returned: {actual_types}")
        
        # Test with non-existent type
        logger.info("\n=== Testing non-existent type ===")
        result = await client.get_memory_items(
            matter_id=matter_id,
            limit=10,
            offset=0,
            type_filter="NonExistentType"
        )
        logger.info(f"NonExistentType: {result.get('total', 0)} items")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        await client.close()
        logger.info("\n=== Debug test completed ===")


if __name__ == "__main__":
    asyncio.run(test_type_filtering())