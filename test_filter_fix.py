#!/usr/bin/env python3
"""
Test the memory filter fix.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from ui.api_client import APIClient
from app.logging_conf import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def test_filters():
    """Test that type filters now work correctly."""
    client = APIClient()
    
    try:
        # Get matters
        matters = await client.list_matters()
        if not matters:
            logger.warning("No matters found")
            return
        
        matter_id = matters[0]['id']
        logger.info(f"Testing with matter: {matters[0]['name']}")
        
        # Test each filter
        filters = [None, "Entity", "Event", "Issue", "Fact", "Interaction", "Raw"]
        
        for filter_type in filters:
            result = await client.get_memory_items(
                matter_id=matter_id,
                limit=20,
                offset=0,
                type_filter=filter_type
            )
            
            items = result.get('items', [])
            total = result.get('total', 0)
            
            if filter_type:
                logger.info(f"Filter '{filter_type}': {len(items)} items returned, total: {total}")
                
                # Check that all returned items match the filter
                if items:
                    types = set(item.get('type') for item in items)
                    if len(types) == 1 and filter_type in types:
                        logger.info(f"  ✓ All items correctly have type '{filter_type}'")
                    else:
                        logger.warning(f"  ✗ Mixed types returned: {types}")
            else:
                logger.info(f"No filter (All): {len(items)} items returned, total: {total}")
                if items:
                    types = set(item.get('type') for item in items)
                    logger.info(f"  Types present: {types}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_filters())