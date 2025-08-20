#!/usr/bin/env python3
"""
Test script for Memory Viewer API endpoints.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from ui.api_client import APIClient
from app.logging_conf import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def test_memory_endpoints():
    """Test the memory viewer API endpoints."""
    client = APIClient()
    
    try:
        # Get list of matters
        logger.info("Fetching matters...")
        matters = await client.list_matters()
        
        if not matters:
            logger.warning("No matters found. Please create a matter first.")
            return
        
        matter = matters[0]
        matter_id = matter['id']
        logger.info(f"Using matter: {matter['name']} (ID: {matter_id})")
        
        # Test get_memory_items endpoint
        logger.info("\nTesting get_memory_items endpoint...")
        try:
            result = await client.get_memory_items(
                matter_id=matter_id,
                limit=10,
                offset=0
            )
            
            items = result.get('items', [])
            total = result.get('total', 0)
            
            logger.info(f"Successfully fetched memory items: {total} total, {len(items)} returned")
            
            if items:
                logger.info("Sample memory item:")
                first_item = items[0]
                logger.info(f"  - ID: {first_item.get('id')}")
                logger.info(f"  - Type: {first_item.get('type')}")
                logger.info(f"  - Text: {first_item.get('text')[:100]}...")
                logger.info(f"  - Created: {first_item.get('created_at')}")
                
                # Test get_memory_item for specific item
                item_id = first_item.get('id')
                if item_id:
                    logger.info(f"\nTesting get_memory_item for ID: {item_id}")
                    try:
                        specific_item = await client.get_memory_item(matter_id, item_id)
                        logger.info(f"Successfully fetched specific item: {specific_item.get('type')}")
                    except Exception as e:
                        logger.error(f"Failed to fetch specific item: {e}")
            else:
                logger.info("No memory items found for this matter")
                
            # Test with type filter
            logger.info("\nTesting with type filter (Entity)...")
            try:
                filtered_result = await client.get_memory_items(
                    matter_id=matter_id,
                    limit=10,
                    offset=0,
                    type_filter="Entity"
                )
                filtered_items = filtered_result.get('items', [])
                logger.info(f"Found {len(filtered_items)} Entity-type memories")
            except Exception as e:
                logger.error(f"Failed to fetch with type filter: {e}")
                
            # Test with search query
            logger.info("\nTesting with search query...")
            try:
                search_result = await client.get_memory_items(
                    matter_id=matter_id,
                    limit=10,
                    offset=0,
                    search_query="contract"
                )
                search_items = search_result.get('items', [])
                logger.info(f"Found {len(search_items)} memories matching 'contract'")
            except Exception as e:
                logger.error(f"Failed to fetch with search query: {e}")
                
        except Exception as e:
            logger.error(f"Failed to fetch memory items: {e}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        await client.close()
        logger.info("\nTest completed")


if __name__ == "__main__":
    asyncio.run(test_memory_endpoints())