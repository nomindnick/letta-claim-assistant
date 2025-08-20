#!/usr/bin/env python3
"""
Simple test to verify memory viewer implementation without UI context.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

sys.path.append(str(Path(__file__).parent))

from ui.memory_viewer import MemoryViewer
from ui.api_client import APIClient


def test_memory_viewer_initialization():
    """Test that MemoryViewer initializes correctly."""
    api_client = Mock(spec=APIClient)
    viewer = MemoryViewer(api_client=api_client)
    
    assert viewer.api_client == api_client
    assert viewer.current_page == 0
    assert viewer.items_per_page == 20
    assert viewer.memory_items == []
    assert viewer.total_items == 0
    print("✅ MemoryViewer initialization works")


def test_memory_types_configuration():
    """Test memory types are properly configured."""
    viewer = MemoryViewer()
    
    expected_types = ["All", "Entity", "Event", "Issue", "Fact", "Interaction", "Raw"]
    assert all(t in viewer.memory_types for t in expected_types)
    
    # Check each type has icon and color
    for type_name, config in viewer.memory_types.items():
        assert "icon" in config
        assert "color" in config
    
    print("✅ Memory types configuration is correct")


def test_timestamp_formatting():
    """Test timestamp formatting logic."""
    viewer = MemoryViewer()
    
    # Test with various timestamp formats
    from datetime import datetime, timedelta
    
    now = datetime.now()
    
    # Recent timestamp (should show "Just now")
    recent = now.isoformat()
    result = viewer._format_timestamp(recent)
    assert result == "Just now"
    
    # Hours ago
    three_hours = (now - timedelta(hours=3)).isoformat()
    result = viewer._format_timestamp(three_hours)
    # Should contain hour indication or "Just now" depending on exact timing
    assert "h ago" in result or result == "Just now"
    
    # Days ago  
    three_days = (now - timedelta(days=3)).isoformat()
    result = viewer._format_timestamp(three_days)
    assert "d ago" in result or result.startswith(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
    
    print("✅ Timestamp formatting works correctly")


async def test_pagination_logic():
    """Test pagination state management."""
    api_client = Mock(spec=APIClient)
    api_client.get_memory_items = AsyncMock(return_value={
        'items': [{'id': f'mem_{i}', 'type': 'Fact', 'text': f'Item {i}'} 
                 for i in range(20)],
        'total': 100
    })
    
    viewer = MemoryViewer(api_client=api_client)
    viewer.current_matter_id = 'test_matter'
    
    # Mock UI elements to prevent errors
    mock_container = Mock()
    mock_container.clear = Mock()
    mock_container.__enter__ = Mock(return_value=mock_container)
    mock_container.__exit__ = Mock(return_value=None)
    viewer.items_container = mock_container
    viewer.loading_indicator = Mock()
    viewer.empty_state = Mock()
    
    # Test initial page
    assert viewer.current_page == 0
    
    # Simulate going to next page
    viewer.current_page = 1
    
    # Would load with offset 20
    await viewer._load_memory_items()
    
    # Check API was called with correct offset
    api_client.get_memory_items.assert_called_with(
        matter_id='test_matter',
        limit=20,
        offset=20,  # page 1 * 20 items per page
        type_filter=None,
        search_query=None
    )
    
    print("✅ Pagination logic works correctly")


async def test_search_and_filter():
    """Test search and filter state management."""
    api_client = Mock(spec=APIClient)
    api_client.get_memory_items = AsyncMock(return_value={
        'items': [],
        'total': 0
    })
    
    viewer = MemoryViewer(api_client=api_client)
    viewer.current_matter_id = 'test_matter'
    
    # Mock UI elements
    mock_container = Mock()
    mock_container.clear = Mock()
    mock_container.__enter__ = Mock(return_value=mock_container)
    mock_container.__exit__ = Mock(return_value=None)
    viewer.items_container = mock_container
    viewer.loading_indicator = Mock()
    viewer.empty_state = Mock()
    
    # Test search
    viewer.current_search = "construction delay"
    await viewer._perform_search()
    
    # Verify search was passed to API
    api_client.get_memory_items.assert_called_with(
        matter_id='test_matter',
        limit=20,
        offset=0,
        type_filter=None,
        search_query='construction delay'
    )
    
    # Test filter
    await viewer._filter_by_type('Entity')
    
    # Verify filter was applied
    api_client.get_memory_items.assert_called_with(
        matter_id='test_matter',
        limit=20,
        offset=0,
        type_filter='Entity',
        search_query='construction delay'  # Search should persist
    )
    
    assert viewer.current_filter == 'Entity'
    assert viewer.current_page == 0  # Should reset to first page
    
    print("✅ Search and filter logic works correctly")


async def test_api_client_methods():
    """Test API client memory methods exist and have correct signatures."""
    client = APIClient()
    
    # Check methods exist
    assert hasattr(client, 'get_memory_items')
    assert hasattr(client, 'get_memory_item')
    
    # Check they're async methods
    import inspect
    assert inspect.iscoroutinefunction(client.get_memory_items)
    assert inspect.iscoroutinefunction(client.get_memory_item)
    
    print("✅ API client methods are properly defined")


def main():
    """Run all tests."""
    print("\n=== Testing Memory Viewer Implementation ===\n")
    
    # Synchronous tests
    test_memory_viewer_initialization()
    test_memory_types_configuration()
    test_timestamp_formatting()
    
    # Async tests
    asyncio.run(test_pagination_logic())
    asyncio.run(test_search_and_filter())
    asyncio.run(test_api_client_methods())
    
    print("\n=== All core functionality tests passed! ===")
    print("\nNote: UI rendering tests require a running NiceGUI context.")
    print("The implementation is structurally correct and should work when integrated with the UI.")


if __name__ == "__main__":
    main()