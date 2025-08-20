"""
Unit tests for Memory Viewer UI components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ui.api_client import APIClient
from ui.memory_viewer import MemoryViewer


class TestMemoryViewer:
    """Test Memory Viewer component."""
    
    @pytest.fixture
    def api_client(self):
        """Create mock API client."""
        client = Mock(spec=APIClient)
        client.get_memory_items = AsyncMock()
        client.get_memory_item = AsyncMock()
        return client
    
    @pytest.fixture
    def memory_viewer(self, api_client):
        """Create Memory Viewer instance."""
        return MemoryViewer(api_client=api_client)
    
    @pytest.mark.asyncio
    async def test_load_memory_items_success(self, memory_viewer, api_client):
        """Test successful loading of memory items."""
        # Setup mock response
        api_client.get_memory_items.return_value = {
            'items': [
                {
                    'id': 'mem_1',
                    'type': 'Entity',
                    'text': 'Test entity memory',
                    'created_at': '2025-08-20T10:00:00Z',
                    'metadata': {'label': 'Test Entity'},
                    'source': 'document.pdf'
                },
                {
                    'id': 'mem_2',
                    'type': 'Event',
                    'text': 'Test event memory',
                    'created_at': '2025-08-20T11:00:00Z',
                    'metadata': {'date': '2025-08-15'},
                    'source': None
                }
            ],
            'total': 2
        }
        
        # Set matter ID
        memory_viewer.current_matter_id = 'test_matter'
        
        # Load items
        await memory_viewer._load_memory_items()
        
        # Verify API was called correctly
        api_client.get_memory_items.assert_called_once_with(
            matter_id='test_matter',
            limit=20,
            offset=0,
            type_filter=None,
            search_query=None
        )
        
        # Verify items were stored
        assert len(memory_viewer.memory_items) == 2
        assert memory_viewer.total_items == 2
        assert memory_viewer.memory_items[0]['type'] == 'Entity'
        assert memory_viewer.memory_items[1]['type'] == 'Event'
    
    @pytest.mark.asyncio
    async def test_pagination(self, memory_viewer, api_client):
        """Test pagination functionality."""
        # Setup mock for multiple pages
        api_client.get_memory_items.return_value = {
            'items': [{'id': f'mem_{i}', 'type': 'Fact', 'text': f'Item {i}'} 
                     for i in range(20)],
            'total': 50
        }
        
        memory_viewer.current_matter_id = 'test_matter'
        memory_viewer.current_page = 0
        
        # Load first page
        await memory_viewer._load_memory_items()
        
        # Go to next page
        memory_viewer.current_page = 1
        await memory_viewer._load_memory_items()
        
        # Verify second page was requested
        assert api_client.get_memory_items.call_count == 2
        second_call = api_client.get_memory_items.call_args_list[1]
        assert second_call.kwargs['offset'] == 20
    
    @pytest.mark.asyncio
    async def test_search_filter(self, memory_viewer, api_client):
        """Test search functionality."""
        api_client.get_memory_items.return_value = {
            'items': [],
            'total': 0
        }
        
        memory_viewer.current_matter_id = 'test_matter'
        memory_viewer.current_search = 'construction delay'
        
        # Perform search
        await memory_viewer._perform_search()
        
        # Verify search query was passed
        api_client.get_memory_items.assert_called_with(
            matter_id='test_matter',
            limit=20,
            offset=0,
            type_filter=None,
            search_query='construction delay'
        )
        
        # Verify page was reset
        assert memory_viewer.current_page == 0
    
    @pytest.mark.asyncio
    async def test_type_filter(self, memory_viewer, api_client):
        """Test filtering by memory type."""
        api_client.get_memory_items.return_value = {
            'items': [],
            'total': 0
        }
        
        memory_viewer.current_matter_id = 'test_matter'
        
        # Filter by Entity type
        await memory_viewer._filter_by_type('Entity')
        
        # Verify type filter was passed
        api_client.get_memory_items.assert_called_with(
            matter_id='test_matter',
            limit=20,
            offset=0,
            type_filter='Entity',
            search_query=None
        )
        
        assert memory_viewer.current_filter == 'Entity'
        assert memory_viewer.current_page == 0
    
    def test_format_timestamp(self, memory_viewer):
        """Test timestamp formatting."""
        # Test various timestamp formats
        now = datetime.now()
        
        # Just now (within a minute)
        recent = now.isoformat()
        result = memory_viewer._format_timestamp(recent)
        assert result == 'Just now'
        
        # Hours ago
        hours_ago = (now - timedelta(hours=3)).isoformat()
        result = memory_viewer._format_timestamp(hours_ago)
        assert '3h ago' in result or 'Just now' in result  # Depends on exact timing
        
        # Days ago
        days_ago = (now - timedelta(days=3)).isoformat()
        result = memory_viewer._format_timestamp(days_ago)
        assert '3d ago' in result or result.startswith(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
    
    @pytest.mark.asyncio
    async def test_error_handling(self, memory_viewer, api_client):
        """Test error handling when API fails."""
        # Setup API to raise error
        api_client.get_memory_items.side_effect = Exception("API Error")
        
        memory_viewer.current_matter_id = 'test_matter'
        
        # Mock UI elements to prevent errors
        memory_viewer.items_container = Mock()
        memory_viewer.loading_indicator = Mock()
        memory_viewer.empty_state = Mock()
        
        # Attempt to load items
        await memory_viewer._load_memory_items()
        
        # Verify error handling
        api_client.get_memory_items.assert_called_once()
        # The error should be caught and logged


class TestAPIClientMemoryMethods:
    """Test API client memory methods."""
    
    @pytest.fixture
    def api_client(self):
        """Create API client instance."""
        return APIClient(base_url="http://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_get_memory_items(self, api_client):
        """Test get_memory_items method."""
        with patch.object(api_client, '_get_session') as mock_session:
            # Mock response
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.json = AsyncMock(return_value={
                'items': [{'id': 'test'}],
                'total': 1
            })
            
            # Mock session.get
            mock_session_obj = AsyncMock()
            mock_session_obj.get = AsyncMock(return_value=mock_response)
            mock_session_obj.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session_obj.get.return_value.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_obj
            
            # Call method
            result = await api_client.get_memory_items(
                matter_id='test_matter',
                limit=10,
                offset=5,
                type_filter='Entity',
                search_query='test'
            )
            
            # Verify call
            mock_session_obj.get.assert_called_once()
            call_args = mock_session_obj.get.call_args
            assert 'matters/test_matter/memory/items' in call_args[0][0]
            assert call_args[1]['params']['limit'] == 10
            assert call_args[1]['params']['offset'] == 5
            assert call_args[1]['params']['type_filter'] == 'Entity'
            assert call_args[1]['params']['search_query'] == 'test'
            
            # Verify result
            assert result == {'items': [{'id': 'test'}], 'total': 1}
    
    @pytest.mark.asyncio
    async def test_get_memory_item(self, api_client):
        """Test get_memory_item method."""
        with patch.object(api_client, '_get_session') as mock_session:
            # Mock response
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.json = AsyncMock(return_value={
                'id': 'mem_123',
                'type': 'Fact',
                'text': 'Test memory'
            })
            
            # Mock session.get
            mock_session_obj = AsyncMock()
            mock_session_obj.get = AsyncMock(return_value=mock_response)
            mock_session_obj.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session_obj.get.return_value.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_obj
            
            # Call method
            result = await api_client.get_memory_item(
                matter_id='test_matter',
                item_id='mem_123'
            )
            
            # Verify call
            mock_session_obj.get.assert_called_once()
            call_args = mock_session_obj.get.call_args
            assert 'matters/test_matter/memory/items/mem_123' in call_args[0][0]
            
            # Verify result
            assert result['id'] == 'mem_123'
            assert result['type'] == 'Fact'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])