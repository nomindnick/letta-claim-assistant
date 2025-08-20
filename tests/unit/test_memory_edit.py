"""
Unit tests for memory edit functionality.

Tests CRUD operations for memory items including create, update, delete,
and audit logging functionality.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pydantic import ValidationError

from app.models import MemoryItem, CreateMemoryItemRequest, UpdateMemoryItemRequest
from app.letta_adapter import LettaAdapter


class TestMemoryEdit:
    """Test memory CRUD operations."""
    
    @pytest.fixture
    async def mock_letta_adapter(self):
        """Create a mock LettaAdapter for testing."""
        with patch('app.letta_adapter.AsyncLetta') as mock_async_letta:
            # Mock the client
            mock_client = AsyncMock()
            mock_async_letta.return_value = mock_client
            
            # Create a temporary matter path
            matter_path = Path("/tmp/test_matter")
            
            # Mock the matter_manager on the adapter
            with patch('app.letta_adapter.matter_manager') as mock_matter_mgr:
                mock_matter = MagicMock()
                mock_matter.id = "test-matter-id"
                mock_matter.name = "Test Matter"
                mock_matter.paths.root = matter_path
                mock_matter_mgr.get_matter_by_id.return_value = mock_matter
                
                # Create adapter
                adapter = LettaAdapter(
                    matter_path=matter_path,
                    matter_name="Test Matter",
                    matter_id="test-matter-id"
                )
                
                # Set the matter_manager on the adapter
                adapter.matter_manager = mock_matter_mgr
            
            # Set up the client on the adapter
            adapter.client = mock_client
            adapter.agent_id = "test-agent-id"
            
            yield adapter, mock_client
    
    @pytest.mark.asyncio
    async def test_create_memory_item_raw(self, mock_letta_adapter):
        """Test creating a raw memory item."""
        adapter, mock_client = mock_letta_adapter
        
        # Mock the passages.create response
        mock_passage = MagicMock()
        mock_passage.id = "new-passage-id"
        mock_client.agents.passages.create = AsyncMock(return_value=mock_passage)
        
        # Mock the audit logging
        with patch.object(adapter, '_log_memory_audit', new=AsyncMock()):
            # Create raw memory item
            item_id = await adapter.create_memory_item(
                text="This is a raw memory item",
                type="Raw"
            )
        
        assert item_id == "new-passage-id"
        mock_client.agents.passages.create.assert_called_once()
        
        # Check the call arguments
        call_args = mock_client.agents.passages.create.call_args
        assert call_args[1]['agent_id'] == "test-agent-id"
        assert call_args[1]['text'] == "This is a raw memory item"
    
    @pytest.mark.asyncio
    async def test_create_memory_item_knowledge(self, mock_letta_adapter):
        """Test creating a knowledge item (Entity, Event, etc.)."""
        adapter, mock_client = mock_letta_adapter
        
        # Mock the passages.create response
        mock_passage = MagicMock()
        mock_passage.id = "new-knowledge-id"
        mock_client.agents.passages.create = AsyncMock(return_value=mock_passage)
        
        # Mock the audit logging
        with patch.object(adapter, '_log_memory_audit', new=AsyncMock()):
            # Create Entity memory item
            item_id = await adapter.create_memory_item(
                text="John Doe is the project manager",
                type="Entity",
                metadata={"role": "manager"}
            )
        
        assert item_id == "new-knowledge-id"
        mock_client.agents.passages.create.assert_called_once()
        
        # Check that it was formatted as JSON
        call_args = mock_client.agents.passages.create.call_args
        text = call_args[1]['text']
        parsed = json.loads(text)
        assert parsed['type'] == "Entity"
        assert parsed['label'] == "John Doe is the project manager"
        assert parsed['doc_refs'] == [{"role": "manager"}]
    
    @pytest.mark.asyncio
    async def test_update_memory_item(self, mock_letta_adapter):
        """Test updating an existing memory item."""
        adapter, mock_client = mock_letta_adapter
        
        # Mock getting the existing item
        existing_item = MemoryItem(
            id="existing-id",
            text="Old text",
            type="Fact",
            metadata={"source": "document.pdf"}
        )
        
        with patch.object(adapter, 'get_memory_item', return_value=existing_item):
            with patch.object(adapter, '_backup_memory_item', new=AsyncMock()):
                with patch.object(adapter, '_log_memory_audit', new=AsyncMock()):
                    # Mock delete
                    mock_client.agents.passages.delete = AsyncMock()
                    
                    # Mock create for the new item
                    mock_passage = MagicMock()
                    mock_passage.id = "updated-id"
                    mock_client.agents.passages.create = AsyncMock(return_value=mock_passage)
                    
                    # Update the item
                    new_id = await adapter.update_memory_item(
                        item_id="existing-id",
                        new_text="Updated text",
                        preserve_type=True
                    )
        
        assert new_id == "updated-id"
        
        # Verify delete was called
        mock_client.agents.passages.delete.assert_called_once_with(
            agent_id="test-agent-id",
            passage_id="existing-id"
        )
        
        # Verify create was called with updated text
        mock_client.agents.passages.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_item(self, mock_letta_adapter):
        """Test updating a non-existent memory item raises error."""
        adapter, mock_client = mock_letta_adapter
        
        # Mock get_memory_item to return None
        with patch.object(adapter, 'get_memory_item', return_value=None):
            with pytest.raises(ValueError, match="Memory item not found"):
                await adapter.update_memory_item(
                    item_id="nonexistent-id",
                    new_text="New text"
                )
    
    @pytest.mark.asyncio
    async def test_delete_memory_item(self, mock_letta_adapter):
        """Test deleting a memory item."""
        adapter, mock_client = mock_letta_adapter
        
        # Mock getting the existing item for backup
        existing_item = MemoryItem(
            id="delete-id",
            text="Text to delete",
            type="Raw"
        )
        
        with patch.object(adapter, 'get_memory_item', return_value=existing_item):
            with patch.object(adapter, '_backup_memory_item', new=AsyncMock()):
                with patch.object(adapter, '_log_memory_audit', new=AsyncMock()):
                    # Mock delete
                    mock_client.agents.passages.delete = AsyncMock()
                    
                    # Delete the item
                    success = await adapter.delete_memory_item("delete-id")
        
        assert success is True
        mock_client.agents.passages.delete.assert_called_once_with(
            agent_id="test-agent-id",
            passage_id="delete-id"
        )
    
    @pytest.mark.asyncio
    async def test_backup_memory_item(self, mock_letta_adapter):
        """Test backing up a memory item before deletion."""
        adapter, mock_client = mock_letta_adapter
        
        # Create a memory item to backup
        item = MemoryItem(
            id="backup-id",
            text="Important memory",
            type="Fact",
            metadata={"important": True},
            created_at=datetime.now()
        )
        
        # Mock file operations
        mock_backup_data = []
        
        def mock_open_func(*args, **kwargs):
            mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
            if mode == 'r':
                return mock_open(read_data=json.dumps(mock_backup_data))()
            else:
                return mock_open()()
        
        with patch('builtins.open', side_effect=mock_open_func):
            with patch('json.dump') as mock_json_dump:
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.mkdir'):
                        await adapter._backup_memory_item(item)
        
        # Verify json.dump was called with backup data
        mock_json_dump.assert_called_once()
        backup_data = mock_json_dump.call_args[0][0]
        assert len(backup_data) == 1
        assert backup_data[0]['id'] == "backup-id"
        assert backup_data[0]['text'] == "Important memory"
        assert backup_data[0]['type'] == "Fact"
        assert 'deleted_at' in backup_data[0]
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, mock_letta_adapter):
        """Test audit logging for memory operations."""
        adapter, mock_client = mock_letta_adapter
        
        # Mock file operations
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file):
            with patch('pathlib.Path.mkdir'):
                await adapter._log_memory_audit(
                    operation="create",
                    item_id="audit-test-id",
                    item_type="Entity",
                    content_preview="Test entity"
                )
        
        # Verify file was opened for append
        mock_file.assert_called_once()
        assert 'a' in str(mock_file.call_args)
        
        # Verify audit entry was written
        handle = mock_file()
        written_data = ''.join(call[0][0] for call in handle.write.call_args_list)
        
        # Parse the written JSON
        audit_entry = json.loads(written_data.strip())
        assert audit_entry['operation'] == "create"
        assert audit_entry['item_id'] == "audit-test-id"
        assert audit_entry['item_type'] == "Entity"
        assert audit_entry['content_preview'] == "Test entity"
        assert 'timestamp' in audit_entry
    
    @pytest.mark.asyncio
    async def test_create_with_validation(self, mock_letta_adapter):
        """Test that empty text is rejected."""
        # This should raise a validation error when the model is created
        with pytest.raises(ValidationError) as exc_info:
            CreateMemoryItemRequest(text="   ", type="Raw")
        
        # Verify the error message
        assert "Text cannot be empty or only whitespace" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_memory_isolation(self, mock_letta_adapter):
        """Test that memory operations are isolated per matter."""
        adapter1, mock_client1 = mock_letta_adapter
        
        # Create second adapter for different matter
        with patch('app.letta_adapter.AsyncLetta') as mock_async_letta2:
            mock_client2 = AsyncMock()
            mock_async_letta2.return_value = mock_client2
            
            matter_path2 = Path("/tmp/different_matter")
            
            with patch('app.letta_adapter.matter_manager') as mock_matter_mgr2:
                mock_matter2 = MagicMock()
                mock_matter2.id = "different-matter-id"
                mock_matter2.name = "Different Matter"
                mock_matter2.paths.root = matter_path2
                mock_matter_mgr2.get_matter_by_id.return_value = mock_matter2
                
                adapter2 = LettaAdapter(
                    matter_path=matter_path2,
                    matter_name="Different Matter",
                    matter_id="different-matter-id"
                )
                
                adapter2.matter_manager = mock_matter_mgr2
            adapter2.client = mock_client2
            adapter2.agent_id = "different-agent-id"
            
            # Create memory in first adapter
            mock_passage1 = MagicMock()
            mock_passage1.id = "matter1-passage"
            mock_client1.agents.passages.create = AsyncMock(return_value=mock_passage1)
            
            # Create memory in second adapter
            mock_passage2 = MagicMock()
            mock_passage2.id = "matter2-passage"
            mock_client2.agents.passages.create = AsyncMock(return_value=mock_passage2)
            
            with patch.object(adapter1, '_log_memory_audit', new=AsyncMock()):
                with patch.object(adapter2, '_log_memory_audit', new=AsyncMock()):
                    id1 = await adapter1.create_memory_item("Matter 1 memory", "Raw")
                    id2 = await adapter2.create_memory_item("Matter 2 memory", "Raw")
            
            # Verify different agents were used
            assert adapter1.agent_id != adapter2.agent_id
            assert id1 != id2
            
            # Verify different agent IDs in API calls
            call1 = mock_client1.agents.passages.create.call_args[1]
            call2 = mock_client2.agents.passages.create.call_args[1]
            assert call1['agent_id'] != call2['agent_id']