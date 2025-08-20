"""
Integration tests for memory command processing.

Tests end-to-end memory command flow including API endpoints,
RAG integration, and Letta adapter interactions.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from app.models import (
    MemoryCommandRequest, MemoryCommandResponse,
    ChatRequest, ChatMode, KnowledgeItem
)
from app.memory_commands import MemoryAction
from app.rag import RAGEngine, RAGResponse
from app.letta_adapter import LettaAdapter


@pytest.mark.asyncio
class TestMemoryCommandAPI:
    """Test memory command API endpoints."""
    
    async def test_process_remember_command(self, test_client, mock_matter):
        """Test processing a remember command via API."""
        # Setup mock Letta adapter
        with patch('app.api.LettaAdapter') as MockLettaAdapter:
            mock_adapter = AsyncMock()
            mock_adapter.create_memory_item = AsyncMock(return_value="test-memory-id")
            MockLettaAdapter.return_value = mock_adapter
            
            # Send remember command
            request = MemoryCommandRequest(
                command="remember that the deadline is March 15th",
                matter_id=mock_matter.id
            )
            
            response = await test_client.post(
                f"/api/matters/{mock_matter.id}/memory/command",
                json=request.model_dump()
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["action"] == "remember"
            assert data["content"] == "the deadline is March 15th"
            assert data["item_id"] == "test-memory-id"
            assert "I'll remember" in data["message"]
            assert data["undo_token"] is not None
    
    async def test_process_forget_command(self, test_client, mock_matter):
        """Test processing a forget command via API."""
        with patch('app.api.LettaAdapter') as MockLettaAdapter:
            mock_adapter = AsyncMock()
            mock_adapter.search_and_delete_memory = AsyncMock(return_value=True)
            MockLettaAdapter.return_value = mock_adapter
            
            request = MemoryCommandRequest(
                command="forget about the old deadline",
                matter_id=mock_matter.id
            )
            
            response = await test_client.post(
                f"/api/matters/{mock_matter.id}/memory/command",
                json=request.model_dump()
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["action"] == "forget"
            assert "forgotten" in data["message"]
    
    async def test_process_update_command(self, test_client, mock_matter):
        """Test processing an update command via API."""
        with patch('app.api.LettaAdapter') as MockLettaAdapter:
            mock_adapter = AsyncMock()
            mock_adapter.search_and_update_memory = AsyncMock(return_value=True)
            MockLettaAdapter.return_value = mock_adapter
            
            request = MemoryCommandRequest(
                command="update the deadline to April 1st",
                matter_id=mock_matter.id
            )
            
            response = await test_client.post(
                f"/api/matters/{mock_matter.id}/memory/command",
                json=request.model_dump()
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["action"] == "update"
            assert "updated" in data["message"]
    
    async def test_process_query_command(self, test_client, mock_matter):
        """Test processing a query command via API."""
        with patch('app.api.LettaAdapter') as MockLettaAdapter:
            mock_adapter = AsyncMock()
            mock_memory = MagicMock()
            mock_memory.text = "The deadline is March 15th"
            mock_adapter.search_memories = AsyncMock(return_value=[mock_memory])
            MockLettaAdapter.return_value = mock_adapter
            
            request = MemoryCommandRequest(
                command="what do you remember about the deadline",
                matter_id=mock_matter.id
            )
            
            response = await test_client.post(
                f"/api/matters/{mock_matter.id}/memory/command",
                json=request.model_dump()
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["action"] == "query"
            assert "remember about" in data["message"]
    
    async def test_invalid_command(self, test_client, mock_matter):
        """Test handling of invalid/unrecognized commands."""
        request = MemoryCommandRequest(
            command="this is not a memory command",
            matter_id=mock_matter.id
        )
        
        response = await test_client.post(
            f"/api/matters/{mock_matter.id}/memory/command",
            json=request.model_dump()
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is False
        assert "couldn't understand" in data["message"].lower()
        # May have a suggestion
        if data.get("suggestion"):
            assert isinstance(data["suggestion"], str)


@pytest.mark.asyncio
class TestRAGMemoryCommandIntegration:
    """Test memory command integration with RAG engine."""
    
    async def test_rag_detects_remember_command(self, mock_matter):
        """Test that RAG engine detects and processes remember commands."""
        # Setup mocks
        mock_vector_store = AsyncMock()
        mock_llm_provider = AsyncMock()
        mock_letta_adapter = AsyncMock()
        mock_letta_adapter.create_memory_item = AsyncMock(return_value="memory-123")
        
        # Create RAG engine
        rag_engine = RAGEngine(
            matter=mock_matter,
            vector_store=mock_vector_store,
            llm_provider=mock_llm_provider,
            letta_adapter=mock_letta_adapter
        )
        
        # Send remember command
        response = await rag_engine.generate_answer(
            query="remember that the contractor is ABC Corp",
            mode=ChatMode.COMBINED
        )
        
        assert response.is_memory_command is True
        assert "I'll remember" in response.answer
        assert response.sources == []  # No document sources for memory commands
        assert mock_letta_adapter.create_memory_item.called
    
    async def test_rag_detects_forget_command(self, mock_matter):
        """Test that RAG engine detects and processes forget commands."""
        mock_vector_store = AsyncMock()
        mock_llm_provider = AsyncMock()
        mock_letta_adapter = AsyncMock()
        mock_letta_adapter.search_and_delete_memory = AsyncMock(return_value=True)
        
        rag_engine = RAGEngine(
            matter=mock_matter,
            vector_store=mock_vector_store,
            llm_provider=mock_llm_provider,
            letta_adapter=mock_letta_adapter
        )
        
        response = await rag_engine.generate_answer(
            query="forget what I said about the old contractor",
            mode=ChatMode.COMBINED
        )
        
        assert response.is_memory_command is True
        assert "forgotten" in response.answer
        assert mock_letta_adapter.search_and_delete_memory.called
    
    async def test_rag_bypasses_commands_in_rag_only_mode(self, mock_matter):
        """Test that memory commands are ignored in RAG_ONLY mode."""
        mock_vector_store = AsyncMock()
        mock_vector_store.search = AsyncMock(return_value=[])
        
        mock_llm_provider = AsyncMock()
        mock_llm_provider.generate = AsyncMock(return_value="Regular RAG response")
        
        mock_letta_adapter = AsyncMock()
        
        rag_engine = RAGEngine(
            matter=mock_matter,
            vector_store=mock_vector_store,
            llm_provider=mock_llm_provider,
            letta_adapter=mock_letta_adapter
        )
        
        # Send remember command in RAG_ONLY mode
        response = await rag_engine.generate_answer(
            query="remember that the deadline is tomorrow",
            mode=ChatMode.RAG_ONLY
        )
        
        # Should process as normal query, not as command
        assert response.is_memory_command is False
        assert "Regular RAG response" in response.answer
        assert not mock_letta_adapter.create_memory_item.called
    
    async def test_memory_query_switches_to_memory_only_mode(self, mock_matter):
        """Test that memory query commands switch to memory-only mode."""
        mock_vector_store = AsyncMock()
        mock_llm_provider = AsyncMock()
        mock_letta_adapter = AsyncMock()
        mock_letta_adapter.memory_only_chat = AsyncMock(
            return_value={"answer": "I remember the deadline is March 15th", "memory_used": True}
        )
        
        rag_engine = RAGEngine(
            matter=mock_matter,
            vector_store=mock_vector_store,
            llm_provider=mock_llm_provider,
            letta_adapter=mock_letta_adapter
        )
        
        response = await rag_engine.generate_answer(
            query="what do you remember about the deadline",
            mode=ChatMode.COMBINED
        )
        
        # Should switch to memory-only mode for query
        assert mock_letta_adapter.memory_only_chat.called
        assert "I remember the deadline is March 15th" in response.answer


@pytest.mark.asyncio
class TestLettaAdapterMemoryOperations:
    """Test Letta adapter memory search and update methods."""
    
    async def test_search_memories_semantic(self):
        """Test semantic memory search."""
        with patch('app.letta_adapter.AsyncLetta') as MockAsyncLetta:
            mock_client = AsyncMock()
            mock_passages = MagicMock()
            mock_passages.list = AsyncMock(return_value=[
                MagicMock(id="1", text="The deadline is March 15th"),
                MagicMock(id="2", text="The budget is $500,000")
            ])
            mock_client.agents.passages = mock_passages
            MockAsyncLetta.return_value = mock_client
            
            adapter = LettaAdapter(
                matter_path=Path("/tmp/test"),
                matter_name="Test",
                matter_id="test-id"
            )
            adapter.client = mock_client
            adapter.agent_id = "test-agent"
            adapter.fallback_mode = False
            adapter._initialized = True
            
            results = await adapter.search_memories("deadline", limit=5, search_type="semantic")
            
            assert len(results) == 2
            assert mock_passages.list.called
    
    async def test_search_and_delete_memory(self):
        """Test searching and deleting memories."""
        with patch('app.letta_adapter.AsyncLetta') as MockAsyncLetta:
            mock_client = AsyncMock()
            adapter = LettaAdapter(
                matter_path=Path("/tmp/test"),
                matter_name="Test",
                matter_id="test-id"
            )
            adapter.client = mock_client
            adapter.agent_id = "test-agent"
            adapter.fallback_mode = False
            adapter._initialized = True
            
            # Mock search_memories to return some results
            mock_memory = MagicMock(id="memory-1", text="Old deadline info")
            adapter.search_memories = AsyncMock(return_value=[mock_memory])
            
            # Mock delete_memory_item
            adapter.delete_memory_item = AsyncMock(return_value=True)
            
            result = await adapter.search_and_delete_memory("old deadline")
            
            assert result is True
            assert adapter.search_memories.called
            assert adapter.delete_memory_item.called
    
    async def test_search_and_update_memory(self):
        """Test searching and updating memories."""
        with patch('app.letta_adapter.AsyncLetta') as MockAsyncLetta:
            mock_client = AsyncMock()
            adapter = LettaAdapter(
                matter_path=Path("/tmp/test"),
                matter_name="Test",
                matter_id="test-id"
            )
            adapter.client = mock_client
            adapter.agent_id = "test-agent"
            adapter.fallback_mode = False
            adapter._initialized = True
            
            # Mock search_memories to return a result
            mock_memory = MagicMock(
                id="memory-1",
                text='{"type": "Fact", "label": "deadline", "support_snippet": "March 15th"}'
            )
            adapter.search_memories = AsyncMock(return_value=[mock_memory])
            
            # Mock update_memory_item
            adapter.update_memory_item = AsyncMock(return_value="memory-1-updated")
            
            result = await adapter.search_and_update_memory(
                target="deadline",
                new_content="April 1st"
            )
            
            assert result is True
            assert adapter.search_memories.called
            assert adapter.update_memory_item.called


# Fixtures for testing
@pytest.fixture
def mock_matter():
    """Create a mock matter for testing."""
    matter = MagicMock()
    matter.id = "test-matter-123"
    matter.name = "Test Matter"
    matter.slug = "test-matter"
    matter.generation_model = "test-model"
    matter.embedding_model = "test-embed"
    matter.paths = MagicMock()
    matter.paths.root = Path("/tmp/test-matter")
    return matter


@pytest.fixture
async def test_client():
    """Create a test client for API testing."""
    from fastapi.testclient import TestClient
    from app.api import app
    
    with TestClient(app) as client:
        yield client