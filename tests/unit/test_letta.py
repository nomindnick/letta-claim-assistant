"""
Unit tests for LettaAdapter functionality.

Tests agent memory management, knowledge recall, interaction storage,
and follow-up generation.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.letta_adapter import LettaAdapter
from app.models import KnowledgeItem, SourceChunk


class TestLettaAdapter:
    """Test suite for LettaAdapter class."""
    
    @pytest.fixture
    def temp_matter_path(self):
        """Create temporary matter directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "test_matter"
            matter_path.mkdir(parents=True, exist_ok=True)
            yield matter_path
    
    @pytest.fixture
    def sample_knowledge_items(self):
        """Sample knowledge items for testing."""
        return [
            KnowledgeItem(
                type="Entity",
                label="ABC Construction Company",
                actors=["Owner", "Contractor"],
                doc_refs=[{"doc": "Contract.pdf", "page": 1}],
                support_snippet="ABC Construction Company is the general contractor"
            ),
            KnowledgeItem(
                type="Event",
                label="Foundation failure",
                date="2023-02-14",
                actors=["ABC Construction"],
                doc_refs=[{"doc": "Inspection_Report.pdf", "page": 3}],
                support_snippet="Foundation showed signs of settling on February 14"
            ),
            KnowledgeItem(
                type="Issue",
                label="Schedule delay",
                date="2023-03-01",
                actors=["ABC Construction", "Steel Supplier"],
                doc_refs=[{"doc": "Change_Order_5.pdf", "page": 2}],
                support_snippet="Steel delivery delayed project by 30 days"
            )
        ]
    
    @pytest.fixture
    def sample_source_chunks(self):
        """Sample source chunks for testing."""
        return [
            SourceChunk(
                doc="Specification.pdf",
                page_start=12,
                page_end=12,
                text="The foundation shall be designed for a minimum bearing capacity of 4000 psf",
                score=0.85
            ),
            SourceChunk(
                doc="Daily_Log_2023-02-15.pdf",
                page_start=1,
                page_end=1,
                text="Noticed settlement cracks in foundation wall, approximately 1/4 inch wide",
                score=0.72
            )
        ]
    
    def test_adapter_initialization_success(self, temp_matter_path):
        """Test successful LettaAdapter initialization."""
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock agent creation
            mock_agent_state = Mock()
            mock_agent_state.id = "test-agent-123"
            mock_client.create_agent.return_value = mock_agent_state
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            assert adapter.matter_path == temp_matter_path
            assert adapter.matter_name == "Test Claim"
            assert adapter.matter_id == "test-matter-123"
            assert adapter.letta_path == temp_matter_path / "knowledge" / "letta_state"
            assert adapter.agent_id == "test-agent-123"
            
            # Verify directories were created
            assert adapter.letta_path.exists()
    
    def test_adapter_initialization_fallback(self, temp_matter_path):
        """Test LettaAdapter initialization with Letta unavailable."""
        with patch('app.letta_adapter.LocalClient', None):
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            assert adapter.client is None
            assert adapter.agent_id is None
            assert adapter.letta_path.exists()  # Directory still created
    
    @pytest.mark.asyncio
    async def test_recall_with_letta_available(self, temp_matter_path):
        """Test memory recall when Letta is available."""
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock agent creation
            mock_agent_state = Mock()
            mock_agent_state.id = "test-agent-123"
            mock_client.create_agent.return_value = mock_agent_state
            
            # Mock memory recall
            mock_memory_obj1 = Mock()
            mock_memory_obj1.text = json.dumps({
                "type": "Entity",
                "label": "ABC Construction",
                "actors": ["Contractor"],
                "doc_refs": [{"doc": "Contract.pdf", "page": 1}]
            })
            
            mock_memory_obj2 = Mock()
            mock_memory_obj2.text = "Foundation failure observed on site"
            
            mock_client.get_archival_memory.return_value = [mock_memory_obj1, mock_memory_obj2]
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            # Test recall
            result = await adapter.recall("foundation issues", top_k=5)
            
            assert len(result) == 2
            assert result[0].type == "Entity"
            assert result[0].label == "ABC Construction"
            assert result[1].type == "Fact"  # Converted from unstructured
            assert "Foundation failure" in result[1].label
            
            # Verify Letta API was called
            mock_client.get_archival_memory.assert_called_once_with(
                agent_id="test-agent-123",
                limit=10  # top_k * 2
            )
    
    @pytest.mark.asyncio
    async def test_recall_with_letta_unavailable(self, temp_matter_path):
        """Test memory recall when Letta is unavailable."""
        with patch('app.letta_adapter.LocalClient', None):
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            result = await adapter.recall("foundation issues", top_k=5)
            
            assert result == []
    
    @pytest.mark.asyncio
    async def test_upsert_interaction_success(self, temp_matter_path, sample_source_chunks, sample_knowledge_items):
        """Test successful interaction storage."""
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock agent creation
            mock_agent_state = Mock()
            mock_agent_state.id = "test-agent-123"
            mock_client.create_agent.return_value = mock_agent_state
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            # Test interaction storage
            await adapter.upsert_interaction(
                user_query="What caused the foundation failure?",
                llm_answer="The foundation failure was likely caused by inadequate soil analysis...",
                sources=sample_source_chunks,
                extracted_facts=sample_knowledge_items[:2]  # First 2 items
            )
            
            # Verify Letta API calls
            # Should be called 3 times: 1 interaction summary + 2 knowledge items
            assert mock_client.insert_archival_memory.call_count == 3
            
            # Verify core memory update was called
            mock_client.update_agent_core_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upsert_interaction_letta_unavailable(self, temp_matter_path, sample_source_chunks, sample_knowledge_items):
        """Test interaction storage when Letta is unavailable."""
        with patch('app.letta_adapter.LocalClient', None):
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            # Should not raise exception
            await adapter.upsert_interaction(
                user_query="What caused the foundation failure?",
                llm_answer="The foundation failure was likely caused by inadequate soil analysis...",
                sources=sample_source_chunks,
                extracted_facts=sample_knowledge_items[:2]
            )
    
    @pytest.mark.asyncio
    async def test_suggest_followups_with_letta(self, temp_matter_path):
        """Test follow-up generation with Letta available."""
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock agent creation
            mock_agent_state = Mock()
            mock_agent_state.id = "test-agent-123"
            mock_client.create_agent.return_value = mock_agent_state
            
            # Mock follow-up generation
            mock_message = Mock()
            mock_message.text = """What additional soil testing was performed?
How much did the foundation repair cost?
Were there any schedule delays due to this issue?
Should we engage a geotechnical expert?"""
            
            mock_response = Mock()
            mock_response.messages = [mock_message]
            mock_client.user_message.return_value = mock_response
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            result = await adapter.suggest_followups(
                user_query="What caused the foundation failure?",
                llm_answer="The foundation failure was caused by inadequate soil analysis."
            )
            
            assert len(result) == 4
            assert "soil testing" in result[0]
            assert "foundation repair cost" in result[1]
            assert "schedule delays" in result[2]
            assert "geotechnical expert" in result[3]
            
            # Verify Letta API was called
            mock_client.user_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_suggest_followups_fallback(self, temp_matter_path):
        """Test follow-up generation fallback when Letta unavailable."""
        with patch('app.letta_adapter.LocalClient', None):
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            result = await adapter.suggest_followups(
                user_query="What caused the foundation failure?",
                llm_answer="The foundation failure was caused by inadequate soil analysis."
            )
            
            # Should return fallback suggestions
            assert len(result) == 4
            assert "additional documentation" in result[0]
            assert "schedule impacts" in result[1]
            assert "damages" in result[2]
            assert "technical experts" in result[3]
    
    @pytest.mark.asyncio
    async def test_get_memory_stats_with_letta(self, temp_matter_path):
        """Test memory statistics retrieval."""
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock agent creation
            mock_agent_state = Mock()
            mock_agent_state.id = "test-agent-123"
            mock_client.create_agent.return_value = mock_agent_state
            
            # Mock memory results
            mock_client.get_archival_memory.return_value = [Mock() for _ in range(25)]
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            stats = await adapter.get_memory_stats()
            
            assert stats["status"] == "active"
            assert stats["memory_items"] == 25
            assert stats["agent_id"] == "test-agent-123"
            assert stats["matter_name"] == "Test Claim"
    
    @pytest.mark.asyncio
    async def test_get_memory_stats_unavailable(self, temp_matter_path):
        """Test memory statistics when Letta unavailable."""
        with patch('app.letta_adapter.LocalClient', None):
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            stats = await adapter.get_memory_stats()
            
            assert stats["status"] == "unavailable"
            assert stats["memory_items"] == 0
    
    def test_agent_persistence_config(self, temp_matter_path):
        """Test agent configuration persistence."""
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock agent creation
            mock_agent_state = Mock()
            mock_agent_state.id = "test-agent-123"
            mock_client.create_agent.return_value = mock_agent_state
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            # Verify config file was created
            config_path = adapter.letta_path / "agent_config.json"
            assert config_path.exists()
            
            # Verify config contents
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            assert config["agent_id"] == "test-agent-123"
            assert config["matter_id"] == "test-matter-123"
            assert config["matter_name"] == "Test Claim"
            assert "created_at" in config
    
    def test_agent_loading_existing(self, temp_matter_path):
        """Test loading existing agent configuration."""
        # Create existing agent config
        letta_path = temp_matter_path / "knowledge" / "letta_state"
        letta_path.mkdir(parents=True, exist_ok=True)
        
        config_path = letta_path / "agent_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "agent_id": "existing-agent-456",
                "matter_id": "test-matter-123",
                "matter_name": "Test Claim",
                "created_at": "2023-01-01T00:00:00"
            }, f)
        
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock existing agent retrieval
            mock_agent_state = Mock()
            mock_agent_state.id = "existing-agent-456"
            mock_client.get_agent.return_value = mock_agent_state
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            # Should load existing agent, not create new one
            assert adapter.agent_id == "existing-agent-456"
            mock_client.get_agent.assert_called_once_with("existing-agent-456")
            mock_client.create_agent.assert_not_called()
    
    def test_construction_domain_configuration(self, temp_matter_path):
        """Test that agent is configured for construction domain."""
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock agent creation
            mock_agent_state = Mock()
            mock_agent_state.id = "test-agent-123"
            mock_client.create_agent.return_value = mock_agent_state
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Foundation Failure Claim",
                matter_id="test-matter-123"
            )
            
            # Verify create_agent was called with construction-specific config
            mock_client.create_agent.assert_called_once()
            call_args = mock_client.create_agent.call_args
            
            # Check persona contains construction-specific content
            persona_content = None
            human_content = None
            
            if 'persona' in call_args.kwargs:
                persona_content = call_args.kwargs['persona']
            elif 'memory' in call_args.kwargs:
                # Alternative API signature
                memory = call_args.kwargs['memory']
                persona_content = memory.persona if hasattr(memory, 'persona') else None
                human_content = memory.human if hasattr(memory, 'human') else None
            
            if persona_content:
                assert "construction claims analyst" in persona_content.lower()
                assert "Foundation Failure Claim" in persona_content
                assert "entities" in persona_content.lower()
                assert "events" in persona_content.lower()
                assert "issues" in persona_content.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling_json_parsing(self, temp_matter_path):
        """Test error handling for malformed JSON in memory."""
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock agent creation
            mock_agent_state = Mock()
            mock_agent_state.id = "test-agent-123"
            mock_client.create_agent.return_value = mock_agent_state
            
            # Mock memory with malformed JSON
            mock_memory_obj = Mock()
            mock_memory_obj.text = '{"type": "Entity", "label": "ABC Construction"'  # Missing closing brace
            
            mock_client.get_archival_memory.return_value = [mock_memory_obj]
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Test Claim",
                matter_id="test-matter-123"
            )
            
            # Should handle malformed JSON gracefully
            result = await adapter.recall("test query", top_k=5)
            
            # Should return empty list due to parsing error
            assert result == []
    
    @pytest.mark.asyncio
    async def test_matter_isolation(self, temp_matter_path):
        """Test that different matters have isolated agents."""
        with patch('app.letta_adapter.LocalClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock agent creation for first matter
            mock_agent_state1 = Mock()
            mock_agent_state1.id = "agent-matter-1"
            
            # Mock agent creation for second matter  
            mock_agent_state2 = Mock()
            mock_agent_state2.id = "agent-matter-2"
            
            mock_client.create_agent.side_effect = [mock_agent_state1, mock_agent_state2]
            
            # Create first adapter
            adapter1 = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="Matter 1",
                matter_id="matter-1"
            )
            
            # Create second matter path
            matter_path_2 = temp_matter_path.parent / "test_matter_2"
            matter_path_2.mkdir(parents=True, exist_ok=True)
            
            # Create second adapter
            adapter2 = LettaAdapter(
                matter_path=matter_path_2,
                matter_name="Matter 2", 
                matter_id="matter-2"
            )
            
            # Verify different agents were created
            assert adapter1.agent_id == "agent-matter-1"
            assert adapter2.agent_id == "agent-matter-2"
            assert adapter1.agent_id != adapter2.agent_id
            
            # Verify separate storage paths
            assert adapter1.letta_path != adapter2.letta_path
            assert adapter1.letta_path.exists()
            assert adapter2.letta_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])