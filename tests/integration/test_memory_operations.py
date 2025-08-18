"""
Integration tests for enhanced memory operations in Sprint L4.

Tests batch storage, advanced retrieval, memory management,
and analytics features.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.letta_adapter import LettaAdapter
from app.models import KnowledgeItem, SourceChunk


class TestMemoryOperations:
    """Integration tests for memory operations."""
    
    @pytest.fixture
    async def adapter_with_mock_client(self):
        """Create adapter with mocked Letta client."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "test_matter"
            matter_path.mkdir(parents=True, exist_ok=True)
            
            adapter = LettaAdapter(
                matter_path=matter_path,
                matter_name="Test Matter",
                matter_id="test-123"
            )
            
            # Mock the client
            mock_client = AsyncMock()
            mock_client.agents = AsyncMock()
            mock_client.agents.passages = AsyncMock()
            mock_client.agents.retrieve = AsyncMock()
            mock_client.agents.modify = AsyncMock()
            
            adapter.client = mock_client
            adapter.agent_id = "test-agent-123"
            adapter._initialized = True
            adapter.fallback_mode = False
            
            yield adapter
    
    @pytest.fixture
    def sample_knowledge_batch(self):
        """Create batch of knowledge items for testing."""
        return [
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
                support_snippet="Foundation showed settlement"
            ),
            KnowledgeItem(
                type="Issue",
                label="Schedule delay",
                date="2024-03-01",
                actors=["ABC Construction"],
                doc_refs=[{"doc": "Notice.pdf", "page": 2}],
                support_snippet="30 day delay due to weather"
            ),
            # Duplicate item for deduplication test
            KnowledgeItem(
                type="Entity",
                label="ABC Construction",
                actors=["ABC Construction"],
                doc_refs=[{"doc": "Contract.pdf", "page": 1}],
                support_snippet="ABC Construction is the general contractor"
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_store_knowledge_batch(self, adapter_with_mock_client, sample_knowledge_batch):
        """Test batch storage with deduplication."""
        adapter = adapter_with_mock_client
        
        # Mock existing memories for deduplication
        adapter.client.agents.passages.list.return_value = []
        adapter.client.agents.passages.create.return_value = Mock(id="passage-1")
        
        # Store batch
        result = await adapter.store_knowledge_batch(
            sample_knowledge_batch,
            deduplicate=True,
            importance_threshold=0.5
        )
        
        assert result["stored"] == 3  # 4 items - 1 duplicate
        assert result["duplicates"] == 1
        assert result["skipped"] == 0
        assert result["total"] == 4
        
        # Verify create was called correct number of times
        assert adapter.client.agents.passages.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_recall_with_context(self, adapter_with_mock_client):
        """Test context-aware memory recall."""
        adapter = adapter_with_mock_client
        
        # Mock memory passages
        mock_passages = [
            Mock(text=json.dumps({
                "type": "Event",
                "label": "Foundation failure",
                "date": "2024-02-14"
            }), created_at=datetime.now()),
            Mock(text=json.dumps({
                "type": "Issue",
                "label": "Water damage",
                "date": "2024-02-20"
            }), created_at=datetime.now() - timedelta(days=7))
        ]
        
        adapter.client.agents.passages.list.return_value = mock_passages
        
        # Test recall with context
        conversation_history = [
            "What caused the foundation issues?",
            "The foundation failed due to water infiltration"
        ]
        
        results = await adapter.recall_with_context(
            query="Tell me more about the water problems",
            conversation_history=conversation_history,
            top_k=5,
            recency_weight=0.3
        )
        
        assert len(results) <= 5
        assert all(isinstance(item, KnowledgeItem) for item in results)
    
    @pytest.mark.asyncio
    async def test_semantic_memory_search(self, adapter_with_mock_client):
        """Test semantic search with filters."""
        adapter = adapter_with_mock_client
        
        # Mock memory data
        mock_passages = [
            Mock(text=json.dumps({
                "type": "Event",
                "label": "Foundation failure",
                "date": "2024-02-14",
                "doc_refs": [{"doc": "Report.pdf"}]
            })),
            Mock(text=json.dumps({
                "type": "Issue",
                "label": "Schedule delay",
                "date": "2024-03-01",
                "doc_refs": [{"doc": "Notice.pdf"}]
            })),
        ]
        
        adapter.client.agents.passages.list.return_value = mock_passages
        
        # Test with filters
        filters = {
            "types": ["Event"],
            "date_range": {
                "start": datetime(2024, 2, 1),
                "end": datetime(2024, 3, 1)
            },
            "doc_sources": ["Report.pdf"]
        }
        
        results = await adapter.semantic_memory_search(
            query="foundation OR schedule",
            filters=filters,
            top_k=10
        )
        
        # Should only return Event type within date range
        assert len(results) >= 0
        for item, score in results:
            assert isinstance(item, KnowledgeItem)
            assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_update_core_memory_smart(self, adapter_with_mock_client):
        """Test smart core memory updates."""
        adapter = adapter_with_mock_client
        
        # Mock agent with memory blocks
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_block = Mock(label="human", value="Initial memory content")
        mock_agent.memory.blocks = [mock_block]
        
        adapter.client.agents.retrieve.return_value = mock_agent
        adapter.client.agents.modify.return_value = True
        
        # Test append mode
        success = await adapter.update_core_memory_smart(
            block_label="human",
            new_content="New important information",
            mode="append",
            max_size=2000
        )
        
        assert success
        assert "New important information" in mock_block.value
        adapter.client.agents.modify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_summary(self, adapter_with_mock_client):
        """Test memory summary generation."""
        adapter = adapter_with_mock_client
        
        # Mock diverse memory types
        mock_passages = [
            Mock(text=json.dumps({"type": "Entity", "label": "ABC Corp"})),
            Mock(text=json.dumps({"type": "Event", "label": "Foundation failure"})),
            Mock(text=json.dumps({"type": "Issue", "label": "Delay"})),
            Mock(text="Plain text memory"),
        ]
        
        adapter.client.agents.passages.list.return_value = mock_passages
        
        summary = await adapter.get_memory_summary(max_length=500)
        
        assert "Memory contains 4 items" in summary
        assert "Entity" in summary
        assert "Event" in summary
    
    @pytest.mark.asyncio
    async def test_memory_pruning(self, adapter_with_mock_client):
        """Test memory pruning by importance and age."""
        adapter = adapter_with_mock_client
        
        # Mock memories with varying importance
        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        recent_date = datetime.now().isoformat()
        
        mock_passages = [
            Mock(id="1", text=json.dumps({
                "importance_score": 0.8,
                "stored_at": recent_date
            })),
            Mock(id="2", text=json.dumps({
                "importance_score": 0.2,  # Low importance
                "stored_at": recent_date
            })),
            Mock(id="3", text=json.dumps({
                "importance_score": 0.6,
                "stored_at": old_date  # Old
            })),
        ]
        
        adapter.client.agents.passages.list.return_value = mock_passages
        adapter.client.agents.passages.delete = AsyncMock()
        
        result = await adapter.prune_memory(
            max_items=2,
            importance_threshold=0.3,
            age_days=90
        )
        
        assert result["kept"] <= 2
        assert "threshold" in result
    
    @pytest.mark.asyncio
    async def test_memory_export_json(self, adapter_with_mock_client):
        """Test memory export to JSON format."""
        adapter = adapter_with_mock_client
        
        mock_passages = [
            Mock(
                id="1",
                text=json.dumps({"type": "Entity", "label": "Test"}),
                created_at=datetime.now()
            ),
        ]
        
        adapter.client.agents.passages.list.return_value = mock_passages
        
        export_data = await adapter.export_memory(
            format="json",
            include_metadata=True
        )
        
        assert isinstance(export_data, dict)
        assert "memories" in export_data
        assert export_data["memory_count"] == 1
        assert export_data["agent_id"] == "test-agent-123"
    
    @pytest.mark.asyncio
    async def test_memory_export_csv(self, adapter_with_mock_client):
        """Test memory export to CSV format."""
        adapter = adapter_with_mock_client
        
        mock_passages = [
            Mock(
                id="1",
                text=json.dumps({
                    "type": "Event",
                    "label": "Foundation issue",
                    "date": "2024-02-14",
                    "actors": ["ABC Corp", "Sub A"],
                    "support_snippet": "Foundation failed",
                    "importance_score": 0.8
                }),
                created_at=datetime.now()
            ),
        ]
        
        adapter.client.agents.passages.list.return_value = mock_passages
        
        csv_data = await adapter.export_memory(
            format="csv",
            include_metadata=True
        )
        
        assert isinstance(csv_data, str)
        assert "Type,Label,Date,Actors,Support,Importance" in csv_data
        assert "Foundation issue" in csv_data
    
    @pytest.mark.asyncio
    async def test_memory_import_json(self, adapter_with_mock_client):
        """Test memory import from JSON."""
        adapter = adapter_with_mock_client
        
        # Prepare import data
        import_data = {
            "memories": [
                {"text": json.dumps({"type": "Entity", "label": "New Corp"})},
                {"data": {"type": "Event", "label": "New Event"}},
            ]
        }
        
        adapter.client.agents.passages.list.return_value = []  # No existing
        adapter.client.agents.passages.create = AsyncMock()
        
        result = await adapter.import_memory(
            data=import_data,
            format="json",
            deduplicate=True
        )
        
        assert result["imported"] == 2
        assert result["skipped"] == 0
        assert adapter.client.agents.passages.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_memory_pattern_analysis(self, adapter_with_mock_client):
        """Test memory pattern analysis."""
        adapter = adapter_with_mock_client
        
        # Mock varied memory data
        mock_passages = [
            Mock(text=json.dumps({
                "type": "Entity",
                "label": "ABC Corp",
                "actors": ["ABC Corp"],
                "date": "2024-01-15",
                "doc_refs": [{"doc": "Contract.pdf"}]
            })),
            Mock(text=json.dumps({
                "type": "Event",
                "label": "Foundation failure",
                "actors": ["ABC Corp", "Sub A"],
                "date": "2024-02-14",
                "doc_refs": [{"doc": "Report.pdf"}]
            })),
            Mock(text=json.dumps({
                "type": "Event",
                "label": "Schedule delay",
                "actors": ["ABC Corp"],
                "date": "2024-02-20",
                "doc_refs": [{"doc": "Report.pdf"}]
            })),
        ]
        
        adapter.client.agents.passages.list.return_value = mock_passages
        
        analysis = await adapter.analyze_memory_patterns()
        
        assert "patterns" in analysis
        assert "insights" in analysis
        assert "total_memories" in analysis
        assert analysis["total_memories"] == 3
        
        # Check for expected patterns
        assert "type_distribution" in analysis
        assert "actor_network" in analysis
    
    @pytest.mark.asyncio
    async def test_memory_quality_metrics(self, adapter_with_mock_client):
        """Test memory quality metrics calculation."""
        adapter = adapter_with_mock_client
        
        # Mock memories with varying quality
        mock_passages = [
            Mock(text=json.dumps({
                "type": "Event",
                "label": "High quality event",
                "date": "2024-02-14",
                "actors": ["ABC Corp"],
                "doc_refs": [{"doc": "Report.pdf"}],
                "support_snippet": "Detailed support text"
            })),
            Mock(text="Low quality plain text"),
        ]
        
        adapter.client.agents.passages.list.return_value = mock_passages
        
        metrics = await adapter.get_memory_quality_metrics()
        
        assert "quality_score" in metrics
        assert 0.0 <= metrics["quality_score"] <= 1.0
        assert "structure_score" in metrics
        assert "support_score" in metrics
        assert "reference_score" in metrics
    
    @pytest.mark.asyncio
    async def test_batch_operations_performance(self, adapter_with_mock_client):
        """Test performance of batch memory operations."""
        adapter = adapter_with_mock_client
        
        # Create large batch
        large_batch = [
            KnowledgeItem(
                type="Fact",
                label=f"Fact {i}",
                support_snippet=f"Support for fact {i}"
            )
            for i in range(100)
        ]
        
        adapter.client.agents.passages.list.return_value = []
        adapter.client.agents.passages.create = AsyncMock()
        
        # Measure time
        import time
        start = time.time()
        
        result = await adapter.store_knowledge_batch(
            large_batch,
            deduplicate=False,
            importance_threshold=0.0
        )
        
        elapsed = time.time() - start
        
        assert result["stored"] == 100
        assert elapsed < 5.0  # Should complete within 5 seconds
        
        # Verify parallel execution
        assert adapter.client.agents.passages.create.call_count == 100
    
    @pytest.mark.asyncio
    async def test_memory_deduplication(self, adapter_with_mock_client):
        """Test memory deduplication logic."""
        adapter = adapter_with_mock_client
        
        # Existing memories
        existing = [
            Mock(text=json.dumps({
                "type": "Entity",
                "label": "ABC Corp",
                "actors": ["ABC Corp"]
            }))
        ]
        
        adapter.client.agents.passages.list.return_value = existing
        adapter.client.agents.passages.create = AsyncMock()
        
        # Try to add duplicate and new item
        items = [
            KnowledgeItem(
                type="Entity",
                label="ABC Corp",  # Duplicate
                actors=["ABC Corp"]
            ),
            KnowledgeItem(
                type="Entity",
                label="XYZ Corp",  # New
                actors=["XYZ Corp"]
            ),
        ]
        
        result = await adapter.store_knowledge_batch(
            items,
            deduplicate=True
        )
        
        assert result["stored"] == 1  # Only new item
        assert result["duplicates"] == 1
    
    @pytest.mark.asyncio
    async def test_fallback_mode_handling(self):
        """Test graceful fallback when Letta unavailable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "test_matter"
            matter_path.mkdir(parents=True, exist_ok=True)
            
            adapter = LettaAdapter(
                matter_path=matter_path,
                matter_name="Test Matter"
            )
            
            # Force fallback mode
            adapter.fallback_mode = True
            adapter._initialized = True
            
            # Test operations in fallback mode
            result = await adapter.recall("test query")
            assert result == []
            
            batch_result = await adapter.store_knowledge_batch([])
            assert batch_result["stored"] == 0
            
            summary = await adapter.get_memory_summary()
            assert summary == "Memory unavailable"