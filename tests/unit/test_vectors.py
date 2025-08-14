"""
Unit tests for vector storage operations.

Tests VectorStore functionality including ChromaDB operations,
embedding integration, and Matter isolation.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from app.vectors import VectorStore, SearchResult, VectorStoreError
from app.chunking import Chunk
from app.models import Matter, MatterPaths


@pytest.fixture
def temp_matter_path():
    """Create temporary Matter directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    matter_path = temp_dir / "Matter_test_matter"
    matter_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (matter_path / "vectors" / "chroma").mkdir(parents=True, exist_ok=True)
    
    yield matter_path
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk-1",
            text="This is the first test chunk about construction delays.",
            doc_id="doc-123",
            doc_name="test_document.pdf",
            page_start=1,
            page_end=1,
            token_count=50,
            char_count=200,
            md5="hash1",
            section_title="Section 1",
            chunk_index=0,
            overlap_info={"has_overlap": False, "overlap_sentences": 0},
            metadata={"test_key": "test_value"}
        ),
        Chunk(
            chunk_id="chunk-2",
            text="This is the second test chunk about material specifications.",
            doc_id="doc-123",
            doc_name="test_document.pdf",
            page_start=2,
            page_end=2,
            token_count=55,
            char_count=220,
            md5="hash2",
            section_title="Section 2",
            chunk_index=1,
            overlap_info={"has_overlap": True, "overlap_sentences": 2},
            metadata={"test_key": "test_value"}
        ),
        Chunk(
            chunk_id="chunk-3",
            text="This is the third test chunk from a different document.",
            doc_id="doc-456",
            doc_name="another_document.pdf",
            page_start=1,
            page_end=1,
            token_count=45,
            char_count=180,
            md5="hash3",
            section_title=None,
            chunk_index=0,
            overlap_info={"has_overlap": False, "overlap_sentences": 0},
            metadata={}
        )
    ]


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager for testing."""
    with patch('app.vectors.embedding_manager') as mock:
        # Mock embedding generation - returns correct number based on input
        def mock_embed_fn(texts):
            return [[0.1 + i*0.1, 0.2 + i*0.1, 0.3 + i*0.1] * 256 for i in range(len(texts))]
        
        mock.embed = AsyncMock(side_effect=mock_embed_fn)
        mock.embed_single = AsyncMock(return_value=[0.15, 0.25, 0.35] * 256)
        yield mock


class TestVectorStore:
    """Test VectorStore functionality."""
    
    def test_init_creates_collection(self, temp_matter_path):
        """Test VectorStore initialization creates ChromaDB collection."""
        vector_store = VectorStore(temp_matter_path)
        
        assert vector_store.matter_path == temp_matter_path
        assert vector_store.collection_name.startswith("matter_")
        assert vector_store.client is not None
        assert vector_store.collection is not None
        assert vector_store.collection.count() == 0
    
    def test_init_with_custom_collection_name(self, temp_matter_path):
        """Test VectorStore initialization with custom collection name."""
        custom_name = "custom_test_collection"
        vector_store = VectorStore(temp_matter_path, collection_name=custom_name)
        
        assert vector_store.collection_name == custom_name
    
    def test_collection_name_sanitization(self, temp_matter_path):
        """Test collection name sanitization for ChromaDB compatibility."""
        # Test with special characters
        matter_path = temp_matter_path.parent / "Matter_test-matter-2024"
        matter_path.mkdir(parents=True, exist_ok=True)
        (matter_path / "vectors" / "chroma").mkdir(parents=True, exist_ok=True)
        
        vector_store = VectorStore(matter_path)
        
        # Should sanitize special characters but keep valid ones
        assert vector_store.collection_name.startswith("matter_")
        # Should only contain valid characters (letters, numbers, hyphens, underscores)
        import re
        assert re.match(r'^[a-zA-Z0-9_-]+$', vector_store.collection_name)
    
    @pytest.mark.asyncio
    async def test_upsert_chunks_success(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test successful chunk upserting."""
        vector_store = VectorStore(temp_matter_path)
        
        await vector_store.upsert_chunks(sample_chunks)
        
        # Verify chunks were added
        assert vector_store.collection.count() == 3
        
        # Verify embedding manager was called
        mock_embedding_manager.embed.assert_called_once()
        call_args = mock_embedding_manager.embed.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[0] == sample_chunks[0].text
    
    @pytest.mark.asyncio
    async def test_upsert_chunks_deduplication(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test chunk deduplication on repeated upserts."""
        vector_store = VectorStore(temp_matter_path)
        
        # First upsert
        await vector_store.upsert_chunks(sample_chunks)
        initial_count = vector_store.collection.count()
        
        # Second upsert with same chunks should not add duplicates
        await vector_store.upsert_chunks(sample_chunks)
        final_count = vector_store.collection.count()
        
        assert initial_count == final_count == 3
        
        # Embedding should only be called once (for first upsert)
        assert mock_embedding_manager.embed.call_count == 1
    
    @pytest.mark.asyncio
    async def test_upsert_chunks_empty_list(self, temp_matter_path, mock_embedding_manager):
        """Test upserting empty chunk list."""
        vector_store = VectorStore(temp_matter_path)
        
        await vector_store.upsert_chunks([])
        
        assert vector_store.collection.count() == 0
        mock_embedding_manager.embed.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_upsert_chunks_embedding_failure(self, temp_matter_path, sample_chunks):
        """Test handling of embedding generation failure."""
        vector_store = VectorStore(temp_matter_path)
        
        with patch('app.vectors.embedding_manager') as mock:
            mock.embed = AsyncMock(side_effect=Exception("Embedding failed"))
            
            with pytest.raises(VectorStoreError) as exc_info:
                await vector_store.upsert_chunks(sample_chunks)
            
            assert "Failed to upsert chunks" in str(exc_info.value)
            assert exc_info.value.recoverable is True
    
    @pytest.mark.asyncio
    async def test_search_success(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test successful vector search."""
        vector_store = VectorStore(temp_matter_path)
        
        # Add chunks first
        await vector_store.upsert_chunks(sample_chunks)
        
        # Search for similar content
        query = "construction delays and material issues"
        results = await vector_store.search(query, k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        
        if results:
            result = results[0]
            assert result.chunk_id
            assert result.doc_name
            assert result.page_start >= 1
            assert result.page_end >= result.page_start
            assert result.text
            assert 0.0 <= result.similarity_score <= 1.0
            assert isinstance(result.metadata, dict)
        
        # Verify query embedding was generated
        mock_embedding_manager.embed_single.assert_called_once_with(query)
    
    @pytest.mark.asyncio
    async def test_search_empty_query(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test search with empty query."""
        vector_store = VectorStore(temp_matter_path)
        
        results = await vector_store.search("", k=5)
        assert results == []
        
        results = await vector_store.search("   ", k=5)
        assert results == []
        
        mock_embedding_manager.embed_single.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_search_no_results(self, temp_matter_path, mock_embedding_manager):
        """Test search when no chunks exist."""
        vector_store = VectorStore(temp_matter_path)
        
        results = await vector_store.search("test query", k=5)
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_with_metadata_filters(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test search with metadata filtering."""
        vector_store = VectorStore(temp_matter_path)
        
        # Add chunks first
        await vector_store.upsert_chunks(sample_chunks)
        
        # Search with document filter
        results = await vector_store.search(
            "test query",
            k=10,
            filter_metadata={"doc_name": "test_document.pdf"}
        )
        
        # Should only return chunks from the filtered document
        for result in results:
            assert result.doc_name == "test_document.pdf"
    
    @pytest.mark.asyncio
    async def test_search_embedding_failure(self, temp_matter_path, sample_chunks):
        """Test handling of query embedding failure."""
        vector_store = VectorStore(temp_matter_path)
        
        with patch('app.vectors.embedding_manager') as mock:
            mock.embed_single = AsyncMock(side_effect=Exception("Embedding failed"))
            
            with pytest.raises(VectorStoreError) as exc_info:
                await vector_store.search("test query")
            
            assert "Vector search failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_empty(self, temp_matter_path):
        """Test getting statistics from empty collection."""
        vector_store = VectorStore(temp_matter_path)
        
        stats = await vector_store.get_collection_stats()
        
        assert stats["total_chunks"] == 0
        assert stats["collection_name"] == vector_store.collection_name
        assert "collection_path" in stats
        assert "matter_path" in stats
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_with_data(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test getting statistics from populated collection."""
        vector_store = VectorStore(temp_matter_path)
        
        # Add chunks first
        await vector_store.upsert_chunks(sample_chunks)
        
        stats = await vector_store.get_collection_stats()
        
        assert stats["total_chunks"] == 3
        assert stats["unique_documents"] == 2  # Two different documents
        assert "test_document.pdf" in stats["document_names"]
        assert "another_document.pdf" in stats["document_names"]
        assert stats["unique_sections"] == 2  # Two named sections
        assert stats["page_range"]["min_page"] == 1
        assert stats["page_range"]["max_page"] == 2
        assert stats["token_stats"]["avg_tokens"] > 0
    
    @pytest.mark.asyncio
    async def test_delete_chunks(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test chunk deletion."""
        vector_store = VectorStore(temp_matter_path)
        
        # Add chunks first
        await vector_store.upsert_chunks(sample_chunks)
        initial_count = vector_store.collection.count()
        
        # Delete one chunk by vector ID (using MD5)
        deleted_count = await vector_store.delete_chunks([f"chunk_{sample_chunks[0].md5}"])
        
        assert deleted_count == 1
        assert vector_store.collection.count() == initial_count - 1
    
    @pytest.mark.asyncio
    async def test_reset_collection(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test collection reset."""
        vector_store = VectorStore(temp_matter_path)
        
        # Add chunks first
        await vector_store.upsert_chunks(sample_chunks)
        assert vector_store.collection.count() > 0
        
        # Reset collection
        await vector_store.reset_collection()
        
        assert vector_store.collection.count() == 0
        assert vector_store.collection.name == vector_store.collection_name


class TestMatterIsolation:
    """Test Matter isolation in vector storage."""
    
    @pytest.mark.asyncio
    async def test_matter_isolation(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test that different matters have isolated collections."""
        # Create two different matter paths
        matter1_path = temp_matter_path
        matter2_path = temp_matter_path.parent / "Matter_another_matter"
        matter2_path.mkdir(parents=True, exist_ok=True)
        (matter2_path / "vectors" / "chroma").mkdir(parents=True, exist_ok=True)
        
        try:
            # Create vector stores for each matter
            vector_store1 = VectorStore(matter1_path)
            vector_store2 = VectorStore(matter2_path)
            
            # Verify different collection names
            assert vector_store1.collection_name != vector_store2.collection_name
            
            # Add chunks to first matter
            await vector_store1.upsert_chunks(sample_chunks[:2])
            assert vector_store1.collection.count() == 2
            assert vector_store2.collection.count() == 0
            
            # Add different chunks to second matter
            await vector_store2.upsert_chunks(sample_chunks[2:])
            assert vector_store1.collection.count() == 2
            assert vector_store2.collection.count() == 1
            
            # Search in first matter should not find chunks from second matter
            results1 = await vector_store1.search("test query", k=10)
            results2 = await vector_store2.search("test query", k=10)
            
            # Results should be isolated
            for result in results1:
                assert result.doc_name in ["test_document.pdf"]
            
            for result in results2:
                assert result.doc_name in ["another_document.pdf"]
            
        finally:
            # Cleanup
            shutil.rmtree(matter2_path, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_collection_switching(self, temp_matter_path, sample_chunks, mock_embedding_manager):
        """Test switching between different collections preserves isolation."""
        # Create first vector store and add data
        vector_store1 = VectorStore(temp_matter_path)
        await vector_store1.upsert_chunks(sample_chunks[:2])
        
        # Create second vector store with same path but different collection name
        vector_store2 = VectorStore(temp_matter_path, collection_name="different_collection")
        await vector_store2.upsert_chunks(sample_chunks[2:])
        
        # Verify isolation
        assert vector_store1.collection.count() == 2
        assert vector_store2.collection.count() == 1
        
        # Create new instance pointing to first collection
        vector_store1_new = VectorStore(temp_matter_path, collection_name=vector_store1.collection_name)
        assert vector_store1_new.collection.count() == 2


class TestEmbeddingIntegration:
    """Test embedding provider integration."""
    
    @pytest.mark.asyncio
    async def test_embedding_dimension_validation(self, temp_matter_path, sample_chunks):
        """Test validation of embedding dimensions."""
        vector_store = VectorStore(temp_matter_path)
        
        with patch('app.vectors.embedding_manager') as mock:
            # Return embeddings with inconsistent dimensions
            mock.embed = AsyncMock(return_value=[
                [0.1] * 768,  # Normal embedding
                [0.2] * 512,  # Different dimension
                [0.3] * 768   # Normal embedding
            ])
            
            # Should handle gracefully (ChromaDB will normalize or reject)
            with pytest.raises(Exception):  # ChromaDB will raise an error
                await vector_store.upsert_chunks(sample_chunks)
    
    @pytest.mark.asyncio
    async def test_embedding_count_mismatch(self, temp_matter_path, sample_chunks):
        """Test handling of embedding count mismatch."""
        vector_store = VectorStore(temp_matter_path)
        
        with patch('app.vectors.embedding_manager') as mock:
            # Return fewer embeddings than expected
            mock.embed = AsyncMock(return_value=[
                [0.1] * 768,  # Only one embedding for multiple chunks
            ])
            
            with pytest.raises(VectorStoreError) as exc_info:
                await vector_store.upsert_chunks(sample_chunks)
            
            assert "Embedding count mismatch" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_invalid_embeddings(self, temp_matter_path, sample_chunks):
        """Test handling of invalid embedding format."""
        vector_store = VectorStore(temp_matter_path)
        
        with patch('app.vectors.embedding_manager') as mock:
            # Return invalid embeddings (match the number of chunks)
            mock.embed = AsyncMock(return_value=[
                None,  # Invalid embedding
                [],    # Empty embedding
                [0.1] * 768  # Valid embedding
            ])
            
            with pytest.raises(VectorStoreError) as exc_info:
                await vector_store.upsert_chunks(sample_chunks)
            
            assert "Invalid embeddings" in str(exc_info.value)


@pytest.mark.asyncio
async def test_vector_store_error_handling():
    """Test VectorStoreError exception handling."""
    # Test VectorStoreError creation
    error = VectorStoreError("Test error", recoverable=True)
    assert error.message == "Test error"
    assert error.recoverable is True
    assert str(error) == "Test error"
    
    error_non_recoverable = VectorStoreError("Fatal error", recoverable=False)
    assert error_non_recoverable.recoverable is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])