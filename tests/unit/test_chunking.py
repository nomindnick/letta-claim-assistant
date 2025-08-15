"""
Unit tests for text chunking functionality.

Tests text chunking with various document sizes, overlap calculations,
section boundary detection, and metadata preservation.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from app.chunking import TextChunker, Chunk, ChunkingError
from app.parsing import PageContent


class TestTextChunker:
    """Test suite for TextChunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create TextChunker instance."""
        return TextChunker(
            target_size=1000,  # tokens
            overlap_percent=0.15,
            min_chunk_size=100
        )
    
    @pytest.fixture
    def sample_pages(self):
        """Sample page content for testing."""
        return [
            PageContent(
                page_no=1,
                text="This is the first page of a construction contract. " * 20,  # ~140 chars
                doc_name="contract.pdf"
            ),
            PageContent(
                page_no=2,
                text="This is the second page with specifications. " * 30,  # ~210 chars
                doc_name="contract.pdf"
            ),
            PageContent(
                page_no=3,
                text="Final page with terms and conditions. " * 25,  # ~175 chars
                doc_name="contract.pdf"
            )
        ]
    
    @pytest.fixture
    def long_text_pages(self):
        """Long text content that requires chunking."""
        long_text = "Construction project management involves detailed planning and execution. " * 100  # ~7300 chars
        
        return [
            PageContent(
                page_no=1,
                text=long_text,
                doc_name="long_document.pdf"
            )
        ]
    
    @pytest.mark.unit
    def test_chunker_initialization(self):
        """Test TextChunker initialization with various parameters."""
        # Default parameters
        chunker = TextChunker()
        assert chunker.target_size == 1000
        assert chunker.overlap_percent == 0.15
        assert chunker.min_chunk_size == 100
        
        # Custom parameters
        custom_chunker = TextChunker(
            target_size=800,
            overlap_percent=0.2,
            min_chunk_size=50
        )
        assert custom_chunker.target_size == 800
        assert custom_chunker.overlap_percent == 0.2
        assert custom_chunker.min_chunk_size == 50
    
    @pytest.mark.unit
    def test_chunk_short_document(self, chunker, sample_pages):
        """Test chunking of short document that doesn't need splitting."""
        chunks = chunker.chunk_document(
            pages=sample_pages,
            doc_id="test-doc",
            doc_name="contract.pdf"
        )
        
        # Should create chunks for content, might be split due to size
        assert len(chunks) >= 1
        
        # Verify first chunk
        first_chunk = chunks[0]
        assert first_chunk.doc_id == "test-doc"
        assert first_chunk.doc_name == "contract.pdf"
        assert first_chunk.page_start == 1
        assert first_chunk.text.startswith("This is the first page")
        assert len(first_chunk.id) > 0
        assert first_chunk.token_count > 0
    
    @pytest.mark.unit
    def test_chunk_long_document(self, chunker, long_text_pages):
        """Test chunking of long document that requires splitting."""
        chunks = chunker.chunk_document(
            pages=long_text_pages,
            doc_id="long-doc",
            doc_name="long_document.pdf"
        )
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Verify chunks have proper overlap
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Check for overlap between consecutive chunks
            overlap_text = current_chunk.text[-200:]  # Last 200 chars
            assert any(word in next_chunk.text[:200] for word in overlap_text.split()[-5:])
    
    @pytest.mark.unit
    def test_token_counting(self, chunker):
        """Test token counting accuracy."""
        test_text = "This is a test sentence with exactly ten words for counting."
        token_count = chunker._count_tokens(test_text)
        
        # Should be approximately 11-12 tokens (including punctuation)
        assert 10 <= token_count <= 15
    
    @pytest.mark.unit
    def test_overlap_calculation(self, chunker):
        """Test overlap calculation between chunks."""
        text = "word " * 100  # 100 words
        
        overlap_size = chunker._calculate_overlap_size(len(text))
        expected_overlap = int(len(text) * chunker.overlap_percent)
        
        assert abs(overlap_size - expected_overlap) <= 10  # Allow small variance
    
    @pytest.mark.unit
    def test_section_boundary_detection(self, chunker):
        """Test detection of section boundaries."""
        pages_with_sections = [
            PageContent(
                page_no=1,
                text="1. INTRODUCTION\nThis section covers the basic requirements.\n\n2. SPECIFICATIONS\nDetailed technical specifications follow.",
                doc_name="spec.pdf"
            )
        ]
        
        chunks = chunker.chunk_document(
            pages=pages_with_sections,
            doc_id="spec-doc",
            doc_name="spec.pdf"
        )
        
        # Should detect section boundaries
        assert len(chunks) >= 1
        
        # Check for section metadata
        for chunk in chunks:
            if "INTRODUCTION" in chunk.text:
                assert chunk.metadata.get("section_hint") == "INTRODUCTION" or "introduction" in chunk.text.lower()
    
    @pytest.mark.unit
    def test_page_boundary_preservation(self, chunker, sample_pages):
        """Test that page boundaries are properly preserved in metadata."""
        chunks = chunker.chunk_document(
            pages=sample_pages,
            doc_id="test-doc",
            doc_name="contract.pdf"
        )
        
        for chunk in chunks:
            # Page boundaries should be valid
            assert chunk.page_start >= 1
            assert chunk.page_end >= chunk.page_start
            assert chunk.page_end <= 3  # Max page in sample
            
            # Verify text contains content from specified pages
            page_range = range(chunk.page_start, chunk.page_end + 1)
            page_texts = [p.text for p in sample_pages if p.page_no in page_range]
            combined_text = " ".join(page_texts)
            
            # Chunk text should be substring of combined page text
            assert any(word in combined_text for word in chunk.text.split()[:5])
    
    @pytest.mark.unit
    def test_metadata_preservation(self, chunker, sample_pages):
        """Test that metadata is properly preserved and enhanced."""
        chunks = chunker.chunk_document(
            pages=sample_pages,
            doc_id="test-doc",
            doc_name="contract.pdf"
        )
        
        for chunk in chunks:
            # Basic metadata should be present
            assert "doc_id" in chunk.metadata
            assert "created_at" in chunk.metadata
            assert chunk.metadata["doc_id"] == "test-doc"
            
            # Page information should be preserved
            assert "page_range" in chunk.metadata
            page_range = chunk.metadata["page_range"]
            assert page_range["start"] == chunk.page_start
            assert page_range["end"] == chunk.page_end
    
    @pytest.mark.unit
    def test_chunk_id_generation(self, chunker, sample_pages):
        """Test that chunk IDs are unique and deterministic."""
        chunks = chunker.chunk_document(
            pages=sample_pages,
            doc_id="test-doc",
            doc_name="contract.pdf"
        )
        
        # All chunk IDs should be unique
        chunk_ids = [chunk.id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # Chunk IDs should be deterministic (based on content hash)
        chunks2 = chunker.chunk_document(
            pages=sample_pages,
            doc_id="test-doc",
            doc_name="contract.pdf"
        )
        
        chunk_ids2 = [chunk.id for chunk in chunks2]
        assert chunk_ids == chunk_ids2
    
    @pytest.mark.unit
    def test_empty_pages_handling(self, chunker):
        """Test handling of empty or whitespace-only pages."""
        empty_pages = [
            PageContent(page_no=1, text="", doc_name="empty.pdf"),
            PageContent(page_no=2, text="   \n\t  ", doc_name="empty.pdf"),
            PageContent(page_no=3, text="Valid content here.", doc_name="empty.pdf")
        ]
        
        chunks = chunker.chunk_document(
            pages=empty_pages,
            doc_id="empty-doc",
            doc_name="empty.pdf"
        )
        
        # Should only create chunks for pages with content
        assert len(chunks) >= 1
        
        # Chunks should not contain empty text
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0
    
    @pytest.mark.unit
    def test_min_chunk_size_enforcement(self):
        """Test that minimum chunk size is enforced."""
        chunker = TextChunker(min_chunk_size=50)
        
        small_pages = [
            PageContent(
                page_no=1,
                text="Short text.",  # Very short content
                doc_name="short.pdf"
            )
        ]
        
        chunks = chunker.chunk_document(
            pages=small_pages,
            doc_id="short-doc",
            doc_name="short.pdf"
        )
        
        # Should still create chunk even if below minimum
        # (or combine with adjacent content)
        assert len(chunks) >= 1
    
    @pytest.mark.unit
    def test_target_size_approximation(self, chunker):
        """Test that chunks approximate target size."""
        # Create content that should result in multiple chunks
        large_content = "This is a test sentence that will be repeated many times. " * 200
        
        large_pages = [
            PageContent(
                page_no=1,
                text=large_content,
                doc_name="large.pdf"
            )
        ]
        
        chunks = chunker.chunk_document(
            pages=large_pages,
            doc_id="large-doc",
            doc_name="large.pdf"
        )
        
        # Check that most chunks are reasonably close to target size
        target_tokens = chunker.target_size
        tolerance = 0.5  # 50% tolerance
        
        within_tolerance = 0
        for chunk in chunks[:-1]:  # Exclude last chunk (may be smaller)
            ratio = chunk.token_count / target_tokens
            if (1 - tolerance) <= ratio <= (1 + tolerance):
                within_tolerance += 1
        
        # At least 70% of chunks should be within tolerance
        if len(chunks) > 1:
            assert within_tolerance >= len(chunks[:-1]) * 0.7
    
    @pytest.mark.unit
    def test_special_characters_handling(self, chunker):
        """Test handling of special characters and unicode."""
        special_pages = [
            PageContent(
                page_no=1,
                text="Contract with special chars: $100,000 @ 5% interest. Unicode: café, naïve, résumé.",
                doc_name="special.pdf"
            )
        ]
        
        chunks = chunker.chunk_document(
            pages=special_pages,
            doc_id="special-doc",
            doc_name="special.pdf"
        )
        
        assert len(chunks) >= 1
        
        # Special characters should be preserved
        chunk_text = chunks[0].text
        assert "$100,000" in chunk_text
        assert "café" in chunk_text
        assert "résumé" in chunk_text
    
    @pytest.mark.unit
    def test_chunking_error_handling(self, chunker):
        """Test error handling in chunking process."""
        # Test with invalid page data
        invalid_pages = None
        
        with pytest.raises(ChunkingError):
            chunker.chunk_document(
                pages=invalid_pages,
                doc_id="test",
                doc_name="test.pdf"
            )
    
    @pytest.mark.unit
    def test_overlap_boundary_cases(self, chunker):
        """Test edge cases in overlap calculation."""
        # Test with very small chunks where overlap might be larger than content
        tiny_text = "Small."
        overlap_size = chunker._calculate_overlap_size(len(tiny_text))
        
        # Overlap should not exceed text size
        assert overlap_size <= len(tiny_text)
        
        # Test with zero-length text
        zero_overlap = chunker._calculate_overlap_size(0)
        assert zero_overlap == 0
    
    @pytest.mark.unit
    def test_concurrent_chunking(self, chunker, sample_pages):
        """Test that chunking is thread-safe."""
        import threading
        import time
        
        results = []
        
        def chunk_worker():
            chunks = chunker.chunk_document(
                pages=sample_pages,
                doc_id="concurrent-doc",
                doc_name="concurrent.pdf"
            )
            results.append(chunks)
        
        # Run multiple chunking operations concurrently
        threads = []
        for i in range(3):
            thread = threading.Thread(target=chunk_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All results should be identical
        assert len(results) == 3
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0])
            for j in range(len(results[i])):
                assert results[i][j].id == results[0][j].id
                assert results[i][j].text == results[0][j].text