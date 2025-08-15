"""
Unit tests for the PDF ingestion pipeline.

Tests OCR processing, PDF parsing, text chunking, and the complete
ingestion workflow with mocked dependencies.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import tempfile
import json
import uuid
from typing import List

# Import modules under test
from app.ocr import OCRProcessor, OCRResult, PDFProcessingError
from app.parsing import PDFParser, PageContent, DocumentMetadata, PDFParsingError
from app.chunking import TextChunker, Chunk, ChunkingError
from app.ingest import IngestionPipeline, IngestionStats, IngestionJob, IngestionError
from app.models import Matter, MatterPaths


class TestOCRProcessor:
    """Test OCR processing functionality."""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCR processor for testing."""
        return OCRProcessor(timeout_seconds=30)
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Create a temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            # Write minimal PDF header
            f.write(b'%PDF-1.4\n')
            f.write(b'1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n')
            f.write(b'2 0 obj\n<<\n/Type /Pages\n/Count 0\n>>\nendobj\n')
            f.write(b'xref\n0 3\n0000000000 65535 f \n')
            f.write(b'trailer\n<<\n/Size 3\n/Root 1 0 R\n>>\n')
            f.write(b'startxref\n0\n%%EOF\n')
            pdf_path = Path(f.name)
        
        yield pdf_path
        
        # Cleanup
        if pdf_path.exists():
            pdf_path.unlink()
    
    @pytest.mark.asyncio
    async def test_check_ocr_dependencies(self, ocr_processor):
        """Test OCR dependency checking."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful OCRmyPDF check
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b'ocrmypdf 15.0.0', b'')
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            deps = await ocr_processor.check_ocr_dependencies()
            
            assert isinstance(deps, dict)
            assert 'ocrmypdf' in deps
            assert 'tesseract' in deps
            assert 'languages' in deps
    
    @pytest.mark.asyncio
    async def test_process_pdf_file_not_found(self, ocr_processor):
        """Test OCR processing with non-existent file."""
        nonexistent_pdf = Path('/nonexistent/file.pdf')
        output_path = Path('/tmp/output.pdf')
        
        result = await ocr_processor.process_pdf(nonexistent_pdf, output_path)
        
        assert not result.success
        assert result.error_message is not None
        assert 'not found' in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_process_pdf_success_mock(self, ocr_processor, sample_pdf_path):
        """Test successful OCR processing with mocked subprocess."""
        output_path = Path(tempfile.mkdtemp()) / 'output.ocr.pdf'
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful OCRmyPDF execution
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.stderr.readline.side_effect = [
                b'INFO - Processing page 1\n',
                b'INFO - OCR completed\n',
                b''  # End of stream
            ]
            mock_process.wait.return_value = None
            mock_subprocess.return_value = mock_process
            
            # Mock successful page count
            with patch.object(ocr_processor, '_get_pdf_page_count', return_value=1):
                # Create mock output file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(sample_pdf_path.read_bytes())
                
                result = await ocr_processor.process_pdf(
                    sample_pdf_path, output_path, force_ocr=False
                )
                
                assert result.success
                assert result.output_path == output_path
                assert result.pages_processed == 1
    
    def test_parse_ocr_error(self, ocr_processor):
        """Test OCR error message parsing."""
        from pathlib import Path
        test_file = Path("/tmp/test.pdf")
        
        # Test encrypted file error
        stderr = "ERROR: PDF is encrypted"
        error = ocr_processor._parse_ocr_error(stderr, test_file)
        assert "encrypted" in error.reason.lower()
        
        # Test corrupted file error
        stderr = "ERROR: PDF appears to be corrupted"
        error = ocr_processor._parse_ocr_error(stderr, test_file)
        assert "corrupted" in error.reason.lower()
        
        # Test generic error
        stderr = "ERROR: Something went wrong"
        error = ocr_processor._parse_ocr_error(stderr, test_file)
        assert "Something went wrong" in error.reason


class TestPDFParser:
    """Test PDF parsing functionality."""
    
    @pytest.fixture
    def pdf_parser(self):
        """Create PDF parser for testing."""
        return PDFParser()
    
    def test_clean_text(self, pdf_parser):
        """Test text cleaning functionality."""
        # Test whitespace normalization
        dirty_text = "This  is\t\ta   test\n\n\n\nwith\nweird\n\n  spacing"
        clean_text = pdf_parser._clean_text(dirty_text)
        assert "  " not in clean_text  # No double spaces
        assert "\n\n\n" not in clean_text  # Max 2 consecutive newlines
        
        # Test encoding fixes
        encoded_text = "Hello\u2018world\u2019 \u201ctest\u201d"
        clean_text = pdf_parser._clean_text(encoded_text)
        assert "'" in clean_text
        assert '"' in clean_text
    
    def test_handle_encoding_issues(self, pdf_parser):
        """Test encoding issue handling."""
        # Test common character replacements
        test_cases = [
            ("\ufeff", ""),  # BOM removal
            ("\u00a0", " "),  # Non-breaking space
            ("\u2018", "'"),  # Left single quote
            ("\u2019", "'"),  # Right single quote
            ("\u201c", '"'),  # Left double quote
            ("\u201d", '"'),  # Right double quote
            ("\u2013", "-"),  # En dash
            ("\u2014", "--"), # Em dash
            ("\u2026", "..."), # Ellipsis
        ]
        
        for input_char, expected in test_cases:
            result = pdf_parser._handle_encoding_issues(input_char)
            assert result == expected
    
    def test_parse_pdf_date(self, pdf_parser):
        """Test PDF date parsing."""
        # Test full date format
        date_str = "D:20230214120000+05'00"
        parsed = pdf_parser._parse_pdf_date(date_str)
        assert parsed is not None
        assert parsed.year == 2023
        assert parsed.month == 2
        assert parsed.day == 14
        
        # Test date only format
        date_str = "D:20230214"
        parsed = pdf_parser._parse_pdf_date(date_str)
        assert parsed is not None
        assert parsed.year == 2023
        
        # Test invalid format
        date_str = "invalid_date"
        parsed = pdf_parser._parse_pdf_date(date_str)
        assert parsed is None
    
    @pytest.mark.asyncio
    async def test_save_and_load_parsed_content(self, pdf_parser):
        """Test saving and loading parsed content."""
        # Create sample page content
        pages = [
            PageContent(
                page_no=1,
                text="This is page 1 content",
                doc_name="test.pdf",
                doc_id="test-doc-id",
                md5="hash1",
                char_count=23,
                has_images=False,
                has_tables=False,
                metadata={"page_width": 612, "page_height": 792}
            ),
            PageContent(
                page_no=2,
                text="This is page 2 content",
                doc_name="test.pdf",
                doc_id="test-doc-id",
                md5="hash2",
                char_count=23,
                has_images=True,
                has_tables=True,
                metadata={"page_width": 612, "page_height": 792}
            )
        ]
        
        metadata = DocumentMetadata(
            title="Test Document",
            author="Test Author",
            page_count=2,
            creation_date=datetime.now()
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "parsed.jsonl"
            
            # Save content
            pdf_parser.save_parsed_content(pages, output_path, metadata)
            assert output_path.exists()
            
            # Load content back
            loaded_pages, loaded_metadata = pdf_parser.load_parsed_content(output_path)
            
            assert len(loaded_pages) == 2
            assert loaded_pages[0].page_no == 1
            assert loaded_pages[0].text == "This is page 1 content"
            assert loaded_pages[1].has_images == True
            
            assert loaded_metadata is not None
            assert loaded_metadata.title == "Test Document"
            assert loaded_metadata.author == "Test Author"


class TestTextChunker:
    """Test text chunking functionality."""
    
    @pytest.fixture
    def text_chunker(self):
        """Create text chunker for testing."""
        return TextChunker(target_tokens=100, overlap_percent=0.15)
    
    @pytest.fixture
    def sample_pages(self):
        """Create sample page content for testing."""
        return [
            PageContent(
                page_no=1,
                text="This is the first page with some content. " * 10,
                doc_name="test.pdf",
                doc_id="test-doc",
                md5="hash1",
                char_count=430,
                has_images=False,
                has_tables=False,
                metadata={}
            ),
            PageContent(
                page_no=2,
                text="This is the second page with different content. " * 10,
                doc_name="test.pdf", 
                doc_id="test-doc",
                md5="hash2",
                char_count=480,
                has_images=False,
                has_tables=False,
                metadata={}
            )
        ]
    
    def test_split_into_sentences(self, text_chunker):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = text_chunker._split_into_sentences(text)
        
        assert len(sentences) == 4
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"
        assert sentences[3] == "Fourth sentence."
    
    def test_chunk_document(self, text_chunker, sample_pages):
        """Test document chunking."""
        chunks = text_chunker.chunk_document(sample_pages, detect_structure=False)
        
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.chunk_id is not None
            assert chunk.text.strip() != ""
            assert chunk.page_start >= 1
            assert chunk.page_end >= chunk.page_start
            assert chunk.token_count > 0
            assert chunk.char_count > 0
            assert chunk.md5 is not None
    
    def test_deduplicate_chunks(self, text_chunker):
        """Test chunk deduplication."""
        # Create duplicate chunks
        chunk1 = Chunk(
            chunk_id="1",
            text="Same content",
            doc_id="doc1",
            doc_name="test.pdf",
            page_start=1,
            page_end=1,
            token_count=10,
            char_count=12,
            md5="same_hash"
        )
        
        chunk2 = Chunk(
            chunk_id="2",
            text="Same content",  # Same content, same hash
            doc_id="doc1",
            doc_name="test.pdf",
            page_start=2,
            page_end=2,
            token_count=10,
            char_count=12,
            md5="same_hash"  # Duplicate hash
        )
        
        chunk3 = Chunk(
            chunk_id="3",
            text="Different content",
            doc_id="doc1",
            doc_name="test.pdf",
            page_start=3,
            page_end=3,
            token_count=15,
            char_count=17,
            md5="different_hash"
        )
        
        chunks = [chunk1, chunk2, chunk3]
        unique_chunks = text_chunker._deduplicate_chunks(chunks)
        
        # Should have 2 unique chunks (chunk2 removed as duplicate)
        assert len(unique_chunks) == 2
        assert any(c.chunk_id == "1" for c in unique_chunks)
        assert any(c.chunk_id == "3" for c in unique_chunks)
        assert not any(c.chunk_id == "2" for c in unique_chunks)
    
    def test_analyze_chunks(self, text_chunker, sample_pages):
        """Test chunk analysis."""
        chunks = text_chunker.chunk_document(sample_pages)
        analysis = text_chunker.analyze_chunks(chunks)
        
        assert isinstance(analysis, dict)
        assert "total_chunks" in analysis
        assert "avg_tokens" in analysis
        assert "min_tokens" in analysis
        assert "max_tokens" in analysis
        assert "avg_chars" in analysis
        assert "sections" in analysis
        assert "page_coverage" in analysis
        
        assert analysis["total_chunks"] == len(chunks)
        assert analysis["avg_tokens"] > 0
    
    @pytest.mark.asyncio
    async def test_save_chunks_jsonl(self, text_chunker, sample_pages):
        """Test saving chunks to JSONL."""
        chunks = text_chunker.chunk_document(sample_pages)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chunks.jsonl"
            text_chunker.save_chunks_jsonl(chunks, output_path)
            
            assert output_path.exists()
            
            # Verify content
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(chunks)
                
                # Test first chunk
                first_chunk_data = json.loads(lines[0])
                assert "chunk_id" in first_chunk_data
                assert "text" in first_chunk_data
                assert "page_start" in first_chunk_data


class TestIngestionPipeline:
    """Test complete ingestion pipeline."""
    
    @pytest.fixture
    def sample_matter(self):
        """Create sample matter for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            paths = MatterPaths.from_root(temp_path)
            
            matter = Matter(
                id="test-matter-id",
                name="Test Matter",
                slug="test-matter",
                created_at=datetime.now(),
                embedding_model="test-embed",
                generation_model="test-gen",
                paths=paths
            )
            
            yield matter
    
    @pytest.fixture
    def ingestion_pipeline(self, sample_matter):
        """Create ingestion pipeline for testing."""
        return IngestionPipeline(sample_matter, {"ocr_timeout_seconds": 30})
    
    @pytest.fixture
    def mock_pdf_file(self):
        """Create mock PDF file."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b'%PDF-1.4\nMock PDF content\n%%EOF\n')
            pdf_path = Path(f.name)
        
        yield pdf_path
        
        if pdf_path.exists():
            pdf_path.unlink()
    
    def test_ensure_directories(self, ingestion_pipeline):
        """Test directory creation."""
        ingestion_pipeline._ensure_directories()
        
        # Check that all required directories exist
        assert ingestion_pipeline.matter.paths.docs.exists()
        assert ingestion_pipeline.matter.paths.docs_ocr.exists()
        assert ingestion_pipeline.matter.paths.parsed.exists()
        assert ingestion_pipeline.matter.paths.vectors.exists()
        assert ingestion_pipeline.matter.paths.knowledge.exists()
        assert ingestion_pipeline.matter.paths.chat.exists()
        assert ingestion_pipeline.matter.paths.logs.exists()
    
    @pytest.mark.asyncio
    async def test_get_ingestion_capabilities(self, ingestion_pipeline):
        """Test capability checking."""
        with patch.object(ingestion_pipeline.ocr_processor, 'check_ocr_dependencies') as mock_deps:
            mock_deps.return_value = {
                "ocrmypdf": True,
                "tesseract": True,
                "languages": ["eng", "spa"]
            }
            
            capabilities = await ingestion_pipeline.get_ingestion_capabilities()
            
            assert isinstance(capabilities, dict)
            assert "ocr" in capabilities
            assert "parsing" in capabilities
            assert "chunking" in capabilities
            assert "matter" in capabilities
            
            assert capabilities["ocr"]["available"] == True
            assert capabilities["parsing"]["available"] == True
            assert capabilities["chunking"]["available"] == True
    
    @pytest.mark.asyncio
    async def test_process_single_pdf_file_not_found(self, ingestion_pipeline):
        """Test processing non-existent PDF file."""
        nonexistent_pdf = Path("/nonexistent/file.pdf")
        
        with pytest.raises(IngestionError) as exc_info:
            await ingestion_pipeline._process_single_pdf(nonexistent_pdf)
        
        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.stage == "validation"
        assert not exc_info.value.recoverable
    
    @pytest.mark.asyncio 
    async def test_process_single_pdf_success_mocked(self, ingestion_pipeline, mock_pdf_file):
        """Test successful PDF processing with mocked components."""
        # Mock OCR result
        mock_ocr_result = OCRResult(
            success=True,
            output_path=mock_pdf_file,
            pages_processed=1,
            ocr_applied=False,
            text_pages_found=1
        )
        
        # Mock page content
        mock_pages = [
            PageContent(
                page_no=1,
                text="Sample PDF content for testing",
                doc_name=mock_pdf_file.name,
                doc_id="mock-doc-id",
                md5="mock-hash",
                char_count=32,
                has_images=False,
                has_tables=False
            )
        ]
        
        # Mock document metadata
        mock_metadata = DocumentMetadata(
            title="Mock Document",
            page_count=1
        )
        
        # Mock chunks
        mock_chunks = [
            Chunk(
                chunk_id="chunk-1",
                text="Sample PDF content for testing",
                doc_id="mock-doc-id",
                doc_name=mock_pdf_file.name,
                page_start=1,
                page_end=1,
                token_count=8,
                char_count=32,
                md5="chunk-hash"
            )
        ]
        
        # Apply mocks
        with patch.object(ingestion_pipeline.ocr_processor, 'process_pdf', return_value=mock_ocr_result), \
             patch.object(ingestion_pipeline.pdf_parser, 'get_document_metadata', return_value=mock_metadata), \
             patch.object(ingestion_pipeline.pdf_parser, 'extract_pages', return_value=mock_pages), \
             patch.object(ingestion_pipeline.pdf_parser, 'save_parsed_content'), \
             patch.object(ingestion_pipeline.text_chunker, 'chunk_document', return_value=mock_chunks), \
             patch.object(ingestion_pipeline.text_chunker, 'save_chunks_jsonl'):
            
            stats = await ingestion_pipeline._process_single_pdf(mock_pdf_file)
            
            assert stats.success
            assert stats.doc_name == mock_pdf_file.name
            assert stats.total_pages == 1
            assert stats.total_chunks == 1
            assert stats.ocr_status == "none"
            assert stats.pages_with_text == 1
            assert stats.avg_chunk_tokens == 8.0
    
    @pytest.mark.asyncio
    async def test_ingest_pdfs_empty_list(self, ingestion_pipeline):
        """Test ingesting empty PDF list."""
        results = await ingestion_pipeline.ingest_pdfs([])
        assert results == {}
    
    @pytest.mark.asyncio
    async def test_ingest_pdfs_with_mocked_processing(self, ingestion_pipeline, mock_pdf_file):
        """Test batch PDF ingestion with mocked processing."""
        mock_stats = IngestionStats(
            doc_name=mock_pdf_file.name,
            doc_id="mock-doc-id",
            total_pages=1,
            total_chunks=1,
            ocr_time_seconds=1.0,
            parse_time_seconds=0.5,
            chunk_time_seconds=0.2,
            total_time_seconds=1.7,
            ocr_status="none",
            file_size_bytes=1024,
            pages_with_text=1,
            pages_with_images=0,
            avg_chunk_tokens=8.0,
            success=True
        )
        
        with patch.object(ingestion_pipeline, '_process_single_pdf', return_value=mock_stats):
            results = await ingestion_pipeline.ingest_pdfs([mock_pdf_file])
            
            assert len(results) == 1
            assert str(mock_pdf_file) in results
            assert results[str(mock_pdf_file)].success
            assert results[str(mock_pdf_file)].total_pages == 1


class TestIngestionJob:
    """Test ingestion job data structure."""
    
    def test_ingestion_job_creation(self):
        """Test creating an ingestion job."""
        # Create mock matter
        paths = MatterPaths.from_root(Path("/tmp/test"))
        matter = Matter(
            id="test-id",
            name="Test Matter",
            slug="test-matter",
            created_at=datetime.now(),
            embedding_model="test-embed",
            generation_model="test-gen",
            paths=paths
        )
        
        job = IngestionJob(
            job_id="job-123",
            matter=matter,
            pdf_files=[Path("/test/file.pdf")],
            force_ocr=True,
            ocr_language="spa"
        )
        
        assert job.job_id == "job-123"
        assert job.matter.id == "test-id"
        assert len(job.pdf_files) == 1
        assert job.force_ocr == True
        assert job.ocr_language == "spa"
        assert job.created_at is not None


# Integration test fixtures and markers
pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])