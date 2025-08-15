"""
Unit tests for PDF parsing functionality.

Tests PDF text extraction, page boundary handling, document metadata
extraction, and handling of corrupted PDF scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import io

from app.parsing import PDFParser, PageContent, DocumentMetadata, PDFParsingError


class TestPDFParser:
    """Test suite for PDFParser."""
    
    @pytest.fixture
    def parser(self):
        """Create PDFParser instance."""
        return PDFParser()
    
    @pytest.fixture
    def mock_fitz_document(self):
        """Mock PyMuPDF document."""
        mock_doc = Mock()
        mock_doc.page_count = 3
        mock_doc.metadata = {
            'title': 'Test Construction Contract',
            'author': 'Legal Department',
            'subject': 'Contract Terms',
            'creator': 'PDF Writer',
            'producer': 'Test Producer',
            'creationDate': 'D:20230115120000+00\'00\'',
            'modDate': 'D:20230116120000+00\'00\''
        }
        
        # Mock pages
        mock_pages = []
        for i in range(3):
            mock_page = Mock()
            mock_page.number = i
            mock_page.get_text.return_value = f"Page {i+1} content with construction contract terms."
            mock_page.get_textpage.return_value = Mock()
            mock_page.get_text_blocks.return_value = [
                (0, 0, 100, 20, f"Page {i+1} content", 0, 0),
                (0, 25, 100, 45, "with construction contract terms.", 1, 0)
            ]
            mock_pages.append(mock_page)
        
        mock_doc.__iter__ = lambda self: iter(mock_pages)
        mock_doc.__getitem__ = lambda self, idx: mock_pages[idx]
        
        return mock_doc
    
    @pytest.fixture
    def temp_pdf_file(self):
        """Create temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            # Write minimal PDF content
            pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            tmp_file.write(pdf_content)
            tmp_file.flush()
            yield Path(tmp_file.name)
        
        # Cleanup
        Path(tmp_file.name).unlink(missing_ok=True)
    
    @pytest.mark.unit
    def test_parser_initialization(self):
        """Test PDFParser initialization."""
        parser = PDFParser()
        assert parser is not None
        assert hasattr(parser, 'extract_pages')
        assert hasattr(parser, 'get_document_metadata')
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_extract_pages_success(self, mock_fitz_open, parser, mock_fitz_document):
        """Test successful page extraction."""
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_fitz_document)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        test_path = Path("/tmp/test.pdf")
        pages = parser.extract_pages(test_path)
        
        assert len(pages) == 3
        
        # Verify first page
        first_page = pages[0]
        assert isinstance(first_page, PageContent)
        assert first_page.page_no == 1
        assert first_page.doc_name == "test.pdf"
        assert "Page 1 content" in first_page.text
        
        # Verify all pages have content
        for i, page in enumerate(pages):
            assert page.page_no == i + 1
            assert f"Page {i+1} content" in page.text
            assert "construction contract terms" in page.text
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_extract_pages_empty_document(self, mock_fitz_open, parser):
        """Test extraction from empty document."""
        mock_doc = Mock()
        mock_doc.page_count = 0
        mock_doc.__iter__ = lambda self: iter([])
        
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_doc)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        test_path = Path("/tmp/empty.pdf")
        pages = parser.extract_pages(test_path)
        
        assert len(pages) == 0
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_extract_pages_with_empty_pages(self, mock_fitz_open, parser):
        """Test extraction with some empty pages."""
        mock_doc = Mock()
        mock_doc.page_count = 3
        
        mock_pages = []
        for i in range(3):
            mock_page = Mock()
            mock_page.number = i
            # Make middle page empty
            if i == 1:
                mock_page.get_text.return_value = ""
            else:
                mock_page.get_text.return_value = f"Page {i+1} has content."
            mock_pages.append(mock_page)
        
        mock_doc.__iter__ = lambda self: iter(mock_pages)
        
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_doc)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        test_path = Path("/tmp/sparse.pdf")
        pages = parser.extract_pages(test_path)
        
        assert len(pages) == 3
        assert pages[0].text == "Page 1 has content."
        assert pages[1].text == ""  # Empty page preserved
        assert pages[2].text == "Page 3 has content."
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_extract_pages_file_not_found(self, mock_fitz_open, parser):
        """Test extraction with non-existent file."""
        mock_fitz_open.side_effect = FileNotFoundError("File not found")
        
        non_existent_path = Path("/tmp/nonexistent.pdf")
        
        with pytest.raises(PDFParsingError) as exc_info:
            parser.extract_pages(non_existent_path)
        
        assert "File not found" in str(exc_info.value)
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_extract_pages_corrupted_pdf(self, mock_fitz_open, parser):
        """Test extraction with corrupted PDF."""
        mock_fitz_open.side_effect = Exception("PDF is corrupted or invalid")
        
        corrupted_path = Path("/tmp/corrupted.pdf")
        
        with pytest.raises(PDFParsingError) as exc_info:
            parser.extract_pages(corrupted_path)
        
        assert "Failed to parse PDF" in str(exc_info.value)
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_get_document_metadata_success(self, mock_fitz_open, parser, mock_fitz_document):
        """Test successful metadata extraction."""
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_fitz_document)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        test_path = Path("/tmp/test.pdf")
        metadata = parser.get_document_metadata(test_path)
        
        assert isinstance(metadata, DocumentMetadata)
        assert metadata.title == "Test Construction Contract"
        assert metadata.author == "Legal Department"
        assert metadata.subject == "Contract Terms"
        assert metadata.total_pages == 3
        assert metadata.file_size > 0  # Should have some file size
        assert metadata.creation_date is not None
        assert metadata.modification_date is not None
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_get_document_metadata_missing_fields(self, mock_fitz_open, parser):
        """Test metadata extraction with missing fields."""
        mock_doc = Mock()
        mock_doc.page_count = 5
        mock_doc.metadata = {
            'title': '',  # Empty title
            'author': None,  # None author
            # Missing other fields
        }
        
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_doc)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024000
            
            test_path = Path("/tmp/minimal.pdf")
            metadata = parser.get_document_metadata(test_path)
            
            assert metadata.title == ""
            assert metadata.author is None
            assert metadata.subject is None
            assert metadata.total_pages == 5
            assert metadata.file_size == 1024000
    
    @pytest.mark.unit
    def test_page_content_creation(self, parser):
        """Test PageContent object creation."""
        page = PageContent(
            page_no=1,
            text="Sample page text with construction terms.",
            doc_name="test.pdf"
        )
        
        assert page.page_no == 1
        assert page.text == "Sample page text with construction terms."
        assert page.doc_name == "test.pdf"
    
    @pytest.mark.unit
    def test_document_metadata_creation(self, parser):
        """Test DocumentMetadata object creation."""
        from datetime import datetime
        
        metadata = DocumentMetadata(
            title="Construction Contract",
            author="Legal Team",
            subject="Project Agreement",
            creator="PDF Generator",
            producer="System",
            total_pages=10,
            file_size=2048000,
            creation_date=datetime.now(),
            modification_date=datetime.now()
        )
        
        assert metadata.title == "Construction Contract"
        assert metadata.author == "Legal Team"
        assert metadata.total_pages == 10
        assert metadata.file_size == 2048000
        assert metadata.creation_date is not None
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_text_extraction_encoding(self, mock_fitz_open, parser):
        """Test text extraction handles various encodings."""
        mock_doc = Mock()
        mock_doc.page_count = 1
        
        # Page with unicode characters
        mock_page = Mock()
        mock_page.number = 0
        mock_page.get_text.return_value = "Contract with special chars: café, naïve, $100,000"
        
        mock_doc.__iter__ = lambda self: iter([mock_page])
        
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_doc)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        test_path = Path("/tmp/unicode.pdf")
        pages = parser.extract_pages(test_path)
        
        assert len(pages) == 1
        page_text = pages[0].text
        assert "café" in page_text
        assert "naïve" in page_text
        assert "$100,000" in page_text
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_large_document_handling(self, mock_fitz_open, parser):
        """Test handling of large documents."""
        mock_doc = Mock()
        mock_doc.page_count = 1000  # Large document
        
        # Create many mock pages
        mock_pages = []
        for i in range(1000):
            mock_page = Mock()
            mock_page.number = i
            mock_page.get_text.return_value = f"Page {i+1} content " * 100  # Long content
            mock_pages.append(mock_page)
        
        mock_doc.__iter__ = lambda self: iter(mock_pages)
        
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_doc)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        test_path = Path("/tmp/large.pdf")
        pages = parser.extract_pages(test_path)
        
        assert len(pages) == 1000
        
        # Verify first and last pages
        assert pages[0].page_no == 1
        assert pages[-1].page_no == 1000
        assert "Page 1 content" in pages[0].text
        assert "Page 1000 content" in pages[-1].text
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_text_cleaning(self, mock_fitz_open, parser):
        """Test that extracted text is properly cleaned."""
        mock_doc = Mock()
        mock_doc.page_count = 1
        
        mock_page = Mock()
        mock_page.number = 0
        # Text with extra whitespace and control characters
        raw_text = "Contract\n\n\n  terms   \t\t\twith\r\nextra    spaces"
        mock_page.get_text.return_value = raw_text
        
        mock_doc.__iter__ = lambda self: iter([mock_page])
        
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_doc)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        test_path = Path("/tmp/messy.pdf")
        pages = parser.extract_pages(test_path)
        
        # Text should be cleaned but preserve content
        cleaned_text = pages[0].text
        assert "Contract" in cleaned_text
        assert "terms" in cleaned_text
        assert "with" in cleaned_text
        assert "extra" in cleaned_text
        assert "spaces" in cleaned_text
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_password_protected_pdf(self, mock_fitz_open, parser):
        """Test handling of password-protected PDFs."""
        mock_fitz_open.side_effect = Exception("PDF requires password")
        
        protected_path = Path("/tmp/protected.pdf")
        
        with pytest.raises(PDFParsingError) as exc_info:
            parser.extract_pages(protected_path)
        
        assert "Failed to parse PDF" in str(exc_info.value)
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_memory_efficient_parsing(self, mock_fitz_open, parser):
        """Test that parsing is memory efficient for large documents."""
        mock_doc = Mock()
        mock_doc.page_count = 100
        
        # Track how many pages are accessed
        access_count = 0
        
        def create_mock_page(page_num):
            nonlocal access_count
            access_count += 1
            mock_page = Mock()
            mock_page.number = page_num
            mock_page.get_text.return_value = f"Page {page_num + 1} content"
            return mock_page
        
        mock_pages = [create_mock_page(i) for i in range(100)]
        mock_doc.__iter__ = lambda self: iter(mock_pages)
        
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_doc)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        test_path = Path("/tmp/memory_test.pdf")
        pages = parser.extract_pages(test_path)
        
        assert len(pages) == 100
        # Should have accessed all pages
        assert access_count == 100
    
    @pytest.mark.unit
    def test_error_message_clarity(self, parser):
        """Test that error messages are clear and actionable."""
        with patch('app.parsing.fitz.open') as mock_fitz_open:
            # Test file not found
            mock_fitz_open.side_effect = FileNotFoundError("No such file")
            
            with pytest.raises(PDFParsingError) as exc_info:
                parser.extract_pages(Path("/tmp/missing.pdf"))
            
            error_msg = str(exc_info.value)
            assert "File not found" in error_msg
            assert "/tmp/missing.pdf" in error_msg
    
    @pytest.mark.unit
    @patch('app.parsing.fitz.open')
    def test_concurrent_parsing(self, mock_fitz_open, parser, mock_fitz_document):
        """Test that parsing is thread-safe."""
        import threading
        import time
        
        mock_fitz_open.return_value.__enter__ = Mock(return_value=mock_fitz_document)
        mock_fitz_open.return_value.__exit__ = Mock(return_value=None)
        
        results = []
        
        def parse_worker():
            test_path = Path("/tmp/concurrent.pdf")
            pages = parser.extract_pages(test_path)
            results.append(pages)
        
        # Run multiple parsing operations concurrently
        threads = []
        for i in range(3):
            thread = threading.Thread(target=parse_worker)
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
                assert results[i][j].text == results[0][j].text
                assert results[i][j].page_no == results[0][j].page_no