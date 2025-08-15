"""
Unit tests for OCR processing functionality.

Tests OCR processing with mocked OCRmyPDF, validates skip-text and
force-OCR modes, tests language support, and handles timeout scenarios.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import asyncio
import subprocess
from datetime import datetime, timedelta

from app.ocr import OCRProcessor, OCRResult, OCRError, OCRTimeoutError


class TestOCRProcessor:
    """Test suite for OCRProcessor."""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCRProcessor instance."""
        return OCRProcessor(
            language="eng",
            skip_text=True,
            timeout_seconds=30
        )
    
    @pytest.fixture
    def temp_pdf_files(self):
        """Create temporary PDF files for testing."""
        input_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        output_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        
        # Write minimal PDF content to input file
        input_file.write(b"%PDF-1.4\ntest content")
        input_file.flush()
        
        try:
            yield Path(input_file.name), Path(output_file.name)
        finally:
            Path(input_file.name).unlink(missing_ok=True)
            Path(output_file.name).unlink(missing_ok=True)
    
    @pytest.mark.unit
    def test_ocr_processor_initialization(self):
        """Test OCRProcessor initialization with various parameters."""
        # Default parameters
        processor = OCRProcessor()
        assert processor.language == "eng"
        assert processor.skip_text is True
        assert processor.timeout_seconds == 120
        assert processor.force_ocr is False
        
        # Custom parameters
        custom_processor = OCRProcessor(
            language="spa",
            skip_text=False,
            force_ocr=True,
            timeout_seconds=60
        )
        assert custom_processor.language == "spa"
        assert custom_processor.skip_text is False
        assert custom_processor.force_ocr is True
        assert custom_processor.timeout_seconds == 60
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_skip_text_success(self, mock_subprocess, ocr_processor, temp_pdf_files):
        """Test successful OCR processing with skip-text mode."""
        input_path, output_path = temp_pdf_files
        
        # Mock successful subprocess
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"OCR completed successfully", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        # Mock file operations
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat.return_value.st_size = 1024000
                
                result = await ocr_processor.process_pdf(input_path, output_path)
                
                assert isinstance(result, OCRResult)
                assert result.success is True
                assert result.input_file == str(input_path)
                assert result.output_file == str(output_path)
                assert result.ocr_mode == "skip_text"
                assert result.processing_time > 0
                assert result.output_size_bytes == 1024000
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_force_ocr_success(self, mock_subprocess, temp_pdf_files):
        """Test successful OCR processing with force-OCR mode."""
        input_path, output_path = temp_pdf_files
        
        # Force OCR processor
        processor = OCRProcessor(force_ocr=True, skip_text=False)
        
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"Force OCR completed", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat.return_value.st_size = 2048000
                
                result = await processor.process_pdf(input_path, output_path)
                
                assert result.success is True
                assert result.ocr_mode == "force_ocr"
                assert result.output_size_bytes == 2048000
                
                # Verify command included --force-ocr
                mock_subprocess.assert_called_once()
                call_args = mock_subprocess.call_args[0][0]
                assert "--force-ocr" in call_args
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_with_language(self, mock_subprocess, temp_pdf_files):
        """Test OCR processing with specific language."""
        input_path, output_path = temp_pdf_files
        
        # Spanish language processor
        processor = OCRProcessor(language="spa")
        
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"Spanish OCR completed", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat.return_value.st_size = 1500000
                
                result = await processor.process_pdf(input_path, output_path)
                
                assert result.success is True
                assert result.language == "spa"
                
                # Verify command included correct language
                call_args = mock_subprocess.call_args[0][0]
                assert "--language" in call_args
                assert "spa" in call_args
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_ocr_failure(self, mock_subprocess, ocr_processor, temp_pdf_files):
        """Test OCR processing failure handling."""
        input_path, output_path = temp_pdf_files
        
        # Mock failed subprocess
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"OCR failed: corrupted PDF")
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process
        
        result = await ocr_processor.process_pdf(input_path, output_path)
        
        assert result.success is False
        assert "OCR processing failed" in result.error_message
        assert "corrupted PDF" in result.error_message
        assert result.processing_time > 0
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_timeout(self, mock_subprocess, temp_pdf_files):
        """Test OCR processing timeout handling."""
        input_path, output_path = temp_pdf_files
        
        # Short timeout processor
        processor = OCRProcessor(timeout_seconds=1)
        
        # Mock subprocess that hangs
        mock_process = AsyncMock()
        mock_process.communicate.side_effect = asyncio.TimeoutError()
        mock_subprocess.return_value = mock_process
        
        with pytest.raises(OCRTimeoutError) as exc_info:
            await processor.process_pdf(input_path, output_path)
        
        assert "OCR processing timed out" in str(exc_info.value)
        assert "1 seconds" in str(exc_info.value)
    
    @pytest.mark.unit
    async def test_process_pdf_input_file_not_found(self, ocr_processor):
        """Test OCR processing with non-existent input file."""
        input_path = Path("/tmp/nonexistent.pdf")
        output_path = Path("/tmp/output.pdf")
        
        with pytest.raises(OCRError) as exc_info:
            await ocr_processor.process_pdf(input_path, output_path)
        
        assert "Input file not found" in str(exc_info.value)
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_output_not_created(self, mock_subprocess, ocr_processor, temp_pdf_files):
        """Test OCR processing when output file is not created."""
        input_path, output_path = temp_pdf_files
        
        # Mock successful subprocess but output file doesn't exist
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"OCR completed", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with patch('pathlib.Path.exists', side_effect=lambda p: p == input_path):
            result = await ocr_processor.process_pdf(input_path, output_path)
            
            assert result.success is False
            assert "Output file was not created" in result.error_message
    
    @pytest.mark.unit
    def test_build_ocr_command_skip_text(self, ocr_processor):
        """Test OCR command building for skip-text mode."""
        input_path = Path("/tmp/input.pdf")
        output_path = Path("/tmp/output.pdf")
        
        command = ocr_processor._build_ocr_command(input_path, output_path)
        
        expected_command = [
            "ocrmypdf",
            "--skip-text",
            "--language", "eng",
            "--output-type", "pdf",
            str(input_path),
            str(output_path)
        ]
        
        assert command == expected_command
    
    @pytest.mark.unit
    def test_build_ocr_command_force_ocr(self):
        """Test OCR command building for force-OCR mode."""
        processor = OCRProcessor(force_ocr=True, skip_text=False, language="fra")
        input_path = Path("/tmp/input.pdf")
        output_path = Path("/tmp/output.pdf")
        
        command = processor._build_ocr_command(input_path, output_path)
        
        expected_command = [
            "ocrmypdf",
            "--force-ocr",
            "--language", "fra",
            "--output-type", "pdf",
            str(input_path),
            str(output_path)
        ]
        
        assert command == expected_command
    
    @pytest.mark.unit
    def test_build_ocr_command_additional_options(self):
        """Test OCR command building with additional options."""
        processor = OCRProcessor(
            language="deu+eng",  # Multiple languages
            deskew=True,
            clean=True,
            optimize=1
        )
        input_path = Path("/tmp/input.pdf")
        output_path = Path("/tmp/output.pdf")
        
        command = processor._build_ocr_command(input_path, output_path)
        
        assert "--language" in command
        assert "deu+eng" in command
        assert "--deskew" in command
        assert "--clean" in command
        assert "--optimize" in command
        assert "1" in command
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_progress_tracking(self, mock_subprocess, ocr_processor, temp_pdf_files):
        """Test OCR processing with progress tracking."""
        input_path, output_path = temp_pdf_files
        
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"Processing page 1 of 3", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        progress_updates = []
        
        def progress_callback(progress: float, message: str):
            progress_updates.append((progress, message))
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat.return_value.st_size = 1024000
                
                result = await ocr_processor.process_pdf(
                    input_path, 
                    output_path,
                    progress_callback=progress_callback
                )
                
                assert result.success is True
                # Progress callback should have been called at least once
                assert len(progress_updates) >= 1
    
    @pytest.mark.unit
    def test_ocr_result_creation(self):
        """Test OCRResult object creation and validation."""
        start_time = datetime.now()
        
        result = OCRResult(
            success=True,
            input_file="/tmp/input.pdf",
            output_file="/tmp/output.pdf",
            processing_time=5.2,
            language="eng",
            ocr_mode="skip_text",
            output_size_bytes=1024000,
            pages_processed=10,
            error_message=None,
            start_time=start_time,
            end_time=start_time + timedelta(seconds=5)
        )
        
        assert result.success is True
        assert result.input_file == "/tmp/input.pdf"
        assert result.processing_time == 5.2
        assert result.language == "eng"
        assert result.ocr_mode == "skip_text"
        assert result.pages_processed == 10
        assert result.error_message is None
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_large_file(self, mock_subprocess, temp_pdf_files):
        """Test OCR processing of large files."""
        input_path, output_path = temp_pdf_files
        
        # Large file processor with extended timeout
        processor = OCRProcessor(timeout_seconds=300)
        
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"Large file processed", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat.return_value.st_size = 50 * 1024 * 1024  # 50MB
                
                result = await processor.process_pdf(input_path, output_path)
                
                assert result.success is True
                assert result.output_size_bytes == 50 * 1024 * 1024
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_memory_monitoring(self, mock_subprocess, ocr_processor, temp_pdf_files):
        """Test OCR processing with memory monitoring."""
        input_path, output_path = temp_pdf_files
        
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"OCR completed", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.stat') as mock_stat:
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_stat.return_value.st_size = 1024000
                    mock_memory.return_value.percent = 85.0  # High memory usage
                    
                    result = await ocr_processor.process_pdf(input_path, output_path)
                    
                    assert result.success is True
                    # Should handle high memory usage gracefully
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_concurrent_processing(self, mock_subprocess, temp_pdf_files):
        """Test concurrent OCR processing."""
        input_paths = []
        output_paths = []
        
        # Create multiple temp file pairs
        for i in range(3):
            input_file = tempfile.NamedTemporaryFile(suffix=f'_input_{i}.pdf', delete=False)
            output_file = tempfile.NamedTemporaryFile(suffix=f'_output_{i}.pdf', delete=False)
            input_file.write(b"%PDF-1.4\ntest content")
            input_file.flush()
            input_paths.append(Path(input_file.name))
            output_paths.append(Path(output_file.name))
        
        try:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"OCR completed", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            processor = OCRProcessor()
            
            async def process_single(input_path, output_path):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.stat') as mock_stat:
                        mock_stat.return_value.st_size = 1024000
                        return await processor.process_pdf(input_path, output_path)
            
            # Process files concurrently
            tasks = []
            for i in range(3):
                task = process_single(input_paths[i], output_paths[i])
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert len(results) == 3
            for result in results:
                assert result.success is True
                
        finally:
            # Cleanup
            for path in input_paths + output_paths:
                path.unlink(missing_ok=True)
    
    @pytest.mark.unit
    def test_ocr_error_types(self):
        """Test different OCR error types."""
        # Basic OCR error
        error = OCRError("OCR processing failed")
        assert str(error) == "OCR processing failed"
        
        # Timeout error
        timeout_error = OCRTimeoutError("Process timed out after 30 seconds")
        assert str(timeout_error) == "Process timed out after 30 seconds"
        assert isinstance(timeout_error, OCRError)
    
    @pytest.mark.unit
    @patch('app.ocr.asyncio.create_subprocess_exec')
    async def test_process_pdf_validation(self, mock_subprocess, ocr_processor):
        """Test input validation for process_pdf method."""
        # Test with None input
        with pytest.raises(OCRError) as exc_info:
            await ocr_processor.process_pdf(None, Path("/tmp/output.pdf"))
        assert "Input path cannot be None" in str(exc_info.value)
        
        # Test with None output
        with pytest.raises(OCRError) as exc_info:
            await ocr_processor.process_pdf(Path("/tmp/input.pdf"), None)
        assert "Output path cannot be None" in str(exc_info.value)
    
    @pytest.mark.unit
    @patch('app.ocr.shutil.which')
    def test_ocrmypdf_availability_check(self, mock_which, ocr_processor):
        """Test checking for OCRmyPDF availability."""
        # OCRmyPDF available
        mock_which.return_value = "/usr/bin/ocrmypdf"
        assert ocr_processor._check_ocrmypdf_available() is True
        
        # OCRmyPDF not available
        mock_which.return_value = None
        assert ocr_processor._check_ocrmypdf_available() is False
    
    @pytest.mark.unit
    async def test_process_pdf_ocrmypdf_not_installed(self, ocr_processor, temp_pdf_files):
        """Test OCR processing when OCRmyPDF is not installed."""
        input_path, output_path = temp_pdf_files
        
        with patch.object(ocr_processor, '_check_ocrmypdf_available', return_value=False):
            with pytest.raises(OCRError) as exc_info:
                await ocr_processor.process_pdf(input_path, output_path)
            
            assert "OCRmyPDF is not installed" in str(exc_info.value)