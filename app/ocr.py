"""
OCR processing with OCRmyPDF integration.

Handles OCR processing for PDF documents with support for skip-text and
force-OCR modes, progress reporting, and error handling.
"""

from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass
import subprocess
import asyncio
import re
import tempfile
import shutil

from .logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """Result of OCR processing operation."""
    success: bool
    output_path: Optional[Path]
    pages_processed: int
    ocr_applied: bool
    text_pages_found: int = 0
    image_pages_found: int = 0
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


class PDFProcessingError(Exception):
    """Raised when PDF processing fails with recoverable error."""
    
    def __init__(self, message: str, retry_possible: bool = True):
        self.message = message
        self.retry_possible = retry_possible
        super().__init__(message)


class OCRProcessor:
    """Handles OCR processing with OCRmyPDF."""
    
    def __init__(self, timeout_seconds: int = 600):
        """
        Initialize OCR processor.
        
        Args:
            timeout_seconds: Maximum time to allow for OCR processing
        """
        self.timeout_seconds = timeout_seconds
    
    async def process_pdf(
        self,
        input_path: Path,
        output_path: Path,
        force_ocr: bool = False,
        language: str = "eng",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> OCRResult:
        """
        Process PDF with OCR and return results.
        
        Args:
            input_path: Source PDF file
            output_path: OCR output PDF file
            force_ocr: Force OCR on all pages regardless of existing text
            language: OCR language code (default: "eng")
            progress_callback: Optional progress reporting callback
            
        Returns:
            OCRResult with processing details
        """
        import time
        start_time = time.time()
        
        # Validate input file
        if not input_path.exists():
            error_msg = f"Input PDF file not found: {input_path}"
            logger.error(error_msg)
            return OCRResult(
                success=False,
                output_path=None,
                pages_processed=0,
                ocr_applied=False,
                error_message=error_msg
            )
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get page count first for progress tracking
            page_count = await self._get_pdf_page_count(input_path)
            
            if progress_callback:
                progress_callback(0.1, f"Starting OCR processing ({page_count} pages)")
            
            # Build OCRmyPDF command
            cmd = self._build_ocr_command(input_path, output_path, force_ocr, language)
            
            logger.info(
                "Starting OCR processing",
                input_path=str(input_path),
                output_path=str(output_path),
                force_ocr=force_ocr,
                language=language,
                page_count=page_count
            )
            
            # Execute OCR with progress monitoring
            result = await self._execute_ocr_with_progress(
                cmd, page_count, progress_callback
            )
            
            processing_time = time.time() - start_time
            
            if result.success and output_path.exists():
                # Analyze what OCR was applied
                ocr_analysis = await self._analyze_ocr_result(
                    input_path, output_path, force_ocr
                )
                
                logger.info(
                    "OCR processing completed",
                    input_path=str(input_path),
                    pages_processed=page_count,
                    ocr_applied=ocr_analysis["ocr_applied"],
                    text_pages=ocr_analysis["text_pages"],
                    image_pages=ocr_analysis["image_pages"],
                    processing_time=f"{processing_time:.2f}s"
                )
                
                return OCRResult(
                    success=True,
                    output_path=output_path,
                    pages_processed=page_count,
                    ocr_applied=ocr_analysis["ocr_applied"],
                    text_pages_found=ocr_analysis["text_pages"],
                    image_pages_found=ocr_analysis["image_pages"],
                    processing_time_seconds=processing_time
                )
            else:
                return OCRResult(
                    success=False,
                    output_path=None,
                    pages_processed=page_count,
                    ocr_applied=False,
                    error_message=result.error_message,
                    processing_time_seconds=processing_time
                )
                
        except PDFProcessingError as e:
            processing_time = time.time() - start_time
            logger.error(
                "PDF processing failed",
                input_path=str(input_path),
                error=e.message,
                retry_possible=e.retry_possible
            )
            return OCRResult(
                success=False,
                output_path=None,
                pages_processed=0,
                ocr_applied=False,
                error_message=e.message,
                processing_time_seconds=processing_time
            )
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error during OCR processing: {str(e)}"
            logger.error(error_msg, input_path=str(input_path))
            return OCRResult(
                success=False,
                output_path=None,
                pages_processed=0,
                ocr_applied=False,
                error_message=error_msg,
                processing_time_seconds=processing_time
            )
    
    def _build_ocr_command(
        self,
        input_path: Path,
        output_path: Path,
        force_ocr: bool,
        language: str
    ) -> List[str]:
        """Build OCRmyPDF command with appropriate options."""
        cmd = [
            "ocrmypdf",
            "--language", language,
            "--output-type", "pdf",
            "--optimize", "1",
            "--jpeg-quality", "95",
            "--png-quality", "95",
        ]
        
        if force_ocr:
            # Force OCR on all pages
            cmd.extend(["--force-ocr"])
            logger.debug("Using force-OCR mode")
        else:
            # Skip pages that already have text (default behavior)
            cmd.extend(["--skip-text"])
            logger.debug("Using skip-text mode")
        
        # Add timeout
        cmd.extend(["--timeout", str(self.timeout_seconds)])
        
        # Input and output files
        cmd.extend([str(input_path), str(output_path)])
        
        return cmd
    
    async def _execute_ocr_with_progress(
        self,
        cmd: List[str],
        page_count: int,
        progress_callback: Optional[Callable[[float, str], None]]
    ) -> OCRResult:
        """Execute OCR command with progress monitoring."""
        try:
            # Create process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Monitor progress by checking stderr output
            stderr_lines = []
            current_page = 0
            
            while True:
                try:
                    # Read stderr with timeout
                    line = await asyncio.wait_for(
                        process.stderr.readline(), 
                        timeout=30.0
                    )
                    
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        stderr_lines.append(line_str)
                        
                        # Look for page progress indicators
                        page_match = re.search(r'page\s+(\d+)', line_str, re.IGNORECASE)
                        if page_match:
                            current_page = max(current_page, int(page_match.group(1)))
                            
                            if progress_callback and page_count > 0:
                                progress = 0.1 + (0.8 * current_page / page_count)
                                progress_callback(
                                    min(progress, 0.9),
                                    f"Processing page {current_page}/{page_count}"
                                )
                
                except asyncio.TimeoutError:
                    # Continue if no output - OCR might be working on a page
                    if progress_callback:
                        progress_callback(
                            0.5, 
                            f"Processing... (current page ~{current_page})"
                        )
                    continue
            
            # Wait for process to complete
            await process.wait()
            
            if progress_callback:
                progress_callback(1.0, "OCR processing completed")
            
            if process.returncode == 0:
                return OCRResult(success=True, output_path=None, pages_processed=0, ocr_applied=True)
            else:
                # Read any remaining stderr
                remaining_stderr = await process.stderr.read()
                if remaining_stderr:
                    stderr_lines.append(remaining_stderr.decode('utf-8', errors='ignore'))
                
                error_message = self._parse_ocr_error('\n'.join(stderr_lines))
                
                return OCRResult(
                    success=False,
                    output_path=None,
                    pages_processed=current_page,
                    ocr_applied=False,
                    error_message=error_message
                )
                
        except asyncio.TimeoutError:
            error_msg = f"OCR processing timed out after {self.timeout_seconds} seconds"
            logger.warning(error_msg)
            
            # Try to terminate the process
            if process:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except:
                    process.kill()
            
            raise PDFProcessingError(error_msg, retry_possible=True)
        
        except Exception as e:
            error_msg = f"Failed to execute OCR command: {str(e)}"
            logger.error(error_msg)
            raise PDFProcessingError(error_msg, retry_possible=False)
    
    def _parse_ocr_error(self, stderr_output: str) -> str:
        """Parse OCRmyPDF error output to provide user-friendly error message."""
        if "encrypted" in stderr_output.lower():
            return "PDF is encrypted and cannot be processed. Please provide an unencrypted version."
        
        if "password" in stderr_output.lower():
            return "PDF is password-protected. Please provide the password or an unprotected version."
        
        if "corrupted" in stderr_output.lower() or "damaged" in stderr_output.lower():
            return "PDF file appears to be corrupted. Please try with a different file."
        
        if "timeout" in stderr_output.lower():
            return "OCR processing timed out. The file may be too large or complex."
        
        if "no space left" in stderr_output.lower():
            return "Insufficient disk space to complete OCR processing."
        
        # Return first meaningful error line
        lines = stderr_output.strip().split('\n')
        for line in lines:
            if line.strip() and not line.startswith('INFO:'):
                return f"OCR processing failed: {line.strip()}"
        
        return "OCR processing failed with unknown error"
    
    async def _get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF file."""
        try:
            # Use pdfinfo if available for faster page counting
            process = await asyncio.create_subprocess_exec(
                "pdfinfo", str(pdf_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode('utf-8')
                for line in output.split('\n'):
                    if line.startswith('Pages:'):
                        return int(line.split(':')[1].strip())
            
            # Fallback to PyMuPDF
            import fitz
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            doc.close()
            return page_count
            
        except Exception:
            # Fallback estimate
            logger.warning("Could not determine page count, estimating based on file size")
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            return max(1, int(file_size_mb / 0.1))  # Rough estimate: ~100KB per page
    
    async def _analyze_ocr_result(
        self,
        input_path: Path,
        output_path: Path,
        force_ocr: bool
    ) -> dict:
        """Analyze OCR results to determine what processing was applied."""
        try:
            # Compare file sizes as a rough indicator
            input_size = input_path.stat().st_size
            output_size = output_path.stat().st_size
            
            # If force OCR was used, OCR was definitely applied
            if force_ocr:
                return {
                    "ocr_applied": True,
                    "text_pages": 0,  # Will be determined by parser
                    "image_pages": 0  # Will be determined by parser
                }
            
            # If output is significantly larger, OCR was likely applied
            size_ratio = output_size / input_size if input_size > 0 else 1.0
            ocr_applied = size_ratio > 1.1  # 10% size increase threshold
            
            return {
                "ocr_applied": ocr_applied,
                "text_pages": 0,  # Will be determined by parser
                "image_pages": 0  # Will be determined by parser
            }
            
        except Exception as e:
            logger.warning("Could not analyze OCR result", error=str(e))
            return {
                "ocr_applied": True,  # Assume OCR was applied if we can't determine
                "text_pages": 0,
                "image_pages": 0
            }
    
    async def check_ocr_dependencies(self) -> dict:
        """Check if OCRmyPDF and dependencies are available."""
        results = {
            "ocrmypdf": False,
            "tesseract": False,
            "pdfinfo": False,
            "languages": []
        }
        
        # Check OCRmyPDF
        try:
            process = await asyncio.create_subprocess_exec(
                "ocrmypdf", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            results["ocrmypdf"] = process.returncode == 0
        except Exception:
            pass
        
        # Check Tesseract
        try:
            process = await asyncio.create_subprocess_exec(
                "tesseract", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            results["tesseract"] = process.returncode == 0
        except Exception:
            pass
        
        # Check pdfinfo
        try:
            process = await asyncio.create_subprocess_exec(
                "pdfinfo", "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            results["pdfinfo"] = process.returncode == 0
        except Exception:
            pass
        
        # Get available Tesseract languages
        if results["tesseract"]:
            try:
                process = await asyncio.create_subprocess_exec(
                    "tesseract", "--list-langs",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    output = stdout.decode('utf-8')
                    lines = output.strip().split('\n')[1:]  # Skip header
                    results["languages"] = [lang.strip() for lang in lines if lang.strip()]
            except Exception:
                pass
        
        return results