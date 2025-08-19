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
from .error_handler import (
    FileProcessingError, ResourceError, handle_error, create_context,
    ErrorSeverity, RecoveryStrategy, RecoveryAction
)
from .resource_monitor import check_disk_space_before_operation

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


class OCRError(Exception):
    """Base class for OCR processing errors."""
    
    def __init__(self, message: str, recoverable: bool = True):
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)


class OCRTimeoutError(OCRError):
    """Raised when OCR processing times out."""
    
    def __init__(self, timeout_seconds: int):
        message = f"OCR processing timed out after {timeout_seconds} seconds"
        super().__init__(message, recoverable=True)
        self.timeout_seconds = timeout_seconds


# Legacy exception for backward compatibility
class PDFProcessingError(FileProcessingError):
    """Legacy PDF processing error - use FileProcessingError instead."""
    
    def __init__(self, message: str, retry_possible: bool = True):
        # Map to new error format
        recovery_strategy = RecoveryStrategy.RETRY if retry_possible else RecoveryStrategy.MANUAL
        super().__init__(
            file_path="unknown",
            operation="OCR processing",
            reason=message,
            recovery_strategy=recovery_strategy
        )
        self.message = message
        self.retry_possible = retry_possible


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
            error = FileProcessingError(
                file_path=input_path,
                operation="OCR processing",
                reason="File not found",
                error_code="OCR_FILE_NOT_FOUND",
                suggestion="Please verify the file path and ensure the file exists"
            )
            handle_error(error, create_context(
                operation="OCR processing",
                file_path=input_path
            ))
            return OCRResult(
                success=False,
                output_path=None,
                pages_processed=0,
                ocr_applied=False,
                error_message=error.user_message
            )
        
        # Check file size and disk space requirements
        try:
            file_size = input_path.stat().st_size
            required_space_gb = (file_size * 3) / (1024**3)  # Estimate 3x file size needed
            check_disk_space_before_operation(required_space_gb, output_path.parent)
        except ResourceError as e:
            handle_error(e, create_context(
                operation="OCR processing",
                file_path=input_path
            ))
            return OCRResult(
                success=False,
                output_path=None,
                pages_processed=0,
                ocr_applied=False,
                error_message=e.user_message
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
                
                stderr_text = '\n'.join(stderr_lines)
                
                # Log the actual error for debugging
                logger.error(
                    "OCR process failed",
                    return_code=process.returncode,
                    stderr_output=stderr_text[:1000],  # First 1000 chars
                    file_path=str(cmd[-2])
                )
                
                error = self._parse_ocr_error(stderr_text, Path(cmd[-2]))  # Input file path
                handle_error(error, create_context(
                    operation="OCR processing", 
                    file_path=cmd[-2]
                ))
                error_message = error.user_message
                
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
    
    def _parse_ocr_error(self, stderr_output: str, file_path: Path) -> FileProcessingError:
        """Parse OCRmyPDF error output and create appropriate FileProcessingError."""
        
        # Check for specific error patterns
        if "encrypted" in stderr_output.lower():
            return FileProcessingError(
                file_path=file_path,
                operation="OCR processing",
                reason="PDF is encrypted",
                error_code="OCR_ENCRYPTED_PDF",
                user_message="This PDF is encrypted and cannot be processed",
                suggestion="Please provide an unencrypted version of the PDF or remove password protection",
                recovery_actions=[
                    RecoveryAction(
                        action_id="decrypt_pdf",
                        label="Remove encryption",
                        description="Remove password protection from the PDF",
                        is_primary=True
                    ),
                    RecoveryAction(
                        action_id="try_different_file",
                        label="Try different file",
                        description="Use a different PDF file"
                    )
                ],
                recovery_strategy=RecoveryStrategy.MANUAL
            )
        
        if "password" in stderr_output.lower():
            return FileProcessingError(
                file_path=file_path,
                operation="OCR processing",
                reason="PDF is password-protected",
                error_code="OCR_PASSWORD_PROTECTED",
                user_message="This PDF is password-protected",
                suggestion="Please provide the password or use an unprotected version",
                recovery_actions=[
                    RecoveryAction(
                        action_id="provide_password",
                        label="Provide password",
                        description="Enter the PDF password to unlock the file",
                        is_primary=True
                    )
                ],
                recovery_strategy=RecoveryStrategy.MANUAL
            )
        
        if "corrupted" in stderr_output.lower() or "damaged" in stderr_output.lower():
            return FileProcessingError(
                file_path=file_path,
                operation="OCR processing",
                reason="PDF file is corrupted",
                error_code="OCR_CORRUPTED_PDF",
                user_message="The PDF file appears to be corrupted or damaged",
                suggestion="Please try with a different file or repair the PDF",
                recovery_actions=[
                    RecoveryAction(
                        action_id="try_repair",
                        label="Try repair",
                        description="Attempt to repair the PDF file"
                    ),
                    RecoveryAction(
                        action_id="try_different_file",
                        label="Use different file",
                        description="Select a different PDF file",
                        is_primary=True
                    )
                ],
                recovery_strategy=RecoveryStrategy.MANUAL
            )
        
        # Check for actual timeout error (not just the word "timeout" in command line args)
        if ("timeout" in stderr_output.lower() and 
            ("timed out" in stderr_output.lower() or 
             "time limit" in stderr_output.lower() or
             "maximum time" in stderr_output.lower())):
            return FileProcessingError(
                file_path=file_path,
                operation="OCR processing",
                reason="Processing timeout",
                error_code="OCR_TIMEOUT",
                user_message="OCR processing timed out - the file may be too large or complex",
                suggestion="Try processing smaller sections or increase the timeout setting",
                recovery_actions=[
                    RecoveryAction(
                        action_id="retry_force_ocr",
                        label="Retry with Force OCR",
                        description="Retry processing with force OCR disabled",
                        is_primary=True
                    ),
                    RecoveryAction(
                        action_id="split_pdf",
                        label="Split PDF",
                        description="Split the PDF into smaller sections"
                    )
                ],
                recovery_strategy=RecoveryStrategy.RETRY
            )
        
        if "no space left" in stderr_output.lower():
            return ResourceError(
                resource_type="disk",
                message="Insufficient disk space for OCR processing",
                user_message="Not enough disk space to complete OCR processing",
                suggestion="Please free up disk space and try again",
                context=create_context(operation="OCR processing", file_path=file_path),
                recovery_actions=[
                    RecoveryAction(
                        action_id="free_space",
                        label="Free disk space",
                        description="Delete unnecessary files to free up space",
                        is_primary=True
                    )
                ]
            )
        
        # Parse for first meaningful error line
        lines = stderr_output.strip().split('\n')
        error_detail = None
        for line in lines:
            if line.strip() and not line.startswith('INFO:'):
                error_detail = line.strip()
                break
        
        if not error_detail:
            error_detail = "Unknown OCR processing error"
        
        return FileProcessingError(
            file_path=file_path,
            operation="OCR processing",
            reason=error_detail,
            error_code="OCR_UNKNOWN_ERROR",
            user_message=f"OCR processing failed: {error_detail}",
            suggestion="Please try again or contact support if the problem persists",
            recovery_actions=[
                RecoveryAction(
                    action_id="retry_ocr",
                    label="Retry",
                    description="Try OCR processing again",
                    is_primary=True
                ),
                RecoveryAction(
                    action_id="skip_ocr",
                    label="Skip OCR",
                    description="Process without OCR (text-based PDFs only)"
                )
            ],
            recovery_strategy=RecoveryStrategy.RETRY
        )
    
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