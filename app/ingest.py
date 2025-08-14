"""
PDF ingestion pipeline with OCR, parsing, chunking, and embedding.

Orchestrates the complete document processing workflow from raw PDFs
to searchable vector embeddings with proper metadata preservation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import hashlib
import asyncio
import time
import shutil
import uuid
from datetime import datetime

from .logging_conf import get_logger
from .ocr import OCRProcessor, OCRResult
from .parsing import PDFParser, PageContent, DocumentMetadata, PDFParsingError
from .chunking import TextChunker, Chunk, ChunkingError
from .models import Matter

logger = get_logger(__name__)


@dataclass
class IngestionStats:
    """Statistics from ingestion pipeline."""
    doc_name: str
    doc_id: str
    total_pages: int
    total_chunks: int
    ocr_time_seconds: float
    parse_time_seconds: float
    chunk_time_seconds: float
    total_time_seconds: float
    ocr_status: str  # "full", "partial", "none"
    file_size_bytes: int
    pages_with_text: int
    pages_with_images: int
    avg_chunk_tokens: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class IngestionJob:
    """Ingestion job parameters and state."""
    job_id: str
    matter: Matter
    pdf_files: List[Path]
    force_ocr: bool = False
    ocr_language: str = "eng"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class IngestionError(Exception):
    """Raised when PDF ingestion fails."""
    
    def __init__(self, message: str, recoverable: bool = True, stage: str = "unknown"):
        self.message = message
        self.recoverable = recoverable
        self.stage = stage
        super().__init__(message)


class IngestionPipeline:
    """Orchestrates the complete PDF ingestion workflow."""
    
    def __init__(self, matter: Matter, settings: Optional[Dict[str, Any]] = None):
        """Initialize ingestion pipeline for a specific Matter."""
        self.matter = matter
        self.matter_path = matter.paths.root
        self.settings = settings or {}
        
        # Initialize processors
        ocr_timeout = self.settings.get('ocr_timeout_seconds', 600)
        self.ocr_processor = OCRProcessor(timeout_seconds=ocr_timeout)
        self.pdf_parser = PDFParser()
        
        # Initialize chunker with settings
        chunk_settings = self.settings.get('chunking', {})
        self.text_chunker = TextChunker(
            target_tokens=chunk_settings.get('target_tokens', 1000),
            overlap_percent=chunk_settings.get('overlap_percent', 0.15),
            min_chunk_tokens=chunk_settings.get('min_chunk_tokens', 50),
            max_chunk_tokens=chunk_settings.get('max_chunk_tokens', 1500)
        )
        
        # Ensure required directories exist
        self._ensure_directories()
        
        logger.info(
            "Ingestion pipeline initialized",
            matter_id=self.matter.id,
            matter_name=self.matter.name,
            target_tokens=self.text_chunker.target_tokens,
            overlap_percent=self.text_chunker.overlap_percent
        )
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.matter.paths.docs,
            self.matter.paths.docs_ocr,
            self.matter.paths.parsed,
            self.matter.paths.vectors,
            self.matter.paths.knowledge,
            self.matter.paths.chat,
            self.matter.paths.logs
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def ingest_pdfs(
        self,
        pdf_files: List[Path],
        force_ocr: bool = False,
        ocr_language: str = "eng",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, IngestionStats]:
        """
        Process multiple PDF files through complete ingestion pipeline.
        
        Args:
            pdf_files: List of PDF file paths to process
            force_ocr: Force OCR on all pages
            ocr_language: OCR language code
            progress_callback: Optional progress reporting callback
            
        Returns:
            Dictionary mapping file paths to IngestionStats
        """
        if not pdf_files:
            logger.warning("No PDF files provided for ingestion")
            return {}
        
        logger.info(
            "Starting batch PDF ingestion",
            matter_id=self.matter.id,
            file_count=len(pdf_files),
            force_ocr=force_ocr,
            ocr_language=ocr_language
        )
        
        results = {}
        total_files = len(pdf_files)
        
        for i, pdf_file in enumerate(pdf_files):
            file_progress_offset = i / total_files
            file_progress_scale = 1.0 / total_files
            
            def file_progress_callback(progress: float, message: str):
                if progress_callback:
                    overall_progress = file_progress_offset + (progress * file_progress_scale)
                    file_name = pdf_file.name
                    overall_message = f"[{i+1}/{total_files}] {file_name}: {message}"
                    progress_callback(overall_progress, overall_message)
            
            try:
                stats = await self._process_single_pdf(
                    pdf_file, force_ocr, ocr_language, file_progress_callback
                )
                results[str(pdf_file)] = stats
                
                logger.info(
                    "PDF processing completed",
                    pdf_file=str(pdf_file),
                    success=stats.success,
                    pages=stats.total_pages,
                    chunks=stats.total_chunks,
                    time_seconds=stats.total_time_seconds
                )
                
            except Exception as e:
                error_msg = f"Failed to process {pdf_file.name}: {str(e)}"
                logger.error(error_msg, pdf_file=str(pdf_file))
                
                # Create error stats
                results[str(pdf_file)] = IngestionStats(
                    doc_name=pdf_file.name,
                    doc_id=pdf_file.stem,
                    total_pages=0,
                    total_chunks=0,
                    ocr_time_seconds=0,
                    parse_time_seconds=0,
                    chunk_time_seconds=0,
                    total_time_seconds=0,
                    ocr_status="failed",
                    file_size_bytes=pdf_file.stat().st_size if pdf_file.exists() else 0,
                    pages_with_text=0,
                    pages_with_images=0,
                    avg_chunk_tokens=0,
                    success=False,
                    error_message=error_msg
                )
        
        if progress_callback:
            progress_callback(1.0, "Batch processing completed")
        
        # Log summary
        successful = sum(1 for s in results.values() if s.success)
        total_pages = sum(s.total_pages for s in results.values())
        total_chunks = sum(s.total_chunks for s in results.values())
        
        logger.info(
            "Batch PDF ingestion completed",
            matter_id=self.matter.id,
            total_files=total_files,
            successful_files=successful,
            failed_files=total_files - successful,
            total_pages=total_pages,
            total_chunks=total_chunks
        )
        
        return results
    
    async def _process_single_pdf(
        self,
        pdf_path: Path,
        force_ocr: bool = False,
        ocr_language: str = "eng",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> IngestionStats:
        """Process a single PDF through the complete pipeline."""
        start_time = time.time()
        doc_name = pdf_path.name
        doc_id = str(uuid.uuid4())  # Unique ID for this processing run
        
        if progress_callback:
            progress_callback(0.0, f"Starting processing of {doc_name}")
        
        logger.info(
            "Processing single PDF",
            pdf_path=str(pdf_path),
            doc_id=doc_id,
            force_ocr=force_ocr,
            ocr_language=ocr_language
        )
        
        try:
            # Validate input file
            if not pdf_path.exists():
                raise IngestionError(f"PDF file not found: {pdf_path}", recoverable=False, stage="validation")
            
            if pdf_path.stat().st_size == 0:
                raise IngestionError(f"PDF file is empty: {pdf_path}", recoverable=False, stage="validation")
            
            # Stage 1: Copy original file to docs directory
            original_path = self.matter.paths.docs / doc_name
            if not original_path.exists() or original_path.stat().st_size != pdf_path.stat().st_size:
                shutil.copy2(pdf_path, original_path)
                logger.debug("Original PDF copied", source=str(pdf_path), dest=str(original_path))
            
            # Stage 2: OCR Processing
            if progress_callback:
                progress_callback(0.1, "Starting OCR processing")
            
            ocr_start = time.time()
            ocr_output_path = self.matter.paths.docs_ocr / f"{pdf_path.stem}.ocr.pdf"
            
            def ocr_progress(progress: float, message: str):
                if progress_callback:
                    overall_progress = 0.1 + (progress * 0.4)  # OCR takes 0.1-0.5 of total
                    progress_callback(overall_progress, f"OCR: {message}")
            
            ocr_result = await self.ocr_processor.process_pdf(
                pdf_path, ocr_output_path, force_ocr, ocr_language, ocr_progress
            )
            ocr_time = time.time() - ocr_start
            
            if not ocr_result.success:
                raise IngestionError(
                    f"OCR processing failed: {ocr_result.error_message}",
                    recoverable=True,
                    stage="ocr"
                )
            
            # Use OCR'd PDF for parsing if successful, otherwise use original
            parsing_pdf = ocr_result.output_path if ocr_result.success else pdf_path
            
            # Stage 3: PDF Parsing
            if progress_callback:
                progress_callback(0.5, "Extracting text from PDF")
            
            parse_start = time.time()
            
            try:
                document_metadata = self.pdf_parser.get_document_metadata(parsing_pdf)
                pages = self.pdf_parser.extract_pages(parsing_pdf, doc_id)
            except PDFParsingError as e:
                raise IngestionError(f"PDF parsing failed: {e.message}", recoverable=e.recoverable, stage="parsing")
            
            parse_time = time.time() - parse_start
            
            if not pages:
                raise IngestionError("No content extracted from PDF", recoverable=True, stage="parsing")
            
            # Save parsed content
            parsed_path = self.matter.paths.parsed / f"{doc_id}.jsonl"
            self.pdf_parser.save_parsed_content(pages, parsed_path, document_metadata)
            
            # Stage 4: Text Chunking
            if progress_callback:
                progress_callback(0.7, "Creating text chunks")
            
            chunk_start = time.time()
            
            try:
                chunks = self.text_chunker.chunk_document(pages, detect_structure=True)
            except ChunkingError as e:
                raise IngestionError(f"Text chunking failed: {e.message}", recoverable=True, stage="chunking")
            
            chunk_time = time.time() - chunk_start
            
            if not chunks:
                logger.warning("No chunks created from PDF", pdf_path=str(pdf_path))
            
            # Save chunks
            chunks_path = self.matter.paths.parsed / f"{doc_id}_chunks.jsonl"
            self.text_chunker.save_chunks_jsonl(chunks, chunks_path)
            
            # Stage 5: Calculate statistics
            total_time = time.time() - start_time
            
            # Determine OCR status
            if force_ocr:
                ocr_status = "full"
            elif ocr_result.ocr_applied:
                if ocr_result.text_pages_found > 0:
                    ocr_status = "partial"
                else:
                    ocr_status = "full"
            else:
                ocr_status = "none"
            
            # Count page characteristics
            pages_with_text = sum(1 for p in pages if len(p.text.strip()) > 10)
            pages_with_images = sum(1 for p in pages if p.has_images)
            avg_chunk_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
            
            stats = IngestionStats(
                doc_name=doc_name,
                doc_id=doc_id,
                total_pages=len(pages),
                total_chunks=len(chunks),
                ocr_time_seconds=ocr_time,
                parse_time_seconds=parse_time,
                chunk_time_seconds=chunk_time,
                total_time_seconds=total_time,
                ocr_status=ocr_status,
                file_size_bytes=pdf_path.stat().st_size,
                pages_with_text=pages_with_text,
                pages_with_images=pages_with_images,
                avg_chunk_tokens=avg_chunk_tokens,
                success=True
            )
            
            if progress_callback:
                progress_callback(1.0, f"Processing completed: {len(chunks)} chunks created")
            
            logger.info(
                "PDF processing pipeline completed",
                pdf_path=str(pdf_path),
                doc_id=doc_id,
                pages=stats.total_pages,
                chunks=stats.total_chunks,
                ocr_status=stats.ocr_status,
                total_time=f"{stats.total_time_seconds:.2f}s"
            )
            
            return stats
            
        except IngestionError:
            # Re-raise ingestion errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            error_msg = f"Unexpected error during PDF processing: {str(e)}"
            logger.error(error_msg, pdf_path=str(pdf_path))
            raise IngestionError(error_msg, recoverable=False, stage="unknown")
    
    async def get_ingestion_capabilities(self) -> Dict[str, Any]:
        """Check ingestion capabilities and dependencies."""
        ocr_deps = await self.ocr_processor.check_ocr_dependencies()
        
        capabilities = {
            "ocr": {
                "available": ocr_deps["ocrmypdf"] and ocr_deps["tesseract"],
                "ocrmypdf": ocr_deps["ocrmypdf"],
                "tesseract": ocr_deps["tesseract"],
                "languages": ocr_deps["languages"]
            },
            "parsing": {
                "available": True,  # PyMuPDF should be available
                "formats": ["pdf"]
            },
            "chunking": {
                "available": True,
                "target_tokens": self.text_chunker.target_tokens,
                "overlap_percent": self.text_chunker.overlap_percent
            },
            "matter": {
                "id": self.matter.id,
                "name": self.matter.name,
                "paths_writable": self._check_paths_writable()
            }
        }
        
        return capabilities
    
    def _check_paths_writable(self) -> bool:
        """Check if all required paths are writable."""
        try:
            test_file = self.matter.paths.root / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """Clean up temporary processing files older than specified age."""
        import time
        from datetime import timedelta
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Clean up temporary OCR files
        temp_extensions = ['.tmp', '.temp', '_temp.pdf']
        
        for directory in [self.matter.paths.docs_ocr, self.matter.paths.parsed]:
            if not directory.exists():
                continue
                
            for file_path in directory.iterdir():
                if not file_path.is_file():
                    continue
                
                # Check if it's a temporary file
                is_temp = any(file_path.name.endswith(ext) for ext in temp_extensions)
                
                # Check age
                if is_temp and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        logger.debug("Cleaned up temp file", file_path=str(file_path))
                    except Exception as e:
                        logger.warning("Failed to clean up temp file", file_path=str(file_path), error=str(e))