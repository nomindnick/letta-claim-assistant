"""
PDF parsing using PyMuPDF for text extraction and metadata.

Handles page-aligned text extraction from PDFs with support for various
document types (born-digital, scanned, mixed) and metadata preservation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json
import re
from datetime import datetime

from .logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    """Parsed content from a single PDF page."""
    page_no: int
    text: str
    doc_name: str
    doc_id: str
    md5: str
    char_count: int
    has_images: bool = False
    has_tables: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class DocumentMetadata:
    """Document metadata extracted from PDF."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: int = 0
    file_size_bytes: int = 0
    pdf_version: Optional[str] = None
    encrypted: bool = False
    has_form_fields: bool = False
    language: Optional[str] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class PDFParsingError(Exception):
    """Raised when PDF parsing fails."""
    
    def __init__(self, message: str, recoverable: bool = False):
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)


class PDFParser:
    """Handles PDF text extraction with PyMuPDF."""
    
    def __init__(self):
        """Initialize PDF parser."""
        self.supported_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    def extract_pages(
        self, 
        pdf_path: Path,
        doc_id: Optional[str] = None
    ) -> List[PageContent]:
        """
        Extract page-aligned text from PDF.
        
        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier (defaults to filename stem)
            
        Returns:
            List of PageContent objects with extracted text
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise PDFParsingError(
                "PyMuPDF (fitz) is required for PDF parsing. Install with: pip install pymupdf",
                recoverable=False
            )
        
        if not pdf_path.exists():
            raise PDFParsingError(f"PDF file not found: {pdf_path}", recoverable=False)
        
        if doc_id is None:
            doc_id = pdf_path.stem
        
        doc_name = pdf_path.name
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            if doc.needs_pass:
                doc.close()
                raise PDFParsingError(
                    f"PDF is password protected: {pdf_path}",
                    recoverable=False
                )
            
            logger.info(
                "Starting PDF text extraction",
                pdf_path=str(pdf_path),
                doc_id=doc_id,
                page_count=doc.page_count
            )
            
            pages = []
            
            for page_num in range(doc.page_count):
                try:
                    page_content = self._extract_page_content(
                        doc, page_num, doc_name, doc_id
                    )
                    pages.append(page_content)
                    
                except Exception as e:
                    logger.warning(
                        "Failed to extract page content",
                        pdf_path=str(pdf_path),
                        page_num=page_num + 1,
                        error=str(e)
                    )
                    
                    # Create empty page content for failed pages
                    empty_page = PageContent(
                        page_no=page_num + 1,
                        text="",
                        doc_name=doc_name,
                        doc_id=doc_id,
                        md5=hashlib.md5(b"").hexdigest(),
                        char_count=0,
                        metadata={"extraction_error": str(e)}
                    )
                    pages.append(empty_page)
            
            doc.close()
            
            # Filter out completely empty pages unless all pages are empty
            non_empty_pages = [p for p in pages if p.text.strip()]
            
            if non_empty_pages:
                logger.info(
                    "PDF text extraction completed",
                    pdf_path=str(pdf_path),
                    total_pages=len(pages),
                    pages_with_text=len(non_empty_pages),
                    total_chars=sum(p.char_count for p in non_empty_pages)
                )
                return pages
            else:
                logger.warning(
                    "No text found in PDF - may be image-only or encrypted",
                    pdf_path=str(pdf_path)
                )
                return pages
            
        except fitz.fitz.FileDataError as e:
            error_msg = f"Corrupted or invalid PDF file: {str(e)}"
            logger.error(error_msg, pdf_path=str(pdf_path))
            raise PDFParsingError(error_msg, recoverable=False)
            
        except fitz.fitz.FileNotFoundError as e:
            error_msg = f"PDF file not accessible: {str(e)}"
            logger.error(error_msg, pdf_path=str(pdf_path))
            raise PDFParsingError(error_msg, recoverable=False)
            
        except Exception as e:
            error_msg = f"Unexpected error during PDF parsing: {str(e)}"
            logger.error(error_msg, pdf_path=str(pdf_path))
            raise PDFParsingError(error_msg, recoverable=True)
    
    def _extract_page_content(
        self, 
        doc, 
        page_num: int, 
        doc_name: str, 
        doc_id: str
    ) -> PageContent:
        """Extract content from a single page."""
        page = doc[page_num]
        
        # Extract text using different methods and choose the best
        text_dict = page.get_text("dict")
        text_blocks = self._extract_text_from_blocks(text_dict.get("blocks", []))
        
        # Fallback to simple text extraction if blocks don't work well
        if len(text_blocks.strip()) < 10:
            text_blocks = page.get_text()
        
        # Clean and normalize text
        text = self._clean_text(text_blocks)
        
        # Calculate MD5 hash
        text_bytes = text.encode('utf-8')
        md5_hash = hashlib.md5(text_bytes).hexdigest()
        
        # Detect content characteristics
        has_images = len(page.get_images()) > 0
        has_tables = self._detect_tables(text_dict)
        
        # Create metadata
        metadata = {
            "page_width": page.rect.width,
            "page_height": page.rect.height,
            "rotation": page.rotation,
            "has_images": has_images,
            "has_tables": has_tables,
            "extraction_method": "blocks" if len(text_blocks) > len(page.get_text()) else "simple"
        }
        
        return PageContent(
            page_no=page_num + 1,  # 1-indexed page numbers
            text=text,
            doc_name=doc_name,
            doc_id=doc_id,
            md5=md5_hash,
            char_count=len(text),
            has_images=has_images,
            has_tables=has_tables,
            metadata=metadata
        )
    
    def _extract_text_from_blocks(self, blocks: List[dict]) -> str:
        """Extract text from PyMuPDF text blocks with proper ordering."""
        text_parts = []
        
        # Sort blocks by position (top to bottom, left to right)
        sorted_blocks = sorted(blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))
        
        for block in sorted_blocks:
            if block.get("type") == 0:  # Text block
                block_text = self._extract_text_from_text_block(block)
                if block_text.strip():
                    text_parts.append(block_text)
        
        return "\n\n".join(text_parts)
    
    def _extract_text_from_text_block(self, block: dict) -> str:
        """Extract text from a single text block."""
        lines = []
        
        for line in block.get("lines", []):
            line_text_parts = []
            
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    line_text_parts.append(text)
            
            if line_text_parts:
                lines.append(" ".join(line_text_parts))
        
        return "\n".join(lines)
    
    def _detect_tables(self, text_dict: dict) -> bool:
        """Simple heuristic to detect if page contains tables."""
        # Look for common table indicators in the text structure
        blocks = text_dict.get("blocks", [])
        
        # Count text blocks that might be table cells (small, aligned)
        potential_cells = 0
        total_blocks = len([b for b in blocks if b.get("type") == 0])
        
        if total_blocks < 4:  # Need minimum blocks to form a table
            return False
        
        for block in blocks:
            if block.get("type") == 0:  # Text block
                bbox = block.get("bbox", [0, 0, 0, 0])
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Small blocks might be table cells
                if width < 200 and height < 50:
                    potential_cells += 1
        
        # If more than 30% of blocks look like cells, probably a table
        return (potential_cells / total_blocks) > 0.3 if total_blocks > 0 else False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Handle different encodings
        cleaned_text = self._handle_encoding_issues(text)
        
        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Remove excessive line breaks while preserving paragraph structure
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        # Remove leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _handle_encoding_issues(self, text: str) -> str:
        """Handle common encoding issues in extracted text."""
        # Common character replacements
        replacements = {
            '\ufeff': '',  # BOM
            '\u00a0': ' ',  # Non-breaking space
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark  
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def get_document_metadata(self, pdf_path: Path) -> DocumentMetadata:
        """Extract document metadata from PDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise PDFParsingError(
                "PyMuPDF (fitz) is required for metadata extraction",
                recoverable=False
            )
        
        if not pdf_path.exists():
            raise PDFParsingError(f"PDF file not found: {pdf_path}", recoverable=False)
        
        try:
            doc = fitz.open(pdf_path)
            
            # Get basic metadata
            metadata_dict = doc.metadata
            
            # Parse dates
            creation_date = self._parse_pdf_date(metadata_dict.get("creationDate"))
            modification_date = self._parse_pdf_date(metadata_dict.get("modDate"))
            
            # Extract keywords
            keywords = []
            if metadata_dict.get("keywords"):
                keywords = [kw.strip() for kw in metadata_dict["keywords"].split(",") if kw.strip()]
            
            # Get file information
            file_size = pdf_path.stat().st_size
            
            # Check for form fields
            has_form_fields = False
            for page in doc:
                if page.widgets():
                    has_form_fields = True
                    break
            
            metadata = DocumentMetadata(
                title=metadata_dict.get("title"),
                author=metadata_dict.get("author"),
                subject=metadata_dict.get("subject"),
                creator=metadata_dict.get("creator"),
                producer=metadata_dict.get("producer"),
                creation_date=creation_date,
                modification_date=modification_date,
                page_count=doc.page_count,
                file_size_bytes=file_size,
                pdf_version=f"1.{doc.pdf_version()}" if hasattr(doc, 'pdf_version') else None,
                encrypted=doc.needs_pass,
                has_form_fields=has_form_fields,
                language=metadata_dict.get("language"),
                keywords=keywords
            )
            
            doc.close()
            
            logger.debug(
                "Document metadata extracted",
                pdf_path=str(pdf_path),
                title=metadata.title,
                author=metadata.author,
                page_count=metadata.page_count,
                file_size_mb=f"{file_size / 1024 / 1024:.1f}"
            )
            
            return metadata
            
        except Exception as e:
            error_msg = f"Failed to extract metadata: {str(e)}"
            logger.error(error_msg, pdf_path=str(pdf_path))
            raise PDFParsingError(error_msg, recoverable=True)
    
    def _parse_pdf_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse PDF date string to datetime object."""
        if not date_str:
            return None
        
        # PDF date format: D:YYYYMMDDHHmmSSOHH'mm
        # Example: D:20230214120000+05'00
        
        try:
            # Remove D: prefix if present
            if date_str.startswith("D:"):
                date_str = date_str[2:]
            
            # Extract basic date components
            if len(date_str) >= 14:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                hour = int(date_str[8:10])
                minute = int(date_str[10:12])
                second = int(date_str[12:14])
                
                return datetime(year, month, day, hour, minute, second)
            elif len(date_str) >= 8:
                # Just date components
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                
                return datetime(year, month, day)
                
        except (ValueError, IndexError) as e:
            logger.debug(f"Could not parse PDF date: {date_str}, error: {e}")
        
        return None
    
    def save_parsed_content(
        self,
        pages: List[PageContent],
        output_path: Path,
        metadata: Optional[DocumentMetadata] = None
    ) -> None:
        """
        Save parsed page content to JSONL file.
        
        Args:
            pages: List of PageContent objects
            output_path: Path to save JSONL file
            metadata: Optional document metadata to include
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write metadata as first line if available
                if metadata:
                    metadata_record = {
                        "record_type": "metadata",
                        "title": metadata.title,
                        "author": metadata.author,
                        "subject": metadata.subject,
                        "creation_date": metadata.creation_date.isoformat() if metadata.creation_date else None,
                        "page_count": metadata.page_count,
                        "file_size_bytes": metadata.file_size_bytes,
                        "keywords": metadata.keywords
                    }
                    f.write(json.dumps(metadata_record, ensure_ascii=False) + '\n')
                
                # Write page content records
                for page in pages:
                    page_record = {
                        "record_type": "page",
                        "page_no": page.page_no,
                        "text": page.text,
                        "doc_name": page.doc_name,
                        "doc_id": page.doc_id,
                        "md5": page.md5,
                        "char_count": page.char_count,
                        "has_images": page.has_images,
                        "has_tables": page.has_tables,
                        "metadata": page.metadata
                    }
                    f.write(json.dumps(page_record, ensure_ascii=False) + '\n')
            
            logger.info(
                "Parsed content saved to JSONL",
                output_path=str(output_path),
                page_count=len(pages),
                total_chars=sum(p.char_count for p in pages)
            )
            
        except Exception as e:
            error_msg = f"Failed to save parsed content: {str(e)}"
            logger.error(error_msg, output_path=str(output_path))
            raise PDFParsingError(error_msg, recoverable=True)
    
    def load_parsed_content(self, jsonl_path: Path) -> Tuple[List[PageContent], Optional[DocumentMetadata]]:
        """
        Load parsed content from JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file
            
        Returns:
            Tuple of (pages, metadata)
        """
        if not jsonl_path.exists():
            raise PDFParsingError(f"JSONL file not found: {jsonl_path}", recoverable=False)
        
        try:
            pages = []
            metadata = None
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        record_type = record.get("record_type")
                        
                        if record_type == "metadata":
                            creation_date = None
                            if record.get("creation_date"):
                                creation_date = datetime.fromisoformat(record["creation_date"])
                            
                            metadata = DocumentMetadata(
                                title=record.get("title"),
                                author=record.get("author"),
                                subject=record.get("subject"),
                                creation_date=creation_date,
                                page_count=record.get("page_count", 0),
                                file_size_bytes=record.get("file_size_bytes", 0),
                                keywords=record.get("keywords", [])
                            )
                        
                        elif record_type == "page":
                            page = PageContent(
                                page_no=record["page_no"],
                                text=record["text"],
                                doc_name=record["doc_name"],
                                doc_id=record["doc_id"],
                                md5=record["md5"],
                                char_count=record["char_count"],
                                has_images=record.get("has_images", False),
                                has_tables=record.get("has_tables", False),
                                metadata=record.get("metadata", {})
                            )
                            pages.append(page)
                    
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Invalid JSON in JSONL file",
                            jsonl_path=str(jsonl_path),
                            line_num=line_num,
                            error=str(e)
                        )
            
            logger.info(
                "Parsed content loaded from JSONL",
                jsonl_path=str(jsonl_path),
                page_count=len(pages),
                has_metadata=metadata is not None
            )
            
            return pages, metadata
            
        except Exception as e:
            error_msg = f"Failed to load parsed content: {str(e)}"
            logger.error(error_msg, jsonl_path=str(jsonl_path))
            raise PDFParsingError(error_msg, recoverable=True)