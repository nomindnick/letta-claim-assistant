"""
Text chunking with overlap and structure awareness.

Creates optimally-sized text chunks from parsed PDF content with proper
overlap, page boundary preservation, and deduplication for embedding.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import hashlib
import re
import uuid
from collections import defaultdict

from .parsing import PageContent
from .logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Text chunk with metadata for embedding."""
    chunk_id: str
    text: str
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    token_count: int
    char_count: int
    md5: str
    section_title: Optional[str] = None
    chunk_index: int = 0
    overlap_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.overlap_info is None:
            self.overlap_info = {}


class ChunkingError(Exception):
    """Raised when text chunking fails."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class TextChunker:
    """Handles text chunking with overlap and structure awareness."""
    
    def __init__(
        self,
        target_tokens: int = 1000,
        overlap_percent: float = 0.15,
        min_chunk_tokens: int = 50,
        max_chunk_tokens: int = 1500
    ):
        """
        Initialize text chunker.
        
        Args:
            target_tokens: Target chunk size in tokens
            overlap_percent: Percentage overlap between chunks (0.0-0.5)
            min_chunk_tokens: Minimum chunk size to keep
            max_chunk_tokens: Maximum chunk size allowed
        """
        self.target_tokens = target_tokens
        self.overlap_percent = max(0.0, min(0.5, overlap_percent))
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        
        # Approximate tokens per character (rough estimate: 1 token ≈ 4 chars)
        self.chars_per_token = 4
        self.target_chars = target_tokens * self.chars_per_token
        self.overlap_chars = int(self.target_chars * self.overlap_percent)
        
        logger.debug(
            "TextChunker initialized",
            target_tokens=target_tokens,
            overlap_percent=overlap_percent,
            target_chars=self.target_chars,
            overlap_chars=self.overlap_chars
        )
    
    def chunk_document(
        self,
        pages: List[PageContent],
        detect_structure: bool = True
    ) -> List[Chunk]:
        """
        Create chunks from document pages.
        
        Args:
            pages: List of PageContent objects
            detect_structure: Whether to detect document structure
            
        Returns:
            List of Chunk objects ready for embedding
        """
        if not pages:
            logger.warning("No pages provided for chunking")
            return []
        
        doc_name = pages[0].doc_name
        doc_id = pages[0].doc_id
        
        logger.info(
            "Starting document chunking",
            doc_name=doc_name,
            page_count=len(pages),
            target_tokens=self.target_tokens,
            overlap_percent=self.overlap_percent
        )
        
        try:
            # Step 1: Structure-aware splitting if enabled
            if detect_structure:
                sections = self._detect_document_structure(pages)
            else:
                # Create single section with all pages
                sections = [{
                    'title': None,
                    'pages': pages,
                    'start_page': pages[0].page_no,
                    'end_page': pages[-1].page_no
                }]
            
            logger.debug(
                "Document structure detected",
                doc_name=doc_name,
                section_count=len(sections)
            )
            
            # Step 2: Create chunks from sections
            all_chunks = []
            chunk_index = 0
            
            for section in sections:
                section_chunks = self._chunk_section(
                    section, doc_id, doc_name, chunk_index
                )
                all_chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
            
            # Step 3: Deduplicate chunks by content hash
            unique_chunks = self._deduplicate_chunks(all_chunks)
            
            # Step 4: Update chunk indices after deduplication
            for i, chunk in enumerate(unique_chunks):
                chunk.chunk_index = i
            
            logger.info(
                "Document chunking completed",
                doc_name=doc_name,
                total_chunks=len(unique_chunks),
                unique_chunks=len(unique_chunks),
                avg_tokens_per_chunk=sum(c.token_count for c in unique_chunks) / len(unique_chunks) if unique_chunks else 0
            )
            
            return unique_chunks
            
        except Exception as e:
            error_msg = f"Failed to chunk document: {str(e)}"
            logger.error(error_msg, doc_name=doc_name)
            raise ChunkingError(error_msg)
    
    def _detect_document_structure(self, pages: List[PageContent]) -> List[Dict[str, Any]]:
        """Detect document structure for better chunking."""
        sections = []
        current_section = None
        
        # Patterns for section headers
        section_patterns = [
            r'^(\d+\.?\s+[A-Z][^.]*)',  # Numbered sections
            r'^([A-Z][A-Z\s]{3,}[A-Z])$',  # ALL CAPS headers
            r'^(SECTION\s+\d+)',  # Section X
            r'^(ARTICLE\s+[IVX\d]+)',  # Article Roman/numbers
            r'^(CHAPTER\s+\d+)',  # Chapter X
            r'^([A-Z]\.?\s+[A-Z][^.]*)',  # A. Header format
        ]
        
        for page in pages:
            lines = page.text.split('\n')
            page_sections = []
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped or len(line_stripped) < 3:
                    continue
                
                # Check if line matches section header patterns
                is_header = False
                header_text = None
                
                for pattern in section_patterns:
                    match = re.match(pattern, line_stripped)
                    if match:
                        is_header = True
                        header_text = match.group(1).strip()
                        break
                
                # Additional heuristics for headers
                if not is_header:
                    # Short lines in all caps might be headers
                    if (len(line_stripped) < 50 and 
                        line_stripped.isupper() and 
                        not re.search(r'\d{4}', line_stripped)):  # Avoid dates
                        is_header = True
                        header_text = line_stripped
                    
                    # Lines that are much shorter than surrounding text
                    elif (i > 0 and i < len(lines) - 1 and
                          len(line_stripped) < 30 and
                          len(lines[i-1].strip()) > 50 and
                          len(lines[i+1].strip()) > 50):
                        is_header = True
                        header_text = line_stripped
                
                if is_header:
                    page_sections.append({
                        'title': header_text,
                        'page_no': page.page_no,
                        'line_index': i
                    })
            
            # If no sections found on this page, continue current section
            if not page_sections:
                if current_section:
                    current_section['pages'].append(page)
                    current_section['end_page'] = page.page_no
                else:
                    # Start first section
                    current_section = {
                        'title': None,
                        'pages': [page],
                        'start_page': page.page_no,
                        'end_page': page.page_no
                    }
            else:
                # Close current section if exists
                if current_section:
                    sections.append(current_section)
                
                # Create new section
                current_section = {
                    'title': page_sections[0]['title'],
                    'pages': [page],
                    'start_page': page.page_no,
                    'end_page': page.page_no
                }
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        # If no structure detected, create single section
        if not sections or len(sections) == 1:
            sections = [{
                'title': None,
                'pages': pages,
                'start_page': pages[0].page_no,
                'end_page': pages[-1].page_no
            }]
        
        return sections
    
    def _chunk_section(
        self,
        section: Dict[str, Any],
        doc_id: str,
        doc_name: str,
        start_chunk_index: int
    ) -> List[Chunk]:
        """Create chunks from a document section."""
        section_title = section.get('title')
        section_pages = section['pages']
        
        # Combine all text from section pages
        full_text = ""
        page_boundaries = []  # Track where each page starts in the text
        
        for page in section_pages:
            page_start_pos = len(full_text)
            page_text = page.text.strip()
            
            if page_text:
                if full_text:
                    full_text += "\n\n"  # Separate pages
                    page_start_pos = len(full_text)
                
                full_text += page_text
                
                page_boundaries.append({
                    'page_no': page.page_no,
                    'start_pos': page_start_pos,
                    'end_pos': len(full_text),
                    'char_count': len(page_text)
                })
        
        if not full_text.strip():
            logger.debug("Section has no text content", section_title=section_title)
            return []
        
        # Create chunks using sliding window approach
        chunks = self._create_sliding_window_chunks(
            full_text, page_boundaries, doc_id, doc_name, 
            start_chunk_index, section_title
        )
        
        return chunks
    
    def _create_sliding_window_chunks(
        self,
        text: str,
        page_boundaries: List[Dict[str, Any]],
        doc_id: str,
        doc_name: str,
        start_chunk_index: int,
        section_title: Optional[str]
    ) -> List[Chunk]:
        """Create overlapping chunks using sliding window approach."""
        chunks = []
        chunk_index = start_chunk_index
        
        # Find optimal sentence/paragraph boundaries for chunking
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Group sentences into chunks
        current_chunk_sentences = []
        current_chunk_chars = 0
        overlap_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence_chars = len(sentence)
            
            # Check if adding this sentence would exceed max size
            if (current_chunk_chars + sentence_chars > self.target_chars * 1.5 and 
                current_chunk_sentences):
                
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk_sentences).strip()
                if chunk_text:
                    chunk = self._create_chunk_from_text(
                        chunk_text, text, page_boundaries, doc_id, doc_name,
                        chunk_index, section_title, overlap_sentences
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                
                # Calculate overlap sentences for next chunk
                total_chars = sum(len(s) for s in current_chunk_sentences)
                overlap_chars = int(total_chars * self.overlap_percent)
                overlap_sentences = []
                overlap_char_count = 0
                
                # Take sentences from the end for overlap
                for j in range(len(current_chunk_sentences) - 1, -1, -1):
                    sent = current_chunk_sentences[j]
                    if overlap_char_count + len(sent) <= overlap_chars:
                        overlap_sentences.insert(0, sent)
                        overlap_char_count += len(sent)
                    else:
                        break
                
                # Start new chunk with overlap + current sentence
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_chars = sum(len(s) for s in current_chunk_sentences)
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_chars += sentence_chars
            
            # If chunk reaches target size, create it
            if current_chunk_chars >= self.target_chars and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences).strip()
                if chunk_text:
                    chunk = self._create_chunk_from_text(
                        chunk_text, text, page_boundaries, doc_id, doc_name,
                        chunk_index, section_title, overlap_sentences
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                
                # Prepare overlap for next chunk
                total_chars = sum(len(s) for s in current_chunk_sentences)
                overlap_chars = int(total_chars * self.overlap_percent)
                new_overlap_sentences = []
                overlap_char_count = 0
                
                for j in range(len(current_chunk_sentences) - 1, -1, -1):
                    sent = current_chunk_sentences[j]
                    if overlap_char_count + len(sent) <= overlap_chars:
                        new_overlap_sentences.insert(0, sent)
                        overlap_char_count += len(sent)
                    else:
                        break
                
                overlap_sentences = new_overlap_sentences
                current_chunk_sentences = new_overlap_sentences.copy()
                current_chunk_chars = overlap_char_count
        
        # Create final chunk if there's remaining content
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text and len(chunk_text) >= self.min_chunk_tokens * self.chars_per_token:
                chunk = self._create_chunk_from_text(
                    chunk_text, text, page_boundaries, doc_id, doc_name,
                    chunk_index, section_title, overlap_sentences
                )
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking boundaries."""
        # Simple sentence splitting - can be improved with nltk/spacy
        sentence_endings = r'[.!?]+(?:\s+|$)'
        
        # Split on sentence boundaries but preserve the ending punctuation
        sentences = re.split(f'({sentence_endings})', text)
        
        # Combine sentences with their endings
        combined = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if i + 1 < len(sentences):
                sentence += sentences[i + 1].strip()
            
            if sentence:
                combined.append(sentence)
        
        # If no sentence boundaries found, split on paragraphs
        if len(combined) <= 1:
            paragraphs = text.split('\n\n')
            combined = [p.strip() for p in paragraphs if p.strip()]
        
        # If still no good splits, split on fixed character boundaries
        if len(combined) <= 1 and len(text) > self.target_chars:
            chunk_size = self.target_chars // 2  # Smaller pieces for overlap
            combined = []
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    combined.append(chunk.strip())
        
        return combined
    
    def _create_chunk_from_text(
        self,
        chunk_text: str,
        full_text: str,
        page_boundaries: List[Dict[str, Any]],
        doc_id: str,
        doc_name: str,
        chunk_index: int,
        section_title: Optional[str],
        overlap_sentences: List[str]
    ) -> Optional[Chunk]:
        """Create a Chunk object from text with proper metadata."""
        if not chunk_text or len(chunk_text.strip()) < 10:
            return None
        
        # Find position in full text
        chunk_start = full_text.find(chunk_text[:100])  # Use first 100 chars to find position
        if chunk_start == -1:
            chunk_start = 0  # Fallback
        
        chunk_end = chunk_start + len(chunk_text)
        
        # Determine page boundaries for this chunk
        page_start = None
        page_end = None
        
        for boundary in page_boundaries:
            # Check if chunk overlaps with this page
            if (chunk_start < boundary['end_pos'] and chunk_end > boundary['start_pos']):
                if page_start is None or boundary['page_no'] < page_start:
                    page_start = boundary['page_no']
                if page_end is None or boundary['page_no'] > page_end:
                    page_end = boundary['page_no']
        
        # Fallback to first and last pages if not found
        if page_start is None:
            page_start = page_boundaries[0]['page_no'] if page_boundaries else 1
        if page_end is None:
            page_end = page_boundaries[-1]['page_no'] if page_boundaries else 1
        
        # Calculate token count (rough estimate)
        token_count = max(1, len(chunk_text) // self.chars_per_token)
        
        # Generate MD5 hash
        text_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
        
        # Generate chunk ID
        chunk_id = str(uuid.uuid4())
        
        # Create overlap info
        overlap_info = {
            "has_overlap": len(overlap_sentences) > 0,
            "overlap_sentences": len(overlap_sentences),
            "overlap_chars": sum(len(s) for s in overlap_sentences) if overlap_sentences else 0
        }
        
        # Create metadata
        metadata = {
            "section_title": section_title,
            "chunk_start_pos": chunk_start,
            "chunk_end_pos": chunk_end,
            "sentence_count": len(chunk_text.split('.')),
            "paragraph_count": len(chunk_text.split('\n\n')),
            "creation_method": "sliding_window"
        }
        
        return Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            doc_id=doc_id,
            doc_name=doc_name,
            page_start=page_start,
            page_end=page_end,
            token_count=token_count,
            char_count=len(chunk_text),
            md5=text_hash,
            section_title=section_title,
            chunk_index=chunk_index,
            overlap_info=overlap_info,
            metadata=metadata
        )
    
    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove duplicate chunks based on MD5 hash."""
        seen_hashes: Set[str] = set()
        unique_chunks = []
        duplicates_removed = 0
        
        for chunk in chunks:
            if chunk.md5 not in seen_hashes:
                seen_hashes.add(chunk.md5)
                unique_chunks.append(chunk)
            else:
                duplicates_removed += 1
                logger.debug(
                    "Duplicate chunk removed",
                    chunk_id=chunk.chunk_id,
                    md5=chunk.md5,
                    doc_name=chunk.doc_name
                )
        
        if duplicates_removed > 0:
            logger.info(
                "Chunk deduplication completed",
                total_chunks=len(chunks),
                unique_chunks=len(unique_chunks),
                duplicates_removed=duplicates_removed
            )
        
        return unique_chunks
    
    def analyze_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Analyze chunk statistics for quality assessment."""
        if not chunks:
            return {
                "error": "No chunks provided",
                "total_chunks": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "avg_chars": 0,
                "avg_page_span": 0,
                "chunks_with_overlap": 0,
                "overlap_percentage": 0,
                "sections": {},
                "page_coverage": {"min_page": 0, "max_page": 0}
            }
        
        token_counts = [c.token_count for c in chunks]
        char_counts = [c.char_count for c in chunks]
        page_spans = [c.page_end - c.page_start + 1 for c in chunks]
        
        # Count chunks with overlaps
        overlapped_chunks = sum(1 for c in chunks if c.overlap_info.get("has_overlap", False))
        
        # Section distribution
        sections = defaultdict(int)
        for chunk in chunks:
            section = chunk.section_title or "No Section"
            sections[section] += 1
        
        return {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "avg_page_span": sum(page_spans) / len(page_spans),
            "chunks_with_overlap": overlapped_chunks,
            "overlap_percentage": (overlapped_chunks / len(chunks)) * 100,
            "sections": dict(sections),
            "page_coverage": {
                "min_page": min(c.page_start for c in chunks),
                "max_page": max(c.page_end for c in chunks)
            }
        }
    
    def save_chunks_jsonl(self, chunks: List[Chunk], output_path: Path) -> None:
        """Save chunks to JSONL format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import json
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    chunk_record = {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "doc_name": chunk.doc_name,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "token_count": chunk.token_count,
                        "char_count": chunk.char_count,
                        "md5": chunk.md5,
                        "section_title": chunk.section_title,
                        "chunk_index": chunk.chunk_index,
                        "overlap_info": chunk.overlap_info,
                        "metadata": chunk.metadata
                    }
                    f.write(json.dumps(chunk_record, ensure_ascii=False) + '\n')
            
            logger.info(
                "Chunks saved to JSONL",
                output_path=str(output_path),
                chunk_count=len(chunks)
            )
            
        except Exception as e:
            error_msg = f"Failed to save chunks: {str(e)}"
            logger.error(error_msg, output_path=str(output_path))
            raise ChunkingError(error_msg)