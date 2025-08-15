"""
Citation Management System for RAG Pipeline.

Provides enhanced citation tracking, validation, and correction
for accurate source attribution in construction claims analysis.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import re
from difflib import SequenceMatcher

from .logging_conf import get_logger
from .vectors import SearchResult
from .models import SourceChunk

logger = get_logger(__name__)


@dataclass
class CitationMapping:
    """Maps a citation to its source chunk with validation info."""
    citation: str
    source_chunk: SourceChunk
    is_valid: bool
    confidence: float
    page_match: bool
    doc_match: bool
    suggested_correction: Optional[str] = None


@dataclass
class CitationMetrics:
    """Citation quality metrics for a response."""
    total_citations: int
    valid_citations: int
    coverage_score: float  # % of key points with citations
    diversity_score: float  # Distribution across different documents
    accuracy_score: float  # % of citations that are valid
    completeness_score: float  # Overall citation quality


class CitationManager:
    """Enhanced citation tracking and validation system."""
    
    def __init__(self, fuzzy_threshold: float = 0.8):
        self.fuzzy_threshold = fuzzy_threshold
        self.citation_pattern = re.compile(r'\[([\w\-\s\.]+\.(?:pdf|PDF))\s+p\.(\d+(?:-\d+)?)\]')
        
    def extract_citations(self, answer: str) -> List[str]:
        """Extract all citations from answer text."""
        matches = self.citation_pattern.findall(answer)
        citations = []
        
        for doc_name, page_range in matches:
            citation = f"[{doc_name} p.{page_range}]"
            if citation not in citations:  # Avoid duplicates
                citations.append(citation)
        
        logger.debug("Citations extracted", count=len(citations), citations=citations)
        return citations
    
    def create_citation_mappings(
        self, 
        citations: List[str], 
        source_chunks: List[SourceChunk]
    ) -> List[CitationMapping]:
        """Create mappings between citations and source chunks with validation."""
        mappings = []
        
        # Create lookup for available sources
        source_lookup = self._build_source_lookup(source_chunks)
        
        for citation in citations:
            mapping = self._map_citation_to_source(citation, source_chunks, source_lookup)
            mappings.append(mapping)
        
        logger.debug(
            "Citation mappings created",
            total_citations=len(citations),
            valid_mappings=sum(1 for m in mappings if m.is_valid)
        )
        
        return mappings
    
    def _build_source_lookup(self, source_chunks: List[SourceChunk]) -> Dict[str, List[Tuple[int, int, SourceChunk]]]:
        """Build efficient lookup for source documents and pages."""
        lookup = {}
        
        for chunk in source_chunks:
            doc_key = chunk.doc.lower()
            if doc_key not in lookup:
                lookup[doc_key] = []
            
            # Store page range and chunk reference
            lookup[doc_key].append((chunk.page_start, chunk.page_end, chunk))
        
        return lookup
    
    def _map_citation_to_source(
        self, 
        citation: str, 
        source_chunks: List[SourceChunk],
        source_lookup: Dict[str, List[Tuple[int, int, SourceChunk]]]
    ) -> CitationMapping:
        """Map a single citation to its best matching source chunk."""
        
        # Parse citation
        match = self.citation_pattern.match(citation)
        if not match:
            logger.warning("Invalid citation format", citation=citation)
            return CitationMapping(
                citation=citation,
                source_chunk=source_chunks[0] if source_chunks else None,
                is_valid=False,
                confidence=0.0,
                page_match=False,
                doc_match=False
            )
        
        cited_doc, page_range_str = match.groups()
        cited_pages = self._parse_page_range(page_range_str)
        
        # Try exact document match first
        doc_key = cited_doc.lower()
        best_match = None
        best_confidence = 0.0
        
        if doc_key in source_lookup:
            # Exact document name match
            for page_start, page_end, chunk in source_lookup[doc_key]:
                confidence = self._calculate_page_overlap_confidence(
                    cited_pages, page_start, page_end
                )
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = chunk
        
        # If no exact match, try fuzzy document matching
        if best_match is None or best_confidence < 0.5:
            fuzzy_match, fuzzy_confidence = self._fuzzy_document_match(
                cited_doc, cited_pages, source_lookup
            )
            if fuzzy_confidence > best_confidence:
                best_match = fuzzy_match
                best_confidence = fuzzy_confidence
        
        # Create mapping
        if best_match:
            page_match = self._check_page_overlap(cited_pages, best_match.page_start, best_match.page_end)
            doc_match = self._check_document_match(cited_doc, best_match.doc)
            is_valid = page_match and doc_match and best_confidence >= 0.5
            
            suggested_correction = None
            if not is_valid and best_confidence > 0.3:
                suggested_correction = f"[{best_match.doc} p.{best_match.page_start}-{best_match.page_end}]"
            
            return CitationMapping(
                citation=citation,
                source_chunk=best_match,
                is_valid=is_valid,
                confidence=best_confidence,
                page_match=page_match,
                doc_match=doc_match,
                suggested_correction=suggested_correction
            )
        else:
            # No reasonable match found
            return CitationMapping(
                citation=citation,
                source_chunk=source_chunks[0] if source_chunks else None,
                is_valid=False,
                confidence=0.0,
                page_match=False,
                doc_match=False
            )
    
    def _parse_page_range(self, page_range_str: str) -> List[int]:
        """Parse page range string into list of page numbers."""
        try:
            if '-' in page_range_str:
                start, end = map(int, page_range_str.split('-'))
                return list(range(start, end + 1))
            else:
                return [int(page_range_str)]
        except ValueError:
            logger.warning("Invalid page range format", page_range=page_range_str)
            return []
    
    def _calculate_page_overlap_confidence(
        self, 
        cited_pages: List[int], 
        chunk_start: int, 
        chunk_end: int
    ) -> float:
        """Calculate confidence based on page overlap."""
        if not cited_pages:
            return 0.0
        
        chunk_pages = set(range(chunk_start, chunk_end + 1))
        cited_pages_set = set(cited_pages)
        
        # Calculate overlap
        overlap = len(chunk_pages.intersection(cited_pages_set))
        total_cited = len(cited_pages_set)
        
        if total_cited == 0:
            return 0.0
        
        # Confidence based on percentage overlap
        return overlap / total_cited
    
    def _check_page_overlap(self, cited_pages: List[int], chunk_start: int, chunk_end: int) -> bool:
        """Check if there's any page overlap between citation and chunk."""
        if not cited_pages:
            return False
        
        chunk_pages = set(range(chunk_start, chunk_end + 1))
        cited_pages_set = set(cited_pages)
        
        return len(chunk_pages.intersection(cited_pages_set)) > 0
    
    def _check_document_match(self, cited_doc: str, chunk_doc: str) -> bool:
        """Check if document names match (with fuzzy matching)."""
        # Exact match
        if cited_doc.lower() == chunk_doc.lower():
            return True
        
        # Fuzzy match
        similarity = SequenceMatcher(None, cited_doc.lower(), chunk_doc.lower()).ratio()
        return similarity >= self.fuzzy_threshold
    
    def _fuzzy_document_match(
        self, 
        cited_doc: str, 
        cited_pages: List[int],
        source_lookup: Dict[str, List[Tuple[int, int, SourceChunk]]]
    ) -> Tuple[Optional[SourceChunk], float]:
        """Find best fuzzy match for document name."""
        best_match = None
        best_confidence = 0.0
        
        for doc_key, page_chunks in source_lookup.items():
            # Calculate document name similarity
            doc_similarity = SequenceMatcher(None, cited_doc.lower(), doc_key).ratio()
            
            if doc_similarity >= self.fuzzy_threshold:
                # Find best page match within this document
                for page_start, page_end, chunk in page_chunks:
                    page_confidence = self._calculate_page_overlap_confidence(
                        cited_pages, page_start, page_end
                    )
                    
                    # Combined confidence: document similarity + page overlap
                    combined_confidence = (doc_similarity + page_confidence) / 2
                    
                    if combined_confidence > best_confidence:
                        best_confidence = combined_confidence
                        best_match = chunk
        
        return best_match, best_confidence
    
    def calculate_citation_metrics(
        self, 
        answer: str, 
        citation_mappings: List[CitationMapping],
        source_chunks: List[SourceChunk]
    ) -> CitationMetrics:
        """Calculate comprehensive citation quality metrics."""
        
        total_citations = len(citation_mappings)
        valid_citations = sum(1 for mapping in citation_mappings if mapping.is_valid)
        
        # Calculate accuracy score
        accuracy_score = valid_citations / total_citations if total_citations > 0 else 0.0
        
        # Calculate coverage score (estimate based on key sentences)
        coverage_score = self._calculate_coverage_score(answer, citation_mappings)
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(citation_mappings, source_chunks)
        
        # Calculate completeness score (weighted combination)
        completeness_score = (
            0.4 * accuracy_score +
            0.3 * coverage_score +
            0.3 * diversity_score
        )
        
        metrics = CitationMetrics(
            total_citations=total_citations,
            valid_citations=valid_citations,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score
        )
        
        logger.debug("Citation metrics calculated", metrics=metrics)
        return metrics
    
    def _calculate_coverage_score(self, answer: str, citation_mappings: List[CitationMapping]) -> float:
        """Estimate what percentage of key statements are cited."""
        
        # Split answer into sentences and identify key point sections
        sentences = self._extract_key_sentences(answer)
        if not sentences:
            return 0.0
        
        # Count sentences with citations
        cited_sentences = 0
        for sentence in sentences:
            if any(mapping.citation in sentence for mapping in citation_mappings):
                cited_sentences += 1
        
        return cited_sentences / len(sentences)
    
    def _extract_key_sentences(self, answer: str) -> List[str]:
        """Extract key sentences that should typically have citations."""
        
        # Focus on sentences in Key Points and Analysis sections
        key_sections = ['## Key Points', '## Analysis']
        key_sentences = []
        
        current_section = None
        for line in answer.split('\n'):
            line = line.strip()
            
            # Check if we're entering a key section
            if any(section in line for section in key_sections):
                current_section = line
                continue
            
            # If we're in a key section and line has content
            if current_section and line and not line.startswith('##'):
                # Split into sentences
                sentences = re.split(r'[.!?]+', line)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:  # Meaningful sentence length
                        key_sentences.append(sentence)
        
        return key_sentences
    
    def _calculate_diversity_score(
        self, 
        citation_mappings: List[CitationMapping], 
        source_chunks: List[SourceChunk]
    ) -> float:
        """Calculate how well citations are distributed across different documents."""
        
        if not citation_mappings or not source_chunks:
            return 0.0
        
        # Count unique documents in citations vs available documents
        cited_docs = set()
        for mapping in citation_mappings:
            if mapping.is_valid and mapping.source_chunk:
                cited_docs.add(mapping.source_chunk.doc.lower())
        
        available_docs = set(chunk.doc.lower() for chunk in source_chunks)
        
        if not available_docs:
            return 0.0
        
        # Diversity score: what fraction of available documents are cited
        return len(cited_docs) / len(available_docs)
    
    def suggest_citation_improvements(
        self, 
        citation_mappings: List[CitationMapping]
    ) -> List[str]:
        """Suggest improvements for citation quality."""
        suggestions = []
        
        invalid_citations = [m for m in citation_mappings if not m.is_valid]
        
        for mapping in invalid_citations:
            if mapping.suggested_correction:
                suggestions.append(
                    f"Consider changing '{mapping.citation}' to '{mapping.suggested_correction}'"
                )
            elif not mapping.doc_match:
                suggestions.append(
                    f"Document name in '{mapping.citation}' may be incorrect"
                )
            elif not mapping.page_match:
                suggestions.append(
                    f"Page number in '{mapping.citation}' may be incorrect"
                )
        
        if len(invalid_citations) > len(citation_mappings) * 0.5:
            suggestions.append("Consider reviewing source materials for more accurate citations")
        
        return suggestions