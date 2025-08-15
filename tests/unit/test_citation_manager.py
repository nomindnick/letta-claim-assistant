"""
Unit tests for the Citation Manager module.

Tests citation extraction, validation, mapping, and quality metrics
for the enhanced RAG citation system.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from app.citation_manager import (
    CitationManager, CitationMapping, CitationMetrics
)
from app.rag import SourceChunk


class TestCitationManager:
    """Test suite for CitationManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.citation_manager = CitationManager(fuzzy_threshold=0.8)
        
        # Sample source chunks
        self.source_chunks = [
            SourceChunk(
                doc="Contract_2023.pdf",
                page_start=5,
                page_end=7,
                text="The contractor shall complete all work in accordance with specifications...",
                score=0.85
            ),
            SourceChunk(
                doc="Daily_Log_Feb15.pdf", 
                page_start=2,
                page_end=2,
                text="Concrete pour delayed due to weather conditions...",
                score=0.78
            ),
            SourceChunk(
                doc="Specification_Section3.pdf",
                page_start=10,
                page_end=12,
                text="All structural steel shall meet AISC standards...",
                score=0.92
            )
        ]
    
    def test_extract_citations_basic(self):
        """Test basic citation extraction."""
        answer = """
        ## Key Points
        - The contract requires completion by June 2023 [Contract_2023.pdf p.5]
        - Weather delays occurred on February 15th [Daily_Log_Feb15.pdf p.2]
        
        ## Analysis
        Multiple issues arose including specifications [Specification_Section3.pdf p.10-12]
        
        ## Citations
        - [Contract_2023.pdf p.5] - Contract completion date
        - [Daily_Log_Feb15.pdf p.2] - Weather delay documentation
        """
        
        citations = self.citation_manager.extract_citations(answer)
        
        expected_citations = [
            "[Contract_2023.pdf p.5]",
            "[Daily_Log_Feb15.pdf p.2]", 
            "[Specification_Section3.pdf p.10-12]"
        ]
        
        assert len(citations) == 3
        assert all(citation in citations for citation in expected_citations)
    
    def test_extract_citations_duplicates(self):
        """Test that duplicate citations are removed."""
        answer = """
        The contract states [Contract_2023.pdf p.5] and further clarifies [Contract_2023.pdf p.5].
        Additional details in [Contract_2023.pdf p.5] support this.
        """
        
        citations = self.citation_manager.extract_citations(answer)
        
        assert len(citations) == 1
        assert citations[0] == "[Contract_2023.pdf p.5]"
    
    def test_extract_citations_various_formats(self):
        """Test extraction of various citation formats."""
        answer = """
        References include [Contract_2023.pdf p.5], [Daily_Log_Feb15.PDF p.2-3],
        and [Spec Section 3.pdf p.10].
        """
        
        citations = self.citation_manager.extract_citations(answer)
        
        expected = [
            "[Contract_2023.pdf p.5]",
            "[Daily_Log_Feb15.PDF p.2-3]",
            "[Spec Section 3.pdf p.10]"
        ]
        
        assert len(citations) == 3
        assert all(citation in citations for citation in expected)
    
    def test_create_citation_mappings_exact_match(self):
        """Test citation mapping with exact document matches."""
        citations = [
            "[Contract_2023.pdf p.5]",
            "[Daily_Log_Feb15.pdf p.2]"
        ]
        
        mappings = self.citation_manager.create_citation_mappings(citations, self.source_chunks)
        
        assert len(mappings) == 2
        
        # Check first mapping
        contract_mapping = next(m for m in mappings if "Contract_2023.pdf" in m.citation)
        assert contract_mapping.is_valid is True
        assert contract_mapping.doc_match is True
        assert contract_mapping.page_match is True
        assert contract_mapping.confidence >= 0.5
        
        # Check second mapping
        log_mapping = next(m for m in mappings if "Daily_Log_Feb15.pdf" in m.citation)
        assert log_mapping.is_valid is True
        assert log_mapping.doc_match is True
        assert log_mapping.page_match is True
    
    def test_create_citation_mappings_page_mismatch(self):
        """Test citation mapping with page mismatches."""
        citations = [
            "[Contract_2023.pdf p.15]",  # Page not in source (which has 5-7)
        ]
        
        mappings = self.citation_manager.create_citation_mappings(citations, self.source_chunks)
        
        assert len(mappings) == 1
        mapping = mappings[0]
        
        assert mapping.doc_match is True  # Document name matches
        assert mapping.page_match is False  # Page doesn't match
        assert mapping.is_valid is False  # Overall invalid
        assert mapping.suggested_correction is not None
    
    def test_create_citation_mappings_fuzzy_document_match(self):
        """Test fuzzy document name matching."""
        # Create citation with slightly different document name
        citations = [
            "[Contract 2023.pdf p.6]"  # Missing underscore from Contract_2023.pdf
        ]
        
        mappings = self.citation_manager.create_citation_mappings(citations, self.source_chunks)
        
        assert len(mappings) == 1
        mapping = mappings[0]
        
        # Should match with fuzzy matching
        assert mapping.confidence > 0.0
        assert mapping.source_chunk.doc == "Contract_2023.pdf"
    
    def test_parse_page_range(self):
        """Test page range parsing."""
        # Single page
        pages = self.citation_manager._parse_page_range("5")
        assert pages == [5]
        
        # Page range
        pages = self.citation_manager._parse_page_range("10-12")
        assert pages == [10, 11, 12]
        
        # Invalid format
        pages = self.citation_manager._parse_page_range("invalid")
        assert pages == []
    
    def test_calculate_page_overlap_confidence(self):
        """Test page overlap confidence calculation."""
        # Full overlap
        confidence = self.citation_manager._calculate_page_overlap_confidence([5, 6], 5, 7)
        assert confidence == 1.0  # Both cited pages in chunk range
        
        # Partial overlap
        confidence = self.citation_manager._calculate_page_overlap_confidence([5, 8], 5, 7)
        assert confidence == 0.5  # Only page 5 overlaps
        
        # No overlap
        confidence = self.citation_manager._calculate_page_overlap_confidence([10, 11], 5, 7)
        assert confidence == 0.0
    
    def test_calculate_citation_metrics(self):
        """Test comprehensive citation metrics calculation."""
        answer = """
        ## Key Points
        - Contract specifies June completion [Contract_2023.pdf p.5]
        - Weather caused delays [Daily_Log_Feb15.pdf p.2]
        - Steel specifications require AISC compliance [Specification_Section3.pdf p.10]
        
        ## Analysis
        The evidence clearly shows multiple factors contributing to project delays.
        
        ## Citations
        - [Contract_2023.pdf p.5] - Completion deadline
        - [Daily_Log_Feb15.pdf p.2] - Weather delay record
        """
        
        citations = [
            "[Contract_2023.pdf p.5]",
            "[Daily_Log_Feb15.pdf p.2]",
            "[Specification_Section3.pdf p.10]"
        ]
        
        mappings = self.citation_manager.create_citation_mappings(citations, self.source_chunks)
        metrics = self.citation_manager.calculate_citation_metrics(answer, mappings, self.source_chunks)
        
        assert isinstance(metrics, CitationMetrics)
        assert metrics.total_citations == 3
        assert metrics.accuracy_score >= 0.0
        assert metrics.coverage_score >= 0.0
        assert metrics.diversity_score >= 0.0
        assert 0.0 <= metrics.completeness_score <= 1.0
    
    def test_calculate_coverage_score(self):
        """Test citation coverage calculation."""
        # Answer with good citation coverage
        answer_with_citations = """
        ## Key Points
        - First point with citation [Doc1.pdf p.1]
        - Second point with citation [Doc2.pdf p.2]
        
        ## Analysis
        Analysis with citation [Doc1.pdf p.1] supporting the conclusion.
        """
        
        mappings = [
            CitationMapping(
                citation="[Doc1.pdf p.1]",
                source_chunk=self.source_chunks[0],
                is_valid=True,
                confidence=0.9,
                page_match=True,
                doc_match=True
            )
        ]
        
        coverage = self.citation_manager._calculate_coverage_score(answer_with_citations, mappings)
        assert coverage > 0.0
        
        # Answer without citations
        answer_no_citations = """
        ## Key Points
        - Unsupported claim
        - Another unsupported claim
        """
        
        coverage_none = self.citation_manager._calculate_coverage_score(answer_no_citations, [])
        assert coverage_none == 0.0
    
    def test_calculate_diversity_score(self):
        """Test citation diversity calculation."""
        # High diversity - citations from different documents
        diverse_mappings = [
            CitationMapping("[Contract_2023.pdf p.5]", self.source_chunks[0], True, 0.9, True, True),
            CitationMapping("[Daily_Log_Feb15.pdf p.2]", self.source_chunks[1], True, 0.8, True, True),
            CitationMapping("[Specification_Section3.pdf p.10]", self.source_chunks[2], True, 0.9, True, True)
        ]
        
        diversity = self.citation_manager._calculate_diversity_score(diverse_mappings, self.source_chunks)
        assert diversity == 1.0  # All available documents cited
        
        # Low diversity - citations from same document
        same_doc_mappings = [
            CitationMapping("[Contract_2023.pdf p.5]", self.source_chunks[0], True, 0.9, True, True),
            CitationMapping("[Contract_2023.pdf p.6]", self.source_chunks[0], True, 0.8, True, True)
        ]
        
        diversity_low = self.citation_manager._calculate_diversity_score(same_doc_mappings, self.source_chunks)
        assert diversity_low < diversity  # Lower diversity score
    
    def test_suggest_citation_improvements(self):
        """Test citation improvement suggestions."""
        # Create mappings with various issues
        mappings = [
            CitationMapping("[Contract_2023.pdf p.5]", self.source_chunks[0], True, 0.9, True, True),  # Valid
            CitationMapping("[Wrong_Doc.pdf p.1]", self.source_chunks[1], False, 0.3, False, False,
                           suggested_correction="[Daily_Log_Feb15.pdf p.2]"),  # Invalid with suggestion
            CitationMapping("[Contract_2023.pdf p.99]", self.source_chunks[0], False, 0.5, False, True)  # Page mismatch
        ]
        
        suggestions = self.citation_manager.suggest_citation_improvements(mappings)
        
        assert len(suggestions) >= 1
        assert any("Wrong_Doc.pdf" in suggestion for suggestion in suggestions)
        assert any("Daily_Log_Feb15.pdf" in suggestion for suggestion in suggestions)
    
    def test_extract_key_sentences(self):
        """Test extraction of key sentences for coverage analysis."""
        answer = """
        ## Key Points
        - This is a key finding that should be cited.
        - Another important point here.
        - Short.
        
        ## Analysis
        This section contains analysis that should have citations.
        
        ## Other Section
        This section is not considered for citation coverage.
        """
        
        sentences = self.citation_manager._extract_key_sentences(answer)
        
        # Should extract sentences from Key Points and Analysis sections
        assert len(sentences) >= 2
        assert any("key finding" in sentence.lower() for sentence in sentences)
        assert any("analysis" in sentence.lower() for sentence in sentences)
        # Should not include very short sentences
        assert not any(sentence.strip() == "Short" for sentence in sentences)
    
    def test_fuzzy_document_match(self):
        """Test fuzzy document matching algorithm."""
        source_lookup = self.citation_manager._build_source_lookup(self.source_chunks)
        
        # Test close match
        match, confidence = self.citation_manager._fuzzy_document_match(
            "Contract 2023.pdf",  # Missing underscore
            [5],
            source_lookup
        )
        
        assert match is not None
        assert confidence > 0.5
        assert match.doc == "Contract_2023.pdf"
        
        # Test very different document name
        match_none, confidence_low = self.citation_manager._fuzzy_document_match(
            "Completely_Different_Doc.pdf",
            [1],
            source_lookup
        )
        
        assert confidence_low < 0.8  # Below fuzzy threshold
    
    def test_citation_manager_integration(self):
        """Test full citation manager integration."""
        answer = """
        ## Key Points  
        - Contract completion required by June [Contract_2023.pdf p.5]
        - Weather delays documented [Daily_Log_Feb15.pdf p.2]
        - Specifications need review [Specification_Section3.pdf p.11]
        
        ## Analysis
        The project faces multiple challenges requiring immediate attention.
        Additional documentation may be needed [Contract_2023.pdf p.6].
        
        ## Citations
        - [Contract_2023.pdf p.5] - Completion deadline
        - [Daily_Log_Feb15.pdf p.2] - Weather documentation
        """
        
        # Extract citations
        citations = self.citation_manager.extract_citations(answer)
        assert len(citations) >= 3
        
        # Create mappings
        mappings = self.citation_manager.create_citation_mappings(citations, self.source_chunks)
        assert len(mappings) == len(citations)
        
        # Calculate metrics
        metrics = self.citation_manager.calculate_citation_metrics(answer, mappings, self.source_chunks)
        assert metrics.total_citations == len(citations)
        assert 0.0 <= metrics.completeness_score <= 1.0
        
        # Get improvement suggestions
        suggestions = self.citation_manager.suggest_citation_improvements(mappings)
        assert isinstance(suggestions, list)


@pytest.fixture
def sample_answer_with_citations():
    """Sample answer with various citation patterns."""
    return """
    ## Key Points
    - The contract specifies completion by June 30, 2023 [Contract_2023.pdf p.5]
    - Daily logs show weather delays on February 15th [Daily_Log_Feb15.pdf p.2]
    - Structural specifications require AISC compliance [Specification_Section3.pdf p.10-12]
    
    ## Analysis
    Based on the documentation review, multiple factors contributed to project delays.
    The contract terms [Contract_2023.pdf p.5-7] clearly outline completion requirements,
    while the daily logs [Daily_Log_Feb15.pdf p.2] document specific delay events.
    
    ## Citations
    - [Contract_2023.pdf p.5] - Project completion deadline
    - [Daily_Log_Feb15.pdf p.2] - Weather delay documentation
    - [Specification_Section3.pdf p.10-12] - Structural requirements
    """


class TestCitationMappingDataClass:
    """Test CitationMapping data class functionality."""
    
    def test_citation_mapping_creation(self):
        """Test CitationMapping creation and attributes."""
        mapping = CitationMapping(
            citation="[Test.pdf p.1]",
            source_chunk=SourceChunk("Test.pdf", 1, 1, "Test content", 0.8),
            is_valid=True,
            confidence=0.9,
            page_match=True,
            doc_match=True,
            suggested_correction="[Test.pdf p.1-2]"
        )
        
        assert mapping.citation == "[Test.pdf p.1]"
        assert mapping.is_valid is True
        assert mapping.confidence == 0.9
        assert mapping.suggested_correction == "[Test.pdf p.1-2]"


class TestCitationMetricsDataClass:
    """Test CitationMetrics data class functionality."""
    
    def test_citation_metrics_creation(self):
        """Test CitationMetrics creation and validation."""
        metrics = CitationMetrics(
            total_citations=5,
            valid_citations=4,
            coverage_score=0.8,
            diversity_score=0.7,
            accuracy_score=0.8,
            completeness_score=0.75
        )
        
        assert metrics.total_citations == 5
        assert metrics.valid_citations == 4
        assert metrics.accuracy_score == 0.8
        assert 0.0 <= metrics.completeness_score <= 1.0