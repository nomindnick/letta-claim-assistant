"""
Unit tests for the Quality Metrics module.

Tests response quality analysis, metrics calculation, and
quality validation for RAG responses.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import List

from app.quality_metrics import (
    QualityAnalyzer, ResponseQualityMetrics, QualityThresholds,
    HistoricalQualityStats
)
from app.citation_manager import CitationMapping, CitationMetrics
from app.followup_engine import FollowupSuggestion, FollowupCategory
from app.models import SourceChunk
from app.models import KnowledgeItem


class TestQualityAnalyzer:
    """Test suite for QualityAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.thresholds = QualityThresholds(
            minimum_citation_coverage=0.6,
            minimum_source_diversity=0.4,
            minimum_answer_completeness=0.7,
            minimum_confidence_score=0.5
        )
        self.analyzer = QualityAnalyzer(self.thresholds)
        
        # Sample data
        self.source_chunks = [
            SourceChunk("Contract_2023.pdf", 5, 7, "Contract text content...", 0.85),
            SourceChunk("Daily_Log_Feb15.pdf", 2, 2, "Daily log entry content...", 0.78),
            SourceChunk("Spec_Section3.pdf", 10, 12, "Specification requirements...", 0.92)
        ]
        
        self.citation_mappings = [
            CitationMapping("[Contract_2023.pdf p.5]", self.source_chunks[0], True, 0.9, True, True),
            CitationMapping("[Daily_Log_Feb15.pdf p.2]", self.source_chunks[1], True, 0.8, True, True),
            CitationMapping("[Spec_Section3.pdf p.10]", self.source_chunks[2], False, 0.3, False, True)
        ]
        
        self.citation_metrics = CitationMetrics(
            total_citations=3,
            valid_citations=2,
            coverage_score=0.75,
            diversity_score=0.8,
            accuracy_score=0.67,
            completeness_score=0.7
        )
        
        self.followup_suggestions = [
            FollowupSuggestion(
                "What contract provisions address delay claims?",
                FollowupCategory.LEGAL, 0.8, "Legal analysis needed", ["Owner", "Contractor"], False
            ),
            FollowupSuggestion(
                "Should we engage a scheduling expert?",
                FollowupCategory.TECHNICAL, 0.7, "Technical expertise required", [], True
            )
        ]
        
        self.memory_items = [
            KnowledgeItem(
                type="Event", label="Project Delay", date="2023-02-15",
                actors=["Contractor"], doc_refs=[{"doc": "Daily_Log_Feb15.pdf", "page": 2}]
            )
        ]
    
    def test_analyze_answer_completeness_good(self):
        """Test answer completeness analysis for good answer."""
        query = "What caused the project delays?"
        answer = """
        ## Key Points
        - Weather conditions caused significant delays in February
        - Contract specifications were unclear regarding timeline requirements
        - Additional approvals were needed from the planning department
        
        ## Analysis
        The project delays resulted from multiple interconnected factors including
        external weather conditions, internal specification issues, and regulatory
        approval processes. The contract should have included provisions for
        weather-related delays and clearer timelines for approval processes.
        
        ## Citations
        - [Daily_Log_Feb15.pdf p.2] - Weather delay documentation
        - [Contract_2023.pdf p.5] - Timeline specifications
        """
        
        completeness = self.analyzer._analyze_answer_completeness(query, answer)
        
        assert completeness >= 0.7  # Should be high for well-structured answer
        # Should find required sections
        assert "key points" in answer.lower()
        assert "analysis" in answer.lower()
        assert "citations" in answer.lower()
    
    def test_analyze_answer_completeness_poor(self):
        """Test answer completeness analysis for poor answer."""
        query = "What caused the delays?"
        answer = "There were some issues with the weather."
        
        completeness = self.analyzer._analyze_answer_completeness(query, answer)
        
        assert completeness < 0.5  # Should be low for minimal answer
    
    def test_analyze_content_coherence_good(self):
        """Test content coherence analysis for well-structured content."""
        answer = """
        ## Key Points
        - First, the weather conditions deteriorated significantly
        - Subsequently, this led to delays in concrete work
        - Therefore, the project timeline was impacted
        
        ## Analysis
        The evidence shows a clear sequence of events. Initially, weather reports
        indicated favorable conditions. However, unexpected storms developed,
        which consequently affected all outdoor operations. As a result, the
        contractor had to reschedule critical activities.
        
        ## Conclusion
        In conclusion, the delays were unavoidable given the circumstances.
        """
        
        coherence = self.analyzer._analyze_content_coherence(answer)
        
        assert coherence >= 0.6  # Should be high for coherent content
    
    def test_analyze_content_coherence_poor(self):
        """Test content coherence analysis for poorly structured content."""
        answer = """
        Weather. Bad. Delays happened. Contract says things. 
        Money issues. Time problems. Weather again. Maybe problems.
        Weather. Contract. Weather. Contract. Same things repeated.
        """
        
        coherence = self.analyzer._analyze_content_coherence(answer)
        
        assert coherence < 0.5  # Should be low for incoherent content
    
    def test_analyze_domain_specificity(self):
        """Test domain specificity analysis."""
        # High domain specificity answer
        domain_answer = """
        The liquidated damages clause in Section 3.2 specifies that delays beyond
        substantial completion will result in penalties. The critical path analysis
        shows that the concrete pour activities were on the project's critical path.
        The contractor failed to provide proper notice under the changed conditions
        clause, which may affect their ability to claim time extensions.
        """
        
        domain_score = self.analyzer._analyze_domain_specificity(domain_answer)
        assert domain_score >= 0.6
        
        # Low domain specificity answer
        generic_answer = """
        There were some problems with the project. Things didn't go as planned.
        People were not happy about the situation. Money was involved.
        """
        
        generic_score = self.analyzer._analyze_domain_specificity(generic_answer)
        assert generic_score < domain_score
    
    def test_check_contextual_term_usage(self):
        """Test contextual term usage checking."""
        answer = """
        The critical path analysis shows that concrete activities are
        on the schedule's critical path, affecting project completion.
        The liquidated damages clause specifies predetermined penalties
        for delays beyond substantial completion.
        """
        
        terms = ["critical path", "liquidated damages", "substantial completion"]
        contextual_score = self.analyzer._check_contextual_term_usage(answer, terms)
        
        assert contextual_score > 0.5  # Should find good contextual usage
    
    def test_analyze_source_diversity(self):
        """Test source diversity analysis."""
        # High diversity - different document types
        diverse_sources = [
            SourceChunk("Contract_Agreement.pdf", 1, 2, "Contract text", 0.8),
            SourceChunk("Daily_Log_Feb15.pdf", 5, 5, "Log entry", 0.7),
            SourceChunk("Specification_Structural.pdf", 10, 12, "Spec text", 0.9),
            SourceChunk("Email_Correspondence.pdf", 1, 1, "Email text", 0.6)
        ]
        
        diversity = self.analyzer._analyze_source_diversity(diverse_sources)
        assert diversity >= 0.5
        
        # Low diversity - same document
        same_doc_sources = [
            SourceChunk("Contract.pdf", 1, 2, "Text 1", 0.8),
            SourceChunk("Contract.pdf", 3, 4, "Text 2", 0.7),
            SourceChunk("Contract.pdf", 5, 6, "Text 3", 0.6)
        ]
        
        low_diversity = self.analyzer._analyze_source_diversity(same_doc_sources)
        assert low_diversity < diversity
    
    def test_analyze_source_relevance(self):
        """Test source relevance analysis."""
        query = "What contract provisions address delay claims?"
        
        # Relevant sources
        relevant_sources = [
            SourceChunk("Contract.pdf", 1, 2, "delay provisions and claims procedures", 0.9),
            SourceChunk("Contract.pdf", 5, 6, "contract terms for delays", 0.8)
        ]
        
        relevance = self.analyzer._analyze_source_relevance(relevant_sources, query)
        assert relevance >= 0.5
        
        # Irrelevant sources
        irrelevant_sources = [
            SourceChunk("Other.pdf", 1, 1, "completely unrelated content", 0.3),
            SourceChunk("Random.pdf", 2, 2, "nothing about the topic", 0.2)
        ]
        
        low_relevance = self.analyzer._analyze_source_relevance(irrelevant_sources, query)
        assert low_relevance < relevance
    
    def test_analyze_followup_relevance(self):
        """Test follow-up relevance analysis."""
        query = "What caused the structural failures?"
        answer = "Structural failures were caused by design defects and poor construction practices."
        
        # Relevant follow-ups
        relevant_followups = [
            FollowupSuggestion(
                "Should we engage a structural engineer for analysis?",
                FollowupCategory.TECHNICAL, 0.8, "Technical expertise", [], True
            ),
            FollowupSuggestion(
                "What design standards were violated?",
                FollowupCategory.LEGAL, 0.7, "Legal compliance", [], False
            )
        ]
        
        relevance = self.analyzer._analyze_followup_relevance(relevant_followups, query, answer)
        assert relevance >= 0.4
        
        # Irrelevant follow-ups
        irrelevant_followups = [
            FollowupSuggestion(
                "What color should we paint the building?",
                FollowupCategory.RESPONSIBILITY, 0.2, "Unrelated", [], False
            )
        ]
        
        low_relevance = self.analyzer._analyze_followup_relevance(irrelevant_followups, query, answer)
        assert low_relevance < relevance
    
    def test_analyze_followup_diversity(self):
        """Test follow-up diversity analysis."""
        # Diverse follow-ups
        diverse_followups = [
            FollowupSuggestion("Legal question?", FollowupCategory.LEGAL, 0.8, "Legal", [], False),
            FollowupSuggestion("Technical question?", FollowupCategory.TECHNICAL, 0.7, "Tech", [], True),
            FollowupSuggestion("Evidence question?", FollowupCategory.EVIDENCE, 0.6, "Evidence", [], False)
        ]
        
        diversity = self.analyzer._analyze_followup_diversity(diverse_followups)
        assert diversity >= 0.5
        
        # Similar follow-ups (low diversity)
        similar_followups = [
            FollowupSuggestion("What contract provisions apply?", FollowupCategory.LEGAL, 0.8, "Legal", [], False),
            FollowupSuggestion("What contract terms are relevant?", FollowupCategory.LEGAL, 0.7, "Legal", [], False)
        ]
        
        low_diversity = self.analyzer._analyze_followup_diversity(similar_followups)
        assert low_diversity < diversity
    
    def test_analyze_followup_actionability(self):
        """Test follow-up actionability analysis."""
        # Actionable follow-ups
        actionable = [
            FollowupSuggestion("What documentation should we request?", FollowupCategory.EVIDENCE, 0.8, "", [], False),
            FollowupSuggestion("Should we engage a structural expert?", FollowupCategory.TECHNICAL, 0.7, "", [], True),
            FollowupSuggestion("How do we calculate delay damages?", FollowupCategory.DAMAGES, 0.6, "", [], False)
        ]
        
        actionability = self.analyzer._analyze_followup_actionability(actionable)
        assert actionability >= 0.5
        
        # Non-actionable follow-ups
        non_actionable = [
            FollowupSuggestion("Things are complicated", FollowupCategory.RESPONSIBILITY, 0.3, "", [], False),
            FollowupSuggestion("Maybe we should think about stuff", FollowupCategory.RESPONSIBILITY, 0.2, "", [], False)
        ]
        
        low_actionability = self.analyzer._analyze_followup_actionability(non_actionable)
        assert low_actionability < actionability
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        # High confidence scenario
        high_confidence = self.analyzer._calculate_confidence_score(
            citation_accuracy=0.9,
            source_relevance=0.8,
            content_coherence=0.85,
            answer_completeness=0.8
        )
        
        assert high_confidence >= 0.7
        
        # Low confidence scenario
        low_confidence = self.analyzer._calculate_confidence_score(
            citation_accuracy=0.3,
            source_relevance=0.4,
            content_coherence=0.5,
            answer_completeness=0.4
        )
        
        assert low_confidence < high_confidence
        assert low_confidence < 0.6  # Should apply penalties
    
    def test_calculate_overall_quality(self):
        """Test overall quality score calculation."""
        quality = self.analyzer._calculate_overall_quality(
            citation_coverage=0.8,
            citation_accuracy=0.9,
            citation_diversity=0.7,
            answer_completeness=0.8,
            content_coherence=0.85,
            domain_specificity=0.7,
            source_diversity=0.6,
            source_relevance=0.8,
            source_recency=0.5,
            followup_relevance=0.7,
            followup_diversity=0.6,
            followup_actionability=0.8
        )
        
        assert 0.0 <= quality <= 1.0
        assert quality >= 0.6  # Should be reasonably high for good inputs
    
    def test_validate_quality_standards(self):
        """Test quality standards validation."""
        # Meets standards
        meets_standards, requires_regen, warnings = self.analyzer._validate_quality_standards(
            overall_quality=0.8,
            citation_coverage=0.7,
            source_diversity=0.5,
            answer_completeness=0.8,
            confidence_score=0.6
        )
        
        assert meets_standards is True
        assert requires_regen is False
        assert len(warnings) == 0
        
        # Below standards
        fails_standards, needs_regen, warning_msgs = self.analyzer._validate_quality_standards(
            overall_quality=0.3,
            citation_coverage=0.4,  # Below threshold
            source_diversity=0.2,   # Below threshold
            answer_completeness=0.5, # Below threshold
            confidence_score=0.3     # Below threshold
        )
        
        assert fails_standards is False
        assert needs_regen is True
        assert len(warning_msgs) > 0
    
    def test_analyze_response_quality_integration(self):
        """Test full response quality analysis integration."""
        query = "What caused the construction delays?"
        answer = """
        ## Key Points
        - Weather conditions in February caused significant delays [Daily_Log_Feb15.pdf p.2]
        - Contract specifications were unclear regarding timeline [Contract_2023.pdf p.5]
        
        ## Analysis
        The construction delays resulted from multiple factors including adverse
        weather conditions and ambiguous contract language. The critical path
        analysis shows these delays affected the project completion schedule.
        
        ## Citations
        - [Daily_Log_Feb15.pdf p.2] - Weather delay documentation
        - [Contract_2023.pdf p.5] - Timeline specifications
        """
        
        metrics = self.analyzer.analyze_response_quality(
            user_query=query,
            assistant_answer=answer,
            source_chunks=self.source_chunks,
            citation_mappings=self.citation_mappings,
            citation_metrics=self.citation_metrics,
            followup_suggestions=self.followup_suggestions,
            memory_items=self.memory_items,
            processing_time=1.5,
            matter_id="test_matter"
        )
        
        assert isinstance(metrics, ResponseQualityMetrics)
        assert 0.0 <= metrics.overall_quality <= 1.0
        assert 0.0 <= metrics.confidence_score <= 1.0
        assert metrics.response_length == len(answer)
        assert metrics.processing_time == 1.5
        assert isinstance(metrics.meets_minimum_standards, bool)
        assert isinstance(metrics.quality_warnings, list)
    
    def test_update_quality_history(self):
        """Test quality history tracking."""
        matter_id = "test_matter"
        
        # Create sample metrics
        metrics1 = ResponseQualityMetrics(
            citation_coverage=0.8, citation_accuracy=0.9, citation_diversity=0.7,
            answer_completeness=0.8, content_coherence=0.85, domain_specificity=0.7,
            source_diversity=0.6, source_relevance=0.8, source_recency=0.5,
            followup_relevance=0.7, followup_diversity=0.6, followup_actionability=0.8,
            confidence_score=0.8, overall_quality=0.75, response_length=500
        )
        
        # Update history
        self.analyzer._update_quality_history(matter_id, metrics1)
        
        assert matter_id in self.analyzer.quality_history
        assert len(self.analyzer.quality_history[matter_id]) == 1
        
        # Add more metrics
        metrics2 = ResponseQualityMetrics(
            citation_coverage=0.7, citation_accuracy=0.8, citation_diversity=0.6,
            answer_completeness=0.7, content_coherence=0.75, domain_specificity=0.6,
            source_diversity=0.5, source_relevance=0.7, source_recency=0.4,
            followup_relevance=0.6, followup_diversity=0.5, followup_actionability=0.7,
            confidence_score=0.7, overall_quality=0.65, response_length=400
        )
        
        self.analyzer._update_quality_history(matter_id, metrics2)
        assert len(self.analyzer.quality_history[matter_id]) == 2
    
    def test_get_historical_stats(self):
        """Test historical statistics calculation."""
        matter_id = "test_matter"
        
        # Add multiple metrics to history
        for i in range(5):
            metrics = ResponseQualityMetrics(
                citation_coverage=0.7 + i*0.05, citation_accuracy=0.8 + i*0.03,
                citation_diversity=0.6, answer_completeness=0.7 + i*0.04,
                content_coherence=0.75, domain_specificity=0.6,
                source_diversity=0.5, source_relevance=0.7, source_recency=0.4,
                followup_relevance=0.6, followup_diversity=0.5, followup_actionability=0.7,
                confidence_score=0.7 + i*0.02, overall_quality=0.65 + i*0.03,
                response_length=400
            )
            self.analyzer._update_quality_history(matter_id, metrics)
        
        stats = self.analyzer.get_historical_stats(matter_id)
        
        assert isinstance(stats, HistoricalQualityStats)
        assert stats.total_responses == 5
        assert 0.0 <= stats.average_quality <= 1.0
        assert stats.quality_trend != 0.0  # Should detect trend
        assert stats.best_quality_score >= stats.worst_quality_score
    
    def test_get_historical_stats_no_data(self):
        """Test historical stats when no data exists."""
        stats = self.analyzer.get_historical_stats("nonexistent_matter")
        assert stats is None
    
    def test_suggest_quality_improvements(self):
        """Test quality improvement suggestions."""
        # Poor quality metrics
        poor_metrics = ResponseQualityMetrics(
            citation_coverage=0.4,     # Below threshold
            citation_accuracy=0.5,    # Below ideal
            citation_diversity=0.3,   # Below threshold
            answer_completeness=0.5,  # Below threshold
            content_coherence=0.4,    # Below ideal
            domain_specificity=0.4,   # Below ideal
            source_diversity=0.3,     # Below threshold
            source_relevance=0.6,     # OK
            source_recency=0.5,       # OK
            followup_relevance=0.4,   # Below ideal
            followup_diversity=0.5,   # OK
            followup_actionability=0.6, # OK
            confidence_score=0.3,     # Below threshold
            overall_quality=0.4,      # Poor
            response_length=200
        )
        
        suggestions = self.analyzer.suggest_quality_improvements(poor_metrics)
        
        assert len(suggestions) > 0
        assert any("citation coverage" in suggestion.lower() for suggestion in suggestions)
        assert any("source diversity" in suggestion.lower() for suggestion in suggestions)
        assert any("completeness" in suggestion.lower() for suggestion in suggestions)


class TestQualityThresholds:
    """Test QualityThresholds data class."""
    
    def test_quality_thresholds_defaults(self):
        """Test default quality thresholds."""
        thresholds = QualityThresholds()
        
        assert thresholds.minimum_citation_coverage == 0.6
        assert thresholds.minimum_source_diversity == 0.4
        assert thresholds.minimum_answer_completeness == 0.7
        assert thresholds.minimum_confidence_score == 0.5
    
    def test_quality_thresholds_custom(self):
        """Test custom quality thresholds."""
        thresholds = QualityThresholds(
            minimum_citation_coverage=0.8,
            minimum_source_diversity=0.6,
            minimum_answer_completeness=0.9,
            minimum_confidence_score=0.7
        )
        
        assert thresholds.minimum_citation_coverage == 0.8
        assert thresholds.minimum_source_diversity == 0.6
        assert thresholds.minimum_answer_completeness == 0.9
        assert thresholds.minimum_confidence_score == 0.7


class TestResponseQualityMetrics:
    """Test ResponseQualityMetrics data class."""
    
    def test_response_quality_metrics_creation(self):
        """Test ResponseQualityMetrics creation."""
        metrics = ResponseQualityMetrics(
            citation_coverage=0.8,
            citation_accuracy=0.9,
            citation_diversity=0.7,
            answer_completeness=0.85,
            content_coherence=0.8,
            domain_specificity=0.75,
            source_diversity=0.6,
            source_relevance=0.8,
            source_recency=0.5,
            followup_relevance=0.7,
            followup_diversity=0.6,
            followup_actionability=0.8,
            confidence_score=0.82,
            overall_quality=0.76,
            response_length=650,
            processing_time=2.1,
            meets_minimum_standards=True,
            requires_regeneration=False,
            quality_warnings=["Minor warning"]
        )
        
        assert metrics.overall_quality == 0.76
        assert metrics.confidence_score == 0.82
        assert metrics.processing_time == 2.1
        assert metrics.meets_minimum_standards is True
        assert metrics.requires_regeneration is False
        assert len(metrics.quality_warnings) == 1