"""
Response Quality Measurement and Analysis.

Provides comprehensive quality metrics for RAG responses including
citation coverage, source diversity, answer completeness, and
confidence scoring for construction claims analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import math
from statistics import mean, stdev

from .logging_conf import get_logger
from .rag import SourceChunk
from .models import KnowledgeItem
from .citation_manager import CitationMapping, CitationMetrics
from .followup_engine import FollowupSuggestion

logger = get_logger(__name__)


@dataclass
class QualityThresholds:
    """Quality thresholds for response validation."""
    minimum_citation_coverage: float = 0.6
    minimum_source_diversity: float = 0.4
    minimum_answer_completeness: float = 0.7
    minimum_confidence_score: float = 0.5
    maximum_followup_redundancy: float = 0.3


@dataclass
class ResponseQualityMetrics:
    """Comprehensive quality metrics for a single response."""
    # Citation quality
    citation_coverage: float
    citation_accuracy: float
    citation_diversity: float
    
    # Content quality
    answer_completeness: float
    content_coherence: float
    domain_specificity: float
    
    # Source quality
    source_diversity: float
    source_relevance: float
    source_recency: float
    
    # Follow-up quality
    followup_relevance: float
    followup_diversity: float
    followup_actionability: float
    
    # Overall scores
    confidence_score: float
    overall_quality: float
    
    # Metadata
    response_length: int
    processing_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Quality flags
    meets_minimum_standards: bool = False
    requires_regeneration: bool = False
    quality_warnings: List[str] = field(default_factory=list)


@dataclass
class HistoricalQualityStats:
    """Historical quality statistics for a matter."""
    total_responses: int
    average_quality: float
    quality_trend: float  # Positive = improving, negative = declining
    best_quality_score: float
    worst_quality_score: float
    quality_consistency: float  # Lower = more consistent
    
    # Category breakdown
    citation_quality_avg: float
    content_quality_avg: float
    source_quality_avg: float
    followup_quality_avg: float
    
    # Time-based metrics
    first_response_date: datetime
    last_response_date: datetime
    quality_by_day: Dict[str, float] = field(default_factory=dict)


class QualityAnalyzer:
    """Comprehensive response quality analysis engine."""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        self.quality_history = {}  # matter_id -> List[ResponseQualityMetrics]
        
    def analyze_response_quality(
        self,
        user_query: str,
        assistant_answer: str,
        source_chunks: List[SourceChunk],
        citation_mappings: List[CitationMapping],
        citation_metrics: CitationMetrics,
        followup_suggestions: List[FollowupSuggestion],
        memory_items: List[KnowledgeItem],
        processing_time: Optional[float] = None,
        matter_id: Optional[str] = None
    ) -> ResponseQualityMetrics:
        """Perform comprehensive quality analysis of a RAG response."""
        
        logger.debug(
            "Starting response quality analysis",
            query_preview=user_query[:100],
            answer_length=len(assistant_answer),
            source_count=len(source_chunks),
            citation_count=len(citation_mappings),
            followup_count=len(followup_suggestions)
        )
        
        # Citation quality metrics (from citation manager)
        citation_coverage = citation_metrics.coverage_score
        citation_accuracy = citation_metrics.accuracy_score
        citation_diversity = citation_metrics.diversity_score
        
        # Content quality analysis
        answer_completeness = self._analyze_answer_completeness(user_query, assistant_answer)
        content_coherence = self._analyze_content_coherence(assistant_answer)
        domain_specificity = self._analyze_domain_specificity(assistant_answer)
        
        # Source quality analysis
        source_diversity = self._analyze_source_diversity(source_chunks)
        source_relevance = self._analyze_source_relevance(source_chunks, user_query)
        source_recency = self._analyze_source_recency(source_chunks, memory_items)
        
        # Follow-up quality analysis
        followup_relevance = self._analyze_followup_relevance(followup_suggestions, user_query, assistant_answer)
        followup_diversity = self._analyze_followup_diversity(followup_suggestions)
        followup_actionability = self._analyze_followup_actionability(followup_suggestions)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            citation_accuracy, source_relevance, content_coherence, answer_completeness
        )
        
        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(
            citation_coverage, citation_accuracy, citation_diversity,
            answer_completeness, content_coherence, domain_specificity,
            source_diversity, source_relevance, source_recency,
            followup_relevance, followup_diversity, followup_actionability
        )
        
        # Quality validation and warnings
        meets_standards, requires_regen, warnings = self._validate_quality_standards(
            overall_quality, citation_coverage, source_diversity, 
            answer_completeness, confidence_score
        )
        
        metrics = ResponseQualityMetrics(
            citation_coverage=citation_coverage,
            citation_accuracy=citation_accuracy,
            citation_diversity=citation_diversity,
            answer_completeness=answer_completeness,
            content_coherence=content_coherence,
            domain_specificity=domain_specificity,
            source_diversity=source_diversity,
            source_relevance=source_relevance,
            source_recency=source_recency,
            followup_relevance=followup_relevance,
            followup_diversity=followup_diversity,
            followup_actionability=followup_actionability,
            confidence_score=confidence_score,
            overall_quality=overall_quality,
            response_length=len(assistant_answer),
            processing_time=processing_time,
            meets_minimum_standards=meets_standards,
            requires_regeneration=requires_regen,
            quality_warnings=warnings
        )
        
        # Store in history if matter_id provided
        if matter_id:
            self._update_quality_history(matter_id, metrics)
        
        logger.debug(
            "Quality analysis completed",
            overall_quality=overall_quality,
            confidence_score=confidence_score,
            meets_standards=meets_standards,
            warnings_count=len(warnings)
        )
        
        return metrics
    
    def _analyze_answer_completeness(self, query: str, answer: str) -> float:
        """Analyze how completely the answer addresses the query."""
        if not answer or not query:
            return 0.0
        
        completeness_score = 0.0
        
        # Check for required sections in construction claims analysis
        required_sections = ['key points', 'analysis', 'citations']
        found_sections = sum(1 for section in required_sections if section in answer.lower())
        completeness_score += 0.3 * (found_sections / len(required_sections))
        
        # Check query keyword coverage
        query_words = set(word.lower() for word in re.findall(r'\w+', query) if len(word) > 3)
        answer_words = set(word.lower() for word in re.findall(r'\w+', answer))
        
        if query_words:
            keyword_coverage = len(query_words.intersection(answer_words)) / len(query_words)
            completeness_score += 0.2 * keyword_coverage
        
        # Check answer length appropriateness
        answer_length = len(answer.split())
        if 100 <= answer_length <= 500:  # Optimal range for detailed answers
            completeness_score += 0.2
        elif 50 <= answer_length < 100 or 500 < answer_length <= 800:  # Acceptable range
            completeness_score += 0.1
        
        # Check for uncertainty acknowledgment when appropriate
        uncertainty_phrases = [
            'unclear', 'uncertain', 'insufficient information', 
            'additional documentation needed', 'cannot determine'
        ]
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        
        # Bonus for acknowledging limitations (shows thoughtfulness)
        if has_uncertainty:
            completeness_score += 0.1
        
        # Check for specific construction claims elements
        construction_elements = [
            'contract', 'specification', 'schedule', 'delay', 'damage', 
            'breach', 'negligence', 'standard of care', 'causation'
        ]
        found_elements = sum(1 for element in construction_elements if element in answer.lower())
        if found_elements > 0:
            completeness_score += 0.2 * min(found_elements / 3, 1.0)  # Up to 3 elements
        
        return min(completeness_score, 1.0)
    
    def _analyze_content_coherence(self, answer: str) -> float:
        """Analyze logical flow and coherence of the answer."""
        if not answer:
            return 0.0
        
        coherence_score = 0.0
        
        # Check for logical structure
        sections = re.split(r'##\s*', answer)
        if len(sections) >= 3:  # At least header + 2 sections
            coherence_score += 0.3
        
        # Check for transition words and phrases
        transition_words = [
            'however', 'therefore', 'furthermore', 'additionally', 'consequently',
            'in contrast', 'similarly', 'moreover', 'as a result', 'in conclusion'
        ]
        found_transitions = sum(1 for word in transition_words if word in answer.lower())
        coherence_score += 0.2 * min(found_transitions / 3, 1.0)
        
        # Check sentence length variation (indicates good flow)
        sentences = re.split(r'[.!?]+', answer)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        
        if len(sentence_lengths) > 3:
            length_std = stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
            avg_length = mean(sentence_lengths)
            
            # Good variation in sentence length indicates better readability
            if 0.3 * avg_length <= length_std <= 0.7 * avg_length:
                coherence_score += 0.2
        
        # Check for chronological or logical ordering
        ordering_indicators = [
            'first', 'second', 'third', 'initially', 'subsequently', 'finally',
            'before', 'after', 'during', 'following'
        ]
        found_ordering = sum(1 for indicator in ordering_indicators if indicator in answer.lower())
        coherence_score += 0.2 * min(found_ordering / 2, 1.0)
        
        # Penalty for repetitive content
        words = answer.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.6:  # Too much repetition
                coherence_score -= 0.1
        
        # Check for balanced section lengths
        if len(sections) >= 3:
            section_lengths = [len(section.split()) for section in sections[1:] if section.strip()]
            if section_lengths:
                max_length = max(section_lengths)
                min_length = min(section_lengths)
                if max_length > 0:
                    balance_ratio = min_length / max_length
                    if balance_ratio >= 0.3:  # Reasonably balanced
                        coherence_score += 0.1
        
        return min(coherence_score, 1.0)
    
    def _analyze_domain_specificity(self, answer: str) -> float:
        """Analyze use of construction claims domain terminology."""
        if not answer:
            return 0.0
        
        answer_lower = answer.lower()
        
        # Construction law terms
        legal_terms = [
            'breach of contract', 'liquidated damages', 'substantial completion',
            'time is of the essence', 'force majeure', 'changed conditions',
            'differing site conditions', 'constructive change', 'cardinal change',
            'acceleration', 'delay damages', 'impact costs', 'inefficiency',
            'disruption', 'standby costs', 'extended overhead'
        ]
        
        # Construction industry terms
        industry_terms = [
            'critical path', 'float', 'schedule compression', 'fast tracking',
            'value engineering', 'shop drawings', 'submittals', 'rfi',
            'change order', 'punch list', 'substantial completion',
            'certificate of occupancy', 'retainage', 'progress payment'
        ]
        
        # Technical construction terms
        technical_terms = [
            'specifications', 'drawings', 'as-built', 'field conditions',
            'quality control', 'quality assurance', 'inspection',
            'testing', 'non-conforming work', 'defective work',
            'remedial work', 'structural', 'geotechnical'
        ]
        
        all_terms = legal_terms + industry_terms + technical_terms
        
        found_terms = sum(1 for term in all_terms if term in answer_lower)
        term_density = found_terms / len(answer.split()) * 100  # Terms per 100 words
        
        # Score based on appropriate density of domain terms
        if 2 <= term_density <= 8:  # Optimal range
            domain_score = 1.0
        elif 1 <= term_density < 2 or 8 < term_density <= 12:  # Good range
            domain_score = 0.8
        elif 0.5 <= term_density < 1 or 12 < term_density <= 15:  # Acceptable
            domain_score = 0.6
        else:  # Too few or too many terms
            domain_score = 0.3
        
        # Bonus for using terms correctly in context
        contextual_usage = self._check_contextual_term_usage(answer_lower, all_terms)
        domain_score += 0.2 * contextual_usage
        
        return min(domain_score, 1.0)
    
    def _check_contextual_term_usage(self, answer: str, terms: List[str]) -> float:
        """Check if domain terms are used correctly in context."""
        correct_usage = 0
        total_found = 0
        
        # Simple context checks for key terms
        context_checks = {
            'breach of contract': ['obligation', 'duty', 'performance'],
            'liquidated damages': ['delay', 'completion', 'predetermined'],
            'critical path': ['schedule', 'delay', 'duration', 'activity'],
            'change order': ['modification', 'scope', 'cost', 'time'],
            'substantial completion': ['project', 'work', 'occupancy']
        }
        
        for term, context_words in context_checks.items():
            if term in answer:
                total_found += 1
                # Check if any context words appear near the term
                term_pos = answer.find(term)
                nearby_text = answer[max(0, term_pos-100):term_pos+100]
                
                if any(word in nearby_text for word in context_words):
                    correct_usage += 1
        
        return correct_usage / total_found if total_found > 0 else 0.0
    
    def _analyze_source_diversity(self, source_chunks: List[SourceChunk]) -> float:
        """Analyze diversity of sources used in the response."""
        if not source_chunks:
            return 0.0
        
        # Document diversity
        unique_docs = set(chunk.doc for chunk in source_chunks)
        doc_diversity = len(unique_docs) / len(source_chunks)
        
        # Document type diversity (if we can infer types)
        doc_types = set()
        for chunk in source_chunks:
            doc_name = chunk.doc.lower()
            if any(term in doc_name for term in ['contract', 'agreement']):
                doc_types.add('contract')
            elif any(term in doc_name for term in ['spec', 'specification']):
                doc_types.add('specification')
            elif any(term in doc_name for term in ['log', 'daily']):
                doc_types.add('log')
            elif any(term in doc_name for term in ['email', 'correspondence']):
                doc_types.add('correspondence')
            elif any(term in doc_name for term in ['report', 'analysis']):
                doc_types.add('report')
            else:
                doc_types.add('other')
        
        type_diversity = len(doc_types) / min(len(source_chunks), 6)  # Max 6 expected types
        
        # Page spread diversity (avoid clustering in same pages)
        if len(source_chunks) > 1:
            pages = [chunk.page_start for chunk in source_chunks]
            page_range = max(pages) - min(pages) + 1
            page_spread = min(page_range / (len(source_chunks) * 5), 1.0)  # Normalize
        else:
            page_spread = 1.0
        
        # Combined diversity score
        diversity_score = 0.5 * doc_diversity + 0.3 * type_diversity + 0.2 * page_spread
        
        return diversity_score
    
    def _analyze_source_relevance(self, source_chunks: List[SourceChunk], query: str) -> float:
        """Analyze how relevant the sources are to the query."""
        if not source_chunks or not query:
            return 0.0
        
        query_words = set(word.lower() for word in re.findall(r'\w+', query) if len(word) > 3)
        if not query_words:
            return 0.5  # No meaningful query words
        
        relevance_scores = []
        
        for chunk in source_chunks:
            chunk_words = set(word.lower() for word in re.findall(r'\w+', chunk.text))
            
            if not chunk_words:
                relevance_scores.append(0.0)
                continue
            
            # Word overlap relevance
            overlap = len(query_words.intersection(chunk_words))
            word_relevance = overlap / len(query_words)
            
            # Boost for high similarity scores
            similarity_boost = min(chunk.score * 0.5, 0.3)  # Up to 0.3 boost
            
            chunk_relevance = word_relevance + similarity_boost
            relevance_scores.append(min(chunk_relevance, 1.0))
        
        return mean(relevance_scores)
    
    def _analyze_source_recency(
        self, 
        source_chunks: List[SourceChunk], 
        memory_items: List[KnowledgeItem]
    ) -> float:
        """Analyze recency/freshness of source information."""
        if not source_chunks:
            return 0.0
        
        recency_score = 0.5  # Baseline score
        
        # Check if any sources are referenced in recent memory items
        memory_docs = set()
        recent_memory_count = 0
        
        for item in memory_items:
            for doc_ref in item.doc_refs:
                memory_docs.add(doc_ref.get('doc', '').lower())
                recent_memory_count += 1
        
        if memory_docs:
            source_docs = set(chunk.doc.lower() for chunk in source_chunks)
            memory_overlap = len(source_docs.intersection(memory_docs))
            
            if source_docs:
                memory_relevance = memory_overlap / len(source_docs)
                recency_score += 0.3 * memory_relevance
        
        # Penalize if all sources are from the same old document
        unique_docs = set(chunk.doc for chunk in source_chunks)
        if len(unique_docs) == 1:
            recency_score -= 0.2  # Diversity penalty
        
        return min(recency_score, 1.0)
    
    def _analyze_followup_relevance(
        self, 
        followups: List[FollowupSuggestion], 
        query: str, 
        answer: str
    ) -> float:
        """Analyze relevance of follow-up suggestions."""
        if not followups:
            return 0.0
        
        query_words = set(word.lower() for word in re.findall(r'\w+', query) if len(word) > 2)
        answer_words = set(word.lower() for word in re.findall(r'\w+', answer) if len(word) > 2)
        context_words = query_words.union(answer_words)
        
        relevance_scores = []
        
        for followup in followups:
            followup_words = set(word.lower() for word in re.findall(r'\w+', followup.question) if len(word) > 2)
            
            if not followup_words or not context_words:
                relevance_scores.append(0.0)
                continue
            
            # Word overlap with context
            overlap = len(followup_words.intersection(context_words))
            overlap_score = overlap / len(followup_words) if followup_words else 0.0
            
            # Priority score from followup engine
            priority_score = followup.priority if hasattr(followup, 'priority') else 0.5
            
            # Combined relevance
            combined_relevance = 0.6 * overlap_score + 0.4 * priority_score
            relevance_scores.append(combined_relevance)
        
        return mean(relevance_scores)
    
    def _analyze_followup_diversity(self, followups: List[FollowupSuggestion]) -> float:
        """Analyze diversity of follow-up suggestion categories."""
        if not followups:
            return 0.0
        
        # Category diversity
        categories = set()
        for followup in followups:
            if hasattr(followup, 'category'):
                categories.add(followup.category)
            else:
                # Simple classification for basic strings
                question = followup.question if hasattr(followup, 'question') else str(followup)
                if any(word in question.lower() for word in ['document', 'evidence', 'record']):
                    categories.add('evidence')
                elif any(word in question.lower() for word in ['legal', 'contract', 'liability']):
                    categories.add('legal')
                elif any(word in question.lower() for word in ['cost', 'damage', 'money']):
                    categories.add('damages')
                else:
                    categories.add('general')
        
        category_diversity = len(categories) / len(followups)
        
        # Question uniqueness (avoid repetitive suggestions)
        question_texts = [
            (followup.question if hasattr(followup, 'question') else str(followup)).lower()
            for followup in followups
        ]
        
        unique_score = 1.0
        for i, q1 in enumerate(question_texts):
            for j, q2 in enumerate(question_texts[i+1:], i+1):
                words1 = set(q1.split())
                words2 = set(q2.split())
                if words1 and words2:
                    overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                    if overlap > 0.7:  # Very similar questions
                        unique_score -= 0.2
        
        unique_score = max(unique_score, 0.0)
        
        return 0.6 * category_diversity + 0.4 * unique_score
    
    def _analyze_followup_actionability(self, followups: List[FollowupSuggestion]) -> float:
        """Analyze how actionable the follow-up suggestions are."""
        if not followups:
            return 0.0
        
        actionability_scores = []
        
        # Actionable question indicators
        actionable_starters = [
            'what', 'how', 'when', 'where', 'who', 'which', 'should we',
            'can we', 'do we need', 'is there', 'are there'
        ]
        
        actionable_keywords = [
            'document', 'evidence', 'report', 'analysis', 'review', 'examine',
            'investigate', 'contact', 'request', 'obtain', 'verify', 'confirm'
        ]
        
        for followup in followups:
            question = followup.question if hasattr(followup, 'question') else str(followup)
            question_lower = question.lower()
            
            actionability = 0.0
            
            # Check for actionable question starters
            if any(question_lower.startswith(starter) for starter in actionable_starters):
                actionability += 0.4
            
            # Check for actionable keywords
            found_keywords = sum(1 for keyword in actionable_keywords if keyword in question_lower)
            actionability += 0.3 * min(found_keywords / 2, 1.0)
            
            # Check question length (too short or too long may be less actionable)
            word_count = len(question.split())
            if 5 <= word_count <= 15:  # Optimal length
                actionability += 0.2
            elif 3 <= word_count < 5 or 15 < word_count <= 20:  # Acceptable
                actionability += 0.1
            
            # Bonus for specificity
            specific_terms = [
                'contract', 'specification', 'drawing', 'schedule', 'cost', 
                'timeline', 'expert', 'analysis', 'report', 'documentation'
            ]
            if any(term in question_lower for term in specific_terms):
                actionability += 0.1
            
            actionability_scores.append(min(actionability, 1.0))
        
        return mean(actionability_scores)
    
    def _calculate_confidence_score(
        self, 
        citation_accuracy: float, 
        source_relevance: float, 
        content_coherence: float, 
        answer_completeness: float
    ) -> float:
        """Calculate overall confidence in the response quality."""
        
        # Weight the most important factors for confidence
        confidence = (
            0.3 * citation_accuracy +      # Accurate citations are crucial
            0.25 * source_relevance +      # Relevant sources build confidence
            0.25 * answer_completeness +   # Complete answers are more trustworthy
            0.2 * content_coherence        # Coherent content suggests understanding
        )
        
        # Apply confidence penalties
        if citation_accuracy < 0.5:  # Poor citation accuracy is a red flag
            confidence *= 0.7
        
        if source_relevance < 0.4:  # Irrelevant sources hurt confidence
            confidence *= 0.8
        
        return confidence
    
    def _calculate_overall_quality(
        self,
        citation_coverage: float,
        citation_accuracy: float, 
        citation_diversity: float,
        answer_completeness: float,
        content_coherence: float,
        domain_specificity: float,
        source_diversity: float,
        source_relevance: float,
        source_recency: float,
        followup_relevance: float,
        followup_diversity: float,
        followup_actionability: float
    ) -> float:
        """Calculate weighted overall quality score."""
        
        # Citation quality (35% of total)
        citation_quality = (
            0.4 * citation_coverage +
            0.4 * citation_accuracy +
            0.2 * citation_diversity
        )
        
        # Content quality (30% of total) 
        content_quality = (
            0.4 * answer_completeness +
            0.35 * content_coherence +
            0.25 * domain_specificity
        )
        
        # Source quality (25% of total)
        source_quality = (
            0.5 * source_relevance +
            0.3 * source_diversity +
            0.2 * source_recency
        )
        
        # Follow-up quality (10% of total)
        followup_quality = (
            0.4 * followup_relevance +
            0.3 * followup_actionability +
            0.3 * followup_diversity
        )
        
        # Overall weighted score
        overall_quality = (
            0.35 * citation_quality +
            0.30 * content_quality +
            0.25 * source_quality +
            0.10 * followup_quality
        )
        
        return overall_quality
    
    def _validate_quality_standards(
        self,
        overall_quality: float,
        citation_coverage: float,
        source_diversity: float,
        answer_completeness: float,
        confidence_score: float
    ) -> Tuple[bool, bool, List[str]]:
        """Validate response against quality standards."""
        
        meets_standards = True
        requires_regeneration = False
        warnings = []
        
        # Check minimum standards
        if citation_coverage < self.thresholds.minimum_citation_coverage:
            meets_standards = False
            warnings.append(f"Citation coverage ({citation_coverage:.2f}) below minimum ({self.thresholds.minimum_citation_coverage})")
        
        if source_diversity < self.thresholds.minimum_source_diversity:
            meets_standards = False
            warnings.append(f"Source diversity ({source_diversity:.2f}) below minimum ({self.thresholds.minimum_source_diversity})")
        
        if answer_completeness < self.thresholds.minimum_answer_completeness:
            meets_standards = False
            warnings.append(f"Answer completeness ({answer_completeness:.2f}) below minimum ({self.thresholds.minimum_answer_completeness})")
        
        if confidence_score < self.thresholds.minimum_confidence_score:
            meets_standards = False
            warnings.append(f"Confidence score ({confidence_score:.2f}) below minimum ({self.thresholds.minimum_confidence_score})")
        
        # Determine if regeneration is recommended
        if overall_quality < 0.4 or confidence_score < 0.3:
            requires_regeneration = True
            warnings.append("Response quality too low - regeneration recommended")
        
        return meets_standards, requires_regeneration, warnings
    
    def _update_quality_history(self, matter_id: str, metrics: ResponseQualityMetrics):
        """Update quality history for matter-based tracking."""
        if matter_id not in self.quality_history:
            self.quality_history[matter_id] = []
        
        self.quality_history[matter_id].append(metrics)
        
        # Keep only last 100 responses per matter
        if len(self.quality_history[matter_id]) > 100:
            self.quality_history[matter_id] = self.quality_history[matter_id][-100:]
    
    def get_historical_stats(self, matter_id: str) -> Optional[HistoricalQualityStats]:
        """Get historical quality statistics for a matter."""
        if matter_id not in self.quality_history or not self.quality_history[matter_id]:
            return None
        
        metrics_list = self.quality_history[matter_id]
        
        # Calculate basic stats
        quality_scores = [m.overall_quality for m in metrics_list]
        total_responses = len(metrics_list)
        average_quality = mean(quality_scores)
        best_quality = max(quality_scores)
        worst_quality = min(quality_scores)
        quality_consistency = stdev(quality_scores) if len(quality_scores) > 1 else 0.0
        
        # Calculate trend (simple linear regression on recent scores)
        recent_scores = quality_scores[-min(10, len(quality_scores)):]  # Last 10 scores
        if len(recent_scores) >= 3:
            # Simple trend calculation
            x_vals = list(range(len(recent_scores)))
            x_mean = mean(x_vals)
            y_mean = mean(recent_scores)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, recent_scores))
            denominator = sum((x - x_mean) ** 2 for x in x_vals)
            
            quality_trend = numerator / denominator if denominator != 0 else 0.0
        else:
            quality_trend = 0.0
        
        # Category averages
        citation_quality_avg = mean([
            (m.citation_coverage + m.citation_accuracy + m.citation_diversity) / 3
            for m in metrics_list
        ])
        
        content_quality_avg = mean([
            (m.answer_completeness + m.content_coherence + m.domain_specificity) / 3
            for m in metrics_list
        ])
        
        source_quality_avg = mean([
            (m.source_diversity + m.source_relevance + m.source_recency) / 3
            for m in metrics_list
        ])
        
        followup_quality_avg = mean([
            (m.followup_relevance + m.followup_diversity + m.followup_actionability) / 3
            for m in metrics_list
        ])
        
        # Date range
        first_date = min(m.timestamp for m in metrics_list)
        last_date = max(m.timestamp for m in metrics_list)
        
        # Quality by day (last 30 days)
        quality_by_day = {}
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for metric in metrics_list:
            if metric.timestamp >= cutoff_date:
                date_key = metric.timestamp.strftime('%Y-%m-%d')
                if date_key not in quality_by_day:
                    quality_by_day[date_key] = []
                quality_by_day[date_key].append(metric.overall_quality)
        
        # Average quality by day
        for date_key in quality_by_day:
            quality_by_day[date_key] = mean(quality_by_day[date_key])
        
        return HistoricalQualityStats(
            total_responses=total_responses,
            average_quality=average_quality,
            quality_trend=quality_trend,
            best_quality_score=best_quality,
            worst_quality_score=worst_quality,
            quality_consistency=quality_consistency,
            citation_quality_avg=citation_quality_avg,
            content_quality_avg=content_quality_avg,
            source_quality_avg=source_quality_avg,
            followup_quality_avg=followup_quality_avg,
            first_response_date=first_date,
            last_response_date=last_date,
            quality_by_day=quality_by_day
        )
    
    def suggest_quality_improvements(self, metrics: ResponseQualityMetrics) -> List[str]:
        """Suggest specific improvements based on quality metrics."""
        suggestions = []
        
        if metrics.citation_coverage < 0.7:
            suggestions.append("Increase citation coverage by ensuring all key points are supported with references")
        
        if metrics.citation_accuracy < 0.8:
            suggestions.append("Improve citation accuracy by verifying document names and page numbers")
        
        if metrics.source_diversity < 0.5:
            suggestions.append("Use more diverse sources across different document types and pages")
        
        if metrics.answer_completeness < 0.7:
            suggestions.append("Provide more complete answers addressing all aspects of the query")
        
        if metrics.content_coherence < 0.6:
            suggestions.append("Improve answer structure and logical flow between sections")
        
        if metrics.domain_specificity < 0.6:
            suggestions.append("Use more construction-specific terminology and legal concepts")
        
        if metrics.followup_relevance < 0.6:
            suggestions.append("Generate more relevant follow-up questions based on the analysis")
        
        if metrics.confidence_score < 0.5:
            suggestions.append("Focus on improving source quality and citation accuracy to build confidence")
        
        return suggestions