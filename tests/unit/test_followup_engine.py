"""
Unit tests for the Follow-up Engine module.

Tests advanced follow-up question generation with domain expertise,
context awareness, and priority scoring.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from app.followup_engine import (
    FollowupEngine, FollowupSuggestion, FollowupContext, FollowupCategory
)
from app.models import KnowledgeItem
from app.llm.base import LLMProvider


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self):
        self.generation_calls = []
        self.responses = []
        self.current_response_index = 0
    
    def set_responses(self, responses: List[str]):
        """Set predefined responses."""
        self.responses = responses
        self.current_response_index = 0
    
    async def generate(self, system: str, messages: List[dict], max_tokens: int = 200, temperature: float = 0.3) -> str:
        """Mock generate method."""
        self.generation_calls.append({
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        })
        
        if self.responses and self.current_response_index < len(self.responses):
            response = self.responses[self.current_response_index]
            self.current_response_index += 1
            return response
        
        # Default response
        return """What additional contract documentation is needed?
Should we engage a structural expert for analysis?
How does this impact the project timeline?
What are the potential cost implications?"""


class TestFollowupEngine:
    """Test suite for FollowupEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.followup_engine = FollowupEngine(self.mock_llm)
        
        # Sample memory items
        self.memory_items = [
            KnowledgeItem(
                type="Event",
                label="Dry Well Failure",
                date="2023-02-14",
                actors=["Contractor X", "Owner"],
                doc_refs=[{"doc": "Daily_Log_Feb15.pdf", "page": 2}],
                support_snippet="Well failed during testing phase"
            ),
            KnowledgeItem(
                type="Issue", 
                label="Design Defect",
                actors=["Engineer A"],
                doc_refs=[{"doc": "Design_Review.pdf", "page": 5}]
            ),
            KnowledgeItem(
                type="Entity",
                label="ABC Construction Company",
                actors=["ABC Construction"],
                doc_refs=[{"doc": "Contract_2023.pdf", "page": 1}]
            )
        ]
    
    def test_followup_context_creation(self):
        """Test FollowupContext creation and attributes."""
        context = FollowupContext(
            user_query="What caused the well failure?",
            assistant_answer="The well failed due to design issues...",
            memory_items=self.memory_items,
            conversation_history=["Previous question", "Previous answer"],
            matter_context={"matter_id": "test_matter", "matter_name": "Test Case"}
        )
        
        assert context.user_query == "What caused the well failure?"
        assert len(context.memory_items) == 3
        assert len(context.conversation_history) == 2
        assert context.matter_context["matter_id"] == "test_matter"
    
    def test_extract_entities(self):
        """Test entity extraction from text."""
        query = "What was ABC Construction's role in the structural design?"
        answer = "ABC Construction was the primary contractor responsible for foundation work. The engineer reviewed specifications."
        
        entities = self.followup_engine._extract_entities(query, answer)
        
        assert "organizations" in entities
        assert "technical_terms" in entities
        assert len(entities["organizations"]) > 0
        assert any("construction" in org.lower() for org in entities["organizations"])
    
    def test_extract_topics(self):
        """Test topic extraction from text."""
        query = "What caused the delay in concrete work?"
        answer = "The delay was due to contract specifications requiring additional testing and inspection."
        
        topics = self.followup_engine._extract_topics(query, answer)
        
        assert len(topics) > 0
        assert any("delay" in topic.lower() for topic in topics)
        assert any("contract" in topic.lower() for topic in topics)
    
    def test_template_library_structure(self):
        """Test that template library is properly structured."""
        templates = self.followup_engine.template_library
        
        # Check all expected categories are present
        expected_categories = [
            FollowupCategory.EVIDENCE,
            FollowupCategory.TIMELINE, 
            FollowupCategory.TECHNICAL,
            FollowupCategory.LEGAL,
            FollowupCategory.DAMAGES,
            FollowupCategory.CAUSATION,
            FollowupCategory.RESPONSIBILITY
        ]
        
        for category in expected_categories:
            assert category in templates
            assert len(templates[category]) > 0
            assert all(isinstance(template, str) for template in templates[category])
    
    def test_fill_template_with_entities(self):
        """Test template filling with extracted entities."""
        template = "What additional documentation exists for {topic}?"
        entities = {
            "organizations": ["ABC Construction"], 
            "technical_terms": ["foundation", "concrete"],
            "documents": ["Contract.pdf"]
        }
        topics = ["delay", "inspection"]
        
        filled = self.followup_engine._fill_template(template, entities, topics)
        
        assert len(filled) > 0
        assert all("{topic}" not in question for question in filled)
        assert any("delay" in question or "inspection" in question for question in filled)
    
    def test_generate_template_suggestions(self):
        """Test template-based suggestion generation."""
        context = FollowupContext(
            user_query="What caused the foundation problems?",
            assistant_answer="The foundation issues were due to poor soil conditions and design defects.",
            memory_items=self.memory_items
        )
        
        suggestions = self.followup_engine._generate_template_suggestions(context)
        
        assert len(suggestions) > 0
        assert all(isinstance(suggestion, FollowupSuggestion) for suggestion in suggestions)
        assert all(hasattr(suggestion, 'category') for suggestion in suggestions)
        assert all(hasattr(suggestion, 'priority') for suggestion in suggestions)
    
    def test_generate_memory_suggestions(self):
        """Test memory-based suggestion generation."""
        context = FollowupContext(
            user_query="Tell me about the dry well failure",
            assistant_answer="The dry well failed on February 14, 2023 during testing.",
            memory_items=self.memory_items
        )
        
        suggestions = self.followup_engine._generate_memory_suggestions(context)
        
        assert len(suggestions) > 0
        # Should generate suggestions related to the "Dry Well Failure" event
        assert any("Dry Well Failure" in suggestion.reasoning for suggestion in suggestions)
    
    def test_generate_memory_type_suggestions_event(self):
        """Test memory suggestions for Event type."""
        event_items = [item for item in self.memory_items if item.type == "Event"]
        context = FollowupContext(
            user_query="What happened with the well?",
            assistant_answer="The well failed during testing.",
            memory_items=self.memory_items
        )
        
        suggestions = self.followup_engine._generate_memory_type_suggestions("Event", event_items, context)
        
        assert len(suggestions) >= 2  # Should generate multiple suggestions per event
        assert all(suggestion.category in [FollowupCategory.EVIDENCE, FollowupCategory.DAMAGES] 
                  for suggestion in suggestions)
    
    def test_generate_memory_type_suggestions_issue(self):
        """Test memory suggestions for Issue type."""
        issue_items = [item for item in self.memory_items if item.type == "Issue"]
        context = FollowupContext(
            user_query="What are the design problems?",
            assistant_answer="There are several design defects identified.",
            memory_items=self.memory_items
        )
        
        suggestions = self.followup_engine._generate_memory_type_suggestions("Issue", issue_items, context)
        
        assert len(suggestions) >= 1
        assert all(suggestion.category == FollowupCategory.LEGAL for suggestion in suggestions)
        assert all(suggestion.requires_expert for suggestion in suggestions)
    
    @pytest.mark.asyncio
    async def test_generate_llm_suggestions(self):
        """Test LLM-based suggestion generation."""
        self.mock_llm.set_responses(["""
        What specific contract provisions address this issue?
        Should we engage a geotechnical expert for soil analysis?
        What documentation exists for the testing procedures?
        How does this impact the project completion schedule?
        """])
        
        context = FollowupContext(
            user_query="What caused the foundation failure?",
            assistant_answer="The foundation failed due to poor soil conditions and inadequate design.",
            memory_items=self.memory_items
        )
        
        suggestions = await self.followup_engine._generate_llm_suggestions(context)
        
        assert len(suggestions) >= 2
        assert all(isinstance(suggestion, FollowupSuggestion) for suggestion in suggestions)
        assert all(len(suggestion.question) <= 150 for suggestion in suggestions)
        # Check that at least one call was made to the LLM
        assert len(self.mock_llm.generation_calls) >= 1
    
    def test_classify_question_category(self):
        """Test question category classification."""
        # Test evidence category
        evidence_q = "What documentation exists for this issue?"
        assert self.followup_engine._classify_question_category(evidence_q) == FollowupCategory.EVIDENCE
        
        # Test timeline category  
        timeline_q = "When did this failure first occur?"
        assert self.followup_engine._classify_question_category(timeline_q) == FollowupCategory.TIMELINE
        
        # Test technical category
        technical_q = "Should we engage a structural expert?"
        assert self.followup_engine._classify_question_category(technical_q) == FollowupCategory.TECHNICAL
        
        # Test legal category
        legal_q = "What contract provisions address this breach?"
        assert self.followup_engine._classify_question_category(legal_q) == FollowupCategory.LEGAL
        
        # Test damages category
        damages_q = "What are the cost implications of this defect?"
        assert self.followup_engine._classify_question_category(damages_q) == FollowupCategory.DAMAGES
        
        # Test causation category
        causation_q = "What caused this structural failure?"
        assert self.followup_engine._classify_question_category(causation_q) == FollowupCategory.CAUSATION
    
    def test_requires_expert_analysis(self):
        """Test expert analysis requirement detection."""
        # Should require expert
        expert_questions = [
            "Should we engage a structural expert?",
            "Is geotechnical analysis needed?",
            "What technical testing is required?"
        ]
        
        for question in expert_questions:
            assert self.followup_engine._requires_expert_analysis(question) is True
        
        # Should not require expert
        non_expert_questions = [
            "What contract provisions apply?",
            "When was this work completed?",
            "Who was the contractor?"
        ]
        
        for question in non_expert_questions:
            assert self.followup_engine._requires_expert_analysis(question) is False
    
    def test_is_relevant_to_context(self):
        """Test context relevance checking."""
        context = FollowupContext(
            user_query="What caused the foundation settlement?",
            assistant_answer="The foundation settled due to poor soil conditions and inadequate compaction.",
            memory_items=[]
        )
        
        # Relevant questions
        relevant_q = "What soil testing was performed before construction?"
        assert self.followup_engine._is_relevant_to_context(relevant_q, context) is True
        
        # Irrelevant question
        irrelevant_q = "What color should we paint the building?"
        assert self.followup_engine._is_relevant_to_context(irrelevant_q, context) is False
    
    def test_calculate_priority(self):
        """Test priority score calculation."""
        context = FollowupContext(
            user_query="What caused the structural failure?",
            assistant_answer="The structure failed due to design defects and poor construction.",
            memory_items=self.memory_items
        )
        
        # Legal question with relevant keywords
        legal_question = "What contract provisions address structural defects?"
        legal_priority = self.followup_engine._calculate_priority(
            legal_question, FollowupCategory.LEGAL, context
        )
        
        # Should get high priority for legal questions
        assert legal_priority >= 0.5
        
        # Generic question
        generic_question = "What should we do next?"
        generic_priority = self.followup_engine._calculate_priority(
            generic_question, FollowupCategory.RESPONSIBILITY, context
        )
        
        # Legal should have higher priority than generic
        assert legal_priority > generic_priority
    
    def test_remove_duplicate_questions(self):
        """Test duplicate question removal."""
        suggestions = [
            FollowupSuggestion("What contract provisions apply?", FollowupCategory.LEGAL, 0.8, "Test", []),
            FollowupSuggestion("What contract terms are relevant?", FollowupCategory.LEGAL, 0.7, "Test", []),  # Similar
            FollowupSuggestion("Should we engage an expert?", FollowupCategory.TECHNICAL, 0.9, "Test", [])
        ]
        
        unique = self.followup_engine._remove_duplicate_questions(suggestions)
        
        # Should remove one of the similar contract questions
        assert len(unique) <= len(suggestions)
        # Should keep the expert question (clearly different)
        assert any("expert" in suggestion.question.lower() for suggestion in unique)
    
    def test_ensure_category_diversity(self):
        """Test category diversity enforcement."""
        suggestions = [
            FollowupSuggestion("Legal question 1", FollowupCategory.LEGAL, 0.9, "Test", []),
            FollowupSuggestion("Legal question 2", FollowupCategory.LEGAL, 0.8, "Test", []),
            FollowupSuggestion("Evidence question", FollowupCategory.EVIDENCE, 0.7, "Test", []),
            FollowupSuggestion("Technical question", FollowupCategory.TECHNICAL, 0.6, "Test", []),
            FollowupSuggestion("Damages question", FollowupCategory.DAMAGES, 0.5, "Test", [])
        ]
        
        diverse = self.followup_engine._ensure_category_diversity(suggestions, 4)
        
        assert len(diverse) == 4
        # Should include different categories, not just highest scoring legal questions
        categories = set(suggestion.category for suggestion in diverse)
        assert len(categories) >= 3  # Should have at least 3 different categories
    
    @pytest.mark.asyncio
    async def test_generate_followups_integration(self):
        """Test full follow-up generation integration."""
        self.mock_llm.set_responses(["""
        What specific geotechnical analysis is needed for the foundation?
        Should we review the original soil boring reports?
        How does this impact the construction schedule?
        What are the potential remediation costs?
        """])
        
        context = FollowupContext(
            user_query="What caused the foundation problems?",
            assistant_answer="Foundation issues arose from poor soil conditions and design inadequacies.",
            memory_items=self.memory_items,
            conversation_history=["Previous Q", "Previous A"]
        )
        
        suggestions = await self.followup_engine.generate_followups(context, max_suggestions=4, min_priority=0.3)
        
        assert len(suggestions) <= 4
        assert all(isinstance(suggestion, FollowupSuggestion) for suggestion in suggestions)
        assert all(suggestion.priority >= 0.3 for suggestion in suggestions)
        
        # Should be sorted by priority (descending)
        priorities = [suggestion.priority for suggestion in suggestions]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_summarize_memory_for_llm(self):
        """Test memory summarization for LLM context."""
        summary = self.followup_engine._summarize_memory_for_llm(self.memory_items)
        
        assert len(summary) > 0
        assert "Dry Well Failure" in summary
        assert "Design Defect" in summary
        assert "ABC Construction" in summary
    
    @pytest.mark.asyncio
    async def test_generate_followups_no_llm(self):
        """Test follow-up generation without LLM provider."""
        engine_no_llm = FollowupEngine(llm_provider=None)
        
        context = FollowupContext(
            user_query="What caused the issues?",
            assistant_answer="Multiple factors contributed to the problems.",
            memory_items=self.memory_items
        )
        
        suggestions = await engine_no_llm.generate_followups(context)
        
        # Should still generate suggestions from templates and memory
        assert len(suggestions) > 0
        assert all(isinstance(suggestion, FollowupSuggestion) for suggestion in suggestions)
    
    def test_filter_and_prioritize(self):
        """Test suggestion filtering and prioritization."""
        suggestions = [
            FollowupSuggestion("High priority legal", FollowupCategory.LEGAL, 0.9, "Test", []),
            FollowupSuggestion("Medium priority evidence", FollowupCategory.EVIDENCE, 0.6, "Test", []),
            FollowupSuggestion("Low priority generic", FollowupCategory.RESPONSIBILITY, 0.2, "Test", []),
            FollowupSuggestion("Good technical", FollowupCategory.TECHNICAL, 0.7, "Test", [])
        ]
        
        filtered = self.followup_engine._filter_and_prioritize(suggestions, min_priority=0.5, max_suggestions=3)
        
        assert len(filtered) == 3  # Should limit to max_suggestions
        assert all(suggestion.priority >= 0.5 for suggestion in filtered)  # Should filter by min_priority
        # Should be sorted by priority descending
        priorities = [suggestion.priority for suggestion in filtered]
        assert priorities == sorted(priorities, reverse=True)


class TestFollowupSuggestion:
    """Test FollowupSuggestion data class."""
    
    def test_followup_suggestion_creation(self):
        """Test FollowupSuggestion creation and attributes."""
        suggestion = FollowupSuggestion(
            question="What contract provisions address this issue?",
            category=FollowupCategory.LEGAL,
            priority=0.8,
            reasoning="Legal analysis needed for contract interpretation",
            related_entities=["ABC Construction", "Owner"],
            requires_expert=False
        )
        
        assert suggestion.question == "What contract provisions address this issue?"
        assert suggestion.category == FollowupCategory.LEGAL
        assert suggestion.priority == 0.8
        assert suggestion.requires_expert is False
        assert len(suggestion.related_entities) == 2


class TestFollowupCategory:
    """Test FollowupCategory enum."""
    
    def test_followup_categories(self):
        """Test that all expected categories exist."""
        expected_categories = [
            "evidence", "timeline", "technical", "legal", 
            "damages", "causation", "responsibility"
        ]
        
        for category_value in expected_categories:
            # Should be able to create category from value
            category = FollowupCategory(category_value)
            assert category.value == category_value