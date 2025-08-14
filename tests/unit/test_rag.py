"""
Unit tests for the RAG (Retrieval-Augmented Generation) engine.

Tests prompt assembly, citation extraction, provider integration,
and the complete RAG pipeline with mocked dependencies.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
import tempfile
import json
from typing import List

from app.rag import RAGEngine, RAGResponse, SourceChunk
from app.vectors import SearchResult, VectorStore
from app.llm.base import LLMProvider
from app.models import Matter, MatterPaths, KnowledgeItem
from app.prompts import (
    assemble_rag_prompt,
    extract_citations_from_answer,
    validate_citations,
    format_doc_context,
    format_memory_context
)


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.generation_calls = []
        self.responses = []
        self.current_response_index = 0
    
    def set_responses(self, responses: List[str]):
        """Set predefined responses for generation calls."""
        self.responses = responses
        self.current_response_index = 0
    
    async def generate(self, system: str, messages: List[dict], max_tokens: int = 900, temperature: float = 0.2) -> str:
        """Mock generate method."""
        call_info = {
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        self.generation_calls.append(call_info)
        
        if self.responses and self.current_response_index < len(self.responses):
            response = self.responses[self.current_response_index]
            self.current_response_index += 1
            return response
        
        # Default response for testing
        return """## Key Points
- Mock finding from document analysis [TestDoc.pdf p.1]
- Second finding with citation [TestDoc.pdf p.2]

## Analysis
This is a mock analysis of the construction claim. The findings indicate potential issues with the project timeline and specifications as documented in the referenced materials.

## Citations
- [TestDoc.pdf p.1] - Supporting evidence for first finding
- [TestDoc.pdf p.2] - Additional context for second finding

## Suggested Follow-ups
- Review project specifications for compliance issues
- Analyze schedule impacts and potential delays
- Assess cost implications of identified issues
- Consider expert analysis for technical matters"""
    
    async def test_connection(self) -> bool:
        """Mock connection test."""
        return True


class TestPromptAssembly:
    """Test prompt assembly and formatting functions."""
    
    def test_format_doc_context(self):
        """Test document context formatting."""
        search_results = [
            SearchResult(
                chunk_id="chunk_1",
                doc_name="Spec2021.pdf",
                page_start=12,
                page_end=12,
                text="The concrete mix design shall comply with ACI standards...",
                similarity_score=0.87,
                metadata={}
            ),
            SearchResult(
                chunk_id="chunk_2", 
                doc_name="DailyLog.pdf",
                page_start=5,
                page_end=6,
                text="Weather conditions prevented concrete pour on 2023-02-14...",
                similarity_score=0.76,
                metadata={}
            )
        ]
        
        context = format_doc_context(search_results)
        
        assert "DOC[1]: Spec2021.pdf (p.12-12, similarity: 0.870)" in context
        assert "DOC[2]: DailyLog.pdf (p.5-6, similarity: 0.760)" in context
        assert "concrete mix design" in context
        assert "Weather conditions" in context
    
    def test_format_memory_context(self):
        """Test memory context formatting."""
        memory_items = [
            KnowledgeItem(
                type="Event",
                label="Dry Well Failure",
                date="2023-02-14",
                actors=["Contractor X", "Site Engineer"],
                doc_refs=[{"doc": "DailyLog.pdf", "page": 5}],
                support_snippet="Equipment malfunction caused dry well failure"
            ),
            KnowledgeItem(
                type="Issue",
                label="Schedule Delay",
                date=None,
                actors=["General Contractor"],
                doc_refs=[],
                support_snippet=None
            )
        ]
        
        context = format_memory_context(memory_items)
        
        assert "MEMORY[1]: Event - Dry Well Failure" in context
        assert "Date: 2023-02-14" in context
        assert "Actors: Contractor X, Site Engineer" in context
        assert "References: DailyLog.pdf p.5" in context
        assert "MEMORY[2]: Issue - Schedule Delay" in context
    
    def test_assemble_rag_prompt(self):
        """Test complete RAG prompt assembly."""
        query = "What caused the dry well failure?"
        
        search_results = [
            SearchResult(
                chunk_id="chunk_1",
                doc_name="TestDoc.pdf",
                page_start=1,
                page_end=1,
                text="Equipment malfunction during concrete operations.",
                similarity_score=0.9,
                metadata={}
            )
        ]
        
        memory_items = [
            KnowledgeItem(
                type="Event",
                label="Equipment Failure",
                date="2023-02-14",
                actors=["Operator"],
                doc_refs=[{"doc": "TestDoc.pdf", "page": 1}],
                support_snippet="Pump malfunction"
            )
        ]
        
        messages = assemble_rag_prompt(query, search_results, memory_items)
        
        assert len(messages) == 1
        user_message = messages[0]
        assert user_message["role"] == "user"
        assert "What caused the dry well failure?" in user_message["content"]
        assert "AGENT MEMORY" in user_message["content"]
        assert "DOCUMENT CONTEXT" in user_message["content"]
        assert "Equipment malfunction" in user_message["content"]
        assert "Key Points" in user_message["content"]  # Output format instruction
    
    def test_extract_citations_from_answer(self):
        """Test citation extraction from LLM answers."""
        answer = """## Key Points
- Equipment failure occurred [TestDoc.pdf p.1]
- Weather delays documented [WeatherLog.pdf p.5-7]

## Analysis
The analysis shows multiple factors [Spec2021.pdf p.12].

## Citations
- [TestDoc.pdf p.1] - Equipment failure details
- [WeatherLog.pdf p.5-7] - Weather impact analysis"""
        
        citations = extract_citations_from_answer(answer)
        
        assert len(citations) == 3
        assert "[TestDoc.pdf p.1]" in citations
        assert "[WeatherLog.pdf p.5-7]" in citations  
        assert "[Spec2021.pdf p.12]" in citations
    
    def test_validate_citations(self):
        """Test citation validation against available sources."""
        citations = [
            "[TestDoc.pdf p.1]",
            "[TestDoc.pdf p.5]",
            "[NonExistent.pdf p.1]"
        ]
        
        search_results = [
            SearchResult(
                chunk_id="chunk_1",
                doc_name="TestDoc.pdf",
                page_start=1,
                page_end=3,
                text="Content from pages 1-3",
                similarity_score=0.9,
                metadata={}
            )
        ]
        
        validity = validate_citations(citations, search_results)
        
        assert validity["[TestDoc.pdf p.1]"] == True  # Page 1 is available
        assert validity["[TestDoc.pdf p.5]"] == False  # Page 5 not available
        assert validity["[NonExistent.pdf p.1]"] == False  # Document not available


class TestRAGEngine:
    """Test RAG engine functionality."""
    
    @pytest.fixture
    def temp_matter_dir(self):
        """Create temporary matter directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_root = Path(temp_dir) / "test_matter"
            matter_root.mkdir()
            
            # Create subdirectories
            paths = MatterPaths.from_root(matter_root)
            for path in [paths.docs, paths.docs_ocr, paths.parsed, paths.vectors, 
                        paths.knowledge, paths.chat, paths.logs]:
                path.mkdir(parents=True, exist_ok=True)
            
            yield matter_root
    
    @pytest.fixture
    def mock_matter(self, temp_matter_dir):
        """Create mock matter for testing."""
        paths = MatterPaths.from_root(temp_matter_dir)
        return Matter(
            id="test-matter-123",
            name="Test Construction Claim",
            slug="test-construction-claim",
            embedding_model="mock-embed",
            generation_model="mock-gen",
            paths=paths
        )
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock(spec=VectorStore)
        store.search = AsyncMock(return_value=[
            SearchResult(
                chunk_id="chunk_1",
                doc_name="TestSpec.pdf",
                page_start=1,
                page_end=1,
                text="The contractor shall complete work by specified deadline.",
                similarity_score=0.85,
                metadata={"section": "timeline"}
            ),
            SearchResult(
                chunk_id="chunk_2",
                doc_name="DailyReport.pdf", 
                page_start=3,
                page_end=3,
                text="Weather delays prevented work completion on schedule.",
                similarity_score=0.78,
                metadata={"date": "2023-02-14"}
            )
        ])
        return store
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        return MockLLMProvider()
    
    @pytest.fixture
    def rag_engine(self, mock_matter, mock_llm_provider, mock_vector_store):
        """Create RAG engine with mocked dependencies."""
        engine = RAGEngine(
            matter=mock_matter,
            llm_provider=mock_llm_provider,
            vector_store=mock_vector_store
        )
        return engine
    
    @pytest.mark.asyncio
    async def test_generate_answer_basic(self, rag_engine, mock_llm_provider):
        """Test basic answer generation."""
        query = "What are the timeline requirements?"
        
        # Set predefined response
        mock_response = """## Key Points
- Timeline specified in contract [TestSpec.pdf p.1]
- Weather delays documented [DailyReport.pdf p.3]

## Analysis
The contract clearly establishes timeline requirements, but weather conditions caused documented delays.

## Citations
- [TestSpec.pdf p.1] - Contract timeline requirements
- [DailyReport.pdf p.3] - Weather delay documentation

## Suggested Follow-ups
- Review weather impact analysis
- Assess schedule mitigation options"""
        
        mock_llm_provider.set_responses([
            mock_response,  # Main answer
            "Review weather data\nAnalyze delay claims",  # Follow-ups
            "[]"  # Knowledge extraction
        ])
        
        response = await rag_engine.generate_answer(query)
        
        assert isinstance(response, RAGResponse)
        assert "Timeline specified" in response.answer
        assert len(response.sources) == 2
        assert response.sources[0].doc == "TestSpec.pdf"
        assert response.sources[1].doc == "DailyReport.pdf"
        assert len(response.followups) >= 1
        
        # Verify LLM was called
        assert len(mock_llm_provider.generation_calls) >= 1
        main_call = mock_llm_provider.generation_calls[0]
        assert query in main_call["messages"][0]["content"]
    
    @pytest.mark.asyncio
    async def test_generate_answer_empty_query(self, rag_engine):
        """Test error handling for empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await rag_engine.generate_answer("")
    
    @pytest.mark.asyncio
    async def test_generate_answer_no_sources(self, rag_engine, mock_vector_store, mock_llm_provider):
        """Test handling when no sources are found."""
        # Mock empty search results
        mock_vector_store.search.return_value = []
        
        mock_llm_provider.set_responses([
            "No relevant documents found to answer this query.",
            "Gather more documentation",
            "[]"
        ])
        
        query = "Obscure technical question with no matches"
        response = await rag_engine.generate_answer(query)
        
        assert len(response.sources) == 0
        assert "No relevant documents" in response.answer
    
    @pytest.mark.asyncio
    async def test_citation_extraction_and_mapping(self, rag_engine, mock_llm_provider):
        """Test citation extraction and source mapping."""
        query = "Test citation mapping"
        
        # Response with specific citations
        mock_response = """## Key Points
- First finding [TestSpec.pdf p.1]
- Second finding [DailyReport.pdf p.3]
- Invalid citation [NonExistent.pdf p.99]

## Analysis
Analysis content here.

## Citations
- [TestSpec.pdf p.1] - Valid citation
- [DailyReport.pdf p.3] - Another valid citation
- [NonExistent.pdf p.99] - Invalid citation"""
        
        mock_llm_provider.set_responses([mock_response, "Follow up question", "[]"])
        
        response = await rag_engine.generate_answer(query)
        
        # Should have 2 valid sources (matching our mock vector store)
        assert len(response.sources) == 2
        assert any(s.doc == "TestSpec.pdf" for s in response.sources)
        assert any(s.doc == "DailyReport.pdf" for s in response.sources)
    
    @pytest.mark.asyncio
    async def test_knowledge_extraction(self, rag_engine, mock_llm_provider):
        """Test knowledge item extraction from answers."""
        query = "Test knowledge extraction"
        
        # Mock knowledge extraction response
        knowledge_response = json.dumps([
            {
                "type": "Event",
                "label": "Contract Delay",
                "date": "2023-02-14",
                "actors": ["General Contractor"],
                "doc_refs": [{"doc": "TestSpec.pdf", "page": 1}],
                "support_snippet": "Timeline requirements not met"
            }
        ])
        
        mock_llm_provider.set_responses([
            "Main answer content",
            "Follow up questions",
            knowledge_response
        ])
        
        response = await rag_engine.generate_answer(query)
        
        # Knowledge extraction is handled internally but doesn't affect response structure
        # This test mainly ensures no errors occur during extraction
        assert isinstance(response, RAGResponse)
        
        # Check that knowledge extraction was attempted
        assert len(mock_llm_provider.generation_calls) >= 3
    
    @pytest.mark.asyncio 
    async def test_followup_generation(self, rag_engine, mock_llm_provider):
        """Test follow-up question generation."""
        query = "What caused the delay?"
        
        followup_response = """Review schedule documentation
Analyze cost implications
Check contractor performance
Assess weather impact"""
        
        mock_llm_provider.set_responses([
            "Main answer about delays",
            followup_response,
            "[]"
        ])
        
        response = await rag_engine.generate_answer(query)
        
        assert len(response.followups) == 4
        assert "Review schedule documentation" in response.followups
        assert "Analyze cost implications" in response.followups
    
    @pytest.mark.asyncio
    async def test_error_handling_llm_failure(self, rag_engine, mock_llm_provider):
        """Test error handling when LLM fails."""
        # Mock LLM to raise exception
        async def failing_generate(*args, **kwargs):
            raise RuntimeError("LLM connection failed")
        
        mock_llm_provider.generate = failing_generate
        
        query = "Test error handling"
        
        with pytest.raises(RuntimeError, match="RAG pipeline failed"):
            await rag_engine.generate_answer(query)
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self, rag_engine):
        """Test parameter validation and limits."""
        query = "Valid query"
        
        # Test with various parameter combinations
        response = await rag_engine.generate_answer(
            query,
            k=5,
            k_memory=3,
            max_tokens=500,
            temperature=0.1
        )
        
        assert isinstance(response, RAGResponse)
        
        # Verify vector search was called with correct k value
        rag_engine.vector_store.search.assert_called_with(query, k=5)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_construction_delay_analysis_scenario(self):
        """Test a realistic construction delay analysis scenario."""
        # This test would use more realistic data and responses
        # to verify the complete workflow
        
        # Mock search results representing actual construction documents
        search_results = [
            SearchResult(
                chunk_id="spec_123",
                doc_name="ProjectSpecifications2023.pdf",
                page_start=15,
                page_end=15,
                text="Section 3.2: All concrete work shall be completed by March 15, 2023. Weather delays exceeding 5 consecutive days shall trigger schedule revision procedures.",
                similarity_score=0.92,
                metadata={"section": "timeline", "spec_version": "v2.1"}
            ),
            SearchResult(
                chunk_id="log_456",
                doc_name="DailyConstructionLog_Feb2023.pdf",
                page_start=28,
                page_end=28, 
                text="February 12-18, 2023: Continuous rainfall prevented concrete pour operations. Site access limited due to muddy conditions. Superintendent notified of potential schedule impact.",
                similarity_score=0.88,
                metadata={"date_range": "2023-02-12_2023-02-18", "weather": "rain"}
            )
        ]
        
        # Create realistic matter setup
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_root = Path(temp_dir) / "delay_analysis_matter"
            matter_root.mkdir()
            paths = MatterPaths.from_root(matter_root)
            for path in [paths.docs, paths.vectors]:
                path.mkdir(parents=True, exist_ok=True)
            
            matter = Matter(
                id="delay-analysis-001",
                name="Downtown Office Delay Analysis", 
                slug="downtown-office-delay-analysis",
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=paths
            )
            
            # Mock vector store with realistic results
            mock_vector_store = Mock(spec=VectorStore)
            mock_vector_store.search = AsyncMock(return_value=search_results)
            
            # Mock LLM with realistic construction claims response
            mock_llm = MockLLMProvider()
            realistic_response = """## Key Points
- Contract specifies March 15, 2023 completion deadline [ProjectSpecifications2023.pdf p.15]
- Weather delays documented February 12-18, 2023 (7 consecutive days) [DailyConstructionLog_Feb2023.pdf p.28]
- Weather delay exceeds 5-day threshold triggering schedule revision procedures [ProjectSpecifications2023.pdf p.15]

## Analysis
The project specifications clearly establish a March 15, 2023 completion deadline for concrete work. The daily construction logs document a weather delay from February 12-18, 2023, totaling 7 consecutive days of rainfall that prevented concrete operations and limited site access.

This delay exceeds the 5-consecutive-day threshold specified in Section 3.2 of the project specifications, which triggers mandatory schedule revision procedures. The contractor appears to have properly documented the weather conditions and notified the superintendent of potential schedule impacts.

## Citations
- [ProjectSpecifications2023.pdf p.15] - Contract completion deadline and weather delay procedures
- [DailyConstructionLog_Feb2023.pdf p.28] - Documentation of February 2023 weather delays

## Suggested Follow-ups
- Review schedule revision procedures outlined in contract specifications
- Analyze impact on overall project timeline and subsequent trade coordination
- Assess whether contractor filed formal schedule revision request
- Determine if weather delays qualify for time extension under contract terms"""
            
            mock_llm.set_responses([
                realistic_response,
                "Review formal schedule revision request\nAnalyze impact on critical path\nAssess time extension eligibility\nReview weather documentation completeness",
                json.dumps([
                    {
                        "type": "Event",
                        "label": "Weather Delay - February 2023",
                        "date": "2023-02-12",
                        "actors": ["General Contractor", "Superintendent"],
                        "doc_refs": [{"doc": "DailyConstructionLog_Feb2023.pdf", "page": 28}],
                        "support_snippet": "7 consecutive days rainfall prevented concrete operations"
                    },
                    {
                        "type": "Issue", 
                        "label": "Schedule Revision Required",
                        "date": "2023-02-18",
                        "actors": ["General Contractor"],
                        "doc_refs": [{"doc": "ProjectSpecifications2023.pdf", "page": 15}],
                        "support_snippet": "Weather delay exceeds 5-day contractual threshold"
                    }
                ])
            ])
            
            # Create RAG engine and test
            rag_engine = RAGEngine(
                matter=matter,
                llm_provider=mock_llm,
                vector_store=mock_vector_store
            )
            
            query = "What weather delays occurred and do they qualify for schedule revision under the contract?"
            
            response = await rag_engine.generate_answer(query, k=8)
            
            # Verify comprehensive response
            assert "March 15, 2023" in response.answer
            assert "February 12-18" in response.answer
            assert "7 consecutive days" in response.answer
            assert "exceeds the 5-day threshold" in response.answer
            
            # Verify proper citations
            assert len(response.sources) == 2
            spec_source = next(s for s in response.sources if "Specifications" in s.doc)
            log_source = next(s for s in response.sources if "ConstructionLog" in s.doc)
            assert spec_source.page_start == 15
            assert log_source.page_start == 28
            
            # Verify actionable follow-ups
            assert len(response.followups) == 4
            assert any("schedule revision" in f.lower() for f in response.followups)
            assert any("time extension" in f.lower() for f in response.followups)
            
            # Verify search was called properly
            mock_vector_store.search.assert_called_once_with(query, k=8)