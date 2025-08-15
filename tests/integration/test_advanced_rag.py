"""
Integration tests for Advanced RAG Features.

Tests end-to-end integration of citation management, follow-up generation,
hybrid retrieval, and quality metrics in the RAG pipeline.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from app.rag import RAGEngine, RAGResponse
from app.models import SourceChunk
from app.vectors import VectorStore
from app.models import Matter, MatterPaths, KnowledgeItem
from app.citation_manager import CitationManager, CitationMetrics
from app.followup_engine import FollowupEngine, FollowupContext
from app.hybrid_retrieval import HybridRetrieval, create_retrieval_context
from app.quality_metrics import QualityAnalyzer, QualityThresholds
from app.llm.base import LLMProvider


class MockLLMProvider:
    """Mock LLM provider for integration testing."""
    
    def __init__(self):
        self.generation_calls = []
        self.responses = {
            'main': """## Key Points
- Contract specifies completion by June 30, 2023 [Contract_2023.pdf p.5]
- Weather delays documented on February 15th [Daily_Log_Feb15.pdf p.2]
- Structural specifications require AISC compliance [Spec_Section3.pdf p.10]

## Analysis
The project faced significant challenges due to adverse weather conditions and
complex structural requirements. The contract deadline of June 30, 2023 became
increasingly difficult to meet after the February weather delays. The AISC
compliance requirements for structural work added additional complexity to the
project timeline and quality control processes.

## Citations
- [Contract_2023.pdf p.5] - Project completion deadline specification
- [Daily_Log_Feb15.pdf p.2] - Documentation of weather-related delays
- [Spec_Section3.pdf p.10] - Structural compliance requirements

## Suggested Follow-ups
- What contingency plans exist for weather-related delays?
- How do AISC compliance requirements affect the construction schedule?
- What penalties apply if the June deadline is missed?
- Should we engage a scheduling expert to assess timeline recovery?""",
            
            'followup': """What additional weather contingency provisions should be reviewed?
Should we engage a structural engineer for AISC compliance analysis?
How can we accelerate the remaining work to meet the June deadline?
What are the liquidated damages implications of missing the deadline?""",
            
            'extraction': """[
    {
        "type": "Event",
        "label": "Weather Delay February 15",
        "date": "2023-02-15",
        "actors": ["General Contractor", "Owner"],
        "doc_refs": [{"doc": "Daily_Log_Feb15.pdf", "page": 2}],
        "support_snippet": "Severe weather conditions prevented concrete pour operations"
    },
    {
        "type": "Issue", 
        "label": "AISC Compliance Requirements",
        "actors": ["Structural Engineer", "Steel Contractor"],
        "doc_refs": [{"doc": "Spec_Section3.pdf", "page": 10}],
        "support_snippet": "All structural steel must meet AISC standards for compliance"
    },
    {
        "type": "Fact",
        "label": "Contract Completion Deadline June 30 2023",
        "date": "2023-06-30",
        "actors": ["Owner", "General Contractor"],
        "doc_refs": [{"doc": "Contract_2023.pdf", "page": 5}],
        "support_snippet": "Project shall achieve substantial completion by June 30, 2023"
    }
]"""
        }
        self.current_response_type = 'main'
    
    async def generate(self, system: str, messages: List[dict], max_tokens: int = 900, temperature: float = 0.2) -> str:
        """Mock generate method with context-aware responses."""
        self.generation_calls.append({
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        })
        
        # Determine response type based on system prompt and content
        if "extract structured items" in system.lower() or "json" in messages[0]["content"].lower():
            return self.responses['extraction']
        elif "follow-up" in system.lower() or "suggestions" in messages[0]["content"].lower():
            return self.responses['followup']
        else:
            return self.responses['main']


class MockVectorStore:
    """Mock vector store for integration testing."""
    
    def __init__(self, matter_path: Path):
        self.matter_path = matter_path
        self.mock_results = [
            {
                "chunk_id": "chunk_1",
                "doc_name": "Contract_2023.pdf",
                "page_start": 5,
                "page_end": 7,
                "text": "The project shall achieve substantial completion by June 30, 2023. Time is of the essence for this contract. Any delays beyond this date may result in liquidated damages as specified in Section 8.3.",
                "similarity_score": 0.89,
                "metadata": {"doc_type": "contract", "section": "timeline"}
            },
            {
                "chunk_id": "chunk_2", 
                "doc_name": "Daily_Log_Feb15.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "February 15, 2023: Severe weather conditions with heavy rain and high winds prevented all outdoor work. Concrete pour scheduled for foundation work had to be postponed. Weather forecast shows improvement by February 18th.",
                "similarity_score": 0.78,
                "metadata": {"doc_type": "log", "date": "2023-02-15"}
            },
            {
                "chunk_id": "chunk_3",
                "doc_name": "Spec_Section3.pdf", 
                "page_start": 10,
                "page_end": 12,
                "text": "All structural steel components shall conform to AISC specifications and standards. Quality control testing must be performed in accordance with AWS D1.1 welding code. Documentation of compliance is required for each structural element.",
                "similarity_score": 0.85,
                "metadata": {"doc_type": "specification", "section": "structural"}
            }
        ]
    
    async def search(self, query: str, k: int = 8, filter_metadata: dict = None):
        """Mock search method."""
        from app.vectors import SearchResult
        
        results = []
        for i, mock_result in enumerate(self.mock_results[:k]):
            result = SearchResult(
                chunk_id=mock_result["chunk_id"],
                doc_name=mock_result["doc_name"],
                page_start=mock_result["page_start"],
                page_end=mock_result["page_end"],
                text=mock_result["text"],
                similarity_score=mock_result["similarity_score"],
                metadata=mock_result["metadata"]
            )
            results.append(result)
        
        return results


class MockLettaAdapter:
    """Mock Letta adapter for integration testing."""
    
    def __init__(self, matter_path: Path, matter_name: str, matter_id: str):
        self.matter_path = matter_path
        self.matter_name = matter_name
        self.matter_id = matter_id
        self.memory_items = [
            KnowledgeItem(
                type="Event",
                label="Project Kickoff",
                date="2023-01-15",
                actors=["Owner", "General Contractor", "Architect"],
                doc_refs=[{"doc": "Contract_2023.pdf", "page": 1}],
                support_snippet="Project officially commenced with all parties present"
            ),
            KnowledgeItem(
                type="Issue",
                label="Weather Risk Assessment",
                actors=["Project Manager", "Scheduler"],
                doc_refs=[{"doc": "Risk_Assessment.pdf", "page": 3}],
                support_snippet="February weather patterns pose significant construction risks"
            )
        ]
    
    async def recall(self, query: str, top_k: int = 6) -> List[KnowledgeItem]:
        """Mock recall method."""
        # Return memory items relevant to query
        relevant_items = []
        query_lower = query.lower()
        
        for item in self.memory_items:
            if (any(word in item.label.lower() for word in query_lower.split()) or
                any(word in query_lower for word in item.label.lower().split())):
                relevant_items.append(item)
        
        return relevant_items[:top_k]
    
    async def upsert_interaction(self, user_query: str, llm_answer: str, sources: List[SourceChunk], extracted_facts: List[KnowledgeItem]) -> None:
        """Mock upsert method."""
        # Add extracted facts to memory
        self.memory_items.extend(extracted_facts[:3])  # Limit to prevent memory bloat in tests
    
    async def suggest_followups(self, user_query: str, llm_answer: str) -> List[str]:
        """Mock followup suggestions."""
        return [
            "What additional documentation should we review for weather contingencies?",
            "How can we mitigate the impact of structural compliance requirements?",
            "Should we consider requesting a time extension based on these delays?",
            "What expert analysis would strengthen our position on schedule impacts?"
        ]


@pytest.fixture
def temp_matter():
    """Create temporary matter for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        matter_root = Path(temp_dir) / "Matter_test"
        matter_paths = MatterPaths.from_root(matter_root)
        
        # Create directories
        for path in [matter_paths.docs, matter_paths.docs_ocr, matter_paths.parsed,
                    matter_paths.vectors, matter_paths.knowledge, matter_paths.chat, matter_paths.logs]:
            path.mkdir(parents=True, exist_ok=True)
        
        matter = Matter(
            id="test_matter_123",
            name="Test Construction Claim",
            slug="test-construction-claim",
            embedding_model="nomic-embed-text",
            generation_model="gpt-oss:20b",
            paths=matter_paths
        )
        
        yield matter


class TestAdvancedRAGIntegration:
    """Integration tests for advanced RAG features."""
    
    @pytest.mark.asyncio
    async def test_rag_engine_with_all_advanced_features(self, temp_matter):
        """Test RAG engine with all advanced features enabled."""
        # Setup
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        # Create RAG engine with advanced features
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            quality_thresholds=QualityThresholds(
                minimum_citation_coverage=0.5,
                minimum_source_diversity=0.3,
                minimum_answer_completeness=0.6,
                minimum_confidence_score=0.4
            ),
            enable_advanced_features=True
        )
        
        # Mock Letta adapter
        rag_engine.letta_adapter = MockLettaAdapter(
            temp_matter.paths.root, temp_matter.name, temp_matter.id
        )
        
        # Test query
        query = "What are the main challenges affecting the project timeline?"
        conversation_history = ["What is the project start date?", "Project started January 15, 2023"]
        recent_documents = ["Contract_2023.pdf", "Daily_Log_Feb15.pdf"]
        
        # Generate answer
        response = await rag_engine.generate_answer(
            query=query,
            k=8,
            k_memory=6,
            max_tokens=900,
            temperature=0.2,
            conversation_history=conversation_history,
            recent_documents=recent_documents,
            enable_mmr=True
        )
        
        # Verify response structure
        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert len(response.sources) > 0
        assert len(response.followups) > 0
        assert len(response.used_memory) >= 0
        
        # Verify advanced features
        assert response.citation_mappings is not None
        assert response.citation_metrics is not None
        assert response.quality_metrics is not None
        assert response.processing_time is not None
        assert response.processing_time > 0
        
        # Verify quality metrics
        quality = response.quality_metrics
        assert 0.0 <= quality.overall_quality <= 1.0
        assert 0.0 <= quality.confidence_score <= 1.0
        assert quality.response_length == len(response.answer)
        assert isinstance(quality.meets_minimum_standards, bool)
        
        # Verify citation analysis
        citations = response.citation_metrics
        assert citations.total_citations >= 0
        assert citations.valid_citations <= citations.total_citations
        assert 0.0 <= citations.accuracy_score <= 1.0
        assert 0.0 <= citations.completeness_score <= 1.0
        
        # Verify LLM was called appropriately
        assert len(mock_llm.generation_calls) >= 2  # Main generation + extraction/followups
    
    @pytest.mark.asyncio
    async def test_rag_engine_fallback_without_advanced_features(self, temp_matter):
        """Test RAG engine fallback when advanced features disabled."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        # Create RAG engine without advanced features
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=False
        )
        
        # Mock Letta adapter
        rag_engine.letta_adapter = MockLettaAdapter(
            temp_matter.paths.root, temp_matter.name, temp_matter.id
        )
        
        # Generate answer
        response = await rag_engine.generate_answer(
            query="What caused the project delays?",
            k=5
        )
        
        # Verify basic response
        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert len(response.sources) > 0
        assert len(response.followups) > 0
        
        # Verify advanced features are None (fallback mode)
        assert response.citation_mappings is None
        assert response.citation_metrics is None
        assert response.quality_metrics is None
    
    @pytest.mark.asyncio
    async def test_citation_manager_integration(self, temp_matter):
        """Test citation manager integration in RAG pipeline."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        response = await rag_engine.generate_answer("What are the contract requirements?")
        
        # Verify citation analysis
        assert response.citation_mappings is not None
        assert len(response.citation_mappings) > 0
        
        # Check citation mapping validity
        valid_mappings = [m for m in response.citation_mappings if m.is_valid]
        assert len(valid_mappings) > 0
        
        # Verify citation metrics calculation
        assert response.citation_metrics is not None
        assert response.citation_metrics.total_citations > 0
        assert 0.0 <= response.citation_metrics.accuracy_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_followup_engine_integration(self, temp_matter):
        """Test follow-up engine integration in RAG pipeline."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        # Mock Letta adapter
        rag_engine.letta_adapter = MockLettaAdapter(
            temp_matter.paths.root, temp_matter.name, temp_matter.id
        )
        
        response = await rag_engine.generate_answer(
            "What caused the delays?",
            conversation_history=["Previous question", "Previous answer"]
        )
        
        # Verify enhanced follow-ups
        assert len(response.followups) > 0
        assert all(len(followup) > 10 for followup in response.followups)  # Meaningful length
        assert all(followup.endswith('?') for followup in response.followups)  # Proper question format
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_integration(self, temp_matter):
        """Test hybrid retrieval integration in RAG pipeline."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        # Mock Letta adapter with memory
        rag_engine.letta_adapter = MockLettaAdapter(
            temp_matter.paths.root, temp_matter.name, temp_matter.id
        )
        
        response = await rag_engine.generate_answer(
            "What weather issues affected construction?",
            conversation_history=["What was the project timeline?"],
            recent_documents=["Daily_Log_Feb15.pdf"],
            enable_mmr=True
        )
        
        # Verify retrieval worked
        assert len(response.sources) > 0
        assert len(response.used_memory) > 0  # Memory should be recalled
        
        # Check for memory integration in response
        assert any("weather" in source.text.lower() for source in response.sources)
    
    @pytest.mark.asyncio
    async def test_quality_analyzer_integration(self, temp_matter):
        """Test quality analyzer integration in RAG pipeline.""" 
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        # Custom thresholds for testing
        thresholds = QualityThresholds(
            minimum_citation_coverage=0.4,
            minimum_source_diversity=0.2,
            minimum_answer_completeness=0.5,
            minimum_confidence_score=0.3
        )
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            quality_thresholds=thresholds,
            enable_advanced_features=True
        )
        
        response = await rag_engine.generate_answer("Analyze the construction delays")
        
        # Verify quality analysis
        quality = response.quality_metrics
        assert quality is not None
        assert 0.0 <= quality.overall_quality <= 1.0
        assert 0.0 <= quality.confidence_score <= 1.0
        assert quality.response_length > 0
        assert isinstance(quality.quality_warnings, list)
        
        # Verify improvement suggestions if quality is low
        if response.quality_metrics.overall_quality < 0.7:
            assert len(response.improvement_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_quality_retry_mechanism(self, temp_matter):
        """Test quality-based retry mechanism."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        # Set high thresholds to trigger retry
        thresholds = QualityThresholds(
            minimum_citation_coverage=0.9,
            minimum_source_diversity=0.8,
            minimum_answer_completeness=0.9,
            minimum_confidence_score=0.8
        )
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            quality_thresholds=thresholds,
            enable_advanced_features=True
        )
        
        # Mock Letta adapter
        rag_engine.letta_adapter = MockLettaAdapter(
            temp_matter.paths.root, temp_matter.name, temp_matter.id
        )
        
        # Test retry mechanism
        response = await rag_engine.generate_answer_with_quality_retry(
            "Simple question",
            max_retry_attempts=1
        )
        
        # Verify response generated
        assert isinstance(response, RAGResponse)
        assert response.quality_metrics is not None
        
        # Check if retry was attempted (multiple LLM calls)
        assert len(mock_llm.generation_calls) >= 2  # Initial + retry attempt
    
    def test_get_quality_insights(self, temp_matter):
        """Test quality insights retrieval."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        insights = rag_engine.get_quality_insights()
        
        assert isinstance(insights, dict)
        assert insights['advanced_features_enabled'] is True
        assert 'quality_thresholds' in insights
        assert 'retrieval_stats' in insights
    
    def test_advanced_features_status(self, temp_matter):
        """Test advanced features status reporting."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        # Test with features enabled
        rag_engine_enabled = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        status_enabled = rag_engine_enabled.get_advanced_features_status()
        assert status_enabled['advanced_features_enabled'] is True
        assert status_enabled['citation_manager_available'] is True
        assert status_enabled['followup_engine_available'] is True
        assert status_enabled['hybrid_retrieval_available'] is True
        assert status_enabled['quality_analyzer_available'] is True
        
        # Test with features disabled
        rag_engine_disabled = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=False
        )
        
        status_disabled = rag_engine_disabled.get_advanced_features_status()
        assert status_disabled['advanced_features_enabled'] is False
        assert status_disabled['citation_manager_available'] is False
        assert status_disabled['followup_engine_available'] is False
        assert status_disabled['hybrid_retrieval_available'] is False
        assert status_disabled['quality_analyzer_available'] is False
    
    @pytest.mark.asyncio
    async def test_error_handling_advanced_features(self, temp_matter):
        """Test error handling when advanced features fail."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        # Mock failure in citation manager
        with patch.object(rag_engine.citation_manager, 'create_citation_mappings', 
                         side_effect=Exception("Citation analysis failed")):
            
            response = await rag_engine.generate_answer("Test query")
            
            # Should still generate response despite citation failure
            assert isinstance(response, RAGResponse)
            assert len(response.answer) > 0
            # Citation mappings might be empty due to failure
            assert response.citation_mappings is not None  # Should be list, possibly empty
    
    @pytest.mark.asyncio 
    async def test_memory_context_enhancement(self, temp_matter):
        """Test that memory context enhances responses."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        # Create Letta adapter with relevant memory
        letta_adapter = MockLettaAdapter(
            temp_matter.paths.root, temp_matter.name, temp_matter.id
        )
        
        # Add specific memory about weather issues
        letta_adapter.memory_items.append(
            KnowledgeItem(
                type="Event",
                label="February Weather Delay",
                date="2023-02-15", 
                actors=["General Contractor", "Weather Service"],
                doc_refs=[{"doc": "Daily_Log_Feb15.pdf", "page": 2}],
                support_snippet="Severe storm prevented all construction activities"
            )
        )
        
        rag_engine.letta_adapter = letta_adapter
        
        response = await rag_engine.generate_answer("What weather issues occurred?")
        
        # Verify memory was used
        assert len(response.used_memory) > 0
        assert any("weather" in item.label.lower() for item in response.used_memory)
        
        # Verify memory influenced the response
        assert "february" in response.answer.lower() or "weather" in response.answer.lower()


class TestCitationQualityIntegration:
    """Integration tests for citation quality analysis."""
    
    @pytest.mark.asyncio
    async def test_citation_accuracy_tracking(self, temp_matter):
        """Test end-to-end citation accuracy tracking."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        response = await rag_engine.generate_answer("What are the key contract dates?")
        
        # Verify citations were extracted and analyzed
        assert response.citation_metrics.total_citations > 0
        
        # Check that citations map to available sources
        for mapping in response.citation_mappings:
            assert mapping.source_chunk is not None
            assert mapping.citation is not None
            assert isinstance(mapping.confidence, float)
            assert 0.0 <= mapping.confidence <= 1.0


class TestPerformanceIntegration:
    """Integration tests for performance aspects of advanced features."""
    
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, temp_matter):
        """Test that processing time is accurately tracked."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        rag_engine = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm, 
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        response = await rag_engine.generate_answer("Performance test query")
        
        # Verify timing is tracked
        assert response.processing_time is not None
        assert response.processing_time > 0
        assert response.quality_metrics.processing_time == response.processing_time
    
    @pytest.mark.asyncio
    async def test_advanced_features_performance_impact(self, temp_matter):
        """Test performance impact of advanced features."""
        mock_llm = MockLLMProvider()
        mock_vector_store = MockVectorStore(temp_matter.paths.root)
        
        # Test with advanced features enabled
        rag_engine_advanced = RAGEngine(
            matter=temp_matter,
            llm_provider=mock_llm,
            vector_store=mock_vector_store,
            enable_advanced_features=True
        )
        
        response_advanced = await rag_engine_advanced.generate_answer("Test query")
        
        # Test with advanced features disabled
        rag_engine_basic = RAGEngine(
            matter=temp_matter,
            llm_provider=MockLLMProvider(),  # Fresh instance to reset call counts
            vector_store=MockVectorStore(temp_matter.paths.root),
            enable_advanced_features=False
        )
        
        response_basic = await rag_engine_basic.generate_answer("Test query")
        
        # Both should generate responses
        assert isinstance(response_advanced, RAGResponse)
        assert isinstance(response_basic, RAGResponse)
        
        # Advanced version should have additional features
        assert response_advanced.quality_metrics is not None
        assert response_basic.quality_metrics is None
        
        assert response_advanced.citation_metrics is not None
        assert response_basic.citation_metrics is None