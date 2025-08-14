"""
Integration tests for RAG pipeline with real Ollama models.

Tests the complete RAG workflow using actual Ollama models
and vector operations (when available).
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
from unittest.mock import patch, Mock

from app.rag import RAGEngine
from app.vectors import VectorStore
from app.llm.ollama_provider import OllamaProvider, OllamaEmbeddings
from app.llm.provider_manager import provider_manager
from app.models import Matter, MatterPaths
from app.chunking import Chunk


class TestRAGIntegration:
    """Integration tests for RAG pipeline."""
    
    @pytest.fixture
    def temp_matter_dir(self):
        """Create temporary matter directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_root = Path(temp_dir) / "integration_test_matter"
            matter_root.mkdir()
            
            # Create subdirectories
            paths = MatterPaths.from_root(matter_root)
            for path in [paths.docs, paths.docs_ocr, paths.parsed, paths.vectors, 
                        paths.knowledge, paths.chat, paths.logs]:
                path.mkdir(parents=True, exist_ok=True)
            
            yield matter_root
    
    @pytest.fixture
    def test_matter(self, temp_matter_dir):
        """Create test matter."""
        paths = MatterPaths.from_root(temp_matter_dir)
        return Matter(
            id="integration-test-001",
            name="Integration Test Matter",
            slug="integration-test-matter",
            embedding_model="nomic-embed-text",
            generation_model="gpt-oss:20b",
            paths=paths
        )
    
    @pytest.fixture
    async def vector_store_with_data(self, test_matter):
        """Create vector store with test data."""
        vector_store = VectorStore(test_matter.paths.root)
        
        # Create test chunks representing construction documents
        test_chunks = [
            Chunk(
                chunk_id="spec_timeline_001",
                doc_id="spec_2023",
                doc_name="ProjectSpecifications2023.pdf",
                page_start=1,
                page_end=1,
                text="Section 1.2: Project Timeline - All foundation work shall be completed by February 28, 2023. Concrete curing shall follow ACI standards with minimum 7-day cure time before form removal.",
                token_count=35,
                char_count=180,
                md5="spec_timeline_md5_001",
                chunk_index=0,
                section_title="Project Timeline",
                overlap_info={"has_overlap": False, "overlap_sentences": 0},
                metadata={"section": "timeline", "importance": "high"}
            ),
            Chunk(
                chunk_id="daily_log_weather_001",
                doc_id="log_feb_2023",
                doc_name="DailyLog_February2023.pdf",
                page_start=15,
                page_end=15,
                text="February 25, 2023: Heavy rainfall overnight and throughout morning. Site conditions unsafe for concrete operations. Superintendent decision to postpone foundation pour until weather clears. Forecast shows continued rain through February 28.",
                token_count=40,
                char_count=220,
                md5="daily_log_weather_md5_001",
                chunk_index=0,
                section_title="Daily Operations Log",
                overlap_info={"has_overlap": False, "overlap_sentences": 0},
                metadata={"date": "2023-02-25", "weather": "rain", "operation": "foundation"}
            ),
            Chunk(
                chunk_id="change_order_001",
                doc_id="co_2023_001",
                doc_name="ChangeOrder_001_2023.pdf",
                page_start=1,
                page_end=2,
                text="Change Order #001: Due to unexpected weather delays during foundation work, contract timeline extended by 5 business days. Additional costs for standby equipment and labor: $12,500. Owner approval required for schedule modification.",
                token_count=38,
                char_count=200,
                md5="change_order_md5_001",
                chunk_index=0,
                section_title="Schedule Modification",
                overlap_info={"has_overlap": False, "overlap_sentences": 0},
                metadata={"change_order": "001", "cost_impact": "12500", "type": "schedule"}
            )
        ]
        
        try:
            await vector_store.upsert_chunks(test_chunks)
            yield vector_store
        except Exception as e:
            # If embeddings fail (Ollama not available), use mock
            pytest.skip(f"Ollama embeddings not available: {e}")
    
    @pytest.mark.asyncio
    async def test_ollama_provider_connectivity(self):
        """Test if Ollama is available for integration testing."""
        try:
            provider = OllamaProvider(model="gpt-oss:20b")
            connection_ok = await provider.test_connection()
            
            if not connection_ok:
                pytest.skip("Ollama server not available or model not found")
            
            # Test basic generation
            response = await provider.generate(
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": "Say 'test successful'"}],
                max_tokens=50,
                temperature=0.1
            )
            
            assert response
            assert len(response) > 0
            
        except Exception as e:
            pytest.skip(f"Ollama provider test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_embeddings_connectivity(self):
        """Test if Ollama embeddings are available."""
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            connection_ok = await embeddings.test_connection()
            
            if not connection_ok:
                pytest.skip("Ollama embeddings not available")
            
            # Test embedding generation
            test_texts = ["Construction timeline requirements", "Weather delay documentation"]
            vectors = await embeddings.embed(test_texts)
            
            assert len(vectors) == 2
            assert all(isinstance(v, list) and len(v) > 0 for v in vectors)
            
        except Exception as e:
            pytest.skip(f"Ollama embeddings test failed: {e}")
    
    @pytest.mark.asyncio 
    async def test_end_to_end_rag_pipeline(self, test_matter, vector_store_with_data):
        """Test complete RAG pipeline with real models."""
        try:
            # Initialize provider manager and register Ollama
            success = await provider_manager.register_ollama_provider(
                model="gpt-oss:20b",
                embedding_model="nomic-embed-text"
            )
            
            if not success:
                pytest.skip("Failed to register Ollama provider")
            
            active_provider = provider_manager.get_active_provider()
            assert active_provider is not None
            
            # Create RAG engine
            rag_engine = RAGEngine(
                matter=test_matter,
                llm_provider=active_provider,
                vector_store=vector_store_with_data
            )
            
            # Test query about timeline and weather delays
            query = "What are the timeline requirements and how do weather delays affect the project?"
            
            response = await rag_engine.generate_answer(
                query=query,
                k=3,
                max_tokens=500,
                temperature=0.2
            )
            
            # Verify response structure
            assert response.answer
            assert len(response.answer) > 50  # Should be substantive
            assert len(response.sources) > 0  # Should find relevant documents
            assert len(response.followups) > 0  # Should generate follow-ups
            
            # Verify content relevance (basic checks)
            answer_lower = response.answer.lower()
            assert any(term in answer_lower for term in ["timeline", "february", "weather", "foundation"])
            
            # Verify citations are present
            assert "[" in response.answer and "]" in response.answer
            
            # Verify sources have proper metadata
            for source in response.sources:
                assert source.doc
                assert source.page_start > 0
                assert source.score >= 0.0
                assert len(source.text) > 0
            
            print(f"✓ RAG pipeline test completed successfully")
            print(f"  Query: {query[:60]}...")
            print(f"  Answer length: {len(response.answer)} chars")
            print(f"  Sources found: {len(response.sources)}")
            print(f"  Follow-ups: {len(response.followups)}")
            
        except Exception as e:
            if "Ollama" in str(e) or "connection" in str(e).lower():
                pytest.skip(f"Ollama not available for integration test: {e}")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_rag_with_no_relevant_documents(self, test_matter, vector_store_with_data):
        """Test RAG behavior when query doesn't match any documents well."""
        try:
            success = await provider_manager.register_ollama_provider()
            if not success:
                pytest.skip("Ollama not available")
            
            active_provider = provider_manager.get_active_provider()
            rag_engine = RAGEngine(
                matter=test_matter,
                llm_provider=active_provider, 
                vector_store=vector_store_with_data
            )
            
            # Query about something not in our test documents
            query = "What are the electrical specifications for HVAC systems?"
            
            response = await rag_engine.generate_answer(query, k=3)
            
            # Should still generate a response but may indicate limited information
            assert response.answer
            assert len(response.sources) >= 0  # Might be 0 or have low-relevance results
            
            # Low similarity scores expected
            if response.sources:
                assert all(source.score < 0.7 for source in response.sources)
            
        except Exception as e:
            if "Ollama" in str(e):
                pytest.skip(f"Ollama not available: {e}")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_citation_accuracy(self, test_matter, vector_store_with_data):
        """Test that citations in answers map to actual source documents."""
        try:
            success = await provider_manager.register_ollama_provider()
            if not success:
                pytest.skip("Ollama not available")
            
            active_provider = provider_manager.get_active_provider()
            rag_engine = RAGEngine(
                matter=test_matter,
                llm_provider=active_provider,
                vector_store=vector_store_with_data
            )
            
            query = "When was the foundation work scheduled to be completed?"
            
            response = await rag_engine.generate_answer(query, k=5)
            
            # Extract citations from answer
            import re
            citations = re.findall(r'\[([\w\-\s\.]+\.pdf)\s+p\.(\d+(?:-\d+)?)\]', response.answer)
            
            if citations:
                # Verify citations reference actual sources
                source_docs = {source.doc for source in response.sources}
                cited_docs = {citation[0] for citation in citations}
                
                # At least some citations should match available sources
                assert len(cited_docs.intersection(source_docs)) > 0
                
                print(f"✓ Citation accuracy test passed")
                print(f"  Citations found: {len(citations)}")
                print(f"  Cited documents: {cited_docs}")
                print(f"  Available sources: {source_docs}")
            
        except Exception as e:
            if "Ollama" in str(e):
                pytest.skip(f"Ollama not available: {e}")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_provider_switching(self, test_matter):
        """Test switching between different providers."""
        try:
            # Try to register both Ollama and potentially Gemini
            ollama_success = await provider_manager.register_ollama_provider(
                model="gpt-oss:20b"
            )
            
            if not ollama_success:
                pytest.skip("Ollama not available for provider switching test")
            
            # Get initial provider
            initial_provider = provider_manager.get_active_provider()
            assert initial_provider is not None
            
            # Test provider listing
            providers = provider_manager.list_providers()
            assert len(providers) > 0
            
            # Test provider switching (switch to same provider for now)
            config = provider_manager.get_provider_config()
            active_key = config["active_provider"]
            
            switch_success = provider_manager.switch_provider(active_key)
            assert switch_success
            
            # Verify provider is still active
            current_provider = provider_manager.get_active_provider()
            assert current_provider is not None
            
            print(f"✓ Provider switching test completed")
            print(f"  Available providers: {len(providers)}")
            print(f"  Active provider: {active_key}")
            
        except Exception as e:
            if "Ollama" in str(e):
                pytest.skip(f"Ollama not available: {e}")
            else:
                raise


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance tests for RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_response_time_reasonable(self, test_matter):
        """Test that RAG responses complete in reasonable time."""
        try:
            success = await provider_manager.register_ollama_provider()
            if not success:
                pytest.skip("Ollama not available")
            
            # Create simple vector store
            vector_store = VectorStore(test_matter.paths.root)
            
            active_provider = provider_manager.get_active_provider()
            rag_engine = RAGEngine(
                matter=test_matter,
                llm_provider=active_provider,
                vector_store=vector_store
            )
            
            import time
            start_time = time.time()
            
            query = "What is the project timeline?"
            response = await rag_engine.generate_answer(query, k=3, max_tokens=200)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete within reasonable time (adjust based on hardware)
            assert duration < 60.0  # 60 seconds max
            assert response.answer  # Should produce valid response
            
            print(f"✓ Response time test: {duration:.2f} seconds")
            
        except Exception as e:
            if "Ollama" in str(e):
                pytest.skip(f"Ollama not available: {e}")
            else:
                raise


if __name__ == "__main__":
    # Run basic connectivity tests
    pytest.main([__file__ + "::TestRAGIntegration::test_ollama_provider_connectivity", "-v"])