"""
End-to-end workflow integration tests.

Tests complete workflow from matter creation to Q&A, document upload → OCR → 
parsing → chunking → embedding → search, validates citation accuracy end-to-end,
and tests matter switching with data isolation.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
import json
import time

from app.matters import matter_manager as matter_service
from app.ingest import IngestionPipeline
from app.vectors import VectorStore
from app.rag import RAGEngine
from app.chat_history import ChatHistoryManager
from app.llm.ollama_provider import OllamaProvider, OllamaEmbeddings
from app.models import Matter, MatterPaths


class TestEndToEndWorkflow:
    """End-to-end workflow integration tests."""
    
    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary base directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "LettaClaims"
            base_path.mkdir()
            yield base_path
    
    @pytest.fixture
    def mock_pdf_content(self):
        """Mock PDF content for testing."""
        return {
            "file_path": "/tmp/construction_contract.pdf",
            "content": b"%PDF-1.4\nMock PDF content for testing",
            "pages": [
                {
                    "page_no": 1,
                    "text": "CONSTRUCTION CONTRACT\n\nThis agreement is between Owner ABC Corp and Contractor XYZ Ltd for the installation of a dry well system. The work must be completed according to specifications in Section 3.",
                    "doc_name": "construction_contract.pdf"
                },
                {
                    "page_no": 2,
                    "text": "SECTION 3: DRY WELL SPECIFICATIONS\n\nThe dry well must be installed at least 10 feet from the foundation. Installation depth should be 8-12 feet depending on soil conditions. Materials must meet ASTM standards.",
                    "doc_name": "construction_contract.pdf"
                },
                {
                    "page_no": 3,
                    "text": "SECTION 4: TIMELINE AND RESPONSIBILITIES\n\nContractor XYZ Ltd is responsible for obtaining all permits. Work must be completed by March 15, 2023. Any delays due to weather will extend the deadline proportionally.",
                    "doc_name": "construction_contract.pdf"
                }
            ]
        }
    
    @pytest.fixture
    def mock_ollama_services(self):
        """Mock Ollama services for testing."""
        # Mock embedding service
        mock_embeddings = AsyncMock()
        mock_embeddings.embed.return_value = [[0.1] * 768] * 10  # 768-dim embeddings
        
        # Mock generation service
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = (
            "Based on the construction contract, the dry well installation requirements are specified in Section 3. "
            "The dry well must be installed at least 10 feet from the foundation according to [construction_contract.pdf p.2]. "
            "Installation depth should be 8-12 feet depending on soil conditions, and materials must meet ASTM standards."
        )
        
        return mock_embeddings, mock_provider
    
    @pytest.fixture
    def mock_ocr_processor(self):
        """Mock OCR processor for testing."""
        mock_processor = AsyncMock()
        mock_processor.process_pdf.return_value = Mock(
            success=True,
            output_file="/tmp/construction_contract_ocr.pdf",
            processing_time=2.5,
            ocr_mode="skip_text",
            pages_processed=3
        )
        return mock_processor
    
    @pytest.mark.integration
    async def test_complete_workflow_new_matter(self, temp_base_dir, mock_pdf_content, mock_ollama_services, mock_ocr_processor):
        """Test complete workflow starting with new matter creation."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            with patch('app.ingest.OCRProcessor', return_value=mock_ocr_processor):
                with patch('app.ingest.PDFParser') as mock_parser:
                    with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings):
                        with patch('app.rag.provider_manager.get_active_provider', return_value=mock_provider):
                            
                            # Mock PDF parser
                            mock_parser.return_value.extract_pages.return_value = [
                                Mock(page_no=p["page_no"], text=p["text"], doc_name=p["doc_name"])
                                for p in mock_pdf_content["pages"]
                            ]
                            mock_parser.return_value.get_document_metadata.return_value = Mock(
                                total_pages=3,
                                file_size=1024000,
                                title="Construction Contract"
                            )
                            
                            # Step 1: Create new matter
                            matter = await matter_service.create_matter("Construction Dry Well Claim")
                            
                            assert matter is not None
                            assert matter.name == "Construction Dry Well Claim"
                            assert matter.slug == "construction-dry-well-claim"
                            assert matter.paths.root.exists()
                            assert matter.paths.docs.exists()
                            assert matter.paths.vectors.exists()
                            
                            # Step 2: Initialize services for the matter
                            vector_store = VectorStore(matter.paths.vectors)
                            rag_engine = RAGEngine()
                            chat_manager = ChatHistoryManager(matter.paths.chat)
                            ingestion_pipeline = IngestionPipeline()
                            
                            # Step 3: Process document (OCR → Parse → Chunk → Embed)
                            with patch('pathlib.Path.exists', return_value=True):
                                with patch('shutil.copy2'):  # Mock file copying
                                    stats = await ingestion_pipeline.process_document(
                                        pdf_path=Path(mock_pdf_content["file_path"]),
                                        matter=matter
                                    )
                            
                            assert stats.success is True
                            assert stats.total_pages == 3
                            assert stats.total_chunks > 0
                            
                            # Step 4: Query the system
                            query = "What are the dry well installation requirements?"
                            
                            response = await rag_engine.generate_answer(
                                query=query,
                                matter=matter,
                                k=5
                            )
                            
                            assert response is not None
                            assert "dry well" in response["answer"].lower()
                            assert "10 feet from the foundation" in response["answer"]
                            assert len(response["sources"]) > 0
                            assert len(response["followups"]) > 0
                            
                            # Step 5: Verify citations are accurate
                            citations_found = []
                            for source in response["sources"]:
                                if source["doc"] == "construction_contract.pdf":
                                    citations_found.append(source)
                            
                            assert len(citations_found) > 0
                            
                            # Step 6: Save chat history
                            from app.chat_history import ChatMessage
                            
                            user_message = ChatMessage(
                                role="user",
                                content=query,
                                timestamp=asyncio.get_event_loop().time()
                            )
                            assistant_message = ChatMessage(
                                role="assistant", 
                                content=response["answer"],
                                timestamp=asyncio.get_event_loop().time(),
                                metadata={
                                    "sources": response["sources"],
                                    "followups": response["followups"]
                                }
                            )
                            
                            chat_manager.add_message(user_message)
                            chat_manager.add_message(assistant_message)
                            
                            # Step 7: Verify chat history persistence
                            history = chat_manager.get_history()
                            assert len(history) == 2
                            assert history[0].role == "user"
                            assert history[1].role == "assistant"
                            assert history[0].content == query
    
    @pytest.mark.integration
    async def test_multi_document_workflow(self, temp_base_dir, mock_ollama_services, mock_ocr_processor):
        """Test workflow with multiple documents."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        # Mock multiple documents
        documents = [
            {
                "name": "contract.pdf",
                "pages": [
                    {"page_no": 1, "text": "Main construction contract with dry well installation requirements.", "doc_name": "contract.pdf"}
                ]
            },
            {
                "name": "specifications.pdf", 
                "pages": [
                    {"page_no": 1, "text": "Technical specifications for dry well systems including depth and materials.", "doc_name": "specifications.pdf"}
                ]
            },
            {
                "name": "daily_log.pdf",
                "pages": [
                    {"page_no": 1, "text": "February 14, 2023: Dry well installation failed due to improper depth.", "doc_name": "daily_log.pdf"}
                ]
            }
        ]
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            with patch('app.ingest.OCRProcessor', return_value=mock_ocr_processor):
                with patch('app.ingest.PDFParser') as mock_parser:
                    with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings):
                        with patch('app.rag.provider_manager.get_active_provider', return_value=mock_provider):
                            
                            # Create matter
                            matter = await matter_service.create_matter("Multi-Document Case")
                            
                            # Process each document
                            ingestion_pipeline = IngestionPipeline()
                            all_stats = []
                            
                            for doc in documents:
                                # Mock parser for this document
                                mock_parser.return_value.extract_pages.return_value = [
                                    Mock(page_no=p["page_no"], text=p["text"], doc_name=p["doc_name"])
                                    for p in doc["pages"]
                                ]
                                mock_parser.return_value.get_document_metadata.return_value = Mock(
                                    total_pages=len(doc["pages"]),
                                    file_size=512000,
                                    title=doc["name"]
                                )
                                
                                with patch('pathlib.Path.exists', return_value=True):
                                    with patch('shutil.copy2'):
                                        stats = await ingestion_pipeline.process_document(
                                            pdf_path=Path(f"/tmp/{doc['name']}"),
                                            matter=matter
                                        )
                                        all_stats.append(stats)
                            
                            # All documents should be processed successfully
                            assert all(stats.success for stats in all_stats)
                            
                            # Query across all documents
                            rag_engine = RAGEngine()
                            
                            # Mock provider to return diverse sources
                            mock_provider.generate.return_value = (
                                "The dry well installation requirements are found in the contract and specifications. "
                                "According to [contract.pdf p.1] and [specifications.pdf p.1], proper depth is crucial. "
                                "The daily log shows that on February 14, 2023, the installation failed due to improper depth "
                                "as noted in [daily_log.pdf p.1]."
                            )
                            
                            response = await rag_engine.generate_answer(
                                query="What caused the dry well installation failure?",
                                matter=matter,
                                k=10
                            )
                            
                            # Should have sources from multiple documents
                            doc_sources = set()
                            for source in response["sources"]:
                                doc_sources.add(source["doc"])
                            
                            assert len(doc_sources) >= 2  # At least 2 different documents
    
    @pytest.mark.integration
    async def test_matter_switching_isolation(self, temp_base_dir, mock_ollama_services, mock_ocr_processor):
        """Test that matter switching maintains data isolation."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            with patch('app.ingest.OCRProcessor', return_value=mock_ocr_processor):
                with patch('app.ingest.PDFParser') as mock_parser:
                    with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings):
                        with patch('app.rag.provider_manager.get_active_provider', return_value=mock_provider):
                            
                            # Create two different matters
                            matter1 = await matter_service.create_matter("Matter One")
                            matter2 = await matter_service.create_matter("Matter Two")
                            
                            # Process different documents for each matter
                            ingestion_pipeline = IngestionPipeline()
                            
                            # Matter 1: Contract document
                            mock_parser.return_value.extract_pages.return_value = [
                                Mock(page_no=1, text="Matter 1 contract with specific terms.", doc_name="matter1_contract.pdf")
                            ]
                            mock_parser.return_value.get_document_metadata.return_value = Mock(
                                total_pages=1, file_size=100000, title="Matter 1 Contract"
                            )
                            
                            with patch('pathlib.Path.exists', return_value=True):
                                with patch('shutil.copy2'):
                                    await ingestion_pipeline.process_document(
                                        pdf_path=Path("/tmp/matter1_contract.pdf"),
                                        matter=matter1
                                    )
                            
                            # Matter 2: Different contract
                            mock_parser.return_value.extract_pages.return_value = [
                                Mock(page_no=1, text="Matter 2 specifications for different project.", doc_name="matter2_specs.pdf")
                            ]
                            mock_parser.return_value.get_document_metadata.return_value = Mock(
                                total_pages=1, file_size=120000, title="Matter 2 Specs"
                            )
                            
                            with patch('pathlib.Path.exists', return_value=True):
                                with patch('shutil.copy2'):
                                    await ingestion_pipeline.process_document(
                                        pdf_path=Path("/tmp/matter2_specs.pdf"),
                                        matter=matter2
                                    )
                            
                            # Switch to matter 1 and query
                            await matter_service.switch_matter(matter1.id)
                            
                            rag_engine = RAGEngine()
                            mock_provider.generate.return_value = "Answer based on matter 1 contract [matter1_contract.pdf p.1]."
                            
                            response1 = await rag_engine.generate_answer(
                                query="What are the contract terms?",
                                matter=matter1,
                                k=5
                            )
                            
                            # Switch to matter 2 and query
                            await matter_service.switch_matter(matter2.id)
                            mock_provider.generate.return_value = "Answer based on matter 2 specifications [matter2_specs.pdf p.1]."
                            
                            response2 = await rag_engine.generate_answer(
                                query="What are the specifications?", 
                                matter=matter2,
                                k=5
                            )
                            
                            # Responses should be isolated (contain different documents)
                            matter1_docs = {source["doc"] for source in response1["sources"]}
                            matter2_docs = {source["doc"] for source in response2["sources"]}
                            
                            # Should have no overlap between matters
                            assert len(matter1_docs.intersection(matter2_docs)) == 0
                            
                            # Verify chat histories are separate
                            chat1 = ChatHistoryManager(matter1.paths.chat)
                            chat2 = ChatHistoryManager(matter2.paths.chat)
                            
                            from app.chat_history import ChatMessage
                            
                            # Add different messages to each matter
                            chat1.add_message(ChatMessage(
                                role="user",
                                content="Question about matter 1",
                                timestamp=asyncio.get_event_loop().time()
                            ))
                            
                            chat2.add_message(ChatMessage(
                                role="user", 
                                content="Question about matter 2",
                                timestamp=asyncio.get_event_loop().time()
                            ))
                            
                            # Histories should be isolated
                            history1 = chat1.get_history()
                            history2 = chat2.get_history()
                            
                            assert len(history1) == 1
                            assert len(history2) == 1
                            assert history1[0].content == "Question about matter 1"
                            assert history2[0].content == "Question about matter 2"
    
    @pytest.mark.integration
    async def test_citation_accuracy_validation(self, temp_base_dir, mock_ollama_services, mock_ocr_processor):
        """Test end-to-end citation accuracy."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        # Mock document with known content
        test_pages = [
            {
                "page_no": 1,
                "text": "Section 1: This document covers the installation requirements for dry well systems.",
                "doc_name": "installation_guide.pdf"
            },
            {
                "page_no": 2, 
                "text": "Section 2: The minimum depth for dry well installation is 10 feet below grade.",
                "doc_name": "installation_guide.pdf"
            },
            {
                "page_no": 3,
                "text": "Section 3: All materials must be approved by the local building department.",
                "doc_name": "installation_guide.pdf"
            }
        ]
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            with patch('app.ingest.OCRProcessor', return_value=mock_ocr_processor):
                with patch('app.ingest.PDFParser') as mock_parser:
                    with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings):
                        with patch('app.rag.provider_manager.get_active_provider', return_value=mock_provider):
                            
                            # Setup document
                            mock_parser.return_value.extract_pages.return_value = [
                                Mock(page_no=p["page_no"], text=p["text"], doc_name=p["doc_name"])
                                for p in test_pages
                            ]
                            mock_parser.return_value.get_document_metadata.return_value = Mock(
                                total_pages=3, file_size=200000, title="Installation Guide"
                            )
                            
                            # Create matter and process document
                            matter = await matter_service.create_matter("Citation Test Case")
                            
                            ingestion_pipeline = IngestionPipeline()
                            with patch('pathlib.Path.exists', return_value=True):
                                with patch('shutil.copy2'):
                                    await ingestion_pipeline.process_document(
                                        pdf_path=Path("/tmp/installation_guide.pdf"),
                                        matter=matter
                                    )
                            
                            # Mock provider response with specific citations
                            mock_provider.generate.return_value = (
                                "The dry well installation requirements are detailed in the installation guide. "
                                "According to [installation_guide.pdf p.2], the minimum depth is 10 feet below grade. "
                                "Additionally, [installation_guide.pdf p.3] states that all materials must be approved "
                                "by the local building department."
                            )
                            
                            # Query the system
                            rag_engine = RAGEngine()
                            response = await rag_engine.generate_answer(
                                query="What are the dry well installation requirements?",
                                matter=matter,
                                k=5
                            )
                            
                            # Validate citations exist and are accurate
                            assert "installation_guide.pdf p.2" in response["answer"]
                            assert "installation_guide.pdf p.3" in response["answer"]
                            
                            # Verify sources contain the cited pages
                            cited_pages = set()
                            for source in response["sources"]:
                                if source["doc"] == "installation_guide.pdf":
                                    cited_pages.update(range(source["page_start"], source["page_end"] + 1))
                            
                            assert 2 in cited_pages  # Page 2 should be in sources
                            assert 3 in cited_pages  # Page 3 should be in sources
                            
                            # Verify source content matches citations
                            for source in response["sources"]:
                                if source["page_start"] == 2:
                                    assert "10 feet below grade" in source["text"]
                                elif source["page_start"] == 3:
                                    assert "building department" in source["text"]
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_performance_with_large_document(self, temp_base_dir, mock_ollama_services, mock_ocr_processor):
        """Test workflow performance with large document."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        # Generate large document content
        large_pages = []
        for page_num in range(1, 51):  # 50 pages
            page_text = f"Page {page_num}: " + "Construction project details and specifications. " * 50
            large_pages.append({
                "page_no": page_num,
                "text": page_text,
                "doc_name": "large_specifications.pdf"
            })
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            with patch('app.ingest.OCRProcessor', return_value=mock_ocr_processor):
                with patch('app.ingest.PDFParser') as mock_parser:
                    with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings):
                        with patch('app.rag.provider_manager.get_active_provider', return_value=mock_provider):
                            
                            # Setup large document
                            mock_parser.return_value.extract_pages.return_value = [
                                Mock(page_no=p["page_no"], text=p["text"], doc_name=p["doc_name"])
                                for p in large_pages
                            ]
                            mock_parser.return_value.get_document_metadata.return_value = Mock(
                                total_pages=50, file_size=5000000, title="Large Specifications"
                            )
                            
                            # Time the processing
                            start_time = time.time()
                            
                            matter = await matter_service.create_matter("Large Document Test")
                            
                            ingestion_pipeline = IngestionPipeline()
                            with patch('pathlib.Path.exists', return_value=True):
                                with patch('shutil.copy2'):
                                    stats = await ingestion_pipeline.process_document(
                                        pdf_path=Path("/tmp/large_specifications.pdf"),
                                        matter=matter
                                    )
                            
                            processing_time = time.time() - start_time
                            
                            # Should complete processing within reasonable time
                            assert processing_time < 30.0  # Less than 30 seconds (mocked)
                            assert stats.success is True
                            assert stats.total_pages == 50
                            assert stats.total_chunks > 50  # Should create multiple chunks
                            
                            # Query should also be performant
                            start_time = time.time()
                            
                            rag_engine = RAGEngine()
                            mock_provider.generate.return_value = "Large document analysis result with citations."
                            
                            response = await rag_engine.generate_answer(
                                query="What are the key project specifications?",
                                matter=matter,
                                k=10
                            )
                            
                            query_time = time.time() - start_time
                            
                            assert query_time < 5.0  # Query should be fast
                            assert response is not None
                            assert len(response["sources"]) > 0
    
    @pytest.mark.integration
    async def test_error_recovery_workflow(self, temp_base_dir, mock_ollama_services):
        """Test workflow error recovery scenarios."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            
            # Test 1: OCR failure recovery
            mock_ocr_fail = AsyncMock()
            mock_ocr_fail.process_pdf.side_effect = Exception("OCR failed")
            
            with patch('app.ingest.OCRProcessor', return_value=mock_ocr_fail):
                matter = await matter_service.create_matter("Error Recovery Test")
                
                ingestion_pipeline = IngestionPipeline()
                with patch('pathlib.Path.exists', return_value=True):
                    stats = await ingestion_pipeline.process_document(
                        pdf_path=Path("/tmp/test.pdf"),
                        matter=matter
                    )
                
                # Should handle OCR failure gracefully
                assert stats.success is False
                assert "OCR failed" in stats.error_message
            
            # Test 2: Embedding failure recovery
            mock_embeddings_fail = AsyncMock()
            mock_embeddings_fail.embed.side_effect = Exception("Embedding failed")
            
            with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings_fail):
                with patch('app.ingest.OCRProcessor') as mock_ocr_success:
                    mock_ocr_success.return_value.process_pdf.return_value = Mock(
                        success=True, output_file="/tmp/test_ocr.pdf"
                    )
                    
                    with patch('app.ingest.PDFParser') as mock_parser:
                        mock_parser.return_value.extract_pages.return_value = [
                            Mock(page_no=1, text="Test content", doc_name="test.pdf")
                        ]
                        
                        with patch('pathlib.Path.exists', return_value=True):
                            with patch('shutil.copy2'):
                                stats = await ingestion_pipeline.process_document(
                                    pdf_path=Path("/tmp/test.pdf"),
                                    matter=matter
                                )
                        
                        # Should handle embedding failure
                        assert stats.success is False
                        assert "embed" in stats.error_message.lower()
            
            # Test 3: Query with no documents
            rag_engine = RAGEngine()
            
            # Empty matter (no documents processed)
            empty_matter = await matter_service.create_matter("Empty Matter")
            
            mock_provider.generate.return_value = "I don't have any documents to reference for this matter."
            
            response = await rag_engine.generate_answer(
                query="What are the requirements?",
                matter=empty_matter,
                k=5
            )
            
            # Should handle gracefully
            assert response is not None
            assert len(response["sources"]) == 0  # No sources available
            assert "don't have any documents" in response["answer"]