"""
Multi-matter isolation integration tests.

Tests creation of multiple matters, uploading different documents to each,
verifying no cross-contamination in search results, and testing concurrent operations.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
import threading
import time
import concurrent.futures

from app.matters import matter_service
from app.vectors import VectorStore
from app.rag import RAGEngine
from app.chat_history import ChatHistoryManager
from app.letta_adapter import LettaAdapter
from app.ingest import IngestionPipeline
from app.models import Matter, MatterPaths


class TestMultiMatterIsolation:
    """Multi-matter isolation integration tests."""
    
    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary base directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "LettaClaims"
            base_path.mkdir()
            yield base_path
    
    @pytest.fixture
    def mock_ollama_services(self):
        """Mock Ollama services for testing."""
        mock_embeddings = AsyncMock()
        mock_embeddings.embed.return_value = [[0.1] * 768] * 10
        
        mock_provider = AsyncMock()
        return mock_embeddings, mock_provider
    
    @pytest.fixture
    def matter_test_data(self):
        """Test data for different matters."""
        return {
            "construction_claim": {
                "name": "Construction Claim - Dry Well",
                "documents": [
                    {
                        "name": "construction_contract.pdf",
                        "pages": [
                            {"page_no": 1, "text": "Construction contract for dry well installation at ABC Corp site. Contractor XYZ Ltd responsible for installation.", "doc_name": "construction_contract.pdf"},
                            {"page_no": 2, "text": "Dry well specifications: 10 feet from foundation, 8-12 feet depth, ASTM materials required.", "doc_name": "construction_contract.pdf"}
                        ]
                    },
                    {
                        "name": "failure_report.pdf", 
                        "pages": [
                            {"page_no": 1, "text": "Dry well failure occurred on February 14, 2023 due to improper installation depth of only 6 feet.", "doc_name": "failure_report.pdf"}
                        ]
                    }
                ]
            },
            "employment_dispute": {
                "name": "Employment Dispute - Jane Doe",
                "documents": [
                    {
                        "name": "employment_contract.pdf",
                        "pages": [
                            {"page_no": 1, "text": "Employment agreement between Jane Doe and Tech Corp for software engineer position starting January 1, 2023.", "doc_name": "employment_contract.pdf"},
                            {"page_no": 2, "text": "Salary: $120,000 annually. Benefits include health insurance, 401k matching, and 3 weeks vacation.", "doc_name": "employment_contract.pdf"}
                        ]
                    },
                    {
                        "name": "termination_notice.pdf",
                        "pages": [
                            {"page_no": 1, "text": "Jane Doe terminated on March 15, 2023 for alleged poor performance. No documentation of performance issues provided.", "doc_name": "termination_notice.pdf"}
                        ]
                    }
                ]
            },
            "property_dispute": {
                "name": "Property Dispute - Boundary Lines", 
                "documents": [
                    {
                        "name": "property_deed.pdf",
                        "pages": [
                            {"page_no": 1, "text": "Property deed for 123 Main Street showing boundary lines as surveyed in 1985. Eastern boundary marked by oak tree.", "doc_name": "property_deed.pdf"}
                        ]
                    },
                    {
                        "name": "survey_report.pdf",
                        "pages": [
                            {"page_no": 1, "text": "2023 survey shows eastern boundary has shifted 5 feet due to erosion. Oak tree landmark no longer accurate.", "doc_name": "survey_report.pdf"}
                        ]
                    }
                ]
            }
        }
    
    @pytest.mark.integration
    async def test_matter_creation_isolation(self, temp_base_dir, matter_test_data):
        """Test that creating multiple matters creates isolated directory structures."""
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            
            matters = {}
            
            # Create multiple matters
            for matter_key, matter_data in matter_test_data.items():
                matter = await matter_service.create_matter(matter_data["name"])
                matters[matter_key] = matter
                
                # Verify each matter has its own directory structure
                assert matter.paths.root.exists()
                assert matter.paths.docs.exists()
                assert matter.paths.vectors.exists()
                assert matter.paths.knowledge.exists()
                assert matter.paths.chat.exists()
                
                # Verify matter-specific subdirectories are isolated
                assert str(matter.paths.root).endswith(matter.slug)
                
                # Check that no matter directories overlap
                for other_key, other_matter in matters.items():
                    if other_key != matter_key:
                        assert matter.paths.root != other_matter.paths.root
                        assert not matter.paths.root.is_relative_to(other_matter.paths.root)
                        assert not other_matter.paths.root.is_relative_to(matter.paths.root)
    
    @pytest.mark.integration
    async def test_vector_store_isolation(self, temp_base_dir, matter_test_data, mock_ollama_services):
        """Test that vector stores for different matters are completely isolated."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings):
                
                matters = {}
                vector_stores = {}
                
                # Create matters and vector stores
                for matter_key, matter_data in matter_test_data.items():
                    matter = await matter_service.create_matter(matter_data["name"])
                    matters[matter_key] = matter
                    vector_stores[matter_key] = VectorStore(matter.paths.vectors)
                
                # Add different chunks to each vector store
                from app.chunking import Chunk
                
                # Construction matter chunks
                construction_chunks = [
                    Chunk(
                        id="const_chunk_1",
                        text="Dry well installation contract terms",
                        doc_id="construction_contract",
                        doc_name="construction_contract.pdf",
                        page_start=1,
                        page_end=1,
                        metadata={"matter": "construction"},
                        token_count=6
                    )
                ]
                
                # Employment matter chunks  
                employment_chunks = [
                    Chunk(
                        id="emp_chunk_1",
                        text="Employment agreement salary details",
                        doc_id="employment_contract",
                        doc_name="employment_contract.pdf",
                        page_start=1,
                        page_end=1,
                        metadata={"matter": "employment"},
                        token_count=5
                    )
                ]
                
                # Property matter chunks
                property_chunks = [
                    Chunk(
                        id="prop_chunk_1", 
                        text="Property boundary survey information",
                        doc_id="property_deed",
                        doc_name="property_deed.pdf",
                        page_start=1,
                        page_end=1,
                        metadata={"matter": "property"},
                        token_count=5
                    )
                ]
                
                # Insert chunks into respective vector stores
                await vector_stores["construction_claim"].upsert_chunks(construction_chunks)
                await vector_stores["employment_dispute"].upsert_chunks(employment_chunks)
                await vector_stores["property_dispute"].upsert_chunks(property_chunks)
                
                # Test isolation: search each vector store
                construction_results = await vector_stores["construction_claim"].search(
                    query="contract terms",
                    k=10
                )
                
                employment_results = await vector_stores["employment_dispute"].search(
                    query="contract terms", 
                    k=10
                )
                
                property_results = await vector_stores["property_dispute"].search(
                    query="contract terms",
                    k=10
                )
                
                # Each vector store should only return its own chunks
                construction_docs = {result.doc_name for result in construction_results}
                employment_docs = {result.doc_name for result in employment_results}
                property_docs = {result.doc_name for result in property_results}
                
                # No cross-contamination
                assert "construction_contract.pdf" in construction_docs
                assert "employment_contract.pdf" not in construction_docs
                assert "property_deed.pdf" not in construction_docs
                
                assert "employment_contract.pdf" in employment_docs
                assert "construction_contract.pdf" not in employment_docs
                assert "property_deed.pdf" not in employment_docs
                
                assert "property_deed.pdf" in property_docs
                assert "construction_contract.pdf" not in property_docs
                assert "employment_contract.pdf" not in property_docs
    
    @pytest.mark.integration
    async def test_chat_history_isolation(self, temp_base_dir, matter_test_data):
        """Test that chat histories are isolated between matters."""
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            
            matters = {}
            chat_managers = {}
            
            # Create matters and chat managers
            for matter_key, matter_data in matter_test_data.items():
                matter = await matter_service.create_matter(matter_data["name"])
                matters[matter_key] = matter
                chat_managers[matter_key] = ChatHistoryManager(matter.paths.chat)
            
            # Add different messages to each chat manager
            from app.chat_history import ChatMessage
            from datetime import datetime
            
            # Construction matter messages
            chat_managers["construction_claim"].add_message(ChatMessage(
                role="user",
                content="What caused the dry well failure?",
                timestamp=datetime.now(),
                metadata={"matter": "construction"}
            ))
            
            chat_managers["construction_claim"].add_message(ChatMessage(
                role="assistant",
                content="The dry well failed due to improper installation depth.",
                timestamp=datetime.now(),
                metadata={"matter": "construction"}
            ))
            
            # Employment matter messages
            chat_managers["employment_dispute"].add_message(ChatMessage(
                role="user",
                content="What was Jane Doe's salary?",
                timestamp=datetime.now(),
                metadata={"matter": "employment"}
            ))
            
            chat_managers["employment_dispute"].add_message(ChatMessage(
                role="assistant", 
                content="Jane Doe's salary was $120,000 annually.",
                timestamp=datetime.now(),
                metadata={"matter": "employment"}
            ))
            
            # Property matter messages
            chat_managers["property_dispute"].add_message(ChatMessage(
                role="user",
                content="Where is the property boundary?",
                timestamp=datetime.now(),
                metadata={"matter": "property"}
            ))
            
            # Verify isolation: each chat manager should only have its own messages
            construction_history = chat_managers["construction_claim"].get_history()
            employment_history = chat_managers["employment_dispute"].get_history()
            property_history = chat_managers["property_dispute"].get_history()
            
            assert len(construction_history) == 2
            assert len(employment_history) == 2
            assert len(property_history) == 1
            
            # Check content isolation
            construction_content = " ".join([msg.content for msg in construction_history])
            employment_content = " ".join([msg.content for msg in employment_history])
            property_content = " ".join([msg.content for msg in property_history])
            
            assert "dry well" in construction_content.lower()
            assert "jane doe" not in construction_content.lower()
            assert "boundary" not in construction_content.lower()
            
            assert "jane doe" in employment_content.lower()
            assert "dry well" not in employment_content.lower()
            assert "boundary" not in employment_content.lower()
            
            assert "boundary" in property_content.lower()
            assert "dry well" not in property_content.lower()
            assert "jane doe" not in property_content.lower()
    
    @pytest.mark.integration
    async def test_letta_agent_isolation(self, temp_base_dir, matter_test_data):
        """Test that Letta agents are isolated between matters."""
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            
            matters = {}
            letta_adapters = {}
            
            # Create matters and Letta adapters
            for matter_key, matter_data in matter_test_data.items():
                matter = await matter_service.create_matter(matter_data["name"])
                matters[matter_key] = matter
                
                # Mock Letta adapter for each matter
                with patch('app.letta_adapter.LettaAdapter') as mock_letta_class:
                    mock_adapter = AsyncMock()
                    mock_letta_class.return_value = mock_adapter
                    letta_adapters[matter_key] = mock_adapter
            
            # Simulate storing different knowledge in each agent
            construction_knowledge = [
                {
                    "type": "Event",
                    "label": "Dry well failure",
                    "date": "2023-02-14",
                    "actors": ["XYZ Ltd"],
                    "doc_refs": [{"doc": "failure_report.pdf", "page": 1}]
                }
            ]
            
            employment_knowledge = [
                {
                    "type": "Event",
                    "label": "Jane Doe termination",
                    "date": "2023-03-15", 
                    "actors": ["Tech Corp"],
                    "doc_refs": [{"doc": "termination_notice.pdf", "page": 1}]
                }
            ]
            
            property_knowledge = [
                {
                    "type": "Fact",
                    "label": "Boundary shift due to erosion",
                    "date": "2023-01-01",
                    "actors": ["Property Owner"],
                    "doc_refs": [{"doc": "survey_report.pdf", "page": 1}]
                }
            ]
            
            # Configure mock responses for knowledge recall
            letta_adapters["construction_claim"].recall.return_value = construction_knowledge
            letta_adapters["employment_dispute"].recall.return_value = employment_knowledge
            letta_adapters["property_dispute"].recall.return_value = property_knowledge
            
            # Test recall isolation
            construction_recall = await letta_adapters["construction_claim"].recall("dry well")
            employment_recall = await letta_adapters["employment_dispute"].recall("termination")
            property_recall = await letta_adapters["property_dispute"].recall("boundary")
            
            # Each adapter should return only its own knowledge
            construction_events = [item["label"] for item in construction_recall]
            employment_events = [item["label"] for item in employment_recall]
            property_events = [item["label"] for item in property_recall]
            
            assert "Dry well failure" in construction_events
            assert "Jane Doe termination" not in construction_events
            assert "Boundary shift due to erosion" not in construction_events
            
            assert "Jane Doe termination" in employment_events
            assert "Dry well failure" not in employment_events
            assert "Boundary shift due to erosion" not in employment_events
            
            assert "Boundary shift due to erosion" in property_events
            assert "Dry well failure" not in property_events
            assert "Jane Doe termination" not in property_events
    
    @pytest.mark.integration
    async def test_concurrent_matter_operations(self, temp_base_dir, matter_test_data, mock_ollama_services):
        """Test concurrent operations on different matters."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings):
                with patch('app.rag.provider_manager.get_active_provider', return_value=mock_provider):
                    
                    matters = {}
                    
                    # Create matters concurrently
                    async def create_matter_task(matter_key, matter_data):
                        return matter_key, await matter_service.create_matter(matter_data["name"])
                    
                    creation_tasks = [
                        create_matter_task(key, data)
                        for key, data in matter_test_data.items()
                    ]
                    
                    creation_results = await asyncio.gather(*creation_tasks)
                    
                    for matter_key, matter in creation_results:
                        matters[matter_key] = matter
                    
                    # Perform concurrent operations on different matters
                    async def concurrent_operations(matter_key, matter):
                        vector_store = VectorStore(matter.paths.vectors)
                        rag_engine = RAGEngine()
                        chat_manager = ChatHistoryManager(matter.paths.chat)
                        
                        # Add some test data
                        from app.chunking import Chunk
                        from app.chat_history import ChatMessage
                        
                        test_chunk = Chunk(
                            id=f"{matter_key}_chunk_1",
                            text=f"Test content for {matter.name}",
                            doc_id=f"{matter_key}_doc",
                            doc_name=f"{matter_key}.pdf",
                            page_start=1,
                            page_end=1,
                            metadata={"matter": matter_key},
                            token_count=5
                        )
                        
                        await vector_store.upsert_chunks([test_chunk])
                        
                        # Mock provider response specific to this matter
                        mock_provider.generate.return_value = f"Response for {matter.name}"
                        
                        # Perform RAG query
                        response = await rag_engine.generate_answer(
                            query=f"What is this matter about?",
                            matter=matter,
                            k=5
                        )
                        
                        # Add chat message
                        chat_manager.add_message(ChatMessage(
                            role="user",
                            content=f"Question about {matter.name}",
                            timestamp=asyncio.get_event_loop().time()
                        ))
                        
                        return {
                            "matter_key": matter_key,
                            "matter_name": matter.name,
                            "response": response,
                            "chat_count": len(chat_manager.get_history())
                        }
                    
                    # Run operations concurrently
                    operation_tasks = [
                        concurrent_operations(key, matter)
                        for key, matter in matters.items()
                    ]
                    
                    operation_results = await asyncio.gather(*operation_tasks)
                    
                    # Verify all operations completed successfully
                    assert len(operation_results) == len(matter_test_data)
                    
                    # Check that each matter's operations were isolated
                    for result in operation_results:
                        assert result["response"] is not None
                        assert result["chat_count"] == 1
                        assert result["matter_name"] in result["response"]["answer"]
    
    @pytest.mark.integration
    async def test_matter_switching_context_isolation(self, temp_base_dir, matter_test_data, mock_ollama_services):
        """Test that switching between matters maintains proper context isolation."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings):
                with patch('app.rag.provider_manager.get_active_provider', return_value=mock_provider):
                    
                    # Create two matters
                    matter1 = await matter_service.create_matter("Matter One")
                    matter2 = await matter_service.create_matter("Matter Two")
                    
                    # Set up different data for each matter
                    vector_store1 = VectorStore(matter1.paths.vectors)
                    vector_store2 = VectorStore(matter2.paths.vectors)
                    rag_engine = RAGEngine()
                    
                    from app.chunking import Chunk
                    
                    # Add matter-specific chunks
                    chunk1 = Chunk(
                        id="matter1_chunk",
                        text="Matter 1 specific content about construction",
                        doc_id="matter1_doc",
                        doc_name="matter1.pdf",
                        page_start=1,
                        page_end=1,
                        metadata={"matter": "matter1"},
                        token_count=6
                    )
                    
                    chunk2 = Chunk(
                        id="matter2_chunk",
                        text="Matter 2 specific content about employment",
                        doc_id="matter2_doc", 
                        doc_name="matter2.pdf",
                        page_start=1,
                        page_end=1,
                        metadata={"matter": "matter2"},
                        token_count=6
                    )
                    
                    await vector_store1.upsert_chunks([chunk1])
                    await vector_store2.upsert_chunks([chunk2])
                    
                    # Switch to matter 1 and query
                    await matter_service.switch_matter(matter1.id)
                    current_matter = matter_service.get_active_matter()
                    assert current_matter.id == matter1.id
                    
                    mock_provider.generate.return_value = "Answer based on matter 1 construction content [matter1.pdf p.1]"
                    
                    response1 = await rag_engine.generate_answer(
                        query="What is this about?",
                        matter=matter1,
                        k=5
                    )
                    
                    # Switch to matter 2 and query
                    await matter_service.switch_matter(matter2.id)
                    current_matter = matter_service.get_active_matter()
                    assert current_matter.id == matter2.id
                    
                    mock_provider.generate.return_value = "Answer based on matter 2 employment content [matter2.pdf p.1]"
                    
                    response2 = await rag_engine.generate_answer(
                        query="What is this about?",
                        matter=matter2,
                        k=5
                    )
                    
                    # Switch back to matter 1
                    await matter_service.switch_matter(matter1.id)
                    current_matter = matter_service.get_active_matter()
                    assert current_matter.id == matter1.id
                    
                    mock_provider.generate.return_value = "Answer based on matter 1 construction content [matter1.pdf p.1]"
                    
                    response1_again = await rag_engine.generate_answer(
                        query="What is this about?",
                        matter=matter1,
                        k=5
                    )
                    
                    # Verify context isolation
                    matter1_docs = {source["doc"] for source in response1["sources"]}
                    matter2_docs = {source["doc"] for source in response2["sources"]}
                    matter1_again_docs = {source["doc"] for source in response1_again["sources"]}
                    
                    # Each matter should only see its own documents
                    assert "matter1.pdf" in matter1_docs
                    assert "matter2.pdf" not in matter1_docs
                    
                    assert "matter2.pdf" in matter2_docs
                    assert "matter1.pdf" not in matter2_docs
                    
                    assert "matter1.pdf" in matter1_again_docs
                    assert "matter2.pdf" not in matter1_again_docs
                    
                    # Context should be consistent when switching back
                    assert matter1_docs == matter1_again_docs
    
    @pytest.mark.integration
    async def test_file_system_isolation(self, temp_base_dir, matter_test_data):
        """Test that file system operations are isolated between matters."""
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            
            matters = {}
            
            # Create matters
            for matter_key, matter_data in matter_test_data.items():
                matter = await matter_service.create_matter(matter_data["name"])
                matters[matter_key] = matter
                
                # Create test files in each matter's directories
                test_doc_path = matter.paths.docs / f"{matter_key}_test.pdf"
                test_chat_path = matter.paths.chat / "history.jsonl"
                test_vectors_path = matter.paths.vectors / "chroma" 
                test_vectors_path.mkdir(parents=True, exist_ok=True)
                
                # Write test content
                test_doc_path.write_text(f"Test document for {matter.name}")
                test_chat_path.write_text(f'{{"role":"user","content":"Test message for {matter.name}"}}\n')
                
                # Create vector database files
                (test_vectors_path / "test.db").write_text(f"Vector data for {matter.name}")
            
            # Verify file isolation
            for matter_key, matter in matters.items():
                # Check that each matter only has its own files
                doc_files = list(matter.paths.docs.glob("*.pdf"))
                chat_files = list(matter.paths.chat.glob("*.jsonl"))
                vector_files = list(matter.paths.vectors.rglob("*.db"))
                
                assert len(doc_files) == 1
                assert len(chat_files) == 1
                assert len(vector_files) == 1
                
                # Verify content is matter-specific
                doc_content = doc_files[0].read_text()
                chat_content = chat_files[0].read_text()
                vector_content = vector_files[0].read_text()
                
                assert matter.name in doc_content
                assert matter.name in chat_content
                assert matter.name in vector_content
                
                # Verify no cross-contamination
                for other_key, other_matter in matters.items():
                    if other_key != matter_key:
                        assert other_matter.name not in doc_content
                        assert other_matter.name not in chat_content
                        assert other_matter.name not in vector_content
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_large_scale_matter_isolation(self, temp_base_dir, mock_ollama_services):
        """Test isolation with many matters and operations."""
        mock_embeddings, mock_provider = mock_ollama_services
        
        with patch('app.matters.settings.get_base_path', return_value=temp_base_dir):
            with patch('app.vectors.OllamaEmbeddings', return_value=mock_embeddings):
                
                # Create many matters
                num_matters = 10
                matters = []
                
                for i in range(num_matters):
                    matter = await matter_service.create_matter(f"Test Matter {i:02d}")
                    matters.append(matter)
                
                # Perform operations on each matter
                async def matter_operations(matter_index, matter):
                    vector_store = VectorStore(matter.paths.vectors)
                    chat_manager = ChatHistoryManager(matter.paths.chat)
                    
                    # Add unique data to each matter
                    from app.chunking import Chunk
                    from app.chat_history import ChatMessage
                    
                    chunk = Chunk(
                        id=f"chunk_{matter_index}",
                        text=f"Unique content for matter {matter_index}",
                        doc_id=f"doc_{matter_index}",
                        doc_name=f"document_{matter_index}.pdf",
                        page_start=1,
                        page_end=1,
                        metadata={"matter_index": matter_index},
                        token_count=5
                    )
                    
                    await vector_store.upsert_chunks([chunk])
                    
                    message = ChatMessage(
                        role="user",
                        content=f"Message for matter {matter_index}",
                        timestamp=asyncio.get_event_loop().time()
                    )
                    
                    chat_manager.add_message(message)
                    
                    # Search for content
                    search_results = await vector_store.search(
                        query=f"content {matter_index}",
                        k=5
                    )
                    
                    # Get chat history
                    history = chat_manager.get_history()
                    
                    return {
                        "matter_index": matter_index,
                        "search_results": search_results,
                        "history": history
                    }
                
                # Run operations on all matters concurrently
                operation_tasks = [
                    matter_operations(i, matter)
                    for i, matter in enumerate(matters)
                ]
                
                results = await asyncio.gather(*operation_tasks)
                
                # Verify isolation across all matters
                for result in results:
                    matter_index = result["matter_index"]
                    
                    # Each matter should only find its own content
                    search_results = result["search_results"]
                    assert len(search_results) >= 1
                    
                    for search_result in search_results:
                        assert search_result.metadata["matter_index"] == matter_index
                        assert f"matter {matter_index}" in search_result.text
                    
                    # Each matter should only have its own chat history
                    history = result["history"]
                    assert len(history) == 1
                    assert f"matter {matter_index}" in history[0].content
                    
                    # Verify no cross-contamination
                    for other_result in results:
                        if other_result["matter_index"] != matter_index:
                            other_index = other_result["matter_index"]
                            
                            # Should not find other matter's content
                            for search_result in search_results:
                                assert f"matter {other_index}" not in search_result.text
                            
                            # Should not have other matter's messages
                            for message in history:
                                assert f"matter {other_index}" not in message.content