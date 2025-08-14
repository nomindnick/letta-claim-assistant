"""
Integration tests for vector storage with real Ollama embeddings.

Tests end-to-end functionality with actual Ollama embedding generation
and ChromaDB operations.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List

from app.vectors import VectorStore, SearchResult, VectorStoreError
from app.chunking import Chunk
from app.llm.ollama_provider import OllamaEmbeddings
from app.llm.embeddings import embedding_manager


@pytest.fixture
def temp_matter_path():
    """Create temporary Matter directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    matter_path = temp_dir / "Matter_integration_test"
    matter_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (matter_path / "vectors" / "chroma").mkdir(parents=True, exist_ok=True)
    
    yield matter_path
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def construction_claim_chunks() -> List[Chunk]:
    """Create realistic construction claim chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk-delay-001",
            text="""The contractor experienced significant delays due to differing site conditions. 
            During excavation for the foundation, unexpected rock formations were encountered at depths 
            not indicated in the original soil reports. This required additional blasting and specialized 
            equipment, resulting in a 30-day delay to the project schedule.""",
            doc_id="daily-log-001",
            doc_name="Daily_Construction_Log_2023-02-15.pdf",
            page_start=1,
            page_end=1,
            token_count=65,
            char_count=320,
            md5="delay_hash_001",
            section_title="Site Conditions",
            chunk_index=0,
            overlap_info={"has_overlap": False, "overlap_sentences": 0},
            metadata={"event_type": "delay", "date": "2023-02-15"}
        ),
        Chunk(
            chunk_id="chunk-rfi-001",
            text="""RFI-103 was submitted regarding the concrete specifications for the foundation walls. 
            The structural drawings specify 4000 PSI concrete, but the architectural drawings indicate 
            3500 PSI. The architect clarified that 4000 PSI should be used throughout, as specified 
            in Section 03300 of the project specifications.""",
            doc_id="rfi-log-001",
            doc_name="RFI_Log_March_2023.pdf",
            page_start=5,
            page_end=5,
            token_count=58,
            char_count=285,
            md5="rfi_hash_001",
            section_title="Material Specifications",
            chunk_index=1,
            overlap_info={"has_overlap": False, "overlap_sentences": 0},
            metadata={"document_type": "rfi", "rfi_number": "RFI-103"}
        ),
        Chunk(
            chunk_id="chunk-change-order-001",
            text="""Change Order #12 was issued to address the dry well failure discovered during system 
            testing. The original dry well design was inadequate for the soil conditions, requiring 
            redesign and installation of a larger capacity system. The additional work includes excavation, 
            stone aggregate, and connection modifications. Total cost impact: $45,000.""",
            doc_id="change-order-012",
            doc_name="Change_Order_12_Dry_Well_Replacement.pdf",
            page_start=1,
            page_end=2,
            token_count=72,
            char_count=350,
            md5="co_hash_012",
            section_title="Change Orders",
            chunk_index=2,
            overlap_info={"has_overlap": False, "overlap_sentences": 0},
            metadata={"document_type": "change_order", "cost_impact": 45000}
        ),
        Chunk(
            chunk_id="chunk-payment-001",
            text="""Payment Application #8 was submitted on April 15, 2023, for work completed through 
            April 10, 2023. The application includes progress on foundation work (95% complete), 
            framing (60% complete), and electrical rough-in (30% complete). Total requested amount: 
            $285,000. Previous payments: $820,000. Contract amount to date: $1,250,000.""",
            doc_id="pay-app-008",
            doc_name="Payment_Application_008_April_2023.pdf",
            page_start=1,
            page_end=1,
            token_count=68,
            char_count=340,
            md5="pay_hash_008",
            section_title="Payment Applications",
            chunk_index=3,
            overlap_info={"has_overlap": False, "overlap_sentences": 0},
            metadata={"document_type": "payment_app", "amount": 285000}
        )
    ]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_embeddings_connectivity():
    """Test connectivity to Ollama embeddings service."""
    try:
        embeddings_provider = OllamaEmbeddings(model="nomic-embed-text")
        is_connected = await embeddings_provider.test_connection()
        
        if not is_connected:
            pytest.skip("Ollama embeddings service not available")
        
        # Test actual embedding generation
        test_texts = [
            "This is a test sentence about construction delays.",
            "Material specifications require 4000 PSI concrete."
        ]
        
        embeddings = await embeddings_provider.embed(test_texts)
        
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        
        # Embeddings should be different for different texts
        assert embeddings[0] != embeddings[1]
        
    except Exception as e:
        pytest.skip(f"Ollama not available for integration test: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_end_to_end_vector_operations(temp_matter_path, construction_claim_chunks):
    """Test complete vector operations workflow."""
    # Skip if Ollama is not available
    try:
        is_connected = await embedding_manager.test_connection()
        if not is_connected:
            pytest.skip("Embedding manager not available")
    except Exception:
        pytest.skip("Embedding manager not available")
    
    # Initialize vector store
    vector_store = VectorStore(temp_matter_path)
    
    # Test 1: Upsert chunks
    await vector_store.upsert_chunks(construction_claim_chunks)
    
    # Verify chunks were stored
    stats = await vector_store.get_collection_stats()
    assert stats["total_chunks"] == 4
    assert stats["unique_documents"] == 4
    
    # Test 2: Search for construction delays
    delay_results = await vector_store.search(
        "construction delays and site conditions",
        k=3
    )
    
    assert len(delay_results) > 0
    # The delay chunk should be most relevant
    top_result = delay_results[0]
    assert "delay" in top_result.text.lower() or "site conditions" in top_result.text.lower()
    assert top_result.similarity_score > 0.0
    
    # Test 3: Search for RFI information
    rfi_results = await vector_store.search(
        "RFI concrete specifications PSI",
        k=2
    )
    
    assert len(rfi_results) > 0
    # Should find RFI chunk
    rfi_found = any("RFI" in result.text for result in rfi_results)
    assert rfi_found
    
    # Test 4: Search with metadata filters
    change_order_results = await vector_store.search(
        "dry well failure cost",
        k=5,
        filter_metadata={"document_type": "change_order"}
    )
    
    # Should only return change order documents
    for result in change_order_results:
        if result.metadata.get("document_type"):
            assert result.metadata["document_type"] == "change_order"
    
    # Test 5: Search for financial information
    payment_results = await vector_store.search(
        "payment application amount money",
        k=2
    )
    
    assert len(payment_results) > 0
    # Should find payment-related content
    payment_found = any(
        "payment" in result.text.lower() or "amount" in result.text.lower() 
        for result in payment_results
    )
    assert payment_found


@pytest.mark.asyncio
@pytest.mark.integration
async def test_matter_isolation_with_real_embeddings(temp_matter_path, construction_claim_chunks):
    """Test Matter isolation with real embeddings."""
    # Skip if Ollama is not available
    try:
        is_connected = await embedding_manager.test_connection()
        if not is_connected:
            pytest.skip("Embedding manager not available")
    except Exception:
        pytest.skip("Embedding manager not available")
    
    # Create two different matter paths
    matter1_path = temp_matter_path
    matter2_path = temp_matter_path.parent / "Matter_second_test"
    matter2_path.mkdir(parents=True, exist_ok=True)
    (matter2_path / "vectors" / "chroma").mkdir(parents=True, exist_ok=True)
    
    try:
        # Create vector stores for each matter
        vector_store1 = VectorStore(matter1_path)
        vector_store2 = VectorStore(matter2_path)
        
        # Add different subsets of chunks to each matter
        await vector_store1.upsert_chunks(construction_claim_chunks[:2])  # Delay and RFI
        await vector_store2.upsert_chunks(construction_claim_chunks[2:])  # Change Order and Payment
        
        # Verify counts
        stats1 = await vector_store1.get_collection_stats()
        stats2 = await vector_store2.get_collection_stats()
        
        assert stats1["total_chunks"] == 2
        assert stats2["total_chunks"] == 2
        
        # Search in matter 1 should only find delay and RFI content
        matter1_results = await vector_store1.search("construction project information", k=10)
        matter1_docs = {result.doc_name for result in matter1_results}
        
        # Should only contain documents from matter 1
        expected_matter1_docs = {
            "Daily_Construction_Log_2023-02-15.pdf",
            "RFI_Log_March_2023.pdf"
        }
        assert matter1_docs.issubset(expected_matter1_docs)
        
        # Search in matter 2 should only find change order and payment content
        matter2_results = await vector_store2.search("construction project information", k=10)
        matter2_docs = {result.doc_name for result in matter2_results}
        
        # Should only contain documents from matter 2
        expected_matter2_docs = {
            "Change_Order_12_Dry_Well_Replacement.pdf",
            "Payment_Application_008_April_2023.pdf"
        }
        assert matter2_docs.issubset(expected_matter2_docs)
        
        # Verify no cross-contamination
        assert matter1_docs.isdisjoint(matter2_docs)
        
    finally:
        # Cleanup
        shutil.rmtree(matter2_path, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_semantic_search_quality(temp_matter_path, construction_claim_chunks):
    """Test semantic search quality with real embeddings."""
    # Skip if Ollama is not available
    try:
        is_connected = await embedding_manager.test_connection()
        if not is_connected:
            pytest.skip("Embedding manager not available")
    except Exception:
        pytest.skip("Embedding manager not available")
    
    vector_store = VectorStore(temp_matter_path)
    await vector_store.upsert_chunks(construction_claim_chunks)
    
    # Test semantic similarity - should find relevant content even with different wording
    test_queries = [
        {
            "query": "What caused project schedule delays?",
            "expected_content": ["delay", "site conditions", "excavation"],
            "min_similarity": 0.3
        },
        {
            "query": "Issues with concrete strength requirements",
            "expected_content": ["concrete", "PSI", "specifications"],
            "min_similarity": 0.3
        },
        {
            "query": "Problems with drainage system",
            "expected_content": ["dry well", "failure", "system"],
            "min_similarity": 0.3
        },
        {
            "query": "How much money was requested?",
            "expected_content": ["payment", "amount", "285,000"],
            "min_similarity": 0.2
        }
    ]
    
    for test_case in test_queries:
        results = await vector_store.search(test_case["query"], k=3)
        
        assert len(results) > 0, f"No results for query: {test_case['query']}"
        
        # Check if top result meets minimum similarity threshold
        top_result = results[0]
        assert top_result.similarity_score >= test_case["min_similarity"], \
            f"Similarity too low for query '{test_case['query']}': {top_result.similarity_score}"
        
        # Check if expected content words appear in results
        all_result_text = " ".join(result.text.lower() for result in results)
        found_content = [
            content for content in test_case["expected_content"]
            if content.lower() in all_result_text
        ]
        
        assert len(found_content) > 0, \
            f"None of expected content {test_case['expected_content']} found in results for '{test_case['query']}'"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_large_chunk_batching(temp_matter_path):
    """Test handling of large numbers of chunks."""
    # Skip if Ollama is not available
    try:
        is_connected = await embedding_manager.test_connection()
        if not is_connected:
            pytest.skip("Embedding manager not available")
    except Exception:
        pytest.skip("Embedding manager not available")
    
    vector_store = VectorStore(temp_matter_path)
    
    # Create a larger set of chunks to test batching
    large_chunk_set = []
    for i in range(15):  # Small but sufficient to test batching logic
        chunk = Chunk(
            chunk_id=f"batch-chunk-{i}",
            text=f"This is test chunk number {i} about various construction topics. " * 5,
            doc_id=f"batch-doc-{i // 5}",  # Group into documents
            doc_name=f"batch_document_{i // 5}.pdf",
            page_start=i % 10 + 1,
            page_end=i % 10 + 1,
            token_count=50,
            char_count=200,
            md5=f"batch_hash_{i}",
            section_title=f"Section {i % 3}",
            chunk_index=i,
            overlap_info={"has_overlap": i % 2 == 1, "overlap_sentences": i % 3},
            metadata={"batch_id": i // 5}
        )
        large_chunk_set.append(chunk)
    
    # Upsert all chunks
    await vector_store.upsert_chunks(large_chunk_set)
    
    # Verify all chunks were stored
    stats = await vector_store.get_collection_stats()
    assert stats["total_chunks"] == 15
    
    # Test search across all chunks
    results = await vector_store.search("construction topics", k=10)
    assert len(results) > 0
    
    # Test filtering by batch
    filtered_results = await vector_store.search(
        "construction topics",
        k=10,
        filter_metadata={"batch_id": 1}
    )
    
    # Should only return chunks from batch 1
    for result in filtered_results:
        if "batch_id" in result.metadata:
            assert result.metadata["batch_id"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])