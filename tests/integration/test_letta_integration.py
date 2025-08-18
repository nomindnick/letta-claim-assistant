"""
Integration tests for Letta agent memory functionality.

Tests end-to-end memory persistence, matter isolation, and 
RAG integration with real Letta instances.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
import uuid

from app.letta_adapter import LettaAdapter
from app.models import KnowledgeItem, SourceChunk, Matter, MatterPaths
from app.rag import RAGEngine
from app.vectors import VectorStore
from app.llm.ollama_provider import OllamaProvider


class TestLettaIntegration:
    """Integration tests for Letta memory functionality."""
    
    @pytest.fixture
    def temp_matter_paths(self):
        """Create temporary matter directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Create two separate matter directories
            matter1_path = base_path / "Matter_test-claim-1"
            matter2_path = base_path / "Matter_test-claim-2"
            
            for path in [matter1_path, matter2_path]:
                (path / "knowledge" / "letta_state").mkdir(parents=True, exist_ok=True)
                (path / "vectors").mkdir(parents=True, exist_ok=True)
                (path / "docs").mkdir(parents=True, exist_ok=True)
                (path / "chat").mkdir(parents=True, exist_ok=True)
            
            yield matter1_path, matter2_path
    
    @pytest.fixture
    def sample_matters(self, temp_matter_paths):
        """Create sample Matter instances for testing."""
        matter1_path, matter2_path = temp_matter_paths
        
        matter1 = Matter(
            id="matter-1",
            name="Foundation Failure Claim",
            slug="foundation-failure-claim",
            created_at=datetime.now(),
            embedding_model="nomic-embed-text",
            generation_model="gpt-oss:20b",
            paths=MatterPaths.from_root(matter1_path)
        )
        
        matter2 = Matter(
            id="matter-2", 
            name="Roof Leak Claim",
            slug="roof-leak-claim",
            created_at=datetime.now(),
            embedding_model="nomic-embed-text",
            generation_model="gpt-oss:20b",
            paths=MatterPaths.from_root(matter2_path)
        )
        
        return matter1, matter2
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_letta_agent_initialization(self, sample_matters):
        """Test that Letta agents can be initialized successfully."""
        matter1, matter2 = sample_matters
        
        # Test adapter initialization
        adapter1 = LettaAdapter(
            matter_path=matter1.paths.root,
            matter_name=matter1.name,
            matter_id=matter1.id
        )
        
        adapter2 = LettaAdapter(
            matter_path=matter2.paths.root,
            matter_name=matter2.name,
            matter_id=matter2.id
        )
        
        # Verify adapters were created
        assert adapter1.matter_name == "Foundation Failure Claim"
        assert adapter2.matter_name == "Roof Leak Claim"
        assert adapter1.matter_id != adapter2.matter_id
        
        # Verify storage isolation
        assert adapter1.letta_path != adapter2.letta_path
        assert adapter1.letta_path.exists()
        assert adapter2.letta_path.exists()
        
        # If Letta is available, verify agents were created
        if adapter1.client and adapter1.agent_id:
            assert adapter1.agent_id != adapter2.agent_id
            
            # Verify agent configs were saved
            config1_path = adapter1.letta_path / "agent_config.json"
            config2_path = adapter2.letta_path / "agent_config.json"
            assert config1_path.exists()
            assert config2_path.exists()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memory_persistence_lifecycle(self, sample_matters):
        """Test complete memory lifecycle: store, recall, persist."""
        matter1, _ = sample_matters
        
        adapter = LettaAdapter(
            matter_path=matter1.paths.root,
            matter_name=matter1.name,
            matter_id=matter1.id
        )
        
        if not adapter.client or not adapter.agent_id:
            pytest.skip("Letta not available for integration testing")
        
        # Test data
        user_query = "What caused the foundation failure in Building A?"
        llm_answer = """The foundation failure in Building A was primarily caused by inadequate soil analysis during the design phase. The geotechnical report failed to identify the presence of expansive clay soils at the site. Key findings:
        
1. Soil samples were taken at insufficient depth
2. Seasonal moisture variations were not considered
3. Foundation design used standard bearing capacity without site-specific analysis
        
This resulted in differential settlement and structural damage requiring complete foundation reconstruction."""
        
        sources = [
            SourceChunk(
                doc="Geotechnical_Report_2022.pdf",
                page_start=5,
                page_end=7,
                text="Soil samples indicate clay content of 45% with high plasticity index",
                score=0.89
            ),
            SourceChunk(
                doc="Foundation_Inspection_2023.pdf",
                page_start=12,
                page_end=12,
                text="Differential settlement measurements show 2.5 inch variance across foundation",
                score=0.82
            )
        ]
        
        extracted_facts = [
            KnowledgeItem(
                type="Issue",
                label="Inadequate soil analysis",
                date="2022-03-15",
                actors=["Geotechnical Engineer", "Design Team"],
                doc_refs=[{"doc": "Geotechnical_Report_2022.pdf", "page": 5}],
                support_snippet="Soil samples taken at insufficient depth"
            ),
            KnowledgeItem(
                type="Event",
                label="Foundation failure Building A",
                date="2023-02-14",
                actors=["ABC Construction", "Owner"],
                doc_refs=[{"doc": "Foundation_Inspection_2023.pdf", "page": 12}],
                support_snippet="Differential settlement of 2.5 inches observed"
            )
        ]
        
        # Step 1: Store interaction
        await adapter.upsert_interaction(
            user_query=user_query,
            llm_answer=llm_answer,
            sources=sources,
            extracted_facts=extracted_facts
        )
        
        # Step 2: Test immediate recall
        recalled_items = await adapter.recall("foundation failure", top_k=5)
        
        # Should find stored knowledge items
        assert len(recalled_items) > 0
        
        # Look for our stored facts
        found_issue = any(item.label == "Inadequate soil analysis" for item in recalled_items)
        found_event = any(item.label == "Foundation failure Building A" for item in recalled_items)
        
        # At least one of our facts should be recalled
        assert found_issue or found_event
        
        # Step 3: Test memory stats
        stats = await adapter.get_memory_stats()
        assert stats["status"] == "active"
        assert stats["memory_items"] > 0
        assert stats["agent_id"] == adapter.agent_id
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_matter_isolation_verification(self, sample_matters):
        """Test that memory is completely isolated between matters."""
        matter1, matter2 = sample_matters
        
        adapter1 = LettaAdapter(
            matter_path=matter1.paths.root,
            matter_name=matter1.name,
            matter_id=matter1.id
        )
        
        adapter2 = LettaAdapter(
            matter_path=matter2.paths.root,
            matter_name=matter2.name,
            matter_id=matter2.id
        )
        
        if not (adapter1.client and adapter1.agent_id and adapter2.client and adapter2.agent_id):
            pytest.skip("Letta not available for integration testing")
        
        # Store different facts in each matter
        foundation_facts = [
            KnowledgeItem(
                type="Event",
                label="Foundation failure",
                date="2023-02-14",
                actors=["ABC Construction"],
                support_snippet="Foundation showed differential settlement"
            )
        ]
        
        roof_facts = [
            KnowledgeItem(
                type="Event",
                label="Roof membrane failure",
                date="2023-03-10",
                actors=["XYZ Roofing"],
                support_snippet="EPDM membrane developed tears"
            )
        ]
        
        # Store foundation facts in matter 1
        await adapter1.upsert_interaction(
            user_query="What happened to the foundation?",
            llm_answer="The foundation failed due to differential settlement.",
            sources=[],
            extracted_facts=foundation_facts
        )
        
        # Store roof facts in matter 2
        await adapter2.upsert_interaction(
            user_query="What happened to the roof?",
            llm_answer="The roof membrane failed and developed tears.",
            sources=[],
            extracted_facts=roof_facts
        )
        
        # Test isolation: recall from matter 1 should not return matter 2 facts
        matter1_recall = await adapter1.recall("roof membrane", top_k=10)
        matter2_recall = await adapter2.recall("foundation failure", top_k=10)
        
        # Matter 1 should not have roof facts
        roof_labels = [item.label for item in matter1_recall]
        assert not any("roof" in label.lower() for label in roof_labels)
        
        # Matter 2 should not have foundation facts
        foundation_labels = [item.label for item in matter2_recall]
        assert not any("foundation" in label.lower() for label in foundation_labels)
        
        # Verify each matter can recall its own facts
        matter1_foundation_recall = await adapter1.recall("foundation", top_k=10)
        matter2_roof_recall = await adapter2.recall("roof", top_k=10)
        
        foundation_found = any("foundation" in item.label.lower() for item in matter1_foundation_recall)
        roof_found = any("roof" in item.label.lower() for item in matter2_roof_recall)
        
        assert foundation_found  # Matter 1 should find foundation facts
        assert roof_found       # Matter 2 should find roof facts
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_rag_engine_letta_integration(self, sample_matters):
        """Test RAG engine integration with Letta memory."""
        matter1, _ = sample_matters
        
        # Create RAG engine which should initialize Letta automatically
        try:
            rag_engine = RAGEngine(matter=matter1)
        except Exception as e:
            pytest.skip(f"Failed to initialize RAG engine: {e}")
        
        if not rag_engine.letta_adapter or not rag_engine.letta_adapter.agent_id:
            pytest.skip("Letta not available in RAG engine")
        
        # First, store some knowledge via manual interaction
        await rag_engine.letta_adapter.upsert_interaction(
            user_query="Tell me about the project timeline",
            llm_answer="The project was originally scheduled for 18 months but experienced delays due to foundation issues.",
            sources=[],
            extracted_facts=[
                KnowledgeItem(
                    type="Event",
                    label="Project schedule delay",
                    date="2023-03-01",
                    actors=["General Contractor", "Owner"],
                    support_snippet="18-month project delayed due to foundation issues"
                )
            ]
        )
        
        # Now test that RAG engine can recall this information
        # Note: This requires vector store to have some content
        # For full integration, we'd need to set up vector store with actual documents
        
        memory_stats = await rag_engine.letta_adapter.get_memory_stats()
        assert memory_stats["status"] == "active"
        assert memory_stats["memory_items"] > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_followup_generation_with_memory(self, sample_matters):
        """Test that follow-up generation uses stored memory context."""
        matter1, _ = sample_matters
        
        adapter = LettaAdapter(
            matter_path=matter1.paths.root,
            matter_name=matter1.name,
            matter_id=matter1.id
        )
        
        if not adapter.client or not adapter.agent_id:
            pytest.skip("Letta not available for integration testing")
        
        # Store some context about foundation issues
        await adapter.upsert_interaction(
            user_query="What are the main foundation problems?",
            llm_answer="The main foundation problems include differential settlement, inadequate soil analysis, and design deficiencies.",
            sources=[],
            extracted_facts=[
                KnowledgeItem(
                    type="Issue",
                    label="Differential settlement",
                    support_snippet="Foundation experienced uneven settling"
                ),
                KnowledgeItem(
                    type="Issue", 
                    label="Inadequate soil analysis",
                    support_snippet="Geotechnical investigation was insufficient"
                )
            ]
        )
        
        # Generate follow-ups - should be contextually aware
        followups = await adapter.suggest_followups(
            user_query="How much will the foundation repairs cost?",
            llm_answer="Foundation repairs are estimated at $250,000 including excavation, soil stabilization, and reconstruction."
        )
        
        # Should return contextual follow-ups
        assert len(followups) > 0
        assert all(isinstance(f, str) for f in followups)
        assert all(len(f) <= 150 for f in followups)  # Reasonable length
        
        # At least some should be questions (end with ?)
        question_count = sum(1 for f in followups if f.endswith('?'))
        assert question_count > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_agent_persistence_across_sessions(self, sample_matters):
        """Test that agent state persists across adapter instances."""
        matter1, _ = sample_matters
        
        # Create first adapter instance
        adapter1 = LettaAdapter(
            matter_path=matter1.paths.root,
            matter_name=matter1.name,
            matter_id=matter1.id
        )
        
        if not adapter1.client or not adapter1.agent_id:
            pytest.skip("Letta not available for integration testing")
        
        original_agent_id = adapter1.agent_id
        
        # Store some knowledge
        await adapter1.upsert_interaction(
            user_query="What is the project status?",
            llm_answer="The project is currently delayed due to foundation issues.",
            sources=[],
            extracted_facts=[
                KnowledgeItem(
                    type="Fact",
                    label="Project currently delayed",
                    support_snippet="Foundation issues causing project delays"
                )
            ]
        )
        
        # Create second adapter instance (simulating app restart)
        adapter2 = LettaAdapter(
            matter_path=matter1.paths.root,
            matter_name=matter1.name,
            matter_id=matter1.id
        )
        
        # Should load the same agent
        assert adapter2.agent_id == original_agent_id
        
        # Should be able to recall previously stored information
        recalled = await adapter2.recall("project delay", top_k=5)
        
        # Should find our previously stored fact
        found_delay_fact = any("delayed" in item.label.lower() for item in recalled)
        assert found_delay_fact
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_large_memory_handling(self, sample_matters):
        """Test handling of large amounts of memory data."""
        matter1, _ = sample_matters
        
        adapter = LettaAdapter(
            matter_path=matter1.paths.root,
            matter_name=matter1.name,
            matter_id=matter1.id
        )
        
        if not adapter.client or not adapter.agent_id:
            pytest.skip("Letta not available for integration testing")
        
        # Store multiple interactions with various knowledge items
        for i in range(5):
            facts = [
                KnowledgeItem(
                    type="Event",
                    label=f"Event {i}: Construction milestone",
                    date=f"2023-0{i+1}-15",
                    actors=["Contractor", "Owner"],
                    support_snippet=f"Milestone {i} completed on schedule"
                ),
                KnowledgeItem(
                    type="Issue", 
                    label=f"Issue {i}: Quality concern",
                    date=f"2023-0{i+1}-20",
                    actors=["QC Inspector"],
                    support_snippet=f"Quality issue {i} identified and resolved"
                )
            ]
            
            await adapter.upsert_interaction(
                user_query=f"What happened in month {i+1}?",
                llm_answer=f"In month {i+1}, we completed milestone {i} but identified quality issue {i}.",
                sources=[],
                extracted_facts=facts
            )
        
        # Test recall with various queries
        event_recall = await adapter.recall("construction milestone", top_k=8)
        issue_recall = await adapter.recall("quality concern", top_k=8)
        
        # Should find multiple relevant items
        assert len(event_recall) > 0
        assert len(issue_recall) > 0
        
        # Verify memory stats reflect the stored data
        stats = await adapter.get_memory_stats()
        assert stats["memory_items"] >= 10  # At least our 10 facts + interaction summaries
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_recovery_and_resilience(self, sample_matters):
        """Test adapter resilience to various error conditions."""
        matter1, _ = sample_matters
        
        adapter = LettaAdapter(
            matter_path=matter1.paths.root,
            matter_name=matter1.name,
            matter_id=matter1.id
        )
        
        # Test operations when Letta is unavailable
        if not adapter.client:
            # Should handle gracefully without errors
            recall_result = await adapter.recall("test query", top_k=5)
            assert recall_result == []
            
            await adapter.upsert_interaction("test", "test", [], [])
            # Should not raise exception
            
            followups = await adapter.suggest_followups("test", "test")
            assert len(followups) == 4  # Fallback suggestions
            
            stats = await adapter.get_memory_stats()
            assert stats["status"] == "unavailable"
        
        # Test with malformed data
        if adapter.client and adapter.agent_id:
            # Should handle malformed knowledge items gracefully
            malformed_facts = [
                KnowledgeItem(
                    type="Event",
                    label="",  # Empty label
                    support_snippet="Test snippet"
                )
            ]
            
            # Should not raise exception
            await adapter.upsert_interaction(
                user_query="test",
                llm_answer="test",
                sources=[],
                extracted_facts=malformed_facts
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])