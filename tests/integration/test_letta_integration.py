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
        ]\n        \n        extracted_facts = [\n            KnowledgeItem(\n                type="Issue",\n                label="Inadequate soil analysis",\n                date="2022-03-15",\n                actors=["Geotechnical Engineer", "Design Team"],\n                doc_refs=[{"doc": "Geotechnical_Report_2022.pdf", "page": 5}],\n                support_snippet="Soil samples taken at insufficient depth"\n            ),\n            KnowledgeItem(\n                type="Event",\n                label="Foundation failure Building A",\n                date="2023-02-14",\n                actors=["ABC Construction", "Owner"],\n                doc_refs=[{"doc": "Foundation_Inspection_2023.pdf", "page": 12}],\n                support_snippet="Differential settlement of 2.5 inches observed"\n            )\n        ]\n        \n        # Step 1: Store interaction\n        await adapter.upsert_interaction(\n            user_query=user_query,\n            llm_answer=llm_answer,\n            sources=sources,\n            extracted_facts=extracted_facts\n        )\n        \n        # Step 2: Test immediate recall\n        recalled_items = await adapter.recall("foundation failure", top_k=5)\n        \n        # Should find stored knowledge items\n        assert len(recalled_items) > 0\n        \n        # Look for our stored facts\n        found_issue = any(item.label == "Inadequate soil analysis" for item in recalled_items)\n        found_event = any(item.label == "Foundation failure Building A" for item in recalled_items)\n        \n        # At least one of our facts should be recalled\n        assert found_issue or found_event\n        \n        # Step 3: Test memory stats\n        stats = await adapter.get_memory_stats()\n        assert stats["status"] == "active"\n        assert stats["memory_items"] > 0\n        assert stats["agent_id"] == adapter.agent_id\n    \n    @pytest.mark.asyncio\n    @pytest.mark.integration\n    async def test_matter_isolation_verification(self, sample_matters):\n        """Test that memory is completely isolated between matters."""\n        matter1, matter2 = sample_matters\n        \n        adapter1 = LettaAdapter(\n            matter_path=matter1.paths.root,\n            matter_name=matter1.name,\n            matter_id=matter1.id\n        )\n        \n        adapter2 = LettaAdapter(\n            matter_path=matter2.paths.root,\n            matter_name=matter2.name,\n            matter_id=matter2.id\n        )\n        \n        if not (adapter1.client and adapter1.agent_id and adapter2.client and adapter2.agent_id):\n            pytest.skip("Letta not available for integration testing")\n        \n        # Store different facts in each matter\n        foundation_facts = [\n            KnowledgeItem(\n                type="Event",\n                label="Foundation failure",\n                date="2023-02-14",\n                actors=["ABC Construction"],\n                support_snippet="Foundation showed differential settlement"\n            )\n        ]\n        \n        roof_facts = [\n            KnowledgeItem(\n                type="Event",\n                label="Roof membrane failure",\n                date="2023-03-10",\n                actors=["XYZ Roofing"],\n                support_snippet="EPDM membrane developed tears"\n            )\n        ]\n        \n        # Store foundation facts in matter 1\n        await adapter1.upsert_interaction(\n            user_query="What happened to the foundation?",\n            llm_answer="The foundation failed due to differential settlement.",\n            sources=[],\n            extracted_facts=foundation_facts\n        )\n        \n        # Store roof facts in matter 2\n        await adapter2.upsert_interaction(\n            user_query="What happened to the roof?",\n            llm_answer="The roof membrane failed and developed tears.",\n            sources=[],\n            extracted_facts=roof_facts\n        )\n        \n        # Test isolation: recall from matter 1 should not return matter 2 facts\n        matter1_recall = await adapter1.recall("roof membrane", top_k=10)\n        matter2_recall = await adapter2.recall("foundation failure", top_k=10)\n        \n        # Matter 1 should not have roof facts\n        roof_labels = [item.label for item in matter1_recall]\n        assert not any("roof" in label.lower() for label in roof_labels)\n        \n        # Matter 2 should not have foundation facts\n        foundation_labels = [item.label for item in matter2_recall]\n        assert not any("foundation" in label.lower() for label in foundation_labels)\n        \n        # Verify each matter can recall its own facts\n        matter1_foundation_recall = await adapter1.recall("foundation", top_k=10)\n        matter2_roof_recall = await adapter2.recall("roof", top_k=10)\n        \n        foundation_found = any("foundation" in item.label.lower() for item in matter1_foundation_recall)\n        roof_found = any("roof" in item.label.lower() for item in matter2_roof_recall)\n        \n        assert foundation_found  # Matter 1 should find foundation facts\n        assert roof_found       # Matter 2 should find roof facts\n    \n    @pytest.mark.asyncio\n    @pytest.mark.integration \n    async def test_rag_engine_letta_integration(self, sample_matters):\n        """Test RAG engine integration with Letta memory."""\n        matter1, _ = sample_matters\n        \n        # Create RAG engine which should initialize Letta automatically\n        try:\n            rag_engine = RAGEngine(matter=matter1)\n        except Exception as e:\n            pytest.skip(f"Failed to initialize RAG engine: {e}")\n        \n        if not rag_engine.letta_adapter or not rag_engine.letta_adapter.agent_id:\n            pytest.skip("Letta not available in RAG engine")\n        \n        # First, store some knowledge via manual interaction\n        await rag_engine.letta_adapter.upsert_interaction(\n            user_query="Tell me about the project timeline",\n            llm_answer="The project was originally scheduled for 18 months but experienced delays due to foundation issues.",\n            sources=[],\n            extracted_facts=[\n                KnowledgeItem(\n                    type="Event",\n                    label="Project schedule delay",\n                    date="2023-03-01",\n                    actors=["General Contractor", "Owner"],\n                    support_snippet="18-month project delayed due to foundation issues"\n                )\n            ]\n        )\n        \n        # Now test that RAG engine can recall this information\n        # Note: This requires vector store to have some content\n        # For full integration, we'd need to set up vector store with actual documents\n        \n        memory_stats = await rag_engine.letta_adapter.get_memory_stats()\n        assert memory_stats[\"status\"] == \"active\"\n        assert memory_stats[\"memory_items\"] > 0\n    \n    @pytest.mark.asyncio\n    @pytest.mark.integration\n    async def test_followup_generation_with_memory(self, sample_matters):\n        """Test that follow-up generation uses stored memory context."""\n        matter1, _ = sample_matters\n        \n        adapter = LettaAdapter(\n            matter_path=matter1.paths.root,\n            matter_name=matter1.name,\n            matter_id=matter1.id\n        )\n        \n        if not adapter.client or not adapter.agent_id:\n            pytest.skip("Letta not available for integration testing")\n        \n        # Store some context about foundation issues\n        await adapter.upsert_interaction(\n            user_query="What are the main foundation problems?",\n            llm_answer="The main foundation problems include differential settlement, inadequate soil analysis, and design deficiencies.",\n            sources=[],\n            extracted_facts=[\n                KnowledgeItem(\n                    type="Issue",\n                    label="Differential settlement",\n                    support_snippet="Foundation experienced uneven settling"\n                ),\n                KnowledgeItem(\n                    type="Issue", \n                    label="Inadequate soil analysis",\n                    support_snippet="Geotechnical investigation was insufficient"\n                )\n            ]\n        )\n        \n        # Generate follow-ups - should be contextually aware\n        followups = await adapter.suggest_followups(\n            user_query="How much will the foundation repairs cost?",\n            llm_answer="Foundation repairs are estimated at $250,000 including excavation, soil stabilization, and reconstruction."\n        )\n        \n        # Should return contextual follow-ups\n        assert len(followups) > 0\n        assert all(isinstance(f, str) for f in followups)\n        assert all(len(f) <= 150 for f in followups)  # Reasonable length\n        \n        # At least some should be questions (end with ?)\n        question_count = sum(1 for f in followups if f.endswith('?'))\n        assert question_count > 0\n    \n    @pytest.mark.asyncio\n    @pytest.mark.integration\n    async def test_agent_persistence_across_sessions(self, sample_matters):\n        """Test that agent state persists across adapter instances."""\n        matter1, _ = sample_matters\n        \n        # Create first adapter instance\n        adapter1 = LettaAdapter(\n            matter_path=matter1.paths.root,\n            matter_name=matter1.name,\n            matter_id=matter1.id\n        )\n        \n        if not adapter1.client or not adapter1.agent_id:\n            pytest.skip("Letta not available for integration testing")\n        \n        original_agent_id = adapter1.agent_id\n        \n        # Store some knowledge\n        await adapter1.upsert_interaction(\n            user_query="What is the project status?",\n            llm_answer="The project is currently delayed due to foundation issues.",\n            sources=[],\n            extracted_facts=[\n                KnowledgeItem(\n                    type="Fact",\n                    label="Project currently delayed",\n                    support_snippet="Foundation issues causing project delays"\n                )\n            ]\n        )\n        \n        # Create second adapter instance (simulating app restart)\n        adapter2 = LettaAdapter(\n            matter_path=matter1.paths.root,\n            matter_name=matter1.name,\n            matter_id=matter1.id\n        )\n        \n        # Should load the same agent\n        assert adapter2.agent_id == original_agent_id\n        \n        # Should be able to recall previously stored information\n        recalled = await adapter2.recall("project delay", top_k=5)\n        \n        # Should find our previously stored fact\n        found_delay_fact = any("delayed" in item.label.lower() for item in recalled)\n        assert found_delay_fact\n    \n    @pytest.mark.asyncio\n    @pytest.mark.integration\n    async def test_large_memory_handling(self, sample_matters):\n        """Test handling of large amounts of memory data."""\n        matter1, _ = sample_matters\n        \n        adapter = LettaAdapter(\n            matter_path=matter1.paths.root,\n            matter_name=matter1.name,\n            matter_id=matter1.id\n        )\n        \n        if not adapter.client or not adapter.agent_id:\n            pytest.skip("Letta not available for integration testing")\n        \n        # Store multiple interactions with various knowledge items\n        for i in range(5):\n            facts = [\n                KnowledgeItem(\n                    type="Event",\n                    label=f"Event {i}: Construction milestone",\n                    date=f"2023-0{i+1}-15",\n                    actors=["Contractor", "Owner"],\n                    support_snippet=f"Milestone {i} completed on schedule"\n                ),\n                KnowledgeItem(\n                    type="Issue", \n                    label=f"Issue {i}: Quality concern",\n                    date=f"2023-0{i+1}-20",\n                    actors=["QC Inspector"],\n                    support_snippet=f"Quality issue {i} identified and resolved"\n                )\n            ]\n            \n            await adapter.upsert_interaction(\n                user_query=f"What happened in month {i+1}?",\n                llm_answer=f"In month {i+1}, we completed milestone {i} but identified quality issue {i}.",\n                sources=[],\n                extracted_facts=facts\n            )\n        \n        # Test recall with various queries\n        event_recall = await adapter.recall("construction milestone", top_k=8)\n        issue_recall = await adapter.recall("quality concern", top_k=8)\n        \n        # Should find multiple relevant items\n        assert len(event_recall) > 0\n        assert len(issue_recall) > 0\n        \n        # Verify memory stats reflect the stored data\n        stats = await adapter.get_memory_stats()\n        assert stats["memory_items"] >= 10  # At least our 10 facts + interaction summaries\n    \n    @pytest.mark.asyncio\n    @pytest.mark.integration\n    async def test_error_recovery_and_resilience(self, sample_matters):\n        """Test adapter resilience to various error conditions."""\n        matter1, _ = sample_matters\n        \n        adapter = LettaAdapter(\n            matter_path=matter1.paths.root,\n            matter_name=matter1.name,\n            matter_id=matter1.id\n        )\n        \n        # Test operations when Letta is unavailable\n        if not adapter.client:\n            # Should handle gracefully without errors\n            recall_result = await adapter.recall("test query", top_k=5)\n            assert recall_result == []\n            \n            await adapter.upsert_interaction("test", "test", [], [])\n            # Should not raise exception\n            \n            followups = await adapter.suggest_followups("test", "test")\n            assert len(followups) == 4  # Fallback suggestions\n            \n            stats = await adapter.get_memory_stats()\n            assert stats["status"] == "unavailable"\n        \n        # Test with malformed data\n        if adapter.client and adapter.agent_id:\n            # Should handle malformed knowledge items gracefully\n            malformed_facts = [\n                KnowledgeItem(\n                    type="Event",\n                    label="",  # Empty label\n                    support_snippet="Test snippet"\n                )\n            ]\n            \n            # Should not raise exception\n            await adapter.upsert_interaction(\n                user_query="test",\n                llm_answer="test",\n                sources=[],\n                extracted_facts=malformed_facts\n            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])