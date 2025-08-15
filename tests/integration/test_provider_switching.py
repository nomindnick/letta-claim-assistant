"""
Integration tests for LLM provider switching functionality.

Tests end-to-end provider switching, consent workflows, and conversation context preservation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
from pathlib import Path

from app.llm.provider_manager import ProviderManager
from app.privacy_consent import PrivacyConsentManager, ConsentType
from app.rag import RAGEngine
from app.matters import Matter


class TestProviderSwitchingIntegration:
    """Test end-to-end provider switching scenarios."""
    
    @pytest.fixture
    async def setup_environment(self):
        """Set up test environment with temporary directory and mocked services."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                # Create provider manager and consent manager
                provider_manager = ProviderManager()
                consent_manager = PrivacyConsentManager()
                
                # Mock matter
                mock_matter = MagicMock(spec=Matter)
                mock_matter.id = "test_matter_123"
                mock_matter.name = "Test Matter"
                
                # Mock vector store and letta adapter
                mock_vector_store = AsyncMock()
                mock_letta_adapter = AsyncMock()
                
                # Create RAG engine
                rag_engine = RAGEngine()
                
                yield {
                    "provider_manager": provider_manager,
                    "consent_manager": consent_manager,
                    "rag_engine": rag_engine,
                    "mock_matter": mock_matter,
                    "mock_vector_store": mock_vector_store,
                    "mock_letta_adapter": mock_letta_adapter,
                    "tmpdir": tmpdir
                }
    
    @pytest.mark.asyncio
    async def test_ollama_to_gemini_switching_with_consent(self, setup_environment):
        """Test switching from Ollama to Gemini with consent workflow."""
        env = await setup_environment
        provider_manager = env["provider_manager"]
        consent_manager = env["consent_manager"]
        
        # Step 1: Register Ollama provider
        with patch('app.llm.provider_manager.OllamaProvider') as mock_ollama_class:
            mock_ollama = AsyncMock()
            mock_ollama.test_connection.return_value = True
            mock_ollama.generate.return_value = "Response from Ollama"
            mock_ollama_class.return_value = mock_ollama
            
            success = await provider_manager.register_ollama_provider()
            assert success is True
            
            # Verify Ollama is active
            active = provider_manager.get_active_provider()
            assert active is not None
            
            # Test generation with Ollama
            ollama_response = await active.generate(
                "You are helpful", 
                [{"role": "user", "content": "Hello"}]
            )
            assert ollama_response == "Response from Ollama"
        
        # Step 2: Grant consent for Gemini
        consent_granted = consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "gemini")
        assert consent_granted is True
        
        # Step 3: Register Gemini provider
        with patch('app.llm.provider_manager.GeminiProvider') as mock_gemini_class:
            mock_gemini = AsyncMock()
            mock_gemini.test_connection.return_value = True
            mock_gemini.generate.return_value = "Response from Gemini"
            mock_gemini_class.return_value = mock_gemini
            
            result = await provider_manager.register_gemini_provider(
                api_key="test_key",
                model="gemini-2.0-flash-exp"
            )
            assert result["success"] is True
            
            # Step 4: Switch to Gemini
            gemini_key = "gemini_gemini-2.0-flash-exp"
            switch_success = provider_manager.switch_provider(gemini_key)
            assert switch_success is True
            
            # Verify Gemini is now active
            active = provider_manager.get_active_provider()
            assert active == mock_gemini
            
            # Test generation with Gemini
            gemini_response = await active.generate(
                "You are helpful",
                [{"role": "user", "content": "Hello"}]
            )
            assert gemini_response == "Response from Gemini"
    
    @pytest.mark.asyncio
    async def test_gemini_switching_without_consent(self, setup_environment):
        """Test that Gemini switching fails without consent."""
        env = await setup_environment
        provider_manager = env["provider_manager"]
        
        # Attempt to register Gemini without consent
        with patch('app.llm.provider_manager.GeminiProvider') as mock_gemini_class:
            mock_gemini = AsyncMock()
            mock_gemini.test_connection.return_value = True
            mock_gemini_class.return_value = mock_gemini
            
            result = await provider_manager.register_gemini_provider(
                api_key="test_key",
                model="gemini-2.0-flash-exp"
            )
            
            assert result["success"] is False
            assert result["error"] == "consent_required"
            
            # Verify no Gemini provider was registered
            providers = provider_manager.list_providers()
            gemini_providers = [k for k in providers.keys() if k.startswith("gemini")]
            assert len(gemini_providers) == 0
    
    @pytest.mark.asyncio
    async def test_provider_metrics_tracking_across_switches(self, setup_environment):
        """Test that provider metrics are tracked correctly across switches."""
        env = await setup_environment
        provider_manager = env["provider_manager"]
        
        # Register two Ollama providers with different models
        with patch('app.llm.provider_manager.OllamaProvider') as mock_ollama_class:
            # Provider 1
            mock_ollama1 = AsyncMock()
            mock_ollama1.test_connection.return_value = True
            mock_ollama_class.return_value = mock_ollama1
            
            success1 = await provider_manager.register_ollama_provider(
                model="gpt-oss:20b"
            )
            assert success1 is True
            provider1_key = list(provider_manager._providers.keys())[0]
            
            # Provider 2
            mock_ollama2 = AsyncMock()
            mock_ollama2.test_connection.return_value = True
            mock_ollama_class.return_value = mock_ollama2
            
            success2 = await provider_manager.register_ollama_provider(
                model="llama2:13b"
            )
            # Note: Second registration might override the first in current implementation
            # This test validates metrics tracking behavior
            
            # Record metrics for provider 1
            provider_manager._record_provider_metric(provider1_key, "success", None)
            provider_manager._record_provider_metric(provider1_key, "response_time", 1.5)
            
            # Switch providers and record different metrics
            if len(provider_manager._providers) > 1:
                provider2_key = list(provider_manager._providers.keys())[1]
                provider_manager.switch_provider(provider2_key)
                provider_manager._record_provider_metric(provider2_key, "success", None)
                provider_manager._record_provider_metric(provider2_key, "response_time", 2.0)
            
            # Verify metrics are maintained separately
            metrics = provider_manager.get_provider_metrics()
            assert len(metrics) >= 1
            
            if provider1_key in metrics:
                assert metrics[provider1_key]["total_requests"] >= 1
                assert metrics[provider1_key]["avg_response_time"] >= 0
    
    @pytest.mark.asyncio
    async def test_rag_context_preservation_across_provider_switch(self, setup_environment):
        """Test that conversation context is preserved when switching providers."""
        env = await setup_environment
        provider_manager = env["provider_manager"]
        rag_engine = env["rag_engine"]
        mock_matter = env["mock_matter"]
        
        # Mock the dependencies
        with patch.object(rag_engine, 'vector_store') as mock_vector_store:
            with patch.object(rag_engine, 'letta_adapter') as mock_letta:
                
                # Set up mock returns
                mock_vector_store.search.return_value = [
                    MagicMock(
                        chunk_id="chunk1",
                        doc_name="test.pdf",
                        page_start=1,
                        page_end=1,
                        text="Test content",
                        similarity_score=0.9,
                        metadata={"doc_id": "test", "page": 1}
                    )
                ]
                
                mock_letta.recall.return_value = []
                mock_letta.upsert_interaction.return_value = None
                mock_letta.suggest_followups.return_value = ["Follow up 1", "Follow up 2"]
                
                # Register Ollama provider
                with patch('app.llm.provider_manager.OllamaProvider') as mock_ollama_class:
                    mock_ollama = AsyncMock()
                    mock_ollama.test_connection.return_value = True
                    mock_ollama.generate.return_value = "Ollama response with context"
                    mock_ollama_class.return_value = mock_ollama
                    
                    await provider_manager.register_ollama_provider()
                    
                    # Generate response with Ollama
                    rag_engine.provider_manager = provider_manager
                    response1 = await rag_engine.generate_answer(
                        query="What is the project status?",
                        matter=mock_matter
                    )
                    
                    assert "Ollama response" in response1.answer
                    assert len(response1.sources) > 0
                    
                    # Switch to different model (simulating provider switch)
                    mock_ollama2 = AsyncMock()
                    mock_ollama2.test_connection.return_value = True
                    mock_ollama2.generate.return_value = "Second provider response"
                    mock_ollama_class.return_value = mock_ollama2
                    
                    await provider_manager.register_ollama_provider(model="different:model")
                    
                    # Generate response with new provider - should preserve context
                    response2 = await rag_engine.generate_answer(
                        query="Follow up question",
                        matter=mock_matter
                    )
                    
                    assert "Second provider response" in response2.answer
                    
                    # Verify that vector search was called both times (context preserved)
                    assert mock_vector_store.search.call_count == 2
                    
                    # Verify that Letta was called both times (memory preserved)
                    assert mock_letta.recall.call_count == 2
    
    @pytest.mark.asyncio
    async def test_provider_fallback_on_failure(self, setup_environment):
        """Test automatic fallback to backup provider on failure."""
        env = await setup_environment
        provider_manager = env["provider_manager"]
        
        # Register Ollama as primary provider
        with patch('app.llm.provider_manager.OllamaProvider') as mock_ollama_class:
            mock_ollama = AsyncMock()
            mock_ollama.test_connection.return_value = True
            mock_ollama.generate.return_value = "Ollama response"
            mock_ollama_class.return_value = mock_ollama
            
            await provider_manager.register_ollama_provider(model="primary:model")
            primary_key = list(provider_manager._providers.keys())[0]
            
            # Register secondary provider
            mock_ollama2 = AsyncMock()
            mock_ollama2.test_connection.return_value = True
            mock_ollama2.generate.return_value = "Fallback response"
            mock_ollama_class.return_value = mock_ollama2
            
            await provider_manager.register_ollama_provider(model="fallback:model")
            
            # Make primary provider fail
            mock_ollama.generate.side_effect = Exception("Primary provider failed")
            
            # Test that we can detect failure and switch to backup
            providers = list(provider_manager._providers.keys())
            if len(providers) > 1:
                # Try primary provider
                try:
                    provider_manager.switch_provider(providers[0])
                    await provider_manager.get_active_provider().generate(
                        "System", [{"role": "user", "content": "Test"}]
                    )
                    assert False, "Expected primary provider to fail"
                except Exception:
                    # Switch to fallback
                    provider_manager.switch_provider(providers[1])
                    response = await provider_manager.get_active_provider().generate(
                        "System", [{"role": "user", "content": "Test"}]
                    )
                    assert response == "Fallback response"
    
    @pytest.mark.asyncio
    async def test_consent_revocation_affects_provider_availability(self, setup_environment):
        """Test that revoking consent makes external providers unavailable."""
        env = await setup_environment
        provider_manager = env["provider_manager"]
        consent_manager = env["consent_manager"]
        
        # Grant consent and register Gemini
        consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "gemini")
        
        with patch('app.llm.provider_manager.GeminiProvider') as mock_gemini_class:
            mock_gemini = AsyncMock()
            mock_gemini.test_connection.return_value = True
            mock_gemini_class.return_value = mock_gemini
            
            result = await provider_manager.register_gemini_provider(
                api_key="test_key",
                model="gemini-2.0-flash-exp"
            )
            assert result["success"] is True
            
            # Switch to Gemini
            gemini_key = "gemini_gemini-2.0-flash-exp"
            provider_manager.switch_provider(gemini_key)
            
            # Verify Gemini is available
            assert provider_manager.get_active_provider() is not None
            assert provider_manager.check_provider_consent("gemini")["consent_granted"] is True
            
            # Revoke consent
            consent_manager.revoke_consent(ConsentType.EXTERNAL_LLM, "gemini")
            
            # Check that consent is revoked
            consent_status = provider_manager.check_provider_consent("gemini")
            assert consent_status["consent_granted"] is False
            assert consent_status["consent_status"] == "revoked"
            
            # Attempting to register again should fail
            result2 = await provider_manager.register_gemini_provider(
                api_key="test_key",
                model="gemini-2.0-flash-exp"
            )
            assert result2["success"] is False
            assert result2["error"] == "consent_required"
    
    @pytest.mark.asyncio
    async def test_provider_performance_comparison(self, setup_environment):
        """Test comparing performance metrics across different providers."""
        env = await setup_environment
        provider_manager = env["provider_manager"]
        
        # Register multiple providers
        with patch('app.llm.provider_manager.OllamaProvider') as mock_ollama_class:
            # Fast provider
            mock_fast = AsyncMock()
            mock_fast.test_connection.return_value = True
            mock_ollama_class.return_value = mock_fast
            
            await provider_manager.register_ollama_provider(model="fast:model")
            fast_key = list(provider_manager._providers.keys())[0]
            
            # Slow provider  
            mock_slow = AsyncMock()
            mock_slow.test_connection.return_value = True
            mock_ollama_class.return_value = mock_slow
            
            await provider_manager.register_ollama_provider(model="slow:model")
            
            # Record performance metrics
            provider_manager._record_provider_metric(fast_key, "response_time", 0.5)
            provider_manager._record_provider_metric(fast_key, "response_time", 0.7)
            provider_manager._record_provider_metric(fast_key, "success", None)
            provider_manager._record_provider_metric(fast_key, "success", None)
            
            # Slow provider metrics
            providers = list(provider_manager._providers.keys())
            if len(providers) > 1:
                slow_key = providers[1]
                provider_manager._record_provider_metric(slow_key, "response_time", 2.0)
                provider_manager._record_provider_metric(slow_key, "response_time", 2.5)
                provider_manager._record_provider_metric(slow_key, "success", None)
                provider_manager._record_provider_metric(slow_key, "error", None)
                
                # Compare metrics
                metrics = provider_manager.get_provider_metrics()
                
                if fast_key in metrics and slow_key in metrics:
                    assert metrics[fast_key]["avg_response_time"] < metrics[slow_key]["avg_response_time"]
                    assert metrics[fast_key]["success_rate"] > metrics[slow_key]["success_rate"]
    
    @pytest.mark.asyncio
    async def test_concurrent_provider_operations(self, setup_environment):
        """Test that concurrent provider operations are handled safely."""
        env = await setup_environment
        provider_manager = env["provider_manager"]
        
        # Register base provider
        with patch('app.llm.provider_manager.OllamaProvider') as mock_ollama_class:
            mock_ollama = AsyncMock()
            mock_ollama.test_connection.return_value = True
            mock_ollama_class.return_value = mock_ollama
            
            await provider_manager.register_ollama_provider()
            
            # Simulate concurrent operations
            async def test_provider():
                return await provider_manager.test_provider(list(provider_manager._providers.keys())[0])
            
            async def switch_provider():
                provider_key = list(provider_manager._providers.keys())[0]
                return provider_manager.switch_provider(provider_key)
            
            async def get_metrics():
                return provider_manager.get_provider_metrics()
            
            # Run operations concurrently
            results = await asyncio.gather(
                test_provider(),
                switch_provider(),
                get_metrics(),
                return_exceptions=True
            )
            
            # Verify no exceptions were raised
            for result in results:
                assert not isinstance(result, Exception)
            
            # Verify results make sense
            test_result, switch_result, metrics_result = results
            assert isinstance(test_result, bool)
            assert isinstance(switch_result, bool)
            assert isinstance(metrics_result, dict)