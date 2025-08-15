"""
Unit tests for LLM provider management system.

Tests provider registration, switching, consent management, and secure credential storage.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
import tempfile
from pathlib import Path

# Import modules under test
from app.llm.provider_manager import ProviderManager, ProviderType
from app.llm.gemini_provider import GeminiProvider
from app.privacy_consent import PrivacyConsentManager, ConsentType, ConsentStatus
from app.settings import Settings


class TestProviderManager:
    """Test the LLM provider manager."""
    
    @pytest.fixture
    def provider_manager(self):
        """Create a fresh provider manager for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the state file location
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                manager = ProviderManager()
                return manager
    
    @pytest.mark.asyncio
    async def test_register_ollama_provider_success(self, provider_manager):
        """Test successful Ollama provider registration."""
        with patch('app.llm.provider_manager.OllamaProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.test_connection.return_value = True
            mock_provider_class.return_value = mock_provider
            
            success = await provider_manager.register_ollama_provider()
            
            assert success is True
            assert len(provider_manager._providers) == 1
            mock_provider.test_connection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_ollama_provider_connection_failure(self, provider_manager):
        """Test Ollama provider registration with connection failure."""
        with patch('app.llm.provider_manager.OllamaProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.test_connection.return_value = False
            mock_provider_class.return_value = mock_provider
            
            success = await provider_manager.register_ollama_provider()
            
            assert success is False
            assert len(provider_manager._providers) == 0
    
    @pytest.mark.asyncio
    async def test_register_gemini_provider_success(self, provider_manager):
        """Test successful Gemini provider registration."""
        with patch('app.llm.provider_manager.consent_manager') as mock_consent:
            mock_consent.get_consent_requirements.return_value = {
                "requires_consent": True,
                "consent_granted": True
            }
            
            with patch('app.llm.provider_manager.GeminiProvider') as mock_provider_class:
                mock_provider = AsyncMock()
                mock_provider.test_connection.return_value = True
                mock_provider_class.return_value = mock_provider
                
                result = await provider_manager.register_gemini_provider(
                    api_key="test_key",
                    model="gemini-2.0-flash-exp"
                )
                
                assert result["success"] is True
                assert "gemini_gemini-2.0-flash-exp" in provider_manager._providers
    
    @pytest.mark.asyncio
    async def test_register_gemini_provider_consent_required(self, provider_manager):
        """Test Gemini provider registration when consent is required but not granted."""
        with patch('app.llm.provider_manager.consent_manager') as mock_consent:
            mock_consent.get_consent_requirements.return_value = {
                "requires_consent": True,
                "consent_granted": False
            }
            
            result = await provider_manager.register_gemini_provider(
                api_key="test_key",
                model="gemini-2.0-flash-exp"
            )
            
            assert result["success"] is False
            assert result["error"] == "consent_required"
            assert len(provider_manager._providers) == 0
    
    def test_switch_provider_success(self, provider_manager):
        """Test successful provider switching."""
        # Add a mock provider
        mock_provider = MagicMock()
        provider_key = "test_provider"
        provider_manager._providers[provider_key] = mock_provider
        
        success = provider_manager.switch_provider(provider_key)
        
        assert success is True
        assert provider_manager._active_provider == provider_key
    
    def test_switch_provider_not_found(self, provider_manager):
        """Test provider switching with non-existent provider."""
        success = provider_manager.switch_provider("non_existent_provider")
        
        assert success is False
        assert provider_manager._active_provider is None
    
    def test_get_active_provider(self, provider_manager):
        """Test getting the active provider."""
        # No active provider initially
        assert provider_manager.get_active_provider() is None
        
        # Add and set active provider
        mock_provider = MagicMock()
        provider_key = "test_provider"
        provider_manager._providers[provider_key] = mock_provider
        provider_manager._active_provider = provider_key
        
        active = provider_manager.get_active_provider()
        assert active == mock_provider
    
    def test_list_providers(self, provider_manager):
        """Test listing all providers."""
        # Add mock providers
        gen_provider = MagicMock()
        gen_provider.model_name = "test-gen-model"
        embed_provider = MagicMock()
        embed_provider.model_name = "test-embed-model"
        
        provider_manager._providers["gen_provider"] = gen_provider
        provider_manager._embedding_providers["embed_provider"] = embed_provider
        provider_manager._active_provider = "gen_provider"
        
        providers = provider_manager.list_providers()
        
        assert "gen_provider" in providers
        assert "embed_provider" in providers
        assert providers["gen_provider"]["active"] is True
        assert providers["embed_provider"]["active"] is False
    
    @pytest.mark.asyncio
    async def test_test_all_providers(self, provider_manager):
        """Test testing all registered providers."""
        # Add mock providers
        provider1 = AsyncMock()
        provider1.test_connection.return_value = True
        provider2 = AsyncMock()
        provider2.test_connection.return_value = False
        
        provider_manager._providers["provider1"] = provider1
        provider_manager._providers["provider2"] = provider2
        
        results = await provider_manager.test_all_providers()
        
        assert results["provider1"] is True
        assert results["provider2"] is False
    
    def test_get_provider_metrics(self, provider_manager):
        """Test getting provider performance metrics."""
        # Set up mock metrics
        provider_manager._provider_metrics = {
            "test_provider": {
                "response_times": [{"time": 1.0}, {"time": 2.0}],
                "success_count": 8,
                "error_count": 2,
                "total_requests": 10,
                "last_used": "2025-01-21T10:00:00"
            }
        }
        
        # Test specific provider metrics
        metrics = provider_manager.get_provider_metrics("test_provider")
        assert metrics["success_count"] == 8
        assert metrics["error_count"] == 2
        
        # Test summary metrics
        summary = provider_manager.get_provider_metrics()
        assert "test_provider" in summary
        assert summary["test_provider"]["avg_response_time"] == 1.5
        assert summary["test_provider"]["success_rate"] == 0.8
    
    def test_record_provider_metric(self, provider_manager):
        """Test recording provider metrics."""
        provider_key = "test_provider"
        
        # Test recording response time
        provider_manager._record_provider_metric(provider_key, "response_time", 1.5)
        
        assert provider_key in provider_manager._provider_metrics
        metrics = provider_manager._provider_metrics[provider_key]
        assert len(metrics["response_times"]) == 1
        assert metrics["response_times"][0]["time"] == 1.5
        
        # Test recording success
        provider_manager._record_provider_metric(provider_key, "success", None)
        assert metrics["success_count"] == 1
        assert metrics["total_requests"] == 1
        
        # Test recording error
        provider_manager._record_provider_metric(provider_key, "error", None)
        assert metrics["error_count"] == 1
        assert metrics["total_requests"] == 2


class TestGeminiProvider:
    """Test the Gemini provider implementation."""
    
    @pytest.fixture
    def gemini_provider(self):
        """Create a Gemini provider for testing."""
        return GeminiProvider(api_key="test_key", model="gemini-2.0-flash-exp")
    
    def test_format_messages_for_gemini(self, gemini_provider):
        """Test message formatting for Gemini API."""
        system_prompt = "You are a helpful assistant."
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        formatted = gemini_provider._format_messages_for_gemini(system_prompt, messages)
        
        assert "System: You are a helpful assistant." in formatted
        assert "Human: Hello" in formatted
        assert "Assistant: Hi there!" in formatted
        assert "Human: How are you?" in formatted
        assert formatted.endswith("Assistant:")
    
    def test_format_messages_empty_system(self, gemini_provider):
        """Test message formatting with empty system prompt."""
        messages = [{"role": "user", "content": "Hello"}]
        
        formatted = gemini_provider._format_messages_for_gemini("", messages)
        
        assert "Human: Hello" in formatted
        assert "System:" not in formatted
    
    @pytest.mark.asyncio
    async def test_generate_no_model(self):
        """Test generation with no model initialized."""
        provider = GeminiProvider(api_key="", model="test")
        
        with pytest.raises(RuntimeError, match="Gemini model not initialized"):
            await provider.generate("System prompt", [{"role": "user", "content": "Hello"}])
    
    @pytest.mark.asyncio
    async def test_generate_no_messages(self, gemini_provider):
        """Test generation with no messages."""
        with pytest.raises(ValueError, match="No messages provided"):
            await gemini_provider.generate("System prompt", [])
    
    def test_default_safety_settings(self, gemini_provider):
        """Test default safety settings."""
        settings = gemini_provider._default_safety_settings()
        
        assert len(settings) == 4
        categories = [s["category"] for s in settings]
        assert "HARM_CATEGORY_HARASSMENT" in categories
        assert "HARM_CATEGORY_HATE_SPEECH" in categories
        assert "HARM_CATEGORY_SEXUALLY_EXPLICIT" in categories
        assert "HARM_CATEGORY_DANGEROUS_CONTENT" in categories


class TestPrivacyConsentManager:
    """Test the privacy consent management system."""
    
    @pytest.fixture
    def consent_manager(self):
        """Create a fresh consent manager for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                return PrivacyConsentManager()
    
    def test_consent_key_generation(self, consent_manager):
        """Test consent key generation."""
        key = consent_manager._consent_key(ConsentType.EXTERNAL_LLM, "gemini")
        assert key == "external_llm:gemini"
    
    def test_grant_consent_success(self, consent_manager):
        """Test successful consent granting."""
        success = consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "gemini")
        
        assert success is True
        assert consent_manager.is_consent_granted(ConsentType.EXTERNAL_LLM, "gemini") is True
        assert consent_manager.get_consent_status(ConsentType.EXTERNAL_LLM, "gemini") == ConsentStatus.GRANTED
    
    def test_deny_consent_success(self, consent_manager):
        """Test successful consent denial."""
        success = consent_manager.deny_consent(ConsentType.EXTERNAL_LLM, "gemini")
        
        assert success is True
        assert consent_manager.is_consent_granted(ConsentType.EXTERNAL_LLM, "gemini") is False
        assert consent_manager.get_consent_status(ConsentType.EXTERNAL_LLM, "gemini") == ConsentStatus.DENIED
    
    def test_revoke_consent_success(self, consent_manager):
        """Test successful consent revocation."""
        # Grant consent first
        consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "gemini")
        assert consent_manager.is_consent_granted(ConsentType.EXTERNAL_LLM, "gemini") is True
        
        # Revoke consent
        success = consent_manager.revoke_consent(ConsentType.EXTERNAL_LLM, "gemini")
        
        assert success is True
        assert consent_manager.is_consent_granted(ConsentType.EXTERNAL_LLM, "gemini") is False
        assert consent_manager.get_consent_status(ConsentType.EXTERNAL_LLM, "gemini") == ConsentStatus.REVOKED
    
    def test_get_consent_status_not_requested(self, consent_manager):
        """Test getting consent status when not yet requested."""
        status = consent_manager.get_consent_status(ConsentType.EXTERNAL_LLM, "gemini")
        assert status == ConsentStatus.NOT_REQUESTED
    
    def test_ensure_external_llm_consent_local(self, consent_manager):
        """Test consent checking for local providers."""
        # Local providers don't require consent
        assert consent_manager.ensure_external_llm_consent("ollama") is True
        assert consent_manager.ensure_external_llm_consent("local") is True
    
    def test_ensure_external_llm_consent_external_granted(self, consent_manager):
        """Test consent checking for external providers with granted consent."""
        consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "gemini")
        assert consent_manager.ensure_external_llm_consent("gemini") is True
    
    def test_ensure_external_llm_consent_external_not_granted(self, consent_manager):
        """Test consent checking for external providers without consent."""
        assert consent_manager.ensure_external_llm_consent("gemini") is False
    
    def test_get_consent_requirements_local(self, consent_manager):
        """Test getting consent requirements for local provider."""
        reqs = consent_manager.get_consent_requirements("ollama")
        
        assert reqs["requires_consent"] is False
        assert reqs["consent_granted"] is True
        assert reqs["consent_status"] == "not_required"
    
    def test_get_consent_requirements_external_not_granted(self, consent_manager):
        """Test getting consent requirements for external provider without consent."""
        reqs = consent_manager.get_consent_requirements("gemini")
        
        assert reqs["requires_consent"] is True
        assert reqs["consent_granted"] is False
        assert reqs["consent_status"] == ConsentStatus.NOT_REQUESTED.value
        assert "data_usage_notice" in reqs
    
    def test_get_consent_requirements_external_granted(self, consent_manager):
        """Test getting consent requirements for external provider with consent."""
        consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "gemini")
        reqs = consent_manager.get_consent_requirements("gemini")
        
        assert reqs["requires_consent"] is True
        assert reqs["consent_granted"] is True
        assert reqs["consent_status"] == ConsentStatus.GRANTED.value
    
    def test_get_all_consents(self, consent_manager):
        """Test getting all consent records."""
        # Grant consent for multiple providers
        consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "gemini")
        consent_manager.deny_consent(ConsentType.EXTERNAL_LLM, "openai")
        
        consents = consent_manager.get_all_consents()
        
        assert len(consents) == 2
        providers = [c["provider"] for c in consents]
        assert "gemini" in providers
        assert "openai" in providers
    
    def test_get_provider_consents(self, consent_manager):
        """Test getting consents for specific provider."""
        consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "gemini")
        consent_manager.grant_consent(ConsentType.DATA_SHARING, "gemini")
        
        consents = consent_manager.get_provider_consents("gemini")
        
        assert len(consents) == 2
        consent_types = [c["consent_type"] for c in consents]
        assert ConsentType.EXTERNAL_LLM.value in consent_types
        assert ConsentType.DATA_SHARING.value in consent_types
    
    def test_clear_provider_consents(self, consent_manager):
        """Test clearing all consents for a provider."""
        # Set up multiple consents for the provider
        consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "gemini")
        consent_manager.grant_consent(ConsentType.DATA_SHARING, "gemini")
        consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, "openai")
        
        # Clear Gemini consents
        success = consent_manager.clear_provider_consents("gemini")
        
        assert success is True
        
        # Check Gemini consents are gone
        gemini_consents = consent_manager.get_provider_consents("gemini")
        assert len(gemini_consents) == 0
        
        # Check other provider consents remain
        openai_consents = consent_manager.get_provider_consents("openai")
        assert len(openai_consents) == 1
    
    def test_data_usage_notice_gemini(self, consent_manager):
        """Test data usage notice for Gemini."""
        notice = consent_manager._get_data_usage_notice("gemini")
        assert "Google Gemini" in notice
        assert "Google's servers" in notice
        assert "privacy policy" in notice
    
    def test_data_usage_notice_generic(self, consent_manager):
        """Test data usage notice for generic provider."""
        notice = consent_manager._get_data_usage_notice("custom_provider")
        assert "custom_provider" in notice
        assert "external servers" in notice


class TestSecureCredentialStorage:
    """Test secure credential storage in settings."""
    
    @pytest.fixture
    def settings_instance(self):
        """Create a fresh settings instance for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                return Settings()
    
    def test_store_and_retrieve_credential(self, settings_instance):
        """Test storing and retrieving a credential."""
        provider = "gemini"
        credential_type = "api_key" 
        credential_value = "test_api_key_12345"
        
        # Store credential
        success = settings_instance.store_credential(provider, credential_type, credential_value)
        assert success is True
        
        # Retrieve credential
        retrieved_value = settings_instance.get_credential(provider, credential_type)
        assert retrieved_value == credential_value
    
    def test_get_nonexistent_credential(self, settings_instance):
        """Test retrieving a credential that doesn't exist."""
        value = settings_instance.get_credential("nonexistent", "api_key")
        assert value is None
    
    def test_delete_credential(self, settings_instance):
        """Test deleting a stored credential."""
        provider = "gemini"
        credential_type = "api_key"
        credential_value = "test_key"
        
        # Store and verify
        settings_instance.store_credential(provider, credential_type, credential_value)
        assert settings_instance.get_credential(provider, credential_type) == credential_value
        
        # Delete and verify
        success = settings_instance.delete_credential(provider, credential_type)
        assert success is True
        assert settings_instance.get_credential(provider, credential_type) is None
    
    def test_list_stored_credentials(self, settings_instance):
        """Test listing all stored credentials."""
        # Store credentials for multiple providers
        settings_instance.store_credential("gemini", "api_key", "gemini_key")
        settings_instance.store_credential("gemini", "project_id", "project_123")
        settings_instance.store_credential("openai", "api_key", "openai_key")
        
        credentials = settings_instance.list_stored_credentials()
        
        assert "gemini" in credentials
        assert "openai" in credentials
        assert "api_key" in credentials["gemini"]
        assert "project_id" in credentials["gemini"]
        assert "api_key" in credentials["openai"]
    
    def test_credential_encryption(self, settings_instance):
        """Test that credentials are actually encrypted on disk."""
        provider = "gemini"
        credential_type = "api_key"
        credential_value = "sensitive_api_key_12345"
        
        # Store credential
        settings_instance.store_credential(provider, credential_type, credential_value)
        
        # Check that raw file content is encrypted (not plaintext)
        credentials_file = settings_instance._credentials_path
        if credentials_file.exists():
            with open(credentials_file, 'r') as f:
                raw_content = f.read()
            
            # The sensitive value should not appear in plaintext
            assert credential_value not in raw_content
            assert "sensitive_api_key_12345" not in raw_content
    
    def test_empty_credential_handling(self, settings_instance):
        """Test handling of empty or invalid credentials."""
        # Empty credential value
        success = settings_instance.store_credential("provider", "type", "")
        assert success is True  # Empty string is valid
        
        retrieved = settings_instance.get_credential("provider", "type")
        assert retrieved == ""