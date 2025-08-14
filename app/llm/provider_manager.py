"""
LLM Provider Manager for runtime provider switching and configuration.

Provides a unified interface for managing different LLM providers
and switching between them without application restart.
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import asyncio

from .base import LLMProvider, EmbeddingProvider
from .ollama_provider import OllamaProvider, OllamaEmbeddings
from ..logging_conf import get_logger

logger = get_logger(__name__)


class ProviderType(Enum):
    """Available LLM provider types."""
    OLLAMA = "ollama"
    GEMINI = "gemini"


class ProviderManager:
    """Manages LLM providers and enables runtime switching."""
    
    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        self._embedding_providers: Dict[str, EmbeddingProvider] = {}
        self._active_provider: Optional[str] = None
        self._active_embedding_provider: Optional[str] = None
        
        logger.info("Provider manager initialized")
    
    async def register_ollama_provider(
        self, 
        model: str = "gpt-oss:20b",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ) -> bool:
        """Register Ollama provider with specified models."""
        try:
            # Register generation provider
            gen_provider = OllamaProvider(model=model, base_url=base_url)
            connection_ok = await gen_provider.test_connection()
            
            if not connection_ok:
                logger.error(
                    "Failed to connect to Ollama generation model",
                    model=model,
                    base_url=base_url
                )
                return False
            
            provider_key = f"ollama_{model}"
            self._providers[provider_key] = gen_provider
            
            # Register embedding provider
            embed_provider = OllamaEmbeddings(model=embedding_model, base_url=base_url)
            embed_connection_ok = await embed_provider.test_connection()
            
            if not embed_connection_ok:
                logger.warning(
                    "Failed to connect to Ollama embedding model",
                    model=embedding_model,
                    base_url=base_url
                )
                # Continue without embedding provider
            else:
                embed_key = f"ollama_{embedding_model}"
                self._embedding_providers[embed_key] = embed_provider
            
            # Set as active if this is the first provider
            if self._active_provider is None:
                self._active_provider = provider_key
            
            if self._active_embedding_provider is None and embed_connection_ok:
                self._active_embedding_provider = embed_key
            
            logger.info(
                "Ollama provider registered successfully",
                generation_model=model,
                embedding_model=embedding_model,
                provider_key=provider_key,
                embed_key=embed_key if embed_connection_ok else None
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to register Ollama provider",
                model=model,
                embedding_model=embedding_model,
                error=str(e)
            )
            return False
    
    async def register_gemini_provider(
        self, 
        api_key: str,
        model: str = "gemini-2.0-flash-exp"
    ) -> bool:
        """Register Gemini provider with API key."""
        try:
            # Import here to avoid dependency issues if google-genai not installed
            from .gemini_provider import GeminiProvider
            
            provider = GeminiProvider(api_key=api_key, model=model)
            connection_ok = await provider.test_connection()
            
            if not connection_ok:
                logger.error(
                    "Failed to connect to Gemini API",
                    model=model
                )
                return False
            
            provider_key = f"gemini_{model}"
            self._providers[provider_key] = provider
            
            logger.info(
                "Gemini provider registered successfully",
                model=model,
                provider_key=provider_key
            )
            
            return True
            
        except ImportError:
            logger.error("Gemini provider not available - google-genai package not installed")
            return False
        except Exception as e:
            logger.error(
                "Failed to register Gemini provider",
                model=model,
                error=str(e)
            )
            return False
    
    def get_active_provider(self) -> Optional[LLMProvider]:
        """Get the currently active LLM provider."""
        if self._active_provider and self._active_provider in self._providers:
            return self._providers[self._active_provider]
        return None
    
    def get_active_embedding_provider(self) -> Optional[EmbeddingProvider]:
        """Get the currently active embedding provider."""
        if self._active_embedding_provider and self._active_embedding_provider in self._embedding_providers:
            return self._embedding_providers[self._active_embedding_provider]
        return None
    
    def switch_provider(self, provider_key: str) -> bool:
        """Switch to a different registered provider."""
        if provider_key not in self._providers:
            logger.error("Provider not found", provider_key=provider_key)
            return False
        
        old_provider = self._active_provider
        self._active_provider = provider_key
        
        logger.info(
            "Provider switched",
            from_provider=old_provider,
            to_provider=provider_key
        )
        
        return True
    
    def switch_embedding_provider(self, provider_key: str) -> bool:
        """Switch to a different registered embedding provider."""
        if provider_key not in self._embedding_providers:
            logger.error("Embedding provider not found", provider_key=provider_key)
            return False
        
        old_provider = self._active_embedding_provider
        self._active_embedding_provider = provider_key
        
        logger.info(
            "Embedding provider switched",
            from_provider=old_provider,
            to_provider=provider_key
        )
        
        return True
    
    def list_providers(self) -> Dict[str, Dict[str, Any]]:
        """List all registered providers with their status."""
        provider_info = {}
        
        for key, provider in self._providers.items():
            provider_info[key] = {
                "type": "generation",
                "model": getattr(provider, 'model_name', 'unknown'),
                "active": key == self._active_provider,
                "provider_class": provider.__class__.__name__
            }
        
        for key, provider in self._embedding_providers.items():
            provider_info[key] = {
                "type": "embedding",
                "model": getattr(provider, 'model_name', 'unknown'),
                "active": key == self._active_embedding_provider,
                "provider_class": provider.__class__.__name__
            }
        
        return provider_info
    
    async def test_provider(self, provider_key: str) -> bool:
        """Test connection to a specific provider."""
        if provider_key in self._providers:
            try:
                return await self._providers[provider_key].test_connection()
            except Exception as e:
                logger.error("Provider test failed", provider_key=provider_key, error=str(e))
                return False
        
        if provider_key in self._embedding_providers:
            try:
                return await self._embedding_providers[provider_key].test_connection()
            except Exception as e:
                logger.error("Embedding provider test failed", provider_key=provider_key, error=str(e))
                return False
        
        logger.error("Provider not found for testing", provider_key=provider_key)
        return False
    
    async def test_all_providers(self) -> Dict[str, bool]:
        """Test all registered providers."""
        test_results = {}
        
        # Test generation providers
        for key in self._providers:
            test_results[key] = await self.test_provider(key)
        
        # Test embedding providers
        for key in self._embedding_providers:
            test_results[key] = await self.test_provider(key)
        
        return test_results
    
    def get_provider_config(self) -> Dict[str, Any]:
        """Get current provider configuration."""
        return {
            "active_provider": self._active_provider,
            "active_embedding_provider": self._active_embedding_provider,
            "registered_providers": list(self._providers.keys()),
            "registered_embedding_providers": list(self._embedding_providers.keys())
        }
    
    async def quick_test_generation(self, provider_key: Optional[str] = None) -> Dict[str, Any]:
        """Perform a quick generation test."""
        if provider_key is None:
            provider_key = self._active_provider
        
        if not provider_key or provider_key not in self._providers:
            return {"success": False, "error": "No provider available"}
        
        provider = self._providers[provider_key]
        
        try:
            test_messages = [{"role": "user", "content": "Hello, please respond with 'Test successful'"}]
            
            start_time = asyncio.get_event_loop().time()
            response = await provider.generate(
                system="You are a helpful assistant.",
                messages=test_messages,
                max_tokens=50,
                temperature=0.1
            )
            end_time = asyncio.get_event_loop().time()
            
            return {
                "success": True,
                "provider": provider_key,
                "response": response,
                "response_time_seconds": round(end_time - start_time, 2),
                "response_length": len(response)
            }
            
        except Exception as e:
            return {
                "success": False,
                "provider": provider_key,
                "error": str(e)
            }


# Global provider manager instance
provider_manager = ProviderManager()