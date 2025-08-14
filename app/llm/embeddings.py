"""
Embedding model abstraction layer.

Provides unified interface for embedding generation across different
providers with automatic provider selection and caching.
"""

from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from .base import EmbeddingProvider
from .ollama_provider import OllamaEmbeddings
from ..settings import settings
from ..logging_conf import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """Manages embedding providers and provides unified interface."""
    
    def __init__(self):
        self._providers: Dict[str, EmbeddingProvider] = {}
        self._active_provider: Optional[EmbeddingProvider] = None
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize available embedding providers."""
        config = settings.global_config
        
        # Initialize Ollama embeddings
        try:
            ollama_embeddings = OllamaEmbeddings(model=config.embeddings_model)
            self._providers["ollama"] = ollama_embeddings
        except Exception as e:
            logger.error("Failed to initialize Ollama embeddings", error=str(e))
        
        # Set active provider based on configuration
        if config.embeddings_provider in self._providers:
            self._active_provider = self._providers[config.embeddings_provider]
        elif self._providers:
            # Fallback to first available provider
            self._active_provider = next(iter(self._providers.values()))
            logger.warning(
                "Configured embedding provider not available, using fallback",
                configured=config.embeddings_provider,
                fallback=list(self._providers.keys())[0]
            )
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If no embedding provider is available
        """
        if not self._active_provider:
            raise RuntimeError("No embedding provider available")
        
        try:
            return await self._active_provider.embed(texts)
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise
    
    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
            
        Raises:
            RuntimeError: If no embedding provider is available
        """
        if not self._active_provider:
            raise RuntimeError("No embedding provider available")
        
        try:
            return await self._active_provider.embed_single(text)
        except Exception as e:
            logger.error("Single embedding generation failed", error=str(e))
            raise
    
    async def test_connection(self) -> bool:
        """Test if the active embedding provider is working."""
        if not self._active_provider:
            return False
        
        try:
            return await self._active_provider.test_connection()
        except Exception as e:
            logger.error("Embedding provider connection test failed", error=str(e))
            return False
    
    def switch_provider(self, provider_name: str, model_name: Optional[str] = None) -> bool:
        """
        Switch to a different embedding provider.
        
        Args:
            provider_name: Name of provider to switch to
            model_name: Optional model name for the provider
            
        Returns:
            True if switch was successful, False otherwise
        """
        if provider_name not in self._providers:
            logger.error("Unknown embedding provider", provider=provider_name)
            return False
        
        try:
            if model_name and provider_name == "ollama":
                # Reinitialize with new model
                self._providers[provider_name] = OllamaEmbeddings(model=model_name)
            
            self._active_provider = self._providers[provider_name]
            logger.info("Switched embedding provider", provider=provider_name, model=model_name)
            return True
        except Exception as e:
            logger.error("Failed to switch embedding provider", error=str(e))
            return False
    
    def get_available_providers(self) -> List[str]:
        """Get list of available embedding providers."""
        return list(self._providers.keys())
    
    def get_active_provider_info(self) -> Dict[str, Any]:
        """Get information about the active embedding provider."""
        if not self._active_provider:
            return {"provider": None, "model": None}
        
        return {
            "provider": next(
                name for name, provider in self._providers.items() 
                if provider == self._active_provider
            ),
            "model": getattr(self._active_provider, "model_name", None)
        }


# Global embedding manager instance
embedding_manager = EmbeddingManager()