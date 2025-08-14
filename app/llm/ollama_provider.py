"""
Ollama provider for local LLM generation and embeddings.

Integrates with local Ollama installation for both text generation
and embedding creation using specified models.
"""

from typing import List, Dict, Any
import aiohttp
import asyncio
import json

from .base import BaseLLMProvider, BaseEmbeddingProvider
from ..logging_conf import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local text generation."""
    
    def __init__(self, model: str = "gpt-oss:20b", base_url: str = "http://localhost:11434"):
        super().__init__(model)
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
    
    async def generate(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 900,
        temperature: float = 0.2
    ) -> str:
        """
        Generate text using Ollama chat API.
        
        Args:
            system: System prompt
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Generated text response
        """
        # TODO: Implement Ollama generation
        raise NotImplementedError("Ollama generation not yet implemented")
    
    async def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if our model is available
                        models = [model["name"] for model in data.get("models", [])]
                        return self.model_name in models
                    return False
        except Exception as e:
            logger.error("Ollama connection test failed", error=str(e))
            return False


class OllamaEmbeddings(BaseEmbeddingProvider):
    """Ollama provider for local embeddings generation."""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        super().__init__(model)
        self.base_url = base_url
        self.embed_url = f"{base_url}/api/embeddings"
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Ollama.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        # TODO: Implement Ollama embeddings
        raise NotImplementedError("Ollama embeddings not yet implemented")
    
    async def test_connection(self) -> bool:
        """Test connection to Ollama embeddings."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if our embedding model is available
                        models = [model["name"] for model in data.get("models", [])]
                        return self.model_name in models
                    return False
        except Exception as e:
            logger.error("Ollama embeddings connection test failed", error=str(e))
            return False