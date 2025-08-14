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
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches to avoid overwhelming the API
        batch_size = 100
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                
                for text in batch:
                    try:
                        payload = {
                            "model": self.model_name,
                            "input": text
                        }
                        
                        async with session.post(
                            self.embed_url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(
                                    "Ollama embedding request failed",
                                    status=response.status,
                                    error=error_text,
                                    model=self.model_name
                                )
                                # Return zero vector as fallback
                                batch_embeddings.append([0.0] * 768)  # Common embedding size
                                continue
                            
                            data = await response.json()
                            if "embedding" in data and data["embedding"]:
                                batch_embeddings.append(data["embedding"])
                            else:
                                logger.warning("Empty embeddings response", text_preview=text[:100])
                                batch_embeddings.append([0.0] * 768)
                                
                    except asyncio.TimeoutError:
                        logger.error("Ollama embedding timeout", text_preview=text[:100])
                        batch_embeddings.append([0.0] * 768)
                    except Exception as e:
                        logger.error("Ollama embedding error", error=str(e), text_preview=text[:100])
                        batch_embeddings.append([0.0] * 768)
                
                embeddings.extend(batch_embeddings)
                
                # Small delay between batches to avoid rate limiting
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
        
        logger.debug("Generated embeddings", count=len(embeddings), model=self.model_name)
        return embeddings
    
    async def test_connection(self) -> bool:
        """Test connection to Ollama embeddings."""
        try:
            async with aiohttp.ClientSession() as session:
                # First check if Ollama server is running
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    if response.status != 200:
                        return False
                    
                    data = await response.json()
                    # Check if our embedding model is available (handle version tags)
                    models = [model["name"] for model in data.get("models", [])]
                    model_found = self.model_name in models or f"{self.model_name}:latest" in models
                    
                    if not model_found:
                        logger.warning(
                            "Embedding model not found in Ollama",
                            model=self.model_name,
                            available=models
                        )
                        return False
                
                # Test actual embedding generation with a simple text
                test_payload = {
                    "model": self.model_name,
                    "input": "test embedding"
                }
                
                async with session.post(
                    self.embed_url,
                    json=test_payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return "embedding" in data and data["embedding"]
                    else:
                        error_text = await response.text()
                        logger.error(
                            "Ollama embedding test failed",
                            status=response.status,
                            error=error_text
                        )
                        return False
                        
        except Exception as e:
            logger.error("Ollama embeddings connection test failed", error=str(e))
            return False