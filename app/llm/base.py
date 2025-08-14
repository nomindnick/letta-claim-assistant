"""
Base protocols and interfaces for LLM providers.

Defines the contract that all LLM providers must implement for consistent
integration across the application.
"""

from typing import Protocol, List, Dict, Any
from abc import ABC, abstractmethod


class LLMProvider(Protocol):
    """Protocol for LLM generation providers."""
    
    async def generate(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 900,
        temperature: float = 0.2
    ) -> str:
        """
        Generate text completion from messages.
        
        Args:
            system: System prompt
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Generated text response
        """
        ...
    
    async def test_connection(self) -> bool:
        """Test if the provider is available and working."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per input text)
        """
        ...
    
    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        ...
    
    async def test_connection(self) -> bool:
        """Test if the embedding provider is available."""
        ...


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers with common functionality."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    async def generate(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 900,
        temperature: float = 0.2
    ) -> str:
        """Generate text completion."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test provider connection."""
        pass
    
    def _format_messages(self, system: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages with system prompt for the provider."""
        formatted = [{"role": "system", "content": system}]
        formatted.extend(messages)
        return formatted


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        results = await self.embed([text])
        return results[0] if results else []
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test embedding provider connection."""
        pass