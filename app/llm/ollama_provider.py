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
        if not messages:
            raise ValueError("No messages provided for generation")
        
        # Format messages with system prompt
        formatted_messages = self._format_messages(system, messages)
        
        # Build options dict
        options = {
            "temperature": temperature
        }
        
        # Only add num_predict if we want to limit tokens
        # Note: gpt-oss model has issues with small num_predict values
        if max_tokens and max_tokens > 100:  # Only set if reasonable limit
            options["num_predict"] = max_tokens
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": False,  # For now, use non-streaming for simplicity
            "options": options
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                logger.debug(
                    "Sending generation request to Ollama",
                    model=self.model_name,
                    message_count=len(formatted_messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages_preview=[{"role": m["role"], "content_length": len(m["content"])} for m in formatted_messages]
                )
                
                async with session.post(
                    self.chat_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 minute timeout for large models
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            "Ollama generation request failed",
                            status=response.status,
                            error=error_text,
                            model=self.model_name
                        )
                        raise RuntimeError(f"Ollama generation failed: HTTP {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    # Log the full response for debugging
                    logger.debug(
                        "Ollama raw response",
                        model=self.model_name,
                        response_keys=list(data.keys()),
                        has_message="message" in data,
                        message_content_length=len(data.get("message", {}).get("content", "")) if "message" in data else 0,
                        done=data.get("done"),
                        total_duration_ms=data.get("total_duration", 0) / 1_000_000 if data.get("total_duration") else 0
                    )
                    
                    # Extract response content
                    if "message" in data:
                        message = data["message"]
                        generated_text = message.get("content", "")
                        
                        # Handle gpt-oss model's quirk: sometimes content is in "thinking" field
                        # when the response is cut off due to num_predict limit
                        if not generated_text and "thinking" in message:
                            thinking = message.get("thinking", "")
                            logger.warning(
                                "gpt-oss model returned content in 'thinking' field",
                                model=self.model_name,
                                thinking_preview=thinking[:200]
                            )
                            # For now, don't use thinking as the response
                            # This usually indicates the response was truncated
                        
                        logger.debug(
                            "Generation completed",
                            model=self.model_name,
                            response_length=len(generated_text),
                            prompt_eval_count=data.get("prompt_eval_count", 0),
                            eval_count=data.get("eval_count", 0),
                            done_reason=data.get("done_reason", "unknown")
                        )
                        
                        # Check for empty response
                        if not generated_text or not generated_text.strip():
                            logger.error(
                                "LLM generated empty response",
                                model=self.model_name,
                                full_response=data,
                                prompt_eval_count=data.get("prompt_eval_count", 0),
                                eval_count=data.get("eval_count", 0),
                                done_reason=data.get("done_reason", "unknown")
                            )
                            raise RuntimeError("LLM generated empty response")
                        
                        return generated_text.strip()
                    else:
                        logger.error("Invalid response format from Ollama", response_data=data)
                        raise RuntimeError("Invalid response format from Ollama API")
                        
        except asyncio.TimeoutError:
            logger.error("Ollama generation timeout", model=self.model_name, timeout=600)
            raise RuntimeError(f"Generation timed out after 600 seconds")
        except Exception as e:
            logger.error("Ollama generation error", error=str(e), model=self.model_name)
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    async def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            async with aiohttp.ClientSession() as session:
                # First check if server is running and model exists
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    if response.status != 200:
                        return False
                    
                    data = await response.json()
                    # Check if our model is available
                    models = [model["name"] for model in data.get("models", [])]
                    if self.model_name not in models:
                        logger.warning(
                            "Model not found in Ollama",
                            model=self.model_name,
                            available_models=models
                        )
                        return False
                
                # Try a simple generation test
                test_payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Reply with 'OK' if you are working."}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 50  # Increased from 10 to avoid truncation
                    }
                }
                
                async with session.post(
                    self.chat_url,
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get("message", {}).get("content", "")
                        logger.debug(
                            "Ollama test generation result",
                            model=self.model_name,
                            response_length=len(content),
                            content_preview=content[:100] if content else "(empty)"
                        )
                        return bool(content and content.strip())
                    else:
                        error_text = await response.text()
                        logger.error(
                            "Ollama test generation failed",
                            status=response.status,
                            error=error_text
                        )
                        return False
                        
        except Exception as e:
            logger.error("Ollama connection test failed", error=str(e))
            return False


class OllamaEmbeddings(BaseEmbeddingProvider):
    """Ollama provider for local embeddings generation."""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        super().__init__(model)
        self.base_url = base_url
        self.embed_url = f"{base_url}/api/embed"
    
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
                            if "embeddings" in data and data["embeddings"] and len(data["embeddings"]) > 0:
                                batch_embeddings.append(data["embeddings"][0])
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
                        return "embeddings" in data and data["embeddings"] and len(data["embeddings"]) > 0
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