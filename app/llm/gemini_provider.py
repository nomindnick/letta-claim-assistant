"""
Google Gemini provider for external LLM generation.

Provides integration with Google's Gemini API for text generation
with proper consent and privacy handling.
"""

from typing import List, Dict, Any, Optional
import google.generativeai as genai

from .base import BaseLLMProvider
from ..logging_conf import get_logger

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider for external text generation."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gemini-2.5-flash",
        safety_settings: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model)
        self.api_key = api_key
        self.safety_settings = safety_settings or self._default_safety_settings()
        
        # Configure the API
        if api_key:
            genai.configure(api_key=api_key)
        
        self.model = None
        if api_key:
            try:
                self.model = genai.GenerativeModel(
                    model_name=model,
                    safety_settings=self.safety_settings
                )
            except Exception as e:
                logger.error("Failed to initialize Gemini model", error=str(e))
    
    async def generate(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 900,
        temperature: float = 0.2
    ) -> str:
        """
        Generate text using Google Gemini API.
        
        Args:
            system: System prompt
            messages: Conversation messages  
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Generated text response
        """
        if not self.model:
            raise RuntimeError("Gemini model not initialized. Check API key.")
        
        if not messages:
            raise ValueError("No messages provided for generation")
        
        try:
            # Format the full prompt including system prompt and messages
            prompt = self._format_messages_for_gemini(system, messages)
            
            logger.debug(
                "Sending generation request to Gemini",
                model=self.model_name,
                prompt_length=len(prompt),
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                top_k=40
            )
            
            # Generate response
            response = await self._async_generate_content(
                prompt=prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            if response and response.text:
                generated_text = response.text.strip()
                
                logger.debug(
                    "Gemini generation completed",
                    model=self.model_name,
                    response_length=len(generated_text),
                    prompt_feedback=getattr(response, 'prompt_feedback', None)
                )
                
                return generated_text
            else:
                error_msg = "Empty response from Gemini API"
                if response and hasattr(response, 'prompt_feedback'):
                    error_msg += f" - Prompt feedback: {response.prompt_feedback}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(
                "Gemini generation error",
                error=str(e),
                model=self.model_name,
                error_type=type(e).__name__
            )
            raise RuntimeError(f"Gemini generation failed: {str(e)}")
    
    async def _async_generate_content(self, prompt: str, generation_config) -> Any:
        """Async wrapper for Gemini's generate_content method."""
        import asyncio
        
        # Run the synchronous Gemini API call in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
        )
    
    async def test_connection(self) -> bool:
        """Test connection to Gemini API."""
        if not self.api_key or not self.model:
            logger.warning("Gemini test failed: API key or model not configured")
            return False
        
        try:
            # Simple test generation using async wrapper
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=10,
                temperature=0.1,
            )
            
            response = await self._async_generate_content(
                prompt="Test connection. Respond with 'OK'.",
                generation_config=generation_config
            )
            
            if response and response.text:
                result = "OK" in response.text.upper()
                logger.debug("Gemini connection test result", success=result, response=response.text)
                return result
            else:
                logger.warning("Gemini connection test: empty response")
                return False
                
        except Exception as e:
            logger.error("Gemini connection test failed", error=str(e), error_type=type(e).__name__)
            return False
    
    def _default_safety_settings(self) -> List[Dict[str, Any]]:
        """Get default safety settings for Gemini."""
        return [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
    
    def _format_messages_for_gemini(
        self, 
        system: str, 
        messages: List[Dict[str, str]]
    ) -> str:
        """Format messages into single prompt for Gemini."""
        # Start with system prompt
        prompt_parts = []
        
        if system:
            prompt_parts.append(f"System: {system}\n")
        
        # Add conversation messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # Skip system messages as we handle them separately
                continue
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                # Handle any other roles as generic
                prompt_parts.append(f"{role.title()}: {content}")
        
        # Add final prompt for assistant response
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)