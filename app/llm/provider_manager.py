"""
LLM Provider Manager for runtime provider switching and configuration.

Provides a unified interface for managing different LLM providers
and switching between them without application restart.
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import asyncio
import json
from pathlib import Path
from datetime import datetime

from .base import LLMProvider, EmbeddingProvider
from .ollama_provider import OllamaProvider, OllamaEmbeddings
from ..logging_conf import get_logger
from ..privacy_consent import consent_manager, ConsentType

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
        self._provider_metrics: Dict[str, Dict[str, Any]] = {}
        
        # State persistence
        self._state_file = Path.home() / ".letta-claim" / "provider_state.json"
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()
        
        logger.info("Provider manager initialized")
    
    def _load_state(self) -> None:
        """Load provider state from disk."""
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                
                self._active_provider = state.get("active_provider")
                self._active_embedding_provider = state.get("active_embedding_provider")
                self._provider_metrics = state.get("provider_metrics", {})
                
                logger.debug("Loaded provider state", active_provider=self._active_provider)
                
            except Exception as e:
                logger.error("Failed to load provider state", error=str(e))
    
    def _save_state(self) -> None:
        """Save provider state to disk."""
        try:
            state = {
                "active_provider": self._active_provider,
                "active_embedding_provider": self._active_embedding_provider,
                "provider_metrics": self._provider_metrics,
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.debug("Saved provider state")
            
        except Exception as e:
            logger.error("Failed to save provider state", error=str(e))
    
    def _record_provider_metric(
        self, 
        provider_key: str, 
        metric_type: str, 
        value: Any,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a metric for a provider."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if provider_key not in self._provider_metrics:
            self._provider_metrics[provider_key] = {
                "response_times": [],
                "success_count": 0,
                "error_count": 0,
                "last_used": None,
                "total_requests": 0
            }
        
        metrics = self._provider_metrics[provider_key]
        
        if metric_type == "response_time":
            metrics["response_times"].append({
                "time": value,
                "timestamp": timestamp.isoformat()
            })
            # Keep only last 100 response times
            if len(metrics["response_times"]) > 100:
                metrics["response_times"] = metrics["response_times"][-100:]
        
        elif metric_type == "success":
            metrics["success_count"] += 1
            metrics["total_requests"] += 1
            metrics["last_used"] = timestamp.isoformat()
        
        elif metric_type == "error":
            metrics["error_count"] += 1
            metrics["total_requests"] += 1
        
        self._save_state()
    
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
        model: str = "gemini-2.0-flash-exp",
        check_consent: bool = True
    ) -> Dict[str, Any]:
        """Register Gemini provider with API key and consent checking."""
        try:
            # Check consent first if required
            if check_consent:
                consent_requirements = consent_manager.get_consent_requirements("gemini")
                if consent_requirements["requires_consent"] and not consent_requirements["consent_granted"]:
                    return {
                        "success": False,
                        "error": "consent_required",
                        "message": "User consent required for external LLM usage",
                        "consent_requirements": consent_requirements
                    }
            
            # Import here to avoid dependency issues if google-genai not installed
            from .gemini_provider import GeminiProvider
            
            provider = GeminiProvider(api_key=api_key, model=model)
            connection_ok = await provider.test_connection()
            
            if not connection_ok:
                logger.error(
                    "Failed to connect to Gemini API",
                    model=model
                )
                return {
                    "success": False,
                    "error": "connection_failed",
                    "message": "Failed to connect to Gemini API. Please check your API key."
                }
            
            provider_key = f"gemini_{model}"
            self._providers[provider_key] = provider
            
            # Initialize metrics for new provider
            if provider_key not in self._provider_metrics:
                self._provider_metrics[provider_key] = {
                    "response_times": [],
                    "success_count": 0,
                    "error_count": 0,
                    "last_used": None,
                    "total_requests": 0
                }
            
            self._save_state()
            
            logger.info(
                "Gemini provider registered successfully",
                model=model,
                provider_key=provider_key
            )
            
            return {
                "success": True,
                "provider_key": provider_key,
                "model": model,
                "message": "Gemini provider registered successfully"
            }
            
        except ImportError:
            logger.error("Gemini provider not available - google-genai package not installed")
            return {
                "success": False,
                "error": "dependency_missing",
                "message": "Gemini provider not available - google-generativeai package not installed"
            }
        except Exception as e:
            logger.error(
                "Failed to register Gemini provider",
                model=model,
                error=str(e)
            )
            return {
                "success": False,
                "error": "registration_failed",
                "message": f"Failed to register Gemini provider: {str(e)}"
            }
    
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
            "registered_embedding_providers": list(self._embedding_providers.keys()),
            "provider_metrics": self._provider_metrics
        }
    
    def get_provider_metrics(self, provider_key: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for providers."""
        if provider_key:
            return self._provider_metrics.get(provider_key, {})
        else:
            # Return summary metrics for all providers
            summary = {}
            for key, metrics in self._provider_metrics.items():
                avg_response_time = 0
                if metrics["response_times"]:
                    avg_response_time = sum(
                        rt["time"] for rt in metrics["response_times"]
                    ) / len(metrics["response_times"])
                
                success_rate = 0
                if metrics["total_requests"] > 0:
                    success_rate = metrics["success_count"] / metrics["total_requests"]
                
                summary[key] = {
                    "avg_response_time": round(avg_response_time, 2),
                    "success_rate": round(success_rate, 2),
                    "total_requests": metrics["total_requests"],
                    "last_used": metrics["last_used"]
                }
            
            return summary
    
    def check_provider_consent(self, provider_name: str) -> Dict[str, Any]:
        """Check consent status for a provider."""
        return consent_manager.get_consent_requirements(provider_name)
    
    def grant_provider_consent(self, provider_name: str) -> bool:
        """Grant consent for external provider usage."""
        return consent_manager.grant_consent(ConsentType.EXTERNAL_LLM, provider_name)
    
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