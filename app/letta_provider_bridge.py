"""
Letta Provider Bridge - Bridge between provider_manager and Letta's LLM configuration.

Provides conversion between different provider configurations and Letta's expected
format, enabling dynamic provider switching and configuration management.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
from pathlib import Path
import json

try:
    from letta_client.types import LlmConfig, EmbeddingConfig
except ImportError:
    # Fallback for older versions
    LlmConfig = None
    EmbeddingConfig = None

from .llm.provider_manager import provider_manager, ProviderType
from .privacy_consent import consent_manager, ConsentType
from .logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class ProviderConfiguration:
    """Unified provider configuration for Letta agents."""
    
    provider_type: str  # ollama, gemini, openai, anthropic
    model_name: str
    api_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    context_window: int = 8192
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Embedding settings
    embedding_model: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embedding_dim: int = 768
    
    # Cost tracking
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    
    # Metadata
    requires_consent: bool = False
    is_local: bool = True


class LettaProviderBridge:
    """
    Bridge between provider_manager and Letta's configuration system.
    Handles provider configuration conversion and dynamic switching.
    """
    
    # Provider pricing (per 1K tokens) - Update these as needed
    PROVIDER_PRICING = {
        "gemini-2.0-flash-exp": {"input": 0.0001, "output": 0.0003},
        "gemini-1.5-flash": {"input": 0.00025, "output": 0.00075},
        "gemini-1.5-pro": {"input": 0.0025, "output": 0.0075},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        # Ollama models are free (local)
        "ollama": {"input": 0.0, "output": 0.0}
    }
    
    def __init__(self):
        """Initialize the provider bridge."""
        self.active_provider: Optional[ProviderConfiguration] = None
        self.fallback_chain: list[ProviderConfiguration] = []
        self._load_saved_preferences()
        
        logger.info("LettaProviderBridge initialized")
    
    def _load_saved_preferences(self) -> None:
        """Load saved provider preferences from disk."""
        pref_file = Path.home() / ".letta-claim" / "provider_preferences.json"
        if pref_file.exists():
            try:
                with open(pref_file, 'r') as f:
                    prefs = json.load(f)
                    # Restore active provider if specified
                    if prefs.get("active_provider"):
                        logger.debug(f"Loaded provider preference: {prefs['active_provider']}")
            except Exception as e:
                logger.warning(f"Could not load provider preferences: {e}")
    
    def get_ollama_config(
        self,
        model: str = "gpt-oss:20b",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ) -> ProviderConfiguration:
        """Get Ollama provider configuration."""
        return ProviderConfiguration(
            provider_type="ollama",
            model_name=model,
            endpoint_url=base_url,
            embedding_model=embedding_model,
            embedding_endpoint=base_url,
            embedding_dim=768 if "nomic" in embedding_model else 1536,
            is_local=True,
            requires_consent=False,
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.0
        )
    
    def get_gemini_config(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp"
    ) -> ProviderConfiguration:
        """Get Gemini provider configuration."""
        pricing = self.PROVIDER_PRICING.get(model, self.PROVIDER_PRICING["gemini-2.0-flash-exp"])
        
        return ProviderConfiguration(
            provider_type="google_ai",  # Letta expects "google_ai" for Gemini
            model_name=model,
            api_key=api_key,
            endpoint_url="https://generativelanguage.googleapis.com",
            context_window=32768 if "flash" in model else 128000,
            embedding_model="models/embedding-001",  # Gemini's embedding model
            embedding_endpoint="https://generativelanguage.googleapis.com",
            embedding_dim=768,
            is_local=False,
            requires_consent=True,
            cost_per_1k_input_tokens=pricing["input"],
            cost_per_1k_output_tokens=pricing["output"]
        )
    
    def get_openai_config(
        self,
        api_key: str,
        model: str = "gpt-4o-mini"
    ) -> ProviderConfiguration:
        """Get OpenAI provider configuration."""
        pricing = self.PROVIDER_PRICING.get(model, self.PROVIDER_PRICING["gpt-4o-mini"])
        
        return ProviderConfiguration(
            provider_type="openai",
            model_name=model,
            api_key=api_key,
            endpoint_url="https://api.openai.com/v1",
            context_window=128000 if "4o" in model else 16384,
            embedding_model="text-embedding-3-small",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_dim=1536,
            is_local=False,
            requires_consent=True,
            cost_per_1k_input_tokens=pricing["input"],
            cost_per_1k_output_tokens=pricing["output"]
        )
    
    def get_anthropic_config(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307"
    ) -> ProviderConfiguration:
        """Get Anthropic provider configuration."""
        # Map model names to pricing keys
        if "haiku" in model:
            pricing_key = "claude-3-haiku"
        elif "sonnet" in model:
            pricing_key = "claude-3-sonnet"
        elif "opus" in model:
            pricing_key = "claude-3-opus"
        else:
            pricing_key = "claude-3-haiku"
        
        pricing = self.PROVIDER_PRICING[pricing_key]
        
        return ProviderConfiguration(
            provider_type="anthropic",
            model_name=model,
            api_key=api_key,
            endpoint_url="https://api.anthropic.com",
            context_window=200000,  # Claude 3 has 200K context
            # Note: Anthropic doesn't provide embeddings, would need separate provider
            is_local=False,
            requires_consent=True,
            cost_per_1k_input_tokens=pricing["input"],
            cost_per_1k_output_tokens=pricing["output"]
        )
    
    def to_letta_llm_config(self, provider_config: ProviderConfiguration) -> Optional[Dict[str, Any]]:
        """
        Convert provider configuration to Letta's LlmConfig format.
        
        Args:
            provider_config: Provider configuration to convert
            
        Returns:
            Dictionary compatible with LlmConfig constructor, or None if LlmConfig not available
        """
        if LlmConfig is None:
            logger.warning("LlmConfig not available, returning dict format")
            return None
        
        # Add provider prefix to model name for Letta v0.10.0 compatibility
        model_name = provider_config.model_name
        if provider_config.provider_type == "ollama" and not model_name.startswith("ollama/"):
            model_name = f"ollama/{model_name}"
        elif provider_config.provider_type == "google_ai" and not model_name.startswith("gemini/"):
            model_name = f"gemini/{model_name}"
        elif provider_config.provider_type == "openai" and not model_name.startswith("openai/"):
            model_name = f"openai/{model_name}"
        elif provider_config.provider_type == "anthropic" and not model_name.startswith("anthropic/"):
            model_name = f"anthropic/{model_name}"
        
        config_dict = {
            "model": model_name,
            "model_endpoint_type": provider_config.provider_type,
            "model_endpoint": provider_config.endpoint_url,
            "context_window": provider_config.context_window,
            "max_tokens": provider_config.max_tokens,
            "temperature": provider_config.temperature,
            "top_p": provider_config.top_p
        }
        
        # Add API key if present
        if provider_config.api_key:
            # For Gemini, the key field is "api_key"
            if provider_config.provider_type == "google_ai":
                config_dict["api_key"] = provider_config.api_key
            # For OpenAI and Anthropic
            elif provider_config.provider_type in ["openai", "anthropic"]:
                config_dict["api_key"] = provider_config.api_key
        
        return config_dict
    
    def to_letta_embedding_config(self, provider_config: ProviderConfiguration) -> Optional[Dict[str, Any]]:
        """
        Convert provider configuration to Letta's EmbeddingConfig format.
        
        Args:
            provider_config: Provider configuration to convert
            
        Returns:
            Dictionary compatible with EmbeddingConfig constructor, or None if not available
        """
        if EmbeddingConfig is None or not provider_config.embedding_model:
            logger.warning("EmbeddingConfig not available or no embedding model specified")
            return None
        
        # Add provider prefix to embedding model name for Letta v0.10.0 compatibility
        embedding_model = provider_config.embedding_model
        if provider_config.provider_type == "ollama" and not embedding_model.startswith("ollama/"):
            embedding_model = f"ollama/{embedding_model}"
        elif provider_config.provider_type == "google_ai" and not embedding_model.startswith("models/"):
            # Gemini uses "models/" prefix for embeddings
            embedding_model = embedding_model
        elif provider_config.provider_type == "openai" and not embedding_model.startswith("openai/"):
            embedding_model = f"openai/{embedding_model}"
        
        config_dict = {
            "embedding_model": embedding_model,
            "embedding_endpoint_type": provider_config.provider_type,
            "embedding_endpoint": provider_config.embedding_endpoint or provider_config.endpoint_url,
            "embedding_dim": provider_config.embedding_dim
        }
        
        # Add API key if present
        if provider_config.api_key and not provider_config.is_local:
            config_dict["api_key"] = provider_config.api_key
        
        return config_dict
    
    def create_letta_configs(
        self,
        provider_config: ProviderConfiguration
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Create both LlmConfig and EmbeddingConfig objects for Letta.
        
        Args:
            provider_config: Provider configuration to use
            
        Returns:
            Tuple of (LlmConfig, EmbeddingConfig) or (None, None) if not available
        """
        if LlmConfig is None or EmbeddingConfig is None:
            logger.error("Letta config types not available")
            return None, None
        
        try:
            # Create LLM config
            llm_dict = self.to_letta_llm_config(provider_config)
            llm_config = LlmConfig(**llm_dict) if llm_dict else None
            
            # Create embedding config
            embed_dict = self.to_letta_embedding_config(provider_config)
            embed_config = EmbeddingConfig(**embed_dict) if embed_dict else None
            
            return llm_config, embed_config
            
        except Exception as e:
            logger.error(f"Failed to create Letta configs: {e}")
            return None, None
    
    def check_provider_consent(self, provider_config: ProviderConfiguration) -> bool:
        """
        Check if user consent is required and granted for a provider.
        
        Args:
            provider_config: Provider configuration to check
            
        Returns:
            True if consent is not required or has been granted
        """
        if not provider_config.requires_consent:
            return True
        
        # Map provider types to consent manager names
        provider_map = {
            "google_ai": "gemini",
            "openai": "openai",
            "anthropic": "anthropic"
        }
        
        provider_name = provider_map.get(provider_config.provider_type, provider_config.provider_type)
        consent_status = consent_manager.get_consent_requirements(provider_name)
        
        return consent_status.get("consent_granted", False)
    
    def setup_fallback_chain(
        self,
        primary: ProviderConfiguration,
        secondary: Optional[ProviderConfiguration] = None,
        tertiary: Optional[ProviderConfiguration] = None
    ) -> None:
        """
        Set up provider fallback chain for automatic failover.
        
        Args:
            primary: Primary provider configuration
            secondary: Optional secondary provider
            tertiary: Optional tertiary provider
        """
        self.fallback_chain = [primary]
        
        if secondary:
            self.fallback_chain.append(secondary)
        if tertiary:
            self.fallback_chain.append(tertiary)
        
        self.active_provider = primary
        
        logger.info(
            f"Fallback chain configured with {len(self.fallback_chain)} providers",
            providers=[p.model_name for p in self.fallback_chain]
        )
    
    def get_next_provider(self) -> Optional[ProviderConfiguration]:
        """
        Get the next provider in the fallback chain.
        
        Returns:
            Next provider configuration or None if no more providers
        """
        if not self.fallback_chain or not self.active_provider:
            return None
        
        try:
            current_idx = self.fallback_chain.index(self.active_provider)
            if current_idx < len(self.fallback_chain) - 1:
                next_provider = self.fallback_chain[current_idx + 1]
                
                # Check consent for external providers
                if next_provider.requires_consent and not self.check_provider_consent(next_provider):
                    logger.warning(f"Consent not granted for {next_provider.model_name}, skipping")
                    # Recursively try next provider
                    self.active_provider = next_provider
                    return self.get_next_provider()
                
                return next_provider
        except ValueError:
            # Current provider not in chain, return first provider
            if self.fallback_chain:
                return self.fallback_chain[0]
        
        return None
    
    def switch_to_provider(self, provider_config: ProviderConfiguration) -> bool:
        """
        Switch to a specific provider configuration.
        
        Args:
            provider_config: Provider to switch to
            
        Returns:
            True if switch successful
        """
        # Check consent if required
        if provider_config.requires_consent and not self.check_provider_consent(provider_config):
            logger.error(f"Cannot switch to {provider_config.model_name}: consent not granted")
            return False
        
        self.active_provider = provider_config
        
        # Save preference
        self._save_provider_preference(provider_config)
        
        logger.info(f"Switched to provider: {provider_config.model_name}")
        return True
    
    def _save_provider_preference(self, provider_config: ProviderConfiguration) -> None:
        """Save provider preference to disk."""
        pref_file = Path.home() / ".letta-claim" / "provider_preferences.json"
        pref_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            prefs = {
                "active_provider": provider_config.model_name,
                "provider_type": provider_config.provider_type,
                "is_local": provider_config.is_local
            }
            
            with open(pref_file, 'w') as f:
                json.dump(prefs, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save provider preference: {e}")
    
    def get_provider_for_matter(
        self,
        matter_id: str,
        matter_preferences: Optional[Dict[str, Any]] = None
    ) -> ProviderConfiguration:
        """
        Get provider configuration for a specific matter.
        
        Args:
            matter_id: Matter identifier
            matter_preferences: Optional matter-specific preferences
            
        Returns:
            Provider configuration for the matter
        """
        # Check for matter-specific preference
        if matter_preferences and "llm_provider" in matter_preferences:
            provider_type = matter_preferences["llm_provider"]
            model = matter_preferences.get("llm_model")
            
            # Build configuration based on preference
            if provider_type == "ollama":
                return self.get_ollama_config(model=model or "gpt-oss:20b")
            elif provider_type == "gemini":
                api_key = matter_preferences.get("api_key") or os.getenv("GEMINI_API_KEY")
                if api_key:
                    return self.get_gemini_config(api_key=api_key, model=model or "gemini-2.0-flash-exp")
            elif provider_type == "openai":
                api_key = matter_preferences.get("api_key") or os.getenv("OPENAI_API_KEY")
                if api_key:
                    return self.get_openai_config(api_key=api_key, model=model or "gpt-4o-mini")
        
        # Fall back to active provider or default Ollama
        return self.active_provider or self.get_ollama_config()
    
    def estimate_cost(
        self,
        provider_config: ProviderConfiguration,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate cost for a request with given token counts.
        
        Args:
            provider_config: Provider configuration
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        input_cost = (input_tokens / 1000) * provider_config.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * provider_config.cost_per_1k_output_tokens
        
        return input_cost + output_cost


# Global provider bridge instance
letta_provider_bridge = LettaProviderBridge()