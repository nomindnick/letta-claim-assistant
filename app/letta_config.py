"""
Letta Server Configuration Module

Provides configuration management for the Letta server and client,
including connection settings, resource limits, and deployment options.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import os

from .logging_conf import get_logger
from .letta_domain_california import california_domain

logger = get_logger(__name__)


@dataclass
class LettaServerConfig:
    """Configuration for Letta server deployment."""
    
    # Server mode: subprocess, docker, or external
    mode: str = "subprocess"
    
    # Connection settings
    host: str = "localhost"
    port: int = 8283
    base_url: Optional[str] = None  # Override for external servers
    
    # Startup behavior
    auto_start: bool = True
    startup_timeout: int = 60
    startup_retries: int = 3
    
    # Health monitoring
    health_check_interval: int = 30
    health_check_timeout: int = 5
    max_health_failures: int = 3
    
    # Resource limits
    max_memory_mb: Optional[int] = 2048
    max_connections: int = 100
    
    # Docker-specific settings
    docker_image: str = "letta/letta:latest"
    docker_volumes: Dict[str, str] = field(default_factory=lambda: {
        str(Path.home() / ".letta"): "/root/.letta"
    })
    
    # Subprocess-specific settings
    subprocess_env: Dict[str, str] = field(default_factory=lambda: {
        "PYTHONUNBUFFERED": "1"
    })
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Set base_url if not provided
        if not self.base_url:
            self.base_url = f"http://{self.host}:{self.port}"
        
        # Validate mode
        valid_modes = ["subprocess", "docker", "external"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}")
        
        # Convert paths to strings in docker_volumes
        self.docker_volumes = {
            str(k): str(v) for k, v in self.docker_volumes.items()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LettaServerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_file(cls, path: Path) -> 'LettaServerConfig':
        """Load configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return cls()
    
    def save_to_file(self, path: Path) -> bool:
        """Save configuration to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
            return False


@dataclass
class LettaClientConfig:
    """Configuration for Letta client connections."""
    
    # Connection settings
    base_url: str = "http://localhost:8283"
    timeout: int = 30
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # Connection pooling
    pool_size: int = 10
    pool_maxsize: int = 20
    
    # Request settings
    verify_ssl: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Fallback behavior
    fallback_enabled: bool = True
    fallback_timeout: int = 5
    
    @classmethod
    def from_server_config(cls, server_config: LettaServerConfig) -> 'LettaClientConfig':
        """Create client config from server config."""
        return cls(
            base_url=server_config.base_url,
            timeout=server_config.health_check_timeout * 2,
            max_retries=server_config.startup_retries
        )


@dataclass
class LettaAgentConfig:
    """Configuration for Letta agents."""
    
    # Agent identification
    agent_id: Optional[str] = None
    name: Optional[str] = None
    
    # LLM configuration
    llm_provider: str = "ollama"
    llm_model: str = "gpt-oss:20b"
    llm_endpoint: str = "http://localhost:11434"
    llm_api_key: Optional[str] = None
    context_window: int = 8192
    max_tokens: int = 2000
    temperature: float = 0.7
    
    # Embedding configuration
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_endpoint: str = "http://localhost:11434"
    embedding_api_key: Optional[str] = None
    embedding_dim: int = 768
    
    # Memory configuration
    memory_blocks: List[Dict[str, str]] = field(default_factory=list)
    archival_memory_limit: int = 10000
    recall_memory_limit: int = 100
    
    # System prompt
    system_prompt: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class LettaConfigManager:
    """Manages all Letta-related configuration."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files (default: ~/.letta-claim)
        """
        self.config_dir = config_dir or (Path.home() / ".letta-claim")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration file paths
        self.server_config_path = self.config_dir / "letta-server.json"
        self.client_config_path = self.config_dir / "letta-client.json"
        
        # Load configurations
        self.server_config = self._load_server_config()
        self.client_config = self._load_client_config()
        
        logger.debug(
            "LettaConfigManager initialized",
            config_dir=str(self.config_dir),
            server_mode=self.server_config.mode
        )
    
    def _load_server_config(self) -> LettaServerConfig:
        """Load or create server configuration."""
        # Check environment variables first
        env_config = self._get_env_config()
        
        if self.server_config_path.exists():
            config = LettaServerConfig.from_file(self.server_config_path)
            # Override with environment variables
            for key, value in env_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return config
        else:
            # Create default config with env overrides
            config = LettaServerConfig(**env_config)
            config.save_to_file(self.server_config_path)
            return config
    
    def _load_client_config(self) -> LettaClientConfig:
        """Load or create client configuration."""
        if self.client_config_path.exists():
            try:
                with open(self.client_config_path, 'r') as f:
                    config_dict = json.load(f)
                return LettaClientConfig(**config_dict)
            except Exception as e:
                logger.warning(f"Failed to load client config: {e}")
        
        # Create from server config
        config = LettaClientConfig.from_server_config(self.server_config)
        self.save_client_config(config)
        return config
    
    def _get_env_config(self) -> Dict[str, Any]:
        """Get configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config fields
        env_mappings = {
            "LETTA_SERVER_MODE": ("mode", str),
            "LETTA_SERVER_HOST": ("host", str),
            "LETTA_SERVER_PORT": ("port", int),
            "LETTA_AUTO_START": ("auto_start", lambda x: x.lower() == "true"),
            "LETTA_STARTUP_TIMEOUT": ("startup_timeout", int),
            "LETTA_HEALTH_CHECK_INTERVAL": ("health_check_interval", int),
            "LETTA_LOG_LEVEL": ("log_level", str),
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    env_config[config_key] = converter(value)
                    logger.debug(f"Using {env_var}={value} for {config_key}")
                except Exception as e:
                    logger.warning(f"Failed to parse {env_var}={value}: {e}")
        
        return env_config
    
    def save_server_config(self, config: Optional[LettaServerConfig] = None) -> bool:
        """Save server configuration."""
        config = config or self.server_config
        return config.save_to_file(self.server_config_path)
    
    def save_client_config(self, config: Optional[LettaClientConfig] = None) -> bool:
        """Save client configuration."""
        config = config or self.client_config
        try:
            with open(self.client_config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save client config: {e}")
            return False
    
    def get_agent_config(
        self,
        matter_name: str,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        use_california_domain: bool = True
    ) -> LettaAgentConfig:
        """
        Get agent configuration for a specific matter.
        
        Args:
            matter_name: Name of the matter
            llm_provider: Override LLM provider
            llm_model: Override LLM model
            use_california_domain: Use California public works specialization
            
        Returns:
            Agent configuration
        """
        # Use California domain configuration if enabled
        if use_california_domain:
            system_prompt = california_domain.get_system_prompt(matter_name)
            memory_blocks = california_domain.get_memory_blocks(matter_name)
        else:
            # Fallback to generic construction claims
            system_prompt = f"""You are a construction claims analyst assistant for the matter: {matter_name}.

Your role is to:
- Analyze construction documents, contracts, and claims
- Track entities, events, issues, and facts
- Maintain memory of important information
- Provide insights about causation, responsibility, and damages
- Remember key dates, parties, and technical details

You have access to the search_documents tool to search case documents when needed.
Always cite sources using the format [DocName.pdf p.X] when referencing documents.
Learn from each interaction and remember important facts about the case.
You have persistent memory and learn from each conversation to provide better context-aware assistance."""
            memory_blocks = [
                {
                    "label": "human",
                    "value": f"Construction attorney working on {matter_name}"
                },
                {
                    "label": "persona",
                    "value": "Expert construction claims analyst with legal knowledge"
                }
            ]
        
        config = LettaAgentConfig(
            name=f"claim-assistant-{matter_name}",
            llm_provider=llm_provider or "ollama",
            llm_model=llm_model or "gpt-oss:20b",
            system_prompt=system_prompt,
            memory_blocks=memory_blocks,
            metadata={
                "matter_name": matter_name,
                "domain": "california_public_works" if use_california_domain else "construction_claims"
            }
        )
        
        return config
    
    def update_server_mode(self, mode: str) -> bool:
        """
        Update server deployment mode.
        
        Args:
            mode: New server mode (subprocess, docker, external)
            
        Returns:
            True if update successful
        """
        if mode not in ["subprocess", "docker", "external"]:
            logger.error(f"Invalid server mode: {mode}")
            return False
        
        self.server_config.mode = mode
        return self.save_server_config()
    
    def update_server_connection(self, host: str, port: int) -> bool:
        """
        Update server connection settings.
        
        Args:
            host: Server host
            port: Server port
            
        Returns:
            True if update successful
        """
        self.server_config.host = host
        self.server_config.port = port
        self.server_config.base_url = f"http://{host}:{port}"
        
        # Update client config to match
        self.client_config.base_url = self.server_config.base_url
        
        return self.save_server_config() and self.save_client_config()


# Global configuration manager
config_manager = LettaConfigManager()