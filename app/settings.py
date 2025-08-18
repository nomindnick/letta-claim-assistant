"""
Configuration management for the Letta Construction Claim Assistant.

Handles global configuration loading from TOML files, per-matter configuration
persistence, environment variable integration, and validation with defaults.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import tomllib
import json
import base64
from dataclasses import dataclass
from cryptography.fernet import Fernet
import os


@dataclass
class GlobalConfig:
    """Global application configuration."""
    
    ui_framework: str = "nicegui"
    ui_native: bool = True
    
    llm_provider: str = "ollama"
    llm_model: str = "gpt-oss:20b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 900
    
    embeddings_provider: str = "ollama"
    embeddings_model: str = "nomic-embed-text"
    
    ocr_enabled: bool = True
    ocr_force_ocr: bool = False
    ocr_language: str = "eng"
    ocr_skip_text: bool = True
    
    data_root: Path = Path.home() / "LettaClaims"
    
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    
    letta_enable_filesystem: bool = True
    
    # Letta server configuration
    letta_server_mode: str = "subprocess"  # subprocess, docker, external
    letta_server_host: str = "localhost"
    letta_server_port: int = 8283
    letta_server_auto_start: bool = True
    letta_server_health_check_interval: int = 30
    letta_server_startup_timeout: int = 60


class Settings:
    """Settings management singleton."""
    
    _instance: Optional['Settings'] = None
    _global_config: GlobalConfig
    _config_path: Path
    _credentials_path: Path
    _encryption_key: bytes
    
    def __new__(cls) -> 'Settings':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize settings from configuration files."""
        self._config_path = Path.home() / ".letta-claim" / "config.toml"
        self._credentials_path = Path.home() / ".letta-claim" / "credentials.json"
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup_encryption()
        self._load_global_config()
    
    def _setup_encryption(self) -> None:
        """Set up encryption for secure credential storage."""
        key_file = Path.home() / ".letta-claim" / ".key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                self._encryption_key = f.read()
        else:
            # Generate new encryption key
            self._encryption_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self._encryption_key)
            # Secure the key file
            os.chmod(key_file, 0o600)
    
    def _load_global_config(self) -> None:
        """Load global configuration with defaults."""
        if self._config_path.exists():
            try:
                with open(self._config_path, "rb") as f:
                    config_data = tomllib.load(f)
                self._global_config = self._merge_config(config_data)
            except Exception as e:
                print(f"Warning: Failed to load config from {self._config_path}: {e}")
                self._global_config = GlobalConfig()
        else:
            self._global_config = GlobalConfig()
            self._save_global_config()
    
    def _merge_config(self, config_data: Dict[str, Any]) -> GlobalConfig:
        """Merge configuration data with defaults."""
        config = GlobalConfig()
        
        if "ui" in config_data:
            ui_config = config_data["ui"]
            config.ui_framework = ui_config.get("framework", config.ui_framework)
            config.ui_native = ui_config.get("native", config.ui_native)
        
        if "llm" in config_data:
            llm_config = config_data["llm"]
            config.llm_provider = llm_config.get("provider", config.llm_provider)
            config.llm_model = llm_config.get("model", config.llm_model)
            config.llm_temperature = llm_config.get("temperature", config.llm_temperature)
            config.llm_max_tokens = llm_config.get("max_tokens", config.llm_max_tokens)
        
        if "embeddings" in config_data:
            embed_config = config_data["embeddings"]
            config.embeddings_provider = embed_config.get("provider", config.embeddings_provider)
            config.embeddings_model = embed_config.get("model", config.embeddings_model)
        
        if "ocr" in config_data:
            ocr_config = config_data["ocr"]
            config.ocr_enabled = ocr_config.get("enabled", config.ocr_enabled)
            config.ocr_force_ocr = ocr_config.get("force_ocr", config.ocr_force_ocr)
            config.ocr_language = ocr_config.get("language", config.ocr_language)
            config.ocr_skip_text = ocr_config.get("skip_text", config.ocr_skip_text)
        
        if "paths" in config_data:
            paths_config = config_data["paths"]
            if "root" in paths_config:
                config.data_root = Path(paths_config["root"]).expanduser()
        
        if "gemini" in config_data:
            gemini_config = config_data["gemini"]
            config.gemini_api_key = gemini_config.get("api_key", config.gemini_api_key)
            config.gemini_model = gemini_config.get("model", config.gemini_model)
        
        if "letta" in config_data:
            letta_config = config_data["letta"]
            config.letta_enable_filesystem = letta_config.get("enable_filesystem", config.letta_enable_filesystem)
            
            # Letta server configuration
            if "server" in letta_config:
                server_config = letta_config["server"]
                config.letta_server_mode = server_config.get("mode", config.letta_server_mode)
                config.letta_server_host = server_config.get("host", config.letta_server_host)
                config.letta_server_port = server_config.get("port", config.letta_server_port)
                config.letta_server_auto_start = server_config.get("auto_start", config.letta_server_auto_start)
                config.letta_server_health_check_interval = server_config.get("health_check_interval", config.letta_server_health_check_interval)
                config.letta_server_startup_timeout = server_config.get("startup_timeout", config.letta_server_startup_timeout)
        
        return config
    
    def _save_global_config(self) -> None:
        """Save current global configuration to TOML file."""
        config_toml = f"""[ui]
framework = "{self._global_config.ui_framework}"
native = {str(self._global_config.ui_native).lower()}

[llm]
provider = "{self._global_config.llm_provider}"
model = "{self._global_config.llm_model}"
temperature = {self._global_config.llm_temperature}
max_tokens = {self._global_config.llm_max_tokens}

[embeddings]
provider = "{self._global_config.embeddings_provider}"
model = "{self._global_config.embeddings_model}"

[ocr]
enabled = {str(self._global_config.ocr_enabled).lower()}
force_ocr = {str(self._global_config.ocr_force_ocr).lower()}
language = "{self._global_config.ocr_language}"
skip_text = {str(self._global_config.ocr_skip_text).lower()}

[paths]
root = "{self._global_config.data_root}"

[gemini]
api_key = ""
model = "{self._global_config.gemini_model}"

[letta]
enable_filesystem = {str(self._global_config.letta_enable_filesystem).lower()}

[letta.server]
mode = "{self._global_config.letta_server_mode}"
host = "{self._global_config.letta_server_host}"
port = {self._global_config.letta_server_port}
auto_start = {str(self._global_config.letta_server_auto_start).lower()}
health_check_interval = {self._global_config.letta_server_health_check_interval}
startup_timeout = {self._global_config.letta_server_startup_timeout}
"""
        
        with open(self._config_path, "w") as f:
            f.write(config_toml)
    
    @property
    def global_config(self) -> GlobalConfig:
        """Get global configuration."""
        return self._global_config
    
    def update_global_config(self, **kwargs) -> None:
        """Update global configuration values."""
        for key, value in kwargs.items():
            if hasattr(self._global_config, key):
                setattr(self._global_config, key, value)
        self._save_global_config()
    
    def load_matter_config(self, matter_path: Path) -> Dict[str, Any]:
        """Load matter-specific configuration."""
        config_path = matter_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load matter config from {config_path}: {e}")
        return {}
    
    def save_matter_config(self, matter_path: Path, config: Dict[str, Any]) -> None:
        """Save matter-specific configuration."""
        config_path = matter_path / "config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            print(f"Error: Failed to save matter config to {config_path}: {e}")
    
    def store_credential(self, provider: str, credential_type: str, credential_value: str) -> bool:
        """Store encrypted credential for a provider."""
        try:
            # Load existing credentials
            credentials = self._load_credentials()
            
            # Encrypt the credential
            fernet = Fernet(self._encryption_key)
            encrypted_value = fernet.encrypt(credential_value.encode())
            
            # Store the encrypted credential
            if provider not in credentials:
                credentials[provider] = {}
            
            credentials[provider][credential_type] = base64.b64encode(encrypted_value).decode()
            
            # Save credentials
            self._save_credentials(credentials)
            return True
            
        except Exception as e:
            print(f"Error: Failed to store credential: {e}")
            return False
    
    def get_credential(self, provider: str, credential_type: str) -> Optional[str]:
        """Get decrypted credential for a provider."""
        try:
            credentials = self._load_credentials()
            
            if provider not in credentials or credential_type not in credentials[provider]:
                return None
            
            # Decrypt the credential
            fernet = Fernet(self._encryption_key)
            encrypted_value = base64.b64decode(credentials[provider][credential_type])
            decrypted_value = fernet.decrypt(encrypted_value).decode()
            
            return decrypted_value
            
        except Exception as e:
            print(f"Error: Failed to get credential: {e}")
            return None
    
    def delete_credential(self, provider: str, credential_type: str) -> bool:
        """Delete a stored credential."""
        try:
            credentials = self._load_credentials()
            
            if provider in credentials and credential_type in credentials[provider]:
                del credentials[provider][credential_type]
                
                # Remove provider entry if empty
                if not credentials[provider]:
                    del credentials[provider]
                
                self._save_credentials(credentials)
            
            return True
            
        except Exception as e:
            print(f"Error: Failed to delete credential: {e}")
            return False
    
    def list_stored_credentials(self) -> Dict[str, List[str]]:
        """List all stored credentials by provider."""
        try:
            credentials = self._load_credentials()
            return {
                provider: list(creds.keys()) 
                for provider, creds in credentials.items()
            }
        except Exception:
            return {}
    
    def _load_credentials(self) -> Dict[str, Dict[str, str]]:
        """Load encrypted credentials from disk."""
        if self._credentials_path.exists():
            try:
                with open(self._credentials_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_credentials(self, credentials: Dict[str, Dict[str, str]]) -> None:
        """Save encrypted credentials to disk."""
        with open(self._credentials_path, 'w') as f:
            json.dump(credentials, f, indent=2)
        # Secure the credentials file
        os.chmod(self._credentials_path, 0o600)


# Global settings instance
settings = Settings()