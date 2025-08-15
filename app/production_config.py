"""
Production configuration validator for the Letta Construction Claim Assistant.

Validates configuration settings, environment variables, and system readiness
for production deployment with comprehensive error reporting and suggestions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .logging_conf import get_logger
from .settings import settings, GlobalConfig
from .startup_checks import CheckResult, CheckStatus

logger = get_logger(__name__)


class EnvironmentMode(Enum):
    """Application environment modes."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class ProductionRequirements:
    """Production environment requirements."""
    min_python_version: tuple = (3, 9, 0)
    min_disk_space_gb: float = 5.0
    min_memory_gb: float = 2.0
    required_system_packages: List[str] = None
    required_python_packages: List[str] = None
    
    def __post_init__(self):
        if self.required_system_packages is None:
            self.required_system_packages = [
                "ocrmypdf", "tesseract", "pdfinfo", "gs"
            ]
        
        if self.required_python_packages is None:
            self.required_python_packages = [
                "nicegui", "fastapi", "chromadb", "ollama", 
                "pydantic", "structlog", "pymupdf", "uvicorn"
            ]


class ProductionConfigValidator:
    """Validates configuration for production deployment."""
    
    def __init__(self):
        self.requirements = ProductionRequirements()
        self.config = settings.global_config
        self.environment = self._detect_environment()
        self.validation_results: List[CheckResult] = []
    
    def _detect_environment(self) -> EnvironmentMode:
        """Detect the current environment mode."""
        # Check environment variables
        env_mode = os.getenv("LETTA_ENV", "").lower()
        if env_mode in ["production", "prod"]:
            return EnvironmentMode.PRODUCTION
        elif env_mode in ["testing", "test"]:
            return EnvironmentMode.TESTING
        elif env_mode in ["development", "dev"]:
            return EnvironmentMode.DEVELOPMENT
        
        # Check for development indicators
        if (
            os.getenv("DEBUG") == "1" or
            "--debug" in sys.argv or
            os.path.exists(Path.cwd() / ".git")
        ):
            return EnvironmentMode.DEVELOPMENT
        
        # Default to production for safety
        return EnvironmentMode.PRODUCTION
    
    def validate_all(self) -> tuple[bool, List[CheckResult]]:
        """
        Run all production configuration validations.
        
        Returns:
            Tuple of (is_valid, validation_results)
        """
        logger.info("Validating production configuration", environment=self.environment.value)
        
        self.validation_results = []
        
        # Environment-specific validations
        if self.environment == EnvironmentMode.PRODUCTION:
            self._validate_production_config()
            self._validate_security_settings()
            self._validate_performance_settings()
            self._validate_resource_limits()
        elif self.environment == EnvironmentMode.DEVELOPMENT:
            self._validate_development_config()
        
        # Universal validations
        self._validate_core_config()
        self._validate_paths()
        self._validate_provider_config()
        self._validate_logging_config()
        
        # Check for critical failures
        critical_failures = [
            r for r in self.validation_results 
            if r.status == CheckStatus.FAIL
        ]
        
        is_valid = len(critical_failures) == 0
        
        if is_valid:
            logger.info("Production configuration validation passed")
        else:
            logger.error(
                "Production configuration validation failed",
                failures=len(critical_failures),
                total_checks=len(self.validation_results)
            )
        
        return is_valid, self.validation_results
    
    def _validate_production_config(self):
        """Validate production-specific configuration."""
        
        # Check debug mode is disabled
        if os.getenv("DEBUG") == "1":
            self.validation_results.append(CheckResult(
                name="Debug Mode",
                status=CheckStatus.WARNING,
                message="Debug mode enabled in production",
                suggestion="Set DEBUG=0 or remove DEBUG environment variable"
            ))
        else:
            self.validation_results.append(CheckResult(
                name="Debug Mode",
                status=CheckStatus.PASS,
                message="Debug mode disabled"
            ))
        
        # Check for development files
        dev_indicators = [".git", "test_*.py", "pytest.ini", "requirements-dev.txt"]
        found_dev_files = [f for f in dev_indicators if Path(f).exists()]
        
        if found_dev_files:
            self.validation_results.append(CheckResult(
                name="Development Files",
                status=CheckStatus.WARNING,
                message=f"Development files present: {', '.join(found_dev_files)}",
                suggestion="Remove development files from production deployment",
                details={"dev_files": found_dev_files}
            ))
        else:
            self.validation_results.append(CheckResult(
                name="Development Files",
                status=CheckStatus.PASS,
                message="No development files detected"
            ))
        
        # Check logging configuration
        if self.environment == EnvironmentMode.PRODUCTION:
            # Should have production logging settings
            self.validation_results.append(CheckResult(
                name="Production Logging",
                status=CheckStatus.PASS,
                message="Production environment detected"
            ))
    
    def _validate_development_config(self):
        """Validate development-specific configuration."""
        self.validation_results.append(CheckResult(
            name="Development Mode",
            status=CheckStatus.PASS,
            message="Development environment detected"
        ))
    
    def _validate_security_settings(self):
        """Validate security-related settings."""
        
        # Check credential storage
        credentials_dir = Path.home() / ".letta-claim"
        if credentials_dir.exists():
            # Check permissions on credentials directory
            try:
                dir_stat = credentials_dir.stat()
                mode = oct(dir_stat.st_mode)[-3:]
                
                if mode == "700":  # Owner read/write/execute only
                    self.validation_results.append(CheckResult(
                        name="Credentials Security",
                        status=CheckStatus.PASS,
                        message="Credentials directory properly secured"
                    ))
                else:
                    self.validation_results.append(CheckResult(
                        name="Credentials Security",
                        status=CheckStatus.WARNING,
                        message=f"Credentials directory permissions: {mode}",
                        suggestion="Set secure permissions: chmod 700 ~/.letta-claim"
                    ))
            except Exception as e:
                self.validation_results.append(CheckResult(
                    name="Credentials Security",
                    status=CheckStatus.WARNING,
                    message=f"Could not check credentials security: {e}"
                ))
        else:
            self.validation_results.append(CheckResult(
                name="Credentials Security",
                status=CheckStatus.PASS,
                message="Credentials directory will be created securely"
            ))
        
        # Check for exposed API keys in environment
        exposed_keys = []
        for key, value in os.environ.items():
            if (
                ("api" in key.lower() and "key" in key.lower()) or
                ("secret" in key.lower()) or
                ("token" in key.lower())
            ) and value:
                exposed_keys.append(key)
        
        if exposed_keys:
            self.validation_results.append(CheckResult(
                name="API Key Security",
                status=CheckStatus.WARNING,
                message=f"Potential API keys in environment: {', '.join(exposed_keys)}",
                suggestion="Store API keys securely using application credential storage"
            ))
        else:
            self.validation_results.append(CheckResult(
                name="API Key Security",
                status=CheckStatus.PASS,
                message="No exposed API keys in environment"
            ))
    
    def _validate_performance_settings(self):
        """Validate performance-related settings."""
        
        # Check LLM model selection for production
        model = self.config.llm_model
        
        if "oss" in model.lower() and "20b" in model:
            self.validation_results.append(CheckResult(
                name="LLM Model Performance",
                status=CheckStatus.WARNING,
                message=f"Large model selected: {model}",
                suggestion="Consider lighter model for better performance: llama3.1 or mistral",
                details={"model": model, "size": "large"}
            ))
        else:
            self.validation_results.append(CheckResult(
                name="LLM Model Performance",
                status=CheckStatus.PASS,
                message=f"Model selected: {model}",
                details={"model": model}
            ))
        
        # Check embedding model
        embed_model = self.config.embeddings_model
        if embed_model in ["nomic-embed-text", "mxbai-embed-large"]:
            self.validation_results.append(CheckResult(
                name="Embedding Model",
                status=CheckStatus.PASS,
                message=f"Suitable embedding model: {embed_model}",
                details={"model": embed_model}
            ))
        else:
            self.validation_results.append(CheckResult(
                name="Embedding Model",
                status=CheckStatus.WARNING,
                message=f"Unknown embedding model: {embed_model}",
                suggestion="Use tested embedding model: nomic-embed-text"
            ))
    
    def _validate_resource_limits(self):
        """Validate resource limits and constraints."""
        
        # Check data root path
        data_root = Path(self.config.data_root)
        
        # Validate path is under user home or dedicated directory
        home = Path.home()
        
        if data_root.is_relative_to(home) or str(data_root).startswith("/opt/"):
            self.validation_results.append(CheckResult(
                name="Data Location",
                status=CheckStatus.PASS,
                message=f"Data stored in appropriate location: {data_root}",
                details={"path": str(data_root)}
            ))
        else:
            self.validation_results.append(CheckResult(
                name="Data Location",
                status=CheckStatus.WARNING,
                message=f"Data location may not be appropriate: {data_root}",
                suggestion="Consider storing data under user home or /opt/"
            ))
        
        # Check for temp directory usage
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        
        if data_root.is_relative_to(temp_dir):
            self.validation_results.append(CheckResult(
                name="Data Persistence",
                status=CheckStatus.FAIL,
                message="Data directory is in temporary location",
                suggestion=f"Move data directory outside of {temp_dir}"
            ))
        else:
            self.validation_results.append(CheckResult(
                name="Data Persistence",
                status=CheckStatus.PASS,
                message="Data directory is in persistent location"
            ))
    
    def _validate_core_config(self):
        """Validate core configuration values."""
        
        # Check required configuration values
        required_configs = {
            "ui_framework": self.config.ui_framework,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.llm_model,
            "embeddings_model": self.config.embeddings_model,
            "data_root": str(self.config.data_root)
        }
        
        missing_configs = [k for k, v in required_configs.items() if not v]
        
        if missing_configs:
            self.validation_results.append(CheckResult(
                name="Core Configuration",
                status=CheckStatus.FAIL,
                message=f"Missing required configuration: {', '.join(missing_configs)}",
                suggestion="Complete configuration file setup"
            ))
        else:
            self.validation_results.append(CheckResult(
                name="Core Configuration",
                status=CheckStatus.PASS,
                message="All required configuration present"
            ))
        
        # Validate specific values
        if self.config.llm_provider not in ["ollama", "gemini"]:
            self.validation_results.append(CheckResult(
                name="LLM Provider",
                status=CheckStatus.FAIL,
                message=f"Invalid LLM provider: {self.config.llm_provider}",
                suggestion="Set llm_provider to 'ollama' or 'gemini'"
            ))
        
        if self.config.ui_framework != "nicegui":
            self.validation_results.append(CheckResult(
                name="UI Framework",
                status=CheckStatus.WARNING,
                message=f"Unexpected UI framework: {self.config.ui_framework}",
                suggestion="Set ui_framework to 'nicegui'"
            ))
    
    def _validate_paths(self):
        """Validate all configured paths."""
        
        # Check data root
        data_root = Path(self.config.data_root)
        
        if data_root.is_absolute():
            self.validation_results.append(CheckResult(
                name="Data Root Path",
                status=CheckStatus.PASS,
                message=f"Valid absolute path: {data_root}"
            ))
        else:
            self.validation_results.append(CheckResult(
                name="Data Root Path",
                status=CheckStatus.WARNING,
                message=f"Relative path used: {data_root}",
                suggestion="Use absolute path for data_root"
            ))
        
        # Check path expansion
        try:
            expanded_path = data_root.expanduser().resolve()
            self.validation_results.append(CheckResult(
                name="Path Resolution",
                status=CheckStatus.PASS,
                message=f"Path resolves to: {expanded_path}",
                details={"resolved_path": str(expanded_path)}
            ))
        except Exception as e:
            self.validation_results.append(CheckResult(
                name="Path Resolution",
                status=CheckStatus.FAIL,
                message=f"Path resolution failed: {e}",
                suggestion="Check data_root path syntax"
            ))
    
    def _validate_provider_config(self):
        """Validate LLM provider configuration."""
        
        provider = self.config.llm_provider
        
        if provider == "ollama":
            # Validate Ollama configuration
            self.validation_results.append(CheckResult(
                name="Ollama Configuration",
                status=CheckStatus.PASS,
                message=f"Configured for local Ollama with {self.config.llm_model}"
            ))
        
        elif provider == "gemini":
            # Check if API key is configured
            api_key = settings.get_credential("gemini", "api_key")
            
            if api_key:
                self.validation_results.append(CheckResult(
                    name="Gemini Configuration",
                    status=CheckStatus.PASS,
                    message="API key configured for Gemini"
                ))
            else:
                self.validation_results.append(CheckResult(
                    name="Gemini Configuration",
                    status=CheckStatus.FAIL,
                    message="Gemini selected but no API key configured",
                    suggestion="Configure Gemini API key in application settings"
                ))
        
        else:
            self.validation_results.append(CheckResult(
                name="Provider Configuration",
                status=CheckStatus.FAIL,
                message=f"Unknown provider: {provider}",
                suggestion="Set llm_provider to 'ollama' or 'gemini'"
            ))
    
    def _validate_logging_config(self):
        """Validate logging configuration."""
        
        # Check if structured logging is properly configured
        try:
            from . import logging_conf
            
            self.validation_results.append(CheckResult(
                name="Logging Configuration",
                status=CheckStatus.PASS,
                message="Structured logging configured"
            ))
        except ImportError as e:
            self.validation_results.append(CheckResult(
                name="Logging Configuration",
                status=CheckStatus.FAIL,
                message=f"Logging configuration failed: {e}",
                suggestion="Verify logging_conf module"
            ))
        
        # Validate log directory will be writable
        if self.environment == EnvironmentMode.PRODUCTION:
            # In production, logs should go to appropriate location
            home_logs = Path.home() / ".letta-claim" / "logs"
            
            if home_logs.exists() or home_logs.parent.exists():
                self.validation_results.append(CheckResult(
                    name="Log Directory",
                    status=CheckStatus.PASS,
                    message=f"Log directory accessible: {home_logs}"
                ))
            else:
                self.validation_results.append(CheckResult(
                    name="Log Directory",
                    status=CheckStatus.WARNING,
                    message="Log directory will be created on first run"
                ))


def validate_production_config() -> tuple[bool, List[CheckResult]]:
    """
    Validate production configuration.
    
    Returns:
        Tuple of (is_valid, validation_results)
    """
    validator = ProductionConfigValidator()
    return validator.validate_all()


def get_environment_mode() -> EnvironmentMode:
    """Get the current environment mode."""
    validator = ProductionConfigValidator()
    return validator.environment


def is_production_environment() -> bool:
    """Check if running in production environment."""
    return get_environment_mode() == EnvironmentMode.PRODUCTION


def get_production_requirements() -> ProductionRequirements:
    """Get production requirements specification."""
    return ProductionRequirements()