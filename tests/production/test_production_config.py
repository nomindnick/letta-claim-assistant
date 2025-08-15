"""
Tests for production configuration validation.

Verifies that the production configuration system correctly validates
settings, environment variables, and security configurations.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.production_config import (
    ProductionConfigValidator,
    validate_production_config,
    get_environment_mode,
    is_production_environment,
    EnvironmentMode,
    ProductionRequirements
)
from app.startup_checks import CheckStatus


class TestProductionConfigValidator:
    """Test the production configuration validator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = ProductionConfigValidator()
    
    def test_environment_detection_production(self):
        """Test production environment detection."""
        with patch.dict(os.environ, {"LETTA_ENV": "production"}):
            validator = ProductionConfigValidator()
            assert validator.environment == EnvironmentMode.PRODUCTION
    
    def test_environment_detection_development(self):
        """Test development environment detection."""
        with patch.dict(os.environ, {"DEBUG": "1"}):
            validator = ProductionConfigValidator()
            assert validator.environment == EnvironmentMode.DEVELOPMENT
    
    def test_environment_detection_default(self):
        """Test default environment detection."""
        # Clear relevant environment variables
        env_vars = ["LETTA_ENV", "DEBUG"]
        with patch.dict(os.environ, {}, clear=True):
            # Mock os.path.exists to return False for .git
            with patch("os.path.exists", return_value=False):
                validator = ProductionConfigValidator()
                assert validator.environment == EnvironmentMode.PRODUCTION
    
    def test_validate_all_success(self):
        """Test successful validation."""
        with patch.object(self.validator, '_validate_core_config'), \
             patch.object(self.validator, '_validate_paths'), \
             patch.object(self.validator, '_validate_provider_config'), \
             patch.object(self.validator, '_validate_logging_config'):
            
            success, results = self.validator.validate_all()
            assert success
            assert isinstance(results, list)
    
    def test_validate_production_config_debug_mode(self):
        """Test production validation with debug mode enabled."""
        with patch.dict(os.environ, {"DEBUG": "1"}):
            validator = ProductionConfigValidator()
            validator.environment = EnvironmentMode.PRODUCTION
            validator._validate_production_config()
            
            # Should have warning about debug mode
            warnings = [r for r in validator.validation_results if r.status == CheckStatus.WARNING]
            assert any("debug mode" in r.message.lower() for r in warnings)
    
    def test_validate_security_settings(self):
        """Test security settings validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create credentials directory with secure permissions
            creds_dir = Path(temp_dir) / ".letta-claim"
            creds_dir.mkdir(mode=0o700)
            
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                validator = ProductionConfigValidator()
                validator._validate_security_settings()
                
                # Should pass security check
                security_results = [r for r in validator.validation_results if "security" in r.name.lower()]
                assert any(r.status == CheckStatus.PASS for r in security_results)
    
    def test_validate_paths_absolute(self):
        """Test path validation with absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config = MagicMock()
            mock_config.data_root = Path(temp_dir)
            
            validator = ProductionConfigValidator()
            validator.config = mock_config
            validator._validate_paths()
            
            # Should pass for absolute path
            path_results = [r for r in validator.validation_results if "path" in r.name.lower()]
            assert any(r.status == CheckStatus.PASS for r in path_results)
    
    def test_validate_paths_relative(self):
        """Test path validation with relative paths."""
        mock_config = MagicMock()
        mock_config.data_root = Path("relative/path")
        
        validator = ProductionConfigValidator()
        validator.config = mock_config
        validator._validate_paths()
        
        # Should warn for relative path
        path_results = [r for r in validator.validation_results if "path" in r.name.lower()]
        assert any(r.status == CheckStatus.WARNING for r in path_results)
    
    def test_validate_provider_config_ollama(self):
        """Test provider configuration validation for Ollama."""
        mock_config = MagicMock()
        mock_config.llm_provider = "ollama"
        mock_config.llm_model = "gpt-oss:20b"
        
        validator = ProductionConfigValidator()
        validator.config = mock_config
        validator._validate_provider_config()
        
        # Should pass for valid Ollama config
        provider_results = [r for r in validator.validation_results if "ollama" in r.name.lower()]
        assert any(r.status == CheckStatus.PASS for r in provider_results)
    
    def test_validate_provider_config_gemini_no_key(self):
        """Test provider configuration validation for Gemini without API key."""
        mock_config = MagicMock()
        mock_config.llm_provider = "gemini"
        
        with patch("app.settings.settings.get_credential", return_value=None):
            validator = ProductionConfigValidator()
            validator.config = mock_config
            validator._validate_provider_config()
            
            # Should fail without API key
            provider_results = [r for r in validator.validation_results if "gemini" in r.name.lower()]
            assert any(r.status == CheckStatus.FAIL for r in provider_results)
    
    def test_validate_provider_config_gemini_with_key(self):
        """Test provider configuration validation for Gemini with API key."""
        mock_config = MagicMock()
        mock_config.llm_provider = "gemini"
        
        with patch("app.settings.settings.get_credential", return_value="test_key"):
            validator = ProductionConfigValidator()
            validator.config = mock_config
            validator._validate_provider_config()
            
            # Should pass with API key
            provider_results = [r for r in validator.validation_results if "gemini" in r.name.lower()]
            assert any(r.status == CheckStatus.PASS for r in provider_results)
    
    def test_performance_settings_large_model(self):
        """Test performance validation with large model."""
        mock_config = MagicMock()
        mock_config.llm_model = "gpt-oss:20b"
        
        validator = ProductionConfigValidator()
        validator.config = mock_config
        validator._validate_performance_settings()
        
        # Should warn about large model
        perf_results = [r for r in validator.validation_results if "performance" in r.name.lower()]
        assert any(r.status == CheckStatus.WARNING for r in perf_results)
    
    def test_performance_settings_light_model(self):
        """Test performance validation with light model."""
        mock_config = MagicMock()
        mock_config.llm_model = "llama3.1"
        
        validator = ProductionConfigValidator()
        validator.config = mock_config
        validator._validate_performance_settings()
        
        # Should pass for light model
        perf_results = [r for r in validator.validation_results if "performance" in r.name.lower()]
        assert any(r.status == CheckStatus.PASS for r in perf_results)


class TestProductionConfigFunctions:
    """Test standalone production configuration functions."""
    
    def test_validate_production_config_function(self):
        """Test the standalone validation function."""
        with patch.object(ProductionConfigValidator, 'validate_all', return_value=(True, [])):
            success, results = validate_production_config()
            assert success
            assert isinstance(results, list)
    
    def test_get_environment_mode(self):
        """Test environment mode detection function."""
        with patch('app.production_config.ProductionConfigValidator') as mock_validator:
            mock_instance = mock_validator.return_value
            mock_instance.environment = EnvironmentMode.PRODUCTION
            mode = get_environment_mode()
            assert isinstance(mode, EnvironmentMode)
    
    def test_is_production_environment_true(self):
        """Test production environment check when true."""
        with patch('app.production_config.get_environment_mode', return_value=EnvironmentMode.PRODUCTION):
            assert is_production_environment() == True
    
    def test_is_production_environment_false(self):
        """Test production environment check when false."""
        with patch('app.production_config.get_environment_mode', return_value=EnvironmentMode.DEVELOPMENT):
            assert is_production_environment() == False


class TestProductionRequirements:
    """Test production requirements specification."""
    
    def test_production_requirements_defaults(self):
        """Test default production requirements."""
        req = ProductionRequirements()
        
        assert req.min_python_version == (3, 9, 0)
        assert req.min_disk_space_gb == 5.0
        assert req.min_memory_gb == 2.0
        assert isinstance(req.required_system_packages, list)
        assert isinstance(req.required_python_packages, list)
        assert len(req.required_system_packages) > 0
        assert len(req.required_python_packages) > 0
    
    def test_production_requirements_system_packages(self):
        """Test required system packages."""
        req = ProductionRequirements()
        
        expected_packages = ["ocrmypdf", "tesseract", "pdfinfo"]
        for package in expected_packages:
            assert package in req.required_system_packages
    
    def test_production_requirements_python_packages(self):
        """Test required Python packages."""
        req = ProductionRequirements()
        
        expected_packages = ["nicegui", "chromadb", "ollama", "pydantic"]
        for package in expected_packages:
            assert package in req.required_python_packages


@pytest.mark.integration
class TestProductionConfigIntegration:
    """Integration tests for production configuration."""
    
    def test_full_validation_cycle(self):
        """Test complete validation cycle."""
        # This test requires actual system dependencies
        success, results = validate_production_config()
        
        assert isinstance(success, bool)
        assert isinstance(results, list)
        
        for result in results:
            assert hasattr(result, 'name')
            assert hasattr(result, 'status')
            assert hasattr(result, 'message')
            assert result.status in [CheckStatus.PASS, CheckStatus.WARNING, CheckStatus.FAIL]
    
    def test_environment_detection_real(self):
        """Test environment detection with real environment."""
        mode = get_environment_mode()
        assert isinstance(mode, EnvironmentMode)
        assert mode in [EnvironmentMode.PRODUCTION, EnvironmentMode.DEVELOPMENT, EnvironmentMode.TESTING]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])