"""
Tests for startup validation checks.

Verifies that the startup validation system correctly checks dependencies,
resources, and system readiness.
"""

import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from app.startup_checks import (
    StartupValidator,
    validate_startup,
    format_check_results,
    CheckStatus,
    CheckResult
)


class TestStartupValidator:
    """Test the startup validator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = StartupValidator()
    
    @pytest.mark.asyncio
    async def test_run_all_checks_success(self):
        """Test successful startup validation."""
        # Mock all check methods to pass
        with patch.object(self.validator, '_check_python_version'), \
             patch.object(self.validator, '_check_system_packages'), \
             patch.object(self.validator, '_check_python_packages'), \
             patch.object(self.validator, '_check_disk_space'), \
             patch.object(self.validator, '_check_memory'), \
             patch.object(self.validator, '_check_permissions'), \
             patch.object(self.validator, '_check_ollama_service'), \
             patch.object(self.validator, '_check_ollama_models'), \
             patch.object(self.validator, '_check_external_services'), \
             patch.object(self.validator, '_check_configuration'), \
             patch.object(self.validator, '_check_data_directories'):
            
            success, results = await self.validator.run_all_checks()
            assert isinstance(success, bool)
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_check_python_version_good(self):
        """Test Python version check with good version."""
        await self.validator._check_python_version()
        
        results = [r for r in self.validator.results if "python version" in r.name.lower()]
        assert len(results) > 0
        # Should pass with current Python version (assuming >= 3.9)
        assert any(r.status in [CheckStatus.PASS, CheckStatus.WARNING] for r in results)
    
    @pytest.mark.asyncio
    async def test_check_system_packages_missing(self):
        """Test system package check with missing packages."""
        with patch('subprocess.run') as mock_run:
            # Mock missing package
            mock_run.return_value.returncode = 1
            
            await self.validator._check_system_packages()
            
            results = [r for r in self.validator.results if "system package" in r.name.lower()]
            assert len(results) > 0
            # Should have failures for missing packages
            assert any(r.status == CheckStatus.FAIL for r in results)
    
    @pytest.mark.asyncio
    async def test_check_system_packages_available(self):
        """Test system package check with available packages."""
        with patch('subprocess.run') as mock_run:
            # Mock available package
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "/usr/bin/ocrmypdf"
            
            await self.validator._check_system_packages()
            
            results = [r for r in self.validator.results if "system package" in r.name.lower()]
            assert len(results) > 0
            # Should have passes for available packages
            assert any(r.status == CheckStatus.PASS for r in results)
    
    @pytest.mark.asyncio
    async def test_check_python_packages_missing(self):
        """Test Python package check with missing packages."""
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No module named 'test_package'")
            
            await self.validator._check_python_packages()
            
            results = [r for r in self.validator.results if "python package" in r.name.lower()]
            assert len(results) > 0
            # Should have failures for missing packages
            assert any(r.status == CheckStatus.FAIL for r in results)
    
    @pytest.mark.asyncio
    async def test_check_python_packages_available(self):
        """Test Python package check with available packages."""
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module
            
            await self.validator._check_python_packages()
            
            results = [r for r in self.validator.results if "python package" in r.name.lower()]
            assert len(results) > 0
            # Should have passes for available packages
            assert any(r.status == CheckStatus.PASS for r in results)
    
    @pytest.mark.asyncio
    async def test_check_disk_space_sufficient(self):
        """Test disk space check with sufficient space."""
        with patch('shutil.disk_usage') as mock_usage:
            # Mock 10GB free space
            mock_usage.return_value = (100 * 1024**3, 50 * 1024**3, 10 * 1024**3)
            
            await self.validator._check_disk_space()
            
            results = [r for r in self.validator.results if "disk space" in r.name.lower()]
            assert len(results) > 0
            assert any(r.status == CheckStatus.PASS for r in results)
    
    @pytest.mark.asyncio
    async def test_check_disk_space_insufficient(self):
        """Test disk space check with insufficient space."""
        with patch('shutil.disk_usage') as mock_usage:
            # Mock 0.5GB free space
            mock_usage.return_value = (100 * 1024**3, 99.5 * 1024**3, 0.5 * 1024**3)
            
            await self.validator._check_disk_space()
            
            results = [r for r in self.validator.results if "disk space" in r.name.lower()]
            assert len(results) > 0
            assert any(r.status == CheckStatus.FAIL for r in results)
    
    @pytest.mark.asyncio
    async def test_check_memory_sufficient(self):
        """Test memory check with sufficient memory."""
        try:
            import psutil
            
            with patch.object(psutil, 'virtual_memory') as mock_memory:
                mock_memory.return_value.available = 4 * 1024**3  # 4GB
                mock_memory.return_value.total = 8 * 1024**3      # 8GB
                mock_memory.return_value.percent = 50.0
                
                await self.validator._check_memory()
                
                results = [r for r in self.validator.results if "memory" in r.name.lower()]
                assert len(results) > 0
                assert any(r.status == CheckStatus.PASS for r in results)
                
        except ImportError:
            # Skip test if psutil not available
            pytest.skip("psutil not available")
    
    @pytest.mark.asyncio
    async def test_check_memory_insufficient(self):
        """Test memory check with insufficient memory."""
        try:
            import psutil
            
            with patch.object(psutil, 'virtual_memory') as mock_memory:
                mock_memory.return_value.available = 0.5 * 1024**3  # 0.5GB
                mock_memory.return_value.total = 8 * 1024**3        # 8GB
                mock_memory.return_value.percent = 95.0
                
                await self.validator._check_memory()
                
                results = [r for r in self.validator.results if "memory" in r.name.lower()]
                assert len(results) > 0
                assert any(r.status == CheckStatus.FAIL for r in results)
                
        except ImportError:
            # Skip test if psutil not available
            pytest.skip("psutil not available")
    
    @pytest.mark.asyncio
    async def test_check_permissions_success(self):
        """Test permissions check with success."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "test_data"
            
            mock_config = MagicMock()
            mock_config.data_root = data_root
            
            validator = StartupValidator()
            validator.config = mock_config
            
            await validator._check_permissions()
            
            results = [r for r in validator.results if "permission" in r.name.lower()]
            assert len(results) > 0
            assert any(r.status == CheckStatus.PASS for r in results)
    
    @pytest.mark.asyncio
    async def test_check_ollama_service_available(self):
        """Test Ollama service check when available."""
        with patch('importlib.import_module') as mock_import:
            mock_ollama = MagicMock()
            mock_ollama.list.return_value = {"models": [{"name": "test"}]}
            mock_import.return_value = mock_ollama
            
            await self.validator._check_ollama_service()
            
            results = [r for r in self.validator.results if "ollama service" in r.name.lower()]
            assert len(results) > 0
            assert any(r.status == CheckStatus.PASS for r in results)
    
    @pytest.mark.asyncio
    async def test_check_ollama_service_unavailable(self):
        """Test Ollama service check when unavailable."""
        with patch('importlib.import_module') as mock_import:
            mock_ollama = MagicMock()
            mock_ollama.list.side_effect = ConnectionError("Connection refused")
            mock_import.return_value = mock_ollama
            
            await self.validator._check_ollama_service()
            
            results = [r for r in self.validator.results if "ollama service" in r.name.lower()]
            assert len(results) > 0
            assert any(r.status == CheckStatus.FAIL for r in results)
    
    @pytest.mark.asyncio
    async def test_check_ollama_models_available(self):
        """Test Ollama models check when available."""
        with patch('importlib.import_module') as mock_import:
            mock_ollama = MagicMock()
            mock_ollama.list.return_value = {
                "models": [
                    {"name": "gpt-oss:20b"},
                    {"name": "nomic-embed-text"}
                ]
            }
            mock_import.return_value = mock_ollama
            
            await self.validator._check_ollama_models()
            
            results = [r for r in self.validator.results if "ollama model" in r.name.lower()]
            assert len(results) > 0
            assert any(r.status == CheckStatus.PASS for r in results)
    
    @pytest.mark.asyncio
    async def test_check_ollama_models_missing(self):
        """Test Ollama models check when missing."""
        with patch('importlib.import_module') as mock_import:
            mock_ollama = MagicMock()
            mock_ollama.list.return_value = {"models": []}
            mock_import.return_value = mock_ollama
            
            await self.validator._check_ollama_models()
            
            results = [r for r in self.validator.results if "ollama model" in r.name.lower()]
            assert len(results) > 0
            assert any(r.status == CheckStatus.FAIL for r in results)
    
    @pytest.mark.asyncio
    async def test_check_configuration_valid(self):
        """Test configuration check with valid config."""
        mock_config = MagicMock()
        mock_config.data_root = Path("/home/user/data").absolute()
        
        validator = StartupValidator()
        validator.config = mock_config
        
        await validator._check_configuration()
        
        results = [r for r in validator.results if "configuration" in r.name.lower()]
        assert len(results) > 0
        assert any(r.status == CheckStatus.PASS for r in results)


class TestStartupValidationFunctions:
    """Test standalone startup validation functions."""
    
    @pytest.mark.asyncio
    async def test_validate_startup_function(self):
        """Test the standalone validation function."""
        with patch.object(StartupValidator, 'run_all_checks', return_value=(True, [])):
            success, results = await validate_startup()
            assert isinstance(success, bool)
            assert isinstance(results, list)
    
    def test_format_check_results(self):
        """Test formatting of check results."""
        results = [
            CheckResult(
                name="Test Check 1",
                status=CheckStatus.PASS,
                message="Everything OK"
            ),
            CheckResult(
                name="Test Check 2",
                status=CheckStatus.FAIL,
                message="Something wrong",
                suggestion="Fix it"
            ),
            CheckResult(
                name="Test Check 3",
                status=CheckStatus.WARNING,
                message="Minor issue"
            )
        ]
        
        formatted = format_check_results(results)
        
        assert isinstance(formatted, str)
        assert "Test Check 1" in formatted
        assert "Test Check 2" in formatted
        assert "Test Check 3" in formatted
        assert "✓" in formatted  # Pass icon
        assert "✗" in formatted  # Fail icon
        assert "⚠" in formatted  # Warning icon
        assert "Fix it" in formatted  # Suggestion
        assert "Summary:" in formatted
    
    def test_format_check_results_empty(self):
        """Test formatting of empty results."""
        results = []
        formatted = format_check_results(results)
        
        assert isinstance(formatted, str)
        assert "Summary: 0 passed, 0 warnings, 0 failed, 0 skipped" in formatted


class TestCheckResult:
    """Test the CheckResult data class."""
    
    def test_check_result_creation(self):
        """Test CheckResult creation."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.PASS,
            message="Test message",
            suggestion="Test suggestion",
            details={"key": "value"}
        )
        
        assert result.name == "Test Check"
        assert result.status == CheckStatus.PASS
        assert result.message == "Test message"
        assert result.suggestion == "Test suggestion"
        assert result.details == {"key": "value"}
    
    def test_check_result_minimal(self):
        """Test CheckResult with minimal parameters."""
        result = CheckResult(
            name="Minimal Check",
            status=CheckStatus.WARNING,
            message="Warning message"
        )
        
        assert result.name == "Minimal Check"
        assert result.status == CheckStatus.WARNING
        assert result.message == "Warning message"
        assert result.suggestion is None
        assert result.details is None


@pytest.mark.integration
class TestStartupValidationIntegration:
    """Integration tests for startup validation."""
    
    @pytest.mark.asyncio
    async def test_full_validation_cycle(self):
        """Test complete validation cycle."""
        # This test requires actual system dependencies
        success, results = await validate_startup()
        
        assert isinstance(success, bool)
        assert isinstance(results, list)
        
        for result in results:
            assert hasattr(result, 'name')
            assert hasattr(result, 'status')
            assert hasattr(result, 'message')
            assert result.status in [CheckStatus.PASS, CheckStatus.WARNING, CheckStatus.FAIL, CheckStatus.SKIP]
    
    @pytest.mark.asyncio
    async def test_real_system_checks(self):
        """Test checks against real system."""
        validator = StartupValidator()
        
        # Test Python version check
        await validator._check_python_version()
        python_results = [r for r in validator.results if "python" in r.name.lower()]
        assert len(python_results) > 0
        
        # Test disk space check
        await validator._check_disk_space()
        disk_results = [r for r in validator.results if "disk" in r.name.lower()]
        assert len(disk_results) > 0
        
        # Test configuration check
        await validator._check_configuration()
        config_results = [r for r in validator.results if "configuration" in r.name.lower()]
        assert len(config_results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])