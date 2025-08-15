"""
Startup validation checks for the Letta Construction Claim Assistant.

Validates system dependencies, resources, and configuration before starting
the application to prevent runtime failures and provide helpful error messages.
"""

import subprocess
import sys
import shutil
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .logging_conf import get_logger
from .settings import settings
from .error_handler import StartupError, handle_error, create_context

logger = get_logger(__name__)


class CheckStatus(Enum):
    """Status of a startup check."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a startup check."""
    name: str
    status: CheckStatus
    message: str
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class StartupValidator:
    """Validates system readiness for application startup."""
    
    def __init__(self):
        self.results: List[CheckResult] = []
        self.config = settings.global_config
    
    async def run_all_checks(self) -> Tuple[bool, List[CheckResult]]:
        """
        Run all startup validation checks.
        
        Returns:
            Tuple of (success, check_results)
        """
        logger.info("Starting application validation checks")
        
        self.results = []
        
        # System dependency checks
        await self._check_python_version()
        await self._check_system_packages()
        await self._check_python_packages()
        
        # Resource checks
        await self._check_disk_space()
        await self._check_memory()
        await self._check_permissions()
        
        # Service checks
        await self._check_ollama_service()
        await self._check_ollama_models()
        await self._check_external_services()
        
        # Configuration checks
        await self._check_configuration()
        await self._check_data_directories()
        
        # Determine overall success
        critical_failures = [r for r in self.results if r.status == CheckStatus.FAIL]
        success = len(critical_failures) == 0
        
        if success:
            logger.info("All startup checks passed successfully")
        else:
            logger.error(
                "Startup validation failed", 
                failures=len(critical_failures),
                total_checks=len(self.results)
            )
        
        return success, self.results
    
    async def _check_python_version(self):
        """Check Python version compatibility."""
        try:
            version = sys.version_info
            
            if version >= (3, 11):
                self.results.append(CheckResult(
                    name="Python Version",
                    status=CheckStatus.PASS,
                    message=f"Python {version.major}.{version.minor}.{version.micro}",
                    details={"version": f"{version.major}.{version.minor}.{version.micro}"}
                ))
            elif version >= (3, 9):
                self.results.append(CheckResult(
                    name="Python Version",
                    status=CheckStatus.WARNING,
                    message=f"Python {version.major}.{version.minor} (recommend 3.11+)",
                    suggestion="Consider upgrading to Python 3.11+ for better performance",
                    details={"version": f"{version.major}.{version.minor}.{version.micro}"}
                ))
            else:
                self.results.append(CheckResult(
                    name="Python Version",
                    status=CheckStatus.FAIL,
                    message=f"Python {version.major}.{version.minor} (requires 3.9+)",
                    suggestion="Upgrade to Python 3.9 or higher",
                    details={"version": f"{version.major}.{version.minor}.{version.micro}"}
                ))
        except Exception as e:
            self.results.append(CheckResult(
                name="Python Version",
                status=CheckStatus.FAIL,
                message=f"Failed to check Python version: {e}",
                suggestion="Verify Python installation"
            ))
    
    async def _check_system_packages(self):
        """Check required system packages."""
        packages = {
            "ocrmypdf": "OCR processing",
            "tesseract": "Text recognition",
            "pdfinfo": "PDF information extraction"
        }
        
        for package, description in packages.items():
            try:
                result = subprocess.run(
                    ["which", package], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                
                if result.returncode == 0:
                    # Get version if possible
                    try:
                        if package == "ocrmypdf":
                            version_result = subprocess.run(
                                ["ocrmypdf", "--version"], 
                                capture_output=True, 
                                text=True, 
                                timeout=5
                            )
                            version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"
                        elif package == "tesseract":
                            version_result = subprocess.run(
                                ["tesseract", "--version"], 
                                capture_output=True, 
                                text=True, 
                                timeout=5
                            )
                            version = version_result.stderr.split('\n')[0] if version_result.returncode == 0 else "unknown"
                        else:
                            version = "installed"
                    except:
                        version = "installed"
                    
                    self.results.append(CheckResult(
                        name=f"System Package: {package}",
                        status=CheckStatus.PASS,
                        message=f"{description} - {version}",
                        details={"package": package, "version": version, "path": result.stdout.strip()}
                    ))
                else:
                    self.results.append(CheckResult(
                        name=f"System Package: {package}",
                        status=CheckStatus.FAIL,
                        message=f"{description} - not found",
                        suggestion=f"Install {package}: sudo apt-get install {self._get_package_name(package)}",
                        details={"package": package, "required": True}
                    ))
            except subprocess.TimeoutExpired:
                self.results.append(CheckResult(
                    name=f"System Package: {package}",
                    status=CheckStatus.WARNING,
                    message=f"{description} - check timed out",
                    suggestion=f"Verify {package} installation manually"
                ))
            except Exception as e:
                self.results.append(CheckResult(
                    name=f"System Package: {package}",
                    status=CheckStatus.FAIL,
                    message=f"{description} - check failed: {e}",
                    suggestion=f"Manually verify {package} installation"
                ))
    
    def _get_package_name(self, command: str) -> str:
        """Get package name for installation."""
        package_map = {
            "ocrmypdf": "ocrmypdf",
            "tesseract": "tesseract-ocr",
            "pdfinfo": "poppler-utils"
        }
        return package_map.get(command, command)
    
    async def _check_python_packages(self):
        """Check critical Python packages."""
        critical_packages = [
            "nicegui", "fastapi", "chromadb", "ollama", 
            "pydantic", "structlog", "pymupdf"
        ]
        
        for package in critical_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
                
                self.results.append(CheckResult(
                    name=f"Python Package: {package}",
                    status=CheckStatus.PASS,
                    message=f"Version {version}",
                    details={"package": package, "version": version}
                ))
            except ImportError:
                self.results.append(CheckResult(
                    name=f"Python Package: {package}",
                    status=CheckStatus.FAIL,
                    message="Not installed",
                    suggestion=f"Install with: pip install {package}",
                    details={"package": package, "required": True}
                ))
            except Exception as e:
                self.results.append(CheckResult(
                    name=f"Python Package: {package}",
                    status=CheckStatus.WARNING,
                    message=f"Import check failed: {e}",
                    suggestion=f"Verify {package} installation"
                ))
    
    async def _check_disk_space(self):
        """Check available disk space."""
        try:
            data_root = Path(self.config.data_root)
            
            # Check if path exists or parent exists
            check_path = data_root if data_root.exists() else data_root.parent
            
            total, used, free = shutil.disk_usage(check_path)
            
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            percent_used = (used / total) * 100
            
            if free_gb >= 5.0:  # 5GB recommended
                status = CheckStatus.PASS
                message = f"{free_gb:.1f} GB free ({percent_used:.1f}% used)"
            elif free_gb >= 1.0:  # 1GB minimum
                status = CheckStatus.WARNING
                message = f"{free_gb:.1f} GB free ({percent_used:.1f}% used) - low space"
                suggestion = "Consider freeing up disk space for better performance"
            else:
                status = CheckStatus.FAIL
                message = f"{free_gb:.1f} GB free ({percent_used:.1f}% used) - insufficient"
                suggestion = "Free up disk space before continuing"
            
            self.results.append(CheckResult(
                name="Disk Space",
                status=status,
                message=message,
                suggestion=getattr(self, 'suggestion', None),
                details={
                    "free_gb": free_gb,
                    "total_gb": total_gb,
                    "percent_used": percent_used,
                    "path": str(check_path)
                }
            ))
        except Exception as e:
            self.results.append(CheckResult(
                name="Disk Space",
                status=CheckStatus.WARNING,
                message=f"Could not check disk space: {e}",
                suggestion="Manually verify sufficient disk space available"
            ))
    
    async def _check_memory(self):
        """Check available memory."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            percent_used = memory.percent
            
            if available_gb >= 2.0:  # 2GB available
                status = CheckStatus.PASS
                message = f"{available_gb:.1f} GB available ({percent_used:.1f}% used)"
            elif available_gb >= 1.0:  # 1GB minimum
                status = CheckStatus.WARNING
                message = f"{available_gb:.1f} GB available ({percent_used:.1f}% used) - low memory"
                suggestion = "Close other applications for better performance"
            else:
                status = CheckStatus.FAIL
                message = f"{available_gb:.1f} GB available ({percent_used:.1f}% used) - insufficient"
                suggestion = "Free up memory before continuing"
            
            self.results.append(CheckResult(
                name="Memory",
                status=status,
                message=message,
                suggestion=getattr(self, 'suggestion', None),
                details={
                    "available_gb": available_gb,
                    "total_gb": total_gb,
                    "percent_used": percent_used
                }
            ))
        except ImportError:
            self.results.append(CheckResult(
                name="Memory",
                status=CheckStatus.SKIP,
                message="psutil not available for memory check",
                suggestion="Install psutil for memory monitoring: pip install psutil"
            ))
        except Exception as e:
            self.results.append(CheckResult(
                name="Memory",
                status=CheckStatus.WARNING,
                message=f"Could not check memory: {e}",
                suggestion="Manually verify sufficient memory available"
            ))
    
    async def _check_permissions(self):
        """Check file system permissions."""
        try:
            data_root = Path(self.config.data_root)
            
            # Test creating the data directory
            if not data_root.exists():
                try:
                    data_root.mkdir(parents=True, exist_ok=True)
                    created = True
                except PermissionError:
                    self.results.append(CheckResult(
                        name="File Permissions",
                        status=CheckStatus.FAIL,
                        message=f"Cannot create data directory: {data_root}",
                        suggestion=f"Check permissions for path: {data_root.parent}",
                        details={"path": str(data_root), "operation": "create"}
                    ))
                    return
            else:
                created = False
            
            # Test write access
            test_file = data_root / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                
                self.results.append(CheckResult(
                    name="File Permissions",
                    status=CheckStatus.PASS,
                    message=f"Read/write access to {data_root}",
                    details={
                        "path": str(data_root), 
                        "created": created, 
                        "writable": True
                    }
                ))
            except PermissionError:
                self.results.append(CheckResult(
                    name="File Permissions",
                    status=CheckStatus.FAIL,
                    message=f"No write access to {data_root}",
                    suggestion=f"Check permissions for directory: {data_root}",
                    details={"path": str(data_root), "writable": False}
                ))
        except Exception as e:
            self.results.append(CheckResult(
                name="File Permissions",
                status=CheckStatus.WARNING,
                message=f"Permission check failed: {e}",
                suggestion="Manually verify file system permissions"
            ))
    
    async def _check_ollama_service(self):
        """Check Ollama service availability."""
        try:
            import ollama
            
            # Test connection to Ollama
            try:
                models = ollama.list()
                
                self.results.append(CheckResult(
                    name="Ollama Service",
                    status=CheckStatus.PASS,
                    message=f"Connected - {len(models.get('models', []))} models available",
                    details={"connected": True, "model_count": len(models.get('models', []))}
                ))
            except Exception as e:
                self.results.append(CheckResult(
                    name="Ollama Service",
                    status=CheckStatus.FAIL,
                    message=f"Cannot connect to Ollama: {e}",
                    suggestion="Start Ollama service: ollama serve",
                    details={"connected": False, "error": str(e)}
                ))
        except ImportError:
            self.results.append(CheckResult(
                name="Ollama Service",
                status=CheckStatus.FAIL,
                message="Ollama package not installed",
                suggestion="Install ollama: pip install ollama",
                details={"package_missing": True}
            ))
    
    async def _check_ollama_models(self):
        """Check required Ollama models."""
        required_models = [
            self.config.llm_model,
            self.config.embeddings_model
        ]
        
        try:
            import ollama
            
            available_models = ollama.list()
            model_names = [m['name'] for m in available_models.get('models', [])]
            
            for model in required_models:
                if model in model_names:
                    self.results.append(CheckResult(
                        name=f"Ollama Model: {model}",
                        status=CheckStatus.PASS,
                        message="Available",
                        details={"model": model, "available": True}
                    ))
                else:
                    self.results.append(CheckResult(
                        name=f"Ollama Model: {model}",
                        status=CheckStatus.FAIL,
                        message="Not available",
                        suggestion=f"Pull model: ollama pull {model}",
                        details={"model": model, "available": False}
                    ))
        except Exception as e:
            self.results.append(CheckResult(
                name="Ollama Models",
                status=CheckStatus.WARNING,
                message=f"Could not check models: {e}",
                suggestion="Manually verify required models are pulled"
            ))
    
    async def _check_external_services(self):
        """Check external service configuration."""
        if self.config.llm_provider == "gemini":
            api_key = settings.get_credential("gemini", "api_key")
            
            if api_key:
                # Could test API key validity here
                self.results.append(CheckResult(
                    name="Gemini API",
                    status=CheckStatus.PASS,
                    message="API key configured",
                    details={"configured": True}
                ))
            else:
                self.results.append(CheckResult(
                    name="Gemini API",
                    status=CheckStatus.FAIL,
                    message="API key not configured",
                    suggestion="Configure Gemini API key in settings",
                    details={"configured": False}
                ))
    
    async def _check_configuration(self):
        """Check configuration validity."""
        try:
            # Validate data root path
            data_root = Path(self.config.data_root)
            if data_root.is_absolute():
                self.results.append(CheckResult(
                    name="Configuration",
                    status=CheckStatus.PASS,
                    message="Valid configuration loaded",
                    details={"data_root": str(data_root)}
                ))
            else:
                self.results.append(CheckResult(
                    name="Configuration",
                    status=CheckStatus.WARNING,
                    message="Data root path is not absolute",
                    suggestion="Use absolute path for data_root in configuration"
                ))
        except Exception as e:
            self.results.append(CheckResult(
                name="Configuration",
                status=CheckStatus.FAIL,
                message=f"Configuration error: {e}",
                suggestion="Check configuration file syntax and values"
            ))
    
    async def _check_data_directories(self):
        """Check data directory structure."""
        try:
            data_root = Path(self.config.data_root)
            
            if data_root.exists() and data_root.is_dir():
                # Count existing matters
                matter_count = len([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("Matter_")])
                
                self.results.append(CheckResult(
                    name="Data Directories",
                    status=CheckStatus.PASS,
                    message=f"Data root exists - {matter_count} matters found",
                    details={"path": str(data_root), "matter_count": matter_count}
                ))
            else:
                self.results.append(CheckResult(
                    name="Data Directories",
                    status=CheckStatus.WARNING,
                    message="Data root directory will be created",
                    details={"path": str(data_root), "exists": False}
                ))
        except Exception as e:
            self.results.append(CheckResult(
                name="Data Directories",
                status=CheckStatus.WARNING,
                message=f"Could not check data directories: {e}",
                suggestion="Manually verify data directory accessibility"
            ))


async def validate_startup() -> Tuple[bool, List[CheckResult]]:
    """
    Run startup validation checks.
    
    Returns:
        Tuple of (success, check_results)
    """
    validator = StartupValidator()
    return await validator.run_all_checks()


def format_check_results(results: List[CheckResult]) -> str:
    """Format check results for display."""
    lines = ["Startup Validation Results:", "=" * 30]
    
    for result in results:
        status_icon = {
            CheckStatus.PASS: "✓",
            CheckStatus.WARNING: "⚠",
            CheckStatus.FAIL: "✗",
            CheckStatus.SKIP: "○"
        }[result.status]
        
        lines.append(f"{status_icon} {result.name}: {result.message}")
        
        if result.suggestion:
            lines.append(f"  → {result.suggestion}")
    
    # Summary
    pass_count = sum(1 for r in results if r.status == CheckStatus.PASS)
    warn_count = sum(1 for r in results if r.status == CheckStatus.WARNING)
    fail_count = sum(1 for r in results if r.status == CheckStatus.FAIL)
    skip_count = sum(1 for r in results if r.status == CheckStatus.SKIP)
    
    lines.extend([
        "",
        f"Summary: {pass_count} passed, {warn_count} warnings, {fail_count} failed, {skip_count} skipped"
    ])
    
    return "\n".join(lines)