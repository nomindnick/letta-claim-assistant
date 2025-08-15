#!/usr/bin/env python3
"""
Setup script for the Letta Construction Claim Assistant.

This script enables packaging the application for distribution and installation
via pip or other Python package managers.
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the project directory
here = Path(__file__).parent.resolve()

# Read the README file
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

# Read version from a version file or use default
def get_version():
    """Get version from git tag or default."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            cwd=here
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            # Convert git describe output to valid version
            if "-" in version:
                # Convert "v1.0.0-5-g1234567" to "1.0.0.dev5"
                parts = version.split("-")
                if len(parts) >= 3 and parts[0].startswith("v"):
                    base_version = parts[0][1:]  # Remove 'v' prefix
                    dev_number = parts[1]
                    return f"{base_version}.dev{dev_number}"
            elif version.startswith("v"):
                return version[1:]  # Remove 'v' prefix
            return version
    except Exception:
        pass
    
    return "1.0.0.dev0"

# Read requirements
def get_requirements():
    """Read requirements from requirements.txt."""
    requirements_file = here / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, "r", encoding="utf-8") as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith("#")
            ]
    return []

# Get data files
def get_data_files():
    """Get list of data files to include in the package."""
    data_files = []
    
    # Include documentation
    docs_dir = here / "docs"
    if docs_dir.exists():
        doc_files = [str(f.relative_to(here)) for f in docs_dir.rglob("*") if f.is_file()]
        data_files.extend(doc_files)
    
    # Include scripts
    scripts_dir = here / "scripts"
    if scripts_dir.exists():
        script_files = [str(f.relative_to(here)) for f in scripts_dir.rglob("*") if f.is_file()]
        data_files.extend(script_files)
    
    # Include desktop files
    desktop_dir = here / "desktop"
    if desktop_dir.exists():
        desktop_files = [str(f.relative_to(here)) for f in desktop_dir.rglob("*") if f.is_file()]
        data_files.extend(desktop_files)
    
    # Include configuration template
    config_files = [
        "config.toml.example",
        "pytest.ini"
    ]
    
    for config_file in config_files:
        if (here / config_file).exists():
            data_files.append(config_file)
    
    return data_files

# Package configuration
setup(
    # Basic package information
    name="letta-claim-assistant",
    version=get_version(),
    author="Letta Development Team",
    author_email="dev@letta.com",
    description="AI-powered construction claim analysis with persistent agent memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/letta-claim-assistant",
    project_urls={
        "Bug Reports": "https://github.com/your-org/letta-claim-assistant/issues",
        "Source": "https://github.com/your-org/letta-claim-assistant",
        "Documentation": "https://github.com/your-org/letta-claim-assistant/wiki",
    },
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "": ["*.toml", "*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    
    # Data files
    data_files=[
        ("share/letta-claim-assistant", get_data_files()),
        ("share/applications", ["desktop/letta-claims.desktop"]),
        ("share/pixmaps", ["desktop/icon.svg", "desktop/icon.png"]),
    ],
    
    # Dependencies
    python_requires=">=3.9",
    install_requires=get_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings>=0.22.0",
        ],
        "performance": [
            "psutil>=5.9.0",
            "py-spy>=0.3.14",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "letta-claim-assistant=main:main",
            "letta-claims=main:main",
        ],
        "gui_scripts": [
            "letta-claim-assistant-gui=main:main",
        ],
    },
    
    # Classification
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        
        # Intended audience
        "Intended Audience :: Legal Industry",
        "Intended Audience :: End Users/Desktop",
        
        # Topic
        "Topic :: Office/Business :: Financial :: Accounting",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating systems
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        
        # Environment
        "Environment :: X11 Applications",
        "Environment :: Web Environment",
    ],
    
    # Keywords
    keywords="construction claims legal ai analysis documents contracts specifications",
    
    # Platforms
    platforms=["Linux", "macOS", "Windows"],
    
    # Additional metadata
    license="MIT",
    include_package_data=True,
    zip_safe=False,
    
    # Command options
    options={
        "build_py": {
            "exclude": ["tests", "tests.*"],
        },
        "sdist": {
            "formats": ["gztar", "zip"],
        },
        "bdist_wheel": {
            "universal": False,
        },
    },
)

# Post-installation setup
def post_install():
    """Run post-installation setup tasks."""
    print("Setting up Letta Construction Claim Assistant...")
    
    # Create configuration directory
    config_dir = Path.home() / ".letta-claim"
    config_dir.mkdir(exist_ok=True, mode=0o700)
    
    # Create data directory
    data_dir = Path.home() / "LettaClaims"
    data_dir.mkdir(exist_ok=True)
    
    # Copy configuration template if it doesn't exist
    config_file = config_dir / "config.toml"
    if not config_file.exists():
        template_file = Path(__file__).parent / "config.toml.example"
        if template_file.exists():
            import shutil
            shutil.copy2(template_file, config_file)
            print(f"Created configuration file: {config_file}")
    
    print("Installation completed successfully!")
    print(f"Configuration: {config_dir}")
    print(f"Data directory: {data_dir}")
    print("Run 'letta-claim-assistant' to start the application.")

if __name__ == "__main__":
    # Run setup
    setup()
    
    # Run post-installation if this is being run directly
    if "install" in os.sys.argv:
        post_install()