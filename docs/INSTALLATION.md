# Letta Construction Claim Assistant - Installation Guide

This guide provides step-by-step instructions for installing the Letta Construction Claim Assistant on Ubuntu Linux desktop systems.

## System Requirements

### Minimum Requirements
- Ubuntu 20.04 LTS or later
- Python 3.9 or later (Python 3.11+ recommended)
- 4 GB RAM (8 GB recommended)
- 5 GB free disk space (10 GB recommended for multiple matters)
- Internet connection for model downloads and updates

### Recommended System Specifications
- Ubuntu 22.04 LTS or later
- Python 3.11+
- 8 GB RAM or more
- SSD storage with 20 GB+ free space
- Stable internet connection

## Installation Methods

### Method 1: Automated Installation (Recommended)

The easiest way to install the application is using the provided installation script:

```bash
# Download and run the installation script
curl -fsSL https://github.com/your-org/letta-claim-assistant/raw/main/scripts/install.sh | bash

# Or download and inspect first
wget https://github.com/your-org/letta-claim-assistant/raw/main/scripts/install.sh
chmod +x install.sh
./install.sh
```

The installation script will:
1. Check system compatibility
2. Install required system packages
3. Set up Python virtual environment
4. Install Python dependencies
5. Download and configure Ollama with required models
6. Create desktop shortcuts
7. Run initial validation tests

### Method 2: Manual Installation

If you prefer manual installation or need custom configuration:

#### Step 1: Install System Dependencies

```bash
# Update package lists
sudo apt-get update

# Install required system packages
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ocrmypdf \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    poppler-utils \
    curl \
    wget \
    git

# Optional: Install additional OCR languages
sudo apt-get install -y tesseract-ocr-spa  # Spanish
sudo apt-get install -y tesseract-ocr-fra  # French
```

#### Step 2: Install Ollama

Ollama provides local LLM capabilities:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull required models (this may take several minutes)
ollama pull gpt-oss:20b          # Generation model (large, ~11GB)
ollama pull nomic-embed-text     # Embedding model (~274MB)

# Optional: Pull alternative models
ollama pull llama3.1            # Faster generation model
ollama pull mxbai-embed-large   # Alternative embedding model
```

**Note:** The `gpt-oss:20b` model is large (~11GB). For faster performance on modest hardware, consider using `llama3.1` instead.

#### Step 3: Download Application

```bash
# Clone the repository
git clone https://github.com/your-org/letta-claim-assistant.git
cd letta-claim-assistant

# Or download release archive
wget https://github.com/your-org/letta-claim-assistant/archive/v1.0.0.tar.gz
tar -xzf v1.0.0.tar.gz
cd letta-claim-assistant-1.0.0
```

#### Step 4: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Step 5: Initial Configuration

```bash
# Copy configuration template
mkdir -p ~/.letta-claim
cp config.toml.example ~/.letta-claim/config.toml

# Edit configuration (optional)
nano ~/.letta-claim/config.toml
```

#### Step 6: Validate Installation

```bash
# Run system validation
python -c "
from app.startup_checks import validate_startup
from app.production_config import validate_production_config
import asyncio

async def main():
    startup_ok, startup_results = await validate_startup()
    config_ok, config_results = validate_production_config()
    
    print('Startup validation:', 'PASS' if startup_ok else 'FAIL')
    print('Config validation:', 'PASS' if config_ok else 'FAIL')
    
    if not startup_ok:
        for result in startup_results:
            if result.status.value == 'fail':
                print(f'  ✗ {result.name}: {result.message}')
                if result.suggestion:
                    print(f'    → {result.suggestion}')

asyncio.run(main())
"
```

## Configuration

### Global Configuration

Edit `~/.letta-claim/config.toml` to customize application settings:

```toml
[ui]
framework = "nicegui"
native = true

[llm]
provider = "ollama"              # "ollama" | "gemini"
model = "gpt-oss:20b"           # or "llama3.1" for faster performance
temperature = 0.2
max_tokens = 900

[embeddings]
provider = "ollama"
model = "nomic-embed-text"

[ocr]
enabled = true
force_ocr = false
language = "eng"
skip_text = true                 # Only OCR pages without text

[paths]
root = "~/LettaClaims"          # Data storage location

[gemini]
api_key = ""                    # Set if using Gemini
model = "gemini-2.5-flash"

[letta]
enable_filesystem = true
```

### External LLM Configuration (Optional)

To use Google Gemini instead of local Ollama:

1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Configure in application settings (not in config file for security)
3. The application will prompt for consent when first using external services

## Running the Application

### Desktop Mode (Recommended)

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the application
python main.py
```

The application will open in a native desktop window. If native mode fails, it will automatically fall back to browser mode.

### Browser Mode

If you prefer browser mode or native mode isn't working:

```bash
# Set environment variable
export LETTA_UI_NATIVE=false

# Start the application
python main.py
```

Then open your browser to `http://localhost:8000`

## Desktop Integration

### Create Desktop Shortcut

```bash
# Create desktop entry
cat > ~/.local/share/applications/letta-claim-assistant.desktop << EOF
[Desktop Entry]
Name=Letta Claim Assistant
Comment=Construction claim analysis with AI
Exec=/path/to/letta-claim-assistant/scripts/launcher.sh
Icon=/path/to/letta-claim-assistant/desktop/icon.png
Type=Application
Categories=Office;Finance;
Terminal=false
StartupNotify=true
EOF

# Make executable
chmod +x ~/.local/share/applications/letta-claim-assistant.desktop

# Update desktop database
update-desktop-database ~/.local/share/applications/
```

### Add to Applications Menu

The desktop entry above will automatically add the application to your applications menu under "Office" applications.

## Verification

### Quick Verification

1. **Start Application**: Run `python main.py` - application should start without errors
2. **Create Matter**: Create a new matter through the UI
3. **Upload PDF**: Upload a sample PDF document
4. **Ask Question**: Ask a simple question about the document
5. **Check Sources**: Verify that sources are displayed with page references

### Health Check

Visit `http://localhost:8000/api/health` to see system health status:

```bash
curl http://localhost:8000/api/health | python3 -m json.tool
```

Expected healthy response:
```json
{
  "status": "healthy",
  "services": {
    "ollama": "healthy",
    "chromadb": "healthy",
    "letta": "healthy"
  },
  "resources": {
    "memory": {"status": "healthy"},
    "disk": {"status": "healthy"},
    "cpu": {"status": "healthy"}
  }
}
```

## Troubleshooting Common Issues

### Issue: Ollama Models Not Found

**Symptoms**: Error messages about missing models

**Solution**:
```bash
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

### Issue: Permission Errors

**Symptoms**: Cannot create directories or write files

**Solution**:
```bash
# Fix ownership of data directory
sudo chown -R $USER:$USER ~/LettaClaims

# Fix permissions
chmod -R 755 ~/LettaClaims
```

### Issue: Python Import Errors

**Symptoms**: `ModuleNotFoundError` when starting

**Solution**:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: OCR Fails

**Symptoms**: PDF processing fails with OCR errors

**Solution**:
```bash
# Install additional OCR dependencies
sudo apt-get install -y ghostscript

# Check Tesseract installation
tesseract --version

# Test OCR manually
ocrmypdf --version
```

### Issue: High Memory Usage

**Symptoms**: System becomes slow, high RAM usage

**Solutions**:
1. Switch to a lighter model: Change `llm.model` to `"llama3.1"`
2. Reduce concurrent operations: Process fewer documents at once
3. Monitor memory: Check `/api/metrics` endpoint for memory usage

### Issue: Slow Performance

**Symptoms**: Long response times, UI feels sluggish

**Solutions**:
1. **Use lighter models**: Switch from `gpt-oss:20b` to `llama3.1`
2. **Reduce context**: Lower `max_tokens` in configuration
3. **Check hardware**: Ensure adequate RAM and CPU
4. **Monitor performance**: Use `/api/metrics` to identify bottlenecks

## Updating

### Update Application

```bash
# Navigate to installation directory
cd letta-claim-assistant

# Pull latest changes
git pull origin main

# Activate virtual environment
source .venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart application
python main.py
```

### Update Models

```bash
# Update Ollama models
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

## Uninstallation

### Remove Application

```bash
# Remove application directory
rm -rf /path/to/letta-claim-assistant

# Remove virtual environment
rm -rf /path/to/letta-claim-assistant/.venv

# Remove desktop entry
rm ~/.local/share/applications/letta-claim-assistant.desktop
```

### Remove User Data (Optional)

**Warning**: This will delete all your matters and documents!

```bash
# Remove user data
rm -rf ~/LettaClaims

# Remove configuration
rm -rf ~/.letta-claim
```

### Remove System Dependencies (Optional)

```bash
# Remove system packages (be careful - may be used by other applications)
sudo apt-get remove ocrmypdf tesseract-ocr poppler-utils

# Remove Ollama
sudo rm /usr/local/bin/ollama
sudo rm -rf ~/.ollama
```

## Security Considerations

### Data Privacy
- All document processing happens locally by default
- External LLM providers (Gemini) require explicit user consent
- API keys are stored encrypted locally
- No data is transmitted to external services without consent

### File Permissions
- Application creates `~/.letta-claim` with restricted permissions (700)
- Document storage under `~/LettaClaims` is user-accessible only
- Log files contain no sensitive information by default

### Network Security
- Local mode requires no internet connection after initial setup
- External API calls only when explicitly configured
- No telemetry or tracking by default

## Getting Help

### Documentation
- User Guide: `docs/USER_GUIDE.md`
- Configuration Reference: `docs/CONFIGURATION.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`

### Support
- Check system health: `/api/health/detailed`
- View logs: `~/.letta-claim/logs/app.log`
- Report issues: [GitHub Issues](https://github.com/your-org/letta-claim-assistant/issues)

### Community
- Discussions: [GitHub Discussions](https://github.com/your-org/letta-claim-assistant/discussions)
- Documentation: [Wiki](https://github.com/your-org/letta-claim-assistant/wiki)

---

**Next Steps**: After installation, see the [User Guide](USER_GUIDE.md) to learn how to use the application effectively.