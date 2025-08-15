#!/bin/bash

# Letta Construction Claim Assistant - Installation Script
# Automated installation for Ubuntu Linux systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="letta-claim-assistant"
PYTHON_MIN_VERSION="3.9"
INSTALL_DIR="$HOME/Applications/$PROJECT_NAME"
DATA_DIR="$HOME/LettaClaims"
CONFIG_DIR="$HOME/.letta-claim"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu
check_os() {
    print_status "Checking operating system..."
    
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        if [[ "$ID" == "ubuntu" ]]; then
            print_success "Ubuntu $VERSION_ID detected"
            
            # Check version
            VERSION_NUM=$(echo $VERSION_ID | cut -d. -f1)
            if [[ $VERSION_NUM -lt 20 ]]; then
                print_warning "Ubuntu $VERSION_ID detected. Ubuntu 20.04+ recommended."
            fi
        else
            print_warning "Non-Ubuntu system detected ($ID). Installation may work but is not officially supported."
        fi
    else
        print_warning "Could not detect OS version. Proceeding anyway."
    fi
}

# Check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
        
        # Check minimum version
        MIN_VERSION_NUM=$(echo $PYTHON_MIN_VERSION | sed 's/\.//')
        CURRENT_VERSION_NUM=$(echo $PYTHON_VERSION | sed 's/\.//' | cut -c1-2)
        
        if [[ $CURRENT_VERSION_NUM -lt $MIN_VERSION_NUM ]]; then
            print_error "Python $PYTHON_MIN_VERSION or higher required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3."
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Update package lists
    print_status "Updating package lists..."
    sudo apt-get update -qq
    
    # Install required packages
    PACKAGES=(
        "python3"
        "python3-pip"
        "python3-venv"
        "python3-dev"
        "ocrmypdf"
        "tesseract-ocr"
        "tesseract-ocr-eng"
        "tesseract-ocr-osd"
        "poppler-utils"
        "ghostscript"
        "curl"
        "wget"
        "git"
        "build-essential"
    )
    
    print_status "Installing packages: ${PACKAGES[*]}"
    
    for package in "${PACKAGES[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            print_status "Installing $package..."
            sudo apt-get install -y "$package"
        else
            print_success "$package already installed"
        fi
    done
    
    print_success "System dependencies installed"
}

# Install Ollama
install_ollama() {
    print_status "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama already installed"
    else
        print_status "Downloading and installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        print_success "Ollama installed"
    fi
    
    # Start Ollama service
    print_status "Starting Ollama service..."
    if pgrep ollama > /dev/null; then
        print_success "Ollama service already running"
    else
        print_status "Starting Ollama..."
        ollama serve &
        sleep 3
        
        if pgrep ollama > /dev/null; then
            print_success "Ollama service started"
        else
            print_warning "Ollama service may not have started. You may need to start it manually."
        fi
    fi
    
    # Pull required models
    print_status "Downloading required AI models (this may take several minutes)..."
    
    print_status "Pulling generation model (gpt-oss:20b)..."
    if ollama list | grep -q "gpt-oss:20b"; then
        print_success "gpt-oss:20b already available"
    else
        print_status "Downloading gpt-oss:20b (~11GB, this will take a while)..."
        ollama pull gpt-oss:20b
        print_success "gpt-oss:20b downloaded"
    fi
    
    print_status "Pulling embedding model (nomic-embed-text)..."
    if ollama list | grep -q "nomic-embed-text"; then
        print_success "nomic-embed-text already available"
    else
        ollama pull nomic-embed-text
        print_success "nomic-embed-text downloaded"
    fi
    
    print_success "Ollama installation complete"
}

# Download application
download_app() {
    print_status "Setting up application..."
    
    # Create installation directory
    mkdir -p "$(dirname "$INSTALL_DIR")"
    
    if [[ -d "$INSTALL_DIR" ]]; then
        print_warning "Installation directory exists. Backing up..."
        mv "$INSTALL_DIR" "${INSTALL_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Check if we're in the project directory already
    if [[ -f "main.py" && -f "requirements.txt" ]]; then
        print_status "Installing from current directory..."
        cp -r "$(pwd)" "$INSTALL_DIR"
    else
        # Try to clone from git (would need actual repository URL)
        print_status "Cloning application repository..."
        if command -v git &> /dev/null; then
            # This would be replaced with actual repository URL
            # git clone https://github.com/your-org/letta-claim-assistant.git "$INSTALL_DIR"
            print_error "Git repository URL not configured. Please download manually."
            exit 1
        else
            print_error "Git not available and not in project directory."
            exit 1
        fi
    fi
    
    print_success "Application files installed to $INSTALL_DIR"
}

# Set up Python environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment
    if [[ -d ".venv" ]]; then
        print_warning "Virtual environment exists. Recreating..."
        rm -rf .venv
    fi
    
    python3 -m venv .venv
    print_success "Virtual environment created"
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Create configuration
setup_configuration() {
    print_status "Setting up configuration..."
    
    # Create config directory
    mkdir -p "$CONFIG_DIR"
    chmod 700 "$CONFIG_DIR"
    
    # Create default configuration
    if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
        if [[ -f "$INSTALL_DIR/config.toml.example" ]]; then
            cp "$INSTALL_DIR/config.toml.example" "$CONFIG_DIR/config.toml"
            print_success "Default configuration created"
        else
            print_warning "Example configuration not found. Creating minimal config."
            cat > "$CONFIG_DIR/config.toml" << EOF
[ui]
framework = "nicegui"
native = true

[llm]
provider = "ollama"
model = "gpt-oss:20b"
temperature = 0.2
max_tokens = 900

[embeddings]
provider = "ollama"
model = "nomic-embed-text"

[ocr]
enabled = true
force_ocr = false
language = "eng"
skip_text = true

[paths]
root = "~/LettaClaims"

[gemini]
api_key = ""
model = "gemini-2.5-flash"

[letta]
enable_filesystem = true
EOF
        fi
    else
        print_success "Configuration already exists"
    fi
    
    # Create data directory
    mkdir -p "$DATA_DIR"
    print_success "Data directory created at $DATA_DIR"
}

# Create desktop integration
setup_desktop_integration() {
    print_status "Setting up desktop integration..."
    
    # Create launcher script
    LAUNCHER_SCRIPT="$INSTALL_DIR/scripts/launcher.sh"
    mkdir -p "$(dirname "$LAUNCHER_SCRIPT")"
    
    cat > "$LAUNCHER_SCRIPT" << EOF
#!/bin/bash
# Letta Construction Claim Assistant Launcher

cd "$INSTALL_DIR"
source .venv/bin/activate
python main.py
EOF
    
    chmod +x "$LAUNCHER_SCRIPT"
    print_success "Launcher script created"
    
    # Create desktop entry
    DESKTOP_FILE="$HOME/.local/share/applications/letta-claim-assistant.desktop"
    mkdir -p "$(dirname "$DESKTOP_FILE")"
    
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=Letta Claim Assistant
Comment=Construction claim analysis with AI
Exec=$LAUNCHER_SCRIPT
Icon=$INSTALL_DIR/desktop/icon.png
Type=Application
Categories=Office;Finance;
Terminal=false
StartupNotify=true
Keywords=construction;claims;legal;AI;analysis;
EOF
    
    chmod +x "$DESKTOP_FILE"
    
    # Update desktop database
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database "$HOME/.local/share/applications/"
    fi
    
    print_success "Desktop integration created"
}

# Run validation tests
run_validation() {
    print_status "Running installation validation..."
    
    cd "$INSTALL_DIR"
    source .venv/bin/activate
    
    # Test Python imports
    print_status "Testing Python imports..."
    python3 -c "
import sys
packages = ['nicegui', 'chromadb', 'ollama', 'pydantic', 'structlog', 'pymupdf']
failed = []

for package in packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError as e:
        print(f'✗ {package}: {e}')
        failed.append(package)

if failed:
    print(f'Failed to import: {failed}')
    sys.exit(1)
else:
    print('All packages imported successfully')
"
    
    if [[ $? -eq 0 ]]; then
        print_success "Python package validation passed"
    else
        print_error "Python package validation failed"
        exit 1
    fi
    
    # Test system dependencies
    print_status "Testing system dependencies..."
    
    REQUIRED_COMMANDS=("ocrmypdf" "tesseract" "pdfinfo" "gs")
    for cmd in "${REQUIRED_COMMANDS[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            print_success "$cmd is available"
        else
            print_error "$cmd is not available"
            exit 1
        fi
    done
    
    # Test Ollama
    print_status "Testing Ollama installation..."
    if ollama list &> /dev/null; then
        MODEL_COUNT=$(ollama list | tail -n +2 | wc -l)
        print_success "Ollama is working with $MODEL_COUNT models"
    else
        print_error "Ollama is not working properly"
        exit 1
    fi
    
    print_success "Installation validation completed successfully"
}

# Display post-installation instructions
show_completion_message() {
    print_success "Installation completed successfully!"
    echo
    echo "=== Next Steps ==="
    echo
    echo "1. Start the application:"
    echo "   $INSTALL_DIR/scripts/launcher.sh"
    echo
    echo "2. Or find 'Letta Claim Assistant' in your applications menu"
    echo
    echo "3. The application will open in desktop mode by default"
    echo
    echo "4. Create your first matter and upload some PDF documents"
    echo
    echo "=== Important Locations ==="
    echo "- Application: $INSTALL_DIR"
    echo "- Configuration: $CONFIG_DIR/config.toml"
    echo "- Data: $DATA_DIR"
    echo "- Logs: $CONFIG_DIR/logs/"
    echo
    echo "=== Getting Help ==="
    echo "- User Guide: $INSTALL_DIR/docs/USER_GUIDE.md"
    echo "- Troubleshooting: $INSTALL_DIR/docs/TROUBLESHOOTING.md"
    echo "- Configuration: $INSTALL_DIR/docs/CONFIGURATION.md"
    echo
    echo "=== Health Check ==="
    echo "After starting the app, visit: http://localhost:8000/api/health"
    echo
    print_success "Ready to analyze construction claims with AI!"
}

# Main installation function
main() {
    echo "========================================"
    echo "Letta Construction Claim Assistant"
    echo "Automated Installation Script"
    echo "========================================"
    echo
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_error "Do not run this script as root. Run as a regular user."
        exit 1
    fi
    
    # Confirm installation
    echo "This script will install the Letta Construction Claim Assistant"
    echo "and all required dependencies on your Ubuntu system."
    echo
    echo "Installation locations:"
    echo "- Application: $INSTALL_DIR"
    echo "- Data: $DATA_DIR"
    echo "- Configuration: $CONFIG_DIR"
    echo
    read -p "Continue with installation? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installation cancelled"
        exit 0
    fi
    
    echo
    print_status "Starting installation..."
    
    # Run installation steps
    check_os
    check_python
    install_system_deps
    install_ollama
    download_app
    setup_python_env
    setup_configuration
    setup_desktop_integration
    run_validation
    show_completion_message
}

# Handle interruption
trap 'print_error "Installation interrupted"; exit 1' INT TERM

# Run main function
main "$@"