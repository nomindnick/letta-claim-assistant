#!/bin/bash

# Letta Construction Claim Assistant - Desktop Launcher
# Launches the application with proper environment setup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="Letta Construction Claim Assistant"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
MAIN_SCRIPT="$PROJECT_DIR/main.py"
CONFIG_DIR="$HOME/.letta-claim"
LOG_FILE="$CONFIG_DIR/logs/launcher.log"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Ensure log directory exists
setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Log launch attempt
    echo "=== Launch Attempt: $(date) ===" >> "$LOG_FILE"
}

# Check if virtual environment exists
check_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        print_error "Virtual environment not found at $VENV_DIR"
        print_error "Please run the installation script first"
        return 1
    fi
    
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        print_error "Virtual environment appears to be corrupted"
        print_error "Please reinstall the application"
        return 1
    fi
    
    return 0
}

# Check if main script exists
check_main_script() {
    if [[ ! -f "$MAIN_SCRIPT" ]]; then
        print_error "Main application script not found at $MAIN_SCRIPT"
        print_error "Please check your installation"
        return 1
    fi
    
    return 0
}

# Check system dependencies
check_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check required commands
    local required_commands=("python3" "ollama" "ocrmypdf" "tesseract")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install missing dependencies or re-run installation"
        return 1
    fi
    
    print_success "System dependencies check passed"
    return 0
}

# Check Ollama service
check_ollama() {
    print_status "Checking Ollama service..."
    
    # Check if Ollama is running
    if ! pgrep ollama > /dev/null; then
        print_warning "Ollama service not running, attempting to start..."
        
        # Try to start Ollama
        ollama serve &
        sleep 3
        
        if ! pgrep ollama > /dev/null; then
            print_error "Failed to start Ollama service"
            print_error "Please start Ollama manually: ollama serve"
            return 1
        fi
        
        print_success "Ollama service started"
    else
        print_success "Ollama service is running"
    fi
    
    # Check if required models are available
    if ! ollama list &> /dev/null; then
        print_warning "Cannot query Ollama models"
        return 1
    fi
    
    local required_models=("gpt-oss:20b" "nomic-embed-text")
    local missing_models=()
    
    for model in "${required_models[@]}"; do
        if ! ollama list | grep -q "$model"; then
            missing_models+=("$model")
        fi
    done
    
    if [[ ${#missing_models[@]} -gt 0 ]]; then
        print_warning "Missing models: ${missing_models[*]}"
        print_warning "The application may work with alternative models"
    else
        print_success "Required models are available"
    fi
    
    return 0
}

# Check available resources
check_resources() {
    print_status "Checking system resources..."
    
    # Check available memory
    if command -v free &> /dev/null; then
        local available_mem=$(free -m | awk 'NR==2{print $7}')
        if [[ $available_mem -lt 1024 ]]; then
            print_warning "Low available memory: ${available_mem}MB (4GB+ recommended)"
        else
            print_success "Sufficient memory available: ${available_mem}MB"
        fi
    fi
    
    # Check disk space
    local data_dir="$HOME/LettaClaims"
    if [[ -d "$data_dir" ]]; then
        local available_space=$(df -BM "$data_dir" | awk 'NR==2{print $4}' | sed 's/M//')
        if [[ $available_space -lt 1024 ]]; then
            print_warning "Low disk space: ${available_space}MB (5GB+ recommended)"
        else
            print_success "Sufficient disk space: ${available_space}MB"
        fi
    fi
    
    return 0
}

# Activate virtual environment and launch application
launch_application() {
    print_status "Launching $APP_NAME..."
    
    # Change to project directory
    cd "$PROJECT_DIR" || {
        print_error "Failed to change to project directory"
        return 1
    }
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate" || {
        print_error "Failed to activate virtual environment"
        return 1
    }
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    
    # Launch application
    print_success "Starting application..."
    python "$MAIN_SCRIPT"
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Application exited normally"
    else
        print_error "Application exited with code $exit_code"
    fi
    
    return $exit_code
}

# Handle different launch modes
handle_launch_mode() {
    local mode="${1:-normal}"
    
    case "$mode" in
        "debug")
            print_status "Launching in debug mode..."
            export DEBUG=1
            export LETTA_LOG_LEVEL=DEBUG
            ;;
        "browser")
            print_status "Launching in browser mode..."
            export LETTA_UI_NATIVE=false
            ;;
        "safe")
            print_status "Launching in safe mode..."
            export LETTA_DISABLE_MONITORING=1
            export LETTA_UI_NATIVE=false
            ;;
        "normal"|*)
            print_status "Launching in normal mode..."
            ;;
    esac
}

# Show help message
show_help() {
    echo "Letta Construction Claim Assistant Launcher"
    echo
    echo "Usage: $0 [MODE]"
    echo
    echo "Modes:"
    echo "  normal  - Launch in normal desktop mode (default)"
    echo "  debug   - Launch with debug logging enabled"
    echo "  browser - Launch in browser mode instead of desktop"
    echo "  safe    - Launch with minimal features for troubleshooting"
    echo "  help    - Show this help message"
    echo
    echo "Examples:"
    echo "  $0           # Normal launch"
    echo "  $0 debug     # Debug mode"
    echo "  $0 browser   # Browser mode"
    echo "  $0 safe      # Safe mode"
}

# Check if port is already in use
check_port_availability() {
    local port="${LETTA_PORT:-8000}"
    
    if command -v lsof &> /dev/null; then
        if lsof -Pi ":$port" -sTCP:LISTEN -t &> /dev/null; then
            print_warning "Port $port is already in use"
            print_warning "Another instance may be running, or try a different port"
            
            # Try to find what's using the port
            local process=$(lsof -Pi ":$port" -sTCP:LISTEN | tail -n +2)
            if [[ -n "$process" ]]; then
                print_warning "Process using port: $process"
            fi
            
            return 1
        fi
    fi
    
    return 0
}

# Pre-flight checks
run_preflight_checks() {
    print_status "Running pre-flight checks..."
    
    local checks_passed=true
    
    if ! check_venv; then
        checks_passed=false
    fi
    
    if ! check_main_script; then
        checks_passed=false
    fi
    
    if ! check_dependencies; then
        checks_passed=false
    fi
    
    if ! check_ollama; then
        # Don't fail on Ollama issues, just warn
        print_warning "Ollama check failed, but continuing anyway"
    fi
    
    if ! check_port_availability; then
        # Don't fail on port issues, application will handle it
        print_warning "Port availability check failed, but continuing anyway"
    fi
    
    check_resources  # This is informational only
    
    if [[ "$checks_passed" == "false" ]]; then
        print_error "Pre-flight checks failed"
        print_error "Please resolve the issues above before launching"
        return 1
    fi
    
    print_success "Pre-flight checks completed"
    return 0
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    
    # Kill any background processes we started
    jobs -p | xargs -r kill
    
    # Log cleanup
    echo "=== Launch Ended: $(date) ===" >> "$LOG_FILE"
}

# Main function
main() {
    local launch_mode="${1:-normal}"
    
    # Handle help
    if [[ "$launch_mode" == "help" || "$launch_mode" == "-h" || "$launch_mode" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    # Set up logging
    setup_logging
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    print_status "Starting $APP_NAME launcher..."
    
    # Handle launch mode
    handle_launch_mode "$launch_mode"
    
    # Run pre-flight checks
    if ! run_preflight_checks; then
        print_error "Pre-flight checks failed. Aborting launch."
        exit 1
    fi
    
    # Launch application
    if ! launch_application; then
        print_error "Application launch failed"
        exit 1
    fi
    
    print_success "Launcher completed successfully"
}

# Handle interruption
trap 'print_error "Launch interrupted"; exit 1' INT TERM

# Run main function
main "$@"