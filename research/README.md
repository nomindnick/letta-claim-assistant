# Letta Integration Research & POC Scripts

This directory contains proof-of-concept scripts for testing and validating the Letta integration with the construction claims assistant.

## Overview

These scripts demonstrate and test:
- Server connectivity and management
- Ollama integration for local LLMs
- Memory operations (storage, retrieval, search)
- Migration from old LocalClient to new API
- External API integration (Gemini, OpenAI, etc.)

## Prerequisites

1. **Python Environment**
   ```bash
   # Activate virtual environment
   source .venv/bin/activate
   
   # Ensure Letta is installed
   pip install letta>=0.11.0 letta-client>=0.1.0
   ```

2. **Ollama** (for local LLM)
   ```bash
   # Start Ollama service
   ollama serve
   
   # Pull required models
   ollama pull gpt-oss:20b
   ollama pull nomic-embed-text
   ```

3. **Letta Server**
   ```bash
   # Start Letta server (runs on port 8283 by default)
   letta server --port 8283
   ```

## Test Scripts

### 1. Server Connectivity Test (`test_letta_server.py`)
Tests basic server operations including startup, health checks, and agent creation.

```bash
python research/test_letta_server.py
```

**What it tests:**
- Server startup and shutdown
- Health endpoint connectivity
- Client connection (sync and async)
- Agent creation and deletion
- Basic memory operations

### 2. Ollama Integration Test (`test_letta_ollama.py`)
Validates integration with Ollama for local LLM inference.

```bash
python research/test_letta_ollama.py
```

**What it tests:**
- Ollama connectivity
- Available model detection
- Agent creation with Ollama models
- Generation quality
- Model switching
- Embedding operations

### 3. Memory Operations Test (`test_letta_memory.py`)
Comprehensive testing of memory storage and retrieval.

```bash
python research/test_letta_memory.py
```

**What it tests:**
- Archival memory storage
- Memory retrieval and search
- Semantic similarity search
- Core memory updates
- Memory persistence
- Performance benchmarks
- Conversation-based memory

### 4. Migration Test (`test_letta_migration.py`)
Demonstrates migration from LocalClient to modern API.

```bash
python research/test_letta_migration.py
```

**What it tests:**
- Old vs new API patterns
- Compatibility layer creation
- Data migration strategies
- Fallback mechanisms
- Migration readiness assessment

### 5. Gemini Integration Test (`test_letta_gemini.py`)
Tests integration with Google Gemini API (requires API key).

```bash
# Set API key first
export GEMINI_API_KEY='your-key-here'

# Run test
python research/test_letta_gemini.py
```

**What it tests:**
- Gemini API connectivity
- Agent creation with Gemini
- Generation quality
- Cost tracking
- Fallback to local models

## Running All Tests

Use the provided test runner script:

```bash
# Run all tests
./research/run_tests.sh

# The script will:
# 1. Check prerequisites
# 2. Start Letta server if needed
# 3. Run each test in sequence
# 4. Provide a summary of results
```

## Configuration Files

The `config/` directory contains templates for:

1. **`letta_server_config.yaml`** - Server configuration
2. **`docker-compose.yml`** - Container deployment
3. **`client_config.json`** - Client settings

Copy and customize these for your environment:

```bash
# For local development
cp config/client_config.json ~/.letta-claim/client_config.json

# Edit as needed
vim ~/.letta-claim/client_config.json
```

## Common Issues & Solutions

### Server Won't Start
```bash
# Check if port is in use
lsof -i :8283

# Try different port
letta server --port 8284
```

### Ollama Not Found
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start service
ollama serve
```

### Import Errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### API Key Issues (Gemini)
```bash
# Set environment variable
export GEMINI_API_KEY='your-key-here'

# Or pass as argument
python research/test_letta_gemini.py --api-key 'your-key-here'
```

## Next Steps

After validating the POC scripts:

1. **Review `LETTA_TECHNICAL_REFERENCE.md`** for detailed implementation guidance
2. **Migrate `app/letta_adapter.py`** to use new API
3. **Implement server lifecycle management** in the main application
4. **Add UI indicators** for memory operations
5. **Test with real construction claim data**

## Performance Considerations

- **Server startup**: ~5-10 seconds
- **Agent creation**: ~2-3 seconds
- **Memory insertion**: ~50-100ms per item
- **Memory search**: ~100-500ms depending on size
- **Generation**: ~1-5 seconds depending on model and prompt

## Security Notes

- Keep API keys secure (use environment variables)
- Run Letta server locally for sensitive data
- Enable authentication for production deployments
- Monitor and limit external API usage
- Implement proper consent mechanisms for external APIs

## Support

For issues or questions:
1. Check the test output for specific error messages
2. Review `LETTA_TECHNICAL_REFERENCE.md` for detailed API documentation
3. Consult the official Letta documentation
4. Check server and client logs for debugging