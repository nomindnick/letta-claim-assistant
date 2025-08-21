# Letta Configuration Bug Report

## Executive Summary
Letta v0.10.0 is not respecting agent-specific LLM configurations and is defaulting to OpenAI API calls instead of using the configured Ollama endpoint, causing 500 Internal Server Errors when attempting to send messages to agents.

## Application Context

### Project Overview
The Letta Construction Claim Assistant is a local-first desktop application that analyzes construction claim PDFs using RAG (Retrieval-Augmented Generation) combined with a stateful agent powered by Letta. The application supports three chat modes:
1. **Documents Only (RAG)** - Search and retrieve from vectorized PDFs
2. **Memory Only** - Direct interaction with Letta agent
3. **Combined** - Both document search and memory recall

### Technology Stack
- **Python**: 3.12
- **Letta**: v0.10.0 (server and agent framework)
- **letta-client**: v0.1.257 (Python SDK)
- **LLM Provider**: Ollama (local) with model `gpt-oss:20b`
- **Embedding Model**: `nomic-embed-text` via Ollama
- **Vector Store**: ChromaDB
- **UI Framework**: NiceGUI

### System Architecture
```
User Interface → Backend API → RAG Engine → Letta Adapter
                                    ↓              ↓
                              Vector Store   Letta Server
                                    ↓              ↓
                                 Ollama      Letta Agent
```

## The Bug

### Symptoms
1. When sending messages to a Letta agent via `client.agents.messages.create()`, the server returns HTTP 500
2. Server logs show: `letta.errors.LLMNotFoundError: NOT_FOUND: Resource not found in OpenAI: 404 page not found`
3. The agent IS correctly configured with Ollama settings but Letta server ignores them

### Error Details
```python
# Error when calling
response = await client.agents.messages.create(
    agent_id=agent_id,
    messages=[MessageCreate(role="user", content="Hello")]
)

# Returns
ApiError: headers: {'date': '...', 'server': 'uvicorn', 'content-length': '46', 'content-type': 'application/json'}, 
status_code: 500, 
body: {'detail': 'An internal server error occurred'}

# Server logs show
letta.errors.LLMNotFoundError: NOT_FOUND: Resource not found in OpenAI: 404 page not found
```

## Technical Details

### Agent Configuration (Working)
```python
# Agent successfully created with:
agent_state = await client.agents.create(
    name="claim-assistant-Test Matter",
    llm_config=LlmConfig(
        model="gpt-oss:20b",
        model_endpoint_type="ollama",
        model_endpoint="http://localhost:11434",
        context_window=32768,
        max_tokens=4096
    ),
    embedding_config=EmbeddingConfig(
        embedding_model="nomic-embed-text",
        embedding_endpoint_type="ollama",
        embedding_endpoint="http://localhost:11434"
    )
)

# Verification shows correct config:
agent = await client.agents.retrieve(agent_id)
print(agent.llm_config.model)  # "gpt-oss:20b"
print(agent.llm_config.model_endpoint_type)  # "ollama"
print(agent.llm_config.model_endpoint)  # "http://localhost:11434"
```

### API Calls That Work
```python
# These work fine (no LLM needed):
passages = await client.agents.passages.list(agent_id=agent_id)  # ✓ Works
agent = await client.agents.retrieve(agent_id)  # ✓ Works
```

### API Calls That Fail
```python
# These fail with 500 error:
response = await client.agents.messages.create(...)  # ✗ Fails - tries OpenAI
```

## Attempted Solutions

### 1. Global Configuration File
**Location**: `~/.letta/config`
```ini
[llm]
model = gpt-oss:20b
model_endpoint_type = ollama
model_endpoint = http://localhost:11434

[embedding]
embedding_model = nomic-embed-text
embedding_endpoint_type = ollama
embedding_endpoint = http://localhost:11434
```
**Result**: No effect - still tries OpenAI

### 2. Environment Variables
```bash
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_API_KEY=dummy
letta server --port 8283
```
**Result**: No effect - still tries OpenAI

### 3. Server Start Parameters
```bash
LETTA_LLM_ENDPOINT=http://localhost:11434 \
LETTA_LLM_ENDPOINT_TYPE=ollama \
LETTA_LLM_MODEL=gpt-oss:20b \
letta server --port 8283
```
**Result**: No effect - still tries OpenAI

### 4. Message Format Variations
```python
# Tried MessageCreate object
message = MessageCreate(role="user", content="Hello")

# Tried dict format
message = {"role": "user", "content": "Hello"}

# Tried with TextContent
message = MessageCreate(
    role="user", 
    content=[TextContent(type="text", text="Hello")]
)
```
**Result**: All fail with same error

## Environment Information

### Versions
- Letta: 0.10.0
- letta-client: 0.1.257
- Python: 3.12
- OS: Ubuntu Linux
- Ollama: Running on http://localhost:11434

### Verification Commands
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Verify Letta server is running
curl http://localhost:8283/v1/health/

# Test Ollama directly
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "Hello"
}'
```

## Root Cause Analysis

The issue appears to be in Letta v0.10.0's message processing pipeline:
1. Agent configuration is stored correctly
2. Non-LLM operations (passages.list) work fine
3. When processing messages.create(), Letta server ignores agent's LLM config
4. Server defaults to OpenAI instead of using agent's Ollama configuration
5. Results in 404 from non-existent OpenAI endpoint

## Code References

### Relevant Files
- `app/letta_adapter.py`: Lines 909-1053 (memory_only_chat implementation)
- `app/letta_connection.py`: Connection management and retry logic
- `~/.letta/config`: Global Letta configuration
- `/tmp/letta_server.log`: Server logs showing the error

### Key Code Sections
```python
# Current workaround in app/letta_adapter.py:922-943
# Temporarily disabled memory-only mode with informative message
return {
    "answer": (
        "Memory-only mode is temporarily unavailable due to a known issue in Letta v0.10.0.\n\n"
        "The Letta server is not properly using the agent's Ollama configuration and is "
        "defaulting to OpenAI instead. This is a bug in Letta that needs to be fixed.\n\n"
        "As a workaround, please use 'Combined' mode which will search both documents and "
        "memory, or 'Documents Only' mode for RAG-only queries."
    ),
    "sources": [],
    "memory_used": False,
    "error": "Letta v0.10.0 configuration bug"
}
```

## Potential Solutions to Investigate

1. **Check Letta GitHub Issues**: Search for similar issues in letta-ai/letta repository
2. **Version Compatibility**: Try different Letta versions (0.9.x or 0.11.x if available)
3. **Direct API Override**: Investigate if there's a way to pass LLM config in the message.create() call
4. **Custom Tool Approach**: Instead of messages.create(), use agent tools API
5. **Letta Server Configuration**: Check if server has separate config from agents
6. **OpenAI Compatibility Layer**: Configure Ollama to fully emulate OpenAI API at system level

## Reproduction Steps

1. Install dependencies:
```bash
pip install letta==0.10.0 letta-client==0.1.257
```

2. Start Ollama with model:
```bash
ollama run gpt-oss:20b
```

3. Start Letta server:
```bash
letta server --port 8283
```

4. Run test script:
```python
import asyncio
from letta_client import AsyncLetta
from letta_client.types import MessageCreate, LlmConfig

async def reproduce_bug():
    client = AsyncLetta(base_url='http://localhost:8283')
    
    # Create agent with Ollama config
    agent = await client.agents.create(
        name="test-agent",
        llm_config=LlmConfig(
            model="gpt-oss:20b",
            model_endpoint_type="ollama",
            model_endpoint="http://localhost:11434"
        )
    )
    
    # This will fail with 500 error
    response = await client.agents.messages.create(
        agent_id=agent.id,
        messages=[MessageCreate(role="user", content="Hello")]
    )
    
asyncio.run(reproduce_bug())
```

## Impact

- **Memory-only mode**: Completely non-functional
- **Combined mode**: Memory recall works, but agent learning/interaction disabled
- **RAG-only mode**: Unaffected
- **User Experience**: Limited to document search, no stateful agent capabilities

## Success Criteria

A solution is successful if:
1. `client.agents.messages.create()` returns successfully
2. Response is generated using Ollama (not OpenAI)
3. Agent maintains conversation state
4. No 500 errors or OpenAI 404s in logs