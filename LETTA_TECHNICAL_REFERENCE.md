# Letta Technical Reference

**Version:** Letta 0.10.0 with letta-client 0.1.257 (Stable)  
**Last Updated:** 2025-08-18  
**Purpose:** Comprehensive technical guide for integrating Letta's persistent agent memory system

> **Important:** This project uses Letta v0.10.0, which is the latest stable version with a working passages API. Version 0.11.x contains a critical bug that breaks memory storage operations.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Server Setup](#server-setup)
3. [Client Connection](#client-connection)
4. [Agent Management](#agent-management)
5. [Memory Operations](#memory-operations)
6. [LLM Provider Configuration](#llm-provider-configuration)
7. [Migration from LocalClient](#migration-from-localclient)
8. [Error Handling & Fallback](#error-handling--fallback)
9. [Performance Considerations](#performance-considerations)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Modern Letta Architecture (v0.11+)

The modern Letta architecture consists of three main components:

```
┌─────────────────┐
│  Application    │
│  (Your Code)    │
└────────┬────────┘
         │ HTTP/WebSocket
┌────────▼────────┐
│  Letta Client   │
│ (letta_client)  │
└────────┬────────┘
         │ REST API
┌────────▼────────┐
│  Letta Server   │
│  (letta server) │
└────────┬────────┘
         │
┌────────▼────────┐
│    Storage      │
│  (SQLite/PG)    │
└─────────────────┘
```

### Key Changes from Previous Versions

| Old (LocalClient) | New (Server-based) |
|-------------------|-------------------|
| `from letta import LocalClient` | `from letta_client import Letta` |
| Direct file access | REST API communication |
| Embedded database | Server-managed database |
| Synchronous only | Sync and Async support |
| Single process | Client-server separation |

---

## Server Setup

### Starting Letta Server

#### Option 1: Command Line (Simplest)

```bash
# Start server on default port (8080)
letta server

# Start on custom port
letta server --port 8283

# Start with debugging
letta server --debug --port 8283
```

#### Option 2: Python Subprocess (Programmatic)

```python
import subprocess
import time
import requests

class LettaServerManager:
    def __init__(self, port=8283, host="localhost"):
        self.port = port
        self.host = host
        self.process = None
    
    def start(self):
        """Start Letta server as subprocess."""
        cmd = ["letta", "server", "--port", str(self.port), "--host", self.host]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait for server to be ready
        self._wait_for_server()
        
    def _wait_for_server(self, timeout=30):
        """Wait for server to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/health")
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(0.5)
        raise TimeoutError("Server failed to start")
    
    def stop(self):
        """Stop the server gracefully."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
```

#### Option 3: Docker Container

```yaml
# docker-compose.yml
version: '3.8'
services:
  letta:
    image: letta/letta:latest
    ports:
      - "8283:8283"
    volumes:
      - ./letta_data:/data
    environment:
      - LETTA_PORT=8283
      - LETTA_HOST=0.0.0.0
    command: server --port 8283 --host 0.0.0.0
```

### Server Configuration

Default configuration location: `~/.letta/config`

```yaml
# Example server configuration
server:
  host: localhost
  port: 8283
  cors_origins: ["http://localhost:*"]
  
database:
  type: sqlite
  path: ~/.letta/letta.db
  
logging:
  level: INFO
  file: ~/.letta/logs/server.log
```

---

## Client Connection

### Basic Client Setup

```python
from letta_client import Letta, AsyncLetta

# Synchronous client
client = Letta(base_url="http://localhost:8283")

# Asynchronous client (recommended for async applications)
async_client = AsyncLetta(base_url="http://localhost:8283")
```

### Connection with Error Handling

```python
import asyncio
from letta_client import AsyncLetta
from letta_client.errors import ClientError

class LettaConnection:
    def __init__(self, base_url="http://localhost:8283"):
        self.base_url = base_url
        self.client = None
        
    async def connect(self, retries=3):
        """Connect to Letta server with retry logic."""
        for attempt in range(retries):
            try:
                self.client = AsyncLetta(base_url=self.base_url)
                # Test connection
                await self.client.health.health_check()
                return True
            except Exception as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        return False
    
    async def ensure_connected(self):
        """Ensure client is connected before operations."""
        if not self.client:
            await self.connect()
        return self.client is not None
```

### Connection Pooling

```python
from contextlib import asynccontextmanager

class LettaClientPool:
    _instance = None
    _client = None
    
    @classmethod
    async def get_client(cls):
        """Get or create singleton client."""
        if cls._client is None:
            cls._client = AsyncLetta(base_url="http://localhost:8283")
        return cls._client
    
    @classmethod
    @asynccontextmanager
    async def client_session(cls):
        """Context manager for client operations."""
        client = await cls.get_client()
        try:
            yield client
        finally:
            # Cleanup if needed
            pass
```

---

## Agent Management

### Creating an Agent

```python
from letta_client import AsyncLetta
from letta_client.types import AgentState, LlmConfig, EmbeddingConfig

async def create_agent(client: AsyncLetta, matter_name: str):
    """Create a new agent for a construction claim matter."""
    
    # Configure LLM (using Ollama)
    llm_config = LlmConfig(
        model="gpt-oss:20b",
        model_endpoint_type="ollama",
        model_endpoint="http://localhost:11434",
        context_window=8192
    )
    
    # Configure embeddings
    embedding_config = EmbeddingConfig(
        embedding_model="nomic-embed-text",
        embedding_endpoint_type="ollama",
        embedding_endpoint="http://localhost:11434",
        embedding_dim=768
    )
    
    # Create agent
    agent = await client.agents.create_agent(
        name=f"claim-assistant-{matter_name}",
        description=f"Construction claims analyst for {matter_name}",
        system=f"""You are a construction claims analyst assistant for the matter: {matter_name}.
        
Your role is to:
- Analyze construction documents, contracts, and claims
- Track entities, events, issues, and facts
- Maintain memory of important information
- Provide insights about causation, responsibility, and damages
- Remember key dates, parties, and technical details""",
        llm_config=llm_config,
        embedding_config=embedding_config,
        # Memory configuration
        memory_blocks=[
            {
                "label": "human",
                "value": f"Construction attorney working on {matter_name}"
            },
            {
                "label": "persona",
                "value": "Expert construction claims analyst with legal knowledge"
            }
        ]
    )
    
    return agent.id
```

### Loading an Existing Agent

```python
async def load_agent(client: AsyncLetta, agent_id: str):
    """Load an existing agent by ID."""
    try:
        agent = await client.agents.get_agent(agent_id)
        return agent
    except Exception as e:
        print(f"Agent not found: {e}")
        return None
```

### Updating Agent Configuration

```python
async def update_agent(client: AsyncLetta, agent_id: str, updates: dict):
    """Update agent configuration."""
    agent = await client.agents.update_agent(
        agent_id=agent_id,
        **updates
    )
    return agent
```

### Deleting an Agent

```python
async def delete_agent(client: AsyncLetta, agent_id: str):
    """Delete an agent and its memory."""
    await client.agents.delete_agent(agent_id)
```

---

## Memory Operations

### Sending Messages (Storing Interactions)

```python
async def store_interaction(client: AsyncLetta, agent_id: str, user_query: str, context: str = None):
    """Store user interaction in agent memory."""
    
    # Prepare message with context if available
    message = user_query
    if context:
        message = f"Context: {context}\n\nQuery: {user_query}"
    
    # Send message to agent
    response = await client.messages.send_message(
        agent_id=agent_id,
        role="user",
        message=message,
        stream=False
    )
    
    return response
```

### Memory Recall

```python
async def recall_memory(client: AsyncLetta, agent_id: str, query: str, limit: int = 10):
    """Search agent's memory for relevant information."""
    
    # Search archival memory
    results = await client.agents.search_archival_memory(
        agent_id=agent_id,
        query=query,
        limit=limit
    )
    
    return results
```

### Archival Memory Management

```python
async def insert_archival_memory(client: AsyncLetta, agent_id: str, content: str):
    """Insert content into agent's archival memory."""
    
    passage = await client.agents.insert_archival_memory(
        agent_id=agent_id,
        memory=content
    )
    return passage

async def get_archival_memory(client: AsyncLetta, agent_id: str, limit: int = 100):
    """Retrieve all archival memory."""
    
    memories = await client.agents.get_archival_memory(
        agent_id=agent_id,
        limit=limit
    )
    return memories
```

### Core Memory Updates

```python
async def update_core_memory(client: AsyncLetta, agent_id: str, block_label: str, new_value: str):
    """Update agent's core memory blocks."""
    
    # Get current agent state
    agent = await client.agents.get_agent(agent_id)
    
    # Update specific memory block
    memory_blocks = agent.memory.blocks
    for block in memory_blocks:
        if block.label == block_label:
            block.value = new_value
            break
    
    # Update agent with new memory
    updated_agent = await client.agents.update_agent(
        agent_id=agent_id,
        memory_blocks=memory_blocks
    )
    
    return updated_agent
```

---

## LLM Provider Configuration

### Ollama Configuration

```python
from letta_client.types import LlmConfig

def create_ollama_config(model_name: str = "gpt-oss:20b"):
    """Create Ollama LLM configuration."""
    return LlmConfig(
        model=model_name,
        model_endpoint_type="ollama",
        model_endpoint="http://localhost:11434",
        context_window=8192,
        # Ollama-specific settings
        temperature=0.7,
        top_p=0.9,
        max_tokens=2000
    )
```

### Gemini API Configuration

```python
import os

def create_gemini_config(api_key: str = None):
    """Create Gemini API configuration."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    
    return LlmConfig(
        model="gemini-1.5-flash",
        model_endpoint_type="google_ai",
        model_endpoint="https://generativelanguage.googleapis.com",
        api_key=api_key,
        context_window=32768,
        temperature=0.7
    )
```

### OpenAI Configuration

```python
def create_openai_config(api_key: str = None):
    """Create OpenAI API configuration."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    return LlmConfig(
        model="gpt-4o-mini",
        model_endpoint_type="openai",
        model_endpoint="https://api.openai.com/v1",
        api_key=api_key,
        context_window=16384,
        temperature=0.7
    )
```

### Dynamic Provider Selection

```python
class LLMProviderManager:
    @staticmethod
    def get_llm_config(provider: str, **kwargs):
        """Get LLM configuration based on provider."""
        providers = {
            "ollama": create_ollama_config,
            "gemini": create_gemini_config,
            "openai": create_openai_config
        }
        
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        return providers[provider](**kwargs)
```

---

## Migration from LocalClient

### Old LocalClient Code

```python
# OLD CODE - No longer works
from letta import LocalClient

client = LocalClient()
agent_state = client.create_agent(
    name="test-agent",
    persona="Assistant persona",
    human="Human description"
)
```

### New Server-based Code

```python
# NEW CODE - Modern approach
from letta_client import AsyncLetta

async def migrate_to_new_api():
    # Start server first (separately or programmatically)
    # letta server --port 8283
    
    # Connect to server
    client = AsyncLetta(base_url="http://localhost:8283")
    
    # Create agent with new API
    agent = await client.agents.create_agent(
        name="test-agent",
        system="Assistant persona",
        memory_blocks=[
            {"label": "persona", "value": "Assistant persona"},
            {"label": "human", "value": "Human description"}
        ],
        llm_config=create_ollama_config()
    )
    
    return agent.id
```

### Migration Checklist

1. **Install new packages**:
   ```bash
   pip install letta>=0.11.0 letta-client>=0.1.0
   ```

2. **Start Letta server**:
   - Run `letta server` or implement programmatic startup

3. **Update imports**:
   - Replace `from letta import LocalClient` with `from letta_client import Letta/AsyncLetta`

4. **Update API calls**:
   - `create_agent()` → `agents.create_agent()`
   - `user_message()` → `messages.send_message()`
   - `get_archival_memory()` → `agents.get_archival_memory()`
   - `insert_archival_memory()` → `agents.insert_archival_memory()`

5. **Handle async operations**:
   - Add `async/await` for all client operations

6. **Update configuration**:
   - Add LLM and embedding configurations explicitly

---

## Error Handling & Fallback

### Comprehensive Error Handling

```python
from letta_client.errors import (
    ClientError,
    BadRequestError,
    NotFoundError,
    InternalServerError
)

class LettaErrorHandler:
    @staticmethod
    async def safe_operation(operation, fallback_value=None):
        """Execute operation with comprehensive error handling."""
        try:
            return await operation()
        except NotFoundError as e:
            print(f"Resource not found: {e}")
            return fallback_value
        except BadRequestError as e:
            print(f"Invalid request: {e}")
            return fallback_value
        except InternalServerError as e:
            print(f"Server error: {e}")
            # Could retry or use fallback
            return fallback_value
        except ClientError as e:
            print(f"Client error: {e}")
            return fallback_value
        except Exception as e:
            print(f"Unexpected error: {e}")
            return fallback_value
```

### Fallback Mode Implementation

```python
class LettaWithFallback:
    def __init__(self, base_url="http://localhost:8283"):
        self.base_url = base_url
        self.client = None
        self.fallback_mode = False
        
    async def initialize(self):
        """Initialize with fallback detection."""
        try:
            self.client = AsyncLetta(base_url=self.base_url)
            await self.client.health.health_check()
            self.fallback_mode = False
        except:
            self.fallback_mode = True
            print("Letta unavailable, using fallback mode")
    
    async def recall_memory(self, query: str):
        """Recall with fallback to empty results."""
        if self.fallback_mode:
            return []
        
        try:
            return await self.client.agents.search_archival_memory(
                agent_id=self.agent_id,
                query=query
            )
        except:
            return []
```

---

## Performance Considerations

### Connection Reuse

```python
# Singleton pattern for connection reuse
class LettaClientSingleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = AsyncLetta(base_url="http://localhost:8283")
        return cls._instance
```

### Batch Operations

```python
async def batch_insert_memories(client: AsyncLetta, agent_id: str, memories: list):
    """Insert multiple memories efficiently."""
    tasks = []
    for memory in memories:
        task = client.agents.insert_archival_memory(
            agent_id=agent_id,
            memory=memory
        )
        tasks.append(task)
    
    # Execute in parallel
    results = await asyncio.gather(*tasks)
    return results
```

### Caching

```python
from functools import lru_cache
import hashlib

class MemoryCache:
    def __init__(self):
        self.cache = {}
    
    def get_cache_key(self, agent_id: str, query: str):
        """Generate cache key."""
        return hashlib.md5(f"{agent_id}:{query}".encode()).hexdigest()
    
    async def recall_with_cache(self, client, agent_id, query):
        """Recall with caching."""
        cache_key = self.get_cache_key(agent_id, query)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = await client.agents.search_archival_memory(
            agent_id=agent_id,
            query=query
        )
        
        self.cache[cache_key] = results
        return results
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Server Won't Start

**Issue**: `letta server` fails to start

**Solutions**:
- Check if port is already in use: `lsof -i :8283`
- Try different port: `letta server --port 8284`
- Check logs: `~/.letta/logs/server.log`

#### 2. Client Connection Fails

**Issue**: Cannot connect to server

**Solutions**:
```python
# Check server health
import requests
response = requests.get("http://localhost:8283/health")
print(response.json())

# Verify base URL
client = AsyncLetta(base_url="http://localhost:8283")  # Include http://
```

#### 3. Agent Creation Fails

**Issue**: Error creating agent

**Solutions**:
- Verify LLM configuration is correct
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Ensure model exists: `ollama list`

#### 4. Memory Operations Timeout

**Issue**: Memory operations are slow

**Solutions**:
- Implement pagination for large recalls
- Use more specific search queries
- Consider memory pruning for large agents

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Letta-specific logging
logger = logging.getLogger("letta_client")
logger.setLevel(logging.DEBUG)
```

### Health Checks

```python
async def comprehensive_health_check(base_url="http://localhost:8283"):
    """Perform comprehensive system health check."""
    checks = {
        "server": False,
        "ollama": False,
        "database": False
    }
    
    # Check Letta server
    try:
        client = AsyncLetta(base_url=base_url)
        await client.health.health_check()
        checks["server"] = True
    except:
        pass
    
    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        checks["ollama"] = response.status_code == 200
    except:
        pass
    
    return checks
```

---

## Appendix: Complete Example

```python
"""
Complete example demonstrating Letta integration for construction claims.
"""

import asyncio
from letta_client import AsyncLetta
from letta_client.types import LlmConfig, EmbeddingConfig

class ConstructionClaimsAgent:
    def __init__(self, matter_name: str):
        self.matter_name = matter_name
        self.client = None
        self.agent_id = None
        
    async def initialize(self):
        """Initialize client and agent."""
        # Connect to server
        self.client = AsyncLetta(base_url="http://localhost:8283")
        
        # Create or load agent
        self.agent_id = await self._get_or_create_agent()
        
    async def _get_or_create_agent(self):
        """Get existing agent or create new one."""
        # Try to find existing agent
        agents = await self.client.agents.list_agents()
        for agent in agents:
            if agent.name == f"claim-{self.matter_name}":
                return agent.id
        
        # Create new agent
        llm_config = LlmConfig(
            model="gpt-oss:20b",
            model_endpoint_type="ollama",
            model_endpoint="http://localhost:11434"
        )
        
        agent = await self.client.agents.create_agent(
            name=f"claim-{self.matter_name}",
            system=f"Construction claims analyst for {self.matter_name}",
            llm_config=llm_config
        )
        
        return agent.id
    
    async def analyze_query(self, query: str):
        """Analyze user query with memory context."""
        # Recall relevant memories
        memories = await self.client.agents.search_archival_memory(
            agent_id=self.agent_id,
            query=query,
            limit=5
        )
        
        # Send message with context
        response = await self.client.messages.send_message(
            agent_id=self.agent_id,
            role="user",
            message=query
        )
        
        return response
    
    async def store_fact(self, fact: str):
        """Store important fact in memory."""
        await self.client.agents.insert_archival_memory(
            agent_id=self.agent_id,
            memory=fact
        )

# Usage
async def main():
    agent = ConstructionClaimsAgent("Foundation-Failure-2024")
    await agent.initialize()
    
    # Store some facts
    await agent.store_fact("ABC Construction was the general contractor")
    await agent.store_fact("Foundation failure occurred on 2024-02-14")
    
    # Analyze query
    response = await agent.analyze_query("What caused the foundation failure?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Next Steps

1. **Test server connectivity** using the POC scripts
2. **Validate Ollama integration** with different models
3. **Implement migration** for existing LettaAdapter
4. **Add production features** like monitoring and logging
5. **Create UI indicators** for memory operations

This reference will be continuously updated as we discover new patterns and best practices during implementation.