# Stateful Agent-First Architecture Specification

**Status**: Phase 1 Complete ✅ (2025-08-21)  
**Current Phase**: Ready for Phase 2 - UI Integration  
**Test Results**: All Phase 1 tests passing (6/6)

## Vision Statement
Transform the Letta Construction Claim Assistant from a RAG-first system with passive memory to an agent-first system where a stateful Letta agent serves as the primary interface, actively learning about cases over time and intelligently leveraging RAG search as a tool when needed.

## Current Implementation Context

### Existing System Components
The application currently has a functional RAG-first system with:
- **PDF Ingestion Pipeline**: OCR, text extraction, chunking (app/ingest.py)
- **Vector Store**: ChromaDB per-matter isolation (app/vectors.py)
- **RAG Engine**: Retrieval and generation pipeline (app/rag.py)
- **Letta Integration**: Memory storage adapter (app/letta_adapter.py)
- **Server Management**: Letta server lifecycle (app/letta_server.py)
- **Connection Management**: Retry logic and health monitoring (app/letta_connection.py)
- **Provider Bridge**: LLM provider configuration (app/letta_provider_bridge.py)

### Known Issues and Constraints

#### Letta Configuration Bug (v0.10.0)
- **Issue**: Letta server not respecting agent-specific LLM configurations when processing messages
- **Symptom**: Server returns HTTP 500 with "Resource not found in OpenAI: 404 page not found" despite agent configured for Ollama
- **Root Cause**: Server requires `OLLAMA_BASE_URL` environment variable to enable Ollama provider system-wide
- **Current Status**: Documented in `letta-config-bug.md` with workarounds
- **Fix Required**: Update server startup in `app/letta_server.py` line 234-236 to set: `env["OLLAMA_BASE_URL"] = "http://localhost:11434"`

#### Agent Model Configuration
- **Technical Capability**: Letta DOES support modifying agents after creation via `agents.modify()` endpoint
- **Product Decision**: We choose to lock provider selection at matter creation for consistency
- **Rationale**: Prevents embedding dimension mismatches and ensures reproducible results
- **Migration Path**: If needed, implement provider switching with re-embedding of all documents

## Architectural Transformation

### Current Architecture (RAG-First)
```
User Input → RAG Pipeline → [Doc Retrieval + Memory Recall] → LLM Generation → Response
                                            ↑                         ↓
                                    Letta (passive storage) ←────────┘
```

**Characteristics:**
- Stateless per query
- Memory is just retrieved context
- Always searches documents
- No conversation continuity
- No progressive learning

### Proposed Architecture (Agent-First)
```
User Input → Letta Agent → [Reasoning & Memory] → Tool Selection → Response
                 ↑              ↓                      ↓
                 └── Stateful ──┘                RAG Search Tool
                 Conversation                    (when needed)
                    Loop
```

**Characteristics:**
- Stateful conversation
- Agent decides when to search
- Learns and remembers
- Maintains context
- Progressive understanding

## Core Components

### 1. Letta Agent as Primary Interface

#### Agent Configuration Requirements

**Provider Selection at Matter Creation:**
- User selects LLM provider/model when creating a new matter
- This selection is permanent for the agent's lifetime
- Configuration stored with matter for consistency

**Core Agent Configuration:**
```python
{
    "name": "claim-assistant-{matter_name}",
    "system_prompt": """You are a legal assistant specializing in construction claims. 
    You have access to case documents and maintain memory of all interactions.
    You learn progressively about the case and can search documents when needed.
    Your goal is to become increasingly knowledgeable about this specific matter.""",
    
    "memory_blocks": [
        {
            "label": "human",
            "value": "Construction attorney working on {matter_name}",
            "limit": 2000
        },
        {
            "label": "persona",
            "value": "Expert construction claims analyst with legal knowledge",
            "limit": 2000
        },
        {
            "label": "case_facts",
            "value": "",
            "description": "Key verified facts with citations. Update only with document-backed information.",
            "limit": 4000
        },
        {
            "label": "entities",
            "value": "",
            "description": "People, companies, and organizations involved in the claim",
            "limit": 3000
        },
        {
            "label": "timeline",
            "value": "",
            "description": "Chronological events and deadlines in YYYY-MM-DD format",
            "limit": 3000
        }
    ],
    
    "tools": [
        "search_documents"  # Start with basic RAG search tool only
    ]
}
```

### 2. Tool Definitions

#### Tool Registration Pattern (Letta SDK)
```python
# Tool definition with proper Letta SDK pattern
def create_search_documents_tool():
    """Create and register the search_documents tool with Letta."""
    
    tool_definition = {
        "name": "search_documents",
        "description": "Search case documents for specific information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
    
    # Tool implementation
    async def search_documents_impl(agent_state, query: str, k: int = 5) -> dict:
        """
        Search implementation with context passing and structured output.
        
        Args:
            agent_state: Letta agent state containing metadata
            query: Search query string
            k: Number of results to return
        
        Returns:
            Structured JSON response
        """
        # Extract matter context from agent metadata
        matter_id = agent_state.metadata.get("matter_id")
        if not matter_id:
            return {
                "status": "error",
                "message": "No matter context available"
            }
        
        # Use existing RAG infrastructure
        from app.rag import RAGEngine
        from app.matters import matter_manager
        
        matter = await matter_manager.get_matter(matter_id)
        rag_engine = RAGEngine(matter)
        
        # Perform search
        try:
            search_results = await rag_engine.search(
                query=query,
                k=k
            )
            
            # Return structured JSON
            return {
                "status": "success",
                "results_count": len(search_results),
                "results": [
                    {
                        "doc_name": r.metadata["doc_name"],
                        "page_start": r.metadata["page_start"],
                        "page_end": r.metadata.get("page_end", r.metadata["page_start"]),
                        "score": r.score,
                        "snippet": r.text[:300] + "..." if len(r.text) > 300 else r.text,
                        "citation": f"[{r.metadata['doc_name']} p.{r.metadata['page_start']}]"
                    }
                    for r in search_results
                ]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}"
            }
    
    return tool_definition, search_documents_impl

# Register tool with Letta
async def register_tools_with_agent(client, agent_id: str, matter_id: str):
    """Register tools with a Letta agent."""
    
    # Create tool
    tool_def, tool_impl = create_search_documents_tool()
    
    # Upsert tool to Letta server
    tool = await client.tools.upsert(
        name=tool_def["name"],
        description=tool_def["description"],
        parameters=tool_def["parameters"],
        source_code=inspect.getsource(tool_impl),
        return_char_limit=10000  # Limit response size
    )
    
    # Attach tool to agent
    await client.agents.tools.attach(
        agent_id=agent_id,
        tool_name=tool.name
    )
    
    # Set agent metadata for context
    await client.agents.modify(
        agent_id=agent_id,
        metadata={"matter_id": matter_id}
    )
```

#### Future Tool Extensions (Not Phase 1)
```python
# These tools will be added in future phases after core functionality works
# - search_by_date_range: Filter documents by date
# - extract_from_document: Extract specific sections from a document
# - summarize_document: Generate document summaries
# - find_contradictions: Identify conflicting information
# - timeline_extraction: Build timeline from documents

# Advanced sub-agents for specialized tasks will be considered after
# the primary agent with basic RAG search is fully operational
```

### 3. Memory Management Strategy

#### Memory Block Limits and Constraints
Letta memory blocks have token limits that must be managed:

```python
MEMORY_BLOCK_LIMITS = {
    "human": 2000,        # User context
    "persona": 2000,      # Agent identity
    "case_facts": 4000,   # Core facts about the case
    "entities": 3000,     # People and organizations
    "timeline": 3000,     # Chronological events
    "conversations": 2000 # Recent interaction summary
}

# Total memory budget: ~16,000 tokens
```

#### Overflow Handling Strategy
```python
class MemoryManager:
    """Manage memory blocks to prevent overflow."""
    
    async def check_memory_usage(self, agent_id: str) -> dict:
        """Monitor memory block token counts."""
        blocks = await client.agents.memory.list(agent_id)
        usage = {}
        for block in blocks:
            token_count = estimate_tokens(block.value)
            usage[block.label] = {
                "used": token_count,
                "limit": MEMORY_BLOCK_LIMITS[block.label],
                "percentage": (token_count / MEMORY_BLOCK_LIMITS[block.label]) * 100
            }
        return usage
    
    async def compress_memory_block(self, agent_id: str, block_label: str):
        """Compress a memory block when approaching limit."""
        if block_label == "case_facts":
            # Summarize older facts, keep recent ones detailed
            await self._summarize_old_facts(agent_id)
        elif block_label == "conversations":
            # Archive old conversations, keep last 10
            await self._archive_old_conversations(agent_id)
        elif block_label == "timeline":
            # Consolidate duplicate events
            await self._consolidate_timeline(agent_id)
    
    async def archive_to_passages(self, agent_id: str, content: str, metadata: dict):
        """Archive overflow content to passages for retrieval."""
        await client.agents.passages.create(
            agent_id=agent_id,
            text=content,
            metadata={
                "type": "archived_memory",
                "original_block": metadata["block_label"],
                "archived_at": datetime.now().isoformat()
            }
        )
```

#### Memory Update Guidelines
The agent's system prompt should include explicit memory management instructions:

```python
MEMORY_MANAGEMENT_PROMPT = """
When updating memory blocks, follow these guidelines:

1. CASE_FACTS Block (4000 token limit):
   - Store only verified facts with citations: "Contract date: 2023-06-15 [Contract.pdf p.12]"
   - When approaching limit, summarize older facts
   - Prioritize legally significant facts

2. ENTITIES Block (3000 token limit):
   - Format: "Role: Name (Company) - key responsibilities"
   - Group related entities together
   - Remove duplicates immediately

3. TIMELINE Block (3000 token limit):
   - Use ISO format: "2023-06-15: Event description [Source]"
   - Consolidate related events
   - Keep only significant dates

4. CONVERSATIONS Block (2000 token limit):
   - Summarize older conversations
   - Keep full detail for last 5 interactions
   - Archive important decisions to case_facts

When any block reaches 80% capacity, actively compress and reorganize.
"""
```

#### Archival Memory Strategy
For long-running matters with extensive history:

```python
async def implement_archival_memory(agent_id: str):
    """Use Letta's passages API for overflow content."""
    
    # Check all memory blocks
    usage = await check_memory_usage(agent_id)
    
    for block_label, stats in usage.items():
        if stats["percentage"] > 80:
            # Extract content to archive
            block = await client.agents.memory.get(agent_id, block_label)
            
            # Split into current and archival
            current_content, archival_content = split_by_recency(
                block.value, 
                keep_recent_percentage=0.5
            )
            
            # Archive older content
            await client.agents.passages.create(
                agent_id=agent_id,
                text=archival_content,
                metadata={
                    "source": "memory_overflow",
                    "block": block_label,
                    "archived_date": datetime.now().isoformat()
                }
            )
            
            # Update block with compressed content
            await client.agents.memory.update(
                agent_id=agent_id,
                block_label=block_label,
                value=current_content
            )
```

### 4. Implementation Phases (Phase 0 & 1 Complete)

#### Phase 0: Technical Discovery & Validation ✅ COMPLETE (2025-08-21)
**Purpose**: Validate Letta v0.10.0 capabilities and establish baseline functionality before implementation.

**Discovery Tasks:**
- [x] Test Letta tool registration mechanisms with simple test tool
- [x] Verify memory block behavior with content exceeding limits
- [x] Test agent creation with Ollama, Gemini, and OpenAI providers
- [x] Document actual Letta database schema and persistence mechanism
- [x] Validate passages API for memory insertion and retrieval
- [x] Test message streaming endpoint for UI integration
- [x] Verify health check and monitoring endpoints
- [x] Check tool timeout and failure recovery mechanisms

**Technical Validation:**
- [x] Confirm OLLAMA_BASE_URL environment variable requirement
- [x] Test agent modification capabilities (agents.modify endpoint)
- [x] Verify context passing to tools via agent metadata
- [x] Test memory block token limits and overflow behavior
- [x] Validate tool return size limits (return_char_limit)
- [x] Check heartbeat and multi-step execution patterns

**Documentation Output:**
- [x] Letta API compatibility matrix (`docs/letta_api_compatibility_matrix.md`)
- [x] Provider configuration requirements documented
- [x] Memory limits and constraints verified
- [x] Tool registration patterns that work identified
- [x] Known issues and workarounds documented

**Implementation Summary:**
- Fixed critical OLLAMA_BASE_URL environment variable issue in `letta_server.py`
- Created comprehensive validation test suite (`tests/phase_0_validation.py`)
- Verified all Letta v0.10.0 capabilities needed for transformation
- Created simplified validation script (`phase_0_validation_simple.py`)

**Key Technical Discoveries:**
1. **OLLAMA_BASE_URL Fix Applied**: Server now properly sets environment variable for Ollama provider recognition
2. **Import Changes**: Use `RESTClient` instead of `AsyncLetta` (doesn't exist)
3. **Synchronous Client**: RESTClient is synchronous, not async
4. **Provider Lock-in**: Design decision to permanently tie agents to specific providers (prevents embedding mismatches)
5. **Dependencies**: sqlite-vec required for Letta ORM

**Validation Results (All Tests Passed - 5/5):**
- ✅ OLLAMA_BASE_URL environment variable properly set
- ✅ Letta v0.10.0 installed and importable
- ✅ Server startup/shutdown lifecycle working
- ✅ Provider bridge configuration functional
- ✅ Ollama running locally with 16 models available

#### Phase 1: Core Agent with RAG Tool ✅ COMPLETE (2025-08-21, 2h)
- [x] ~~Fix Letta server startup to set `OLLAMA_BASE_URL` environment variable~~ ✅ Completed in Phase 0
- [x] Create basic `search_documents` tool using existing vector store ✅
- [x] Implement agent creation with tool at matter creation ✅
- [x] Connect tool to existing RAG infrastructure ✅
- [x] Ensure proper citation formatting ✅
- [x] Add provider-prefixed model names (e.g., "ollama/gpt-oss:20b") ✅
- [x] Test basic agent creation and message handling ✅

**Implementation Summary:**
- Created `app/letta_tools.py` with search_documents tool
- Updated `app/letta_adapter.py` for tool registration
- Enhanced `app/letta_provider_bridge.py` with provider prefixes
- Created `app/letta_agent.py` for stateful agent handling
- All tests passing (6/6)

#### Phase 2: UI Integration (Week 1-2)
- [ ] Remove chat mode selector
- [ ] Single conversation interface per matter
- [ ] Show when agent uses search tool
- [ ] Display sources with citations

#### Phase 3: Testing and Refinement (Week 2)
- [ ] Test conversation continuity
- [ ] Verify memory persistence
- [ ] Ensure tool reliability
- [ ] Performance optimization

#### Future Phases (Post-MVP)
- Additional document analysis tools
- Sub-agents for specialized tasks
- Advanced learning systems
- Proactive research capabilities
- Multi-agent coordination

## User Experience Changes

### Current UX
```
User: "What was the completion date?"
System: [Always searches] "According to documents..."
User: "What was the completion date?" (same question later)
System: [Searches again] "According to documents..."
```

### New UX
```
User: "What was the completion date?"
Agent: "Let me search for that information..." 
       [Uses search tool]
       "The original completion date was June 15, 2023, according to 
       the contract on page 12. I'll remember this for future reference."

User: "What was the completion date?" (same question later)
Agent: "As I mentioned earlier, the completion date was June 15, 2023."
       [No search needed]

User: "How does that relate to the delays?"
Agent: "Based on the June 15 completion date and let me check for delay notices..."
       [Selective search]
       "I found three delay notices that pushed the date to September 30..."
```

## Technical Implementation Details

### Prerequisites: Fix Letta Server Configuration
```python
# app/letta_server.py - Update _start_subprocess() method (line 234-236)
def _start_subprocess(self) -> bool:
    """Start server as a subprocess."""
    # ... existing code ...
    
    # Set up environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # Ensure output is not buffered
    env["OLLAMA_BASE_URL"] = "http://localhost:11434"  # CRITICAL FIX: Enable Ollama provider
    
    # Additional provider environment variables as needed
    if self.use_docker:
        # Docker on macOS/Windows needs special host
        env["OLLAMA_BASE_URL"] = "http://host.docker.internal:11434"
    
    # Start server process with fixed environment
    self.process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,  # Pass the fixed environment
        start_new_session=True
    )
```

### Agent Creation with Provider Selection
```python
async def create_matter_agent(matter: Matter, provider_config: ProviderConfiguration) -> str:
    """Create a Letta agent for a matter with RAG tools."""
    
    # Build LLM configuration based on selected provider
    if provider_config.provider_type == "ollama":
        llm_config = LlmConfig(
            model=f"ollama/{provider_config.model_name}",  # Use prefixed name
            model_endpoint_type="ollama",
            model_endpoint="http://localhost:11434",
            context_window=provider_config.context_window,
            max_tokens=provider_config.max_tokens
        )
        embedding_config = EmbeddingConfig(
            embedding_model=f"ollama/{provider_config.embedding_model}",
            embedding_endpoint_type="ollama",
            embedding_endpoint="http://localhost:11434"
        )
    else:
        # Handle other providers (Gemini, OpenAI) in future phases
        raise NotImplementedError(f"Provider {provider_config.provider_type} not yet supported")
    
    # Create agent with permanent configuration
    agent = await letta_client.agents.create(
        name=f"claim-assistant-{matter.slug}",
        system=AGENT_SYSTEM_PROMPT,
        tools=["search_documents"],  # Start with basic search tool
        llm_config=llm_config,
        embedding_config=embedding_config,
        memory_blocks=MEMORY_BLOCKS,
        metadata={"matter_id": matter.id}
    )
    
    # Store agent configuration with matter
    matter.agent_id = agent.id
    matter.llm_provider = provider_config.provider_type
    matter.llm_model = provider_config.model_name
    await matter.save()
    
    return agent.id
```

### Message Handling (Simplified)
```python
async def handle_user_message(matter_id: str, message: str) -> AgentResponse:
    """Send message to matter's agent."""
    
    # Get existing agent (created at matter creation)
    agent_id = await get_matter_agent(matter_id)
    
    if not agent_id:
        raise ValueError("No agent found for this matter")
    
    # Send message - agent decides whether to use search tool
    response = await letta_client.agents.messages.create(
        agent_id=agent_id,
        messages=[MessageCreate(role="user", content=message)]
    )
    
    # Extract tool usage and format response
    return format_agent_response(response)
```

### Tool Implementation Using Existing Infrastructure
```python
@letta_tool
async def search_documents(
    query: str,
    k: int = 5
) -> str:
    """Search tool callable by Letta agent - uses existing RAG infrastructure."""
    
    # Get matter ID from agent context
    matter_id = get_current_matter_id()
    
    # Use existing RAG engine
    from app.rag import RAGEngine
    rag_engine = RAGEngine()
    
    # Perform search using existing implementation
    search_results = await rag_engine.search(
        matter_id=matter_id,
        query=query,
        k=k
    )
    
    # Format with precise citations for agent
    formatted = []
    for result in search_results:
        citation = f"[{result.doc_name} p.{result.page_start}]"
        snippet = result.text[:300] + "..." if len(result.text) > 300 else result.text
        formatted.append(f"{citation} {snippet}")
    
    return "\n\n".join(formatted) if formatted else "No relevant documents found."
```

## Migration Strategy

### Phase 1: Minimal Breaking Changes
- Keep existing RAG infrastructure intact
- Agent uses existing vector stores and RAG engine through tools
- No data migration required initially

### Phase 2: New Matter Flow
- New matters create Letta agent at inception
- User selects LLM provider during matter creation
- Agent configuration stored with matter

### Phase 3: Existing Matter Migration
- Optional migration for existing matters
- Create agent with same provider as matter's current configuration
- Import any existing conversation history into agent memory

### Phase 4: UI Simplification
- Remove chat mode selector
- Single conversation interface
- Tool usage indicators

## Comprehensive Testing Matrix

### Unit Tests
| Component | Test Focus | Priority |
|-----------|------------|----------|
| Tool Registration | Verify upsert/attach patterns | Critical |
| Context Passing | Matter ID propagation to tools | Critical |
| Memory Limits | Token counting and overflow | High |
| Provider Config | OLLAMA_BASE_URL setting | Critical |
| JSON Output | Tool response structure | High |
| Error Handling | Graceful failure modes | High |

### Integration Tests
| Workflow | Test Scenario | Success Criteria |
|----------|--------------|------------------|
| Provider Discovery | Start server with Ollama | Models listed correctly |
| Agent Creation | Create with each provider | Agent configured properly |
| Tool Invocation | Call search_documents | Returns structured JSON |
| Memory Persistence | Restart server | Agent state preserved |
| Conversation Flow | Multi-turn dialogue | Context maintained |
| Memory Overflow | Exceed block limits | Automatic compression |

### End-to-End Tests
```python
async def test_complete_workflow():
    """Test complete agent lifecycle."""
    
    # 1. Start Letta server with Ollama
    server = await start_letta_server(env={"OLLAMA_BASE_URL": "http://localhost:11434"})
    
    # 2. Create agent with tools
    agent_id = await create_matter_agent(
        matter_name="Test Matter",
        provider="ollama",
        model="gpt-oss:20b"
    )
    
    # 3. Register search tool
    await register_tools_with_agent(client, agent_id, matter_id)
    
    # 4. Send message requiring search
    response = await client.agents.messages.create(
        agent_id=agent_id,
        messages=[{"role": "user", "content": "What is the completion date?"}]
    )
    
    # 5. Verify tool was called
    assert "search_documents" in response.tool_calls
    
    # 6. Verify memory updated
    memory = await client.agents.memory.get(agent_id, "case_facts")
    assert "completion date" in memory.value.lower()
    
    # 7. Test conversation continuity
    response2 = await client.agents.messages.create(
        agent_id=agent_id,
        messages=[{"role": "user", "content": "What was that date again?"}]
    )
    
    # Should recall without searching
    assert "search_documents" not in response2.tool_calls
```

### Performance Benchmarks
| Metric | Target | Acceptable | Critical |
|--------|--------|------------|----------|
| Agent Creation | < 2s | < 5s | > 10s |
| Tool Invocation | < 1s | < 3s | > 5s |
| Memory Recall | < 500ms | < 1s | > 2s |
| Search + Response | < 5s | < 10s | > 15s |
| Memory Compression | < 3s | < 7s | > 10s |

## Error Recovery & Resilience

### Agent Crash Recovery
```python
class AgentRecovery:
    """Handle agent failures and recovery."""
    
    async def recover_from_crash(self, agent_id: str):
        """Recover agent after unexpected failure."""
        try:
            # Check if agent still exists
            agent = await client.agents.retrieve(agent_id)
            if agent:
                # Verify memory integrity
                await self.verify_memory_integrity(agent_id)
                return agent
        except AgentNotFoundError:
            # Recreate from backup
            return await self.recreate_from_backup(agent_id)
    
    async def verify_memory_integrity(self, agent_id: str):
        """Check and repair memory blocks."""
        blocks = await client.agents.memory.list(agent_id)
        for block in blocks:
            if not self.is_valid_json(block.value):
                # Repair corrupted memory
                await self.repair_memory_block(agent_id, block.label)
    
    async def backup_agent_state(self, agent_id: str):
        """Periodic backup of agent state."""
        state = {
            "memory": await client.agents.memory.export(agent_id),
            "passages": await client.agents.passages.list(agent_id),
            "metadata": await client.agents.retrieve(agent_id).metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        backup_path = f"backups/{agent_id}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(backup_path, 'w') as f:
            json.dump(state, f)
```

### Tool Failure Handling
```python
async def robust_tool_execution(agent_id: str, tool_name: str, params: dict):
    """Execute tool with retry and fallback."""
    
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # Set timeout for tool execution
            result = await asyncio.wait_for(
                execute_tool(tool_name, params),
                timeout=30.0
            )
            return result
            
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                # Return timeout error
                return {
                    "status": "timeout",
                    "message": f"Tool {tool_name} timed out after {max_retries} attempts"
                }
                
        except Exception as e:
            logger.error(f"Tool {tool_name} failed", error=str(e))
            
            # Return graceful error
            return {
                "status": "error",
                "message": f"Tool execution failed: {str(e)}",
                "fallback": "Please try rephrasing your question"
            }
```

### Server Connection Resilience
```python
class ResilientConnection:
    """Maintain resilient connection to Letta server."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = None
        self.reconnect_attempts = 0
        self.max_reconnects = 5
    
    async def ensure_connected(self):
        """Ensure connection is alive, reconnect if needed."""
        if not await self.is_healthy():
            await self.reconnect()
    
    async def reconnect(self):
        """Reconnect with exponential backoff."""
        for attempt in range(self.max_reconnects):
            try:
                self.client = AsyncLetta(base_url=self.base_url)
                # Test connection
                await self.client.health.check()
                self.reconnect_attempts = 0
                return True
                
            except Exception as e:
                delay = min(60, 2 ** attempt)
                logger.warning(f"Reconnect attempt {attempt + 1} failed, retrying in {delay}s")
                await asyncio.sleep(delay)
        
        raise ConnectionError("Failed to reconnect to Letta server")
```

## Legal Compliance & Audit Features

### Audit Logging
```python
class ComplianceAuditLogger:
    """Log all agent interactions for legal compliance."""
    
    async def log_interaction(self, agent_id: str, interaction: dict):
        """Log interaction with legal metadata."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "matter_id": interaction.get("matter_id"),
            "user_message": interaction.get("user_message"),
            "tool_calls": interaction.get("tool_calls", []),
            "citations_used": self.extract_citations(interaction),
            "memory_updates": interaction.get("memory_updates", []),
            "response_hash": hashlib.sha256(
                interaction.get("response", "").encode()
            ).hexdigest()
        }
        
        # Write to immutable audit log
        await self.write_to_audit_log(audit_entry)
    
    async def export_for_discovery(self, matter_id: str, date_range: tuple):
        """Export all interactions for legal discovery."""
        interactions = await self.query_audit_log(
            matter_id=matter_id,
            start_date=date_range[0],
            end_date=date_range[1]
        )
        
        # Format for legal review
        return self.format_legal_export(interactions)
```

### Conversation Export
```python
async def export_conversation(agent_id: str, format: str = "pdf") -> bytes:
    """Export conversation with full citations for legal records."""
    
    # Get all messages
    messages = await client.agents.messages.list(agent_id)
    
    # Get memory state
    memory = await client.agents.memory.export(agent_id)
    
    # Get all tool calls and results
    tool_history = await get_tool_execution_history(agent_id)
    
    # Create structured export
    export_data = {
        "export_date": datetime.now().isoformat(),
        "matter_id": agent.metadata.get("matter_id"),
        "agent_id": agent_id,
        "messages": messages,
        "memory_state": memory,
        "tool_executions": tool_history,
        "citations": extract_all_citations(messages)
    }
    
    if format == "pdf":
        return generate_legal_pdf(export_data)
    elif format == "json":
        return json.dumps(export_data, indent=2).encode()
```

### Read-Only Policy Blocks
```python
# Add read-only compliance block to agent configuration
COMPLIANCE_MEMORY_BLOCK = {
    "label": "legal_policies",
    "value": """
    LEGAL COMPLIANCE REQUIREMENTS:
    1. All factual assertions must include citations
    2. Maintain attorney-client privilege
    3. No ex parte communications
    4. Preserve all work product
    5. Flag potential conflicts of interest
    6. Report discovery deadlines immediately
    7. Document all document reviews
    """,
    "limit": 1000,
    "read_only": True  # Agent cannot modify
}
```

## Success Metrics

### Functional Metrics (Realistic Targets)
- **Tool Invocation Success**: > 95% (tools execute without errors)
- **Response Time (Ollama)**: < 10 seconds for typical queries
- **Response Time (Cloud APIs)**: < 5 seconds for typical queries
- **Memory Recall Precision**: > 80% (correct facts recalled)
- **Conversation Continuity**: Maintained across sessions
- **Zero Data Leakage**: Between different matters

### Memory Management Metrics
- **Block Utilization**: Maintain < 80% capacity
- **Compression Trigger**: Automatic at 80% threshold
- **Archive Success Rate**: > 99% for overflow content
- **Memory Retrieval Time**: < 1 second for recent facts

### Search Optimization Metrics
- **Search Reduction**: 40% fewer searches after 10 conversations
- **Citation Accuracy**: 100% of facts have valid citations
- **Duplicate Detection**: 90% of redundant queries answered from memory
- **Context Relevance**: > 85% of retrieved chunks relevant

### User Experience Metrics
- **Answer Completeness**: > 90% satisfaction score
- **Tool Transparency**: Users understand when/why tools are used
- **Error Recovery**: < 2% of interactions require manual intervention
- **Learning Curve**: Agent improves noticeably within 5 interactions

## Risk Mitigation

### Risk: Agent Hallucination
**Mitigation**: 
- Always cite sources
- Confidence thresholds
- Fallback to search
- User verification prompts

### Risk: Performance Degradation
**Mitigation**:
- Async tool execution
- Response streaming
- Cache frequently accessed data
- Tool timeout limits

### Risk: Memory Overload
**Mitigation**:
- Memory pruning strategies
- Importance scoring
- Archival of old memories
- Selective recall

### Risk: Tool Failures
**Mitigation**:
- Graceful degradation
- Fallback tools
- Error explanations
- Manual override options

## Configuration Requirements

### Critical Server Fix Required
```python
# app/letta_server.py - MUST add this to enable Ollama
env["OLLAMA_BASE_URL"] = "http://localhost:11434"
```

### Provider-Specific Agent Configuration
```python
# Each agent is permanently tied to one provider/model
PROVIDER_CONFIGS = {
    "ollama": {
        "model_prefix": "ollama/",
        "endpoint": "http://localhost:11434",
        "requires_env": {"OLLAMA_BASE_URL": "http://localhost:11434"}
    },
    "gemini": {
        "model_prefix": "gemini/",
        "requires_api_key": True,
        "requires_env": {"GEMINI_API_KEY": "<user_key>"}
    },
    "openai": {
        "model_prefix": "openai/",
        "requires_api_key": True,
        "requires_env": {"OPENAI_API_KEY": "<user_key>"}
    }
}
```

### Phase 1 Agent Template
```python
AGENT_TEMPLATE = {
    "system_prompt": """You are a legal assistant specializing in construction claims.
    You have access to case documents through the search_documents tool.
    Always cite sources using the format [DocName.pdf p.X] when referencing documents.
    Learn from each interaction and remember important facts about the case.""",
    
    "tools": ["search_documents"],  # Start with one tool
    
    "memory_blocks": [
        {"name": "case_facts", "description": "Key facts about the construction claim"},
        {"name": "entities", "description": "People, companies, and organizations"},
        {"name": "timeline", "description": "Important dates and deadlines"},
        {"name": "conversations", "description": "Summary of our discussions"}
    ]
}
```

## Development Priorities

### Phase 1: Core Functionality (Must Have)
1. Fix Letta server configuration bug
2. Basic agent with document search tool
3. Provider selection at matter creation
4. Single conversation interface
5. Memory persistence through Letta

### Phase 2: Enhanced Tools (Future)
1. Additional document analysis tools
2. Date range filtering
3. Document summarization
4. Timeline extraction

### Phase 3: Advanced Features (Future)
1. Sub-agents for specialized tasks
2. Custom learning systems
3. Proactive research suggestions
4. Multi-agent coordination
5. Knowledge graph visualization

## Key Implementation Notes

### Provider Configuration is Permanent
- Each Letta agent is locked to one LLM provider/model at creation
- Provider selection happens at matter creation time
- Cannot be changed after agent creation
- Tools can potentially use different models (future enhancement)
- **Phase 1 Learning**: Provider prefixing (ollama/, gemini/) is critical for Letta v0.10.0

### Leveraging Existing Infrastructure
- Use existing RAG engine and vector stores
- No need to rebuild document search functionality
- Agent calls existing infrastructure through tools
- Minimal changes to current system
- **Phase 1 Learning**: Tool implementation must be synchronous for Letta v0.10.0

### Focus on Core Functionality First
- Start with single search_documents tool ✅ Implemented
- Rely on Letta's built-in memory management ✅ Working
- Add advanced features incrementally
- Ensure basic stateful agent works before adding complexity
- **Phase 1 Learning**: Mix of sync/async clients requires careful handling

### Technical Discoveries from Phase 1
- Letta v0.10.0 uses RESTClient (synchronous), not AsyncLetta
- Tool registration uses `create_tool()` not `tools.upsert()`
- Agent messages API may return 500 errors with complex tools
- Memory API differs from expected interface (no `agents.memory` attribute)
- OLLAMA_BASE_URL environment variable is required for Ollama provider

## Conclusion

This specification outlines a pragmatic transformation from RAG-first to agent-first architecture. By focusing on core functionality first (fixing the Letta configuration bug and implementing a basic search tool), we can quickly achieve a working stateful agent. The system will maintain conversation context, remember facts from previous interactions, and intelligently decide when to search documents versus using its memory.

Future enhancements like additional tools, sub-agents, and advanced learning systems can be added incrementally once the foundation is solid. The key insight is that each agent is permanently tied to a specific LLM provider/model, which simplifies configuration management while still allowing flexibility through tools and future sub-agents.