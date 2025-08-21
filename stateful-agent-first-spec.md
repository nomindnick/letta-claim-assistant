# Stateful Agent-First Architecture Specification

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
- **Issue**: Letta server not respecting agent-specific LLM configurations
- **Root Cause**: Server requires `OLLAMA_BASE_URL` environment variable to enable Ollama provider
- **Fix Required**: Update server startup in `app/letta_server.py` to set provider environment variables

#### Agent Model Permanence
- **Constraint**: Once a Letta agent is created with a specific model/provider, it cannot be changed
- **Implication**: Provider selection must happen at matter creation time
- **Opportunity**: Tools and sub-agents can use different models than the main agent

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
            "name": "case_facts",
            "description": "Key facts about the construction claim"
        },
        {
            "name": "entities",
            "description": "People, companies, and organizations involved"
        },
        {
            "name": "timeline",
            "description": "Chronological events and deadlines"
        },
        {
            "name": "conversations",
            "description": "Summary of past interactions and questions"
        }
    ],
    
    "tools": [
        "search_documents"  # Start with basic RAG search tool only
    ]
}
```

### 2. Tool Definitions

#### Phase 1: Core RAG Search Tool (Immediate Priority)
```python
@letta_tool
async def search_documents(
    query: str,
    k: int = 5
) -> str:
    """
    Search case documents for specific information.
    
    Args:
        query: Search query string
        k: Number of results to return (default: 5)
    
    Returns:
        Formatted search results with citations
    """
    # Get matter context from agent metadata
    matter_id = get_agent_matter_id()
    
    # Use existing vector store infrastructure
    vector_store = get_vector_store(matter_id)
    
    # Perform search using existing RAG engine
    results = await vector_store.search(query, k=k)
    
    # Format for agent with precise citations
    formatted = []
    for r in results:
        citation = f"[{r.doc_name} p.{r.page_start}]"
        text_snippet = r.text[:300] + "..." if len(r.text) > 300 else r.text
        formatted.append(f"{citation} {text_snippet}")
    
    return "\n\n".join(formatted)
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

### 3. Agent Learning Mechanisms (Built into Letta)

Letta provides built-in learning capabilities through its memory system:

#### Automatic Memory Management
- **Conversation History**: Letta automatically maintains conversation context
- **Memory Blocks**: Agent updates its memory blocks based on interactions
- **Fact Retention**: Important information is automatically stored in agent memory
- **Context Awareness**: Agent remembers previous questions and answers

#### Phase 1 Focus
For initial implementation, we rely on Letta's built-in memory capabilities:
- Agent automatically remembers facts from document searches
- No custom knowledge graph implementation needed initially
- Conversation continuity handled by Letta's memory system
- Tool usage patterns learned implicitly through memory

#### Future Enhancements (Not Phase 1)
```python
# Advanced learning systems can be added later:
# - Custom knowledge graphs
# - Confidence scoring systems
# - Pattern recognition
# - Proactive research suggestions
# These are not needed for basic stateful agent functionality
```

### 4. Implementation Phases (Revised for Core Functionality First)

#### Phase 1: Fix Letta Configuration (Immediate)
- [ ] Fix Letta server startup to set `OLLAMA_BASE_URL` environment variable
- [ ] Add provider-prefixed model names (e.g., "ollama/gpt-oss:20b")
- [ ] Implement provider verification on startup
- [ ] Test basic agent creation and message handling

#### Phase 2: Core Agent with RAG Tool (Week 1)
- [ ] Create basic `search_documents` tool using existing vector store
- [ ] Implement agent creation with tool at matter creation
- [ ] Connect tool to existing RAG infrastructure
- [ ] Ensure proper citation formatting

#### Phase 3: UI Integration (Week 1-2)
- [ ] Remove chat mode selector
- [ ] Single conversation interface per matter
- [ ] Show when agent uses search tool
- [ ] Display sources with citations

#### Phase 4: Testing and Refinement (Week 2)
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
# app/letta_server.py - Add to _start_subprocess() method
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["OLLAMA_BASE_URL"] = "http://localhost:11434"  # CRITICAL: Enable Ollama provider
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

## Success Metrics

### Functional Metrics
- Tool usage accuracy > 90%
- Response time < 5 seconds
- Memory recall precision > 85%
- Conversation continuity score > 80%

### Learning Metrics
- Knowledge graph growth rate
- Fact extraction accuracy
- Contradiction detection rate
- Search reduction over time (fewer searches needed)

### User Experience Metrics
- Reduced redundant searches
- Increased answer relevance
- Better context awareness
- Natural conversation flow

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

### Leveraging Existing Infrastructure
- Use existing RAG engine and vector stores
- No need to rebuild document search functionality
- Agent calls existing infrastructure through tools
- Minimal changes to current system

### Focus on Core Functionality First
- Start with single search_documents tool
- Rely on Letta's built-in memory management
- Add advanced features incrementally
- Ensure basic stateful agent works before adding complexity

## Conclusion

This specification outlines a pragmatic transformation from RAG-first to agent-first architecture. By focusing on core functionality first (fixing the Letta configuration bug and implementing a basic search tool), we can quickly achieve a working stateful agent. The system will maintain conversation context, remember facts from previous interactions, and intelligently decide when to search documents versus using its memory.

Future enhancements like additional tools, sub-agents, and advanced learning systems can be added incrementally once the foundation is solid. The key insight is that each agent is permanently tied to a specific LLM provider/model, which simplifies configuration management while still allowing flexibility through tools and future sub-agents.