# Future Features Roadmap

## Multi-Turn Conversation Support

### Overview
Add support for multi-turn conversations where previous exchanges are maintained in context, enabling more natural, contextual dialogue especially useful for complex legal analysis requiring clarification and follow-up questions.

### Motivation
Currently, each chat message is treated independently. This limits the ability to:
- Ask follow-up questions that reference previous answers
- Clarify or refine previous queries
- Build complex arguments over multiple exchanges
- Maintain context for pronoun resolution ("it", "that", "they")

### Technical Design

#### 1. Conversation Management
```python
class Conversation(BaseModel):
    id: str
    matter_id: str
    title: str  # Auto-generated or user-provided
    created_at: datetime
    last_message_at: datetime
    message_count: int
    total_tokens: int  # Track context size
    mode: ChatMode  # RAG, Memory, or Combined
    status: Literal["active", "archived", "context_full"]
```

#### 2. Context Window Management

**Ollama Integration:**
```python
async def get_model_info(model_name: str) -> Dict:
    # Query Ollama for model capabilities
    response = await ollama_client.show(model_name)
    return {
        "context_window": response.get("context_length", 4096),
        "max_tokens": response.get("num_ctx", 2048)
    }
```

**Context Tracking:**
- Display token counter in UI (e.g., "2,341 / 4,096 tokens")
- Warning at 50% capacity (yellow indicator)
- Prevent new messages at 70% capacity
- Offer to start new conversation or summarize

**For Gemini:** 
- 1M token window makes this nearly a non-issue
- Still track for cost awareness
- Could enable very long conversations

#### 3. Context Building Strategy

```python
async def build_conversation_context(
    conversation_id: str,
    new_query: str,
    max_tokens: int
) -> str:
    # Get conversation history
    messages = await get_conversation_messages(conversation_id)
    
    # Build context intelligently
    if total_tokens < max_tokens * 0.5:
        # Include all messages
        return format_all_messages(messages)
    else:
        # Smart truncation
        return await compress_context(messages, preserve_recent=3)
```

### Implementation Sprints

#### Sprint A: Conversation Data Model (2 hours)
- Create conversation tables/models
- Add conversation CRUD operations
- Update chat to store conversation_id

#### Sprint B: Multi-Turn Context (2.5 hours)
- Modify RAG engine to accept conversation history
- Implement context building logic
- Update Letta adapter for conversation awareness

#### Sprint C: Context Window Management (3 hours)
- Query Ollama for model limits
- Implement token counting (using tiktoken or estimate)
- Add context limit enforcement
- Create truncation strategies

#### Sprint D: Conversation UI (3 hours)
- Conversation sidebar with list
- "New Conversation" button
- Conversation title editing
- Archive/delete options
- Token usage indicator

#### Sprint E: Advanced Features (2 hours)
- Auto-title generation from first message
- Conversation search
- Export conversation as markdown/PDF
- Summarize conversation feature

### UI Mockup
```
+------------------+------------------------+
| Conversations    | Chat Area              |
+------------------+                        |
| [New Chat]       | Previous message...    |
|                  |                        |
| ▼ Active         | Previous response...   |
| • Contract Rev.. |                        |
| • Deadline Iss.. | Current message...     |
|                  |                        |
| ▼ Archived       | [Current response]     |
| • Old Analysis   |                        |
+------------------+------------------------+
| Tokens: 2.3k/4k  | [Message input box]    |
+------------------+------------------------+
```

### Considerations

#### Pros:
- Natural conversation flow
- Better context for complex discussions
- Reduced need to repeat information
- More efficient for iterative analysis

#### Cons:
- Added complexity in context management
- Token limit challenges with local models
- Need for conversation management UI
- Potential confusion with multiple active conversations

### Dependencies
- Requires chat mode selection (Memory Features Sprint 3-4)
- Benefits from memory-only mode for faster interactions
- May need rate limiting for Gemini API costs

### Success Metrics
- Average conversation length > 5 messages
- User satisfaction with follow-up accuracy
- Reduced query reformulation
- Context limit hit rate < 10%

---

## Multi-Domain Legal Matter Support

### Overview
Expand the application beyond construction claims to support multiple legal practice areas including real estate transactions, general litigation, and other document-heavy legal work. Leverage the existing matter isolation and Letta's memory system for cross-domain functionality.

### Key Benefits
- **Shared Infrastructure**: OCR, chunking, and vector search remain domain-agnostic
- **Cross-Matter Learning**: Letta can identify patterns across different matter types
- **Unified Knowledge Base**: Statute collections and regulations benefit all practice areas
- **Single Codebase**: Maintain one application instead of multiple forks

### Implementation Approach

#### Matter Type System
```python
class MatterType(Enum):
    CONSTRUCTION_CLAIM = "construction_claim"
    REAL_ESTATE_TRANSACTION = "real_estate" 
    GENERAL_LITIGATION = "general_litigation"
    BUSINESS_TRANSACTION = "business_transaction"
    FACILITIES_MANAGEMENT = "facilities"

class Matter(BaseModel):
    id: str
    name: str
    matter_type: MatterType
    custom_prompts: Optional[Dict[str, str]] = None
    statute_collections: Optional[List[str]] = None
    template_library: Optional[List[str]] = None
```

#### Domain-Specific Components
- **Custom Prompt Templates**: Each matter type gets specialized system prompts and RAG query templates
- **Document Templates**: Standard contracts, forms, and checklists per matter type
- **Statute Tagging**: Legal references tagged by applicable practice areas
- **Matter-Specific Tools**: Custom analysis tools (e.g., timeline extraction for litigation, title chain for real estate)

#### Shared Knowledge Libraries
```python
class Statute(BaseModel):
    id: str
    citation: str
    text: str
    domains: List[MatterType]
    keywords: List[str]
    last_updated: datetime

class KnowledgeLibrary(BaseModel):
    name: str
    type: Literal["statutes", "regulations", "forms", "precedents"]
    domains: List[MatterType]
    path: Path
```

### Long-Term Project Advantages
- Letta's memory compounds over months/years of a matter
- Agent learns client preferences and case-specific terminology
- Historical context preserved across document revisions
- Proactive identification of changes and emerging issues

---

## Agentic Research System

### Overview
Implement a ReAct-style (Reasoning + Acting) pattern where Letta develops research plans, executes multiple parallel searches across documents and memory, and synthesizes comprehensive answers with full traceability.

### Core Workflow
1. **Query Analysis**: Letta analyzes the question to identify key components
2. **Research Planning**: Agent creates structured plan with sub-questions and search strategies
3. **Parallel Execution**: Multiple RAG and memory searches run simultaneously
4. **Result Synthesis**: Letta combines findings into coherent, cited answer
5. **Confidence Assessment**: Agent evaluates answer completeness and suggests follow-ups

### Implementation Design

#### Research Plan Structure
```python
class ResearchPlan(BaseModel):
    original_query: str
    decomposed_questions: List[str]
    search_strategies: List[SearchStrategy]
    memory_queries: List[str]
    priority_documents: Optional[List[str]]
    confidence_threshold: float

class SearchStrategy(BaseModel):
    query: str
    search_type: Literal["semantic", "keyword", "hybrid"]
    filters: Dict[str, Any]
    expected_info_type: str  # "dates", "parties", "obligations", etc.
    k: int = 8
```

#### Agentic Search Implementation
```python
class LettaResearchAgent:
    async def research_and_answer(self, query: str, matter: Matter) -> ResearchResult:
        # 1. Generate research plan
        plan = await self.letta_adapter.create_research_plan(
            query=query,
            matter_context=matter,
            available_documents=self.get_document_list(matter)
        )
        
        # 2. Execute searches in parallel
        search_tasks = [
            self.execute_search_strategy(strategy, matter)
            for strategy in plan.search_strategies
        ]
        memory_tasks = [
            self.letta_adapter.search_memory(q, matter)
            for q in plan.memory_queries
        ]
        
        # 3. Gather results with progress tracking
        rag_results = await asyncio.gather(*search_tasks)
        memory_results = await asyncio.gather(*memory_tasks)
        
        # 4. Synthesize comprehensive answer
        answer = await self.letta_adapter.synthesize_answer(
            plan=plan,
            rag_results=rag_results,
            memory_results=memory_results,
            matter_context=matter
        )
        
        # 5. Assess and suggest follow-ups
        assessment = await self.assess_answer_completeness(answer, plan)
        
        return ResearchResult(
            answer=answer,
            plan=plan,
            confidence=assessment.confidence,
            suggested_followups=assessment.followups
        )
```

#### Enhanced Traceability
- **Research Plan Visibility**: Users see the agent's research strategy before execution
- **Progress Tracking**: Real-time updates as each search completes
- **Source Attribution**: Clear mapping of which searches yielded which insights
- **Reasoning Chain**: Explicit documentation of synthesis logic
- **Confidence Scoring**: Per-statement confidence based on source quality

### UI Enhancements
```python
# New UI components for research visibility
class ResearchPlanCard:
    """Shows decomposed questions and search strategies"""
    
class SearchProgressBar:
    """Real-time progress through parallel searches"""
    
class SynthesisExplainer:
    """Interactive view of how results were combined"""
    
class ConfidenceIndicator:
    """Visual confidence scoring with source breakdown"""
```

### Benefits Over Simple RAG
- **Comprehensive Coverage**: Multiple search angles ensure nothing is missed
- **Intelligent Prioritization**: Agent learns which strategies work for different query types
- **Memory Integration**: Combines document search with accumulated case knowledge
- **Explainability**: Full visibility into the research and reasoning process
- **Iterative Refinement**: Agent can identify gaps and suggest follow-up searches

---

## Additional Future Features

### 1. Memory Templates
Pre-defined memory structures for common legal patterns:
- Contract analysis template
- Timeline template
- Party relationship template

### 2. Memory Versioning
- Track changes to memories over time
- Ability to revert to previous versions
- Diff view for memory changes

### 3. Collaborative Memory Editing
- Multiple users can suggest memory edits
- Approval workflow for memory changes
- Change attribution and audit trail

### 4. Smart Memory Suggestions
- AI suggests memories to add based on document analysis
- Conflict detection between memories
- Memory deduplication

### 5. Memory Visualization
- Graph view of entity relationships
- Timeline visualization of events
- Memory clustering by topic

### 6. Advanced RAG Features
- Document-specific chat (only search specific PDFs)
- Citation verification (check if citations are accurate)
- Passage highlighting in PDF viewer

### 7. Export and Reporting
- Generate case summary reports
- Export memory as knowledge graph
- Create legal briefs from memory + documents

### 8. Integration Features
- Import from legal databases
- Export to case management systems
- API for external tool integration

### 9. Performance Optimizations
- Memory caching strategies
- Incremental indexing for large documents
- Background memory optimization

### 10. Advanced Security
- Memory encryption at rest
- Role-based access control
- Memory sharing with granular permissions

---

## Implementation Priority

### High Priority (Next 3 months)
1. Multi-turn conversations
2. Multi-Domain Legal Matter Support
3. Agentic Research System

### Medium Priority (3-6 months)
4. Memory templates
5. Memory versioning
6. Smart memory suggestions
7. Memory visualization

### Low Priority (6+ months)
8. Advanced RAG features
9. Collaborative editing
10. External integrations
11. Advanced security features

---

## Notes
- Each feature should be implemented incrementally
- User feedback should drive priority changes
- Performance impact must be measured for each feature
- All features must maintain matter isolation