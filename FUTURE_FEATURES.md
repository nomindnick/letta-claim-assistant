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
2. Memory templates
3. Memory versioning

### Medium Priority (3-6 months)
4. Smart memory suggestions
5. Memory visualization
6. Advanced RAG features

### Low Priority (6+ months)
7. Collaborative editing
8. External integrations
9. Advanced security features

---

## Notes
- Each feature should be implemented incrementally
- User feedback should drive priority changes
- Performance impact must be measured for each feature
- All features must maintain matter isolation