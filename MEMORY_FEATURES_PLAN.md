# Memory Features Implementation Plan

## Overview

### Current System Architecture
The Letta Claim Assistant is a local-first construction claim analysis application that combines two AI technologies:

1. **RAG (Retrieval-Augmented Generation)**: Searches through uploaded PDF documents to find relevant passages for answering questions. Documents are processed through OCR, chunked, embedded as vectors, and stored in a ChromaDB database for semantic search.

2. **Letta Agent Memory**: A stateful AI agent that maintains persistent memory about each matter (case). As users interact with the system, the agent learns and remembers facts, entities, events, and issues. This memory accumulates over time, making the assistant more knowledgeable about the specific case.

Currently, when a user asks a question in the chat:
1. The system searches relevant document chunks using vector similarity (RAG)
2. The Letta agent recalls relevant memories about the matter
3. Both contexts are combined and sent to the LLM to generate an answer
4. The interaction and any extracted facts are stored back into the agent's memory

### Current Limitations
While powerful, the current system has several limitations:
- **Opacity**: Users can only see a basic summary of what's in memory - they don't know what specific facts the agent has learned or might be using
- **No Control**: If the agent memorizes incorrect information, users cannot correct it
- **Mixed Context**: Users cannot choose to query just documents or just memory - it's always both
- **No Direct Memory Interaction**: Users cannot explicitly tell the agent to remember or forget specific information

### What These Features Will Accomplish
This implementation plan adds comprehensive memory management capabilities that will:

1. **Provide Transparency**: Users can see exactly what the agent remembers - every fact, entity, event, and issue stored in memory with full details

2. **Enable Control**: Users can edit incorrect memories, delete irrelevant ones, and manually add important facts that might not be explicitly stated in documents

3. **Offer Flexibility**: Three distinct chat modes allow users to:
   - **RAG-only**: Search only documents (useful for finding specific passages)
   - **Memory-only**: Query only the agent's learned knowledge (faster, focuses on established facts)
   - **Combined**: Current behavior, using both sources

4. **Support Natural Interaction**: Users can manage memory through natural language ("Remember that the deadline is March 15th" or "Forget what I said about the contractor")

5. **Ensure Accuracy**: For legal work, precision is critical. These features ensure the AI assistant's knowledge base is accurate, complete, and under the user's control

Each sprint in this plan incrementally builds these capabilities while maintaining system stability and ensuring existing functionality continues to work perfectly.

## Core Principles
- **Non-breaking**: All changes are additive; existing features continue to work
- **Incremental**: Each sprint delivers usable functionality
- **Testable**: Each sprint includes verification steps
- **Reversible**: Features can be disabled if issues arise

---

## Sprint 1: Memory Items List API (2 hours)
**Goal**: Create backend infrastructure to list and retrieve individual memory items

### Tasks:
1. **Add Memory Item Models** (`app/models.py`)
   ```python
   class MemoryItem(BaseModel):
       id: str
       text: str
       type: Literal["Entity", "Event", "Issue", "Fact", "Interaction"]
       created_at: datetime
       metadata: Dict[str, Any]
       source: Optional[str]
   ```

2. **Add API Endpoints** (`app/api.py`)
   - `GET /api/matters/{id}/memory/items` - List all memory items with pagination
   - `GET /api/matters/{id}/memory/items/{item_id}` - Get specific item
   - Parameters: `limit`, `offset`, `type_filter`, `search_query`

3. **Extend LettaAdapter** (`app/letta_adapter.py`)
   ```python
   async def get_memory_items(
       self, 
       limit: int = 50, 
       offset: int = 0,
       type_filter: Optional[str] = None,
       search_query: Optional[str] = None
   ) -> List[MemoryItem]
   ```

### Verification:
- Test endpoints with curl/Postman
- Verify pagination works
- Check memory isolation between matters

### Rollback: Simply don't call new endpoints; existing code unaffected

---

## Sprint 2: Memory Viewer UI (2.5 hours)
**Goal**: Create read-only UI to view all memory items

### Tasks:
1. **Add API Client Methods** (`ui/api_client.py`)
   ```python
   async def get_memory_items(matter_id: str, limit=50, offset=0)
   async def get_memory_item(matter_id: str, item_id: str)
   ```

2. **Create Memory Viewer Component** (`ui/memory_viewer.py`)
   - Tabbed interface or searchable list
   - Memory item cards showing:
     - Type badge (color-coded)
     - Full text (expandable)
     - Timestamp
     - Source reference
   - Search/filter controls
   - Pagination

3. **Add "View All Memories" Button** (`ui/memory_components.py`)
   - Place next to existing "View Summary" button
   - Opens memory viewer dialog/panel

### Verification:
- View memories for existing matter
- Test search and filters
- Verify pagination
- Ensure UI remains responsive

### Rollback: Remove button; existing summary still works

---

## Sprint 3: Chat Mode Infrastructure (2 hours)
**Goal**: Add backend support for different chat modes

### Tasks:
1. **Define Chat Modes** (`app/models.py`)
   ```python
   class ChatMode(str, Enum):
       RAG_ONLY = "rag"        # Documents only
       MEMORY_ONLY = "memory"  # Agent memory only
       COMBINED = "combined"   # Current behavior (default)
   ```

2. **Update Chat Endpoint** (`app/api.py`)
   - Add `mode` parameter to chat request
   - Keep "combined" as default for backward compatibility

3. **Implement Mode Logic** (`app/rag.py`)
   ```python
   async def generate_answer(query, mode=ChatMode.COMBINED):
       if mode == ChatMode.RAG_ONLY:
           # Skip memory recall
       elif mode == ChatMode.MEMORY_ONLY:
           # Skip vector search
       else:  # COMBINED
           # Current behavior
   ```

4. **Add Memory-Only Chat Handler** (`app/letta_adapter.py`)
   ```python
   async def memory_only_chat(self, query: str) -> Dict[str, Any]:
       # Direct interaction with Letta agent
       # No RAG pipeline involvement
   ```

### Verification:
- Test each mode via API
- Ensure default behavior unchanged
- Verify memory-only mode is faster

### Rollback: Default mode continues to work as before

---

## Sprint 4: Chat Mode UI (2 hours)
**Goal**: Add UI controls to switch between chat modes

### Tasks:
1. **Add Mode Selector Widget** (`ui/chat_components.py`)
   - Radio buttons or dropdown
   - Tooltips explaining each mode
   - Visual indicator of active mode

2. **Update Chat Interface** (`ui/main.py`)
   - Add mode selector above chat input
   - Pass mode to API calls
   - Show mode indicator in chat messages

3. **Update API Client** (`ui/api_client.py`)
   - Add `mode` parameter to `send_chat_message()`
   - Default to "combined" for compatibility

### Verification:
- Switch between modes
- Verify correct behavior for each
- Check that mode persists during session

### Rollback: Hide selector; default mode works

---

## Sprint 5: Memory Edit API (2.5 hours)
**Goal**: Backend support for editing and deleting memories

### Tasks:
1. **Add Mutation Endpoints** (`app/api.py`)
   - `PUT /api/matters/{id}/memory/items/{item_id}` - Update item
   - `DELETE /api/matters/{id}/memory/items/{item_id}` - Delete item
   - `POST /api/matters/{id}/memory/items` - Create new item

2. **Implement in LettaAdapter** (`app/letta_adapter.py`)
   ```python
   async def update_memory_item(item_id: str, new_text: str)
   async def delete_memory_item(item_id: str)
   async def create_memory_item(text: str, type: str)
   ```

3. **Add Audit Logging**
   - Log all memory modifications
   - Include user, timestamp, before/after

### Verification:
- Test CRUD operations
- Verify audit log entries
- Check memory persistence

### Rollback: Don't expose edit UI; view-only remains

---

## Sprint 6: Memory Editor UI (3 hours)
**Goal**: UI for editing, adding, and deleting memories

### Tasks:
1. **Add Edit Mode to Viewer** (`ui/memory_viewer.py`)
   - Edit/Delete buttons on each card
   - "Add Memory" button
   - Confirmation dialogs

2. **Create Memory Editor Dialog** (`ui/memory_editor.py`)
   - Form with:
     - Text area for memory content
     - Type selector
     - Save/Cancel buttons
   - Validation

3. **Update API Client** (`ui/api_client.py`)
   - Add CRUD methods for memory items

### Verification:
- Edit existing memory
- Add new memory
- Delete memory
- Verify changes persist

### Rollback: Hide edit buttons; read-only viewer remains

---

## Sprint 7: Natural Language Memory Management (3 hours)
**Goal**: Allow memory management through chat

### Tasks:
1. **Add Memory Command Parser** (`app/memory_commands.py`)
   - Detect memory management intents:
     - "Remember that..."
     - "Forget about..."
     - "Update your memory..."
     - "What do you remember about..."

2. **Implement Command Handlers** (`app/api.py`)
   - New endpoint: `/api/matters/{id}/memory/command`
   - Parse natural language
   - Execute memory operations
   - Return confirmation

3. **Add UI Indicators** (`ui/main.py`)
   - Show when memory operation detected
   - Confirmation messages
   - Option to undo

### Verification:
- Test various command patterns
- Verify memory updates
- Check error handling

### Rollback: Disable command detection

---

## Sprint 8: Memory Search and Analytics (2 hours)
**Goal**: Advanced memory search and insights

### Tasks:
1. **Add Search Endpoint** (`app/api.py`)
   - `/api/matters/{id}/memory/search`
   - Full-text search
   - Semantic search option

2. **Add Analytics Endpoint**
   - `/api/matters/{id}/memory/analytics`
   - Memory growth over time
   - Type distribution
   - Most referenced entities

3. **Create Analytics Dashboard** (`ui/memory_analytics.py`)
   - Charts showing memory statistics
   - Timeline view
   - Entity relationship graph (optional)

### Verification:
- Search for specific memories
- View analytics
- Check performance with large memory sets

---

## Sprint 9: Memory Import/Export (2 hours)
**Goal**: Backup and restore memory

### Tasks:
1. **Add Import/Export Endpoints** (`app/api.py`)
   - `GET /api/matters/{id}/memory/export` - Download JSON/CSV
   - `POST /api/matters/{id}/memory/import` - Upload memories

2. **Add UI Controls** (`ui/memory_viewer.py`)
   - Export button
   - Import button with file picker
   - Format selector

### Verification:
- Export memories
- Import to new matter
- Verify data integrity

---

## Sprint 10: Performance and Polish (2 hours)
**Goal**: Optimize and refine all memory features

### Tasks:
1. **Performance Optimization**
   - Add caching for memory lists
   - Implement lazy loading
   - Optimize search queries

2. **UI Polish**
   - Loading states
   - Error boundaries
   - Keyboard shortcuts
   - Better mobile responsiveness

3. **Documentation**
   - Update user guide
   - Add tooltips
   - Create memory management tutorial

---

## Testing Strategy

### After Each Sprint:
1. Run existing tests to ensure no regression
2. Test new feature in isolation
3. Test integration with existing features
4. Document any issues or limitations

### Integration Testing (After Sprint 6):
- Full workflow: View → Edit → Verify
- Multi-matter memory isolation
- Performance with 1000+ memories
- Concurrent user operations

---

## Risk Mitigation

### Feature Flags
Add to `app/settings.py`:
```python
MEMORY_FEATURES = {
    "view_all": True,      # Sprint 2
    "edit": False,         # Sprint 6
    "chat_modes": False,   # Sprint 4
    "nl_commands": False,  # Sprint 7
}
```

### Gradual Rollout
1. Test with single matter first
2. Enable for power users
3. Full rollout after stability confirmed

### Monitoring
- Log all memory operations
- Track performance metrics
- Monitor error rates
- User feedback collection

---

## Success Metrics
- **Performance**: Memory-only chat 50% faster than RAG
- **Accuracy**: Users correct <5% of memories
- **Adoption**: 80% of users view memories
- **Reliability**: <0.1% error rate on memory operations

---

## Future Enhancements (Post-MVP)
- Memory versioning/history
- Collaborative memory editing
- Memory templates for common legal patterns
- Integration with external knowledge bases
- Memory confidence scores
- Automatic memory validation
- Memory merge/conflict resolution