# Letta Agent Memory Enhancement - Revised Implementation Plan

**Goal:** Integrate Letta's persistent agent memory system into the existing construction claims assistant to provide stateful, context-aware interactions that improve over time.

**Approach:** Research-first implementation using modern Letta architecture (server + client), with granular sprints for clear progress tracking and risk mitigation.

**Status:** Research phase - addressing deprecated LocalClient API and migrating to new architecture.

**Version:** Letta >= 0.11.0 with letta-client >= 0.1.0

---

## 📚 Critical Reference Document

**IMPORTANT:** Before implementing any sprint, consult **`LETTA_TECHNICAL_REFERENCE.md`** for:
- Detailed API examples and code patterns
- Server setup instructions
- Configuration templates
- Troubleshooting guides
- Proven integration patterns

This reference document will be created in Sprint L-R and continuously updated with findings.

---

## 🔄 Major Architecture Change

**Previous Assumption:** Use LocalClient for serverless operation (Sprint 5 implementation)
**Current Reality:** LocalClient is deprecated; must use server-based architecture
**New Approach:** Local Letta server (Docker or process) with modern client API

---

## Sprint Overview

| Sprint | Duration | Focus | Dependencies | Risk Level |
|--------|----------|-------|--------------|------------|
| **L-R** | 2h | Research & Documentation | None | Low |
| **L0** | ✅ 0.5h | Data Migration Check | None | Low |
| **L1** | 1h | Server Infrastructure Setup | L-R | Medium |
| **L2** | 1h | Client Connection & Fallback | L1 | Low |
| **L3** | 1.5h | Agent Lifecycle Management | L2 | Medium |
| **L4** | 1.5h | Memory Operations | L3 | Low |
| **L5** | 1.5h | LLM Provider Integration | L2 | Medium |
| **L6** | 1.5h | Construction Domain Optimization | L4, L5 | Low |
| **L7** | 1.5h | Testing & Reliability | L6 | Low |
| **L8** | 1h | Documentation & Polish | L7 | Low |

**Total Estimated Time:** 11.5 hours (was 6 hours in original plan)

---

## Sprint L-R: Research & Technical Documentation (2 hours)

### Goal
Create comprehensive technical reference for all subsequent Letta implementation work.

### Deliverables

1. **Create `LETTA_TECHNICAL_REFERENCE.md`** containing:
   - Complete Letta architecture overview (server, client, storage)
   - Working code examples for each operation
   - Server setup instructions (Docker and local process)
   - API migration guide (LocalClient → modern client)
   - LLM provider configuration (Ollama, Gemini, OpenAI)
   - Memory persistence patterns
   - Error handling strategies
   - Performance considerations

2. **Proof of Concept Scripts**:
   ```
   research/
   ├── test_letta_server.py      # Server connectivity test
   ├── test_letta_ollama.py      # Ollama integration test
   ├── test_letta_gemini.py      # External API integration test
   ├── test_letta_memory.py      # Memory operations test
   └── test_letta_migration.py   # LocalClient migration test
   ```

3. **Configuration Templates**:
   ```
   config/
   ├── letta_server_config.yaml  # Server configuration
   ├── docker-compose.yml         # Docker setup
   └── client_config.json         # Client connection settings
   ```

### Acceptance Criteria
- [ ] Technical reference document is comprehensive and accurate
- [ ] All POC scripts run successfully
- [ ] Clear migration path from current implementation documented
- [ ] LLM provider flexibility confirmed (Ollama + external APIs)
- [ ] Server setup instructions tested and working

### Research Questions to Answer
1. Can we run Letta server as a subprocess managed by our app?
2. What's the minimal Docker configuration needed?
3. How do we configure multiple LLM providers?
4. What's the data migration path from old LocalClient?
5. How do we handle server unavailability gracefully?
6. What are the performance implications of server architecture?

---

## Sprint L0: Data Migration Check (0.5 hours) ✅ COMPLETED

### Goal
Verify existing Letta data and provide migration guidance.

### Status
**Completed on 2025-08-18**

### Implementation
- Added `_check_existing_data()` method to detect existing agent data
- Version tracking in agent_config.json
- Comprehensive warnings and backup guidance
- Non-destructive verification only

---

## Sprint L1: Letta Server Infrastructure (1 hour)

### Goal
Set up local Letta server infrastructure with multiple deployment options.

### Deliverables

1. **Server Management Module** (`app/letta_server.py`):
   - Server lifecycle management (start, stop, health check)
   - Automatic server startup on application launch
   - Graceful shutdown on application exit
   - Port conflict resolution
   - Server process monitoring

2. **Deployment Options**:
   - **Option A:** Docker container (preferred for isolation)
   - **Option B:** Local Python subprocess (simpler deployment)
   - **Option C:** External server connection (advanced users)

3. **Configuration** (`app/letta_config.py`):
   - Server connection settings
   - Timeout and retry configuration
   - Resource limits
   - Logging configuration

### Implementation Tasks
Refer to `LETTA_TECHNICAL_REFERENCE.md` Section 2: Server Setup

### Acceptance Criteria
- [ ] Server starts automatically with application
- [ ] Health endpoint responds within 5 seconds
- [ ] Server stops cleanly on application exit
- [ ] Port conflicts handled gracefully
- [ ] Works without Docker if not available
- [ ] Configuration can be overridden by users

---

## Sprint L2: Client Connection & Fallback (1 hour)

### Goal
Establish reliable client connection with comprehensive fallback behavior.

### Deliverables

1. **Update `app/letta_adapter.py`**:
   - Replace LocalClient with modern `letta_client.Letta`
   - Implement connection retry logic
   - Add connection pooling
   - Health check before operations
   - Graceful fallback when server unavailable

2. **Connection Manager**:
   - Singleton pattern for client reuse
   - Automatic reconnection on failure
   - Connection state monitoring
   - Metrics collection (latency, success rate)

### Implementation Tasks
Refer to `LETTA_TECHNICAL_REFERENCE.md` Section 3: Client Connection

### Acceptance Criteria
- [ ] Client connects to local server successfully
- [ ] Automatic retry on transient failures
- [ ] Fallback mode maintains basic functionality
- [ ] Connection errors logged clearly
- [ ] No blocking operations in UI thread
- [ ] Connection state visible in UI

---

## Sprint L3: Agent Lifecycle Management (1.5 hours)

### Goal
Implement complete agent lifecycle with matter-specific isolation.

### Deliverables

1. **Agent Management in `app/letta_adapter.py`**:
   - Create agents with matter-specific configuration
   - Load existing agents by ID
   - Update agent configuration
   - Delete agents on matter deletion
   - Agent state persistence

2. **Agent Configuration**:
   - Matter-specific system prompts
   - Memory block initialization
   - Tool registration (if applicable)
   - Model configuration per agent

3. **Migration Support**:
   - Detect old LocalClient agents
   - Provide migration path to new format
   - Preserve existing memory where possible

### Implementation Tasks
Refer to `LETTA_TECHNICAL_REFERENCE.md` Section 4: Agent Management

### Acceptance Criteria
- [ ] Each matter has isolated agent
- [ ] Agents persist across restarts
- [ ] Agent configuration respects matter settings
- [ ] Old agents detected and migration offered
- [ ] Agent creation errors handled gracefully
- [ ] Agent metadata stored locally

---

## Sprint L4: Memory Operations (1.5 hours)

### Goal
Implement robust memory storage and retrieval operations.

### Deliverables

1. **Memory Storage**:
   - Store conversation interactions
   - Extract and store knowledge items
   - Update core memory blocks
   - Archive important facts
   - Maintain conversation context

2. **Memory Retrieval**:
   - Semantic search through memory
   - Relevance scoring
   - Recency weighting
   - Memory summarization
   - Context assembly for RAG

3. **Memory Management**:
   - Memory size monitoring
   - Old memory archival
   - Memory export/import
   - Memory statistics API

### Implementation Tasks
Refer to `LETTA_TECHNICAL_REFERENCE.md` Section 5: Memory Operations

### Acceptance Criteria
- [ ] Interactions stored with full context
- [ ] Memory recall improves answer relevance
- [ ] Knowledge items properly structured
- [ ] Memory search returns relevant results
- [ ] Memory persists across sessions
- [ ] Memory size stays within limits

---

## Sprint L5: LLM Provider Integration (1.5 hours)

### Goal
Integrate Letta with multiple LLM providers while maintaining flexibility.

### Deliverables

1. **Provider Configuration**:
   - **Primary:** Ollama integration (local, default)
   - **Secondary:** External APIs (Gemini, OpenAI, Anthropic)
   - User-selectable providers per matter
   - Provider-specific parameters
   - Automatic fallback chain

2. **Update `app/letta_adapter.py`**:
   - Configure Letta agents with selected LLM
   - Pass provider credentials securely
   - Handle provider-specific errors
   - Monitor provider performance
   - Cost tracking for external APIs

3. **Provider Management**:
   - Runtime provider switching
   - Provider health checks
   - Usage limits and quotas
   - Provider preference persistence

### Implementation Tasks
Refer to `LETTA_TECHNICAL_REFERENCE.md` Section 6: LLM Providers

### Acceptance Criteria
- [ ] Ollama works as default provider
- [ ] External APIs work with user consent
- [ ] Provider selection persists per matter
- [ ] Provider errors handled gracefully
- [ ] Costs displayed for external APIs
- [ ] Automatic fallback when provider fails

### Security Considerations
- API keys stored securely (encrypted)
- External API usage requires explicit consent
- Data sent to external APIs logged for audit
- Sensitive data filtering before external calls

---

## Sprint L6: Construction Domain Optimization (1.5 hours)

### Goal
Optimize Letta agents for construction claims analysis.

### Deliverables

1. **Domain-Specific Configuration**:
   - Construction claims system prompts
   - Entity extraction templates (parties, dates, amounts)
   - Document reference patterns
   - Legal terminology handling
   - Timeline extraction

2. **Knowledge Structuring**:
   - Construction-specific knowledge schemas
   - Claim type categorization
   - Damage assessment templates
   - Responsibility mapping
   - Precedent tracking

3. **Enhanced Follow-ups**:
   - Domain-specific follow-up templates
   - Claim investigation prompts
   - Evidence gathering suggestions
   - Expert consultation recommendations

### Implementation Tasks
Refer to `LETTA_TECHNICAL_REFERENCE.md` Section 7: Domain Optimization

### Acceptance Criteria
- [ ] Agents understand construction terminology
- [ ] Extracts entities accurately (parties, dates, costs)
- [ ] Generates relevant follow-up questions
- [ ] Maintains claim timeline in memory
- [ ] Identifies contradictions and gaps
- [ ] Suggests relevant document requests

---

## Sprint L7: Testing & Reliability (1.5 hours)

### Goal
Ensure robust, reliable Letta integration with comprehensive testing.

### Deliverables

1. **Test Suites**:
   - Unit tests for all Letta operations
   - Integration tests with server
   - Performance benchmarks
   - Failure scenario tests
   - Memory leak tests
   - Concurrency tests

2. **Test Infrastructure**:
   - Mock Letta server for testing
   - Test data generators
   - Performance profiling
   - Memory usage monitoring
   - Load testing scripts

3. **Reliability Features**:
   - Circuit breaker for server calls
   - Request queuing and batching
   - Timeout management
   - Resource cleanup
   - Error recovery procedures

### Implementation Tasks
Refer to `LETTA_TECHNICAL_REFERENCE.md` Section 8: Testing

### Acceptance Criteria
- [ ] 90%+ test coverage for Letta code
- [ ] All failure scenarios handled
- [ ] No memory leaks detected
- [ ] Performance meets targets
- [ ] Concurrent operations work correctly
- [ ] Recovery procedures documented

---

## Sprint L8: Documentation & Polish (1 hour)

### Goal
Complete documentation and polish for production readiness.

### Deliverables

1. **User Documentation**:
   - Letta features user guide
   - Configuration instructions
   - Troubleshooting guide
   - FAQ section
   - Best practices

2. **Developer Documentation**:
   - API reference
   - Architecture diagrams
   - Deployment guide
   - Performance tuning
   - Debugging procedures

3. **Polish**:
   - UI indicators for memory operations
   - Memory statistics dashboard
   - Agent health monitoring
   - Performance optimizations
   - Error message improvements

### Acceptance Criteria
- [ ] User guide covers all features
- [ ] Troubleshooting covers common issues
- [ ] API fully documented
- [ ] Performance meets production standards
- [ ] UI provides clear feedback
- [ ] All messages user-friendly

---

## Implementation Strategy

### Before Starting Any Sprint

1. **Read `LETTA_TECHNICAL_REFERENCE.md`** completely
2. **Check server status** - ensure Letta server is running
3. **Verify dependencies** - letta, letta-client installed
4. **Review previous sprint outcomes** - check for any issues or learnings
5. **Update test environment** - ensure test matter exists

### Development Workflow

1. Create feature branch: `letta-sprint-{number}`
2. Implement according to reference document
3. Test with both Ollama and external providers
4. Update reference document with findings
5. Commit with message: `[Letta Sprint {number}] Description`

### Risk Mitigation

- **Server dependency**: Always maintain fallback mode
- **API changes**: Pin specific Letta versions
- **Performance**: Implement caching and request batching
- **Data loss**: Regular backups of agent data
- **Provider failures**: Multiple provider fallback chain

---

## Success Metrics

### Technical Metrics
- Server uptime > 99% during application runtime
- Memory operations < 500ms average latency
- Agent creation < 2 seconds
- Memory recall improves answer relevance by 30%+
- Zero data leakage between matters

### User Experience Metrics
- Seamless fallback when Letta unavailable
- No UI blocking during memory operations
- Clear indication of memory-enhanced responses
- Helpful follow-up suggestions based on context
- Visible improvement in assistance over time

### Business Value
- Reduced time finding relevant information
- Better claim analysis through accumulated knowledge
- Improved consistency across related queries
- Proactive identification of patterns and issues
- Enhanced user confidence through citation tracking

---

## Technical Architecture

### Component Diagram
```
┌─────────────────┐
│   NiceGUI UI    │
└────────┬────────┘
         │
┌────────▼────────┐
│  LettaAdapter   │
└────────┬────────┘
         │
┌────────▼────────┐     ┌──────────────┐
│  Letta Client   │────▶│ Letta Server │
└─────────────────┘     └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │   Storage    │
                        │  (SQLite)    │
                        └──────────────┘
```

### Data Flow
1. User query → RAG engine
2. RAG engine → Letta adapter (memory recall)
3. Enhanced context → LLM provider (Ollama/External)
4. Response → Knowledge extraction
5. Knowledge → Letta adapter (memory storage)
6. Updated memory → Future queries

### Provider Architecture
```
LettaAdapter
    ├── Primary: Ollama (local)
    │   ├── gpt-oss:20b
    │   ├── gemma2
    │   └── phi4
    │
    └── Secondary: External APIs
        ├── Gemini (with consent)
        ├── OpenAI (with consent)
        └── Anthropic (with consent)
```

---

## Configuration Management

### Server Configuration (`~/.letta-claim/letta-server.yaml`)
```yaml
server:
  mode: local  # local, docker, external
  host: localhost
  port: 8283
  auto_start: true
  
storage:
  type: sqlite
  path: ~/.letta-claim/letta.db
  
logging:
  level: info
  file: ~/.letta-claim/logs/letta-server.log
```

### Client Configuration (per-matter)
```json
{
  "letta": {
    "enabled": true,
    "agent_id": "uuid",
    "provider": "ollama",
    "model": "gpt-oss:20b",
    "external_api": {
      "provider": "gemini",
      "consent_given": false,
      "last_consent_date": null
    }
  }
}
```

---

## Migration Path

### From LocalClient (Sprint 5) to Server Architecture

1. **Data Migration**:
   - Export existing agent memory to JSON
   - Create new agent with server API
   - Import memory into new agent
   - Verify memory integrity

2. **Code Migration**:
   - Replace `LocalClient` imports
   - Update method signatures
   - Add server management
   - Implement new error handling

3. **Testing Migration**:
   - Update test mocks
   - Add server fixtures
   - Update integration tests
   - Performance revalidation

---

## Next Actions

1. **Immediate** (Sprint L-R):
   - Research current Letta architecture thoroughly
   - Create proof of concept scripts
   - Document all findings in technical reference
   - Validate LLM provider flexibility

2. **Infrastructure** (Sprint L1-L2):
   - Set up Letta server
   - Establish client connection
   - Implement fallback behavior
   - Test with real matters

3. **Core Features** (Sprint L3-L6):
   - Agent lifecycle management
   - Memory operations
   - Provider integration
   - Domain optimization

4. **Production Readiness** (Sprint L7-L8):
   - Comprehensive testing
   - Performance optimization
   - Complete documentation
   - User interface polish

---

## Appendix: Key Decisions

### Why Server Architecture?
- LocalClient deprecated in Letta 0.11+
- Server provides better isolation
- Enables future scaling options
- Consistent with modern Letta design

### Why Support Multiple LLM Providers?
- User flexibility and choice
- Cost optimization (local vs cloud)
- Fallback options for reliability
- Privacy considerations (local-first)

### Why Granular Sprints?
- Reduces implementation risk
- Clearer progress tracking
- Easier debugging and testing
- Better handoff between sessions

---

*This revised plan addresses the deprecated LocalClient issue and provides a clear path forward with modern Letta architecture while maintaining LLM provider flexibility.*