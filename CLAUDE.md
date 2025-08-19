# Claude Development Instructions - Letta Construction Claim Assistant

**Project:** Local-first construction claim analysis with RAG + stateful agent memory  
**Target Platform:** Ubuntu Linux desktop application  
**Last Updated:** 2025-08-14

---

## 🚨 MANDATORY PRE-WORK CHECKLIST

Before starting ANY development work, you MUST:

1. **Read the Complete Specification**
   - Review `spec.md` thoroughly to understand project requirements
   - Pay special attention to user stories and acceptance criteria (§4)
   - Note the technical constraints and architecture decisions (§6)

2. **Understand the Implementation Plan**
   - Study `IMPLEMENTATION_PLAN.md` for sprint-based development approach
   - Identify which sprint you're working on and its specific deliverables
   - Review dependencies and prerequisites for the current sprint

3. **Check Development Progress**
   - Read `DEVELOPMENT_PROGRESS.md` to understand current project state
   - Review completed sprints and key technical decisions
   - Note any known issues or risks that may affect your work

4. **Verify Environment Setup**
   - Confirm all system dependencies are installed
   - Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
   - Check Python environment and package installations
   - Validate external services (Ollama, models) are available

**⚠️ Do NOT proceed with implementation until all four items above are completed.**

---

## Project Overview & Context

### Core Mission
Build a desktop application that analyzes construction claim PDFs using RAG (Retrieval-Augmented Generation) combined with a **stateful agent** powered by Letta. The agent's matter-specific memory compounds over time, making answers contextually aware, traceable, and proactive.

### Key Principles
- **Local-first**: All data stays on disk under `~/LettaClaims/` unless user explicitly chooses external LLMs
- **Privacy-focused**: External API calls only with clear user consent
- **Matter isolation**: Each case/claim has completely separate data and memory
- **Responsiveness**: UI never blocks during background processing
- **Traceability**: Every answer cites specific document pages

### Technical Architecture
```
[ NiceGUI Desktop (native=True) ]
          |  (async calls)
  [ Ingestion + RAG + Letta services ]
          |
  +-------+-----------------------+------------------+
  |                               |                  |
[Chroma (per Matter)]      [Ollama (gen+embed)]   [Gemini API (opt-in)]
```

---

## Development Standards & Conventions

### Code Quality Requirements
```python
# MANDATORY: All functions must have complete type hints
async def process_pdf(
    self, 
    input_path: Path, 
    output_path: Path,
    force_ocr: bool = False
) -> OCRResult:
    """Process PDF with OCR and return results."""
    # Implementation here
```

### Error Handling Patterns
```python
# Use explicit error types with user-friendly messages
class PDFProcessingError(Exception):
    """Raised when PDF processing fails with recoverable error."""
    
    def __init__(self, message: str, retry_possible: bool = True):
        self.message = message
        self.retry_possible = retry_possible
        super().__init__(message)

# Always provide actionable error information
try:
    result = await process_pdf(pdf_path)
except PDFProcessingError as e:
    logger.error(f"PDF processing failed: {e.message}")
    if e.retry_possible:
        # Offer retry UI option
    else:
        # Show permanent error message
```

### Async Patterns
```python
# Consistent async/await usage for I/O operations
async def ingest_documents(self, matter: Matter, files: List[Path]) -> str:
    """Start background ingestion job and return job_id."""
    job_id = await self.job_queue.submit_job(
        job_type="pdf_ingestion",
        params={"matter_id": matter.id, "files": [str(f) for f in files]},
        progress_callback=self._update_progress
    )
    return job_id
```

### Logging Standards
```python
import structlog

logger = structlog.get_logger(__name__)

# Include matter context in all logs
logger.info(
    "PDF ingestion started",
    matter_id=matter.id,
    matter_name=matter.name,
    file_count=len(files),
    job_id=job_id
)
```

---

## File Organization & Module Structure

### Core Application (`app/`)
- **`__init__.py`** - Package initialization
- **`settings.py`** - Configuration management (global + per-matter)
- **`logging_conf.py`** - Structured logging setup
- **`matters.py`** - Matter CRUD operations and filesystem management
- **`ingest.py`** - PDF processing pipeline (OCR → parse → chunk → embed)
- **`vectors.py`** - ChromaDB operations and vector search
- **`rag.py`** - RAG engine with prompt assembly and response generation
- **`jobs.py`** - Background job queue with progress tracking
- **`letta_adapter.py`** - Letta integration for agent memory
- **`api.py`** - FastAPI router mounted at `/api`

### LLM Providers (`app/llm/`)
- **`base.py`** - Provider protocols and interfaces
- **`ollama_provider.py`** - Local generation and embeddings
- **`gemini_provider.py`** - External generation with consent
- **`embeddings.py`** - Embedding model abstraction

### User Interface (`ui/`)
- **`main.py`** - NiceGUI application entry point
- **`api_client.py`** - HTTP client for backend API
- **`widgets/`** - Reusable UI components

### Data Models
Use Pydantic for all data structures:

```python
class Matter(BaseModel):
    id: str
    name: str
    slug: str
    created_at: datetime
    embedding_model: str
    generation_model: str
    paths: MatterPaths

class SourceChunk(BaseModel):
    doc: str
    page_start: int
    page_end: int
    text: str
    score: float

class KnowledgeItem(BaseModel):
    type: Literal["Entity", "Event", "Issue", "Fact"]
    label: str
    date: Optional[str] = None
    actors: List[str] = []
    doc_refs: List[dict] = []
    support_snippet: Optional[str] = None
```

---

## Critical Implementation Guidelines

### Matter Isolation
```python
# CRITICAL: Never mix data between matters
def get_vector_store(matter: Matter) -> VectorStore:
    """Get isolated vector store for specific matter."""
    collection_path = matter.paths.vectors / "chroma"
    return VectorStore(collection_path)

# WRONG - this could leak data between matters
# global_vector_store = VectorStore("global_path")
```

### Citation Accuracy
```python
# Citations MUST map precisely to source documents and pages
def format_sources(chunks: List[SearchResult]) -> List[SourceChunk]:
    """Convert search results to UI-displayable sources with precise citations."""
    sources = []
    for chunk in chunks:
        # Ensure page numbers are accurate and verifiable
        source = SourceChunk(
            doc=chunk.metadata["doc_name"],
            page_start=chunk.metadata["page_start"],
            page_end=chunk.metadata["page_end"],
            text=chunk.text[:600],  # Truncate for UI display
            score=chunk.similarity_score
        )
        sources.append(source)
    return sources
```

### Background Processing
```python
# UI must remain responsive during long operations
async def upload_files_endpoint(matter_id: str, files: List[UploadFile]):
    """Handle file upload with background processing."""
    # Start background job immediately
    job_id = await job_queue.submit_job(
        job_type="pdf_ingestion",
        params={"matter_id": matter_id, "files": files}
    )
    
    # Return job_id immediately, don't wait for completion
    return {"job_id": job_id}
```

---

## Testing Requirements

### Before Sprint Completion
You MUST run these commands and ensure they pass:

```bash
# Activate virtual environment first
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install test dependencies if not present
pip install pytest pytest-asyncio

# Run unit tests
python -m pytest tests/unit/ -v

# Run integration tests  
python -m pytest tests/integration/ -v

# Type checking (if mypy available)
python -m mypy app/ ui/ --ignore-missing-imports

# Code formatting check
python -m black --check app/ ui/ tests/
```

### Test Coverage Requirements
- **Unit Tests**: All core business logic functions
- **Integration Tests**: End-to-end user workflows
- **Error Cases**: Expected failures and recovery paths
- **Provider Tests**: Both Ollama and Gemini integration

---

## Sprint Completion Protocol

After completing each sprint:

1. **Verify Acceptance Criteria**
   - Check each item in the sprint's acceptance criteria
   - Test the functionality manually if needed
   - Ensure no regressions in existing features

2. **Update Development Progress**
   - Edit `DEVELOPMENT_PROGRESS.md`
   - Mark sprint as completed with timestamp
   - Document key implementation decisions
   - Note any deviations from the plan
   - Add new dependencies or configuration changes

3. **Record Technical Decisions**
   - Document why specific approaches were chosen
   - Note any trade-offs made
   - Record lessons learned or issues encountered

4. **Prepare for Next Sprint**
   - Review next sprint requirements
   - Note any changes to approach based on current sprint learnings
   - Update risk assessments if needed

### Example Progress Update
```markdown
### Sprint 2: PDF Ingestion Pipeline (Completed 2025-08-14, 3.5h)

**Implementation Summary:**
- Created OCRProcessor class with skip-text and force-OCR modes
- Implemented PyMuPDF-based text extraction with page alignment
- Built chunking algorithm targeting ~1000 tokens with 15% overlap
- Added MD5-based deduplication for chunks

**Key Technical Decisions:**
- Used OCRmyPDF's --skip-text by default to preserve born-digital text
- Stored page boundaries in chunk metadata for precise citation
- Implemented progress callbacks using asyncio for UI responsiveness

**Dependencies Added:**
- ocrmypdf>=15.0.0
- pymupdf>=1.23.0

**Issues Encountered:**
- OCRmyPDF timeout on large files - added timeout configuration
- Page numbering inconsistency between PyMuPDF and PDF viewers - documented limitation

**Next Sprint Prep:**
- Sprint 3 ready to proceed
- OCR output verified compatible with vector embedding pipeline
```

---

## Configuration Management

### Global Configuration
File: `~/.letta-claim/config.toml`

```python
# Loading configuration
import tomllib
from pathlib import Path

def load_global_config() -> dict:
    """Load global configuration with defaults."""
    config_path = Path.home() / ".letta-claim" / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return get_default_config()
```

### Per-Matter Configuration
File: `Matter_<slug>/config.json`

```python
# Matter-specific settings
def save_matter_config(matter: Matter, config: dict) -> None:
    """Save matter-specific configuration."""
    config_path = matter.paths.root / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
```

---

## Error Handling Patterns

### User-Facing Errors
```python
# Provide clear, actionable error messages
class UserFacingError(Exception):
    """Base for errors that should be shown to users."""
    
    def __init__(self, message: str, suggestion: str = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)

# Usage in UI
try:
    result = await some_operation()
except UserFacingError as e:
    ui.notify(f"Error: {e.message}", type="error")
    if e.suggestion:
        ui.notify(f"Try: {e.suggestion}", type="info")
```

### Retry Logic
```python
import asyncio
import random

async def retry_with_backoff(
    operation: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> Any:
    """Retry operation with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s", error=str(e))
            await asyncio.sleep(delay)
```

---

## Security & Privacy Guidelines

### Local Data Protection
```python
# Ensure all sensitive data stays local
def validate_local_path(path: Path) -> bool:
    """Ensure path is within allowed local directories."""
    home = Path.home()
    letta_claims = home / "LettaClaims"
    
    try:
        # Resolve and check if path is under allowed directories
        resolved = path.resolve()
        return resolved.is_relative_to(letta_claims) or resolved.is_relative_to(home / ".letta-claim")
    except (OSError, ValueError):
        return False
```

### External API Consent
```python
# Always get user consent before external API calls
async def ensure_external_consent(provider: str) -> bool:
    """Ensure user has consented to external API usage."""
    consent_key = f"consent_{provider}_api"
    
    if not settings.get(consent_key, False):
        # Show consent dialog
        consent_given = await show_consent_dialog(provider)
        if consent_given:
            settings.set(consent_key, True)
        return consent_given
    
    return True
```

---

## Performance Considerations

### Memory Management
```python
# Process large documents in chunks to avoid memory issues
async def process_large_pdf(pdf_path: Path, chunk_size: int = 50) -> List[ProcessedChunk]:
    """Process PDF in manageable chunks."""
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    
    processed_chunks = []
    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        
        # Process subset of pages
        page_chunks = await process_page_range(doc, start_page, end_page)
        processed_chunks.extend(page_chunks)
        
        # Yield control to prevent blocking
        await asyncio.sleep(0)
    
    doc.close()
    return processed_chunks
```

### Vector Search Optimization
```python
# Optimize vector searches with proper indexing
class VectorStore:
    def __init__(self, collection_path: Path):
        self.client = chromadb.PersistentClient(str(collection_path))
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Optimize for similarity search
        )
    
    async def search_with_metadata_filter(
        self,
        query: str,
        k: int = 8,
        filter_dict: dict = None
    ) -> List[SearchResult]:
        """Search with optional metadata filtering for performance."""
        query_embedding = await self.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict,  # Pre-filter to reduce search space
            include=["documents", "metadatas", "distances"]
        )
        
        return self._format_results(results)
```

---

## Git Workflow & Commit Standards

### Commit Message Format
```
[Sprint X] Brief description of changes

Longer description if needed explaining:
- What was implemented
- Why specific approaches were chosen
- Any issues encountered and resolved

Closes #issue-number (if applicable)
```

### Branch Strategy
- `main`: Stable, tested code
- `sprint-X`: Feature branch for each sprint
- `hotfix-*`: Critical fixes

### Pre-commit Checklist
- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] DEVELOPMENT_PROGRESS.md updated
- [ ] No secrets or sensitive data in commit

---

## Common Pitfalls to Avoid

### 1. Matter Data Leakage
```python
# WRONG - shared global state
global_embeddings = {}

# RIGHT - matter-isolated state
def get_embeddings_for_matter(matter_id: str) -> VectorStore:
    return VectorStore(get_matter_path(matter_id) / "vectors")
```

### 2. Blocking UI Operations
```python
# WRONG - blocks UI thread
def upload_handler(files):
    for file in files:
        process_pdf(file)  # This blocks!
    return "Done"

# RIGHT - background processing
async def upload_handler(files):
    job_id = await job_queue.submit_job("process_pdfs", {"files": files})
    return {"job_id": job_id}
```

### 3. Imprecise Citations
```python
# WRONG - vague citations
citation = "[Document.pdf]"

# RIGHT - precise page references
citation = f"[{doc_name} p.{page_start}-{page_end}]"
```

### 4. Missing Error Recovery
```python
# WRONG - silent failures
try:
    result = risky_operation()
except Exception:
    pass  # User has no idea what happened

# RIGHT - actionable error handling
try:
    result = risky_operation()
except SpecificError as e:
    logger.error("Operation failed", error=str(e))
    ui.notify("Operation failed. Please try again or contact support.", type="error")
    return error_response(str(e))
```

---

## Final Reminders

### Before Every Development Session
1. ✅ Read spec.md completely
2. ✅ Review IMPLEMENTATION_PLAN.md for current sprint
3. ✅ Check DEVELOPMENT_PROGRESS.md for latest status
4. ✅ Understand sprint acceptance criteria

### After Every Sprint
1. ✅ Verify all acceptance criteria met
2. ✅ Run full test suite
3. ✅ Update DEVELOPMENT_PROGRESS.md
4. ✅ Document key decisions and learnings

### Quality Gates
- **Code**: Type hints, error handling, async patterns
- **Tests**: Unit coverage, integration workflows, error cases  
- **Documentation**: Progress updates, decision rationale
- **Performance**: UI responsiveness, memory efficiency

**Remember: The goal is a production-ready application that attorneys can trust with sensitive legal documents. Every implementation decision should prioritize correctness, security, and user experience.**