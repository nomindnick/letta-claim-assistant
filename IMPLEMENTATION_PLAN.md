# Letta Construction Claim Assistant - Implementation Plan

**Project:** Local-first construction claim analysis with RAG + stateful agent memory  
**Target:** Ubuntu Linux desktop application using NiceGUI  
**Architecture:** Single Python app with async job processing  

---

## Sprint Overview

| Sprint | Focus Area | Duration | Dependencies |
|--------|------------|----------|--------------|
| 0 | Project Setup & Foundation | 2-3h | System packages, Python env |
| 1 | Core Architecture & Matter Management | 3-4h | Sprint 0 |
| 2 | PDF Ingestion Pipeline | 4h | OCR dependencies |
| 3 | Vector Database & Embeddings | 3h | Ollama setup |
| 4 | Basic RAG Implementation | 3-4h | Sprint 3 |
| 5 | Letta Agent Integration | 4h | Letta installation |
| 6 | NiceGUI Desktop Interface - Part 1 | 3-4h | UI framework |
| 7 | NiceGUI Desktop Interface - Part 2 | 3-4h | Sprint 6 |
| 8 | Advanced RAG Features | 3h | Sprint 4, 5 |
| 9 | LLM Provider Management | 2-3h | External API setup |
| 10 | Job Queue & Background Processing | 3h | Async infrastructure |
| 11 | Error Handling & Edge Cases | 2-3h | All core features |
| 12 | Testing & Polish | 3-4h | Testing frameworks |
| 13 | Production Readiness | 2-3h | Final integration |

**Total Estimated Time: 38-47 hours**

---

## Sprint 0: Project Setup & Foundation (2-3h)

### Objectives
- Initialize project structure and version control
- Set up Python environment with all dependencies
- Install and configure system packages
- Create basic configuration management

### Technical Requirements
```bash
# System packages
sudo apt-get install -y \
  ocrmypdf tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
  poppler-utils

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

### Deliverables
1. **Project Structure**
   ```
   letta-claim-assistant/
   ├── app/
   │   ├── __init__.py
   │   ├── settings.py
   │   ├── logging_conf.py
   │   ├── matters.py
   │   ├── ingest.py
   │   ├── vectors.py
   │   ├── rag.py
   │   ├── jobs.py
   │   ├── llm/
   │   │   ├── base.py
   │   │   ├── ollama_provider.py
   │   │   └── gemini_provider.py
   │   ├── letta_adapter.py
   │   └── api.py
   ├── ui/
   │   ├── main.py
   │   ├── api_client.py
   │   └── widgets/
   ├── tests/
   │   ├── unit/
   │   └── integration/
   ├── requirements.txt
   ├── main.py
   └── config.toml
   ```

2. **Dependencies Installation**
   ```python
   # requirements.txt content
   nicegui>=1.4.0
   chromadb>=0.4.0
   pymupdf>=1.23.0
   pydantic>=2.0.0
   uvicorn>=0.23.0
   structlog>=23.0.0
   ollama>=0.2.0
   google-genai>=0.5.0
   letta>=0.3.0
   ocrmypdf>=15.0.0
   asyncio-mqtt>=0.13.0
   ```

3. **Git Repository Setup**
   - Initialize git repository
   - Create .gitignore for Python/IDE files
   - Initial commit with project structure

4. **Global Configuration Template**
   - `~/.letta-claim/config.toml` with default settings
   - Environment detection and path resolution

### Acceptance Criteria
- [ ] All system dependencies installed successfully
- [ ] Python virtual environment created and activated
- [ ] All Python packages install without conflicts
- [ ] Ollama running with required models pulled
- [ ] Project structure matches specification
- [ ] Git repository initialized with clean history
- [ ] Configuration file loads without errors

---

## Sprint 1: Core Architecture & Matter Management (3-4h)

### Objectives
- Implement Matter creation, switching, and management
- Create filesystem layout per specification
- Build foundation classes and data models
- Implement configuration management

### Technical Requirements
- Filesystem operations for Matter directories
- Pydantic models for data validation
- Thread-safe Matter switching
- Configuration persistence

### Deliverables
1. **Data Models** (`app/models.py`)
   ```python
   class Matter(BaseModel):
       id: str
       name: str
       slug: str
       created_at: datetime
       embedding_model: str
       generation_model: str
       paths: MatterPaths

   class MatterPaths(BaseModel):
       root: Path
       docs: Path
       docs_ocr: Path
       parsed: Path
       vectors: Path
       knowledge: Path
       chat: Path
       logs: Path
   ```

2. **Matter Management** (`app/matters.py`)
   - `create_matter(name: str) -> Matter`
   - `list_matters() -> List[Matter]`
   - `switch_matter(matter_id: str) -> Matter`
   - `get_active_matter() -> Matter | None`
   - Filesystem directory creation with proper structure

3. **Settings Management** (`app/settings.py`)
   - Global configuration loading from TOML
   - Per-matter configuration persistence
   - Environment variable integration
   - Validation and defaults

4. **Logging Configuration** (`app/logging_conf.py`)
   - Structured logging with matter context
   - File rotation and cleanup
   - Debug/production modes

### Acceptance Criteria
- [ ] Can create new Matter with unique slug
- [ ] Matter directory structure created correctly under `~/LettaClaims/`
- [ ] Can list and switch between multiple Matters
- [ ] No data leakage between Matter contexts
- [ ] Configuration persists between application restarts
- [ ] All filesystem operations are thread-safe
- [ ] Logging captures matter-specific context

---

## Sprint 2: PDF Ingestion Pipeline (4h)

### Objectives
- Implement OCR processing with OCRmyPDF
- Build PDF parsing with PyMuPDF
- Create text chunking with overlap and metadata
- Handle various PDF formats (born-digital, scanned, mixed)

### Technical Requirements
- OCRmyPDF integration with skip-text and force-ocr modes
- PyMuPDF for text extraction and page handling
- Text chunking algorithm (~1000 tokens, 15% overlap)
- MD5-based deduplication
- Progress tracking for background jobs

### Deliverables
1. **OCR Processing** (`app/ocr.py`)
   ```python
   class OCRProcessor:
       async def process_pdf(
           self, 
           input_path: Path, 
           output_path: Path,
           force_ocr: bool = False,
           language: str = "eng"
       ) -> OCRResult
   ```

2. **PDF Parsing** (`app/parsing.py`)
   ```python
   class PDFParser:
       def extract_pages(self, pdf_path: Path) -> List[PageContent]
       def get_document_metadata(self, pdf_path: Path) -> DocumentMetadata
   ```

3. **Text Chunking** (`app/chunking.py`)
   ```python
   class TextChunker:
       def chunk_document(
           self, 
           pages: List[PageContent],
           target_size: int = 1000,
           overlap_percent: float = 0.15
       ) -> List[Chunk]
   ```

4. **Ingestion Pipeline** (`app/ingest.py`)
   - Orchestrates OCR → Parse → Chunk workflow
   - Progress reporting and error handling
   - Duplicate detection and skip logic
   - Metadata preservation throughout pipeline

### Acceptance Criteria
- [ ] Born-digital PDFs processed without OCR when skip-text enabled
- [ ] Image-only pages OCR'd automatically
- [ ] Force-OCR mode processes all pages regardless of existing text
- [ ] Chunks maintain page boundary information for citations
- [ ] Text chunks target ~1000 tokens with proper overlap
- [ ] Duplicate chunks detected and skipped based on MD5
- [ ] Progress tracking works for large PDF files
- [ ] Error handling for corrupted or encrypted PDFs

---

## Sprint 3: Vector Database & Embeddings (3h)

### Objectives
- Set up Chroma persistent collections per Matter
- Implement embedding generation via Ollama
- Create vector storage and retrieval operations
- Build search functionality with metadata filtering

### Technical Requirements
- Chroma PersistentClient configuration
- Ollama embeddings API integration
- Vector upsert with metadata preservation
- Similarity search with configurable k values

### Deliverables
1. **Vector Store** (`app/vectors.py`)
   ```python
   class VectorStore:
       def __init__(self, matter_path: Path): ...
       async def upsert_chunks(self, chunks: List[Chunk]) -> None
       async def search(
           self, 
           query: str, 
           k: int = 8,
           filter_metadata: dict = None
       ) -> List[SearchResult]
   ```

2. **Embedding Provider** (`app/llm/embeddings.py`)
   ```python
   class OllamaEmbeddings(EmbeddingProvider):
       def __init__(self, model: str = "nomic-embed-text"): ...
       async def embed(self, texts: List[str]) -> List[List[float]]
       async def embed_single(self, text: str) -> List[float]
   ```

3. **Search Results** 
   ```python
   class SearchResult(BaseModel):
       chunk_id: str
       doc_name: str
       page_start: int
       page_end: int
       text: str
       similarity_score: float
       metadata: dict
   ```

### Acceptance Criteria
- [ ] Each Matter has isolated Chroma collection
- [ ] Embeddings generated consistently with Ollama
- [ ] Vector search returns results with proper metadata
- [ ] Similarity scores are meaningful and comparable
- [ ] Large document collections handle efficiently
- [ ] Collection switching works without cross-contamination
- [ ] Embedding model can be changed per Matter

---

## Sprint 4: Basic RAG Implementation (3-4h)

### Objectives
- Implement retrieval-augmented generation pipeline
- Create prompt templates for construction claims analysis
- Build LLM provider abstraction layer
- Integrate Ollama for local generation

### Technical Requirements
- Modular LLM provider system
- Prompt template management
- Context window optimization
- Citation extraction and formatting

### Deliverables
1. **LLM Provider Base** (`app/llm/base.py`)
   ```python
   class LLMProvider(Protocol):
       async def generate(
           self,
           system: str,
           messages: List[dict],
           max_tokens: int = 900,
           temperature: float = 0.2
       ) -> str
   ```

2. **Ollama Provider** (`app/llm/ollama_provider.py`)
   ```python
   class OllamaProvider(LLMProvider):
       def __init__(self, model: str = "gpt-oss:20b"): ...
       async def generate(...) -> str
   ```

3. **RAG Engine** (`app/rag.py`)
   ```python
   class RAGEngine:
       async def generate_answer(
           self,
           query: str,
           matter: Matter,
           k: int = 8
       ) -> RAGResponse
   ```

4. **Prompt Templates**
   - System prompt for construction claims analysis
   - Context formatting for retrieved chunks
   - Citation requirements and formatting rules

### Acceptance Criteria
- [ ] RAG pipeline retrieves relevant chunks based on query
- [ ] Generated answers cite sources with [DocName p.N] format
- [ ] System prompt enforces construction domain expertise
- [ ] Context window management prevents token limit issues
- [ ] Multiple LLM providers can be swapped easily
- [ ] Citation mapping enables UI source tracking
- [ ] Answers follow required format (Key Points, Analysis, Citations)

---

## Sprint 5: Letta Agent Integration (4h)

### Objectives
- Integrate Letta for persistent agent memory
- Implement matter-specific knowledge management
- Build recall and upsert operations
- Create follow-up suggestion system

### Technical Requirements
- Letta agent lifecycle management
- Knowledge item extraction from conversations
- Semantic recall from agent memory
- Integration with RAG pipeline

### Deliverables
1. **Letta Adapter** (`app/letta_adapter.py`)
   ```python
   class LettaAdapter:
       def __init__(self, matter_path: Path): ...
       async def recall(self, query: str, top_k: int = 6) -> List[KnowledgeItem]
       async def upsert_interaction(
           self,
           user_query: str,
           llm_answer: str,
           sources: List[SourceChunk],
           extracted_facts: List[KnowledgeItem]
       ) -> None
       async def suggest_followups(
           self, 
           user_query: str, 
           llm_answer: str
       ) -> List[str]
   ```

2. **Knowledge Models**
   ```python
   class KnowledgeItem(BaseModel):
       type: Literal["Entity", "Event", "Issue", "Fact"]
       label: str
       date: Optional[str] = None
       actors: List[str] = []
       doc_refs: List[dict] = []
       support_snippet: Optional[str] = None
   ```

3. **Information Extraction**
   - Post-answer fact extraction from LLM responses
   - Structured data parsing and validation
   - Domain-specific entity recognition

4. **Enhanced RAG Pipeline**
   - Memory integration in retrieval phase
   - Context enrichment with agent knowledge
   - Follow-up generation based on conversation history

### Acceptance Criteria
- [ ] Each Matter has isolated Letta agent instance
- [ ] Agent memory persists between sessions
- [ ] Facts from conversations are extracted and stored
- [ ] Subsequent queries benefit from prior context
- [ ] Follow-up suggestions are contextually relevant
- [ ] Memory recall enhances answer quality
- [ ] Domain ontology (Entities, Events, Issues, Facts) is respected

---

## Sprint 6: NiceGUI Desktop Interface - Part 1 (3-4h)

### Objectives
- Create NiceGUI desktop application with 3-pane layout
- Implement Matter management UI
- Build document upload and display interface
- Add settings panel

### Technical Requirements
- NiceGUI native desktop mode
- Responsive 3-pane layout design
- File upload with progress indicators
- Real-time status updates

### Deliverables
1. **Main Application** (`ui/main.py`)
   ```python
   def create_app():
       # NiceGUI app with native=True
       # 3-pane layout setup
       # Route handling for SPA behavior
   ```

2. **Left Pane: Matter & Documents**
   - Matter selector dropdown
   - "Create Matter" dialog
   - "Upload PDFs" multi-file selector
   - Document list with status indicators
   - Upload progress bars

3. **Settings Drawer**
   - LLM provider selection (Ollama/Gemini)
   - Model selection with availability checking
   - OCR language and options
   - Test connection buttons

4. **API Client** (`ui/api_client.py`)
   ```python
   class APIClient:
       async def create_matter(self, name: str) -> Matter
       async def upload_files(self, matter_id: str, files: List) -> str
       async def get_job_status(self, job_id: str) -> JobStatus
   ```

### Acceptance Criteria
- [ ] Desktop window launches with NiceGUI native mode
- [ ] 3-pane layout is responsive and functional
- [ ] Can create new Matter via UI form
- [ ] Matter switching updates document list
- [ ] File upload shows progress and completion status
- [ ] Settings drawer persists configuration changes
- [ ] UI remains responsive during file processing
- [ ] Error states are handled gracefully

---

## Sprint 7: NiceGUI Desktop Interface - Part 2 (3-4h)

### Objectives
- Complete chat interface with message history
- Implement sources pane with citations
- Add PDF viewer integration
- Connect UI to RAG backend

### Technical Requirements
- Chat message threading and display
- Source chunk presentation with metadata
- System PDF viewer integration
- Real-time response streaming

### Deliverables
1. **Center Pane: Chat Interface**
   - Message history with timestamps
   - User input with send button
   - Loading indicators during generation
   - Follow-up suggestion chips

2. **Right Pane: Sources Display**
   - Source chunks with document/page info
   - Similarity scores and snippets
   - "Open PDF" and "Copy Citation" buttons
   - Source highlighting for active answer

3. **PDF Integration**
   ```python
   async def open_pdf_at_page(doc_path: Path, page: int):
       # Launch system viewer with page targeting
       # Handle different PDF viewers (evince, etc.)
   ```

4. **Real-time Updates**
   - WebSocket or polling for job progress
   - Chat response streaming
   - Dynamic source loading

### Acceptance Criteria
- [ ] Chat interface displays conversation history
- [ ] Messages sent to backend and responses received
- [ ] Sources pane populates with relevant chunks
- [ ] PDF viewer opens to correct page when possible
- [ ] Follow-up suggestions are clickable and functional
- [ ] UI updates in real-time during processing
- [ ] Copy citation functionality works properly

---

## Sprint 8: Advanced RAG Features (3h)

### Objectives
- Enhance citation tracking and accuracy
- Implement follow-up suggestion generation
- Add memory-based response improvement
- Optimize retrieval with hybrid scoring

### Technical Requirements
- Citation-to-source mapping system
- Follow-up generation prompts
- Memory-augmented retrieval
- Scoring algorithm refinements

### Deliverables
1. **Enhanced Citation System**
   - Precise chunk-to-citation mapping
   - Multi-page span handling
   - Citation validation and correction

2. **Follow-up Generation**
   - Context-aware suggestion prompts
   - Domain-specific follow-up templates
   - Conversation history integration

3. **Memory-Enhanced Retrieval**
   - Combined vector + memory search
   - Relevance scoring improvements
   - Context window optimization

4. **Response Quality Metrics**
   - Citation coverage analysis
   - Source diversity scoring
   - Follow-up relevance evaluation

### Acceptance Criteria
- [ ] Citations accurately map to source documents and pages
- [ ] Follow-up suggestions are contextually relevant
- [ ] Agent memory improves subsequent answer quality
- [ ] Retrieval balances recency and relevance effectively
- [ ] Multi-document queries cite diverse sources appropriately
- [ ] Suggested follow-ups lead to productive conversations

---

## Sprint 9: LLM Provider Management (2-3h)

### Objectives
- Implement Gemini API integration
- Add provider switching without restart
- Create model testing and validation
- Build consent and privacy controls

### Technical Requirements
- Google Gemini API client
- Runtime provider switching
- API key management
- Privacy consent workflow

### Deliverables
1. **Gemini Provider** (`app/llm/gemini_provider.py`)
   ```python
   class GeminiProvider(LLMProvider):
       def __init__(self, api_key: str, model: str): ...
       async def generate(...) -> str
   ```

2. **Provider Management**
   - Runtime switching between Ollama and Gemini
   - Model availability checking
   - Connection testing and validation

3. **Privacy Controls**
   - First-time consent dialog for external LLMs
   - Clear data usage notifications
   - Local-first preference enforcement

4. **Settings Integration**
   - API key secure storage
   - Model parameter configuration
   - Provider-specific options

### Acceptance Criteria
- [ ] Can switch from Ollama to Gemini without restart
- [ ] API key validation works correctly
- [ ] Privacy consent shown before external API use
- [ ] Model testing confirms connectivity
- [ ] Provider switching preserves conversation context
- [ ] Configuration persists between sessions

---

## Sprint 10: Job Queue & Background Processing (3h)

### Objectives
- Implement async job queue for long-running tasks
- Add progress tracking and status reporting
- Create cancellation and retry mechanisms
- Optimize UI responsiveness

### Technical Requirements
- AsyncIO-based job queue
- Progress callback system
- Job persistence and recovery
- Error handling and retries

### Deliverables
1. **Job Queue System** (`app/jobs.py`)
   ```python
   class JobQueue:
       async def submit_job(
           self, 
           job_type: str, 
           params: dict,
           progress_callback: Callable = None
       ) -> str
       async def get_job_status(self, job_id: str) -> JobStatus
       async def cancel_job(self, job_id: str) -> bool
   ```

2. **Job Types**
   - PDF ingestion (OCR + parsing + embedding)
   - Large model operations
   - Batch document processing

3. **Progress Tracking**
   - Real-time progress updates
   - Detailed status messages
   - Error reporting and recovery

4. **UI Integration**
   - Progress bars and status indicators
   - Cancellation controls
   - Background task notifications

### Acceptance Criteria
- [ ] Large PDF uploads don't block UI
- [ ] Progress indicators update in real-time
- [ ] Jobs can be cancelled cleanly
- [ ] Failed jobs provide clear error messages
- [ ] Multiple jobs can run concurrently
- [ ] Job status persists across app restarts

---

## Sprint 11: Error Handling & Edge Cases (2-3h)

### Objectives
- Implement comprehensive error handling
- Add retry mechanisms for failures
- Create graceful degradation modes
- Handle edge cases and corrupted data

### Technical Requirements
- Error classification and recovery
- User-friendly error messages
- Fallback modes for service failures
- Data validation and sanitization

### Deliverables
1. **Error Handling Framework**
   - Centralized error logging and reporting
   - User-friendly error messages
   - Recovery suggestions and actions

2. **Retry Mechanisms**
   - Exponential backoff for API calls
   - Manual retry controls in UI
   - Automatic recovery for transient failures

3. **Edge Case Handling**
   - Encrypted PDFs
   - Corrupted documents
   - Network connectivity issues
   - Disk space limitations
   - Model availability problems

4. **Graceful Degradation**
   - Fallback to local models when external APIs fail
   - Reduced functionality modes
   - Clear capability limitations communication

### Acceptance Criteria
- [ ] App handles corrupted PDFs gracefully
- [ ] Network failures don't crash the application
- [ ] Missing models trigger helpful installation prompts
- [ ] Disk space issues are detected and communicated
- [ ] API rate limits are respected and handled
- [ ] Users can retry failed operations easily

---

## Sprint 12: Testing & Polish (3-4h)

### Objectives
- Create comprehensive test suite
- Implement integration testing
- Add performance optimization
- Polish user experience

### Technical Requirements
- Unit tests for core components
- Integration tests for full workflows
- Performance benchmarking
- UI/UX improvements

### Deliverables
1. **Unit Tests** (`tests/unit/`)
   - Matter management operations
   - PDF parsing and chunking
   - Vector operations and search
   - LLM provider implementations

2. **Integration Tests** (`tests/integration/`)
   - End-to-end ingestion pipeline
   - RAG workflow with real documents
   - Provider switching scenarios
   - Multi-matter isolation

3. **Performance Optimization**
   - Embedding batch processing
   - Memory usage optimization
   - Response time improvements
   - Resource cleanup

4. **UI Polish**
   - Loading states and animations
   - Error state styling
   - Keyboard shortcuts
   - Accessibility improvements

### Acceptance Criteria
- [ ] All unit tests pass consistently
- [ ] Integration tests cover major user workflows
- [ ] Application performance meets targets (§19)
- [ ] UI provides clear feedback for all user actions
- [ ] Memory usage is stable over long sessions
- [ ] Error states are visually clear and actionable

---

## Sprint 13: Production Readiness (2-3h)

### Objectives
- Finalize configuration management
- Create deployment documentation
- Add monitoring and logging
- Prepare distribution package

### Technical Requirements
- Production configuration validation
- Comprehensive logging setup
- Resource monitoring
- Package distribution preparation

### Deliverables
1. **Production Configuration**
   - Environment-specific settings
   - Security configuration validation
   - Resource limits and monitoring

2. **Documentation**
   - Installation guide for Ubuntu
   - Configuration reference
   - Troubleshooting guide
   - User manual sections

3. **Logging & Monitoring**
   - Structured application logging
   - Performance metrics collection
   - Error tracking and alerting

4. **Distribution Package**
   - Requirements.txt finalization
   - Setup script for dependencies
   - Basic packaging with PyInstaller (optional)

### Acceptance Criteria
- [ ] Application runs cleanly in production environment
- [ ] All dependencies are properly documented and versioned
- [ ] Logging provides adequate troubleshooting information
- [ ] Configuration is validated on startup
- [ ] Resource usage is monitored and reported
- [ ] Installation process is documented and tested

---

## Post-Sprint Validation Checklist

After completing all sprints, verify against the milestone checklist from §30:

- [ ] Create/Switch Matter
- [ ] Upload PDFs → background OCR+ingest → progress and stats
- [ ] Ask 3 questions → answers show **Citations** and **Sources pane** with page-level snippets
- [ ] Prior fact is recalled in later answers (Agent Knowledge proven)
- [ ] Follow-ups appear after each answer
- [ ] Model switching (Ollama ↔ Gemini) works without restart
- [ ] All data stored under the Matter folder

---

## Development Notes

### Critical Dependencies
- **Ollama**: Must be running with models pulled
- **System Packages**: OCRmyPDF, Tesseract, Poppler
- **Letta**: Version compatibility crucial for agent persistence
- **NiceGUI**: Native desktop mode requirements

### Technical Risks
- **Model Performance**: `gpt-oss:20b` may be slow on modest hardware
- **OCR Quality**: Mixed documents may require manual force-OCR
- **Memory Usage**: Large document collections may require optimization
- **Native UI**: Fallback to browser mode may be necessary

### Success Metrics
- **Responsiveness**: UI remains interactive during processing
- **Accuracy**: Citations map correctly to source documents
- **Memory**: Agent knowledge demonstrably improves over sessions
- **Usability**: Non-technical users can complete core workflows