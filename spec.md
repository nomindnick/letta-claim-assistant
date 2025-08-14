# “Letta” Construction Claim Assistant — Proof of Concept (PoC) Specification

**Version:** 1.1
**Target OS:** Ubuntu Linux (desktop)
**Goal:** A local‑first desktop PoC that analyzes construction‑claim PDFs using RAG + a **stateful agent**. The agent’s **Matter‑specific memory** (via Letta) compounds over time so answers become context‑aware, traceable, and proactive.

---

## 1) Overview & Vision

* **Problem:** Lawyers analyzing construction claims need fast answers, traceable sources, and continuity across sessions.
* **Solution:** A desktop app that:

  * Ingests PDFs → parses/embeds locally → retrieves relevant chunks.
  * Uses **Letta** to build **persistent Agent Knowledge** per **Matter** (case).
  * Answers questions in a chat UI, **citing page‑level sources**.
  * Suggests **follow‑up lines of inquiry** based on evolving knowledge.
* **Why local:** Privacy, speed, and control. Data stays on disk unless the user explicitly chooses an external LLM.

---

## 2) Scope

### In Scope (PoC)

* Desktop UI (3 panes) built for **fastest setup** using **NiceGUI** (`ui.run(native=True)`).
* **PDF only** ingestion (born‑digital, scanned, or mixed).
* **OCR** (auto for image pages; optional “force OCR”).
* RAG over **local vectors** (Chroma) per Matter.
* **Letta**-powered, Matter‑scoped Agent Knowledge.
* Switchable **LLM providers**: local **Ollama** (default: `gpt-oss:20b`) or external **Gemini** (2.5 Flash).
* Local, per‑Matter storage layout.

### Out of Scope (PoC)

* Multi‑user auth/roles.
* Cloud/web deployment.
* Non‑PDF formats.
* Tagging/folders, collaboration, advanced theming.

---

## 3) Core Concepts

* **Matter**: An isolated workspace for a single claim (docs, vectors, memory, logs).
* **Agent Knowledge**: Letta‑backed, structured memory for that Matter (entities, events, issues, facts, relationships).

### Lightweight Domain Ontology

* **Entities:** Party (Owner/Contractor/Sub), Person, Organization, Project, Document, Clause, Location.
* **Events:** Delay, RFI, Change Order, Non‑Conformance, Inspection, Payment, Failure (e.g., Dry‑Well Failure).
* **Issues:** Differing Site Conditions, Design Defect, Schedule Delay, LDs, Extra Work, Prompt Pay.
* **Relations:** `involves(Party, Event)`, `mentions(Document, X)`, `supports/contradicts(Document, Fact)`, `causes(Event, Issue)`.

---

## 4) User Stories → Acceptance Criteria

**US1 – Create Matter**

* When I enter a unique name and click **Create Matter**, a new folder tree is created (see §8), a fresh Chroma collection and Letta agent are initialized, and the Matter becomes active.

**US2 – Switch Matters**

* Selecting another Matter updates the doc list, chat history, and memory stats with **no cross‑leakage**.

**US3 – Upload PDFs**

* Upload 1..N PDFs. Background job runs OCR (image pages only by default), parse, chunk, embed. UI shows progress; on completion: page count, chunk count, and “OCR: full/partial/none”.

**US4 – Ask Questions**

* Enter a question in chat. Receive a structured answer and a **Sources** pane with page‑level snippets (doc name + page). UI stays responsive.

**US5 – Traceability**

* Each snippet is labeled `[DocName.pdf p.N sim=…]`; clicking “Open” launches the system PDF viewer **at that page** (or nearest supported equivalent).

**US6 – Memory Within Matter**

* Facts established in prior turns (e.g., “Dry well failed on 2023‑02‑14”) inform future answers without restating.

**US7 – Proactive Suggestions**

* After each answer, 2–4 follow‑up suggestions appear, grounded in Agent Knowledge and the current context.

**US8 – LLM Switching**

* Settings allows choosing **Ollama** (local) or **Gemini** (external), selecting model, and testing connectivity without restarting the app.

---

## 5) Non‑Functional Requirements

* **Local‑first:** All PDFs, vectors, and memory on disk under `~/LettaClaims/<Matter_...>/`.
* **Responsiveness:** All heavy tasks are **non‑blocking**; show “processing/thinking” indicators.
* **Reliability:** If OCR/parse/embed fails for a file, show a non‑blocking error with retry actions (“Force OCR”, re‑parse).
* **Privacy:** No data leaves the machine unless the user selects an external provider (Gemini). Show one‑time consent when enabling external LLMs.

---

## 6) Architecture

```
[ NiceGUI Desktop (native=True) ]
          |  (async calls)
  [ Ingestion + RAG + Letta services ]   # single Python app, asyncio jobs
          |
  +-------+-----------------------+------------------+
  |                               |                  |
[Chroma (per Matter)]      [Ollama (gen+embed)]   [Gemini API (opt-in)]
   local disk                local/localhost          external, guarded
```

* **UI:** NiceGUI 3‑pane desktop (single‑process with internal FastAPI router mounted at `/api` for clean boundaries).
* **Jobs:** Simple **async job queue** (e.g., `asyncio.Queue`) for ingestion and long LLM calls; job status polled by UI.
* **Vectors:** **Chroma PersistentClient** → each Matter has its own collection directory.
* **Embeddings:** via **Ollama** (default: `nomic-embed-text`), switchable.
* **Agent:** **Letta** (per‑Matter agent state persisted under `knowledge/`).

---

## 7) Filesystem Layout (per Matter)

```
~/LettaClaims/
  Matter_<slug>/
    config.json
    docs/                   # originals
    docs_ocr/               # OCR’d PDFs (+ optional sidecar .txt)
    parsed/                 # <docId>.jsonl (page_no, text, blocks?, md5)
    vectors/                # chroma/ (persistent collection)
      chroma/
    knowledge/
      letta_state/          # Letta agent persistence
      graph.sqlite3         # optional local graph mirror (PoC optional)
    chat/
      history.jsonl         # user/assistant turns + metadata
    logs/
      app.log
```

---

## 8) GUI (NiceGUI) — Minimal 3‑Pane UX

**Pane 1: Matter & Docs (left)**

* “Create Matter” (unique name → slug).
* Matter selector (by recency).
* “Upload PDFs” (multi‑file).
* Document list: `name.pdf — pages: N, chunks: M — OCR: full/partial/none — status ✓/…`

**Pane 2: Chat (center)**

* Scrollable message history (timestamps).
* Input box + “Send”.
* Spinner on pending turns.
* Chips: **Suggested follow‑ups** after each answer.

**Pane 3: Sources (right)**

* For the selected answer:

  * Rows of **source chunks**: `[Doc p.N sim=0.84]` + snippet (≤600 chars).
  * Buttons: **Open** (launch system viewer at page), **Copy citation**.

**Settings Drawer (🛠)**

* **Provider:** Local (Ollama) / External (Gemini).
* **Local model:** default `gpt-oss:20b`; dropdown lists `ollama list` models; “Pull model” button if missing.
* **Embeddings:** default `nomic-embed-text`; alternatives (e.g., `mxbai-embed-large`, `bge-m3`).
* **OCR:** language (default `eng`), **Skip‑text** (on), **Force OCR** toggle.
* **Test buttons:** “Test local model”, “Test Gemini key”.

**Desktop launch:** `ui.run(native=True)`; fallback to browser if native window fails.

---

## 9) Ingestion Pipeline (with OCR)

**Trigger:** On “Upload PDFs”.

1. **OCR phase**

   * Run **OCRmyPDF** with `--skip-text` (default) to OCR **only image‑only pages**; mixed PDFs retain original text pages.
   * If user enables **Force OCR**, run with `--force-ocr`.
   * Output to `docs_ocr/<file>.ocr.pdf` (+ optional `.txt` sidecar).

2. **Parsing (PyMuPDF)**

   * Read the **OCR’d** PDF.
   * Extract page‑aligned text (and optionally block order for heuristics).
   * Store per‑page records → `parsed/<docId>.jsonl` (fields: `page_no`, `text`, `doc_name`, `md5`).

3. **Chunking**

   * **Structure pass:** split on headings/section labels (e.g., numbered headings, “CHANGE ORDER”, “RFI”, etc.) and page breaks.
   * **Windowing pass:** target \~**1000 tokens** (\~4k chars) with **15% overlap**.
   * **Metadata:** `doc_id`, `doc_name`, `page_start`, `page_end`, `section_title?`, `md5`.

4. **Embeddings & Index**

   * Embed via **Ollama embeddings** (default `nomic-embed-text`).
   * Upsert vectors + metadata into **Chroma collection** under the Matter’s `vectors/chroma/`.

5. **Index Stats**

   * Save counts (pages, chunks), times (parse, embed), OCR status.

**Retry/Edge Cases**

* Encrypted/bad pages → show a warning with **Retry** and/or **Force OCR**.
* Duplicate chunks (identical MD5) → skip to avoid index bloat.

---

## 10) Retrieval‑Augmented Generation (RAG) Flow

1. **Vector search** on the active Matter’s Chroma (default **top‑k: 8**; MMR optional).
2. **Letta recall**: fetch up to **k\_mem: 6** relevant Agent Knowledge items (facts/events/entities/issues).
3. **Prompt assembly** (see §16):

   * System rules (construction‑claims analyst).
   * Context blocks: **MEMORY\[n]** then **DOC\[n]** (chunk label, doc, page(s), text).
   * Style: Key Points → Analysis → Citations → Suggested Follow‑ups.
4. **LLM call** via chosen provider.
5. **Display** answer + sources; **post‑answer IE** extracts structured facts and updates Letta; generate follow‑ups.

**Citation policy**

* Require model to cite `[DocName p.N]` that **directly support** key points.
* Maintain a mapping from chunk labels → doc/page so the UI can render Sources accurately.

---

## 11) Agent Knowledge (Letta)

**Purpose:** Give the agent **persistent, queryable context** per Matter.

**Letta Adapter Interface (`letta_adapter.py`)**

```python
class KnowledgeItem(TypedDict):
    type: Literal["Entity","Event","Issue","Fact"]
    label: str
    date: str | None
    actors: list[str]
    doc_refs: list[dict]  # {doc: str, page: int}
    support_snippet: str | None

class LettaAdapter:
    def __init__(self, matter_path: Path): ...
    def recall(self, query: str, top_k: int = 6) -> list[KnowledgeItem]: ...
    def upsert_interaction(
        self,
        user_query: str,
        llm_answer: str,
        sources: list[SourceChunk],
        extracted_facts: list[KnowledgeItem],
    ) -> None: ...
    def suggest_followups(self, user_query: str, llm_answer: str) -> list[str]: ...
```

**Initialization**

* On Matter creation, create a **new Letta agent** and persist under `knowledge/letta_state/`.
* Optionally enable **Letta Filesystem** (keeps doc handles for better references).

**Data flow**

* **Recall:** semantic fetch of related memory objects to include in the prompt.
* **Upsert:** after each Q\&A, write back structured items (events, facts, issues) linked to doc/page.
* **Suggestions:** small prompt that proposes 2–4 concise follow‑ups grounded in memory.

---

## 12) LLM Providers & Models

**Default local provider:** **Ollama**

* **Generation model (default):** `gpt-oss:20b` (switchable in Settings).
* **Embeddings (default):** `nomic-embed-text` (alternatives available).

**External provider (opt‑in):** **Gemini**

* **Model:** `gemini-2.5-flash` (configurable).
* **Consent:** show a one‑time notice that questions/context are sent to Google if enabled.

**Switching models**

* Settings allow changing provider, model, temperature, and max tokens.
* A “Test model” button performs a short round‑trip generation.

---

## 13) Backend Surface (FastAPI mounted at `/api`)

> Even though NiceGUI can call Python directly, using a small HTTP surface keeps things clean and portable.

### 13.1 Matter Management

* `POST /api/matters`
  **Body:** `{ "name": "New Claim – Dry Well" }`
  **200:** `{ "id": "...", "slug": "new-claim-dry-well", "paths": {...} }`

* `GET /api/matters` → list of matters with minimal stats

* `POST /api/matters/{id}/switch` → sets active Matter

### 13.2 Upload & Ingestion

* `POST /api/matters/{id}/upload` (multipart) → `{ "job_id": "..." }`
* `GET /api/jobs/{job_id}` → `{ status: "queued|running|done|error", progress: 0..1, detail?: str }`

### 13.3 Chat / RAG

* `POST /api/chat`
  **Body:**

  ```json
  {
    "matter_id": "abc123",
    "query": "What caused the dry well failure?",
    "k": 8,
    "model": "active",
    "max_tokens": 900
  }
  ```

  **200:**

  ```json
  {
    "answer": "...",
    "sources": [
      {"doc":"Spec_2021.pdf","page_start":12,"page_end":12,"text":"...","score":0.84}
    ],
    "followups": ["Check whether CO-12 references RFI-103", "..."],
    "used_memory": [
      {"type":"Event","label":"Dry Well Failure","date":"2023-02-14","actors":["Contractor X"],"doc_refs":[{"doc":"DailyLog_2023-02-15.pdf","page":2}],"support_snippet":"..."}
    ]
  }
  ```

### 13.4 Settings / Models

* `GET /api/settings/models` → `{ provider, generation_model, embedding_model, local_models:[...]}`
* `POST /api/settings/models` → set provider/model and (if Gemini) API key; run a short test and return `ok: true/false`.

---

## 14) Data Models (Pydantic)

```python
class SourceChunk(BaseModel):
    doc: str
    page_start: int
    page_end: int
    text: str
    score: float

class KnowledgeItem(BaseModel):
    type: Literal["Entity","Event","Issue","Fact"]
    label: str
    date: str | None = None
    actors: list[str] = []
    doc_refs: list[dict] = []
    support_snippet: str | None = None

class ChatRequest(BaseModel):
    matter_id: str
    query: str
    k: int = 8
    model: str | None = None
    max_tokens: int = 900

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    followups: list[str]
    used_memory: list[KnowledgeItem]
```

---

## 15) Prompting & Output Contracts

**System Prompt (prepend each chat call)**

> You are a construction‑claims analyst assisting an attorney.
> Answer using the MEMORY and DOC contexts provided.
> Be precise and conservative—if uncertain or sources conflict, say so explicitly.
> Cite sources with `[DocName p.N]` that directly support key points.
> Do not invent facts or citations.

**Answer Format (enforced in final prompt)**

```
1) Key Points
   - ...
2) Analysis
   ...
3) Citations
   - [Spec_2021.pdf p.12]
   - [DailyLog_2023-02-15.pdf p.2]
4) Suggested Follow-ups
   - ...
```

**Post‑Answer Information Extraction (IE) Prompt**

```
Extract structured items as a JSON array. Each item:
- type: "Event" | "Fact" | "Issue" | "Entity"
- label: string
- date?: ISO8601 or null
- actors?: [string]
- doc_refs?: [{doc: string, page: number}]
- support_snippet?: string (<=300 chars)
Return only valid JSON (array).
```

**Suggestions Prompt (for Letta adapter)**

```
Given the user's question, the answer, and the MEMORY items,
propose 2–4 concise, concrete follow-up questions that help a
construction-claims lawyer uncover causation, responsibility, or damages.
Each ≤ 18 words. No fluff.
```

---

## 16) Retrieval & Scoring

* **Top‑k:** 8 chunks by default.
* **Hybrid score (optional):** `0.7 * similarity + 0.3 * recency_boost` if chunk metadata dates exist.
* **MMR toggle:** to reduce redundancy when many similar chunks exist.

---

## 17) Error Handling

* **OCR failures:** present non‑blocking toast + allow “Force OCR” retry.
* **Parse/Embed failures:** mark the doc row with an error icon and “Retry” action.
* **Model missing:** show “Pull model” button in Settings (Ollama).
* **Gemini invalid key:** show a red banner until test passes.
* **Letta unavailable:** run in stateless mode with a small banner (“Agent Knowledge disabled”).

---

## 18) Security & Privacy

* All data is local under `~/LettaClaims`.
* External LLM calls (Gemini) **only** if selected; show one‑time consent explaining that prompts and selected context will be sent to the provider.
* Logs avoid dumping full chunk texts (store doc/page refs + hashes).

---

## 19) Performance Targets (PoC guidance)

* **Ingestion:** A 200‑page spec + 50 pages of logs completes OCR+parse+embed within a few minutes on typical dev hardware (background job).
* **Query latency:** Depends on hardware and model. If `gpt-oss:20b` is slow, switch to a lighter local model in Settings.
* **UI:** Main thread never blocked; animations remain smooth during jobs.

---

## 20) Installation (Ubuntu)

**System packages**

```bash
sudo apt-get update
sudo apt-get install -y \
  ocrmypdf tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
  poppler-utils
```

> Add more Tesseract language packs as needed (e.g., `tesseract-ocr-spa`).

**Python environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install nicegui chromadb pymupdf pydantic uvicorn structlog ollama google-genai
```

**Ollama + models**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
# Optional embeddings:
# ollama pull mxbai-embed-large
# ollama pull bge-m3
```

**Gemini (optional external)**

* `pip install google-genai`
* Set API key in Settings when choosing Gemini as provider.

---

## 21) Run

* Start the app (single process):

```bash
python main.py
```

* The UI opens as a desktop window (`native=True`). If native launch fails, it will open in your browser.

---

## 22) Module Layout (suggested)

```
app/
  __init__.py
  settings.py
  logging_conf.py

  # Core services
  matters.py            # create/switch/list + filesystem ops
  ingest.py             # OCR → parse → chunk → embed
  vectors.py            # Chroma helpers (PersistentClient per Matter)
  rag.py                # retrieval + prompt assembly + post-answer IE
  jobs.py               # asyncio job queue + progress tracking

  # LLM providers
  llm/
    base.py             # Provider protocol
    ollama_provider.py  # local gen + embeddings
    gemini_provider.py  # external gen
    embeddings.py       # embedding model abstraction

  # Agent Knowledge
  letta_adapter.py      # Letta integration (recall/upsert/suggest)
  suggestions.py        # small helper for follow-ups

  # API surface (mounted under /api)
  api.py                # FastAPI router: matters, upload, jobs, chat, settings

ui/
  main.py               # NiceGUI app, panes, settings drawer
  api_client.py         # thin HTTP client to /api
  widgets/              # optional: modular UI components

tests/
  unit/
  integration/
```

---

## 23) Key Implementation Notes

* **OCR**

  * Default: `ocrmypdf --skip-text input.pdf output.ocr.pdf`
  * Force: `ocrmypdf --force-ocr ...`
  * Use the OCR’d PDF for parsing.

* **Open PDF at Page**

  * On “Open” in Sources, attempt:

    * `evince -i <page_index> "<path>"` (page index is zero‑based in many viewers).
    * If not supported, open file and display a tooltip that the exact page jump may vary by viewer.

* **Chunking**

  * Use a simple tokenizer or character‑based window to approximate **\~1000 tokens** (≈ 4000 chars).
  * 15% overlap (≈ 600 chars).
  * Store `page_start/page_end` for citation.

* **Embeddings**

  * Default `nomic-embed-text` via Ollama embeddings API.
  * Configure embedding model independently of generation model.

* **Chroma**

  * Create one collection per Matter; path points to `vectors/chroma/`.

---

## 24) Testing Plan

**Unit**

* Ingestion:

  * OCR “image‑only” PDF produces searchable text.
  * Chunker respects section boundaries and window sizes.
  * Duplicate chunk dedupe by `md5` works.
* Vectors:

  * Upsert/retrieve round‑trip; metadata preserved.
* Letta:

  * `recall()` returns typed items; `upsert_interaction()` persists and can be recalled.
* Providers:

  * Ollama & Gemini providers conform to `LLMProvider` interface.

**Integration**

* Create/Switch Matter; ensure isolation of vectors and memory.
* Upload 3 PDFs (specs, daily logs, emails) → answer timeline Q with ≥3 unique citations.
* Proactive suggestions reflect memory (e.g., mention an RFI referenced earlier).
* Switch from `gpt-oss:20b` to a lighter local model and back; then to Gemini and back.

**UX**

* UI remains responsive during ingestion of a large PDF (progress visibly moves).
* “Open” in Sources successfully launches viewer.

---

## 25) Build/Packaging (optional for PoC)

* Keep it as a Python app for now.
* For distribution, use **PyInstaller** to bundle `ui/main.py` and include data files.
* Alternatively, ship a simple launcher shell script.

---

## 26) Future Enhancements (beyond PoC)

* Timeline visualization from Agent Knowledge (events over time).
* Contradiction detection (e.g., Spec §2.1 vs CO‑12).
* Clause graph and cross‑references.
* Hybrid retrieval (keyword + dense).
* Multi‑Matter dashboard with stats.

---

## 27) Configuration Files

**Global app config:** `~/.letta-claim/config.toml`

```toml
[ui]
framework = "nicegui"
native = true

[llm]
provider = "ollama"              # "ollama" | "gemini"
model = "gpt-oss:20b"
temperature = 0.2
max_tokens = 900

[embeddings]
provider = "ollama"
model = "nomic-embed-text"

[ocr]
enabled = true
force_ocr = false
language = "eng"
skip_text = true                 # only OCR pages without text

[paths]
root = "~/LettaClaims"

[gemini]
api_key = ""
model = "gemini-2.5-flash"

[letta]
enable_filesystem = true
```

**Per‑Matter config:** `Matter_<slug>/config.json`

```json
{
  "id": "abc123",
  "name": "Dry Well Claim",
  "created_at": "2025-08-14T12:00:00Z",
  "embedding_model": "nomic-embed-text",
  "generation_model": "gpt-oss:20b",
  "vector_path": "vectors/chroma",
  "letta_path": "knowledge/letta_state"
}
```

---

## 28) Minimal Interfaces (for code generation)

**LLM Provider Protocol (`llm/base.py`)**

```python
class LLMProvider(Protocol):
    async def generate(
        self,
        system: str,
        messages: list[dict],   # [{"role":"user"|"assistant"|"system","content":"..."}]
        max_tokens: int,
        temperature: float
    ) -> str: ...

class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

**Ollama Provider (`llm/ollama_provider.py`)**

```python
class OllamaProvider(LLMProvider):
    def __init__(self, model: str): ...
    async def generate(...): ...
class OllamaEmbeddings(EmbeddingProvider):
    def __init__(self, model: str): ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

**Gemini Provider (`llm/gemini_provider.py`)**

```python
class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str): ...
    async def generate(...): ...
```

---

## 29) RAG Assembly (pseudo)

```python
# 1) retrieve chunks from Chroma
chunks = vectors.search(query, k=request.k)

# 2) recall Letta memory
mem_items = letta.recall(query, top_k=6)

# 3) build messages
system = SYSTEM_PROMPT
context_blocks = format_memory(mem_items) + format_chunks(chunks)
messages = [
  {"role":"system","content":system},
  {"role":"user","content": f"{query}\n\nCONTEXT:\n{context_blocks}\n\nFollow the required output format."}
]

# 4) generate
answer = provider.generate(system, messages, max_tokens, temperature)

# 5) post-answer IE + upsert
facts = extract_facts(answer, chunks)
letta.upsert_interaction(query, answer, chunks, facts)

# 6) suggestions 
followups = letta.suggest_followups(query, answer)
```

---

## 30) Milestone “Done” Checklist

* [ ] Create/Switch Matter.
* [ ] Upload PDFs → background OCR+ingest → progress and stats.
* [ ] Ask 3 questions → answers show **Citations** and **Sources pane** with page‑level snippets.
* [ ] Prior fact is recalled in later answers (Agent Knowledge proven).
* [ ] Follow‑ups appear after each answer.
* [ ] Model switching (Ollama ↔ Gemini) works without restart.
* [ ] All data stored under the Matter folder.

---


