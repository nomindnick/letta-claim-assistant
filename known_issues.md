# Known Issues

## Document Chunking Strategy Not Optimized for Legal Documents

**Issue Date:** 2025-08-19  
**Priority:** High  
**Component:** Document Processing / Chunking

### Current Behavior
The document chunking system is producing too few chunks for large legal documents:
- 133-page document → only 31 chunks
- Average ~980 tokens per chunk (targeting 1000)
- Only ~940 characters extracted per page (typical documents have 2000-3000)

### Problems
1. **Text Combination Before Chunking**: The chunker combines ALL pages in a section into one long text string, then chunks it. With only 2 sections detected for 133 pages, it treats the document as mostly continuous text.

2. **Aggressive Deduplication**: MD5 hashing removes duplicate chunks, which might be removing important repeated content (headers/footers, repeated legal language that is intentionally duplicated).

3. **Loss of Page Boundaries**: Legal documents require precise page citations, but current chunking loses page boundary information when combining text.

4. **Too Few Chunks**: 31 chunks for 133 pages means each chunk covers ~4.3 pages on average, making precise retrieval difficult.

5. **Context Loss**: Important context might be split across chunks inappropriately.

### Impact on Legal/Construction Documents
- Cannot provide precise page citations required for legal work
- Repeated clauses/terms (intentionally duplicated in contracts) are removed
- Retrieval is too coarse-grained for specific clause or section lookup
- Context windows are too large for precise Q&A

### Recommended Solution
For legal/construction documents, implement:
- **Smaller chunks**: 200-500 tokens for precision
- **Page-aware chunking**: Preserve page boundaries for citations
- **Higher overlap**: 25-30% overlap to maintain context
- **No deduplication**: Keep repeated content for legal accuracy
- **Hierarchical chunking**: Preserve document structure (sections, subsections, clauses)
- **Metadata preservation**: Keep page numbers, section titles, document structure

### Expected Metrics
For a 133-page legal document:
- With 500-token chunks: ~250-400 chunks
- With page preservation: At least 133 chunks (one per page minimum)
- With proper overlap: 300-500 chunks total

### Files to Modify
- `/app/chunking.py` - Core chunking logic
- `/app/ingest.py` - Document ingestion pipeline configuration

### Related Code
- `TextChunker` class in `app/chunking.py`
- `chunk_document()` method that combines pages before chunking
- `_deduplicate_chunks()` method that removes duplicates via MD5

### Workaround
Currently none. Users processing legal documents will experience poor retrieval accuracy and cannot get precise citations.