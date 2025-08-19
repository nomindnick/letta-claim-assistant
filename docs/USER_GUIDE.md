# Letta Construction Claim Assistant - User Guide

Welcome to the Letta Construction Claim Assistant! This guide will help you effectively use the application to analyze construction claim documents with AI-powered insights.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Matter Management](#matter-management)
4. [Document Upload and Processing](#document-upload-and-processing)
5. [Asking Questions and Chat Interface](#asking-questions-and-chat-interface)
6. [Understanding Sources and Citations](#understanding-sources-and-citations)
7. [Configuration and Settings](#configuration-and-settings)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### First Launch

1. **Start the Application**
   ```bash
   python main.py
   ```
   The application opens in desktop mode by default.

2. **Initial Setup**
   - The application validates your system on first run
   - If any issues are found, follow the suggestions provided
   - Check that Ollama is running and models are available

3. **Interface Overview**
   The application has a 3-pane layout:
   - **Left Pane**: Matter management and document list
   - **Center Pane**: Chat interface for questions and answers
   - **Right Pane**: Sources and citations for responses

### Your First Matter

1. Click **"Create Matter"** in the left pane
2. Enter a descriptive name (e.g., "Dry Well Claim - Project ABC")
3. The application creates a folder structure under `~/LettaClaims/`
4. Your new matter becomes active automatically

## Core Concepts

### What is a Matter?

A **Matter** represents a single construction claim or case. Each matter has:
- **Isolated data**: Documents, analysis, and AI memory are completely separate
- **Persistent memory**: The AI remembers facts and context across sessions
- **Document library**: All PDFs and processed content for this case
- **Chat history**: All your questions and the AI's responses

### AI Memory and Learning

The application uses **Letta** for persistent AI memory that makes your assistant smarter about your specific case over time:

#### How Memory Works
- **Entity Recognition**: Automatically identifies parties, people, organizations, projects, amounts, and dates
- **Event Tracking**: Records delays, RFIs, change orders, inspections, and other key events
- **Issue Identification**: Recognizes differing site conditions, design defects, delays, and claims
- **Fact Accumulation**: Key facts and relationships build over time across all interactions
- **Cross-Reference Memory**: Connects information between documents and conversations

#### Visual Memory Indicators
- **Purple Badge**: "Memory Enhanced" appears on responses using stored knowledge
- **Memory Dashboard**: Right panel shows memory item count, connection status, and last sync
- **Agent Health**: Header shows agent status (green = healthy, yellow = issues, red = error)
- **Memory Tooltips**: Hover over enhanced messages to see which memories were used
- **Operation Toasts**: Notifications show when storing or recalling from memory

#### California Construction Specialization
The agent has specialized knowledge for California public works construction:
- **Statutory Recognition**: Identifies California Public Contract Code, Government Code references
- **Agency Awareness**: Recognizes Caltrans, DGS, DSA, counties, school districts, and their requirements
- **Deadline Tracking**: Calculates statutory deadlines with day counts and consequences
- **Compliance Validation**: Checks prevailing wage, DIR compliance, government claim requirements
- **Expert Triggers**: Suggests when specialized analysis (quantum, delay, legal) may be needed

#### Memory Quality and Management
- **Importance Scoring**: More critical information prioritized for recall
- **Deduplication**: Prevents storing duplicate or redundant information
- **Recency Weighting**: Recent conversations weighted more heavily in context
- **Automatic Pruning**: Old or low-importance items can be automatically removed
- **Export/Import**: Memory can be backed up, exported, or transferred between systems

*For detailed memory feature documentation, see the [Letta User Guide](LETTA_USER_GUIDE.md).*

### Local-First Privacy

- All processing happens on your machine by default
- Documents never leave your computer without explicit consent
- External AI services (like Gemini) require one-time consent
- All data stored under `~/LettaClaims/` with user-only access

## Matter Management

### Creating a New Matter

1. **Click "Create Matter"** in the left pane
2. **Enter Matter Name**: Use descriptive names like:
   - "Smith Residence - Water Damage Claim"
   - "Highway 101 Project - Delay Claim"
   - "Office Building - Design Defect"
3. **Matter Created**: The system creates a unique folder structure
4. **Auto-Switch**: The new matter becomes active immediately

### Switching Between Matters

1. **Use Matter Dropdown**: In the left pane, select from existing matters
2. **Complete Isolation**: Each matter has its own:
   - Documents and processing
   - AI memory and context
   - Chat history
   - Configuration settings

### Matter Organization

Each matter is stored under `~/LettaClaims/Matter_<slug>/`:
```
Matter_smith-residence-water-damage/
├── config.json              # Matter-specific settings
├── docs/                     # Original PDF files
├── docs_ocr/                 # OCR-processed PDFs
├── parsed/                   # Extracted text and metadata
├── vectors/                  # Vector embeddings for search
├── knowledge/                # AI memory and agent state
├── chat/                     # Chat history
└── logs/                     # Processing logs
```

## Document Upload and Processing

### Supported Document Types

- **PDF Files**: Primary supported format
- **Born-digital PDFs**: Text-based documents (specifications, contracts)
- **Scanned PDFs**: Image-based documents (require OCR)
- **Mixed PDFs**: Combination of text and scanned pages

### Uploading Documents

1. **Drag and Drop**: Drag PDF files to the upload area
2. **Click Upload**: Use the "Upload PDFs" button to select files
3. **Multiple Files**: Upload multiple PDFs at once
4. **Background Processing**: Processing happens in the background

### Processing Pipeline

The application processes documents through several stages:

1. **OCR (if needed)**
   - Automatically detects image-only pages
   - Applies OCR only where needed (preserves original text)
   - Option to force OCR on all pages

2. **Text Extraction**
   - Extracts text with page-level precision
   - Maintains document structure and formatting
   - Preserves page boundaries for accurate citations

3. **Chunking**
   - Breaks text into ~1000 token chunks
   - Maintains 15% overlap between chunks
   - Preserves section boundaries and headings

4. **Embedding Generation**
   - Creates vector embeddings for semantic search
   - Uses local Ollama models (default: nomic-embed-text)
   - Enables fast, relevant content retrieval

### Processing Status

Monitor processing progress in the document list:
- **⏳ Processing**: Document is being processed
- **✅ Complete**: Processing finished successfully
- **❌ Error**: Processing failed (click for details)
- **📄 Pages**: Number of pages processed
- **🧩 Chunks**: Number of text chunks created
- **👁️ OCR**: OCR status (full/partial/none)

### Document Management

- **View Status**: See processing progress and statistics
- **Retry Processing**: Reprocess failed documents
- **Force OCR**: Re-process with OCR for better text extraction
- **Remove Documents**: Delete documents you no longer need

## Asking Questions and Chat Interface

### Chat Interface Basics

1. **Ask Questions**: Type questions in the chat input
2. **Send Message**: Press Enter or click Send
3. **View Responses**: Answers appear with structured formatting
4. **Follow-up Questions**: Get AI-suggested follow-ups after each answer

### Question Types

**Factual Questions**
- "When did the dry well failure occur?"
- "Who was the general contractor?"
- "What was the original completion date?"

**Analysis Questions**
- "What caused the delay in concrete delivery?"
- "How did weather impact the schedule?"
- "What evidence supports the extra work claim?"

**Comparative Questions**
- "How does the actual timeline compare to the planned schedule?"
- "What differences exist between the spec and actual work?"

**Legal/Claim Questions**
- "What evidence supports differing site conditions?"
- "Are there any liquidated damages clauses?"
- "What notice requirements were not met?"

### Response Format

Answers follow a consistent structure:

```
1) Key Points
   - Primary findings and facts
   - Direct answers to your question

2) Analysis
   - Deeper examination of the evidence
   - Connections between different documents
   - Implications for the claim

3) Citations
   - [DocumentName.pdf p.5-7]
   - [DailyLog_2023-02-15.pdf p.2]
   - Precise page references to source material

4) Suggested Follow-ups
   - Related questions to explore further
   - Next steps in your analysis
```

### Follow-up Suggestions

After each answer, the AI suggests relevant follow-up questions:
- **Evidence-based**: "What documentation supports this timeline?"
- **Legal**: "Are there any contract clauses that apply?"
- **Technical**: "What was the specified vs. actual material?"
- **Financial**: "What costs are associated with this issue?"

### Memory and Context

The AI's Letta-powered memory system provides sophisticated context management:

#### How Memory Enhances Responses
- **Facts accumulate**: Established facts from documents and conversations inform future answers
- **Context builds**: Relationships between events, parties, and issues become clearer over time
- **Memory persists**: Information remains between application sessions and survives restarts
- **Pattern Recognition**: Agent identifies contradictions, gaps, and patterns across documents
- **Proactive Suggestions**: Follow-up questions generated based on stored knowledge

#### Memory-Enhanced Chat Experience
- **Purple Glow**: Messages with purple outline used agent memory for enhanced context
- **Memory Badge**: Green "Memory Enhanced" badge appears on contextually-aware responses
- **Contextual Follow-ups**: Suggested questions reference previous conversations and stored facts
- **Cross-Reference Answers**: Agent connects current questions to related previous discussions

#### Understanding Memory Operations
- **Storing Notifications**: "Storing to memory..." appears when saving new information
- **Recalling Notifications**: "Recalling from memory..." shows when retrieving context
- **Memory Dashboard**: Right panel displays memory statistics and connection status
- **Auto-Refresh**: Memory dashboard updates every 30 seconds to show current status

#### Leveraging Memory Effectively
- **Build on Previous Work**: Reference earlier conversations: "What did we determine about the delay claim?"
- **Ask for Patterns**: "What contradictions have you identified across documents?"
- **Request Comparisons**: "How does this payment dispute relate to the contract amendment?"
- **Seek Timeline Analysis**: "What events led up to the stop work order?"
- **Validate Understanding**: Agent will use stored knowledge to confirm or clarify facts

*For comprehensive memory usage guidance, see the [Letta User Guide](LETTA_USER_GUIDE.md).*

## Understanding Sources and Citations

### Citation Format

All answers include precise citations:
- **[DocName.pdf p.5]**: Single page reference
- **[DocName.pdf p.5-7]**: Multi-page reference
- **[DocName.pdf p.5, 12]**: Multiple specific pages

### Sources Pane

The right pane shows detailed source information:
- **Document name**: Source file
- **Page range**: Specific pages referenced
- **Relevance score**: How well the content matches your question
- **Text snippet**: Preview of the relevant content

### Source Actions

For each source, you can:
- **Open PDF**: Opens the document at the specific page
- **Copy Citation**: Copies the citation to your clipboard
- **View Snippet**: See the relevant text excerpt

### Citation Accuracy

The application ensures citation accuracy:
- **Page-level precision**: Citations map to exact page numbers
- **Verification**: AI validates that cited content supports claims
- **Traceability**: Every fact can be traced back to source documents

## Configuration and Settings

### Accessing Settings

Click the **⚙️ Settings** button in the center pane to open the settings drawer.

### LLM Provider Settings

**Local Provider (Ollama)**
- **Model**: Choose generation model (gpt-oss:20b, llama3.1, etc.)
- **Test Connection**: Verify Ollama is running and models are available
- **Pull Models**: Download additional models if needed

**External Provider (Gemini)**
- **API Key**: Enter your Google Gemini API key
- **Model**: Select Gemini model (gemini-2.5-flash recommended)
- **Consent**: One-time consent for external data processing

### Embedding Settings

- **Model**: Choose embedding model (nomic-embed-text recommended)
- **Quality**: Higher quality models may be slower but more accurate

### OCR Settings

- **Language**: Set OCR language (default: English)
- **Skip Text**: Only OCR image-only pages (recommended)
- **Force OCR**: Always apply OCR (useful for poor-quality text)

### Performance Settings

- **Temperature**: Control AI response creativity (0.1-1.0)
- **Max Tokens**: Limit response length (600-1200)
- **Context Size**: Number of document chunks to include

### Privacy Settings

- **Local Only**: Keep all processing local (default)
- **External Consent**: Manage consent for external AI services
- **Data Retention**: Configure log and cache retention

## Best Practices

### Document Organization

1. **Descriptive Matter Names**: Use clear, specific names for matters
2. **Document Naming**: Keep original file names descriptive
3. **Chronological Upload**: Upload documents in logical order
4. **Quality Check**: Ensure PDFs are readable before upload

### Effective Questioning

1. **Start Broad**: Begin with overview questions
2. **Get Specific**: Drill down into specific issues
3. **Follow Citations**: Check sources for additional context
4. **Use Follow-ups**: The AI's suggestions often reveal important angles

### Query Strategies

**Timeline Analysis**
1. "What is the overall project timeline?"
2. "When did delays first occur?"
3. "What caused each delay?"
4. "How do delays relate to damages?"

**Causation Analysis**
1. "What factors contributed to [issue]?"
2. "What was the root cause of [problem]?"
3. "How did [event] impact [outcome]?"

**Evidence Building**
1. "What evidence supports [claim]?"
2. "Are there any contradictory statements?"
3. "What witnesses observed [event]?"
4. "What documentation exists for [issue]?"

### Performance Optimization

1. **Model Selection**: Use lighter models for faster responses
2. **Batch Upload**: Upload related documents together
3. **Regular Cleanup**: Remove unnecessary documents periodically
4. **Monitor Resources**: Check memory and disk usage

### Data Management

1. **Regular Backups**: Backup `~/LettaClaims/` periodically
2. **Matter Isolation**: Keep different cases in separate matters
3. **Clean Organization**: Remove test or duplicate documents
4. **Version Control**: Keep track of document versions

## Advanced Features

### Quality Metrics

The application tracks response quality:
- **Citation Coverage**: Percentage of claims supported by citations
- **Source Diversity**: Number of different documents referenced
- **Answer Completeness**: How thoroughly questions are addressed

### Memory Insights

The Letta memory system provides comprehensive insights into your agent's accumulated knowledge:

#### Memory Statistics Dashboard
- **Memory Items Count**: Total facts, entities, events, and issues stored
- **Connection Status**: Real-time Letta server connectivity (green/yellow/red)
- **Last Sync Time**: When memory was last updated ("Just now", "5m ago", etc.)
- **Memory Usage Bar**: Visual indicator of memory capacity utilization
- **Auto-Refresh**: Dashboard updates automatically every 30 seconds

#### Agent Health Monitoring
- **Health Indicator**: Header shows agent status with color-coded icons
  - 🟢 **Green Check**: Agent healthy and fully operational
  - 🟡 **Yellow Warning**: Minor issues but functional
  - 🔴 **Red Error**: Significant problems, may be in fallback mode
- **Detailed Health Info**: Click info button for connection status, response times, provider info

#### Memory Summary and Analytics
- **View Summary Button**: Click in memory dashboard for detailed breakdown
- **Knowledge Categories**: Shows counts of entities, events, issues, facts
- **Pattern Analysis**: Key actors, temporal trends, document focus areas
- **Memory Quality Metrics**: Completeness, accuracy, and relevance scores

#### Memory Export and Management
- **Export Options**: JSON, CSV, or Markdown formats for analysis or backup
- **Import Capability**: Restore memory from backups or migrate between systems
- **Memory Pruning**: Automatic or manual cleanup of old/low-importance items
- **Backup Integration**: Memory included in automated system backups

*For detailed memory management instructions, see the [Letta Configuration Guide](LETTA_CONFIGURATION.md).*

### Performance Monitoring

Monitor application performance:
- **Processing Speed**: Document ingestion and query response times
- **Resource Usage**: Memory, disk, and CPU utilization
- **Error Rates**: Processing failures and their causes

## Troubleshooting

### Common Issues

**Slow Performance**
- Switch to lighter AI models (llama3.1 vs gpt-oss:20b)
- Reduce max_tokens setting
- Close other applications to free memory

**Poor OCR Results**
- Enable "Force OCR" for better text extraction
- Check document quality (resolution, contrast)
- Use appropriate OCR language setting

**Missing Citations**
- Verify documents processed successfully
- Check that questions relate to uploaded content
- Ensure AI model has access to processed documents

**Memory Issues**
- Monitor `/api/health` for resource status
- Process fewer documents at once

**Letta Memory Problems**
- Check connection status in memory dashboard (should be green)
- Restart application if server shows as disconnected
- Verify memory operations with purple badges and notifications
- Review memory summary for accurate information storage
- Clear browser cache if UI indicators not updating

**Agent Not Responding**
- Check agent health indicator in header (should be green)
- Look for "Memory Enhanced" badges on responses - if missing, memory may be offline
- Restart application to reinitialize Letta server connection
- Check logs at `~/.letta-claim/logs/` for specific error messages

*For comprehensive troubleshooting, see the [Letta Troubleshooting Guide](LETTA_TROUBLESHOOTING.md).*

### Getting Help

1. **Health Check**: Visit `/api/health/detailed` for system status
2. **Logs**: Check `~/.letta-claim/logs/app.log` for error details
3. **Validation**: Run system validation from settings
4. **Documentation**: Refer to TROUBLESHOOTING.md for detailed solutions

### Error Recovery

**Document Processing Errors**
1. Check document format and file integrity
2. Try "Force OCR" option
3. Remove and re-upload problematic documents

**AI Response Errors**
1. Verify model availability in settings
2. Test connection to LLM provider
3. Try rephrasing your question

**Application Startup Errors**
1. Run startup validation checks
2. Verify all dependencies are installed
3. Check system requirements

## Tips for Legal Professionals

### Claim Development

1. **Establish Timeline**: Start with chronological questions
2. **Identify Parties**: Map all involved parties and their roles
3. **Document Chain**: Trace communication and decision chains
4. **Evidence Gaps**: Identify missing or needed documentation

### Damage Analysis

1. **Causation**: Establish clear cause-and-effect relationships
2. **Quantification**: Look for cost and time impact data
3. **Mitigation**: Identify efforts to minimize damages
4. **Comparison**: Compare planned vs. actual performance

### Risk Assessment

1. **Contract Terms**: Identify relevant contract provisions
2. **Notice Requirements**: Check for proper notice compliance
3. **Statute Limitations**: Be aware of timing requirements
4. **Evidence Quality**: Assess strength of supporting documentation

---

**Need More Help?** 
- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [Configuration Reference](CONFIGURATION.md)
- Visit the [Installation Guide](INSTALLATION.md) for setup issues