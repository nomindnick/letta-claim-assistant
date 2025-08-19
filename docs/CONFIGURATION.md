# Letta Construction Claim Assistant - Configuration Reference

This guide provides comprehensive documentation for all configuration options in the Letta Construction Claim Assistant.

## Configuration Overview

The application uses a two-level configuration system:

1. **Global Configuration**: System-wide settings stored in `~/.letta-claim/config.toml`
2. **Matter Configuration**: Per-matter settings stored in each matter's `config.json`

## Global Configuration File

### Location and Format

**File:** `~/.letta-claim/config.toml`
**Format:** TOML (Tom's Obvious Minimal Language)

### Configuration Template

```toml
[ui]
framework = "nicegui"
native = true

[llm]
provider = "ollama"
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
skip_text = true

[paths]
root = "~/LettaClaims"

[gemini]
api_key = ""
model = "gemini-2.5-flash"

[letta]
enable_filesystem = true

[monitoring]
enabled = true
interval_seconds = 30

[logging]
level = "INFO"
enable_rotation = true
enable_masking = true
max_file_size_mb = 10
backup_count = 5

[performance]
enable_profiling = false
cache_size_mb = 256
max_concurrent_jobs = 3

[security]
enable_encryption = true
credential_timeout_hours = 24
```

## Configuration Sections

### [ui] - User Interface Settings

Controls the desktop application interface behavior.

#### framework
- **Type:** String
- **Default:** `"nicegui"`
- **Options:** `"nicegui"`
- **Description:** UI framework to use (currently only NiceGUI supported)

#### native
- **Type:** Boolean
- **Default:** `true`
- **Description:** Whether to run in native desktop mode
- **Environment Override:** `LETTA_UI_NATIVE=false`

**Example:**
```toml
[ui]
framework = "nicegui"
native = true  # Set to false for browser mode
```

### [llm] - Language Model Settings

Configures the primary AI model for text generation and analysis.

#### provider
- **Type:** String
- **Default:** `"ollama"`
- **Options:** `"ollama"`, `"gemini"`
- **Description:** LLM provider to use for generation

#### model
- **Type:** String
- **Default:** `"gpt-oss:20b"`
- **Ollama Options:** 
  - `"gpt-oss:20b"` - Large, high-quality model (~11GB)
  - `"llama3.1"` - Medium, faster model (~4.7GB)
  - `"mistral"` - Small, fastest model (~4.1GB)
- **Gemini Options:**
  - `"gemini-2.5-flash"` - Fast, efficient model
  - `"gemini-pro"` - Higher quality model
- **Description:** Specific model to use for generation

#### temperature
- **Type:** Float
- **Default:** `0.2`
- **Range:** `0.0` to `1.0`
- **Description:** Controls randomness in responses (lower = more focused)

#### max_tokens
- **Type:** Integer
- **Default:** `900`
- **Range:** `100` to `2000`
- **Description:** Maximum length of generated responses

**Example:**
```toml
[llm]
provider = "ollama"
model = "llama3.1"        # Faster than gpt-oss:20b
temperature = 0.1         # Very focused responses
max_tokens = 1200         # Longer responses allowed
```

### [embeddings] - Vector Embedding Settings

Configures models used for document search and retrieval.

#### provider
- **Type:** String
- **Default:** `"ollama"`
- **Options:** `"ollama"`
- **Description:** Embedding provider (currently only Ollama supported)

#### model
- **Type:** String
- **Default:** `"nomic-embed-text"`
- **Options:**
  - `"nomic-embed-text"` - Balanced performance (768 dimensions)
  - `"mxbai-embed-large"` - Higher quality (1024 dimensions)
  - `"bge-m3"` - Multilingual support (1024 dimensions)
- **Description:** Embedding model for document vectorization

**Example:**
```toml
[embeddings]
provider = "ollama"
model = "mxbai-embed-large"  # Higher quality embeddings
```

### [ocr] - Optical Character Recognition Settings

Controls how scanned documents are processed.

#### enabled
- **Type:** Boolean
- **Default:** `true`
- **Description:** Whether to enable OCR processing

#### force_ocr
- **Type:** Boolean
- **Default:** `false`
- **Description:** Whether to force OCR on all pages (even those with text)

#### language
- **Type:** String
- **Default:** `"eng"`
- **Options:** `"eng"`, `"spa"`, `"fra"`, `"deu"`, etc.
- **Description:** Primary OCR language (requires tesseract-ocr-[lang] package)

#### skip_text
- **Type:** Boolean
- **Default:** `true`
- **Description:** Only OCR image-only pages (preserves original text quality)

**Example:**
```toml
[ocr]
enabled = true
force_ocr = false
language = "eng"
skip_text = true

# For multilingual documents
# language = "eng+spa"  # English + Spanish
```

### [paths] - File System Paths

Defines where application data is stored.

#### root
- **Type:** String (Path)
- **Default:** `"~/LettaClaims"`
- **Description:** Root directory for all matter data
- **Environment Override:** `LETTA_DATA_ROOT`

**Directory Structure:**
```
~/LettaClaims/
├── Matter_project-abc/
│   ├── config.json
│   ├── docs/           # Original PDFs
│   ├── docs_ocr/       # OCR-processed PDFs
│   ├── parsed/         # Extracted text
│   ├── vectors/        # ChromaDB collections
│   ├── knowledge/      # Letta agent state
│   ├── chat/          # Chat history
│   └── logs/          # Matter-specific logs
└── Matter_project-xyz/
    └── ...
```

**Example:**
```toml
[paths]
root = "/data/LettaClaims"    # Custom data location
# root = "~/Documents/Claims"  # Alternative location
```

### [gemini] - Google Gemini Settings

Configuration for external Gemini API (optional).

#### api_key
- **Type:** String
- **Default:** `""`
- **Description:** Google AI API key (leave empty, set via UI for security)
- **Security Note:** API keys are stored encrypted, not in this config file

#### model
- **Type:** String
- **Default:** `"gemini-2.5-flash"`
- **Options:**
  - `"gemini-2.5-flash"` - Fast, cost-effective
  - `"gemini-pro"` - Higher quality, slower
- **Description:** Gemini model to use when provider is "gemini"

**Example:**
```toml
[gemini]
api_key = ""                    # Set via application UI
model = "gemini-2.5-flash"
```

### [letta] - Agent Memory Settings

Controls the AI agent's persistent memory system. For comprehensive Letta configuration, see the [Letta Configuration Guide](LETTA_CONFIGURATION.md).

#### Basic Settings
```toml
[letta]
enabled = true                    # Enable Letta memory features
server_mode = "local"            # local, docker, external
server_host = "localhost"
server_port = 8283
auto_start = true               # Start Letta server automatically
fallback_mode = true           # Graceful degradation when unavailable
```

#### Memory Configuration
```toml
[letta.memory]
max_items_per_agent = 10000     # Maximum memory items per matter
prune_old_memories = true       # Automatically remove old memories
prune_after_days = 90          # Remove memories older than N days
importance_threshold = 0.5      # Minimum importance score to keep
enable_deduplication = true     # Remove duplicate memories
```

#### Performance Settings
```toml
[letta.performance]
connection_timeout = 30         # Connection timeout in seconds
request_timeout = 60           # Request timeout in seconds
max_retries = 3                # Maximum retry attempts
batch_size = 100               # Batch size for memory operations
health_check_interval = 30      # Health check frequency
```

#### California Domain Specialization
```toml
[letta.domain]
enabled = true                  # Enable California construction specialization
entity_extraction = true       # Extract California entities (Caltrans, etc.)
compliance_validation = true   # Validate California statutory compliance
followup_generation = true     # Generate California-specific follow-ups
expert_triggers = true         # Suggest expert consultation when needed
```

#### Advanced Settings
```toml
[letta.advanced]
debug_logging = false          # Enable debug logging
log_requests = false          # Log requests (don't log responses - sensitive data)
circuit_breaker = true        # Enable circuit breaker for fault tolerance
request_queue = true          # Enable request queuing and batching
memory_export_encryption = true  # Encrypt memory exports
```

*See [Letta Configuration Guide](LETTA_CONFIGURATION.md) for detailed configuration instructions.*

### [monitoring] - Performance Monitoring

Controls system monitoring and metrics collection.

#### enabled
- **Type:** Boolean
- **Default:** `true`
- **Description:** Whether to enable performance monitoring

#### interval_seconds
- **Type:** Integer
- **Default:** `30`
- **Range:** `10` to `300`
- **Description:** How often to collect system metrics (seconds)

**Example:**
```toml
[monitoring]
enabled = true
interval_seconds = 60  # Collect metrics every minute
```

### [logging] - Logging Configuration

Controls application logging behavior.

#### level
- **Type:** String
- **Default:** `"INFO"`
- **Options:** `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`
- **Description:** Minimum log level to record
- **Environment Override:** `LETTA_LOG_LEVEL=DEBUG`

#### enable_rotation
- **Type:** Boolean
- **Default:** `true`
- **Description:** Whether to rotate log files when they get large

#### enable_masking
- **Type:** Boolean
- **Default:** `true`
- **Description:** Whether to mask sensitive data in logs

#### max_file_size_mb
- **Type:** Integer
- **Default:** `10`
- **Description:** Maximum log file size before rotation (MB)

#### backup_count
- **Type:** Integer
- **Default:** `5`
- **Description:** Number of backup log files to keep

**Example:**
```toml
[logging]
level = "DEBUG"           # Verbose logging for troubleshooting
enable_rotation = true
enable_masking = true
max_file_size_mb = 20    # Larger log files
backup_count = 10        # More backup files
```

### [performance] - Performance Tuning

Advanced performance and resource management settings.

#### enable_profiling
- **Type:** Boolean
- **Default:** `false`
- **Description:** Whether to enable detailed performance profiling

#### cache_size_mb
- **Type:** Integer
- **Default:** `256`
- **Description:** Memory cache size for embeddings and chunks (MB)

#### max_concurrent_jobs
- **Type:** Integer
- **Default:** `3`
- **Description:** Maximum number of concurrent background jobs

**Example:**
```toml
[performance]
enable_profiling = true   # Enable for performance analysis
cache_size_mb = 512      # Larger cache for better performance
max_concurrent_jobs = 2   # Fewer concurrent jobs on modest hardware
```

### [security] - Security Settings

Controls security and encryption features.

#### enable_encryption
- **Type:** Boolean
- **Default:** `true`
- **Description:** Whether to encrypt stored credentials

#### credential_timeout_hours
- **Type:** Integer
- **Default:** `24`
- **Description:** How long to cache decrypted credentials (hours)

**Example:**
```toml
[security]
enable_encryption = true
credential_timeout_hours = 8  # Re-prompt for credentials more frequently
```

## Matter-Specific Configuration

### Location and Format

**File:** `~/LettaClaims/Matter_<slug>/config.json`
**Format:** JSON

### Configuration Structure

```json
{
  "id": "abc123-def456-ghi789",
  "name": "Project ABC - Water Damage Claim",
  "slug": "project-abc-water-damage-claim",
  "created_at": "2025-08-15T10:30:00Z",
  "embedding_model": "nomic-embed-text",
  "generation_model": "gpt-oss:20b",
  "vector_path": "vectors/chroma",
  "letta_path": "knowledge/letta_state",
  "custom_settings": {
    "ocr_language": "eng+spa",
    "force_ocr": false,
    "max_tokens": 1200
  }
}
```

### Configuration Fields

#### Core Fields

- **id**: Unique identifier for the matter
- **name**: Human-readable matter name
- **slug**: URL-safe version of the name
- **created_at**: ISO 8601 timestamp of creation

#### Model Configuration

- **embedding_model**: Embedding model used for this matter
- **generation_model**: Generation model used for this matter
- **vector_path**: Relative path to vector database
- **letta_path**: Relative path to Letta agent state

#### Custom Settings

Matter-specific overrides for global settings:

```json
{
  "custom_settings": {
    "ocr_language": "spa",      # Spanish OCR for this matter
    "force_ocr": true,          # Force OCR for poor-quality scans
    "temperature": 0.1,         # More focused responses
    "max_tokens": 800,          # Shorter responses
    "enable_profiling": true    # Performance monitoring
  }
}
```

## Environment Variables

Environment variables override configuration file settings:

### Core Settings
```bash
export LETTA_ENV=production          # Environment mode
export LETTA_DATA_ROOT=/custom/path  # Override data root
export LETTA_UI_NATIVE=false        # Force browser mode
export LETTA_PORT=8080              # Custom port
```

### Logging
```bash
export LETTA_LOG_LEVEL=DEBUG        # Override log level
export DEBUG=1                      # Enable debug mode
```

### Development
```bash
export LETTA_ENV=development        # Development mode
export LETTA_DISABLE_MONITORING=1   # Disable monitoring
export LETTA_MOCK_OLLAMA=1          # Use mock Ollama for testing
```

## Configuration Validation

### Automatic Validation

The application validates configuration on startup:

```bash
# Check configuration validity
curl http://localhost:8000/api/system/validation
```

### Manual Validation

```python
from app.production_config import validate_production_config

success, results = validate_production_config()
for result in results:
    print(f"{result.status.value}: {result.name} - {result.message}")
```

### Common Validation Errors

1. **Invalid Paths**
   - Non-existent directories
   - Permission issues
   - Relative paths in production

2. **Model Availability**
   - Ollama models not pulled
   - Invalid model names
   - Network connectivity issues

3. **Resource Constraints**
   - Insufficient disk space
   - Low memory
   - Missing system dependencies

## Performance Tuning

### Hardware-Specific Settings

**High-End Workstation (16GB+ RAM, SSD)**
```toml
[llm]
model = "gpt-oss:20b"
max_tokens = 1200

[embeddings]
model = "mxbai-embed-large"

[performance]
cache_size_mb = 1024
max_concurrent_jobs = 4
```

**Standard Desktop (8GB RAM)**
```toml
[llm]
model = "llama3.1"
max_tokens = 900

[embeddings]
model = "nomic-embed-text"

[performance]
cache_size_mb = 256
max_concurrent_jobs = 2
```

**Laptop/Low-Resource (4GB RAM)**
```toml
[llm]
model = "mistral"
max_tokens = 600

[embeddings]
model = "nomic-embed-text"

[performance]
cache_size_mb = 128
max_concurrent_jobs = 1
```

### Network-Specific Settings

**Slow Internet Connection**
```toml
[llm]
provider = "ollama"  # Prefer local models

[monitoring]
interval_seconds = 120  # Less frequent monitoring
```

**Fast Internet, External API Preferred**
```toml
[llm]
provider = "gemini"
model = "gemini-2.5-flash"
```

## Security Considerations

### Sensitive Data Handling

1. **API Keys**: Stored encrypted, never in config files
2. **Logs**: Sensitive data masked automatically
3. **File Permissions**: Restricted to user access only
4. **Network**: Local-first, external APIs require consent

### Production Security

```toml
[security]
enable_encryption = true
credential_timeout_hours = 8

[logging]
enable_masking = true
level = "WARNING"  # Less verbose logging

[monitoring]
enabled = true     # Monitor for security events
```

### Development Security

```toml
[security]
enable_encryption = true
credential_timeout_hours = 24

[logging]
enable_masking = false  # Easier debugging
level = "DEBUG"         # Verbose logging
```

## Migration and Upgrades

### Configuration Migration

When upgrading, the application will:

1. Backup existing configuration
2. Merge new default settings
3. Preserve user customizations
4. Validate merged configuration

### Manual Migration

```bash
# Backup current config
cp ~/.letta-claim/config.toml ~/.letta-claim/config.toml.backup

# Merge with new template
# (Application handles this automatically)
```

### Version Compatibility

- **v1.0.x**: Initial configuration format
- **v1.1.x**: Added monitoring and security sections
- **v1.2.x**: Added performance tuning options

## Troubleshooting Configuration

### Common Issues

1. **Configuration Not Loading**
   - Check file syntax with TOML validator
   - Verify file permissions
   - Check for special characters

2. **Settings Not Applied**
   - Restart application after changes
   - Check environment variable overrides
   - Verify section and key names

3. **Performance Issues**
   - Review hardware-specific settings
   - Check resource constraints
   - Monitor system metrics

### Diagnostic Commands

```bash
# Validate TOML syntax
python3 -c "import tomllib; print(tomllib.load(open('~/.letta-claim/config.toml', 'rb')))"

# Check effective configuration
curl http://localhost:8000/api/system/validation

# Monitor resource usage
curl http://localhost:8000/api/metrics
```

---

**Need Help?**
- Review the [Troubleshooting Guide](TROUBLESHOOTING.md) for specific issues
- Check the [Installation Guide](INSTALLATION.md) for setup problems
- Consult the [User Guide](USER_GUIDE.md) for usage questions