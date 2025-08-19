# Letta Configuration Guide

**Version:** 1.0  
**Last Updated:** 2025-08-19

This guide provides comprehensive configuration instructions for the Letta agent memory system in the Construction Claim Assistant.

---

## Overview

The Letta integration consists of three main components:
1. **Letta Server**: Local server process that manages agent memory
2. **Letta Client**: Connection interface between application and server
3. **Agent Configuration**: Matter-specific agent settings and preferences

---

## Server Configuration

### Configuration File Location
Global server configuration is stored in:
```
~/.letta-claim/letta-server.yaml
```

### Default Server Configuration
```yaml
# Letta Server Configuration
server:
  mode: local              # local, docker, external
  host: localhost
  port: 8283
  auto_start: true
  startup_timeout: 30
  health_check_interval: 30
  max_retries: 3
  retry_delay: 5

# Storage Configuration
storage:
  type: sqlite
  path: ~/.letta-claim/letta.db
  backup_interval: 3600    # seconds
  max_backups: 5
  
# Memory Configuration
memory:
  max_memory_items: 10000
  archival_memory_limit: 50000
  prune_old_memories: true
  prune_after_days: 90
  
# Logging Configuration
logging:
  level: info              # debug, info, warning, error
  file: ~/.letta-claim/logs/letta-server.log
  max_size: 10mb
  backup_count: 5
  
# Performance Configuration
performance:
  workers: 3
  batch_size: 100
  request_timeout: 30
  connection_pool_size: 5
```

### Server Deployment Modes

#### 1. Local Mode (Default)
Runs Letta server as a subprocess managed by the application.

```yaml
server:
  mode: local
  auto_start: true
  startup_timeout: 30
```

**Advantages:**
- Automatic lifecycle management
- No external dependencies
- Easy setup and maintenance

**Requirements:**
- Letta installed in Python environment
- Sufficient system resources

#### 2. Docker Mode
Runs Letta server in a Docker container.

```yaml
server:
  mode: docker
  docker_image: letta/server:latest
  docker_port_mapping: "8283:8283"
  docker_volumes:
    - "~/.letta-claim/data:/app/data"
```

**Advantages:**
- Isolated execution environment
- Consistent deployment across systems
- Resource limits and monitoring

**Requirements:**
- Docker installed and running
- Docker image available locally or via registry

#### 3. External Mode
Connects to an externally managed Letta server.

```yaml
server:
  mode: external
  host: your-letta-server.com
  port: 8283
  auth_token: "your-auth-token"  # if required
```

**Advantages:**
- Shared server across multiple applications
- Centralized management and monitoring
- Scalability for team environments

**Requirements:**
- External Letta server accessible
- Network connectivity and credentials

### Port Configuration

#### Default Ports
- **Letta Server**: 8283
- **Application API**: 8000

#### Port Conflict Resolution
The system automatically handles port conflicts:
1. Checks if default port (8283) is available
2. If occupied, tries ports 8284, 8285, 8286...
3. Updates configuration with selected port
4. Logs port selection for troubleshooting

#### Custom Port Configuration
```yaml
server:
  port: 9283              # Custom port
  port_range_start: 9283  # Start of search range
  port_range_end: 9293    # End of search range
```

### Storage Configuration

#### SQLite Database Settings
```yaml
storage:
  type: sqlite
  path: ~/.letta-claim/letta.db
  pragma_settings:
    journal_mode: WAL
    synchronous: NORMAL
    cache_size: -64000    # 64MB cache
    temp_store: MEMORY
```

#### Backup Configuration
```yaml
storage:
  backup_interval: 3600   # Backup every hour
  max_backups: 24        # Keep 24 hourly backups
  backup_path: ~/.letta-claim/backups/
  compress_backups: true
```

### Memory Management Settings

#### Memory Limits
```yaml
memory:
  max_memory_items: 10000           # Max items per agent
  archival_memory_limit: 50000      # Max archival items
  core_memory_limit: 8192          # Max core memory chars
  conversation_buffer: 32          # Recent messages kept
```

#### Memory Pruning
```yaml
memory:
  prune_old_memories: true
  prune_after_days: 90            # Delete memories older than 90 days
  prune_low_importance: true      # Remove low-importance items first
  importance_threshold: 0.3       # Minimum importance score
```

#### Memory Quality Settings
```yaml
memory:
  deduplication: true             # Remove duplicate memories
  similarity_threshold: 0.85      # Similarity for deduplication
  quality_scoring: true           # Enable quality scoring
  min_quality_score: 0.5         # Minimum quality to store
```

---

## Client Configuration

### Connection Settings
Client configuration is stored per-matter in:
```
~/LettaClaims/Matter_<slug>/config.json
```

### Default Client Configuration
```json
{
  "letta": {
    "enabled": true,
    "server_host": "localhost",
    "server_port": 8283,
    "connection_timeout": 10,
    "request_timeout": 30,
    "max_retries": 3,
    "retry_backoff": 2.0,
    "health_check_interval": 30,
    "agent_id": null,
    "fallback_mode": true
  }
}
```

### Connection Management
```json
{
  "letta": {
    "connection": {
      "pool_size": 5,
      "pool_timeout": 30,
      "keep_alive": true,
      "tcp_nodelay": true,
      "connection_retry_interval": 5,
      "max_connection_attempts": 10
    }
  }
}
```

### Circuit Breaker Settings
```json
{
  "letta": {
    "circuit_breaker": {
      "enabled": true,
      "failure_threshold": 5,
      "recovery_timeout": 30,
      "half_open_max_calls": 3,
      "expected_exception_types": [
        "ConnectionError",
        "TimeoutError"
      ]
    }
  }
}
```

---

## Agent Configuration

### Agent Creation Settings
```json
{
  "letta": {
    "agent": {
      "name": "Construction Claims Agent",
      "persona": "california_construction_expert",
      "human_description": "Legal professional analyzing construction claims",
      "system_prompt_template": "california_public_works",
      "memory_capacity": 10000,
      "context_window": 8192
    }
  }
}
```

### LLM Provider Configuration

#### Ollama (Local)
```json
{
  "letta": {
    "llm_provider": {
      "type": "ollama",
      "host": "localhost",
      "port": 11434,
      "model": "gpt-oss:20b",
      "temperature": 0.2,
      "max_tokens": 900,
      "context_length": 8192
    }
  }
}
```

#### Gemini (External)
```json
{
  "letta": {
    "llm_provider": {
      "type": "gemini",
      "api_key": "your-gemini-api-key",
      "model": "gemini-2.5-flash",
      "temperature": 0.3,
      "max_tokens": 1000,
      "safety_settings": {
        "harassment": "BLOCK_NONE",
        "hate_speech": "BLOCK_NONE",
        "sexually_explicit": "BLOCK_NONE",
        "dangerous_content": "BLOCK_NONE"
      }
    }
  }
}
```

#### OpenAI (External)
```json
{
  "letta": {
    "llm_provider": {
      "type": "openai",
      "api_key": "your-openai-api-key",
      "model": "gpt-4",
      "temperature": 0.2,
      "max_tokens": 900,
      "organization": "your-org-id"
    }
  }
}
```

### Embedding Configuration
```json
{
  "letta": {
    "embedding_provider": {
      "type": "ollama",
      "model": "nomic-embed-text",
      "host": "localhost",
      "port": 11434,
      "batch_size": 100
    }
  }
}
```

### Domain Specialization
```json
{
  "letta": {
    "domain": {
      "enabled": true,
      "type": "california_construction",
      "entity_extraction": true,
      "compliance_validation": true,
      "followup_generation": true,
      "expert_triggers": true
    }
  }
}
```

---

## Performance Configuration

### Request Queue Settings
```json
{
  "letta": {
    "request_queue": {
      "enabled": true,
      "max_queue_size": 1000,
      "batch_processing": true,
      "batch_size": 10,
      "batch_timeout": 5,
      "priority_levels": ["CRITICAL", "HIGH", "NORMAL", "LOW"]
    }
  }
}
```

### Caching Configuration
```json
{
  "letta": {
    "caching": {
      "enabled": true,
      "memory_cache_size": 1000,
      "memory_cache_ttl": 3600,
      "response_cache_size": 100,
      "response_cache_ttl": 300
    }
  }
}
```

### Logging Configuration
```json
{
  "letta": {
    "logging": {
      "level": "info",
      "file": "~/.letta-claim/logs/letta-client.log",
      "max_size": "10MB",
      "backup_count": 5,
      "log_requests": true,
      "log_responses": false,
      "sensitive_data_masking": true
    }
  }
}
```

---

## Security Configuration

### Authentication Settings
```yaml
server:
  auth:
    enabled: false           # Enable for production
    jwt_secret: "your-secret-key"
    token_expiration: 3600   # 1 hour
    refresh_token_expiration: 86400  # 24 hours
```

### SSL/TLS Configuration
```yaml
server:
  ssl:
    enabled: false
    cert_file: ~/.letta-claim/ssl/cert.pem
    key_file: ~/.letta-claim/ssl/key.pem
    ca_file: ~/.letta-claim/ssl/ca.pem
```

### Data Privacy Settings
```json
{
  "letta": {
    "privacy": {
      "local_only": true,
      "external_api_consent": false,
      "data_encryption": true,
      "log_sanitization": true,
      "memory_export_encryption": true
    }
  }
}
```

---

## Environment Variables

### Server Environment Variables
```bash
# Server Configuration
LETTA_SERVER_HOST=localhost
LETTA_SERVER_PORT=8283
LETTA_SERVER_MODE=local

# Database Configuration
LETTA_DB_PATH=~/.letta-claim/letta.db
LETTA_DB_BACKUP_INTERVAL=3600

# Logging Configuration
LETTA_LOG_LEVEL=info
LETTA_LOG_FILE=~/.letta-claim/logs/letta-server.log

# Performance Configuration
LETTA_WORKERS=3
LETTA_BATCH_SIZE=100
```

### Client Environment Variables
```bash
# Client Configuration
LETTA_CLIENT_TIMEOUT=30
LETTA_CLIENT_RETRIES=3
LETTA_CLIENT_FALLBACK=true

# Provider Configuration
LETTA_OLLAMA_HOST=localhost
LETTA_OLLAMA_PORT=11434
GEMINI_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
```

---

## Configuration Validation

### Server Configuration Validation
```bash
# Validate server configuration
letta-claim-assistant --validate-server-config

# Check server connectivity
letta-claim-assistant --check-server-health

# Test server performance
letta-claim-assistant --benchmark-server
```

### Client Configuration Validation
```bash
# Validate client configuration
letta-claim-assistant --validate-client-config

# Test agent creation
letta-claim-assistant --test-agent-creation

# Validate memory operations
letta-claim-assistant --test-memory-ops
```

---

## Configuration Best Practices

### 1. Development Environment
```yaml
# Use local mode with debug logging
server:
  mode: local
  auto_start: true
  
logging:
  level: debug
  
memory:
  prune_old_memories: false  # Keep all memories for debugging
```

### 2. Production Environment
```yaml
# Use Docker mode with proper logging
server:
  mode: docker
  health_check_interval: 60
  
logging:
  level: info
  
memory:
  prune_old_memories: true
  backup_interval: 3600
  
security:
  auth_enabled: true
  ssl_enabled: true
```

### 3. Team Environment
```yaml
# Use external server with shared configuration
server:
  mode: external
  host: team-letta-server.internal
  auth_token: ${LETTA_TEAM_TOKEN}
  
memory:
  max_memory_items: 50000    # Higher limits for team use
```

### 4. High-Performance Setup
```yaml
# Optimize for performance
server:
  workers: 6
  
performance:
  batch_size: 250
  connection_pool_size: 10
  
memory:
  prune_old_memories: true
  deduplication: true
```

---

## Configuration Troubleshooting

### Common Issues

#### Server Won't Start
1. Check port availability: `netstat -an | grep 8283`
2. Verify Letta installation: `letta --version`
3. Check configuration syntax: `letta config validate`
4. Review logs: `~/.letta-claim/logs/letta-server.log`

#### Connection Timeouts
1. Increase timeout values in client configuration
2. Check network connectivity
3. Verify server is responsive: `curl http://localhost:8283/v1/health/`
4. Enable connection pooling

#### Memory Issues
1. Check memory limits in configuration
2. Enable memory pruning
3. Monitor memory usage in dashboard
4. Consider increasing system resources

#### Performance Problems
1. Increase worker count
2. Enable request batching
3. Tune batch sizes and timeouts
4. Monitor system resource usage

---

## Migration and Upgrades

### Configuration Migration
When upgrading, configuration files are automatically migrated:
1. Backup existing configuration
2. Apply new default settings
3. Merge user customizations
4. Validate final configuration

### Version Compatibility
Configuration version tracking ensures compatibility:
```json
{
  "config_version": "1.0",
  "letta_version": "0.10.0",
  "last_updated": "2025-08-19T10:30:00Z"
}
```

---

## Advanced Configuration Topics

### Custom Personas
Create domain-specific agent personas:
```json
{
  "personas": {
    "california_construction_expert": {
      "description": "Expert in California public works construction law",
      "traits": ["analytical", "detail-oriented", "legally-focused"],
      "knowledge_areas": ["PCC", "Government Code", "Civil Code"]
    }
  }
}
```

### Memory Optimization
Fine-tune memory performance:
```json
{
  "memory_optimization": {
    "semantic_search_threshold": 0.7,
    "importance_scoring_weights": {
      "recency": 0.3,
      "relevance": 0.4,
      "completeness": 0.3
    },
    "context_window_optimization": true
  }
}
```

### Provider Fallback Chains
Configure automatic provider fallback:
```json
{
  "provider_fallback": {
    "primary": "ollama",
    "secondary": "gemini",
    "tertiary": "openai",
    "fallback_conditions": ["timeout", "rate_limit", "error"]
  }
}
```

---

## Support and Resources

### Configuration Help
- **Configuration reference**: Complete parameter documentation
- **Best practices guide**: Recommended settings for common scenarios
- **Troubleshooting guide**: Solutions for configuration issues
- **Performance tuning**: Optimization recommendations

### Community Resources
- **Configuration examples**: Sample configurations for various setups
- **Migration guides**: Step-by-step upgrade instructions
- **Performance benchmarks**: Configuration impact analysis
- **Security guidelines**: Best practices for production deployment

---

*For additional configuration assistance, see the Letta Technical Reference and consult the troubleshooting documentation.*