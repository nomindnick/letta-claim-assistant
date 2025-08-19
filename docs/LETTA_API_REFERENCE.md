# Letta API Reference

**Version:** 1.0  
**Last Updated:** 2025-08-19

This document provides complete API reference for the Letta agent memory integration in the Construction Claim Assistant.

---

## Overview

The Letta API provides endpoints for:
- **Memory Operations**: Store, retrieve, and manage agent memory
- **Agent Management**: Create, configure, and control agents
- **Health Monitoring**: Server status and connection health
- **Statistics**: Memory usage and performance metrics

### Base URLs
- **Application API**: `http://localhost:8000/api`
- **Letta Server**: `http://localhost:8283/v1`

---

## Memory Operations API

### Get Memory Statistics

**Endpoint:** `GET /api/matters/{matter_id}/memory/stats`

**Description:** Retrieve memory usage statistics for a specific matter's agent.

**Parameters:**
- `matter_id` (path, required): Matter UUID

**Response:**
```json
{
  "memory_items": 245,
  "connection_state": "connected",
  "last_sync": "2025-08-19T10:30:00Z",
  "memory_usage": 0.35,
  "agent_id": "agent_12345",
  "server_status": "healthy",
  "stats": {
    "entities": 45,
    "events": 32,
    "issues": 18,
    "facts": 150
  }
}
```

**Error Responses:**
- `404`: Matter not found
- `503`: Letta server unavailable

---

### Get Memory Summary

**Endpoint:** `POST /api/matters/{matter_id}/memory/summary`

**Description:** Generate a comprehensive summary of stored memory.

**Parameters:**
- `matter_id` (path, required): Matter UUID

**Request Body:**
```json
{
  "max_items": 50,
  "category_filter": ["entities", "events"],
  "date_range": {
    "start": "2025-08-01",
    "end": "2025-08-19"
  },
  "importance_threshold": 0.5
}
```

**Response:**
```json
{
  "summary": {
    "total_items": 245,
    "categories": {
      "entities": 45,
      "events": 32,
      "issues": 18,
      "facts": 150
    },
    "date_range": {
      "earliest": "2025-08-01T00:00:00Z",
      "latest": "2025-08-19T10:30:00Z"
    }
  },
  "items": [
    {
      "id": "mem_001",
      "type": "Entity",
      "label": "Caltrans District 4",
      "date": "2025-08-15",
      "actors": ["Caltrans"],
      "doc_refs": [
        {
          "doc": "Contract_Amendment.pdf",
          "page": 3
        }
      ],
      "support_snippet": "Caltrans District 4 approved the change order",
      "importance": 0.85,
      "last_updated": "2025-08-19T09:15:00Z"
    }
  ],
  "patterns": {
    "key_actors": ["Caltrans", "ABC Construction", "XYZ Engineering"],
    "temporal_trends": "Increased activity in Q3 2025",
    "document_focus": "Contract amendments and change orders"
  }
}
```

---

### Store Knowledge Items

**Endpoint:** `POST /api/matters/{matter_id}/memory/store`

**Description:** Store new knowledge items in agent memory.

**Parameters:**
- `matter_id` (path, required): Matter UUID

**Request Body:**
```json
{
  "items": [
    {
      "type": "Entity",
      "label": "ABC Construction Company",
      "date": "2025-08-15",
      "actors": ["ABC Construction"],
      "doc_refs": [
        {
          "doc": "Contract.pdf",
          "page": 1
        }
      ],
      "support_snippet": "Prime contractor for the highway project"
    },
    {
      "type": "Event", 
      "label": "Change Order #3 Approved",
      "date": "2025-08-10",
      "actors": ["Caltrans", "ABC Construction"],
      "doc_refs": [
        {
          "doc": "Change_Order_3.pdf",
          "page": 2
        }
      ],
      "support_snippet": "Additional work approved for $125,000"
    }
  ],
  "batch_options": {
    "deduplication": true,
    "importance_scoring": true,
    "parallel_processing": true
  }
}
```

**Response:**
```json
{
  "stored_items": 2,
  "duplicates_found": 0,
  "batch_id": "batch_789",
  "processing_time": 1.23,
  "results": [
    {
      "item_index": 0,
      "status": "stored",
      "memory_id": "mem_456",
      "importance_score": 0.75
    },
    {
      "item_index": 1,
      "status": "stored", 
      "memory_id": "mem_457",
      "importance_score": 0.82
    }
  ]
}
```

---

### Recall Memory Context

**Endpoint:** `POST /api/matters/{matter_id}/memory/recall`

**Description:** Retrieve relevant memory items for a given query context.

**Parameters:**
- `matter_id` (path, required): Matter UUID

**Request Body:**
```json
{
  "query": "payment disputes with Caltrans",
  "max_results": 10,
  "context_window": 7,
  "filters": {
    "types": ["Event", "Issue"],
    "actors": ["Caltrans"],
    "date_range": {
      "start": "2025-07-01",
      "end": "2025-08-19"
    }
  },
  "options": {
    "include_snippets": true,
    "recency_weight": 0.3,
    "relevance_threshold": 0.6
  }
}
```

**Response:**
```json
{
  "total_matches": 8,
  "context_items": [
    {
      "id": "mem_123",
      "type": "Issue",
      "label": "Payment Delay Claim",
      "date": "2025-08-05",
      "actors": ["Caltrans", "ABC Construction"],
      "relevance_score": 0.92,
      "recency_score": 0.85,
      "support_snippet": "Caltrans delayed payment for 45 days beyond contract terms",
      "doc_refs": [
        {
          "doc": "Payment_Dispute.pdf",
          "page": 5
        }
      ]
    }
  ],
  "context_summary": "Multiple payment disputes identified with Caltrans, primarily related to change order approvals and retention releases.",
  "recall_time": 0.45
}
```

---

## Agent Management API

### Get Agent Status

**Endpoint:** `GET /api/matters/{matter_id}/agent/status`

**Description:** Get current status and configuration of matter's agent.

**Response:**
```json
{
  "agent_id": "agent_12345",
  "status": "healthy",
  "created_at": "2025-08-01T10:00:00Z",
  "last_active": "2025-08-19T10:30:00Z",
  "configuration": {
    "name": "Construction Claims Agent - Matter ABC",
    "persona": "california_construction_expert",
    "llm_provider": "ollama",
    "model": "gpt-oss:20b",
    "temperature": 0.2,
    "max_tokens": 900
  },
  "memory_stats": {
    "core_memory_usage": "2,156 / 8,192 chars",
    "archival_items": 245,
    "conversation_buffer": 12
  },
  "performance": {
    "avg_response_time": 2.3,
    "memory_operations": 1847,
    "uptime": "18 days"
  }
}
```

---

### Update Agent Configuration

**Endpoint:** `PUT /api/matters/{matter_id}/agent/config`

**Description:** Update agent configuration settings.

**Request Body:**
```json
{
  "llm_provider": {
    "type": "gemini",
    "model": "gemini-2.5-flash",
    "temperature": 0.3,
    "max_tokens": 1000
  },
  "domain_config": {
    "type": "california_construction",
    "entity_extraction": true,
    "compliance_validation": true
  },
  "memory_settings": {
    "max_items": 15000,
    "pruning_enabled": true,
    "importance_threshold": 0.4
  }
}
```

**Response:**
```json
{
  "updated": true,
  "agent_id": "agent_12345",
  "changes_applied": [
    "llm_provider",
    "domain_config",
    "memory_settings"
  ],
  "restart_required": false
}
```

---

### Delete Agent

**Endpoint:** `DELETE /api/matters/{matter_id}/agent`

**Description:** Delete agent and all associated memory.

**Query Parameters:**
- `backup` (optional): Create backup before deletion (default: true)

**Response:**
```json
{
  "deleted": true,
  "agent_id": "agent_12345",
  "backup_created": true,
  "backup_path": "~/.letta-claim/backups/agent_12345_2025-08-19.json",
  "items_deleted": {
    "memory_items": 245,
    "conversations": 18,
    "configuration": 1
  }
}
```

---

## Health and Monitoring API

### Get Letta Health Status

**Endpoint:** `GET /api/letta/health`

**Description:** Comprehensive health check for Letta server and integration.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-19T10:30:00Z",
  "server": {
    "status": "running",
    "version": "0.10.0",
    "uptime": "18 days, 5 hours",
    "host": "localhost",
    "port": 8283,
    "response_time": 8.2
  },
  "connection": {
    "status": "connected",
    "pool_size": 5,
    "active_connections": 2,
    "connection_time": 1.1,
    "retry_count": 0
  },
  "memory": {
    "total_agents": 3,
    "total_items": 1247,
    "database_size": "45.2 MB",
    "backup_status": "current"
  },
  "performance": {
    "avg_response_time": 245.6,
    "requests_per_minute": 12.4,
    "error_rate": 0.002,
    "memory_operations_per_hour": 48
  },
  "issues": []
}
```

**Health Status Codes:**
- `healthy`: All systems operational
- `degraded`: Minor issues, functionality reduced
- `error`: Significant problems, core features affected
- `unknown`: Unable to determine status

---

### Get Performance Metrics

**Endpoint:** `GET /api/letta/metrics`

**Description:** Detailed performance and usage metrics.

**Query Parameters:**
- `window` (optional): Time window for metrics (1h, 24h, 7d, 30d)
- `detailed` (optional): Include detailed breakdown

**Response:**
```json
{
  "time_window": "24h",
  "timestamp": "2025-08-19T10:30:00Z",
  "request_metrics": {
    "total_requests": 1247,
    "successful_requests": 1244,
    "failed_requests": 3,
    "avg_response_time": 245.6,
    "p95_response_time": 567.2,
    "requests_per_hour": [48, 52, 43, 67, 55]
  },
  "memory_metrics": {
    "memory_operations": 387,
    "storage_operations": 45,
    "recall_operations": 342,
    "avg_recall_time": 156.3,
    "cache_hit_rate": 0.34
  },
  "agent_metrics": {
    "active_agents": 3,
    "total_conversations": 89,
    "avg_conversation_length": 12.4,
    "memory_items_created": 156
  },
  "resource_usage": {
    "cpu_usage": 12.5,
    "memory_usage": 245.6,
    "disk_usage": 1247.3,
    "network_io": {
      "bytes_sent": 15678234,
      "bytes_received": 9876543
    }
  }
}
```

---

## LettaAdapter Class API

### Core Methods

#### `__init__(matter: Matter)`
Initialize LettaAdapter for a specific matter.

**Parameters:**
- `matter`: Matter instance

**Raises:**
- `LettaConnectionError`: If unable to connect to server
- `LettaConfigurationError`: If configuration invalid

---

#### `async store_interaction(query: str, response: str, sources: List[SourceChunk]) -> bool`
Store a chat interaction in agent memory.

**Parameters:**
- `query`: User's question
- `response`: Assistant's response  
- `sources`: Source chunks used in response

**Returns:** `bool` - Success status

**Raises:**
- `LettaMemoryError`: If storage operation fails

---

#### `async recall_context(query: str, k: int = 8) -> List[str]`
Recall relevant context for a query.

**Parameters:**
- `query`: Query string for context recall
- `k`: Maximum number of context items

**Returns:** `List[str]` - List of relevant memory items

---

#### `async store_knowledge_batch(items: List[KnowledgeItem]) -> Dict[str, Any]`
Store multiple knowledge items efficiently.

**Parameters:**
- `items`: List of knowledge items to store

**Returns:** Dictionary with storage results and statistics

---

#### `async get_memory_stats() -> Dict[str, Any]`
Get comprehensive memory statistics.

**Returns:** Dictionary with memory usage and performance data

---

### Memory Management Methods

#### `async update_core_memory_smart(updates: Dict[str, str]) -> bool`
Intelligently update core memory blocks.

#### `async semantic_memory_search(query: str, filters: Dict = None) -> List[Dict]`
Perform semantic search through memory.

#### `async get_memory_summary(max_items: int = 50) -> Dict[str, Any]`
Generate comprehensive memory summary.

#### `async export_memory(format: str = "json") -> str`
Export all memory data.

#### `async import_memory(data: str, format: str = "json") -> bool`
Import memory data from backup.

---

### Provider Management Methods

#### `async switch_provider(provider: str, model: str, **kwargs) -> bool`
Switch LLM provider for agent.

#### `async test_provider_configuration(provider: str, **kwargs) -> Dict[str, Any]`
Test provider configuration without switching.

#### `async get_usage_stats() -> Dict[str, Any]`
Get LLM usage statistics and costs.

#### `async set_spending_limit(limit_type: str, amount: float) -> bool`
Set spending limits for external providers.

---

### California Domain Methods

#### `async extract_california_entities(text: str) -> List[Dict[str, Any]]`
Extract California-specific legal entities from text.

#### `async validate_california_compliance(claim_data: Dict) -> Dict[str, Any]`
Validate compliance with California construction law.

#### `async generate_followup_questions(context: str, category: str = None) -> List[str]`
Generate domain-specific follow-up questions.

---

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "LETTA_CONNECTION_ERROR",
    "message": "Unable to connect to Letta server",
    "details": {
      "host": "localhost",
      "port": 8283,
      "timeout": 10
    },
    "timestamp": "2025-08-19T10:30:00Z",
    "trace_id": "trace_12345"
  }
}
```

### Error Codes

#### Connection Errors
- `LETTA_CONNECTION_ERROR`: Cannot connect to server
- `LETTA_TIMEOUT_ERROR`: Request timed out
- `LETTA_AUTH_ERROR`: Authentication failed

#### Memory Errors
- `LETTA_MEMORY_ERROR`: Memory operation failed
- `LETTA_STORAGE_ERROR`: Storage operation failed
- `LETTA_CAPACITY_ERROR`: Memory capacity exceeded

#### Agent Errors
- `LETTA_AGENT_ERROR`: Agent operation failed
- `LETTA_AGENT_NOT_FOUND`: Agent not found
- `LETTA_CONFIGURATION_ERROR`: Invalid configuration

#### Provider Errors
- `LETTA_PROVIDER_ERROR`: LLM provider error
- `LETTA_PROVIDER_RATE_LIMIT`: Rate limit exceeded
- `LETTA_PROVIDER_AUTH_ERROR`: Provider authentication failed

---

## Rate Limits

### Default Limits
- **Memory operations**: 100 per minute
- **Agent operations**: 50 per minute
- **Health checks**: 200 per minute
- **Statistics requests**: 20 per minute

### Rate Limit Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1692441600
X-RateLimit-Window: 60
```

### Rate Limit Error Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Memory operation rate limit exceeded",
    "retry_after": 23
  }
}
```

---

## Authentication

### API Key Authentication
For production deployments with authentication enabled:

```bash
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8000/api/letta/health
```

### JWT Authentication
```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
     http://localhost:8000/api/matters/matter-id/memory/stats
```

---

## Versioning

### API Version Header
```
Accept: application/json; version=1.0
```

### Version Compatibility
- **v1.0**: Current stable version
- **v0.9**: Legacy version (deprecated)

### Breaking Changes
Breaking changes are introduced only in major version updates and are documented in the changelog.

---

## SDKs and Libraries

### Python Client
```python
from letta_client import LettaClient

client = LettaClient(
    host="localhost",
    port=8000,
    api_key="your-api-key"
)

# Get memory stats
stats = await client.get_memory_stats(matter_id="uuid")
```

### JavaScript Client
```javascript
import { LettaClient } from 'letta-client-js';

const client = new LettaClient({
  host: 'localhost',
  port: 8000,
  apiKey: 'your-api-key'
});

// Store knowledge items
await client.storeKnowledge(matterId, items);
```

---

## Testing and Development

### Test Endpoints
Development-only endpoints for testing:

- `POST /api/test/letta/reset` - Reset test agent
- `GET /api/test/letta/mock-data` - Generate mock memory data
- `POST /api/test/letta/simulate-error` - Simulate error conditions

### Mock Server
Use the mock Letta server for testing:

```bash
letta-claim-assistant --mock-letta-server --port 8283
```

---

## Support and Resources

### API Documentation
- **Interactive docs**: http://localhost:8000/docs
- **OpenAPI spec**: http://localhost:8000/openapi.json
- **Redoc**: http://localhost:8000/redoc

### Code Examples
- **Python examples**: `/examples/python/`
- **JavaScript examples**: `/examples/javascript/`
- **curl examples**: `/examples/curl/`

### Community
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: API questions and best practices
- **Wiki**: Additional documentation and tutorials

---

*For implementation details and advanced usage, see the Letta Technical Reference and source code documentation.*