"""
Shared pytest configuration and fixtures for Letta Construction Claim Assistant tests.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.models import Matter, MatterPaths


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_matter_dir():
    """Create temporary matter directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        matter_root = Path(temp_dir) / "test_matter"
        matter_root.mkdir()
        
        # Create subdirectories
        paths = MatterPaths.from_root(matter_root)
        for path in [paths.docs, paths.docs_ocr, paths.parsed, paths.vectors, 
                    paths.knowledge, paths.chat, paths.logs]:
            path.mkdir(parents=True, exist_ok=True)
        
        yield matter_root


@pytest.fixture
def test_matter(temp_matter_dir):
    """Create test matter with temporary directory."""
    paths = MatterPaths.from_root(temp_matter_dir)
    return Matter(
        id="test-matter-001",
        name="Test Matter",
        slug="test-matter",
        embedding_model="nomic-embed-text",
        generation_model="gpt-oss:20b",
        paths=paths
    )


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return {
        "pages": [
            {
                "page_no": 1,
                "text": "This is the first page of a construction contract. It contains important terms and conditions for the project.",
                "doc_name": "contract.pdf"
            },
            {
                "page_no": 2,
                "text": "This is the second page containing details about the dry well installation requirements and specifications.",
                "doc_name": "contract.pdf"
            }
        ],
        "metadata": {
            "total_pages": 2,
            "doc_name": "contract.pdf",
            "file_size": 1024000
        }
    }


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    from app.chunking import Chunk
    
    return [
        Chunk(
            id="chunk-1",
            text="This is the first page of a construction contract. It contains important terms and conditions.",
            doc_id="contract",
            doc_name="contract.pdf",
            page_start=1,
            page_end=1,
            metadata={"section": "terms"},
            token_count=15
        ),
        Chunk(
            id="chunk-2", 
            text="This is the second page containing details about the dry well installation requirements.",
            doc_id="contract",
            doc_name="contract.pdf",
            page_start=2,
            page_end=2,
            metadata={"section": "specifications"},
            token_count=13
        )
    ]


@pytest.fixture
def mock_ollama_provider():
    """Mock Ollama provider for testing."""
    provider = AsyncMock()
    provider.generate.return_value = "Mocked LLM response with [contract.pdf p.1] citation."
    provider.embed.return_value = [[0.1] * 768]  # 768-dim embedding
    return provider


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = AsyncMock()
    store.search.return_value = [
        {
            "chunk_id": "chunk-1",
            "doc_name": "contract.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Sample text from document",
            "similarity_score": 0.85,
            "metadata": {"section": "terms"}
        }
    ]
    return store


@pytest.fixture
def mock_letta_adapter():
    """Mock Letta adapter for testing."""
    adapter = AsyncMock()
    adapter.recall.return_value = [
        {
            "type": "Fact",
            "label": "Dry well installation required",
            "date": "2023-02-15",
            "actors": ["Contractor"],
            "doc_refs": [{"doc": "contract.pdf", "page": 2}],
            "support_snippet": "Installation must meet specifications"
        }
    ]
    adapter.suggest_followups.return_value = [
        "What are the timeline requirements?",
        "Who is responsible for permits?"
    ]
    return adapter


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.get_global_config.return_value = {
        "llm": {
            "provider": "ollama",
            "model": "gpt-oss:20b",
            "temperature": 0.2,
            "max_tokens": 900
        },
        "embeddings": {
            "provider": "ollama", 
            "model": "nomic-embed-text"
        },
        "ocr": {
            "enabled": True,
            "force_ocr": False,
            "language": "eng",
            "skip_text": True
        }
    }
    return settings


class MockJobQueue:
    """Mock job queue for testing."""
    
    def __init__(self):
        self.jobs = {}
        self.job_counter = 0
    
    async def submit_job(self, job_type: str, params: Dict[str, Any], progress_callback=None):
        job_id = f"job-{self.job_counter}"
        self.job_counter += 1
        self.jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "params": params,
            "status": "completed",
            "progress": 1.0,
            "result": {"success": True}
        }
        return job_id
    
    async def get_job_status(self, job_id: str):
        return self.jobs.get(job_id, {"status": "not_found"})


@pytest.fixture
def mock_job_queue():
    """Mock job queue for testing."""
    return MockJobQueue()


# Performance test utilities
@pytest.fixture
def performance_timer():
    """Timer utility for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
    
    return Timer()


# Test data generators
def generate_large_text(num_pages: int = 100, words_per_page: int = 500):
    """Generate large text content for performance testing."""
    import random
    
    words = ["construction", "project", "contract", "specifications", "requirements",
             "installation", "materials", "timeline", "schedule", "delivery",
             "quality", "standards", "inspection", "compliance", "documentation"]
    
    pages = []
    for page_num in range(1, num_pages + 1):
        text = " ".join(random.choices(words, k=words_per_page))
        pages.append({
            "page_no": page_num,
            "text": text,
            "doc_name": f"large_document.pdf"
        })
    
    return pages


@pytest.fixture
def large_document():
    """Generate large document for performance testing."""
    return {
        "pages": generate_large_text(50, 400),  # 50 pages, 400 words each
        "metadata": {
            "total_pages": 50,
            "doc_name": "large_document.pdf",
            "file_size": 5000000  # 5MB
        }
    }


# Letta-specific fixtures
@pytest.fixture
def mock_letta_client():
    """Mock Letta client for testing."""
    from unittest.mock import AsyncMock
    
    client = AsyncMock()
    
    # Mock health check
    client.health.check.return_value = None
    
    # Mock agent operations
    client.agents.create.return_value = Mock(id="test-agent-123", name="test-agent")
    client.agents.retrieve.return_value = Mock(id="test-agent-123", name="test-agent")
    client.agents.delete.return_value = None
    client.agents.list.return_value = []
    
    # Mock memory operations
    client.agents.search_archival_memory.return_value = []
    client.agents.insert_archival_memory.return_value = Mock(id="mem-123")
    client.agents.get_archival_memory.return_value = []
    
    return client


@pytest.fixture
def mock_letta_server():
    """Mock Letta server manager for testing."""
    from unittest.mock import MagicMock
    
    server = MagicMock()
    server._is_running = True
    server.port = 8283
    server.host = "localhost"
    server.get_base_url.return_value = "http://localhost:8283"
    server.start.return_value = True
    server.stop.return_value = True
    
    return server


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for testing."""
    from app.letta_circuit_breaker import CircuitBreaker, CircuitState
    
    breaker = CircuitBreaker(
        name="test_breaker",
        failure_threshold=3,
        success_threshold=2,
        timeout=1.0
    )
    return breaker


@pytest.fixture
def mock_request_queue():
    """Mock request queue for testing."""
    from app.letta_request_queue import RequestQueue
    
    queue = RequestQueue(
        max_queue_size=100,
        batch_size=10,
        batch_timeout=0.5
    )
    return queue


@pytest.fixture
async def letta_test_adapter(test_matter):
    """Create test Letta adapter with mocked client."""
    from app.letta_adapter import LettaAdapter
    from unittest.mock import patch, AsyncMock
    
    with patch('app.letta_adapter.AsyncLetta') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock successful initialization
        mock_client.agents.create.return_value = Mock(id="test-agent")
        mock_client.health.check.return_value = None
        
        adapter = LettaAdapter(test_matter)
        await adapter.initialize()
        
        yield adapter


@pytest.fixture
def letta_knowledge_items():
    """Sample knowledge items for Letta testing."""
    return [
        {
            "type": "Entity",
            "label": "ABC Construction Inc.",
            "actors": ["General Contractor"],
            "doc_refs": [{"doc": "contract.pdf", "page": 1}]
        },
        {
            "type": "Event",
            "label": "Foundation failure",
            "date": "2024-02-14",
            "actors": ["ABC Construction Inc."],
            "doc_refs": [{"doc": "incident_report.pdf", "page": 3}],
            "support_snippet": "Significant cracking observed in foundation"
        },
        {
            "type": "Issue",
            "label": "Delay claim for weather",
            "date": "2024-03-01",
            "actors": ["ABC Construction Inc.", "Owner"],
            "doc_refs": [{"doc": "claim.pdf", "page": 1}]
        },
        {
            "type": "Fact",
            "label": "Project completion date extended by 30 days",
            "date": "2024-03-15",
            "doc_refs": [{"doc": "change_order.pdf", "page": 2}]
        }
    ]


@pytest.fixture
def letta_memory_data():
    """Sample memory data for testing."""
    return {
        "conversations": [
            {
                "timestamp": "2024-01-01T10:00:00",
                "user": "What caused the foundation failure?",
                "assistant": "The foundation failure was caused by inadequate soil preparation.",
                "sources": ["report.pdf p.5-7"]
            }
        ],
        "facts": [
            "Foundation failure occurred on 2024-02-14",
            "ABC Construction was the general contractor",
            "Weather delays totaled 15 days"
        ],
        "entities": {
            "contractors": ["ABC Construction Inc.", "XYZ Subcontractor"],
            "owners": ["Property Owner LLC"],
            "dates": ["2024-02-14", "2024-03-01", "2024-03-15"]
        }
    }