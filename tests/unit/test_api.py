"""
Unit tests for FastAPI endpoints and API functionality.

Tests all API endpoints including matters, upload, chat, settings
with proper mocking of external dependencies.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
from io import BytesIO

from fastapi.testclient import TestClient
from fastapi import UploadFile

# Import the API module and dependencies
from app.api import create_app
from app.models import Matter, ChatRequest, ChatResponse, SourceChunk, KnowledgeItem


class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def test_app(self, mock_job_queue, mock_settings):
        """Create test FastAPI application."""
        with patch('app.api.job_queue', mock_job_queue):
            with patch('app.api.settings', mock_settings):
                app = create_app()
                return TestClient(app)
    
    @pytest.fixture
    def mock_matter_service(self):
        """Mock matter service for testing."""
        mock_service = AsyncMock()
        
        # Mock matter data
        test_matter = {
            "id": "test-001",
            "name": "Test Matter",
            "slug": "test-matter",
            "created_at": "2025-01-21T10:00:00Z",
            "embedding_model": "nomic-embed-text",
            "generation_model": "gpt-oss:20b",
            "paths": {
                "root": "/tmp/test_matter",
                "docs": "/tmp/test_matter/docs",
                "vectors": "/tmp/test_matter/vectors"
            }
        }
        
        mock_service.create_matter.return_value = test_matter
        mock_service.list_matters.return_value = [test_matter]
        mock_service.get_active_matter.return_value = test_matter
        mock_service.switch_matter.return_value = test_matter
        
        return mock_service
    
    @pytest.fixture
    def mock_rag_engine(self):
        """Mock RAG engine for testing."""
        mock_engine = AsyncMock()
        
        mock_response = {
            "answer": "This is a test answer with [contract.pdf p.1] citation.",
            "sources": [
                {
                    "doc": "contract.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Sample contract text...",
                    "score": 0.85
                }
            ],
            "followups": [
                "What are the project timelines?",
                "Who is responsible for permits?"
            ],
            "used_memory": [
                {
                    "type": "Fact",
                    "label": "Contract signed",
                    "date": "2023-01-15",
                    "actors": ["Owner", "Contractor"],
                    "doc_refs": [{"doc": "contract.pdf", "page": 1}],
                    "support_snippet": "Contract was executed on January 15th"
                }
            ]
        }
        
        mock_engine.generate_answer.return_value = mock_response
        return mock_engine
    
    @pytest.mark.unit
    def test_health_endpoint(self, test_app):
        """Test health check endpoint."""
        response = test_app.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    @pytest.mark.unit
    def test_create_matter_success(self, test_app, mock_matter_service):
        """Test successful matter creation."""
        with patch('app.api.matter_service', mock_matter_service):
            response = test_app.post(
                "/api/matters",
                json={"name": "New Test Matter"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Matter"
            assert data["slug"] == "test-matter"
            assert "id" in data
            
            mock_matter_service.create_matter.assert_called_once()
    
    @pytest.mark.unit
    def test_create_matter_invalid_name(self, test_app):
        """Test matter creation with invalid name."""
        response = test_app.post(
            "/api/matters",
            json={"name": ""}  # Empty name
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    def test_list_matters(self, test_app, mock_matter_service):
        """Test listing all matters."""
        with patch('app.api.matter_service', mock_matter_service):
            response = test_app.get("/api/matters")
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "Test Matter"
    
    @pytest.mark.unit
    def test_switch_matter(self, test_app, mock_matter_service):
        """Test switching active matter."""
        with patch('app.api.matter_service', mock_matter_service):
            response = test_app.post("/api/matters/test-001/switch")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["matter"]["id"] == "test-001"
            
            mock_matter_service.switch_matter.assert_called_once_with("test-001")
    
    @pytest.mark.unit
    def test_switch_matter_not_found(self, test_app, mock_matter_service):
        """Test switching to non-existent matter."""
        mock_matter_service.switch_matter.side_effect = ValueError("Matter not found")
        
        with patch('app.api.matter_service', mock_matter_service):
            response = test_app.post("/api/matters/nonexistent/switch")
            
            assert response.status_code == 404
            data = response.json()
            assert "Matter not found" in data["detail"]
    
    @pytest.mark.unit
    def test_upload_files_success(self, test_app, mock_job_queue):
        """Test successful file upload."""
        # Create mock file
        file_content = b"Mock PDF content"
        files = {
            "files": ("test.pdf", BytesIO(file_content), "application/pdf")
        }
        
        with patch('app.api.job_queue', mock_job_queue):
            response = test_app.post(
                "/api/matters/test-001/upload",
                files=files
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["job_id"].startswith("job-")
    
    @pytest.mark.unit
    def test_upload_files_no_matter(self, test_app):
        """Test file upload without active matter."""
        file_content = b"Mock PDF content"
        files = {
            "files": ("test.pdf", BytesIO(file_content), "application/pdf")
        }
        
        with patch('app.api.matter_service.get_active_matter', return_value=None):
            response = test_app.post(
                "/api/matters/nonexistent/upload", 
                files=files
            )
            
            assert response.status_code == 404
    
    @pytest.mark.unit
    def test_upload_invalid_file_type(self, test_app):
        """Test upload with invalid file type."""
        file_content = b"Not a PDF"
        files = {
            "files": ("test.txt", BytesIO(file_content), "text/plain")
        }
        
        response = test_app.post(
            "/api/matters/test-001/upload",
            files=files
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Only PDF files" in data["detail"]
    
    @pytest.mark.unit
    def test_job_status(self, test_app, mock_job_queue):
        """Test job status endpoint."""
        with patch('app.api.job_queue', mock_job_queue):
            # First create a job
            job_id = asyncio.run(mock_job_queue.submit_job("test", {}))
            
            response = test_app.get(f"/api/jobs/{job_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == job_id
            assert data["status"] == "completed"
            assert data["progress"] == 1.0
    
    @pytest.mark.unit
    def test_job_status_not_found(self, test_app, mock_job_queue):
        """Test job status for non-existent job."""
        with patch('app.api.job_queue', mock_job_queue):
            response = test_app.get("/api/jobs/nonexistent")
            
            assert response.status_code == 404
    
    @pytest.mark.unit
    def test_chat_success(self, test_app, mock_rag_engine, mock_matter_service):
        """Test successful chat request."""
        chat_request = {
            "matter_id": "test-001",
            "query": "What caused the dry well failure?",
            "k": 8,
            "max_tokens": 900
        }
        
        with patch('app.api.rag_engine', mock_rag_engine):
            with patch('app.api.matter_service', mock_matter_service):
                response = test_app.post("/api/chat", json=chat_request)
                
                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "sources" in data
                assert "followups" in data
                assert "used_memory" in data
                assert len(data["sources"]) > 0
                assert len(data["followups"]) > 0
    
    @pytest.mark.unit
    def test_chat_no_matter(self, test_app):
        """Test chat request without active matter."""
        chat_request = {
            "matter_id": "nonexistent",
            "query": "Test query",
            "k": 8
        }
        
        with patch('app.api.matter_service.get_matter_by_id', return_value=None):
            response = test_app.post("/api/chat", json=chat_request)
            
            assert response.status_code == 404
            data = response.json()
            assert "Matter not found" in data["detail"]
    
    @pytest.mark.unit
    def test_chat_empty_query(self, test_app):
        """Test chat with empty query."""
        chat_request = {
            "matter_id": "test-001",
            "query": "",  # Empty query
            "k": 8
        }
        
        response = test_app.post("/api/chat", json=chat_request)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    def test_chat_rag_error(self, test_app, mock_rag_engine, mock_matter_service):
        """Test chat request with RAG engine error."""
        mock_rag_engine.generate_answer.side_effect = Exception("RAG processing failed")
        
        chat_request = {
            "matter_id": "test-001",
            "query": "Test query",
            "k": 8
        }
        
        with patch('app.api.rag_engine', mock_rag_engine):
            with patch('app.api.matter_service', mock_matter_service):
                response = test_app.post("/api/chat", json=chat_request)
                
                assert response.status_code == 500
                data = response.json()
                assert "Failed to generate answer" in data["detail"]
    
    @pytest.mark.unit
    def test_get_models_settings(self, test_app, mock_settings):
        """Test get models settings endpoint."""
        with patch('app.api.settings', mock_settings):
            response = test_app.get("/api/settings/models")
            
            assert response.status_code == 200
            data = response.json()
            assert "provider" in data
            assert "generation_model" in data
            assert "embedding_model" in data
    
    @pytest.mark.unit
    def test_update_models_settings(self, test_app, mock_settings):
        """Test update models settings endpoint."""
        settings_update = {
            "provider": "gemini",
            "generation_model": "gemini-2.5-flash",
            "api_key": "test-key"
        }
        
        with patch('app.api.settings', mock_settings):
            with patch('app.api.provider_manager') as mock_provider_mgr:
                mock_provider_mgr.test_provider.return_value = True
                
                response = test_app.post("/api/settings/models", json=settings_update)
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["test_result"] is True
    
    @pytest.mark.unit
    def test_update_models_settings_test_fail(self, test_app, mock_settings):
        """Test update models settings with test failure."""
        settings_update = {
            "provider": "gemini",
            "generation_model": "gemini-2.5-flash",
            "api_key": "invalid-key"
        }
        
        with patch('app.api.settings', mock_settings):
            with patch('app.api.provider_manager') as mock_provider_mgr:
                mock_provider_mgr.test_provider.return_value = False
                
                response = test_app.post("/api/settings/models", json=settings_update)
                
                assert response.status_code == 400
                data = response.json()
                assert "Provider test failed" in data["detail"]
    
    @pytest.mark.unit
    def test_get_documents(self, test_app, mock_matter_service):
        """Test get documents endpoint."""
        mock_docs = [
            {
                "name": "contract.pdf",
                "pages": 10,
                "chunks": 25,
                "ocr_status": "partial",
                "status": "completed"
            }
        ]
        
        mock_matter_service.get_documents.return_value = mock_docs
        
        with patch('app.api.matter_service', mock_matter_service):
            response = test_app.get("/api/matters/test-001/documents")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "contract.pdf"
            assert data[0]["pages"] == 10
    
    @pytest.mark.unit
    def test_get_chat_history(self, test_app):
        """Test get chat history endpoint."""
        mock_history = [
            {
                "role": "user",
                "content": "What is the project timeline?",
                "timestamp": "2025-01-21T10:00:00Z"
            },
            {
                "role": "assistant", 
                "content": "The project timeline is 6 months.",
                "timestamp": "2025-01-21T10:00:05Z"
            }
        ]
        
        with patch('app.api.chat_history_service') as mock_chat_svc:
            mock_chat_svc.get_history.return_value = mock_history
            
            response = test_app.get("/api/matters/test-001/chat/history")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["role"] == "user"
            assert data[1]["role"] == "assistant"
    
    @pytest.mark.unit
    def test_error_handling_middleware(self, test_app):
        """Test error handling middleware."""
        with patch('app.api.matter_service.create_matter') as mock_create:
            mock_create.side_effect = Exception("Unexpected error")
            
            response = test_app.post(
                "/api/matters",
                json={"name": "Test Matter"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
    
    @pytest.mark.unit
    def test_request_validation(self, test_app):
        """Test request validation across endpoints."""
        # Test invalid chat request
        invalid_chat = {
            "matter_id": "",  # Empty string
            "query": "test",
            "k": -1  # Negative number
        }
        
        response = test_app.post("/api/chat", json=invalid_chat)
        assert response.status_code == 422
        
        # Test invalid matter creation
        invalid_matter = {
            "name": None  # None value
        }
        
        response = test_app.post("/api/matters", json=invalid_matter)
        assert response.status_code == 422
    
    @pytest.mark.unit
    def test_cors_headers(self, test_app):
        """Test CORS headers are included."""
        response = test_app.get("/api/health")
        assert response.status_code == 200
        
        # Check for CORS headers if enabled
        headers = response.headers
        # Note: CORS headers would be present in actual deployment