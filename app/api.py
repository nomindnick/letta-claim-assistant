"""
FastAPI backend surface for the Letta Construction Claim Assistant.

Provides HTTP API endpoints for matter management, file upload,
job status tracking, chat/RAG operations, and settings management.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from .models import (
    CreateMatterRequest, CreateMatterResponse, MatterSummary,
    ChatRequest, ChatResponse, JobStatus, DocumentInfo
)
from .logging_conf import get_logger
from .matters import matter_manager
from .jobs import job_queue

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(title="Letta Claim Assistant API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models are now imported from models.py

class ModelSettings(BaseModel):
    provider: str
    generation_model: str
    embedding_model: str
    api_key: Optional[str] = None


# Matter Management Endpoints
@app.post("/api/matters", response_model=CreateMatterResponse)
async def create_matter(request: CreateMatterRequest):
    """Create a new Matter with filesystem structure."""
    try:
        matter = matter_manager.create_matter(request.name)
        return CreateMatterResponse(
            id=matter.id,
            slug=matter.slug,
            paths={
                "root": str(matter.paths.root),
                "docs": str(matter.paths.docs),
                "docs_ocr": str(matter.paths.docs_ocr),
                "parsed": str(matter.paths.parsed),
                "vectors": str(matter.paths.vectors),
                "knowledge": str(matter.paths.knowledge),
                "chat": str(matter.paths.chat),
                "logs": str(matter.paths.logs)
            }
        )
    except ValueError as e:
        logger.error("Invalid matter creation request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create matter", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/matters", response_model=List[MatterSummary])
async def list_matters():
    """List all available Matters with summary information."""
    try:
        matter_summaries = matter_manager.list_matter_summaries()
        return matter_summaries
    except Exception as e:
        logger.error("Failed to list matters", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/matters/{matter_id}/switch")
async def switch_matter(matter_id: str):
    """Switch active Matter context."""
    try:
        matter = matter_manager.switch_matter(matter_id)
        logger.info(
            "Matter switched via API",
            matter_id=matter.id,
            matter_name=matter.name,
            matter_slug=matter.slug
        )
        return {
            "status": "switched", 
            "matter_id": matter.id, 
            "matter_name": matter.name,
            "matter_slug": matter.slug
        }
    except ValueError as e:
        logger.warning("Matter not found for switching", matter_id=matter_id)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to switch matter", matter_id=matter_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/matters/active")
async def get_active_matter():
    """Get the currently active Matter."""
    try:
        active_matter = matter_manager.get_active_matter()
        if active_matter is None:
            return {"active_matter": None}
        
        return {
            "active_matter": {
                "id": active_matter.id,
                "name": active_matter.name,
                "slug": active_matter.slug,
                "created_at": active_matter.created_at.isoformat()
            }
        }
    except Exception as e:
        logger.error("Failed to get active matter", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Upload & Ingestion Endpoints
@app.post("/api/matters/{matter_id}/upload")
async def upload_files(matter_id: str, files: List[UploadFile] = File(...)):
    """Upload and process PDF files for a Matter."""
    try:
        # TODO: Implement file upload and processing
        # This should:
        # 1. Validate matter exists
        # 2. Save uploaded files to matter's docs directory
        # 3. Submit background job for processing
        # 4. Return job ID
        
        job_id = await job_queue.submit_job(
            "pdf_ingestion",
            {"matter_id": matter_id, "files": [f.filename for f in files]}
        )
        
        return {"job_id": job_id}
    except Exception as e:
        logger.error("Failed to upload files", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a background job."""
    try:
        job_info = await job_queue.get_job_status(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "id": job_info.job_id,
            "status": job_info.status.value,
            "progress": job_info.progress,
            "detail": job_info.message,
            "error": job_info.error_message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get job status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Chat / RAG Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat query using RAG pipeline."""
    try:
        # TODO: Implement chat/RAG processing
        # This should:
        # 1. Validate matter exists and is active
        # 2. Initialize RAG engine for matter
        # 3. Process query through RAG pipeline
        # 4. Return structured response
        
        raise NotImplementedError("Chat endpoint not yet implemented")
    except Exception as e:
        logger.error("Failed to process chat request", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Settings / Models Endpoints
@app.get("/api/settings/models")
async def get_model_settings():
    """Get current model and provider settings."""
    try:
        # TODO: Implement settings retrieval
        # Return current provider, models, and available options
        raise NotImplementedError("Model settings endpoint not yet implemented")
    except Exception as e:
        logger.error("Failed to get model settings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/settings/models")
async def update_model_settings(settings: ModelSettings):
    """Update model and provider settings."""
    try:
        # TODO: Implement settings update
        # Test connection, update configuration, return success status
        raise NotImplementedError("Model settings update not yet implemented")
    except Exception as e:
        logger.error("Failed to update model settings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "message": "Letta Claim Assistant API is running"}


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Letta Claim Assistant API")
    await job_queue.start_workers()


# Shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Letta Claim Assistant API")