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

from .logging_conf import get_logger
from .matters import matter_manager, Matter
from .jobs import job_queue, JobInfo
from .rag import RAGResponse

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


# Pydantic models for request/response
class CreateMatterRequest(BaseModel):
    name: str


class MatterResponse(BaseModel):
    id: str
    name: str
    slug: str
    created_at: str
    paths: Dict[str, str]


class ChatRequest(BaseModel):
    matter_id: str
    query: str
    k: int = 8
    model: Optional[str] = None
    max_tokens: int = 900


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    followups: List[str]
    used_memory: List[Dict[str, Any]]


class ModelSettings(BaseModel):
    provider: str
    generation_model: str
    embedding_model: str
    api_key: Optional[str] = None


# Matter Management Endpoints
@app.post("/api/matters", response_model=MatterResponse)
async def create_matter(request: CreateMatterRequest):
    """Create a new Matter with filesystem structure."""
    try:
        matter = matter_manager.create_matter(request.name)
        return MatterResponse(
            id=matter.id,
            name=matter.name,
            slug=matter.slug,
            created_at=matter.created_at.isoformat(),
            paths={
                "root": str(matter.paths.root),
                "docs": str(matter.paths.docs),
                "vectors": str(matter.paths.vectors),
                "knowledge": str(matter.paths.knowledge)
            }
        )
    except Exception as e:
        logger.error("Failed to create matter", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/matters")
async def list_matters():
    """List all available Matters."""
    try:
        matters = matter_manager.list_matters()
        return [
            MatterResponse(
                id=matter.id,
                name=matter.name,
                slug=matter.slug,
                created_at=matter.created_at.isoformat(),
                paths={
                    "root": str(matter.paths.root),
                    "docs": str(matter.paths.docs),
                    "vectors": str(matter.paths.vectors),
                    "knowledge": str(matter.paths.knowledge)
                }
            )
            for matter in matters
        ]
    except Exception as e:
        logger.error("Failed to list matters", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/matters/{matter_id}/switch")
async def switch_matter(matter_id: str):
    """Switch active Matter context."""
    try:
        matter = matter_manager.switch_matter(matter_id)
        return {"status": "switched", "matter_id": matter.id, "matter_name": matter.name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to switch matter", error=str(e))
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
            "job_id": job_info.job_id,
            "status": job_info.status.value,
            "progress": job_info.progress,
            "message": job_info.message,
            "error_message": job_info.error_message
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