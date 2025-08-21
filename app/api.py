"""
FastAPI backend surface for the Letta Construction Claim Assistant.

Provides HTTP API endpoints for matter management, file upload,
job status tracking, chat/RAG operations, and settings management.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
import json
import time
import traceback
import io
from datetime import datetime

from .models import (
    CreateMatterRequest, CreateMatterResponse, MatterSummary,
    ChatRequest, ChatResponse, ChatMode, JobStatus, DocumentInfo,
    QualityInsights, RetrievalWeights,
    CreateMemoryItemRequest, UpdateMemoryItemRequest, MemoryOperationResponse,
    MemoryCommandRequest, MemoryCommandResponse
)
from .logging_conf import get_logger
from .matters import matter_manager
from .jobs import job_queue, ensure_job_queue_started
from .rag import RAGEngine
from .llm.provider_manager import provider_manager
from .letta_agent import LettaAgentHandler, AgentResponse
from .quality_metrics import QualityThresholds as QualityThresholdsClass
from .hybrid_retrieval import RetrievalWeights as RetrievalWeightsClass
from .privacy_consent import consent_manager, ConsentType
from .settings import settings
from .error_handler import (
    BaseApplicationError, handle_error, create_context, ValidationError as AppValidationError,
    ResourceError, ServiceUnavailableError, FileProcessingError
)
from .resource_monitor import resource_monitor
from .degradation import degradation_manager
from .monitoring import get_health_status, get_metrics_summary, get_recent_metrics, start_monitoring
from .startup_checks import validate_startup, format_check_results
from .production_config import validate_production_config, get_environment_mode, is_production_environment

logger = get_logger(__name__)

# Initialize the Letta agent handler
agent_handler = LettaAgentHandler()

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


# Custom exception handlers
@app.exception_handler(BaseApplicationError)
async def application_error_handler(request: Request, exc: BaseApplicationError):
    """Handle application-specific errors."""
    logger.error(
        f"Application error in {request.method} {request.url.path}",
        error_code=exc.error_code,
        severity=exc.severity.value,
        user_message=exc.user_message
    )
    
    # Determine HTTP status code based on error type
    if isinstance(exc, AppValidationError):
        status_code = 400
    elif isinstance(exc, ResourceError):
        status_code = 507  # Insufficient Storage
    elif isinstance(exc, ServiceUnavailableError):
        status_code = 503
    elif isinstance(exc, FileProcessingError):
        status_code = 422  # Unprocessable Entity
    else:
        status_code = 500
    
    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict()
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI validation errors."""
    logger.warning(
        f"Validation error in {request.method} {request.url.path}",
        errors=str(exc.errors())
    )
    
    # Convert to our error format
    validation_error = AppValidationError(
        field="request",
        value=str(exc),
        constraint="Request validation failed",
        suggestion="Please check your request format and try again"
    )
    
    return JSONResponse(
        status_code=400,
        content=validation_error.to_dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        f"Unexpected error in {request.method} {request.url.path}",
        error=str(exc),
        traceback=traceback.format_exc()
    )
    
    # Convert to application error
    app_error = handle_error(exc, create_context(
        operation=f"{request.method} {request.url.path}"
    ), notify_user=False)
    
    return JSONResponse(
        status_code=500,
        content=app_error.to_dict()
    )


# Middleware for request validation and error context
@app.middleware("http")
async def error_context_middleware(request: Request, call_next):
    """Middleware to add error context and handle request validation."""
    start_time = time.time()
    
    try:
        # Add request context for error handling
        request.state.error_context = create_context(
            operation=f"{request.method} {request.url.path}"
        )
        
        response = await call_next(request)
        
        # Log successful requests
        process_time = time.time() - start_time
        logger.debug(
            f"Request completed",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            process_time=f"{process_time:.3f}s"
        )
        
        return response
        
    except Exception as exc:
        # Let exception handlers deal with the error
        raise exc


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    logger.info("Starting Letta Claim Assistant API", environment=get_environment_mode().value)
    
    # Start job queue
    await ensure_job_queue_started()
    
    # Start monitoring if in production
    if is_production_environment():
        await start_monitoring(interval_seconds=30)
        logger.info("Production monitoring started")


# Health and monitoring endpoints
@app.get("/api/health")
async def health_check():
    """Get application health status."""
    try:
        # Get comprehensive health status from new monitoring system
        health_status = await get_health_status()
        
        # Add legacy system status for backwards compatibility
        try:
            system_status = await resource_monitor.get_comprehensive_status()
            degradation_status = degradation_manager.get_degradation_status()
            
            from .error_handler import error_handler
            error_stats = error_handler.get_error_stats()
            
            # Combine new and legacy status
            response = health_status.to_dict()
            response.update({
                "legacy_system": system_status,
                "degradation": degradation_status,
                "errors": error_stats,
                "environment": get_environment_mode().value
            })
            
            return response
            
        except Exception as legacy_error:
            # If legacy monitoring fails, return new monitoring data only
            logger.warning("Legacy monitoring failed", error=str(legacy_error))
            response = health_status.to_dict()
            response["environment"] = get_environment_mode().value
            return response
            
    except Exception as e:
        # Health check failure
        logger.error("Health check completely failed", error=str(e))
        
        app_error = handle_error(e, create_context(operation="health_check"), notify_user=False)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": app_error.to_dict(),
                "timestamp": time.time(),
                "environment": get_environment_mode().value
            }
        )


@app.get("/api/health/detailed")
async def detailed_health_check():
    """Get detailed health and system information."""
    try:
        # Run startup validation checks
        startup_success, startup_results = await validate_startup()
        
        # Run production config validation
        config_success, config_results = validate_production_config()
        
        # Get health status
        health_status = await get_health_status()
        
        # Get metrics summary
        metrics_summary = get_metrics_summary()
        
        return {
            "status": "healthy" if startup_success and config_success and health_status.status == "healthy" else "unhealthy",
            "timestamp": time.time(),
            "environment": get_environment_mode().value,
            "startup_validation": {
                "success": startup_success,
                "results": [r.__dict__ for r in startup_results]
            },
            "config_validation": {
                "success": config_success,
                "results": [r.__dict__ for r in config_results]
            },
            "health": health_status.to_dict(),
            "metrics": metrics_summary
        }
        
    except Exception as e:
        app_error = handle_error(e, create_context(operation="detailed_health_check"), notify_user=False)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": app_error.to_dict(),
                "timestamp": time.time()
            }
        )


@app.get("/api/metrics")
async def get_application_metrics():
    """Get application performance metrics."""
    try:
        return {
            "summary": get_metrics_summary(),
            "recent": get_recent_metrics(minutes=5),
            "timestamp": time.time()
        }
    except Exception as e:
        app_error = handle_error(e, create_context(operation="get_metrics"), notify_user=False)
        return JSONResponse(
            status_code=500,
            content=app_error.to_dict()
        )


@app.get("/api/metrics/recent")
async def get_recent_application_metrics(minutes: int = 5):
    """Get recent application metrics."""
    if minutes < 1 or minutes > 60:
        raise HTTPException(status_code=400, detail="Minutes must be between 1 and 60")
    
    try:
        return {
            "metrics": get_recent_metrics(minutes=minutes),
            "time_window_minutes": minutes,
            "timestamp": time.time()
        }
    except Exception as e:
        app_error = handle_error(e, create_context(operation="get_recent_metrics"), notify_user=False)
        return JSONResponse(
            status_code=500,
            content=app_error.to_dict()
        )


@app.get("/api/system/validation")
async def system_validation():
    """Run system validation checks."""
    try:
        # Run startup validation
        startup_success, startup_results = await validate_startup()
        
        # Run production config validation  
        config_success, config_results = validate_production_config()
        
        # Format results for display
        startup_formatted = format_check_results(startup_results)
        
        overall_success = startup_success and config_success
        
        return {
            "success": overall_success,
            "timestamp": time.time(),
            "startup_validation": {
                "success": startup_success,
                "results": [r.__dict__ for r in startup_results],
                "formatted": startup_formatted
            },
            "config_validation": {
                "success": config_success,
                "results": [r.__dict__ for r in config_results]
            }
        }
        
    except Exception as e:
        app_error = handle_error(e, create_context(operation="system_validation"), notify_user=False)
        return JSONResponse(
            status_code=500,
            content=app_error.to_dict()
        )


@app.get("/api/system/status")
async def system_status():
    """Get detailed system status."""
    try:
        status = await resource_monitor.get_comprehensive_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system/degradation")
async def degradation_status():
    """Get service degradation status."""
    try:
        status = degradation_manager.get_degradation_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/system/recovery/{service_name}")
async def attempt_service_recovery(service_name: str):
    """Attempt to recover a degraded service."""
    try:
        recovery_result = await degradation_manager.attempt_service_recovery(service_name)
        return {
            "service": service_name,
            "recovery_attempted": True,
            "recovery_successful": recovery_result
        }
    except Exception as e:
        app_error = handle_error(e, create_context(
            operation="service_recovery",
            provider=service_name
        ), notify_user=False)
        raise HTTPException(status_code=500, detail=app_error.to_dict())
    
    # Initialize default Ollama provider
    try:
        success = await provider_manager.register_ollama_provider()
        if success:
            logger.info("Default Ollama provider registered successfully")
        else:
            logger.warning("Failed to register default Ollama provider - models may not be available")
    except Exception as e:
        logger.warning("Could not initialize Ollama provider on startup", error=str(e))
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown."""
    logger.info("Shutting down Letta Claim Assistant API")


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


# Pydantic models are now imported from models.py

class ModelSettings(BaseModel):
    provider: str
    generation_model: str
    embedding_model: str
    api_key: Optional[str] = None

class ConsentRequest(BaseModel):
    provider: str
    consent_granted: bool
    consent_version: str = "1.0"

class ProviderTestRequest(BaseModel):
    provider_type: str  # "ollama" or "gemini"
    generation_model: str
    embedding_model: Optional[str] = None
    api_key: Optional[str] = None
    test_only: bool = False

class ProviderSwitchRequest(BaseModel):
    provider_key: str


# Matter Management Endpoints
@app.post("/api/matters", response_model=CreateMatterResponse)
async def create_matter(request: CreateMatterRequest):
    """Create a new Matter with filesystem structure and provider configuration."""
    try:
        # Check if provider configuration is provided
        if request.provider and request.generation_model:
            # Create matter with provider configuration
            matter = matter_manager.create_matter_with_provider(
                name=request.name,
                provider=request.provider,
                generation_model=request.generation_model,
                embedding_model=request.embedding_model
            )
            
            # If API key provided, store it securely (for external providers)
            if request.api_key and request.provider in ['gemini', 'openai']:
                # Store API key in provider configuration
                from .provider_management import provider_manager
                await provider_manager.store_api_key(request.provider, request.api_key)
            
            # Create Letta agent with the specified provider configuration
            if matter:
                try:
                    from .letta_agent import agent_handler
                    from .letta_adapter import LettaAdapter
                    from .letta_provider_bridge import LettaProviderBridge
                    
                    # Create adapter for the matter
                    adapter = LettaAdapter(
                        matter_path=matter.paths.root,
                        matter_name=matter.name,
                        matter_id=matter.id
                    )
                    
                    # Configure provider for the agent
                    bridge = LettaProviderBridge()
                    if request.provider == 'ollama':
                        provider_config = bridge.get_ollama_config(
                            model=request.generation_model,
                            embedding_model=request.embedding_model
                        )
                    elif request.provider == 'gemini':
                        provider_config = bridge.get_gemini_config(
                            api_key=request.api_key,
                            model=request.generation_model
                        )
                    elif request.provider == 'openai':
                        provider_config = bridge.get_openai_config(
                            api_key=request.api_key,
                            model=request.generation_model,
                            embedding_model=request.embedding_model
                        )
                    else:
                        provider_config = None
                    
                    # Set provider configuration for the adapter
                    if provider_config:
                        adapter.provider_config = provider_config
                    
                    # Initialize the adapter (creates agent with provider config)
                    await adapter._ensure_initialized()
                    
                    # Store agent ID in matter
                    if adapter.agent_id:
                        matter.agent_id = adapter.agent_id
                        matter_manager.update_matter(matter)
                    
                    logger.info(
                        "Created matter with provider configuration",
                        matter_id=matter.id,
                        provider=request.provider,
                        model=request.generation_model,
                        agent_id=adapter.agent_id
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to create Letta agent for matter: {e}")
                    # Continue - matter is created even if agent creation fails
        else:
            # Legacy: Create matter without provider configuration
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


@app.delete("/api/matters/{matter_id}")
async def delete_matter(matter_id: str):
    """
    Delete a matter and all associated data.
    
    This will permanently delete:
    - All documents and OCR files
    - Vector store and embeddings
    - Letta agent and memory
    - Chat history
    - Matter configuration
    """
    try:
        # Check if matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Check if it's the currently active matter
        active_matter = matter_manager.get_active_matter()
        if active_matter and active_matter.id == matter_id:
            # Switch to another matter or clear active matter
            matters = matter_manager.list_matters()
            other_matters = [m for m in matters if m.id != matter_id]
            if other_matters:
                # Switch to the first available matter
                matter_manager.switch_matter(other_matters[0].id)
            else:
                # No other matters, clear active matter
                matter_manager._current_matter = None
        
        # Delete the matter
        success = await matter_manager.delete_matter(matter_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete matter")
        
        logger.info("Matter deleted via API", matter_id=matter_id)
        
        return {
            "status": "success",
            "message": f"Matter {matter_id} deleted successfully",
            "deleted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete matter", matter_id=matter_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Upload & Ingestion Endpoints
@app.post("/api/matters/{matter_id}/upload")
async def upload_files(matter_id: str, files: List[UploadFile] = File(...)):
    """Upload and process PDF files for a Matter."""
    import tempfile
    import shutil
    from pathlib import Path
    
    try:
        # Ensure job queue is started
        await ensure_job_queue_started()
        
        # 1. Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # 2. Validate and save uploaded files
        uploaded_files = []
        temp_files = []
        
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only PDF files are supported. Got: {file.filename}"
                )
            
            # Check file size (limit to 100MB per file)
            if hasattr(file, 'size') and file.size > 100 * 1024 * 1024:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File too large: {file.filename}. Maximum size is 100MB."
                )
            
            # Create temporary file to save upload
            temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            temp_path = Path(temp_file.name)
            temp_files.append(temp_path)
            
            try:
                # Copy uploaded file content to temp file
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                temp_file.close()
                
                # Validate it's a real PDF by checking header
                with open(temp_path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid PDF file: {file.filename}"
                        )
                
                uploaded_files.append({
                    "filename": file.filename,
                    "path": str(temp_path),
                    "size": temp_path.stat().st_size,
                    "content_type": file.content_type
                })
                
                logger.info(
                    "File uploaded temporarily",
                    matter_id=matter_id,
                    filename=file.filename,
                    size_bytes=temp_path.stat().st_size,
                    temp_path=str(temp_path)
                )
                
            except Exception as e:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process file {file.filename}: {str(e)}"
                )
        
        # 3. Submit background job for processing
        job_params = {
            "matter_id": matter_id,
            "uploaded_files": uploaded_files,
            "force_ocr": False,  # Could be made configurable
            "ocr_language": "eng"  # Could be made configurable
        }
        
        job_id = await job_queue.submit_job("pdf_ingestion", job_params)
        
        logger.info(
            "PDF ingestion job submitted",
            matter_id=matter_id,
            job_id=job_id,
            file_count=len(uploaded_files),
            total_size_mb=sum(f["size"] for f in uploaded_files) / 1024 / 1024
        )
        
        # 4. Return job ID and file info
        return {
            "job_id": job_id,
            "files_uploaded": len(uploaded_files),
            "total_size_bytes": sum(f["size"] for f in uploaded_files),
            "files": [{
                "filename": f["filename"],
                "size_bytes": f["size"]
            } for f in uploaded_files]
        }
        
    except HTTPException:
        # Clean up temp files on HTTP errors
        for temp_path in temp_files:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
        raise
    except Exception as e:
        # Clean up temp files on unexpected errors
        for temp_path in temp_files:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
        
        logger.error("Failed to upload files", matter_id=matter_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get enhanced status of a background job."""
    try:
        job_info = await job_queue.get_job_status(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")
        
        response = {
            "id": job_info.job_id,
            "type": job_info.job_type,
            "status": job_info.status.value,
            "progress": job_info.progress,
            "message": job_info.message,
            "created_at": job_info.created_at.isoformat() if job_info.created_at else None,
            "started_at": job_info.started_at.isoformat() if job_info.started_at else None,
            "completed_at": job_info.completed_at.isoformat() if job_info.completed_at else None,
            "error_message": job_info.error_message,
            "result": job_info.result,
            "retry_count": getattr(job_info, "retry_count", 0),
            "max_retries": getattr(job_info, "max_retries", 3),
            "priority": getattr(job_info, "priority", 0),
            "estimated_duration": getattr(job_info, "estimated_duration", None)
        }
        
        # Include ETA if available
        if (hasattr(job_info, 'progress_history') and 
            job_info.progress_history):
            latest_progress = job_info.progress_history[-1]
            if "eta_seconds" in latest_progress:
                response["eta_seconds"] = latest_progress["eta_seconds"]
            if "elapsed_seconds" in latest_progress:
                response["elapsed_seconds"] = latest_progress["elapsed_seconds"]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get job status", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs")
async def get_all_jobs(limit: int = 50, status_filter: str = None):
    """Get list of recent jobs with optional status filtering."""
    try:
        if status_filter:
            # Get from persistent storage with filter
            jobs = await job_queue.get_job_history(limit, status_filter)
        else:
            # Get recent jobs from memory
            jobs = await job_queue.get_all_jobs(limit)
        
        return {
            "jobs": [{
                "id": job.job_id,
                "type": job.job_type,
                "status": job.status.value,
                "progress": job.progress,
                "message": job.message,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "retry_count": getattr(job, "retry_count", 0),
                "priority": getattr(job, "priority", 0)
            } for job in jobs],
            "count": len(jobs)
        }
        
    except Exception as e:
        logger.error("Failed to get jobs list", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running or queued job."""
    try:
        success = await job_queue.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    """Manually retry a failed job."""
    try:
        success = await job_queue.retry_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be retried")
        
        return {"message": f"Job {job_id} queued for retry"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/queue/status")
async def get_queue_status():
    """Get overall job queue status."""
    try:
        running_jobs = await job_queue.get_running_jobs()
        queued_jobs = await job_queue.get_queued_jobs()
        
        return {
            "running_count": len(running_jobs),
            "queued_count": len(queued_jobs),
            "max_concurrent": job_queue.max_concurrent,
            "running_jobs": [{
                "id": job.job_id,
                "type": job.job_type,
                "progress": job.progress,
                "message": job.message,
                "started_at": job.started_at.isoformat() if job.started_at else None
            } for job in running_jobs],
            "queued_jobs": [{
                "id": job.job_id,
                "type": job.job_type,
                "priority": getattr(job, "priority", 0),
                "created_at": job.created_at.isoformat() if job.created_at else None
            } for job in queued_jobs[:10]]  # Show only first 10 queued
        }
        
    except Exception as e:
        logger.error("Failed to get queue status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/cleanup")
async def cleanup_old_jobs(older_than_hours: int = 24):
    """Clean up old completed jobs."""
    try:
        cleaned_count = await job_queue.cleanup_completed_jobs(older_than_hours)
        return {
            "message": f"Cleaned up {cleaned_count} old jobs",
            "cleaned_count": cleaned_count
        }
        
    except Exception as e:
        logger.error("Failed to cleanup jobs", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# New Job Type Submission Endpoints
@app.post("/api/jobs/batch-processing")
async def submit_batch_processing_job(
    file_batches: List[Dict[str, Any]],
    force_ocr: bool = False,
    ocr_language: str = "eng",
    max_concurrent_files: int = 3,
    priority: int = 1
):
    """Submit a batch document processing job."""
    try:
        job_params = {
            "file_batches": file_batches,
            "force_ocr": force_ocr,
            "ocr_language": ocr_language,
            "max_concurrent_files": max_concurrent_files
        }
        
        job_id = await job_queue.submit_job(
            job_type="batch_document_processing",
            params=job_params,
            priority=priority,
            max_retries=2
        )
        
        return {"job_id": job_id}
        
    except Exception as e:
        logger.error("Failed to submit batch processing job", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/large-model-operation")
async def submit_large_model_operation_job(
    operation_type: str,
    matter_id: str = None,
    parameters: Dict[str, Any] = None,
    priority: int = 1
):
    """Submit a large model operation job."""
    try:
        job_params = {
            "operation_type": operation_type,
            "matter_id": matter_id,
            **(parameters or {})
        }
        
        job_id = await job_queue.submit_job(
            job_type="large_model_operation",
            params=job_params,
            priority=priority,
            max_retries=3
        )
        
        return {"job_id": job_id}
        
    except Exception as e:
        logger.error("Failed to submit large model operation job", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/matter-analysis")
async def submit_matter_analysis_job(
    matter_id: str,
    analysis_types: List[str] = ["overview", "timeline", "entities"],
    priority: int = 0
):
    """Submit a matter analysis job."""
    try:
        job_params = {
            "matter_id": matter_id,
            "analysis_types": analysis_types
        }
        
        job_id = await job_queue.submit_job(
            job_type="matter_analysis",
            params=job_params,
            priority=priority,
            max_retries=2
        )
        
        return {"job_id": job_id}
        
    except Exception as e:
        logger.error("Failed to submit matter analysis job", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Chat / RAG Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat query using stateful Letta agent."""
    try:
        logger.info(
            "Processing chat request via agent",
            matter_id=request.matter_id,
            query_preview=request.query[:100],
            k=request.k,
            max_tokens=request.max_tokens
        )
        
        # 1. Validate matter exists and get matter object
        matter = matter_manager.get_matter_by_id(request.matter_id)
        if not matter:
            raise HTTPException(
                status_code=404, 
                detail=f"Matter with ID {request.matter_id} not found"
            )
        
        # 2. Send message to the agent (agent decides when to use tools)
        agent_response = await agent_handler.send_message(
            matter_id=request.matter_id,
            message=request.query,
            k=request.k,  # Pass k for tool context
            max_tokens=request.max_tokens
        )
        
        # 3. Convert agent response to ChatResponse format
        # Extract sources from search results if search was performed
        sources = []
        if agent_response.search_performed and agent_response.search_results:
            for result in agent_response.search_results:
                sources.append({
                    "doc": result.get("doc_name", ""),
                    "page_start": result.get("page_start", 1),
                    "page_end": result.get("page_end", 1),
                    "text": result.get("snippet", ""),
                    "score": result.get("score", 0.0)
                })
        
        # 4. Create the response with tool usage information
        rag_response = ChatResponse(
            answer=agent_response.message,
            sources=sources,
            followups=[],  # Agent doesn't generate followups yet
            used_memory=[],  # Memory items handled internally by agent
            tools_used=agent_response.tools_used,  # Include tools used
            search_performed=agent_response.search_performed,  # Include search status
            processing_time=agent_response.response_time,
            confidence_score=0.9 if agent_response.search_performed else 0.8
        )
        
        # 5. Save interaction to chat history
        try:
            from .chat_history import get_chat_history_manager
            chat_manager = get_chat_history_manager(matter)
            await chat_manager.save_interaction(
                user_query=request.query,
                assistant_response=rag_response.answer,
                sources=sources,
                followups=[],
                used_memory=[],
                query_metadata={
                    "k": request.k,
                    "max_tokens": request.max_tokens,
                    "model": request.model,
                    "tools_used": agent_response.tools_used,
                    "search_performed": agent_response.search_performed
                }
            )
        except Exception as e:
            logger.warning("Failed to save chat history", error=str(e))
            # Don't fail the request if history saving fails
        
        logger.info(
            "Chat request completed via agent",
            matter_id=request.matter_id,
            answer_length=len(agent_response.message),
            sources_count=len(sources),
            tools_used=agent_response.tools_used,
            search_performed=agent_response.search_performed
        )
        
        return rag_response
        
    except HTTPException:
        # Re-raise HTTP exceptions (404, 503, etc.)
        raise
    except ValueError as e:
        logger.error("Invalid chat request", error=str(e), matter_id=request.matter_id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to process chat request", 
            error=str(e), 
            matter_id=request.matter_id,
            query_preview=request.query[:100] if request.query else "None"
        )
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# Settings / Models Endpoints
@app.get("/api/settings/models")
async def get_model_settings():
    """Get current model and provider settings."""
    try:
        logger.debug("Getting model settings")
        
        # Get provider configuration
        provider_config = provider_manager.get_provider_config()
        
        # Get list of all providers with details
        all_providers = provider_manager.list_providers()
        
        # Test connectivity of active providers
        test_results = {}
        if provider_config.get("active_provider"):
            test_results["generation"] = await provider_manager.test_provider(
                provider_config["active_provider"]
            )
        
        if provider_config.get("active_embedding_provider"):
            test_results["embedding"] = await provider_manager.test_provider(
                provider_config["active_embedding_provider"]
            )
        
        response = {
            "active_provider": provider_config.get("active_provider"),
            "active_embedding_provider": provider_config.get("active_embedding_provider"),
            "providers": all_providers,
            "connectivity": test_results,
            "provider_count": len(all_providers)
        }
        
        logger.debug(
            "Model settings retrieved",
            active_provider=provider_config.get("active_provider"),
            provider_count=len(all_providers)
        )
        
        return response
        
    except Exception as e:
        logger.error("Failed to get model settings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class ProviderTestRequest(BaseModel):
    """Request model for provider testing and configuration."""
    provider_type: str  # "ollama" or "gemini"
    generation_model: Optional[str] = None
    embedding_model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    test_only: bool = False  # If True, only test connection without registering


@app.post("/api/settings/models")
async def update_model_settings(request: ProviderTestRequest):
    """Update model and provider settings."""
    try:
        logger.info(
            "Updating model settings",
            provider_type=request.provider_type,
            generation_model=request.generation_model,
            test_only=request.test_only
        )
        
        success = False
        error_message = None
        
        if request.provider_type.lower() == "ollama":
            # Configure Ollama provider
            generation_model = request.generation_model or "gpt-oss:20b"
            embedding_model = request.embedding_model or "nomic-embed-text"
            base_url = request.base_url or "http://localhost:11434"
            
            success = await provider_manager.register_ollama_provider(
                model=generation_model,
                embedding_model=embedding_model,
                base_url=base_url
            )
            
            if not success:
                error_message = f"Failed to connect to Ollama at {base_url} with model {generation_model}"
        
        elif request.provider_type.lower() == "gemini":
            # Configure Gemini provider
            if not request.api_key:
                raise HTTPException(status_code=400, detail="API key required for Gemini provider")
            
            generation_model = request.generation_model or "gemini-2.0-flash-exp"
            
            success = await provider_manager.register_gemini_provider(
                api_key=request.api_key,
                model=generation_model
            )
            
            if not success:
                error_message = f"Failed to connect to Gemini API with model {generation_model}"
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported provider type: {request.provider_type}"
            )
        
        # Perform a quick test if requested
        test_result = None
        if success and not request.test_only:
            test_result = await provider_manager.quick_test_generation()
        
        response = {
            "success": success,
            "provider_type": request.provider_type,
            "generation_model": request.generation_model,
            "embedding_model": request.embedding_model,
            "test_result": test_result,
            "error": error_message
        }
        
        if success:
            logger.info(
                "Model settings updated successfully",
                provider_type=request.provider_type,
                generation_model=request.generation_model
            )
        else:
            logger.warning(
                "Model settings update failed",
                provider_type=request.provider_type,
                error=error_message
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update model settings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/settings/test-provider")
async def test_provider_connectivity(request: ProviderTestRequest):
    """Test provider connectivity without registering."""
    request.test_only = True
    return await update_model_settings(request)


# Privacy Consent Endpoints
@app.get("/api/consent/{provider}")
async def get_consent_status(provider: str):
    """Get consent status and requirements for a provider."""
    try:
        consent_requirements = consent_manager.get_consent_requirements(provider)
        return {
            "success": True,
            **consent_requirements
        }
    except Exception as e:
        logger.error("Failed to get consent status", provider=provider, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/consent")
async def update_consent(request: ConsentRequest):
    """Update consent for a provider."""
    try:
        if request.consent_granted:
            success = consent_manager.grant_consent(
                ConsentType.EXTERNAL_LLM, 
                request.provider,
                request.consent_version
            )
            action = "granted"
        else:
            success = consent_manager.deny_consent(
                ConsentType.EXTERNAL_LLM,
                request.provider
            )
            action = "denied"
        
        if success:
            logger.info(
                f"Consent {action}",
                provider=request.provider,
                consent_version=request.consent_version
            )
            return {
                "success": True,
                "message": f"Consent {action} for {request.provider}",
                "provider": request.provider,
                "consent_granted": request.consent_granted
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to {action.lower()} consent for {request.provider}"
            )
            
    except Exception as e:
        logger.error("Failed to update consent", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/consent/{provider}")
async def revoke_consent(provider: str):
    """Revoke consent for a provider."""
    try:
        success = consent_manager.revoke_consent(ConsentType.EXTERNAL_LLM, provider)
        
        if success:
            # Also remove stored credentials for this provider
            settings.delete_credential(provider, "api_key")
            
            logger.info("Consent revoked", provider=provider)
            return {
                "success": True,
                "message": f"Consent revoked for {provider}",
                "provider": provider
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to revoke consent for {provider}"
            )
            
    except Exception as e:
        logger.error("Failed to revoke consent", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced Provider Management Endpoints
@app.get("/api/providers")
async def get_all_providers():
    """Get all registered providers with metrics and status."""
    try:
        providers = provider_manager.list_providers()
        metrics = provider_manager.get_provider_metrics()
        config = provider_manager.get_provider_config()
        
        # Add consent status for each provider
        for provider_key in providers:
            provider_name = provider_key.split('_')[0]  # Extract base name (e.g., 'gemini' from 'gemini_2.0-flash-exp')
            providers[provider_key]["consent"] = consent_manager.get_consent_requirements(provider_name)
        
        return {
            "success": True,
            "providers": providers,
            "metrics": metrics,
            "config": config
        }
        
    except Exception as e:
        logger.error("Failed to get providers", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/providers/register")
async def register_provider(request: ProviderTestRequest):
    """Register a new provider with consent checking."""
    try:
        if request.provider_type.lower() == "gemini":
            # Store API key securely if provided
            if request.api_key:
                settings.store_credential("gemini", "api_key", request.api_key)
            
            # Get stored API key if not provided
            api_key = request.api_key or settings.get_credential("gemini", "api_key")
            
            if not api_key:
                return {
                    "success": False,
                    "error": "api_key_required",
                    "message": "API key required for Gemini provider"
                }
            
            # Register Gemini provider with consent checking
            result = await provider_manager.register_gemini_provider(
                api_key=api_key,
                model=request.generation_model,
                check_consent=True
            )
            
            return result
            
        elif request.provider_type.lower() == "ollama":
            success = await provider_manager.register_ollama_provider(
                model=request.generation_model,
                embedding_model=request.embedding_model or "nomic-embed-text"
            )
            
            if success:
                return {
                    "success": True,
                    "message": "Ollama provider registered successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "registration_failed",
                    "message": "Failed to register Ollama provider"
                }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider type: {request.provider_type}"
            )
            
    except Exception as e:
        logger.error("Failed to register provider", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/providers/switch")
async def switch_active_provider(request: ProviderSwitchRequest):
    """Switch the active provider."""
    try:
        success = provider_manager.switch_provider(request.provider_key)
        
        if success:
            return {
                "success": True,
                "message": f"Switched to provider: {request.provider_key}",
                "active_provider": request.provider_key
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Provider not found: {request.provider_key}"
            )
            
    except Exception as e:
        logger.error("Failed to switch provider", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/providers/{provider_key}/metrics")
async def get_provider_metrics(provider_key: str):
    """Get detailed metrics for a specific provider."""
    try:
        metrics = provider_manager.get_provider_metrics(provider_key)
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics found for provider: {provider_key}"
            )
        
        return {
            "success": True,
            "provider_key": provider_key,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error("Failed to get provider metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/providers/test-all")
async def test_all_providers():
    """Test connectivity for all registered providers."""
    try:
        test_results = await provider_manager.test_all_providers()
        
        return {
            "success": True,
            "test_results": test_results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Failed to test all providers", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Document Management Endpoints
@app.get("/api/matters/{matter_id}/documents")
async def get_matter_documents(matter_id: str):
    """Get list of documents for a matter."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get document information
        documents = matter_manager.get_matter_documents(matter)
        
        logger.info("Retrieved documents for matter", matter_id=matter_id, document_count=len(documents))
        
        return {"documents": documents}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get matter documents", error=str(e), matter_id=matter_id)
        raise HTTPException(status_code=500, detail=str(e))


# Chat History Endpoints
@app.get("/api/matters/{matter_id}/chat/history")
async def get_chat_history(matter_id: str, limit: Optional[int] = 50):
    """Get chat history for a matter."""
    try:
        from .chat_history import get_chat_history_manager
        
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get chat history
        chat_manager = get_chat_history_manager(matter)
        messages = await chat_manager.load_history(limit=limit)
        
        # Convert to API format
        api_messages = []
        for message in messages:
            api_message = {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "sources": message.sources,
                "followups": message.followups,
                "used_memory": message.used_memory
            }
            api_messages.append(api_message)
        
        logger.info("Retrieved chat history", matter_id=matter_id, message_count=len(api_messages))
        
        return {"messages": api_messages}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get chat history", error=str(e), matter_id=matter_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/matters/{matter_id}/chat/history")
async def clear_chat_history(matter_id: str):
    """Clear chat history for a matter."""
    try:
        from .chat_history import get_chat_history_manager
        
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Clear chat history
        chat_manager = get_chat_history_manager(matter)
        await chat_manager.clear_history()
        
        logger.info("Cleared chat history", matter_id=matter_id)
        
        return {"message": "Chat history cleared successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to clear chat history", error=str(e), matter_id=matter_id)
        raise HTTPException(status_code=500, detail=str(e))


# Quality and Advanced Features Endpoints

@app.get("/api/matters/{matter_id}/quality/insights", response_model=QualityInsights)
async def get_quality_insights(matter_id: str):
    """Get quality insights and statistics for a matter."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get active provider
        active_provider = provider_manager.get_active_provider()
        if not active_provider:
            raise HTTPException(
                status_code=503,
                detail="No LLM provider available. Please configure a provider in settings."
            )
        
        # Initialize RAG engine with advanced features
        rag_engine = RAGEngine(
            matter=matter,
            llm_provider=active_provider,
            enable_advanced_features=True
        )
        
        # Get insights
        insights_data = rag_engine.get_quality_insights(matter_id)
        
        return QualityInsights(**insights_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quality insights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get quality insights: {str(e)}")


@app.get("/api/matters/{matter_id}/advanced-features/status")
async def get_advanced_features_status(matter_id: str):
    """Get status of advanced RAG features for a matter."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get active provider
        active_provider = provider_manager.get_active_provider()
        if not active_provider:
            return {
                "advanced_features_enabled": False,
                "error": "No LLM provider available"
            }
        
        # Initialize RAG engine with advanced features
        rag_engine = RAGEngine(
            matter=matter,
            llm_provider=active_provider,
            enable_advanced_features=True
        )
        
        # Get feature status
        status = rag_engine.get_advanced_features_status()
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get advanced features status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get advanced features status: {str(e)}")


# Letta Memory and Agent Endpoints
@app.get("/api/matters/{matter_id}/memory/stats")
async def get_memory_stats(matter_id: str):
    """Get Letta agent memory statistics for a matter."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get Letta adapter for this matter
        from .letta_adapter import LettaAdapter
        letta_adapter = LettaAdapter(
            matter_path=matter.paths.root,
            matter_name=matter.name,
            matter_id=matter.id
        )
        
        # Get memory stats
        stats = await letta_adapter.get_memory_stats()
        
        # Add additional context
        stats["matter_name"] = matter.name
        stats["matter_id"] = matter.id
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory stats: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "memory_items": 0,
            "matter_name": matter.name if matter else "Unknown",
            "matter_id": matter_id
        }


@app.get("/api/letta/health")
async def get_letta_health():
    """Get Letta server and connection health status."""
    try:
        from .letta_connection import connection_manager
        from .letta_server import server_manager
        from .letta_provider_health import provider_health_monitor
        
        # Get connection state and metrics
        connection_state = connection_manager.get_state().value
        connection_metrics = connection_manager.get_metrics()
        
        # Get server status
        server_status = server_manager.health_check()
        
        # Get provider health
        provider_status = provider_health_monitor.get_health_summary()
        
        return {
            "status": "healthy" if connection_state == "connected" else "degraded",
            "connection": {
                "state": connection_state,
                "metrics": connection_metrics
            },
            "server": {
                "running": server_status,
                "url": server_manager.get_base_url() if server_status else None
            },
            "providers": provider_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get Letta health: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "connection": {"state": "error"},
            "server": {"running": False},
            "providers": {},
            "timestamp": datetime.now().isoformat()
        }


@app.post("/api/matters/{matter_id}/memory/summary")
async def get_memory_summary(matter_id: str, max_length: Optional[int] = 500):
    """Get a summary of the agent's memory for a matter."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get Letta adapter for this matter
        from .letta_adapter import LettaAdapter
        letta_adapter = LettaAdapter(
            matter_path=matter.paths.root,
            matter_name=matter.name,
            matter_id=matter.id
        )
        
        # Get memory summary
        summary = await letta_adapter.get_memory_summary(max_length=max_length)
        
        return {
            "summary": summary,
            "matter_id": matter_id,
            "matter_name": matter.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory summary: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get memory summary: {str(e)}")


@app.get("/api/matters/{matter_id}/memory/items")
async def get_memory_items(
    matter_id: str,
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    type_filter: Optional[str] = Query(None, description="Filter by memory type (Entity, Event, Issue, Fact, Interaction, Raw)"),
    search_query: Optional[str] = Query(None, description="Search query to filter memories"),
    search_type: Optional[str] = Query("semantic", description="Search type: semantic, keyword, exact, or regex")
):
    """Get individual memory items with pagination and filtering."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get Letta adapter for this matter
        from .letta_adapter import LettaAdapter
        letta_adapter = LettaAdapter(
            matter_path=matter.paths.root,
            matter_name=matter.name,
            matter_id=matter.id
        )
        
        # Get memory items
        items = await letta_adapter.get_memory_items(
            limit=limit,
            offset=offset,
            type_filter=type_filter,
            search_query=search_query,
            search_type=search_type
        )
        
        # For better UX, try to get a rough total count
        # This is not perfect but gives users an idea of how many items exist
        # We fetch a larger batch to estimate the total
        if len(items) == limit:
            # There might be more items
            estimated_total = offset + limit + 1  # At least one more page
        else:
            # This is the last page
            estimated_total = offset + len(items)
        
        return {
            "items": [item.model_dump() for item in items],
            "total": estimated_total,  # Estimated total for pagination
            "count": len(items),  # Actual items returned
            "limit": limit,
            "offset": offset,
            "matter_id": matter_id,
            "matter_name": matter.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory items: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get memory items: {str(e)}")


@app.get("/api/matters/{matter_id}/memory/items/{item_id}")
async def get_memory_item(matter_id: str, item_id: str):
    """Get a specific memory item by ID."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get Letta adapter for this matter
        from .letta_adapter import LettaAdapter
        letta_adapter = LettaAdapter(
            matter_path=matter.paths.root,
            matter_name=matter.name,
            matter_id=matter.id
        )
        
        # Get specific memory item
        item = await letta_adapter.get_memory_item(item_id)
        
        if not item:
            raise HTTPException(status_code=404, detail=f"Memory item not found: {item_id}")
        
        return {
            "item": item.model_dump(),
            "matter_id": matter_id,
            "matter_name": matter.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory item: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get memory item: {str(e)}")


@app.post("/api/matters/{matter_id}/memory/items")
async def create_memory_item(
    matter_id: str,
    request: CreateMemoryItemRequest
):
    """Create a new memory item."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get Letta adapter for this matter
        from .letta_adapter import LettaAdapter
        letta_adapter = LettaAdapter(
            matter_id=matter_id,
            matter_manager=matter_manager,
            model_provider=model_manager.get_provider(matter.generation_model)
        )
        
        # Create the memory item
        item_id = await letta_adapter.create_memory_item(
            text=request.text,
            type=request.type,
            metadata=request.metadata
        )
        
        return MemoryOperationResponse(
            success=True,
            item_id=item_id,
            message=f"Successfully created memory item of type {request.type}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create memory item: {str(e)}", exc_info=True)
        return MemoryOperationResponse(
            success=False,
            item_id=None,
            message="Failed to create memory item",
            error=str(e)
        )


@app.put("/api/matters/{matter_id}/memory/items/{item_id}")
async def update_memory_item(
    matter_id: str,
    item_id: str,
    request: UpdateMemoryItemRequest
):
    """Update an existing memory item."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get Letta adapter for this matter
        from .letta_adapter import LettaAdapter
        letta_adapter = LettaAdapter(
            matter_id=matter_id,
            matter_manager=matter_manager,
            model_provider=model_manager.get_provider(matter.generation_model)
        )
        
        # Update the memory item
        new_id = await letta_adapter.update_memory_item(
            item_id=item_id,
            new_text=request.new_text,
            preserve_type=request.preserve_type
        )
        
        return MemoryOperationResponse(
            success=True,
            item_id=new_id,
            message=f"Successfully updated memory item. New ID: {new_id}"
        )
        
    except ValueError as e:
        # Memory item not found
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update memory item: {str(e)}", exc_info=True)
        return MemoryOperationResponse(
            success=False,
            item_id=item_id,
            message="Failed to update memory item",
            error=str(e)
        )


@app.delete("/api/matters/{matter_id}/memory/items/{item_id}")
async def delete_memory_item(matter_id: str, item_id: str):
    """Delete a memory item."""
    try:
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get Letta adapter for this matter
        from .letta_adapter import LettaAdapter
        letta_adapter = LettaAdapter(
            matter_id=matter_id,
            matter_manager=matter_manager,
            model_provider=model_manager.get_provider(matter.generation_model)
        )
        
        # Delete the memory item
        success = await letta_adapter.delete_memory_item(item_id)
        
        if success:
            return MemoryOperationResponse(
                success=True,
                item_id=item_id,
                message=f"Successfully deleted memory item {item_id}"
            )
        else:
            return MemoryOperationResponse(
                success=False,
                item_id=item_id,
                message="Failed to delete memory item",
                error="Deletion failed"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory item: {str(e)}", exc_info=True)
        return MemoryOperationResponse(
            success=False,
            item_id=item_id,
            message="Failed to delete memory item",
            error=str(e)
        )


@app.post("/api/matters/{matter_id}/memory/command")
async def process_memory_command(matter_id: str, request: MemoryCommandRequest):
    """Process natural language memory management commands."""
    try:
        # Import command parser
        from .memory_commands import parse_memory_command, MemoryCommandParser, MemoryAction
        
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Parse the command
        command = parse_memory_command(request.command)
        
        if not command:
            # Command not recognized, provide suggestion
            suggestion = MemoryCommandParser.suggest_command(request.command)
            return MemoryCommandResponse(
                success=False,
                message="I couldn't understand that as a memory command. Try being more specific.",
                suggestion=suggestion
            )
        
        # Get Letta adapter for this matter
        from .letta_adapter import LettaAdapter
        letta_adapter = LettaAdapter(
            matter_path=matter.paths.root,
            matter_name=matter.name,
            matter_id=matter.id
        )
        
        # Execute command based on action type
        if command.action == MemoryAction.REMEMBER:
            # Create new memory item
            from .models import KnowledgeItem
            
            # Create a Fact type knowledge item
            knowledge_item = KnowledgeItem(
                type="Fact",
                label=command.content[:100],  # First 100 chars as label
                support_snippet=command.content
            )
            
            item_id = await letta_adapter.create_memory_item(
                text=knowledge_item.model_dump_json(),
                type="Fact"
            )
            
            # Generate undo token (simple approach - could be enhanced)
            import uuid
            undo_token = str(uuid.uuid4())
            
            # Store undo info (in production, this would be cached)
            # For now, we'll just return the token
            
            return MemoryCommandResponse(
                success=True,
                action=command.action.value,
                content=command.content,
                confidence=command.confidence,
                message=MemoryCommandParser.format_confirmation(command),
                item_id=item_id,
                undo_token=undo_token
            )
        
        elif command.action == MemoryAction.FORGET:
            # Search for and delete matching memories
            success = await letta_adapter.search_and_delete_memory(command.content)
            
            if success:
                return MemoryCommandResponse(
                    success=True,
                    action=command.action.value,
                    content=command.content,
                    confidence=command.confidence,
                    message=MemoryCommandParser.format_confirmation(command)
                )
            else:
                return MemoryCommandResponse(
                    success=False,
                    action=command.action.value,
                    content=command.content,
                    confidence=command.confidence,
                    message=f"I couldn't find any memories matching: {command.content}"
                )
        
        elif command.action == MemoryAction.UPDATE:
            # Update existing memory
            success = await letta_adapter.search_and_update_memory(
                target=command.target,
                new_content=command.content
            )
            
            if success:
                return MemoryCommandResponse(
                    success=True,
                    action=command.action.value,
                    content=command.content,
                    confidence=command.confidence,
                    message=MemoryCommandParser.format_confirmation(command)
                )
            else:
                return MemoryCommandResponse(
                    success=False,
                    action=command.action.value,
                    content=command.content,
                    confidence=command.confidence,
                    message=f"I couldn't find memories about '{command.target}' to update"
                )
        
        elif command.action == MemoryAction.QUERY:
            # Query memory (this would typically be handled by chat, but we can provide a direct response)
            memories = await letta_adapter.search_memories(command.content, limit=5)
            
            if memories:
                memory_summary = "\n".join([f"• {m.text[:200]}" for m in memories[:3]])
                return MemoryCommandResponse(
                    success=True,
                    action=command.action.value,
                    content=command.content,
                    confidence=command.confidence,
                    message=f"Here's what I remember about {command.content}:\n{memory_summary}"
                )
            else:
                return MemoryCommandResponse(
                    success=True,
                    action=command.action.value,
                    content=command.content,
                    confidence=command.confidence,
                    message=f"I don't have any specific memories about {command.content}"
                )
        
        else:
            return MemoryCommandResponse(
                success=False,
                message="Unknown command action"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process memory command: {str(e)}", exc_info=True)
        return MemoryCommandResponse(
            success=False,
            message=f"Failed to process memory command: {str(e)}"
        )


@app.get("/api/matters/{matter_id}/memory/analytics")
async def get_memory_analytics(matter_id: str):
    """Get comprehensive memory analytics for a matter."""
    try:
        # Import models
        from .models import MemoryAnalytics, MemoryPattern, MemoryInsight
        
        # Validate matter exists
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter not found: {matter_id}")
        
        # Get Letta adapter for this matter
        from .letta_adapter import LettaAdapter
        letta_adapter = LettaAdapter(
            matter_path=matter.paths.root,
            matter_name=matter.name,
            matter_id=matter_id
        )
        
        # Get analytics data from existing analyze_memory_patterns method
        analytics_data = await letta_adapter.analyze_memory_patterns()
        
        # Check for errors
        if "error" in analytics_data:
            return MemoryAnalytics(
                total_memories=0,
                error=analytics_data["error"]
            )
        
        # Get quality metrics if available
        quality_metrics = None
        try:
            quality_metrics = await letta_adapter.get_memory_quality_metrics()
        except:
            pass  # Quality metrics are optional
        
        # Convert patterns to model format
        patterns = []
        for pattern in analytics_data.get("patterns", []):
            pattern_model = MemoryPattern(
                type=pattern.get("type"),
                value=pattern.get("value"),
                count=pattern.get("count"),
                percentage=pattern.get("percentage"),
                actors=pattern.get("actors"),
                documents=pattern.get("documents"),
                month=pattern.get("month")
            )
            patterns.append(pattern_model)
        
        # Convert insights to model format
        insights = []
        for insight in analytics_data.get("insights", []):
            insight_model = MemoryInsight(
                insight=insight.get("insight"),
                score=insight.get("score"),
                rate=insight.get("rate"),
                avg_connections=insight.get("avg_connections"),
                interpretation=insight.get("interpretation")
            )
            insights.append(insight_model)
        
        # Build growth timeline from temporal distribution
        growth_timeline = []
        temporal_dist = analytics_data.get("temporal_distribution", {})
        if temporal_dist:
            for month, count in sorted(temporal_dist.items()):
                growth_timeline.append({
                    "month": month,
                    "count": count
                })
        
        # Return analytics response
        return MemoryAnalytics(
            total_memories=analytics_data.get("total_memories", 0),
            patterns=patterns,
            insights=insights,
            type_distribution=analytics_data.get("type_distribution", {}),
            actor_network=analytics_data.get("actor_network", {}),
            temporal_distribution=temporal_dist,
            quality_metrics=quality_metrics,
            growth_timeline=growth_timeline if growth_timeline else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory analytics: {str(e)}", exc_info=True)
        return MemoryAnalytics(
            total_memories=0,
            error=f"Failed to retrieve analytics: {str(e)}"
        )


@app.get("/api/matters/{matter_id}/memory/export")
async def export_memory(
    matter_id: str,
    format: str = Query("json", description="Export format (json or csv)"),
    include_metadata: bool = Query(True, description="Include metadata in export")
):
    """
    Export memory to JSON or CSV format.
    
    Args:
        matter_id: The matter ID
        format: Export format (json or csv)
        include_metadata: Whether to include metadata
        
    Returns:
        File download response
    """
    try:
        matter = get_matter(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter {matter_id} not found")
        
        # Get Letta adapter
        letta_adapter = get_letta_adapter(matter)
        if not letta_adapter:
            raise HTTPException(status_code=503, detail="Memory service not available")
        
        # Export memory
        export_data = await letta_adapter.export_memory(
            format=format,
            include_metadata=include_metadata
        )
        
        # Check for errors
        if isinstance(export_data, dict) and "error" in export_data:
            raise HTTPException(status_code=500, detail=export_data["error"])
        
        # Prepare response based on format
        if format == "json":
            # Convert dict to JSON string
            content = json.dumps(export_data, indent=2, default=str)
            media_type = "application/json"
            filename = f"{matter.slug}_memory_export.json"
        elif format == "csv":
            # Content is already a CSV string
            content = export_data
            media_type = "text/csv"
            filename = f"{matter.slug}_memory_export.csv"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        # Return as streaming response with download headers
        return StreamingResponse(
            io.BytesIO(content.encode('utf-8')),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export memory: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export memory: {str(e)}")


@app.post("/api/matters/{matter_id}/memory/import")
async def import_memory(
    matter_id: str,
    file: UploadFile = File(...),
    format: str = Query("json", description="Import format (json or csv)"),
    deduplicate: bool = Query(True, description="Check for duplicates")
):
    """
    Import memory from uploaded file.
    
    Args:
        matter_id: The matter ID
        file: Uploaded file containing memory data
        format: Import format (json or csv)
        deduplicate: Whether to check for duplicates
        
    Returns:
        Import statistics
    """
    try:
        matter = get_matter(matter_id)
        if not matter:
            raise HTTPException(status_code=404, detail=f"Matter {matter_id} not found")
        
        # Get Letta adapter
        letta_adapter = get_letta_adapter(matter)
        if not letta_adapter:
            raise HTTPException(status_code=503, detail="Memory service not available")
        
        # Read file content
        content = await file.read()
        
        # Decode content
        try:
            data_str = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
        
        # Parse data based on format
        if format == "json":
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        elif format == "csv":
            # Pass CSV string directly
            data = data_str
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        # Import memory
        result = await letta_adapter.import_memory(
            data=data,
            format=format,
            deduplicate=deduplicate
        )
        
        # Check for errors
        if "error" in result:
            # Still return the partial results if some imports succeeded
            if result.get("imported", 0) > 0:
                logger.warning(f"Partial import success: {result}")
                return result
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        logger.info(
            "Memory import completed",
            matter_id=matter_id,
            filename=file.filename,
            imported=result.get("imported", 0),
            skipped=result.get("skipped", 0)
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import memory: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to import memory: {str(e)}")


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


def create_app() -> FastAPI:
    """Create and return the configured FastAPI application.
    
    This function is used by tests and deployment scripts to get the app instance.
    """
    return app