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
import time

from .models import (
    CreateMatterRequest, CreateMatterResponse, MatterSummary,
    ChatRequest, ChatResponse, JobStatus, DocumentInfo
)
from .logging_conf import get_logger
from .matters import matter_manager
from .jobs import job_queue, ensure_job_queue_started
from .rag import RAGEngine
from .llm.provider_manager import provider_manager

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


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    logger.info("Starting Letta Claim Assistant API")
    await ensure_job_queue_started()
    
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
        logger.info(
            "Processing chat request",
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
        
        # 2. Get active LLM provider
        active_provider = provider_manager.get_active_provider()
        if not active_provider:
            raise HTTPException(
                status_code=503,
                detail="No LLM provider available. Please configure a provider in settings."
            )
        
        # 3. Initialize RAG engine for the matter
        rag_engine = RAGEngine(
            matter=matter,
            llm_provider=active_provider
        )
        
        # 4. Process query through RAG pipeline
        rag_response = await rag_engine.generate_answer(
            query=request.query,
            k=request.k,
            k_memory=6,  # Default memory items to recall
            max_tokens=request.max_tokens,
            temperature=0.2  # Conservative temperature for legal analysis
        )
        
        # 5. Save interaction to chat history
        try:
            from .chat_history import get_chat_history_manager
            chat_manager = get_chat_history_manager(matter)
            await chat_manager.save_interaction(
                user_query=request.query,
                assistant_response=rag_response.answer,
                sources=[source.model_dump() for source in rag_response.sources],
                followups=rag_response.followups,
                used_memory=[item.model_dump() for item in rag_response.used_memory],
                query_metadata={
                    "k": request.k,
                    "max_tokens": request.max_tokens,
                    "model": request.model
                }
            )
        except Exception as e:
            logger.warning("Failed to save chat history", error=str(e))
            # Don't fail the request if history saving fails
        
        # 6. Convert RAG response to API response format
        api_response = ChatResponse(
            answer=rag_response.answer,
            sources=rag_response.sources,
            followups=rag_response.followups,
            used_memory=rag_response.used_memory
        )
        
        logger.info(
            "Chat request completed successfully",
            matter_id=request.matter_id,
            answer_length=len(rag_response.answer),
            sources_count=len(rag_response.sources),
            followups_count=len(rag_response.followups)
        )
        
        return api_response
        
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