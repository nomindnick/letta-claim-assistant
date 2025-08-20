"""
HTTP client for backend API communication.

Provides async methods to communicate with the FastAPI backend
for all application operations.
"""

from typing import List, Dict, Any, Optional
import aiohttp
import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.logging_conf import get_logger

logger = get_logger(__name__)


class APIClient:
    """Async HTTP client for backend API."""
    
    def __init__(self, base_url: str = None):
        if base_url is None:
            # Check if backend port was dynamically assigned
            import os
            port = os.environ.get('BACKEND_PORT', '8000')
            base_url = f"http://localhost:{port}"
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def post(self, endpoint: str, json_data: Dict[str, Any] = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generic POST request method. By default sends JSON data."""
        session = await self._get_session()
        try:
            # Ensure endpoint doesn't start with double slash
            endpoint = endpoint.lstrip('/')
            url = f"{self.base_url}/{endpoint}"
            
            kwargs = {}
            if json_data is not None:
                kwargs['json'] = json_data
            elif data is not None:
                # If data is a dict and not explicitly form data, send as JSON
                kwargs['json'] = data
            
            async with session.post(url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"POST request failed for {endpoint}", error=str(e))
            raise
    
    async def get(self, endpoint: str) -> Dict[str, Any]:
        """Generic GET request method."""
        session = await self._get_session()
        try:
            # Ensure endpoint doesn't start with double slash
            endpoint = endpoint.lstrip('/')
            url = f"{self.base_url}/{endpoint}"
            
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"GET request failed for {endpoint}", error=str(e))
            raise
    
    # Matter Management
    async def create_matter(self, name: str) -> Dict[str, Any]:
        """Create a new Matter."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/api/matters",
                json={"name": name}
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to create matter", error=str(e))
            raise
    
    async def list_matters(self) -> List[Dict[str, Any]]:
        """List all available Matters."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/matters") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to list matters", error=str(e))
            raise
    
    async def switch_matter(self, matter_id: str) -> Dict[str, Any]:
        """Switch active Matter."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/api/matters/{matter_id}/switch"
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to switch matter", error=str(e))
            raise
    
    async def get_active_matter(self) -> Optional[Dict[str, Any]]:
        """Get the currently active Matter."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/matters/active") as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("active_matter")
        except Exception as e:
            logger.error("Failed to get active matter", error=str(e))
            raise
    
    # File Upload
    async def upload_files(
        self,
        matter_id: str,
        files: List[Path]
    ) -> str:
        """
        Upload files to a Matter and return job ID.
        
        Args:
            matter_id: Target Matter ID
            files: List of file paths to upload
            
        Returns:
            Job ID for tracking upload progress
        """
        session = await self._get_session()
        
        # Prepare multipart data
        data = aiohttp.FormData()
        file_handles = []
        
        try:
            for file_path in files:
                file_handle = open(file_path, 'rb')
                file_handles.append(file_handle)
                data.add_field(
                    'files',
                    file_handle,
                    filename=file_path.name,
                    content_type='application/pdf'
                )
            
            async with session.post(
                f"{self.base_url}/api/matters/{matter_id}/upload",
                data=data
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["job_id"]
                
        except Exception as e:
            logger.error("Failed to upload files", error=str(e))
            raise
        finally:
            # Close file handles
            for file_handle in file_handles:
                try:
                    file_handle.close()
                except:
                    pass
    
    # Job Management
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a background job."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/jobs/{job_id}") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to get job status", error=str(e))
            raise
    
    async def get_all_jobs(self, limit: int = 50, status_filter: str = None) -> Dict[str, Any]:
        """Get list of recent jobs with optional status filtering."""
        session = await self._get_session()
        try:
            params = {"limit": limit}
            if status_filter:
                params["status_filter"] = status_filter
                
            async with session.get(f"{self.base_url}/api/jobs", params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to get jobs list", error=str(e))
            raise
    
    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running or queued job."""
        session = await self._get_session()
        try:
            async with session.post(f"{self.base_url}/api/jobs/{job_id}/cancel") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to cancel job", job_id=job_id, error=str(e))
            raise
    
    async def retry_job(self, job_id: str) -> Dict[str, Any]:
        """Manually retry a failed job."""
        session = await self._get_session()
        try:
            async with session.post(f"{self.base_url}/api/jobs/{job_id}/retry") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to retry job", job_id=job_id, error=str(e))
            raise
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get overall job queue status."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/jobs/queue/status") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to get queue status", error=str(e))
            raise
    
    async def cleanup_old_jobs(self, older_than_hours: int = 24) -> Dict[str, Any]:
        """Clean up old completed jobs."""
        session = await self._get_session()
        try:
            async with session.post(f"{self.base_url}/api/jobs/cleanup", 
                                  params={"older_than_hours": older_than_hours}) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to cleanup jobs", error=str(e))
            raise
    
    async def submit_batch_processing_job(
        self,
        file_batches: List[Dict[str, Any]],
        force_ocr: bool = False,
        ocr_language: str = "eng",
        max_concurrent_files: int = 3,
        priority: int = 1
    ) -> Dict[str, Any]:
        """Submit a batch document processing job."""
        session = await self._get_session()
        try:
            data = {
                "file_batches": file_batches,
                "force_ocr": force_ocr,
                "ocr_language": ocr_language,
                "max_concurrent_files": max_concurrent_files,
                "priority": priority
            }
            
            async with session.post(f"{self.base_url}/api/jobs/batch-processing", 
                                  json=data) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to submit batch processing job", error=str(e))
            raise
    
    async def submit_large_model_operation_job(
        self,
        operation_type: str,
        matter_id: str = None,
        parameters: Dict[str, Any] = None,
        priority: int = 1
    ) -> Dict[str, Any]:
        """Submit a large model operation job."""
        session = await self._get_session()
        try:
            data = {
                "operation_type": operation_type,
                "matter_id": matter_id,
                "parameters": parameters or {},
                "priority": priority
            }
            
            async with session.post(f"{self.base_url}/api/jobs/large-model-operation", 
                                  json=data) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to submit large model operation job", error=str(e))
            raise
    
    async def submit_matter_analysis_job(
        self,
        matter_id: str,
        analysis_types: List[str] = ["overview", "timeline", "entities"],
        priority: int = 0
    ) -> Dict[str, Any]:
        """Submit a matter analysis job."""
        session = await self._get_session()
        try:
            data = {
                "matter_id": matter_id,
                "analysis_types": analysis_types,
                "priority": priority
            }
            
            async with session.post(f"{self.base_url}/api/jobs/matter-analysis", 
                                  json=data) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to submit matter analysis job", error=str(e))
            raise
    
    # Chat / RAG
    async def send_chat_message(
        self,
        matter_id: str,
        query: str,
        k: int = 8,
        mode: str = "combined",  # Add mode parameter with default
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Send chat message and get RAG response.
        
        Args:
            matter_id: ID of the matter to query
            query: User's question
            k: Number of document chunks to retrieve
            mode: Chat mode - "rag" (documents only), "memory" (agent only), or "combined" (both)
            model: Optional model override
            max_tokens: Optional max tokens override
            
        Returns:
            Response with answer, sources, and metadata
        """
        session = await self._get_session()
        try:
            # Build request payload
            payload = {
                "matter_id": matter_id,
                "query": query,
                "k": k,
                "mode": mode  # Include mode in payload
            }
            if model is not None:
                payload["model"] = model
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
                
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to send chat message", error=str(e))
            raise
    
    # Settings
    async def get_model_settings(self) -> Dict[str, Any]:
        """Get current model and provider settings."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/settings/models") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to get model settings", error=str(e))
            raise
    
    async def update_model_settings(
        self,
        provider: str,
        generation_model: str,
        embedding_model: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update model and provider settings."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/api/settings/models",
                json={
                    "provider_type": provider,  # Changed from 'provider' to 'provider_type'
                    "generation_model": generation_model,
                    "embedding_model": embedding_model,
                    "api_key": api_key
                }
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to update model settings", error=str(e))
            raise
    
    # Document Management
    async def get_matter_documents(self, matter_id: str) -> List[Dict[str, Any]]:
        """Get list of documents for a matter."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/matters/{matter_id}/documents") as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("documents", [])
        except Exception as e:
            logger.error("Failed to get matter documents", error=str(e))
            raise
    
    # Chat History Management
    async def get_chat_history(
        self,
        matter_id: str,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Get chat history for a matter."""
        session = await self._get_session()
        try:
            params = {}
            if limit:
                params["limit"] = limit
                
            async with session.get(
                f"{self.base_url}/api/matters/{matter_id}/chat/history",
                params=params
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("messages", [])
        except Exception as e:
            logger.error("Failed to get chat history", error=str(e))
            raise
    
    async def clear_chat_history(self, matter_id: str) -> bool:
        """Clear chat history for a matter."""
        session = await self._get_session()
        try:
            async with session.delete(f"{self.base_url}/api/matters/{matter_id}/chat/history") as response:
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
            raise
    
    # Memory Management
    async def get_memory_summary(self, matter_id: str, max_length: int = 500) -> Dict[str, Any]:
        """Get memory summary for a matter."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/api/matters/{matter_id}/memory/summary",
                params={"max_length": max_length}
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to get memory summary", error=str(e))
            raise
    
    async def get_memory_items(
        self, 
        matter_id: str, 
        limit: int = 50, 
        offset: int = 0,
        type_filter: Optional[str] = None,
        search_query: Optional[str] = None,
        search_type: Optional[str] = "semantic"
    ) -> Dict[str, Any]:
        """Get memory items for a matter with pagination and filtering."""
        session = await self._get_session()
        try:
            params = {
                "limit": limit,
                "offset": offset,
                "search_type": search_type
            }
            if type_filter:
                params["type_filter"] = type_filter
            if search_query:
                params["search_query"] = search_query
            
            async with session.get(
                f"{self.base_url}/api/matters/{matter_id}/memory/items",
                params=params
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to get memory items", error=str(e))
            raise
    
    async def get_memory_item(self, matter_id: str, item_id: str) -> Dict[str, Any]:
        """Get a specific memory item by ID."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/api/matters/{matter_id}/memory/items/{item_id}"
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to get memory item", error=str(e))
            raise
    
    async def create_memory_item(
        self,
        matter_id: str,
        text: str,
        type: str = "Raw",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new memory item."""
        session = await self._get_session()
        try:
            payload = {
                "text": text,
                "type": type
            }
            if metadata:
                payload["metadata"] = metadata
            
            async with session.post(
                f"{self.base_url}/api/matters/{matter_id}/memory/items",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to create memory item", error=str(e))
            raise
    
    async def update_memory_item(
        self,
        matter_id: str,
        item_id: str,
        new_text: str,
        preserve_type: bool = True
    ) -> Dict[str, Any]:
        """Update an existing memory item."""
        session = await self._get_session()
        try:
            payload = {
                "new_text": new_text,
                "preserve_type": preserve_type
            }
            
            async with session.put(
                f"{self.base_url}/api/matters/{matter_id}/memory/items/{item_id}",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to update memory item", item_id=item_id, error=str(e))
            raise
    
    async def delete_memory_item(
        self,
        matter_id: str,
        item_id: str
    ) -> Dict[str, Any]:
        """Delete a memory item."""
        session = await self._get_session()
        try:
            async with session.delete(
                f"{self.base_url}/api/matters/{matter_id}/memory/items/{item_id}"
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to delete memory item", item_id=item_id, error=str(e))
            raise
    
    async def get_memory_analytics(self, matter_id: str) -> Dict[str, Any]:
        """Get memory analytics for a matter."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/api/matters/{matter_id}/memory/analytics"
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("Failed to get memory analytics", error=str(e))
            raise
    
    # Health Check
    async def health_check(self) -> bool:
        """Check if backend API is healthy."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False