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
    
    def __init__(self, base_url: str = "http://localhost:8000"):
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
    
    # Job Status
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
    
    # Chat / RAG
    async def send_chat_message(
        self,
        matter_id: str,
        query: str,
        k: int = 8,
        model: Optional[str] = None,
        max_tokens: int = 900
    ) -> Dict[str, Any]:
        """Send chat message and get RAG response."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={
                    "matter_id": matter_id,
                    "query": query,
                    "k": k,
                    "model": model,
                    "max_tokens": max_tokens
                }
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
                    "provider": provider,
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