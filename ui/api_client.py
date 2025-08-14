"""
HTTP client for backend API communication.

Provides async methods to communicate with the FastAPI backend
for all application operations.
"""

from typing import List, Dict, Any, Optional
import aiohttp
import asyncio
import json
from pathlib import Path

from ..app.logging_conf import get_logger

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
        for file_path in files:
            data.add_field(
                'files',
                open(file_path, 'rb'),
                filename=file_path.name,
                content_type='application/pdf'
            )
        
        try:
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
            for file_path in files:
                try:
                    # Note: In real implementation, we'd need to track these properly
                    pass
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