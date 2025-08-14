"""
Background job queue with progress tracking.

Provides AsyncIO-based job processing for long-running tasks like
PDF ingestion and large model operations.
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import asyncio
import uuid
import time
from datetime import datetime

from .logging_conf import get_logger

logger = get_logger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Job information and status."""
    job_id: str
    job_type: str
    status: JobStatus
    progress: float  # 0.0 to 1.0
    message: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None


class JobQueue:
    """Async job queue for background processing."""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.jobs: Dict[str, JobInfo] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self._workers_started = False
    
    async def start_workers(self) -> None:
        """Start background job processing workers."""
        if self._workers_started:
            return
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            worker_task = asyncio.create_task(self._job_worker(f"worker-{i}"))
            # Don't await - let workers run in background
            
        self._workers_started = True
        logger.info("Job queue workers started", max_concurrent=self.max_concurrent)
    
    async def submit_job(
        self,
        job_type: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Submit a job for background processing.
        
        Args:
            job_type: Type of job to execute
            params: Job parameters
            progress_callback: Optional progress callback
            
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        job_info = JobInfo(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            progress=0.0,
            message="Queued",
            created_at=datetime.utcnow()
        )
        
        self.jobs[job_id] = job_info
        
        # Add job to queue for processing
        await self.job_queue.put((job_id, job_type, params, progress_callback))
        
        logger.info("Job submitted", job_id=job_id, job_type=job_type)
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[JobInfo]:
        """Get current status of a job."""
        return self.jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Attempt to cancel a job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled, False otherwise
        """
        # TODO: Implement job cancellation
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            
            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.CANCELLED
                self.jobs[job_id].message = "Cancelled"
                self.jobs[job_id].completed_at = datetime.utcnow()
            
            logger.info("Job cancelled", job_id=job_id)
            return True
        
        return False
    
    async def _job_worker(self, worker_name: str) -> None:
        """Worker coroutine for processing jobs."""
        logger.info(f"Job worker {worker_name} started")
        
        while True:
            try:
                # Wait for a job from the queue
                job_data = await self.job_queue.get()
                job_id, job_type, params, progress_callback = job_data
                
                # Update job status to running
                if job_id in self.jobs:
                    job_info = self.jobs[job_id]
                    job_info.status = JobStatus.RUNNING
                    job_info.started_at = datetime.utcnow()
                    job_info.message = f"Processing with {worker_name}"
                    
                    logger.info(
                        "Job started",
                        job_id=job_id,
                        job_type=job_type,
                        worker=worker_name
                    )
                
                # Create tracking task
                self.running_jobs[job_id] = asyncio.current_task()
                
                try:
                    # Execute the job
                    result = await self._execute_job(job_id, job_type, params, progress_callback)
                    
                    # Mark as completed
                    if job_id in self.jobs:
                        job_info = self.jobs[job_id]
                        job_info.status = JobStatus.COMPLETED
                        job_info.completed_at = datetime.utcnow()
                        job_info.message = "Completed successfully"
                        job_info.result = result
                        job_info.progress = 1.0
                        
                        logger.info(
                            "Job completed",
                            job_id=job_id,
                            job_type=job_type,
                            worker=worker_name,
                            duration_seconds=(job_info.completed_at - job_info.started_at).total_seconds()
                        )
                        
                except asyncio.CancelledError:
                    # Job was cancelled
                    if job_id in self.jobs:
                        job_info = self.jobs[job_id]
                        job_info.status = JobStatus.CANCELLED
                        job_info.completed_at = datetime.utcnow()
                        job_info.message = "Job was cancelled"
                    
                    logger.info("Job cancelled", job_id=job_id, worker=worker_name)
                    
                except Exception as e:
                    # Job failed
                    if job_id in self.jobs:
                        job_info = self.jobs[job_id]
                        job_info.status = JobStatus.FAILED
                        job_info.completed_at = datetime.utcnow()
                        job_info.error_message = str(e)
                        job_info.message = f"Failed: {str(e)}"
                    
                    logger.error(
                        "Job failed",
                        job_id=job_id,
                        job_type=job_type,
                        worker=worker_name,
                        error=str(e)
                    )
                
                finally:
                    # Clean up tracking
                    self.running_jobs.pop(job_id, None)
                    self.job_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Unexpected error in job worker {worker_name}: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def _execute_job(
        self,
        job_id: str,
        job_type: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Any:
        """Execute a specific job type."""
        
        def update_progress(progress: float, message: str):
            """Update job progress and call callback if provided."""
            if job_id in self.jobs:
                self.jobs[job_id].progress = progress
                self.jobs[job_id].message = message
            
            if progress_callback:
                progress_callback(progress, message)
        
        if job_type == "pdf_ingestion":
            return await self._execute_pdf_ingestion(job_id, params, update_progress)
        elif job_type == "test_job":
            return await self._execute_test_job(job_id, params, update_progress)
        else:
            raise ValueError(f"Unknown job type: {job_type}")
    
    async def _execute_pdf_ingestion(
        self,
        job_id: str,
        params: Dict[str, Any],
        progress_callback: Callable[[float, str], None]
    ) -> Dict[str, Any]:
        """Execute PDF ingestion job."""
        from .matters import matter_manager
        from .ingest import IngestionPipeline, IngestionError
        from pathlib import Path
        
        # Extract parameters
        matter_id = params.get("matter_id")
        uploaded_files = params.get("uploaded_files", [])
        force_ocr = params.get("force_ocr", False)
        ocr_language = params.get("ocr_language", "eng")
        
        if not matter_id:
            raise ValueError("matter_id is required for PDF ingestion")
        
        if not uploaded_files:
            raise ValueError("No files provided for ingestion")
        
        # Get the matter
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise ValueError(f"Matter not found: {matter_id}")
        
        # Convert file paths to Path objects
        pdf_files = [Path(f["path"]) for f in uploaded_files if f.get("path")]
        
        # Initialize ingestion pipeline
        pipeline = IngestionPipeline(matter)
        
        try:
            # Process the PDFs
            results = await pipeline.ingest_pdfs(
                pdf_files=pdf_files,
                force_ocr=force_ocr,
                ocr_language=ocr_language,
                progress_callback=progress_callback
            )
            
            # Calculate summary statistics
            successful_files = sum(1 for r in results.values() if r.success)
            failed_files = len(results) - successful_files
            total_pages = sum(r.total_pages for r in results.values())
            total_chunks = sum(r.total_chunks for r in results.values())
            
            summary = {
                "total_files": len(pdf_files),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "results": {str(k): {
                    "doc_name": v.doc_name,
                    "success": v.success,
                    "pages": v.total_pages,
                    "chunks": v.total_chunks,
                    "ocr_status": v.ocr_status,
                    "error": v.error_message
                } for k, v in results.items()}
            }
            
            return summary
            
        except IngestionError as e:
            logger.error(
                "PDF ingestion failed",
                job_id=job_id,
                matter_id=matter_id,
                error=e.message,
                stage=e.stage
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during PDF ingestion",
                job_id=job_id,
                matter_id=matter_id,
                error=str(e)
            )
            raise
    
    async def _execute_test_job(
        self,
        job_id: str,
        params: Dict[str, Any],
        progress_callback: Callable[[float, str], None]
    ) -> str:
        """Execute a test job for debugging."""
        duration = params.get("duration", 5)
        steps = params.get("steps", 10)
        
        for i in range(steps):
            progress = i / steps
            message = f"Test step {i+1}/{steps}"
            progress_callback(progress, message)
            
            await asyncio.sleep(duration / steps)
        
        progress_callback(1.0, "Test completed")
        return f"Test job completed with {steps} steps over {duration} seconds"


# Global job queue instance  
job_queue = JobQueue()


async def ensure_job_queue_started():
    """Ensure the job queue workers are started."""
    await job_queue.start_workers()