"""
Background job queue with progress tracking.

Provides AsyncIO-based job processing for long-running tasks like
PDF ingestion and large model operations.
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
import time
import random
import heapq
from datetime import datetime, timedelta

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
    # New fields for enhanced functionality
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 0  # Higher numbers = higher priority
    estimated_duration: Optional[float] = None  # Seconds
    progress_history: Optional[List[Dict[str, Any]]] = None


class JobQueue:
    """Enhanced async job queue with persistence and recovery."""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.jobs: Dict[str, JobInfo] = {}
        self.job_params: Dict[str, Dict[str, Any]] = {}  # Store job parameters
        self.job_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()  # Priority-based queue
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.cancelled_jobs: set = set()
        self._workers_started = False
        self._recovery_completed = False
        self._job_counter = 0  # For priority queue ordering
    
    async def start_workers(self) -> None:
        """Start background job processing workers and recover jobs."""
        if self._workers_started:
            return
        
        # Import persistence here to avoid circular imports
        from .job_persistence import job_persistence
        
        # Recover jobs from previous session
        if not self._recovery_completed:
            await self._recover_jobs(job_persistence)
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            worker_task = asyncio.create_task(self._job_worker(f"worker-{i}"))
            # Don't await - let workers run in background
            
        self._workers_started = True
        logger.info("Job queue workers started", max_concurrent=self.max_concurrent)
    
    async def _recover_jobs(self, persistence) -> None:
        """Recover jobs from persistent storage."""
        try:
            recoverable_jobs = await persistence.load_recoverable_jobs()
            
            for job_info, params in recoverable_jobs:
                # Re-queue the job
                self.jobs[job_info.job_id] = job_info
                self.job_params[job_info.job_id] = params
                
                # Add to priority queue (negative priority for max-heap behavior)
                priority = (-job_info.priority, self._job_counter)
                self._job_counter += 1
                
                progress_callback = params.get('progress_callback')
                await self.job_queue.put((priority, (job_info.job_id, job_info.job_type, params, progress_callback)))
            
            self._recovery_completed = True
            
            if recoverable_jobs:
                logger.info("Recovered jobs from previous session", count=len(recoverable_jobs))
                
        except Exception as e:
            logger.error("Failed to recover jobs", error=str(e))
    
    async def submit_job(
        self,
        job_type: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        priority: int = 0,
        max_retries: int = 3
    ) -> str:
        """
        Submit a job for background processing.
        
        Args:
            job_type: Type of job to execute
            params: Job parameters
            progress_callback: Optional progress callback
            priority: Job priority (higher = more important)
            max_retries: Maximum retry attempts
            
        Returns:
            Job ID for tracking
        """
        from .job_persistence import job_persistence
        
        job_id = str(uuid.uuid4())
        
        job_info = JobInfo(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            progress=0.0,
            message="Queued",
            created_at=datetime.utcnow(),
            priority=priority,
            max_retries=max_retries,
            progress_history=[]
        )
        
        self.jobs[job_id] = job_info
        self.job_params[job_id] = params.copy()
        
        # Save to persistent storage
        try:
            await job_persistence.save_job(job_info, params)
        except Exception as e:
            logger.warning("Failed to persist job", job_id=job_id, error=str(e))
        
        # Add job to priority queue (negative priority for max-heap behavior)
        queue_priority = (-priority, self._job_counter)
        self._job_counter += 1
        
        await self.job_queue.put((queue_priority, (job_id, job_type, params, progress_callback)))
        
        logger.info("Job submitted", job_id=job_id, job_type=job_type, priority=priority)
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[JobInfo]:
        """Get current status of a job."""
        return self.jobs.get(job_id)
    
    async def get_all_jobs(self, limit: int = 50) -> List[JobInfo]:
        """Get all jobs in memory, sorted by creation time."""
        jobs = list(self.jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]
    
    async def get_job_history(self, limit: int = 100, status_filter: Optional[str] = None) -> List[JobInfo]:
        """Get job history from persistent storage."""
        from .job_persistence import job_persistence
        return await job_persistence.get_job_history(limit, status_filter)
    
    async def get_running_jobs(self) -> List[JobInfo]:
        """Get currently running jobs."""
        return [job for job in self.jobs.values() if job.status == JobStatus.RUNNING]
    
    async def get_queued_jobs(self) -> List[JobInfo]:
        """Get queued jobs."""
        return [job for job in self.jobs.values() if job.status == JobStatus.QUEUED]
    
    async def retry_job(self, job_id: str) -> bool:
        """Manually retry a failed job."""
        if job_id not in self.jobs or job_id not in self.job_params:
            return False
        
        job_info = self.jobs[job_id]
        
        # Only retry failed jobs
        if job_info.status != JobStatus.FAILED:
            return False
        
        # Reset job status
        job_info.status = JobStatus.QUEUED
        job_info.progress = 0.0
        job_info.message = "Manually retried"
        job_info.started_at = None
        job_info.completed_at = None
        job_info.error_message = None
        job_info.result = None
        job_info.retry_count = 0  # Reset retry count for manual retry
        
        # Re-queue with original parameters
        params = self.job_params[job_id]
        priority = (-job_info.priority, self._job_counter)
        self._job_counter += 1
        
        await self.job_queue.put((priority, (job_id, job_info.job_type, params, None)))
        
        logger.info("Job manually retried", job_id=job_id)
        return True
    
    async def cleanup_completed_jobs(self, older_than_hours: int = 24) -> int:
        """Clean up old completed jobs from memory and storage."""
        from .job_persistence import job_persistence
        
        # Clean up from persistent storage
        keep_days = max(1, older_than_hours // 24)
        deleted_count = await job_persistence.cleanup_old_jobs(keep_days)
        
        # Clean up from memory
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        jobs_to_remove = []
        
        for job_id, job_info in self.jobs.items():
            if (job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                job_info.completed_at and job_info.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            self.jobs.pop(job_id, None)
            self.job_params.pop(job_id, None)
            self.cancelled_jobs.discard(job_id)
        
        memory_cleaned = len(jobs_to_remove)
        
        if memory_cleaned > 0:
            logger.info("Cleaned up jobs", 
                       storage_deleted=deleted_count,
                       memory_cleaned=memory_cleaned)
        
        return deleted_count + memory_cleaned
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Attempt to cancel a job with proper cleanup.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled, False otherwise
        """
        from .job_persistence import job_persistence
        
        if job_id not in self.jobs:
            return False
        
        # Mark as cancelled
        self.cancelled_jobs.add(job_id)
        job_info = self.jobs[job_id]
        
        # Cancel running job
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            
            # Wait briefly for graceful cancellation
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass  # Expected
        
        # Update job status
        job_info.status = JobStatus.CANCELLED
        job_info.message = "Cancelled by user"
        job_info.completed_at = datetime.utcnow()
        
        # Persist the cancellation
        try:
            if job_id in self.job_params:
                await job_persistence.save_job(job_info, self.job_params[job_id])
        except Exception as e:
            logger.warning("Failed to persist job cancellation", job_id=job_id, error=str(e))
        
        logger.info("Job cancelled", job_id=job_id)
        return True
    
    async def _job_worker(self, worker_name: str) -> None:
        """Enhanced worker coroutine for processing jobs."""
        logger.info(f"Job worker {worker_name} started")
        
        while True:
            try:
                # Wait for a job from the priority queue
                priority, job_data = await self.job_queue.get()
                job_id, job_type, params, progress_callback = job_data
                
                # Check if job was cancelled while queued
                if job_id in self.cancelled_jobs:
                    self.job_queue.task_done()
                    continue
                
                # Update job status to running
                if job_id in self.jobs:
                    job_info = self.jobs[job_id]
                    job_info.status = JobStatus.RUNNING
                    job_info.started_at = datetime.utcnow()
                    job_info.message = f"Processing with {worker_name}"
                    
                    # Estimate duration if not set
                    if job_info.estimated_duration is None:
                        job_info.estimated_duration = self._estimate_job_duration(job_type)
                    
                    logger.info(
                        "Job started",
                        job_id=job_id,
                        job_type=job_type,
                        worker=worker_name,
                        retry_count=job_info.retry_count
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
                    # Job failed - check if we should retry
                    if job_id in self.jobs:
                        job_info = self.jobs[job_id]
                        
                        # Check if we should retry
                        should_retry = (job_info.retry_count < job_info.max_retries and 
                                      self._is_retryable_error(e))
                        
                        if should_retry:
                            # Increment retry count and re-queue
                            job_info.retry_count += 1
                            job_info.status = JobStatus.QUEUED
                            job_info.message = f"Retry {job_info.retry_count}/{job_info.max_retries}: {str(e)}"
                            job_info.started_at = None
                            job_info.progress = 0.0
                            
                            # Calculate backoff delay
                            delay = min(300, (2 ** job_info.retry_count) + random.uniform(0, 1))
                            
                            # Re-queue after delay
                            asyncio.create_task(self._retry_job_after_delay(
                                job_id, job_type, params, progress_callback, delay, priority
                            ))
                            
                            logger.warning(
                                "Job failed, will retry",
                                job_id=job_id,
                                job_type=job_type,
                                retry_count=job_info.retry_count,
                                max_retries=job_info.max_retries,
                                delay_seconds=delay,
                                error=str(e)
                            )
                        else:
                            # Mark as failed
                            job_info.status = JobStatus.FAILED
                            job_info.completed_at = datetime.utcnow()
                            job_info.error_message = str(e)
                            job_info.message = f"Failed after {job_info.retry_count} retries: {str(e)}"
                            
                            logger.error(
                                "Job failed permanently",
                                job_id=job_id,
                                job_type=job_type,
                                worker=worker_name,
                                retry_count=job_info.retry_count,
                                error=str(e)
                            )
                
                finally:
                    # Clean up tracking and persist final state
                    self.running_jobs.pop(job_id, None)
                    
                    # Persist job state
                    try:
                        from .job_persistence import job_persistence
                        if job_id in self.jobs and job_id in self.job_params:
                            await job_persistence.save_job(self.jobs[job_id], self.job_params[job_id])
                    except Exception as e:
                        logger.warning("Failed to persist job state", job_id=job_id, error=str(e))
                    
                    self.job_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Unexpected error in job worker {worker_name}: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def _retry_job_after_delay(
        self, 
        job_id: str, 
        job_type: str, 
        params: Dict[str, Any],
        progress_callback: Optional[Callable],
        delay: float,
        original_priority: tuple
    ) -> None:
        """Re-queue a job after a delay for retry."""
        try:
            await asyncio.sleep(delay)
            
            # Check if job was cancelled during delay
            if job_id in self.cancelled_jobs:
                return
            
            # Re-queue the job
            await self.job_queue.put((original_priority, (job_id, job_type, params, progress_callback)))
            
        except Exception as e:
            logger.error("Failed to retry job", job_id=job_id, error=str(e))
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        # Network/connection errors are typically retryable
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            OSError,  # Network-related OS errors
        )
        
        # Check error type
        if isinstance(error, retryable_errors):
            return True
        
        # Check error message for known retryable patterns
        error_msg = str(error).lower()
        retryable_patterns = [
            "connection",
            "timeout", 
            "network",
            "temporary",
            "service unavailable",
            "rate limit"
        ]
        
        return any(pattern in error_msg for pattern in retryable_patterns)
    
    def _estimate_job_duration(self, job_type: str) -> float:
        """Estimate job duration in seconds based on job type."""
        duration_estimates = {
            "pdf_ingestion": 180.0,  # 3 minutes
            "batch_document_processing": 600.0,  # 10 minutes
            "large_model_operation": 300.0,  # 5 minutes
            "matter_analysis": 480.0,  # 8 minutes
            "test_job": 10.0,  # 10 seconds
        }
        
        return duration_estimates.get(job_type, 120.0)  # 2 minutes default
    
    async def _execute_job(
        self,
        job_id: str,
        job_type: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Any:
        """Execute a specific job type with enhanced progress tracking."""
        from .job_persistence import job_persistence
        
        start_time = time.time()
        
        def update_progress(progress: float, message: str, sub_progress: Optional[Dict[str, Any]] = None):
            """Enhanced progress update with history tracking."""
            current_time = time.time()
            elapsed = current_time - start_time
            
            if job_id in self.jobs:
                job_info = self.jobs[job_id]
                job_info.progress = progress
                job_info.message = message
                
                # Add to progress history
                if job_info.progress_history is None:
                    job_info.progress_history = []
                
                progress_entry = {
                    "progress": progress,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat(),
                    "elapsed_seconds": elapsed
                }
                
                if sub_progress:
                    progress_entry["sub_progress"] = sub_progress
                
                # Calculate ETA
                if progress > 0 and job_info.estimated_duration:
                    estimated_total = elapsed / progress
                    eta_seconds = max(0, estimated_total - elapsed)
                    progress_entry["eta_seconds"] = eta_seconds
                
                job_info.progress_history.append(progress_entry)
                
                # Keep only last 50 progress entries to avoid memory bloat
                if len(job_info.progress_history) > 50:
                    job_info.progress_history = job_info.progress_history[-50:]
            
            # Save progress to persistent storage
            asyncio.create_task(job_persistence.save_progress_history(job_id, progress, message))
            
            if progress_callback:
                progress_callback(progress, message)
        
        if job_type == "pdf_ingestion":
            return await self._execute_pdf_ingestion(job_id, params, update_progress)
        elif job_type == "batch_document_processing":
            from .job_types import execute_batch_document_processing
            return await execute_batch_document_processing(job_id, params, update_progress)
        elif job_type == "large_model_operation":
            from .job_types import execute_large_model_operation
            return await execute_large_model_operation(job_id, params, update_progress)
        elif job_type == "matter_analysis":
            from .job_types import execute_matter_analysis
            return await execute_matter_analysis(job_id, params, update_progress)
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
        from .vectors import VectorStore
        from .chunking import Chunk
        from pathlib import Path
        import json
        
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
                progress_callback=lambda p, m: progress_callback(p * 0.7, m)  # OCR/parsing/chunking is 70% of work
            )
            
            # Now embed and store chunks in vector database
            progress_callback(0.7, "Embedding and storing document chunks...")
            
            # Initialize vector store
            vector_store = VectorStore(matter.paths.root)
            
            # Collect all chunks from successful ingestions
            all_chunks = []
            for file_path, stats in results.items():
                if stats.success and stats.total_chunks > 0:
                    # Load chunks from the saved JSONL file
                    chunks_path = matter.paths.parsed / f"{stats.doc_id}_chunks.jsonl"
                    if chunks_path.exists():
                        with open(chunks_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                chunk_data = json.loads(line)
                                # Reconstruct Chunk object from JSON data
                                chunk = Chunk(
                                    chunk_id=chunk_data["chunk_id"],
                                    text=chunk_data["text"],
                                    doc_id=chunk_data["doc_id"],
                                    doc_name=chunk_data["doc_name"],
                                    page_start=chunk_data["page_start"],
                                    page_end=chunk_data["page_end"],
                                    token_count=chunk_data["token_count"],
                                    char_count=chunk_data["char_count"],
                                    md5=chunk_data["md5"],
                                    section_title=chunk_data.get("section_title"),
                                    chunk_index=chunk_data["chunk_index"],
                                    overlap_info=chunk_data.get("overlap_info", {}),
                                    metadata=chunk_data.get("metadata", {})
                                )
                                all_chunks.append(chunk)
            
            # Embed and store chunks
            if all_chunks:
                progress_callback(0.8, f"Embedding {len(all_chunks)} chunks...")
                await vector_store.upsert_chunks(all_chunks)
                progress_callback(0.95, "Finalizing vector storage...")
            
            progress_callback(1.0, "Ingestion completed")
            
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
                "chunks_embedded": len(all_chunks),
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