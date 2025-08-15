"""
Job persistence and recovery system.

Provides persistent storage for job state to enable recovery after
application restarts and maintain job history.
"""

import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .models import Matter
from .jobs import JobInfo, JobStatus
from .settings import settings
from .logging_conf import get_logger

logger = get_logger(__name__)


class JobPersistence:
    """Handles job persistence and recovery operations."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize job persistence."""
        if storage_path is None:
            storage_path = Path.home() / ".letta-claim" / "jobs.db"
        
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the SQLite database for job storage."""
        try:
            with sqlite3.connect(str(self.storage_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress REAL DEFAULT 0.0,
                        message TEXT DEFAULT '',
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        error_message TEXT,
                        result_json TEXT,
                        params_json TEXT NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        priority INTEGER DEFAULT 0
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS job_progress_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        progress REAL NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (job_id) REFERENCES jobs (job_id)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_status 
                    ON jobs(status)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_created_at 
                    ON jobs(created_at DESC)
                """)
                
                logger.debug("Job database initialized", path=str(self.storage_path))
                
        except Exception as e:
            logger.error("Failed to initialize job database", error=str(e))
            raise
    
    async def save_job(self, job_info: JobInfo, params: Dict[str, Any]) -> None:
        """Save job to persistent storage."""
        try:
            def _save():
                with sqlite3.connect(str(self.storage_path)) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO jobs 
                        (job_id, job_type, status, progress, message, created_at, 
                         started_at, completed_at, error_message, result_json, params_json,
                         retry_count, max_retries, priority)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        job_info.job_id,
                        job_info.job_type,
                        job_info.status.value,
                        job_info.progress,
                        job_info.message,
                        job_info.created_at.isoformat() if job_info.created_at else None,
                        job_info.started_at.isoformat() if job_info.started_at else None,
                        job_info.completed_at.isoformat() if job_info.completed_at else None,
                        job_info.error_message,
                        json.dumps(job_info.result) if job_info.result else None,
                        json.dumps(params),
                        getattr(job_info, 'retry_count', 0),
                        getattr(job_info, 'max_retries', 3),
                        getattr(job_info, 'priority', 0)
                    ))
            
            # Run in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(None, _save)
            
        except Exception as e:
            logger.error("Failed to save job", job_id=job_info.job_id, error=str(e))
            raise
    
    async def load_job(self, job_id: str) -> Optional[tuple[JobInfo, Dict[str, Any]]]:
        """Load job from persistent storage."""
        try:
            def _load():
                with sqlite3.connect(str(self.storage_path)) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT * FROM jobs WHERE job_id = ?
                    """, (job_id,))
                    return cursor.fetchone()
            
            row = await asyncio.get_event_loop().run_in_executor(None, _load)
            
            if not row:
                return None
            
            # Convert row to JobInfo
            job_info = JobInfo(
                job_id=row['job_id'],
                job_type=row['job_type'],
                status=JobStatus(row['status']),
                progress=row['progress'],
                message=row['message'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                error_message=row['error_message'],
                result=json.loads(row['result_json']) if row['result_json'] else None
            )
            
            # Add persistence-specific fields
            job_info.retry_count = row['retry_count']
            job_info.max_retries = row['max_retries']
            job_info.priority = row['priority']
            
            params = json.loads(row['params_json'])
            
            return job_info, params
            
        except Exception as e:
            logger.error("Failed to load job", job_id=job_id, error=str(e))
            return None
    
    async def load_recoverable_jobs(self) -> List[tuple[JobInfo, Dict[str, Any]]]:
        """Load jobs that can be recovered (running or queued jobs)."""
        try:
            def _load():
                with sqlite3.connect(str(self.storage_path)) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT * FROM jobs 
                        WHERE status IN ('queued', 'running')
                        ORDER BY created_at ASC
                    """)
                    return cursor.fetchall()
            
            rows = await asyncio.get_event_loop().run_in_executor(None, _load)
            
            recoverable_jobs = []
            for row in rows:
                try:
                    job_info = JobInfo(
                        job_id=row['job_id'],
                        job_type=row['job_type'],
                        status=JobStatus.QUEUED,  # Reset to queued for recovery
                        progress=0.0,  # Reset progress
                        message="Recovered and re-queued",
                        created_at=datetime.fromisoformat(row['created_at']),
                        started_at=None,  # Reset start time
                        completed_at=None,
                        error_message=None,
                        result=None
                    )
                    
                    # Preserve persistence fields
                    job_info.retry_count = row['retry_count']
                    job_info.max_retries = row['max_retries'] 
                    job_info.priority = row['priority']
                    
                    params = json.loads(row['params_json'])
                    recoverable_jobs.append((job_info, params))
                    
                except Exception as e:
                    logger.warning(
                        "Failed to recover job",
                        job_id=row['job_id'],
                        error=str(e)
                    )
                    continue
            
            logger.info("Recovered jobs for restart", count=len(recoverable_jobs))
            return recoverable_jobs
            
        except Exception as e:
            logger.error("Failed to load recoverable jobs", error=str(e))
            return []
    
    async def save_progress_history(
        self, 
        job_id: str, 
        progress: float, 
        message: str
    ) -> None:
        """Save progress history for job."""
        try:
            def _save():
                with sqlite3.connect(str(self.storage_path)) as conn:
                    conn.execute("""
                        INSERT INTO job_progress_history 
                        (job_id, progress, message, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (job_id, progress, message, datetime.utcnow().isoformat()))
            
            await asyncio.get_event_loop().run_in_executor(None, _save)
            
        except Exception as e:
            logger.warning("Failed to save progress history", job_id=job_id, error=str(e))
    
    async def get_job_history(
        self, 
        limit: int = 100,
        status_filter: Optional[str] = None
    ) -> List[JobInfo]:
        """Get job history with optional filtering."""
        try:
            def _load():
                with sqlite3.connect(str(self.storage_path)) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    query = "SELECT * FROM jobs"
                    params = []
                    
                    if status_filter:
                        query += " WHERE status = ?"
                        params.append(status_filter)
                    
                    query += " ORDER BY created_at DESC LIMIT ?"
                    params.append(limit)
                    
                    cursor = conn.execute(query, params)
                    return cursor.fetchall()
            
            rows = await asyncio.get_event_loop().run_in_executor(None, _load)
            
            jobs = []
            for row in rows:
                try:
                    job_info = JobInfo(
                        job_id=row['job_id'],
                        job_type=row['job_type'],
                        status=JobStatus(row['status']),
                        progress=row['progress'],
                        message=row['message'],
                        created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                        started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                        completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                        error_message=row['error_message'],
                        result=json.loads(row['result_json']) if row['result_json'] else None
                    )
                    
                    # Add persistence fields
                    job_info.retry_count = row['retry_count']
                    job_info.max_retries = row['max_retries']
                    job_info.priority = row['priority']
                    
                    jobs.append(job_info)
                    
                except Exception as e:
                    logger.warning("Failed to parse job history entry", error=str(e))
                    continue
            
            return jobs
            
        except Exception as e:
            logger.error("Failed to load job history", error=str(e))
            return []
    
    async def cleanup_old_jobs(self, keep_days: int = 7) -> int:
        """Clean up old completed jobs."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
            
            def _cleanup():
                with sqlite3.connect(str(self.storage_path)) as conn:
                    # Clean up old completed/failed jobs
                    cursor = conn.execute("""
                        DELETE FROM jobs 
                        WHERE status IN ('completed', 'failed', 'cancelled')
                        AND created_at < ?
                    """, (cutoff_date.isoformat(),))
                    
                    jobs_deleted = cursor.rowcount
                    
                    # Clean up orphaned progress history
                    conn.execute("""
                        DELETE FROM job_progress_history 
                        WHERE job_id NOT IN (SELECT job_id FROM jobs)
                    """)
                    
                    return jobs_deleted
            
            deleted_count = await asyncio.get_event_loop().run_in_executor(None, _cleanup)
            
            if deleted_count > 0:
                logger.info("Cleaned up old jobs", deleted_count=deleted_count, keep_days=keep_days)
            
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to cleanup old jobs", error=str(e))
            return 0


# Global job persistence instance
job_persistence = JobPersistence()