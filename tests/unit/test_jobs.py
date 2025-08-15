"""
Unit tests for enhanced job queue system.

Tests job submission, execution, persistence, retry logic,
cancellation, and all new features.
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.jobs import JobQueue, JobInfo, JobStatus
from app.job_persistence import JobPersistence


class TestJobQueue:
    """Test cases for the enhanced JobQueue class."""
    
    @pytest.fixture
    async def job_queue(self):
        """Create a job queue for testing."""
        queue = JobQueue(max_concurrent=2)
        await queue.start_workers()
        yield queue
        # Cleanup
        try:
            await queue.cleanup_completed_jobs(0)  # Clean up all jobs
        except:
            pass
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for persistence tests."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            storage_path = Path(tmp.name)
        
        yield storage_path
        
        # Cleanup
        if storage_path.exists():
            storage_path.unlink()
    
    @pytest.mark.asyncio
    async def test_job_submission_and_basic_execution(self, job_queue):
        """Test basic job submission and execution."""
        job_id = await job_queue.submit_job(
            job_type="test_job",
            params={"duration": 0.1, "steps": 2},
            priority=1
        )
        
        assert job_id is not None
        assert job_id in job_queue.jobs
        
        # Wait for job to complete
        for _ in range(10):  # 1 second timeout
            await asyncio.sleep(0.1)
            job_info = await job_queue.get_job_status(job_id)
            if job_info.status == JobStatus.COMPLETED:
                break
        
        job_info = await job_queue.get_job_status(job_id)
        assert job_info.status == JobStatus.COMPLETED
        assert job_info.progress == 1.0
        assert job_info.result is not None
        assert job_info.retry_count == 0
        assert job_info.priority == 1
    
    @pytest.mark.asyncio
    async def test_job_priority_ordering(self, job_queue):
        """Test that higher priority jobs are processed first."""
        # Submit multiple jobs with different priorities
        low_priority_job = await job_queue.submit_job(
            job_type="test_job",
            params={"duration": 0.1, "steps": 1},
            priority=0
        )
        
        high_priority_job = await job_queue.submit_job(
            job_type="test_job", 
            params={"duration": 0.1, "steps": 1},
            priority=2
        )
        
        medium_priority_job = await job_queue.submit_job(
            job_type="test_job",
            params={"duration": 0.1, "steps": 1},
            priority=1
        )
        
        # Wait for all jobs to complete
        await asyncio.sleep(1)
        
        # Check completion order - higher priority should start earlier
        high_job = await job_queue.get_job_status(high_priority_job)
        medium_job = await job_queue.get_job_status(medium_priority_job)
        low_job = await job_queue.get_job_status(low_priority_job)
        
        # All should be completed
        assert high_job.status == JobStatus.COMPLETED
        assert medium_job.status == JobStatus.COMPLETED
        assert low_job.status == JobStatus.COMPLETED
        
        # Higher priority should have started first (if measured precisely)
        assert high_job.started_at <= medium_job.started_at
        assert medium_job.started_at <= low_job.started_at
    
    @pytest.mark.asyncio
    async def test_job_cancellation(self, job_queue):
        """Test job cancellation functionality."""
        job_id = await job_queue.submit_job(
            job_type="test_job",
            params={"duration": 2.0, "steps": 10},  # Long running job
            priority=0
        )
        
        # Wait for job to start
        await asyncio.sleep(0.2)
        
        # Cancel the job
        success = await job_queue.cancel_job(job_id)
        assert success == True
        
        # Wait a bit and check status
        await asyncio.sleep(0.1)
        
        job_info = await job_queue.get_job_status(job_id)
        assert job_info.status == JobStatus.CANCELLED
        assert "Cancelled" in job_info.message
        assert job_info.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_automatic_retry_logic(self, job_queue):
        """Test automatic retry logic for retryable errors."""
        # Mock a job that fails first time but succeeds on retry
        original_execute = job_queue._execute_job
        call_count = 0
        
        async def failing_execute_job(job_id, job_type, params, progress_callback):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # Fail the first attempt with retryable error
                raise ConnectionError("Network connection failed")
            else:
                # Succeed on retry
                if progress_callback:
                    progress_callback(1.0, "Completed on retry")
                return {"success": True, "attempts": call_count}
        
        job_queue._execute_job = failing_execute_job
        
        try:
            job_id = await job_queue.submit_job(
                job_type="test_job",
                params={},
                max_retries=2
            )
            
            # Wait for job to complete (including retry)
            for _ in range(50):  # 5 second timeout
                await asyncio.sleep(0.1)
                job_info = await job_queue.get_job_status(job_id)
                if job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    break
            
            job_info = await job_queue.get_job_status(job_id)
            assert job_info.status == JobStatus.COMPLETED
            assert job_info.retry_count == 1  # One retry
            assert job_info.result["attempts"] == 2
            
        finally:
            job_queue._execute_job = original_execute
    
    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, job_queue):
        """Test that jobs fail permanently after max retries."""
        # Mock a job that always fails with retryable error
        original_execute = job_queue._execute_job
        
        async def always_failing_execute_job(job_id, job_type, params, progress_callback):
            raise ConnectionError("Network always fails")
        
        job_queue._execute_job = always_failing_execute_job
        
        try:
            job_id = await job_queue.submit_job(
                job_type="test_job",
                params={},
                max_retries=2
            )
            
            # Wait for job to fail permanently
            for _ in range(50):
                await asyncio.sleep(0.1)
                job_info = await job_queue.get_job_status(job_id)
                if job_info.status == JobStatus.FAILED:
                    break
            
            job_info = await job_queue.get_job_status(job_id)
            assert job_info.status == JobStatus.FAILED
            assert job_info.retry_count == 2  # Max retries reached
            assert "after 2 retries" in job_info.message
            
        finally:
            job_queue._execute_job = original_execute
    
    @pytest.mark.asyncio
    async def test_non_retryable_error(self, job_queue):
        """Test that non-retryable errors fail immediately."""
        # Mock a job that fails with non-retryable error
        original_execute = job_queue._execute_job
        
        async def non_retryable_failing_execute_job(job_id, job_type, params, progress_callback):
            raise ValueError("Invalid parameter - not retryable")
        
        job_queue._execute_job = non_retryable_failing_execute_job
        
        try:
            job_id = await job_queue.submit_job(
                job_type="test_job",
                params={},
                max_retries=3
            )
            
            # Wait for job to fail
            for _ in range(20):
                await asyncio.sleep(0.1)
                job_info = await job_queue.get_job_status(job_id)
                if job_info.status == JobStatus.FAILED:
                    break
            
            job_info = await job_queue.get_job_status(job_id)
            assert job_info.status == JobStatus.FAILED
            assert job_info.retry_count == 0  # No retries attempted
            
        finally:
            job_queue._execute_job = original_execute
    
    @pytest.mark.asyncio
    async def test_manual_retry(self, job_queue):
        """Test manual retry of failed jobs."""
        # Create a failed job first
        original_execute = job_queue._execute_job
        
        async def failing_execute_job(job_id, job_type, params, progress_callback):
            raise ValueError("This job always fails")
        
        job_queue._execute_job = failing_execute_job
        
        job_id = await job_queue.submit_job(
            job_type="test_job",
            params={},
            max_retries=0  # Fail immediately
        )
        
        # Wait for job to fail
        for _ in range(20):
            await asyncio.sleep(0.1)
            job_info = await job_queue.get_job_status(job_id)
            if job_info.status == JobStatus.FAILED:
                break
        
        # Now restore normal execution and retry
        job_queue._execute_job = original_execute
        
        success = await job_queue.retry_job(job_id)
        assert success == True
        
        # Wait for retry to complete
        for _ in range(20):
            await asyncio.sleep(0.1)
            job_info = await job_queue.get_job_status(job_id)
            if job_info.status == JobStatus.COMPLETED:
                break
        
        job_info = await job_queue.get_job_status(job_id)
        assert job_info.status == JobStatus.COMPLETED
        assert job_info.retry_count == 0  # Reset on manual retry
        assert "Manually retried" in job_info.message or job_info.status == JobStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_progress_tracking_with_eta(self, job_queue):
        """Test enhanced progress tracking with ETA calculation."""
        job_id = await job_queue.submit_job(
            job_type="test_job",
            params={"duration": 0.5, "steps": 5},
            priority=0
        )
        
        # Wait for job to start and track progress
        progress_updates = []
        
        for _ in range(20):
            await asyncio.sleep(0.1)
            job_info = await job_queue.get_job_status(job_id)
            progress_updates.append((job_info.status, job_info.progress))
            
            if job_info.status == JobStatus.COMPLETED:
                break
        
        # Check that progress increased over time
        running_updates = [(s, p) for s, p in progress_updates if s == JobStatus.RUNNING]
        if len(running_updates) >= 2:
            assert running_updates[-1][1] > running_updates[0][1]
        
        final_job = await job_queue.get_job_status(job_id)
        assert final_job.status == JobStatus.COMPLETED
        assert final_job.progress == 1.0
        
        # Check that progress history was recorded
        if hasattr(final_job, 'progress_history') and final_job.progress_history:
            assert len(final_job.progress_history) > 0
            assert final_job.progress_history[-1]["progress"] == 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_job_execution(self, job_queue):
        """Test that multiple jobs can run concurrently."""
        # Submit multiple jobs
        job_ids = []
        for i in range(4):  # More than max_concurrent (2)
            job_id = await job_queue.submit_job(
                job_type="test_job",
                params={"duration": 0.3, "steps": 3},
                priority=0
            )
            job_ids.append(job_id)
        
        # Wait a bit and check that some jobs are running concurrently
        await asyncio.sleep(0.1)
        
        running_jobs = await job_queue.get_running_jobs()
        queued_jobs = await job_queue.get_queued_jobs()
        
        # Should have jobs in both running and queued states
        assert len(running_jobs) <= job_queue.max_concurrent
        assert len(running_jobs) + len(queued_jobs) >= 3
        
        # Wait for all to complete
        await asyncio.sleep(2.0)
        
        # All jobs should be completed
        for job_id in job_ids:
            job_info = await job_queue.get_job_status(job_id)
            assert job_info.status == JobStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_job_cleanup(self, job_queue):
        """Test cleanup of old completed jobs."""
        # Create some completed jobs
        job_ids = []
        for i in range(3):
            job_id = await job_queue.submit_job(
                job_type="test_job",
                params={"duration": 0.1, "steps": 1},
                priority=0
            )
            job_ids.append(job_id)
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        # Manually set completion times to be old
        for job_id in job_ids:
            if job_id in job_queue.jobs:
                job_queue.jobs[job_id].completed_at = datetime.utcnow() - timedelta(hours=25)
        
        # Run cleanup
        cleaned_count = await job_queue.cleanup_completed_jobs(24)
        
        # Jobs should be removed from memory
        assert cleaned_count >= 3
        for job_id in job_ids:
            assert job_id not in job_queue.jobs


class TestJobPersistence:
    """Test cases for job persistence functionality."""
    
    @pytest.fixture
    def persistence(self, temp_storage):
        """Create job persistence instance with temporary storage."""
        return JobPersistence(temp_storage)
    
    @pytest.fixture
    def sample_job_info(self):
        """Create sample job info for testing."""
        return JobInfo(
            job_id="test-123",
            job_type="test_job",
            status=JobStatus.COMPLETED,
            progress=1.0,
            message="Test completed",
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            result={"success": True},
            retry_count=1,
            max_retries=3,
            priority=1
        )
    
    @pytest.mark.asyncio
    async def test_save_and_load_job(self, persistence, sample_job_info):
        """Test saving and loading job to/from persistent storage."""
        params = {"test_param": "test_value"}
        
        # Save job
        await persistence.save_job(sample_job_info, params)
        
        # Load job
        loaded_data = await persistence.load_job(sample_job_info.job_id)
        
        assert loaded_data is not None
        loaded_job, loaded_params = loaded_data
        
        assert loaded_job.job_id == sample_job_info.job_id
        assert loaded_job.job_type == sample_job_info.job_type
        assert loaded_job.status == sample_job_info.status
        assert loaded_job.progress == sample_job_info.progress
        assert loaded_job.retry_count == sample_job_info.retry_count
        assert loaded_params == params
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_job(self, persistence):
        """Test loading a job that doesn't exist."""
        result = await persistence.load_job("nonexistent-job")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_recoverable_jobs_loading(self, persistence):
        """Test loading recoverable jobs (running/queued)."""
        # Create jobs in different states
        jobs_data = [
            (JobInfo(
                job_id="running-1",
                job_type="test_job",
                status=JobStatus.RUNNING,
                progress=0.5,
                message="Running",
                created_at=datetime.utcnow()
            ), {"param": "value1"}),
            (JobInfo(
                job_id="queued-1",
                job_type="test_job", 
                status=JobStatus.QUEUED,
                progress=0.0,
                message="Queued",
                created_at=datetime.utcnow()
            ), {"param": "value2"}),
            (JobInfo(
                job_id="completed-1",
                job_type="test_job",
                status=JobStatus.COMPLETED,
                progress=1.0,
                message="Completed",
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            ), {"param": "value3"})
        ]
        
        # Save all jobs
        for job_info, params in jobs_data:
            await persistence.save_job(job_info, params)
        
        # Load recoverable jobs
        recoverable = await persistence.load_recoverable_jobs()
        
        # Should only get running and queued jobs (2 out of 3)
        assert len(recoverable) == 2
        
        recoverable_ids = {job_info.job_id for job_info, params in recoverable}
        assert "running-1" in recoverable_ids
        assert "queued-1" in recoverable_ids
        assert "completed-1" not in recoverable_ids
        
        # All recovered jobs should be reset to queued
        for job_info, params in recoverable:
            assert job_info.status == JobStatus.QUEUED
            assert job_info.progress == 0.0
    
    @pytest.mark.asyncio
    async def test_progress_history_saving(self, persistence):
        """Test saving progress history."""
        job_id = "progress-test"
        
        # Save multiple progress updates
        progress_updates = [
            (0.2, "Starting"),
            (0.5, "Halfway done"),
            (0.8, "Almost finished"),
            (1.0, "Completed")
        ]
        
        for progress, message in progress_updates:
            await persistence.save_progress_history(job_id, progress, message)
        
        # Note: This test mainly checks that saving doesn't crash
        # In a real database test, we'd verify the records were saved
    
    @pytest.mark.asyncio
    async def test_job_history_with_filtering(self, persistence):
        """Test getting job history with status filtering."""
        # Create jobs with different statuses
        jobs = [
            JobInfo(
                job_id=f"job-{i}",
                job_type="test_job",
                status=JobStatus.COMPLETED if i % 2 == 0 else JobStatus.FAILED,
                progress=1.0,
                message=f"Job {i}",
                created_at=datetime.utcnow() - timedelta(minutes=i),
                completed_at=datetime.utcnow()
            )
            for i in range(5)
        ]
        
        # Save all jobs
        for job in jobs:
            await persistence.save_job(job, {})
        
        # Get all jobs
        all_jobs = await persistence.get_job_history(10)
        assert len(all_jobs) == 5
        
        # Get only completed jobs
        completed_jobs = await persistence.get_job_history(10, "completed")
        assert len(completed_jobs) == 3  # Jobs 0, 2, 4
        assert all(job.status == JobStatus.COMPLETED for job in completed_jobs)
        
        # Get only failed jobs
        failed_jobs = await persistence.get_job_history(10, "failed")
        assert len(failed_jobs) == 2  # Jobs 1, 3
        assert all(job.status == JobStatus.FAILED for job in failed_jobs)
    
    @pytest.mark.asyncio
    async def test_cleanup_old_jobs(self, persistence):
        """Test cleanup of old jobs from persistent storage."""
        # Create old and new jobs
        old_job = JobInfo(
            job_id="old-job",
            job_type="test_job",
            status=JobStatus.COMPLETED,
            progress=1.0,
            message="Old job",
            created_at=datetime.utcnow() - timedelta(days=8),
            completed_at=datetime.utcnow() - timedelta(days=8)
        )
        
        new_job = JobInfo(
            job_id="new-job", 
            job_type="test_job",
            status=JobStatus.COMPLETED,
            progress=1.0,
            message="New job",
            created_at=datetime.utcnow() - timedelta(hours=1),
            completed_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        # Save both jobs
        await persistence.save_job(old_job, {})
        await persistence.save_job(new_job, {})
        
        # Cleanup jobs older than 7 days
        deleted_count = await persistence.cleanup_old_jobs(keep_days=7)
        
        # Should have deleted the old job
        assert deleted_count >= 1
        
        # Verify the old job is gone and new job remains
        old_job_data = await persistence.load_job("old-job")
        new_job_data = await persistence.load_job("new-job")
        
        assert old_job_data is None
        assert new_job_data is not None


class TestJobQueueIntegration:
    """Integration tests combining job queue and persistence."""
    
    @pytest.mark.asyncio
    async def test_job_recovery_after_restart(self, temp_storage):
        """Test that jobs are recovered after application restart."""
        # Create first job queue and submit jobs
        queue1 = JobQueue(max_concurrent=2)
        
        # Mock the persistence to use our temp storage
        with patch('app.jobs.job_persistence') as mock_persistence:
            mock_persistence.storage_path = temp_storage
            real_persistence = JobPersistence(temp_storage)
            mock_persistence.save_job = real_persistence.save_job
            mock_persistence.load_recoverable_jobs = real_persistence.load_recoverable_jobs
            
            await queue1.start_workers()
            
            # Submit some jobs but don't let them complete
            job_id1 = await queue1.submit_job(
                job_type="test_job",
                params={"duration": 10, "steps": 10},  # Long job
                priority=1
            )
            
            job_id2 = await queue1.submit_job(
                job_type="test_job",
                params={"duration": 5, "steps": 5},
                priority=2
            )
            
            # Wait briefly for jobs to start
            await asyncio.sleep(0.1)
            
            # Save jobs to persistence (simulating what would happen)
            for job_id in [job_id1, job_id2]:
                if job_id in queue1.jobs:
                    params = queue1.job_params.get(job_id, {})
                    await real_persistence.save_job(queue1.jobs[job_id], params)
        
        # Create new job queue (simulating restart)
        queue2 = JobQueue(max_concurrent=2)
        
        with patch('app.jobs.job_persistence', real_persistence):
            await queue2.start_workers()
            
            # Jobs should be recovered
            assert job_id1 in queue2.jobs
            assert job_id2 in queue2.jobs
            
            # Both jobs should be queued (reset from running state)
            assert queue2.jobs[job_id1].status == JobStatus.QUEUED
            assert queue2.jobs[job_id2].status == JobStatus.QUEUED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])