"""
Integration tests for the enhanced job queue system.

Tests end-to-end job workflows including API integration,
new job types, and real system interactions.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.jobs import job_queue
from app.job_persistence import job_persistence
from app.matters import matter_manager
from app.models import Matter, MatterPaths


class TestJobQueueIntegration:
    """Integration tests for job queue with real dependencies."""
    
    @pytest.fixture
    async def setup_job_queue(self):
        """Set up job queue for testing."""
        # Ensure job queue is started
        await job_queue.start_workers()
        
        # Clean up any existing jobs
        await job_queue.cleanup_completed_jobs(0)
        
        yield job_queue
        
        # Cleanup after tests
        await job_queue.cleanup_completed_jobs(0)
    
    @pytest.fixture
    def sample_matter(self):
        """Create a sample matter for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            matter_path = Path(temp_dir) / "test_matter"
            matter_path.mkdir(parents=True, exist_ok=True)
            
            # Create matter paths structure
            paths = MatterPaths(
                root=matter_path,
                docs=matter_path / "docs",
                docs_ocr=matter_path / "docs_ocr",
                parsed=matter_path / "parsed",
                vectors=matter_path / "vectors",
                knowledge=matter_path / "knowledge",
                chat=matter_path / "chat",
                logs=matter_path / "logs"
            )
            
            # Create directories
            for path in [paths.docs, paths.docs_ocr, paths.parsed, 
                        paths.vectors, paths.knowledge, paths.chat, paths.logs]:
                path.mkdir(parents=True, exist_ok=True)
            
            matter = Matter(
                id="test-matter-123",
                name="Test Construction Claim",
                slug="test-construction-claim",
                created_at=None,  # Will be set by matter manager
                embedding_model="nomic-embed-text",
                generation_model="gpt-oss:20b",
                paths=paths
            )
            
            yield matter
    
    @pytest.mark.asyncio
    async def test_test_job_execution(self, setup_job_queue):
        """Test that test jobs execute correctly."""
        job_id = await setup_job_queue.submit_job(
            job_type="test_job",
            params={"duration": 0.2, "steps": 4},
            priority=0
        )
        
        # Wait for job to complete
        for _ in range(30):
            await asyncio.sleep(0.1)
            job_info = await setup_job_queue.get_job_status(job_id)
            if job_info.status.value in ["completed", "failed"]:
                break
        
        job_info = await setup_job_queue.get_job_status(job_id)
        assert job_info.status.value == "completed"
        assert job_info.progress == 1.0
        assert "completed with 4 steps" in job_info.result
    
    @pytest.mark.asyncio 
    async def test_batch_document_processing_job_structure(self, setup_job_queue, sample_matter):
        """Test batch document processing job structure (mocked implementation)."""
        # Mock the batch processing to avoid file system dependencies
        with patch('app.job_types.execute_batch_document_processing') as mock_execute:
            mock_execute.return_value = {
                "total_files": 3,
                "successful_files": 3,
                "failed_files": 0,
                "batch_count": 2,
                "batch_results": []
            }
            
            # Submit batch processing job
            file_batches = [
                {
                    "matter_id": sample_matter.id,
                    "files": ["/tmp/doc1.pdf", "/tmp/doc2.pdf"]
                },
                {
                    "matter_id": sample_matter.id,
                    "files": ["/tmp/doc3.pdf"]
                }
            ]
            
            job_id = await setup_job_queue.submit_job(
                job_type="batch_document_processing",
                params={
                    "file_batches": file_batches,
                    "force_ocr": False,
                    "ocr_language": "eng",
                    "max_concurrent_files": 2
                },
                priority=1
            )
            
            # Wait for job to complete
            for _ in range(30):
                await asyncio.sleep(0.1)
                job_info = await setup_job_queue.get_job_status(job_id)
                if job_info.status.value in ["completed", "failed"]:
                    break
            
            job_info = await setup_job_queue.get_job_status(job_id)
            assert job_info.status.value == "completed"
            assert job_info.result["total_files"] == 3
            assert job_info.result["successful_files"] == 3
    
    @pytest.mark.asyncio
    async def test_large_model_operation_job_structure(self, setup_job_queue, sample_matter):
        """Test large model operation job structure (mocked implementation)."""
        # Mock the large model operation
        with patch('app.job_types.execute_large_model_operation') as mock_execute:
            mock_execute.return_value = {
                "operation_type": "bulk_re_embedding",
                "chunks_processed": 150,
                "embedding_model": "mxbai-embed-large",
                "matter_id": sample_matter.id,
                "completed_at": "2025-01-21T10:00:00"
            }
            
            job_id = await setup_job_queue.submit_job(
                job_type="large_model_operation",
                params={
                    "operation_type": "bulk_re_embedding",
                    "matter_id": sample_matter.id,
                    "new_embedding_model": "mxbai-embed-large"
                },
                priority=2
            )
            
            # Wait for job to complete
            for _ in range(30):
                await asyncio.sleep(0.1)
                job_info = await setup_job_queue.get_job_status(job_id)
                if job_info.status.value in ["completed", "failed"]:
                    break
            
            job_info = await setup_job_queue.get_job_status(job_id)
            assert job_info.status.value == "completed"
            assert job_info.result["operation_type"] == "bulk_re_embedding"
            assert job_info.result["chunks_processed"] == 150
    
    @pytest.mark.asyncio
    async def test_matter_analysis_job_structure(self, setup_job_queue, sample_matter):
        """Test matter analysis job structure (mocked implementation)."""
        # Mock the matter analysis
        with patch('app.job_types.execute_matter_analysis') as mock_execute:
            mock_execute.return_value = {
                "matter_id": sample_matter.id,
                "matter_name": sample_matter.name,
                "analysis_types": ["overview", "timeline"],
                "total_documents": 5,
                "results": {
                    "overview": {
                        "summary": "Construction dispute involving delays and cost overruns.",
                        "sources_count": 12
                    },
                    "timeline": {
                        "narrative": "Project started in Q1 2023...",
                        "events": []
                    }
                }
            }
            
            job_id = await setup_job_queue.submit_job(
                job_type="matter_analysis",
                params={
                    "matter_id": sample_matter.id,
                    "analysis_types": ["overview", "timeline"]
                },
                priority=1
            )
            
            # Wait for job to complete
            for _ in range(30):
                await asyncio.sleep(0.1)
                job_info = await setup_job_queue.get_job_status(job_id)
                if job_info.status.value in ["completed", "failed"]:
                    break
            
            job_info = await setup_job_queue.get_job_status(job_id)
            assert job_info.status.value == "completed"
            assert job_info.result["matter_id"] == sample_matter.id
            assert "overview" in job_info.result["results"]
    
    @pytest.mark.asyncio
    async def test_job_queue_status_reporting(self, setup_job_queue):
        """Test job queue status reporting functionality."""
        # Submit several jobs with different priorities
        job_ids = []
        
        # High priority job
        job_ids.append(await setup_job_queue.submit_job(
            job_type="test_job",
            params={"duration": 0.5, "steps": 3},
            priority=3
        ))
        
        # Medium priority jobs
        for i in range(2):
            job_ids.append(await setup_job_queue.submit_job(
                job_type="test_job",
                params={"duration": 0.3, "steps": 2},
                priority=1
            ))
        
        # Low priority job
        job_ids.append(await setup_job_queue.submit_job(
            job_type="test_job",
            params={"duration": 0.2, "steps": 1},
            priority=0
        ))
        
        # Check queue status
        await asyncio.sleep(0.1)  # Let some jobs start
        
        running_jobs = await setup_job_queue.get_running_jobs()
        queued_jobs = await setup_job_queue.get_queued_jobs()
        
        assert len(running_jobs) <= setup_job_queue.max_concurrent
        assert len(running_jobs) + len(queued_jobs) >= 2
        
        # Get all jobs
        all_jobs = await setup_job_queue.get_all_jobs()
        assert len(all_jobs) == 4
        
        # Wait for all to complete
        await asyncio.sleep(2.0)
        
        # Check final status
        completed_jobs = []
        for job_id in job_ids:
            job_info = await setup_job_queue.get_job_status(job_id)
            if job_info.status.value == "completed":
                completed_jobs.append(job_info)
        
        assert len(completed_jobs) == 4
    
    @pytest.mark.asyncio
    async def test_job_progress_tracking_integration(self, setup_job_queue):
        """Test that progress tracking works end-to-end."""
        job_id = await setup_job_queue.submit_job(
            job_type="test_job",
            params={"duration": 0.4, "steps": 8},
            priority=0
        )
        
        # Track progress over time
        progress_snapshots = []
        
        for _ in range(20):
            await asyncio.sleep(0.05)
            job_info = await setup_job_queue.get_job_status(job_id)
            progress_snapshots.append({
                "status": job_info.status.value,
                "progress": job_info.progress,
                "message": job_info.message,
                "eta_available": hasattr(job_info, 'eta_seconds')
            })
            
            if job_info.status.value == "completed":
                break
        
        # Analyze progress snapshots
        running_snapshots = [s for s in progress_snapshots if s["status"] == "running"]
        
        if len(running_snapshots) >= 2:
            # Progress should increase over time
            first_progress = running_snapshots[0]["progress"]
            last_progress = running_snapshots[-1]["progress"]
            assert last_progress > first_progress
        
        # Final job should be completed
        final_job = await setup_job_queue.get_job_status(job_id)
        assert final_job.status.value == "completed"
        assert final_job.progress == 1.0
    
    @pytest.mark.asyncio 
    async def test_concurrent_job_execution_integration(self, setup_job_queue):
        """Test that the job queue handles concurrent execution properly."""
        # Submit more jobs than max_concurrent
        job_ids = []
        start_times = {}
        
        for i in range(5):  # More than max_concurrent (usually 2-3)
            job_id = await setup_job_queue.submit_job(
                job_type="test_job",
                params={"duration": 0.3, "steps": 2},
                priority=0
            )
            job_ids.append(job_id)
            start_times[job_id] = None
        
        # Monitor job starts
        for _ in range(15):  # 1.5 seconds monitoring
            await asyncio.sleep(0.1)
            
            for job_id in job_ids:
                job_info = await setup_job_queue.get_job_status(job_id)
                if (job_info.status.value == "running" and 
                    start_times[job_id] is None):
                    start_times[job_id] = job_info.started_at
        
        # Wait for all jobs to complete
        await asyncio.sleep(2.0)
        
        # Check that jobs were run concurrently
        running_jobs = await setup_job_queue.get_running_jobs()
        assert len(running_jobs) == 0  # All should be done
        
        completed_count = 0
        for job_id in job_ids:
            job_info = await setup_job_queue.get_job_status(job_id)
            if job_info.status.value == "completed":
                completed_count += 1
        
        assert completed_count == 5
        
        # Check that not all jobs started at the same time (concurrency limit)
        actual_start_times = [t for t in start_times.values() if t is not None]
        if len(actual_start_times) >= 3:
            # There should be some time difference between starts
            time_diffs = []
            sorted_times = sorted(actual_start_times)
            for i in range(1, len(sorted_times)):
                diff = (sorted_times[i] - sorted_times[0]).total_seconds()
                time_diffs.append(diff)
            
            # At least some jobs should have started at different times
            assert max(time_diffs) > 0.05  # At least 50ms difference
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, setup_job_queue):
        """Test error handling in realistic scenarios."""
        # Test with unknown job type
        with pytest.raises(Exception):
            await setup_job_queue.submit_job(
                job_type="nonexistent_job_type",
                params={},
                priority=0
            )
        
        # Test job that fails during execution
        with patch('app.jobs.JobQueue._execute_test_job') as mock_test_job:
            mock_test_job.side_effect = Exception("Simulated job failure")
            
            job_id = await setup_job_queue.submit_job(
                job_type="test_job",
                params={"duration": 0.1},
                max_retries=1
            )
            
            # Wait for job to fail
            for _ in range(20):
                await asyncio.sleep(0.1)
                job_info = await setup_job_queue.get_job_status(job_id)
                if job_info.status.value == "failed":
                    break
            
            job_info = await setup_job_queue.get_job_status(job_id)
            assert job_info.status.value == "failed"
            assert "Simulated job failure" in job_info.error_message
    
    @pytest.mark.asyncio
    async def test_job_persistence_integration(self, setup_job_queue):
        """Test that job persistence works with the queue."""
        # Submit a job
        job_id = await setup_job_queue.submit_job(
            job_type="test_job",
            params={"duration": 0.1, "steps": 2},
            priority=1
        )
        
        # Wait for job to complete
        for _ in range(20):
            await asyncio.sleep(0.1)
            job_info = await setup_job_queue.get_job_status(job_id)
            if job_info.status.value == "completed":
                break
        
        # Check that job was persisted
        job_history = await setup_job_queue.get_job_history(10)
        job_ids_in_history = [job.job_id for job in job_history]
        
        assert job_id in job_ids_in_history
        
        # Test status filtering
        completed_history = await setup_job_queue.get_job_history(10, "completed")
        assert len(completed_history) >= 1
        assert all(job.status.value == "completed" for job in completed_history)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])