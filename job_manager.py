#!/usr/bin/env python3
"""
Job Manager - Handles background video processing jobs with persistence
Simple threading-based approach without Celery/Redis
"""

import json
import uuid
import threading
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job:
    """Represents a video processing job"""
    
    def __init__(self, job_id: str, video_source: str, options: Dict[str, Any]):
        self.id = job_id
        self.video_source = video_source
        self.options = options
        self.status = JobStatus.PENDING
        self.progress = 0
        self.current_step = ""
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.cancel_event = threading.Event()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'video_source': self.video_source,
            'options': self.options,
            'status': self.status.value,
            'progress': self.progress,
            'current_step': self.current_step,
            'result': self.result,
            'error': str(self.error) if self.error else None,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create job from dictionary"""
        job = cls(
            job_id=data['id'],
            video_source=data['video_source'],
            options=data['options']
        )
        job.status = JobStatus(data['status'])
        job.progress = data['progress']
        job.current_step = data['current_step']
        job.result = data['result']
        job.error = data['error']
        job.created_at = datetime.fromisoformat(data['created_at'])
        job.started_at = datetime.fromisoformat(data['started_at']) if data['started_at'] else None
        job.completed_at = datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None
        return job


class JobManager:
    """
    Manages video processing jobs with persistence
    Uses threading for background processing and JSON files for persistence
    """
    
    def __init__(self, jobs_dir: str = "jobs"):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(exist_ok=True)
        self.active_jobs: Dict[str, Job] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        
        # Load existing jobs from disk
        self._load_jobs()
    
    def _load_jobs(self):
        """Load jobs from disk on startup"""
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    data = json.load(f)
                job = Job.from_dict(data)
                
                # Only load jobs that are not completed/failed/cancelled
                if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
                    # Reset processing jobs to pending (they were interrupted)
                    if job.status == JobStatus.PROCESSING:
                        job.status = JobStatus.PENDING
                        job.current_step = "Interrupted - ready to restart"
                    self.active_jobs[job.id] = job
                    logger.info(f"Loaded job {job.id} with status {job.status.value}")
            except Exception as e:
                logger.error(f"Error loading job from {job_file}: {e}")
    
    def _save_job(self, job: Job):
        """Save job to disk"""
        job_file = self.jobs_dir / f"{job.id}.json"
        try:
            with open(job_file, 'w') as f:
                json.dump(job.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving job {job.id}: {e}")
    
    def create_job(self, video_source: str, options: Dict[str, Any]) -> str:
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        job = Job(job_id, video_source, options)
        
        with self._lock:
            self.active_jobs[job_id] = job
            self._save_job(job)
        
        logger.info(f"Created job {job_id} for {video_source}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self._lock:
            return self.active_jobs.get(job_id)
    
    def list_jobs(self, limit: int = 50) -> list[Job]:
        """List all jobs, most recent first"""
        # Load all jobs from disk (including completed ones)
        all_jobs = []
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    data = json.load(f)
                job = Job.from_dict(data)
                all_jobs.append(job)
            except Exception as e:
                logger.error(f"Error loading job from {job_file}: {e}")
        
        # Sort by created_at descending
        all_jobs.sort(key=lambda j: j.created_at, reverse=True)
        return all_jobs[:limit]
    
    def start_job(self, job_id: str, worker_func: Callable):
        """Start processing a job in background thread"""
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        if job.status != JobStatus.PENDING:
            logger.warning(f"Job {job_id} is not pending (status: {job.status.value})")
            return
        
        # Update job status
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()
        self._save_job(job)
        
        # Create progress callback
        def progress_callback(status: str, progress: float):
            if job.cancel_event.is_set():
                raise Exception("Job cancelled by user")
            job.current_step = status
            job.progress = int(progress)
            self._save_job(job)
        
        # Worker wrapper
        def worker():
            try:
                logger.info(f"Starting job {job_id}")
                result = worker_func(job, progress_callback)
                
                # Job completed successfully
                job.status = JobStatus.COMPLETED
                job.result = result
                job.completed_at = datetime.now()
                job.progress = 100
                logger.info(f"Job {job_id} completed successfully")
                
            except Exception as e:
                # Job failed
                logger.error(f"Job {job_id} failed: {e}")
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()
            
            finally:
                self._save_job(job)
                # Clean up thread reference
                with self._lock:
                    if job_id in self.threads:
                        del self.threads[job_id]
        
        # Start thread
        thread = threading.Thread(target=worker, daemon=True)
        with self._lock:
            self.threads[job_id] = thread
        thread.start()
        
        logger.info(f"Job {job_id} started in background thread")
    
    def cancel_job(self, job_id: str):
        """Cancel a running job"""
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        if job.status != JobStatus.PROCESSING:
            logger.warning(f"Job {job_id} is not processing (status: {job.status.value})")
            return
        
        # Set cancel event
        job.cancel_event.set()
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        self._save_job(job)
        
        logger.info(f"Job {job_id} cancelled")
    
    def delete_job(self, job_id: str):
        """Delete a job and its data"""
        job_file = self.jobs_dir / f"{job_id}.json"
        if job_file.exists():
            job_file.unlink()
        
        with self._lock:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            if job_id in self.threads:
                del self.threads[job_id]
        
        logger.info(f"Job {job_id} deleted")
    
    def cleanup_old_jobs(self, days: int = 7):
        """Delete jobs older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    data = json.load(f)
                created_at = datetime.fromisoformat(data['created_at'])
                
                if created_at.timestamp() < cutoff:
                    job_file.unlink()
                    logger.info(f"Deleted old job {data['id']}")
            except Exception as e:
                logger.error(f"Error cleaning up {job_file}: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get job statistics"""
        all_jobs = self.list_jobs(limit=1000)
        
        stats = {
            'total': len(all_jobs),
            'pending': sum(1 for j in all_jobs if j.status == JobStatus.PENDING),
            'processing': sum(1 for j in all_jobs if j.status == JobStatus.PROCESSING),
            'completed': sum(1 for j in all_jobs if j.status == JobStatus.COMPLETED),
            'failed': sum(1 for j in all_jobs if j.status == JobStatus.FAILED),
            'cancelled': sum(1 for j in all_jobs if j.status == JobStatus.CANCELLED),
        }
        
        return stats


# Global job manager instance
_job_manager = None

def get_job_manager() -> JobManager:
    """Get or create global job manager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
