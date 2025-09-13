"""
Advanced ASR Scheduler for Ensemble Transcription
U7 Upgrade: Intelligent scheduling with concurrency control and prioritization
"""

import asyncio
import time
import queue
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import logging
from pathlib import Path
import json

from utils.bounded_executor import BoundedThreadPoolExecutor
from utils.intelligent_cache import get_cache_manager
from utils.deterministic_processing import get_deterministic_processor
from utils.enhanced_structured_logger import create_enhanced_logger

# Configure logging
scheduler_logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ASRTask:
    """Represents an ASR processing task"""
    task_id: str
    audio_file_path: str
    asr_config: Dict[str, Any]
    priority: TaskPriority
    estimated_duration: float  # seconds
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Task lifecycle
    creation_time: float = field(default_factory=time.time)
    queue_time: Optional[float] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

@dataclass
class SchedulerStats:
    """Scheduler statistics"""
    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_tasks_cancelled: int = 0
    active_tasks: int = 0
    queued_tasks: int = 0
    average_queue_time: float = 0.0
    average_processing_time: float = 0.0
    throughput_tasks_per_minute: float = 0.0
    cache_hit_rate: float = 0.0

class AdvancedASRScheduler:
    """
    Advanced ASR scheduler with intelligent prioritization and concurrency control.
    Optimizes resource utilization and minimizes perceived latency.
    """
    
    def __init__(self, max_concurrent_workers: int = 3,
                 enable_caching: bool = True,
                 enable_prioritization: bool = True,
                 queue_size_limit: int = 100):
        """
        Initialize advanced ASR scheduler.
        
        Args:
            max_concurrent_workers: Maximum concurrent ASR workers
            enable_caching: Whether to use intelligent caching
            enable_prioritization: Whether to enable task prioritization
            queue_size_limit: Maximum queue size before backpressure
        """
        self.max_concurrent_workers = max_concurrent_workers
        self.enable_caching = enable_caching
        self.enable_prioritization = enable_prioritization
        self.queue_size_limit = queue_size_limit
        
        # Task management
        self.task_queue = queue.PriorityQueue(maxsize=queue_size_limit)
        self.active_tasks: Dict[str, ASRTask] = {}
        self.completed_tasks: Dict[str, ASRTask] = {}
        self.task_history: List[ASRTask] = []
        
        # Worker management
        self.executor = BoundedThreadPoolExecutor(
            max_workers=max_concurrent_workers,
            queue_size=queue_size_limit,
            thread_name_prefix="asr_worker"
        )
        self.worker_futures: Dict[str, Future] = {}
        
        # Scheduler control
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Caching and deterministic processing
        self.cache_manager = get_cache_manager() if enable_caching else None
        self.deterministic_processor = get_deterministic_processor()
        
        # Statistics and monitoring
        self.stats = SchedulerStats()
        self.stats_lock = threading.Lock()
        
        # Enhanced logging
        self.logger = create_enhanced_logger("asr_scheduler")
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.last_throughput_calculation = time.time()
        
        scheduler_logger.info(f"Initialized ASR scheduler with {max_concurrent_workers} workers")
        scheduler_logger.info(f"Caching: {enable_caching}, Prioritization: {enable_prioritization}")
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            scheduler_logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="asr_scheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        
        scheduler_logger.info("ASR scheduler started")
    
    def stop(self, timeout: float = 30.0):
        """
        Stop the scheduler gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self.running:
            return
        
        scheduler_logger.info("Stopping ASR scheduler...")
        
        # Signal shutdown
        self.running = False
        self.shutdown_event.set()
        
        # Wait for scheduler thread
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=timeout)
        
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=timeout)
        
        scheduler_logger.info("ASR scheduler stopped")
    
    def submit_task(self, audio_file_path: str, asr_config: Dict[str, Any],
                   priority: TaskPriority = TaskPriority.NORMAL,
                   estimated_duration: Optional[float] = None,
                   callback: Optional[Callable] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit an ASR task to the scheduler.
        
        Args:
            audio_file_path: Path to audio file
            asr_config: ASR configuration parameters
            priority: Task priority
            estimated_duration: Estimated processing duration in seconds
            callback: Optional completion callback
            metadata: Optional task metadata
            
        Returns:
            Task ID
        """
        if not self.running:
            raise RuntimeError("Scheduler is not running")
        
        # Generate task ID
        task_id = self._generate_task_id(audio_file_path, asr_config)
        
        # Estimate duration if not provided
        if estimated_duration is None:
            estimated_duration = self._estimate_processing_duration(audio_file_path)
        
        # Create task
        task = ASRTask(
            task_id=task_id,
            audio_file_path=audio_file_path,
            asr_config=asr_config,
            priority=priority,
            estimated_duration=estimated_duration,
            callback=callback,
            metadata=metadata or {}
        )
        
        # Check cache first if enabled
        if self.enable_caching and self.cache_manager:
            cached_result = self.cache_manager.get("asr_processing", audio_file_path, asr_config)
            if cached_result is not None:
                # Cache hit - complete task immediately
                task.result = cached_result
                task.status = TaskStatus.COMPLETED
                task.completion_time = time.time()
                task.processing_time = 0.0
                
                self.completed_tasks[task_id] = task
                
                # Execute callback if provided
                if callback:
                    try:
                        callback(task)
                    except Exception as e:
                        scheduler_logger.error(f"Error in task callback: {e}")
                
                # Update stats
                with self.stats_lock:
                    self.stats.total_tasks_completed += 1
                    self.stats.cache_hit_rate = self._calculate_cache_hit_rate()
                
                scheduler_logger.debug(f"Task {task_id} completed from cache")
                return task_id
        
        # Queue task for processing
        try:
            # Use priority for queue ordering (lower number = higher priority)
            priority_value = priority.value
            
            # Adjust priority based on estimated duration (shorter tasks get slight boost)
            if self.enable_prioritization and estimated_duration < 5.0:
                priority_value -= 0.1  # Slight priority boost for short tasks
            
            self.task_queue.put((priority_value, time.time(), task), timeout=1.0)
            task.status = TaskStatus.QUEUED
            task.queue_time = time.time()
            
            # Update stats
            with self.stats_lock:
                self.stats.total_tasks_submitted += 1
                self.stats.queued_tasks += 1
            
            scheduler_logger.debug(f"Queued task {task_id} with priority {priority.name}")
            return task_id
            
        except queue.Full:
            task.status = TaskStatus.FAILED
            task.error = "Queue is full - backpressure activated"
            
            scheduler_logger.warning(f"Task {task_id} rejected - queue full")
            raise RuntimeError("ASR scheduler queue is full")
    
    def get_task_status(self, task_id: str) -> Optional[ASRTask]:
        """Get current status of a task."""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Check queue (expensive operation)
        with self.task_queue.mutex:
            for priority, timestamp, task in self.task_queue.queue:
                if task.task_id == task_id:
                    return task
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued or running task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if successfully cancelled, False otherwise
        """
        # Check if task is running
        if task_id in self.active_tasks:
            if task_id in self.worker_futures:
                future = self.worker_futures[task_id]
                if future.cancel():
                    task = self.active_tasks[task_id]
                    task.status = TaskStatus.CANCELLED
                    task.completion_time = time.time()
                    
                    # Move to completed
                    self.completed_tasks[task_id] = task
                    del self.active_tasks[task_id]
                    del self.worker_futures[task_id]
                    
                    with self.stats_lock:
                        self.stats.total_tasks_cancelled += 1
                        self.stats.active_tasks -= 1
                    
                    scheduler_logger.info(f"Cancelled running task {task_id}")
                    return True
        
        # Check if task is queued (remove from queue)
        # Note: This is expensive for PriorityQueue, would need custom implementation for efficiency
        temp_items = []
        cancelled = False
        
        try:
            while not self.task_queue.empty():
                priority, timestamp, task = self.task_queue.get_nowait()
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    task.completion_time = time.time()
                    self.completed_tasks[task_id] = task
                    cancelled = True
                    
                    with self.stats_lock:
                        self.stats.total_tasks_cancelled += 1
                        self.stats.queued_tasks -= 1
                    
                    scheduler_logger.info(f"Cancelled queued task {task_id}")
                else:
                    temp_items.append((priority, timestamp, task))
        except queue.Empty:
            pass
        
        # Re-queue remaining items
        for item in temp_items:
            try:
                self.task_queue.put_nowait(item)
            except queue.Full:
                # This shouldn't happen but handle gracefully
                scheduler_logger.error("Failed to re-queue task during cancellation")
        
        return cancelled
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self.stats_lock:
            return {
                'queued_tasks': self.stats.queued_tasks,
                'active_tasks': self.stats.active_tasks,
                'queue_size_limit': self.queue_size_limit,
                'queue_utilization': self.stats.queued_tasks / self.queue_size_limit,
                'worker_utilization': self.stats.active_tasks / self.max_concurrent_workers,
                'total_submitted': self.stats.total_tasks_submitted,
                'total_completed': self.stats.total_tasks_completed,
                'total_failed': self.stats.total_tasks_failed,
                'average_queue_time': self.stats.average_queue_time,
                'average_processing_time': self.stats.average_processing_time,
                'throughput': self.stats.throughput_tasks_per_minute,
                'cache_hit_rate': self.stats.cache_hit_rate
            }
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        scheduler_logger.info("Scheduler loop started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get next task from queue
                try:
                    priority, timestamp, task = self.task_queue.get(timeout=1.0)
                    
                    with self.stats_lock:
                        self.stats.queued_tasks -= 1
                    
                except queue.Empty:
                    continue
                
                # Submit task to executor
                future = self.executor.submit(self._process_task, task)
                
                # Track active task
                self.active_tasks[task.task_id] = task
                self.worker_futures[task.task_id] = future
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()
                
                with self.stats_lock:
                    self.stats.active_tasks += 1
                
                # Add completion callback
                future.add_done_callback(lambda f, tid=task.task_id: self._task_completed(tid, f))
                
                scheduler_logger.debug(f"Started processing task {task.task_id}")
                
            except Exception as e:
                scheduler_logger.error(f"Error in scheduler loop: {e}")
        
        scheduler_logger.info("Scheduler loop ended")
    
    def _process_task(self, task: ASRTask) -> Any:
        """
        Process a single ASR task.
        
        Args:
            task: Task to process
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            # Set deterministic seed for reproducible results
            self.deterministic_processor.set_deterministic_seed(
                "asr_processing",
                {
                    'task_id': task.task_id,
                    'audio_file': task.audio_file_path,
                    'config': task.asr_config
                }
            )
            
            # Import ASR engine here to avoid circular imports
            from core.asr_engine import ASREngine
            asr_engine = ASREngine()
            
            # Process with ASR engine
            result = asr_engine._make_transcription_api_call(
                task.audio_file_path,
                **task.asr_config
            )
            
            # Cache result if caching is enabled
            if self.enable_caching and self.cache_manager:
                self.cache_manager.set(
                    "asr_processing",
                    result,
                    task.audio_file_path,
                    task.asr_config
                )
            
            task.result = result
            task.processing_time = time.time() - start_time
            
            scheduler_logger.debug(f"Task {task.task_id} processed in {task.processing_time:.2f}s")
            return result
            
        except Exception as e:
            task.error = str(e)
            task.processing_time = time.time() - start_time
            scheduler_logger.error(f"Task {task.task_id} failed: {e}")
            raise
    
    def _task_completed(self, task_id: str, future: Future):
        """Handle task completion."""
        try:
            if task_id not in self.active_tasks:
                return
            
            task = self.active_tasks[task_id]
            task.completion_time = time.time()
            
            # Update task status based on future result
            try:
                future.result()  # This will raise exception if task failed
                task.status = TaskStatus.COMPLETED
                
                with self.stats_lock:
                    self.stats.total_tasks_completed += 1
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                
                with self.stats_lock:
                    self.stats.total_tasks_failed += 1
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            if task_id in self.worker_futures:
                del self.worker_futures[task_id]
            
            with self.stats_lock:
                self.stats.active_tasks -= 1
            
            # Execute callback if provided
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    scheduler_logger.error(f"Error in task callback: {e}")
            
            # Update statistics
            self._update_statistics(task)
            
            # Add to history (keep limited history)
            self.task_history.append(task)
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-500:]  # Keep last 500
            
            scheduler_logger.debug(f"Task {task_id} completed with status {task.status.value}")
            
        except Exception as e:
            scheduler_logger.error(f"Error in task completion handler: {e}")
    
    def _generate_task_id(self, audio_file_path: str, asr_config: Dict[str, Any]) -> str:
        """Generate unique task ID."""
        import hashlib
        
        # Create deterministic task ID
        components = [
            str(Path(audio_file_path).name),
            str(asr_config),
            str(time.time())
        ]
        
        combined = "_".join(components)
        task_hash = hashlib.md5(combined.encode()).hexdigest()[:12]
        
        return f"asr_task_{task_hash}"
    
    def _estimate_processing_duration(self, audio_file_path: str) -> float:
        """Estimate processing duration based on audio file."""
        try:
            import librosa
            duration = librosa.get_duration(path=audio_file_path)
            
            # Rough estimate: ASR takes about 10% of audio duration
            estimated = duration * 0.1
            
            # Add base overhead
            estimated += 2.0
            
            # Use historical data if available
            if self.performance_history:
                avg_ratio = sum(h.get('processing_ratio', 0.1) for h in self.performance_history[-10:]) / len(self.performance_history[-10:])
                estimated = duration * avg_ratio + 2.0
            
            return max(1.0, estimated)
            
        except Exception:
            return 10.0  # Default estimate
    
    def _update_statistics(self, task: ASRTask):
        """Update scheduler statistics."""
        with self.stats_lock:
            # Queue time
            if task.queue_time and task.start_time:
                queue_time = task.start_time - task.queue_time
                # Running average
                if self.stats.average_queue_time == 0:
                    self.stats.average_queue_time = queue_time
                else:
                    self.stats.average_queue_time = (self.stats.average_queue_time * 0.9) + (queue_time * 0.1)
            
            # Processing time
            if task.processing_time:
                if self.stats.average_processing_time == 0:
                    self.stats.average_processing_time = task.processing_time
                else:
                    self.stats.average_processing_time = (self.stats.average_processing_time * 0.9) + (task.processing_time * 0.1)
            
            # Throughput calculation
            current_time = time.time()
            if current_time - self.last_throughput_calculation > 60:  # Update every minute
                completed_in_period = len([
                    t for t in self.task_history 
                    if t.completion_time and (current_time - t.completion_time) < 60
                ])
                self.stats.throughput_tasks_per_minute = completed_in_period
                self.last_throughput_calculation = current_time
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.stats.total_tasks_submitted == 0:
            return 0.0
        
        cache_hits = self.stats.total_tasks_completed - len(self.task_history)
        return cache_hits / self.stats.total_tasks_submitted
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self.stats_lock:
            stats_copy = {
                'total_submitted': self.stats.total_tasks_submitted,
                'total_completed': self.stats.total_tasks_completed,
                'total_failed': self.stats.total_tasks_failed,
                'total_cancelled': self.stats.total_tasks_cancelled,
                'active_tasks': self.stats.active_tasks,
                'queued_tasks': self.stats.queued_tasks,
                'average_queue_time': self.stats.average_queue_time,
                'average_processing_time': self.stats.average_processing_time,
                'throughput_per_minute': self.stats.throughput_tasks_per_minute,
                'cache_hit_rate': self.stats.cache_hit_rate
            }
        
        # Add additional metrics
        stats_copy.update({
            'scheduler_running': self.running,
            'max_workers': self.max_concurrent_workers,
            'queue_size_limit': self.queue_size_limit,
            'worker_utilization': self.stats.active_tasks / self.max_concurrent_workers,
            'queue_utilization': self.stats.queued_tasks / self.queue_size_limit,
            'success_rate': (self.stats.total_tasks_completed / max(1, self.stats.total_tasks_submitted)) * 100,
            'tasks_in_history': len(self.task_history)
        })
        
        return stats_copy


# Global ASR scheduler instance
_asr_scheduler: Optional[AdvancedASRScheduler] = None

def get_asr_scheduler() -> AdvancedASRScheduler:
    """Get or create global ASR scheduler instance."""
    global _asr_scheduler
    if _asr_scheduler is None:
        _asr_scheduler = AdvancedASRScheduler()
        _asr_scheduler.start()
    return _asr_scheduler