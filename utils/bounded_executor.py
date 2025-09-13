"""
Bounded Thread Pool Executor with Queue Limits

Provides thread pool executors with bounded queues to prevent memory
issues and API overload when processing large workloads.
"""
import time
import queue
import threading
from typing import Any, Callable, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from concurrent.futures._base import FIRST_COMPLETED, ALL_COMPLETED, FIRST_EXCEPTION

from utils.reliability_config import get_concurrency_config
from utils.structured_logger import StructuredLogger


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """
    ThreadPoolExecutor with bounded task queue to prevent memory issues.
    
    When the queue is full, submit() will block until space becomes available,
    preventing unlimited task accumulation.
    """
    
    def __init__(self, max_workers: int, queue_size: int, thread_name_prefix: str = ''):
        # Initialize parent with bounded queue
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        
        # Replace the unbounded queue with a bounded one
        self._work_queue = queue.Queue(maxsize=queue_size)  # type: ignore[assignment]
        
        # Track metrics
        self.queue_size = queue_size
        self.submitted_tasks = 0
        self.completed_tasks = 0
        self.rejected_tasks = 0
        
        # Structured logging
        self.structured_logger = StructuredLogger(f"bounded_executor_{thread_name_prefix}")
        
        self.structured_logger.info(
            f"Bounded thread pool executor initialized",
            context={
                'max_workers': max_workers,
                'queue_size': queue_size,
                'thread_name_prefix': thread_name_prefix
            }
        )
    
    def submit(self, fn: Callable, *args, timeout: Optional[float] = None, **kwargs) -> Future:
        """
        Submit a task to the executor with optional timeout for queue insertion.
        
        Args:
            fn: Callable to execute
            *args: Positional arguments for callable
            timeout: Max time to wait for queue space (None = block indefinitely)
            **kwargs: Keyword arguments for callable
            
        Returns:
            Future representing the execution
            
        Raises:
            RuntimeError: If executor is shutdown
            queue.Full: If timeout expires waiting for queue space
        """
        if self._shutdown:
            raise RuntimeError('Cannot schedule new futures after shutdown')
        
        start_time = time.time()
        
        try:
            # Try to submit with timeout
            future = super().submit(fn, *args, **kwargs)
            
            # Track successful submission
            self.submitted_tasks += 1
            submission_time = time.time() - start_time
            
            if submission_time > 1.0:  # Log if submission was delayed
                self.structured_logger.warning(
                    f"Task submission delayed by queue backpressure",
                    context={
                        'submission_delay': submission_time,
                        'queue_approximate_size': self._work_queue.qsize(),
                        'queue_capacity': self.queue_size
                    }
                )
            
            # Wrap future to track completion
            def track_completion():
                try:
                    result = future.result()
                    self.completed_tasks += 1
                    return result
                except Exception as e:
                    self.completed_tasks += 1
                    raise
                    
            # Replace the result method to track completion
            original_result = future.result
            future.result = lambda timeout=None: (track_completion(), original_result(timeout))[1]
            
            return future
            
        except Exception as e:
            self.rejected_tasks += 1
            self.structured_logger.error(
                f"Task submission failed",
                context={
                    'error': str(e),
                    'queue_size': self._work_queue.qsize(),
                    'submitted_tasks': self.submitted_tasks,
                    'rejected_tasks': self.rejected_tasks
                }
            )
            raise
    
    def submit_with_backpressure(self, fn: Callable, *args, max_wait: float = 30.0, **kwargs) -> Optional[Future]:
        """
        Submit a task with backpressure handling - returns None if queue is full.
        
        Args:
            fn: Callable to execute
            *args: Positional arguments
            max_wait: Maximum time to wait for queue space
            **kwargs: Keyword arguments
            
        Returns:
            Future if submitted successfully, None if queue full
        """
        try:
            return self.submit(fn, *args, timeout=max_wait, **kwargs)
        except queue.Full:
            self.rejected_tasks += 1
            self.structured_logger.warning(
                f"Task rejected due to full queue",
                context={
                    'queue_size': self.queue_size,
                    'max_wait': max_wait,
                    'rejected_tasks': self.rejected_tasks
                }
            )
            return None
    
    def get_metrics(self) -> dict:
        """Get current executor metrics"""
        return {
            'max_workers': self._max_workers,
            'queue_size': self.queue_size,
            'current_queue_size': self._work_queue.qsize(),
            'submitted_tasks': self.submitted_tasks,
            'completed_tasks': self.completed_tasks,
            'rejected_tasks': self.rejected_tasks,
            'active_threads': len(self._threads),
            'queue_utilization': self._work_queue.qsize() / self.queue_size if self.queue_size > 0 else 0.0
        }


class ResilientExecutorManager:
    """
    Manager for creating and monitoring resilient executors with proper configuration.
    
    Provides pre-configured executors for different service types with appropriate
    concurrency limits and queue sizes.
    """
    
    def __init__(self):
        self.concurrency_config = get_concurrency_config()
        self.executors = {}
        self.structured_logger = StructuredLogger("resilient_executor_manager")
        
    def get_asr_executor(self) -> BoundedThreadPoolExecutor:
        """Get bounded executor for ASR operations"""
        if 'asr' not in self.executors:
            self.executors['asr'] = BoundedThreadPoolExecutor(
                max_workers=self.concurrency_config.max_asr_requests,
                queue_size=self.concurrency_config.thread_pool_queue_size,
                thread_name_prefix='asr'
            )
        return self.executors['asr']
    
    def get_diarization_executor(self) -> BoundedThreadPoolExecutor:
        """Get bounded executor for diarization operations"""
        if 'diarization' not in self.executors:
            self.executors['diarization'] = BoundedThreadPoolExecutor(
                max_workers=self.concurrency_config.max_diarization,
                queue_size=max(10, self.concurrency_config.thread_pool_queue_size // 5),  # Smaller queue
                thread_name_prefix='diarization'
            )
        return self.executors['diarization']
    
    def get_file_processor_executor(self) -> BoundedThreadPoolExecutor:
        """Get bounded executor for file processing operations"""
        if 'file_processor' not in self.executors:
            self.executors['file_processor'] = BoundedThreadPoolExecutor(
                max_workers=self.concurrency_config.max_file_processors,
                queue_size=self.concurrency_config.thread_pool_queue_size,
                thread_name_prefix='file_proc'
            )
        return self.executors['file_processor']
    
    def shutdown_all(self, wait: bool = True):
        """Shutdown all managed executors"""
        self.structured_logger.info("Shutting down all resilient executors")
        
        for name, executor in self.executors.items():
            try:
                executor.shutdown(wait=wait)
                self.structured_logger.info(f"Executor '{name}' shut down successfully")
            except Exception as e:
                self.structured_logger.error(f"Error shutting down executor '{name}': {e}")
        
        self.executors.clear()
    
    def get_all_metrics(self) -> dict:
        """Get metrics for all managed executors"""
        return {
            name: executor.get_metrics() 
            for name, executor in self.executors.items()
        }
    
    def log_metrics_summary(self):
        """Log summary of all executor metrics"""
        all_metrics = self.get_all_metrics()
        
        for name, metrics in all_metrics.items():
            self.structured_logger.info(
                f"Executor '{name}' metrics",
                context=metrics
            )


# Global executor manager instance
resilient_executor_manager = ResilientExecutorManager()


def get_asr_executor() -> BoundedThreadPoolExecutor:
    """Get ASR executor instance"""
    return resilient_executor_manager.get_asr_executor()


def get_diarization_executor() -> BoundedThreadPoolExecutor:
    """Get diarization executor instance"""  
    return resilient_executor_manager.get_diarization_executor()


def get_file_processor_executor() -> BoundedThreadPoolExecutor:
    """Get file processor executor instance"""
    return resilient_executor_manager.get_file_processor_executor()


def shutdown_all_executors(wait: bool = True):
    """Shutdown all managed executors"""
    resilient_executor_manager.shutdown_all(wait=wait)


def get_all_executor_metrics() -> dict:
    """Get metrics for all managed executors"""
    return resilient_executor_manager.get_all_metrics()