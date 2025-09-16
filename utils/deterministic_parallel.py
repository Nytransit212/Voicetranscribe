"""
Deterministic Parallel Processing Utilities

This module provides utilities for concurrent processing with deterministic ordering,
ensuring that parallel execution produces consistent results regardless of timing variations.
"""

import time
import threading
from typing import Dict, List, Any, Callable, Optional, Tuple, Union, TypeVar
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from collections import defaultdict
import heapq

from utils.enhanced_structured_logger import create_enhanced_logger

T = TypeVar('T')

@dataclass
class DeterministicTask:
    """A task with deterministic ordering information"""
    task_id: str
    order_key: Union[str, int, tuple]  # For deterministic sorting
    callable_fn: Callable
    args: tuple
    kwargs: dict
    submitted_at: float
    future: Optional[Future] = None
    result: Any = None
    error: Optional[Exception] = None
    completed_at: Optional[float] = None

@dataclass
class DeterministicResult:
    """Result with deterministic ordering preserved"""
    task_id: str
    order_key: Union[str, int, tuple]
    result: Any
    processing_time: float
    error: Optional[Exception] = None


class DeterministicThreadPoolExecutor:
    """
    ThreadPoolExecutor wrapper that ensures deterministic result ordering
    regardless of completion timing variations.
    """
    
    def __init__(self, max_workers: Optional[int] = None, thread_name_prefix: str = "DeterministicWorker"):
        """
        Initialize deterministic thread pool executor
        
        Args:
            max_workers: Maximum number of worker threads
            thread_name_prefix: Prefix for worker thread names
        """
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.logger = create_enhanced_logger("deterministic_executor")
        
        # Task tracking
        self._tasks: Dict[str, DeterministicTask] = {}
        self._task_counter = 0
        self._lock = threading.RLock()
        
        # Internal executor
        self._executor: Optional[ThreadPoolExecutor] = None
    
    def __enter__(self):
        """Context manager entry"""
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=self.thread_name_prefix
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def submit_ordered(self, 
                      order_key: Union[str, int, tuple],
                      fn: Callable[..., T], 
                      *args, 
                      task_id: Optional[str] = None,
                      **kwargs) -> str:
        """
        Submit a task with deterministic ordering
        
        Args:
            order_key: Key used for deterministic result ordering
            fn: Callable to execute
            *args: Positional arguments for callable
            task_id: Optional task identifier (auto-generated if None)
            **kwargs: Keyword arguments for callable
            
        Returns:
            Task ID for tracking
        """
        if not self._executor:
            raise RuntimeError("Executor not initialized. Use within context manager.")
        
        with self._lock:
            # Generate task ID if not provided
            if task_id is None:
                self._task_counter += 1
                task_id = f"task_{self._task_counter:04d}"
            
            # Create task
            task = DeterministicTask(
                task_id=task_id,
                order_key=order_key,
                callable_fn=fn,
                args=args,
                kwargs=kwargs,
                submitted_at=time.time()
            )
            
            # Submit to underlying executor
            task.future = self._executor.submit(fn, *args, **kwargs)
            
            # Track task
            self._tasks[task_id] = task
            
            self.logger.debug(f"Submitted task {task_id} with order_key {order_key}")
            
            return task_id
    
    def get_results_ordered(self, timeout: Optional[float] = None) -> List[DeterministicResult]:
        """
        Get all results in deterministic order
        
        Args:
            timeout: Maximum time to wait for all tasks to complete
            
        Returns:
            List of results ordered by order_key
        """
        if not self._tasks:
            return []
        
        start_time = time.time()
        completed_tasks = []
        
        # Wait for all futures to complete
        futures_to_tasks = {task.future: task for task in self._tasks.values() if task.future is not None}
        
        try:
            # Use as_completed but collect all results before ordering
            for future in as_completed(futures_to_tasks.keys(), timeout=timeout):
                task = futures_to_tasks[future]
                
                try:
                    result = future.result()
                    task.result = result
                    task.completed_at = time.time()
                    completed_tasks.append(task)
                    
                    self.logger.debug(f"Task {task.task_id} completed successfully")
                    
                except Exception as e:
                    task.error = e
                    task.completed_at = time.time()
                    completed_tasks.append(task)
                    
                    self.logger.error(f"Task {task.task_id} failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Error waiting for task completion: {e}")
            # Still process any completed tasks
        
        # Sort results by order_key for deterministic ordering
        completed_tasks.sort(key=lambda t: t.order_key)
        
        # Convert to deterministic results
        results = []
        for task in completed_tasks:
            processing_time = (task.completed_at or time.time()) - task.submitted_at
            result = DeterministicResult(
                task_id=task.task_id,
                order_key=task.order_key,
                result=task.result,
                processing_time=processing_time,
                error=task.error
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        self.logger.info(
            f"Completed {len(results)} tasks in deterministic order "
            f"(total time: {total_time:.2f}s)"
        )
        
        # Log deterministic ordering telemetry
        self.logger.info("Deterministic execution complete",
            event_type="deterministic_execution_complete",
            metrics={
                "task_count": len(results),
                "total_duration": total_time,
                "success_count": sum(1 for r in results if r.error is None),
                "error_count": sum(1 for r in results if r.error is not None),
                "avg_processing_time": sum(r.processing_time for r in results) / len(results) if results else 0
            })
        
        return results
    
    def map_ordered(self, 
                   fn: Callable[..., T], 
                   iterable: List[Any],
                   order_key_fn: Optional[Callable[[Any], Union[str, int, tuple]]] = None,
                   timeout: Optional[float] = None) -> List[T]:
        """
        Map function over iterable with deterministic ordering
        
        Args:
            fn: Function to apply to each item
            iterable: Items to process
            order_key_fn: Function to extract order key from item (uses index if None)
            timeout: Maximum time to wait for all tasks
            
        Returns:
            Results in the same order as input iterable
        """
        # Submit all tasks
        for i, item in enumerate(iterable):
            order_key = order_key_fn(item) if order_key_fn else i
            self.submit_ordered(order_key, fn, item)
        
        # Get results and extract values
        results = self.get_results_ordered(timeout=timeout)
        
        # Return results in order, raising any errors
        final_results = []
        for result in results:
            if result.error:
                raise result.error
            final_results.append(result.result)
        
        return final_results


class StableTieBreaker:
    """Implements stable tie-breaking rules for deterministic consensus"""
    
    @staticmethod
    def break_ties_by_score_then_lexical(candidates: List[Dict[str, Any]], 
                                       score_key: str = 'confidence_scores',
                                       fallback_key: str = 'candidate_id') -> Dict[str, Any]:
        """
        Break ties using score first, then lexicographic comparison
        
        Args:
            candidates: List of candidate dictionaries
            score_key: Key to access score for primary sorting
            fallback_key: Key to access fallback value for tie-breaking
            
        Returns:
            Winner candidate using stable tie-breaking
        """
        if not candidates:
            raise ValueError("No candidates provided for tie-breaking")
        
        # Sort by score (descending) then by fallback_key (ascending) for stability
        def sort_key(candidate):
            score = candidate.get(score_key, {})
            if isinstance(score, dict):
                primary_score = score.get('final_score', 0.0)
            else:
                primary_score = float(score)
            
            fallback_value = str(candidate.get(fallback_key, ''))
            
            # Return tuple for stable sorting: (-score, fallback_value)
            # Negative score for descending order, fallback_value for ascending
            return (-primary_score, fallback_value)
        
        sorted_candidates = sorted(candidates, key=sort_key)
        
        # Log tie-breaking decision
        logger = create_enhanced_logger("stable_tie_breaker")
        winner = sorted_candidates[0]
        
        if len(candidates) > 1:
            # Check if there was actually a tie
            winner_score = winner.get(score_key, {})
            if isinstance(winner_score, dict):
                winner_final_score = winner_score.get('final_score', 0.0)
            else:
                winner_final_score = float(winner_score)
            
            tied_count = sum(
                1 for c in candidates 
                if abs(
                    (c.get(score_key, {}).get('final_score', 0.0) 
                     if isinstance(c.get(score_key, {}), dict) 
                     else float(c.get(score_key, 0.0))) - winner_final_score
                ) < 0.001
            )
            
            if tied_count > 1:
                logger.info(
                    f"Stable tie-breaking applied: {tied_count} candidates tied at score {winner_final_score:.3f}, "
                    f"winner: {winner.get(fallback_key, 'unknown')}"
                )
        
        return winner
    
    @staticmethod
    def sort_manifests_deterministically(manifests: List[Dict[str, Any]], 
                                       sort_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Sort input manifests deterministically before parallel processing
        
        Args:
            manifests: List of manifest dictionaries
            sort_keys: Keys to use for sorting (default: ['timestamp', 'path', 'sha256'])
            
        Returns:
            Deterministically sorted manifests
        """
        if not manifests:
            return []
        
        if sort_keys is None:
            sort_keys = ['timestamp', 'path', 'sha256', 'size']
        
        def get_sort_key(manifest):
            """Extract multi-level sort key from manifest"""
            key_parts = []
            for key in sort_keys:
                value = manifest.get(key, '')
                # Convert to string for consistent comparison
                if isinstance(value, (int, float)):
                    key_parts.append(f"{value:020.6f}")  # Fixed-width numeric formatting
                else:
                    key_parts.append(str(value))
            return tuple(key_parts)
        
        sorted_manifests = sorted(manifests, key=get_sort_key)
        
        logger = create_enhanced_logger("deterministic_sorting")
        logger.info(f"Sorted {len(manifests)} manifests deterministically using keys: {sort_keys}")
        
        return sorted_manifests


def run_deterministic_parallel(tasks: List[Tuple[Callable, tuple, dict]], 
                              max_workers: Optional[int] = None,
                              order_key_fn: Optional[Callable[[int], Union[str, int, tuple]]] = None,
                              timeout: Optional[float] = None) -> List[Any]:
    """
    Convenience function to run tasks with deterministic parallel execution
    
    Args:
        tasks: List of (callable, args, kwargs) tuples
        max_workers: Maximum number of worker threads
        order_key_fn: Function to generate order key from task index
        timeout: Maximum time to wait for completion
        
    Returns:
        Results in deterministic order
    """
    if not tasks:
        return []
    
    with DeterministicThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for i, (fn, args, kwargs) in enumerate(tasks):
            order_key = order_key_fn(i) if order_key_fn else i
            executor.submit_ordered(order_key, fn, *args, **kwargs)
        
        # Get results
        results = executor.get_results_ordered(timeout=timeout)
        
        # Extract values and raise any errors
        final_results = []
        for result in results:
            if result.error:
                raise result.error
            final_results.append(result.result)
        
        return final_results


def ensure_deterministic_futures_processing(futures_dict: Dict[Future, Any], 
                                           order_key_fn: Callable[[Any], Union[str, int, tuple]]) -> List[Tuple[Any, Any]]:
    """
    Process concurrent.futures results in deterministic order
    
    Args:
        futures_dict: Dictionary mapping Future objects to associated data
        order_key_fn: Function to extract order key from associated data
        
    Returns:
        List of (data, result) tuples in deterministic order
    """
    # Collect all completions first
    completions = []
    
    for future in as_completed(futures_dict.keys()):
        data = futures_dict[future]
        try:
            result = future.result()
            completions.append((data, result, None))
        except Exception as e:
            completions.append((data, None, e))
    
    # Sort by order key for deterministic processing
    completions.sort(key=lambda x: order_key_fn(x[0]))
    
    # Extract results and raise any errors
    ordered_results = []
    for data, result, error in completions:
        if error:
            raise error
        ordered_results.append((data, result))
    
    return ordered_results