"""
Intelligent Caching System for Ensemble Transcription
U8 Upgrade: Enhanced with comprehensive cache correctness and validation

This module provides a bridge between the legacy IntelligentCacheManager API
and the new cache correctness system with comprehensive validation.
"""

import os
import json
import hashlib
import time
import tempfile
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
import numpy as np
import diskcache as dc
from functools import wraps
import pickle
import logging
from datetime import datetime, timedelta

# Import new cache correctness system
from utils.cache_operations import AtomicCacheOperations, get_cache_operations, cached_stage_operation
from utils.cache_key import CacheKey, CacheScope, generate_cache_key_for_stage, validate_cached_data
from utils.enhanced_structured_logger import create_enhanced_logger
from core.run_context import get_global_run_context

# Configure logging for cache operations
cache_logger = create_enhanced_logger(__name__)

class IntelligentCacheManager:
    """
    Enhanced caching system with cache correctness and validation.
    
    This class now integrates with the comprehensive cache correctness system
    while maintaining backward compatibility with existing code.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, 
                 max_memory_cache_mb: int = 500,
                 max_disk_cache_gb: int = 5,
                 enable_compression: bool = True,
                 ttl_hours: int = 168,  # 1 week default TTL
                 use_cache_correctness: bool = True):
        """
        Initialize enhanced intelligent caching system.
        
        Args:
            cache_dir: Directory for persistent cache (defaults to system temp)
            max_memory_cache_mb: Maximum memory cache size in MB
            max_disk_cache_gb: Maximum disk cache size in GB
            enable_compression: Whether to compress cached data
            ttl_hours: Time-to-live for cache entries in hours
            use_cache_correctness: Whether to use new cache correctness system
        """
        self.use_cache_correctness = use_cache_correctness
        
        # Initialize new cache correctness system if enabled
        if self.use_cache_correctness:
            self.atomic_cache = AtomicCacheOperations(cache_dir)
            # Use the new cache directory structure
            self.cache_dir = self.atomic_cache.cache_base_dir
        else:
            # Legacy setup
            if cache_dir is None:
                cache_dir = os.path.join(tempfile.gettempdir(), "ensemble_transcription_cache")
            
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize diskcache with intelligent configuration
            self.disk_cache = dc.Cache(
                directory=str(self.cache_dir),
                size_limit=max_disk_cache_gb * 1024**3,  # Convert GB to bytes
                timeout=60,  # Timeout for cache operations
                tag_index=True,  # Enable tag-based operations
                statistics=True  # Enable hit/miss statistics
            )
        
        # In-memory cache for frequently accessed items (still used for both modes)
        self.memory_cache: Dict[str, Tuple[Any, float]] = {}  # (value, timestamp)
        self.max_memory_cache_size = max_memory_cache_mb * 1024 * 1024  # Convert to bytes
        self.current_memory_cache_size = 0
        
        # Configuration
        self.enable_compression = enable_compression
        self.ttl_seconds = ttl_hours * 3600
        
        # Enhanced cache statistics
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'validated_hits': 0,
            'validation_failures': 0,
            'misses': 0,
            'evictions': 0,
            'cache_sets': 0,
            'correctness_enabled': self.use_cache_correctness
        }
        
        cache_logger.info(
            f"Initialized enhanced intelligent cache: {self.cache_dir}",
            cache_correctness_enabled=self.use_cache_correctness,
            memory_cache_mb=max_memory_cache_mb,
            disk_cache_gb=max_disk_cache_gb
        )
    
    def _generate_cache_key(self, operation: str, *args, **kwargs) -> str:
        """
        Generate deterministic cache key from operation and parameters.
        
        Args:
            operation: Name of the cached operation
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Deterministic cache key string
        """
        # Create deterministic hash from operation and parameters
        key_components = [operation]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, np.ndarray):
                # For numpy arrays, use shape and statistics
                arg_hash = self._hash_numpy_array(arg)
            elif isinstance(arg, (dict, list)):
                # For complex objects, serialize to JSON
                arg_hash = hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()[:16]
            elif hasattr(arg, '__dict__'):
                # For objects with attributes, use their string representation
                arg_hash = hashlib.md5(str(arg).encode()).hexdigest()[:16]
            else:
                # For simple types, use string representation
                arg_hash = hashlib.md5(str(arg).encode()).hexdigest()[:16]
            key_components.append(arg_hash)
        
        # Add keyword arguments (sorted for determinism)
        for key in sorted(kwargs.keys()):
            value = kwargs[key]
            if isinstance(value, np.ndarray):
                value_hash = self._hash_numpy_array(value)
            elif isinstance(value, (dict, list)):
                value_hash = hashlib.md5(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
            else:
                value_hash = hashlib.md5(str(value).encode()).hexdigest()[:16]
            key_components.append(f"{key}_{value_hash}")
        
        # Create final cache key
        combined = "_".join(key_components)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def _hash_numpy_array(self, arr: np.ndarray) -> str:
        """
        Generate stable hash for numpy array based on shape and statistics.
        More stable than hashing raw data for caching purposes.
        
        Args:
            arr: Numpy array to hash
            
        Returns:
            Hash string
        """
        if arr.size == 0:
            return "empty_array"
        
        # Use array statistics for stable hashing
        stats = [
            str(arr.shape),
            str(arr.dtype),
            f"{float(np.mean(arr)):.8f}",
            f"{float(np.std(arr)):.8f}",
            f"{float(np.min(arr)):.8f}",
            f"{float(np.max(arr)):.8f}",
            str(arr.size)
        ]
        
        stats_str = "_".join(stats)
        return hashlib.md5(stats_str.encode()).hexdigest()[:16]
    
    def _hash_audio_file(self, file_path: str) -> str:
        """
        Generate hash for audio file based on content and metadata.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Hash string for file
        """
        try:
            # Get file stats
            stat = os.stat(file_path)
            file_info = [
                str(stat.st_size),
                str(int(stat.st_mtime)),
                str(Path(file_path).name)
            ]
            
            # Add content sample for verification
            with open(file_path, 'rb') as f:
                # Read first and last 1KB for quick hash
                start_chunk = f.read(1024)
                f.seek(-min(1024, stat.st_size), os.SEEK_END)
                end_chunk = f.read(1024)
                
            content_hash = hashlib.md5(start_chunk + end_chunk).hexdigest()[:16]
            file_info.append(content_hash)
            
            combined = "_".join(file_info)
            return hashlib.sha256(combined.encode()).hexdigest()[:32]
            
        except Exception as e:
            cache_logger.warning(f"Failed to hash audio file {file_path}: {e}")
            # Fallback to basic file hash
            return hashlib.md5(str(file_path).encode()).hexdigest()[:16]
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object in bytes."""
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # Fallback estimation
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            else:
                return 1024  # Default estimate
    
    def _cleanup_memory_cache(self):
        """Remove old entries from memory cache if size limit exceeded."""
        if self.current_memory_cache_size <= self.max_memory_cache_size:
            return
        
        # Sort by timestamp and remove oldest entries
        sorted_items = sorted(self.memory_cache.items(), key=lambda x: x[1][1])
        
        while (self.current_memory_cache_size > self.max_memory_cache_size and 
               sorted_items):
            key, (value, timestamp) = sorted_items.pop(0)
            size = self._estimate_size(value)
            self.current_memory_cache_size -= size
            del self.memory_cache[key]
            self.stats['evictions'] += 1
    
    def _add_to_memory_cache(self, cache_key: str, result: Any):
        """Add result to memory cache with size management"""
        size = self._estimate_size(result)
        if size < self.max_memory_cache_size * 0.1:  # Only cache smaller items in memory
            current_time = time.time()
            self.memory_cache[cache_key] = (result, current_time)
            self.current_memory_cache_size += size
            self._cleanup_memory_cache()
    
    def get(self, operation: str, *args, **kwargs) -> Optional[Any]:
        """
        Get cached result for operation with given parameters.
        Enhanced with cache correctness validation when enabled.
        
        Args:
            operation: Name of the cached operation
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cached result if available, None otherwise
        """
        if self.use_cache_correctness:
            return self._get_with_correctness(operation, *args, **kwargs)
        else:
            return self._get_legacy(operation, *args, **kwargs)
    
    def _get_with_correctness(self, operation: str, *args, **kwargs) -> Optional[Any]:
        """Get with cache correctness validation"""
        run_context = get_global_run_context()
        if run_context is None:
            cache_logger.debug("No run context available, falling back to legacy cache")
            return self._get_legacy(operation, *args, **kwargs)
        
        # Try to construct a cache key for validation
        try:
            # Extract stage parameters from kwargs
            stage_params = kwargs.get('stage_params', {})
            component_name = kwargs.get('component_name', operation)
            
            # Generate cache key using new system
            cache_key_str, cache_key_obj = generate_cache_key_for_stage(
                stage_name=operation,
                component_name=component_name,
                run_context=run_context,
                stage_params=stage_params,
                scope=CacheScope.PROJECT
            )
            
            # Try to get from atomic cache with validation
            result = self.atomic_cache.cache_get(
                cache_key_obj,
                run_context,
                format_hint="pickle"
            )
            
            if result is not None:
                # Cache hit with validation
                self.stats['validated_hits'] += 1
                self.stats['disk_hits'] += 1
                
                # Also add to memory cache for faster future access
                self._add_to_memory_cache(cache_key_str, result)
                
                cache_logger.info(
                    f"Validated cache hit for {operation}",
                    cache_hit=True,
                    cache_key=cache_key_obj.generate_short_key(),
                    validated=True,
                    stage_name=operation,
                    component_name=component_name
                )
                return result
            else:
                # Cache miss or validation failed
                self.stats['misses'] += 1
                cache_logger.debug(f"Cache miss or validation failed for {operation}")
                return None
                
        except Exception as e:
            cache_logger.warning(f"Cache correctness lookup failed for {operation}: {e}")
            # Fall back to legacy cache
            return self._get_legacy(operation, *args, **kwargs)
    
    def _get_legacy(self, operation: str, *args, **kwargs) -> Optional[Any]:
        """Legacy cache get implementation"""
        cache_key = self._generate_cache_key(operation, *args, **kwargs)
        current_time = time.time()
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            value, timestamp = self.memory_cache[cache_key]
            if current_time - timestamp < self.ttl_seconds:
                self.stats['memory_hits'] += 1
                cache_logger.debug(f"Memory cache hit for {operation}")
                return value
            else:
                # Expired, remove from memory cache
                del self.memory_cache[cache_key]
        
        # Check disk cache
        try:
            result = self.disk_cache.get(cache_key)
            if result is not None:
                # Move to memory cache for faster future access
                self._add_to_memory_cache(cache_key, result)
                
                self.stats['disk_hits'] += 1
                cache_logger.debug(f"Disk cache hit for {operation}")
                return result
        except Exception as e:
            cache_logger.warning(f"Error reading from disk cache: {e}")
        
        # Cache miss
        self.stats['misses'] += 1
        cache_logger.debug(f"Cache miss for {operation}")
        return None
    
    def set(self, operation: str, result: Any, *args, **kwargs) -> bool:
        """
        Store result in cache for operation with given parameters.
        Enhanced with cache correctness when enabled.
        
        Args:
            operation: Name of the cached operation
            result: Result to cache
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            True if successfully cached, False otherwise
        """
        if self.use_cache_correctness:
            return self._set_with_correctness(operation, result, *args, **kwargs)
        else:
            return self._set_legacy(operation, result, *args, **kwargs)
    
    def _set_with_correctness(self, operation: str, result: Any, *args, **kwargs) -> bool:
        """Set with cache correctness validation"""
        run_context = get_global_run_context()
        if run_context is None:
            cache_logger.debug("No run context available, falling back to legacy cache")
            return self._set_legacy(operation, result, *args, **kwargs)
        
        try:
            # Extract parameters
            stage_params = kwargs.get('stage_params', {})
            component_name = kwargs.get('component_name', operation)
            creating_component = kwargs.get('creating_component', component_name)
            component_version = kwargs.get('component_version', '1.0.0')
            
            # Generate cache key using new system
            cache_key_str, cache_key_obj = generate_cache_key_for_stage(
                stage_name=operation,
                component_name=component_name,
                run_context=run_context,
                stage_params=stage_params,
                scope=CacheScope.PROJECT
            )
            
            # Store using atomic cache operations
            success = self.atomic_cache.cache_set(
                cache_key_obj,
                result,
                creating_component=creating_component,
                component_version=component_version,
                format_hint="pickle",
                compression="gzip" if self.enable_compression else ""
            )
            
            if success:
                # Also add to memory cache for faster future access
                self._add_to_memory_cache(cache_key_str, result)
                
                self.stats['cache_sets'] += 1
                cache_logger.info(
                    f"Cached result for {operation} with validation",
                    cache_set=True,
                    cache_key=cache_key_obj.generate_short_key(),
                    stage_name=operation,
                    component_name=component_name,
                    data_size_bytes=self._estimate_size(result)
                )
                return True
            else:
                cache_logger.warning(f"Failed to cache result for {operation}")
                return False
                
        except Exception as e:
            cache_logger.error(f"Cache correctness set failed for {operation}: {e}")
            # Fall back to legacy cache
            return self._set_legacy(operation, result, *args, **kwargs)
    
    def _set_legacy(self, operation: str, result: Any, *args, **kwargs) -> bool:
        """Legacy cache set implementation"""
        cache_key = self._generate_cache_key(operation, *args, **kwargs)
        current_time = time.time()
        
        try:
            # Store in disk cache with TTL
            self.disk_cache.set(
                cache_key, 
                result, 
                expire=current_time + self.ttl_seconds,
                tag=operation  # Tag for bulk operations
            )
            
            # Store in memory cache if not too large
            self._add_to_memory_cache(cache_key, result)
            
            self.stats['cache_sets'] += 1
            size = self._estimate_size(result)
            cache_logger.debug(f"Cached result for {operation} (size: {size} bytes)")
            return True
            
        except Exception as e:
            cache_logger.error(f"Error storing in cache: {e}")
            return False
    
    def cached_function(self, operation_name: Optional[str] = None):
        """
        Decorator to automatically cache function results.
        
        Args:
            operation_name: Custom name for the operation (defaults to function name)
            
        Returns:
            Decorated function with caching
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation_name or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Try to get from cache first
                cached_result = self.get(op_name, *args, **kwargs)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(op_name, result, *args, **kwargs)
                return result
            
            return wrapper
        return decorator
    
    def invalidate_operation(self, operation: str) -> int:
        """
        Invalidate all cache entries for a specific operation.
        
        Args:
            operation: Name of operation to invalidate
            
        Returns:
            Number of entries invalidated
        """
        try:
            # Remove from disk cache using tag
            count = 0
            for key in list(self.disk_cache.iterkeys()):
                if key in self.disk_cache and self.disk_cache.tag(key) == operation:
                    del self.disk_cache[key]
                    count += 1
            
            # Remove from memory cache
            memory_keys_to_remove = []
            for key in self.memory_cache:
                if key.startswith(operation):
                    memory_keys_to_remove.append(key)
            
            for key in memory_keys_to_remove:
                value, _ = self.memory_cache[key]
                size = self._estimate_size(value)
                self.current_memory_cache_size -= size
                del self.memory_cache[key]
                count += 1
            
            cache_logger.info(f"Invalidated {count} cache entries for operation: {operation}")
            return count
            
        except Exception as e:
            cache_logger.error(f"Error invalidating cache for {operation}: {e}")
            return 0
    
    def clear_expired(self) -> int:
        """
        Remove all expired cache entries.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        count = 0
        
        # Clear expired memory cache entries
        expired_keys = []
        for key, (value, timestamp) in self.memory_cache.items():
            if current_time - timestamp >= self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            value, _ = self.memory_cache[key]
            size = self._estimate_size(value)
            self.current_memory_cache_size -= size
            del self.memory_cache[key]
            count += 1
        
        # Disk cache expiration is handled automatically by diskcache
        cache_logger.info(f"Removed {count} expired cache entries")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics with telemetry integration."""
        
        # Get stats from atomic cache if using cache correctness
        if self.use_cache_correctness:
            atomic_stats = self.atomic_cache.get_cache_stats()
            
            # Combined stats with validation information
            total_hits = self.stats['memory_hits'] + self.stats['disk_hits'] + self.stats['validated_hits']
            total_requests = total_hits + self.stats['misses']
            
            combined_stats = {
                'cache_correctness_enabled': True,
                'memory': {
                    'entries': len(self.memory_cache),
                    'size_mb': self.current_memory_cache_size / (1024 * 1024),
                    'max_size_mb': self.max_memory_cache_size / (1024 * 1024),
                    'hit_rate': self.stats['memory_hits'] / max(1, self.stats['memory_hits'] + self.stats['misses'])
                },
                'atomic_cache': atomic_stats,
                'validation': {
                    'validated_hits': self.stats['validated_hits'],
                    'validation_failures': self.stats['validation_failures'],
                    'validation_success_rate': self.stats['validated_hits'] / max(1, self.stats['validated_hits'] + self.stats['validation_failures'])
                },
                'combined': {
                    'total_hits': total_hits,
                    'total_misses': self.stats['misses'],
                    'overall_hit_rate': total_hits / max(1, total_requests),
                    'cache_sets': self.stats['cache_sets'],
                    'evictions': self.stats['evictions']
                }
            }
            
            # Add telemetry metrics
            cache_logger.info(
                "Cache statistics updated",
                cache_hit_rate=combined_stats['combined']['overall_hit_rate'],
                validation_success_rate=combined_stats['validation']['validation_success_rate'],
                validated_hits=self.stats['validated_hits'],
                validation_failures=self.stats['validation_failures'],
                total_cache_requests=total_requests
            )
            
            return combined_stats
        else:
            # Legacy stats
            disk_stats = dict(self.disk_cache.stats(enable=True))
            
            legacy_stats = {
                'cache_correctness_enabled': False,
                'memory': {
                    'entries': len(self.memory_cache),
                    'size_mb': self.current_memory_cache_size / (1024 * 1024),
                    'max_size_mb': self.max_memory_cache_size / (1024 * 1024),
                    'hit_rate': self.stats['memory_hits'] / max(1, self.stats['memory_hits'] + self.stats['misses'])
                },
                'disk': {
                    'entries': len(self.disk_cache),
                    'size_mb': disk_stats.get('size', 0) / (1024 * 1024),
                    'max_size_gb': self.disk_cache.size_limit / (1024**3),
                    'hit_rate': disk_stats.get('cache_hits', 0) / max(1, disk_stats.get('cache_hits', 0) + disk_stats.get('cache_misses', 1))
                },
                'combined': {
                    'total_hits': self.stats['memory_hits'] + self.stats['disk_hits'],
                    'total_misses': self.stats['misses'],
                    'overall_hit_rate': (self.stats['memory_hits'] + self.stats['disk_hits']) / max(1, self.stats['memory_hits'] + self.stats['disk_hits'] + self.stats['misses']),
                    'cache_sets': self.stats['cache_sets'],
                    'evictions': self.stats['evictions']
                }
            }
            
            return legacy_stats
    
    def cleanup(self, aggressive: bool = False):
        """
        Perform cache cleanup operations.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        cache_logger.info("Starting cache cleanup...")
        
        # Remove expired entries
        expired_count = self.clear_expired()
        
        if aggressive:
            # Clear memory cache
            self.memory_cache.clear()
            self.current_memory_cache_size = 0
            
            # Clean disk cache
            self.disk_cache.expire()
            
            cache_logger.info("Performed aggressive cache cleanup")
        
        cache_logger.info(f"Cache cleanup complete. Removed {expired_count} expired entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'memory_cache': {
                'hit_count': self.stats['memory_hits'],
                'size_bytes': self.current_memory_cache_size,
                'max_size_bytes': self.max_memory_cache_size,
                'entry_count': len(self.memory_cache)
            },
            'disk_cache': {
                'hit_count': self.stats['disk_hits'],
                'size_bytes': self.disk_cache.volume(),
                'entry_count': len(self.disk_cache)
            },
            'overall': {
                'total_hits': self.stats['memory_hits'] + self.stats['disk_hits'],
                'total_misses': self.stats['misses'],
                'hit_rate': (self.stats['memory_hits'] + self.stats['disk_hits']) / max(1, self.stats['memory_hits'] + self.stats['disk_hits'] + self.stats['misses']),
                'cache_sets': self.stats['cache_sets'],
                'evictions': self.stats['evictions']
            }
        }
    
    def clear_all(self) -> bool:
        """Clear all cache data (memory and disk)."""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            self.current_memory_cache_size = 0
            
            # Clear disk cache
            self.disk_cache.clear()
            
            # Reset statistics
            self.stats = {
                'memory_hits': 0,
                'disk_hits': 0,
                'misses': 0,
                'evictions': 0,
                'cache_sets': 0
            }
            
            cache_logger.info("All cache data cleared successfully")
            return True
        except Exception as e:
            cache_logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache configuration and status information."""
        return {
            'config': {
                'cache_dir': str(self.cache_dir),
                'max_memory_cache_mb': self.max_memory_cache_size / (1024 * 1024),
                'max_disk_cache_gb': self.disk_cache.size_limit / (1024**3),
                'ttl_hours': self.ttl_seconds / 3600,
                'compression_enabled': self.enable_compression
            },
            'status': {
                'memory_usage_percent': (self.current_memory_cache_size / self.max_memory_cache_size) * 100,
                'disk_usage_percent': (self.disk_cache.volume() / self.disk_cache.size_limit) * 100,
                'total_operations': self.stats['memory_hits'] + self.stats['disk_hits'] + self.stats['misses']
            }
        }
    
    def close(self):
        """Close cache connections and cleanup."""
        try:
            self.disk_cache.close()
            cache_logger.info("Cache closed successfully")
        except Exception as e:
            cache_logger.error(f"Error closing cache: {e}")


# Global cache manager instance
_cache_manager: Optional[IntelligentCacheManager] = None

def get_cache_manager() -> IntelligentCacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = IntelligentCacheManager()
    return _cache_manager

def cached_operation(operation_name: Optional[str] = None):
    """Convenience decorator for caching operations."""
    return get_cache_manager().cached_function(operation_name)