"""
Atomic Cache Operations with Validation and Sidecar Metadata

This module provides bulletproof cache operations that ensure:
- Atomic writes with temp files and fsync
- Comprehensive validation against RunContext fingerprints
- Sidecar .meta.json metadata for integrity and auditing
- Thread-safe operations for concurrent access
"""

import os
import json
import time
import threading
import tempfile
import shutil
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import pickle
import gzip

from utils.cache_key import CacheKey, CacheMetadata, CacheScope, generate_cache_key_for_stage, validate_cached_data
from utils.enhanced_structured_logger import create_enhanced_logger


class AtomicCacheOperations:
    """
    Thread-safe cache operations with atomic writes and comprehensive validation.
    
    This class provides the core cache operations that ensure data integrity
    and prevent stale artifacts through comprehensive validation.
    """
    
    def __init__(self, cache_base_dir: Optional[str] = None):
        """
        Initialize atomic cache operations.
        
        Args:
            cache_base_dir: Base directory for cache storage
        """
        if cache_base_dir is None:
            cache_base_dir = os.path.join(tempfile.gettempdir(), "ensemble_cache_v2")
        
        self.cache_base_dir = Path(cache_base_dir)
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = create_enhanced_logger("atomic_cache_operations")
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        
        # Statistics tracking
        self.stats = {
            'cache_reads': 0,
            'cache_writes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_failures': 0,
            'atomic_write_failures': 0,
            'metadata_errors': 0
        }
        
        self.logger.info(f"Initialized atomic cache operations at {self.cache_base_dir}")
    
    def _get_cache_paths(self, cache_key: str, scope: CacheScope, 
                        project_id: str = "", session_id: str = "", 
                        run_id: str = "") -> Tuple[Path, Path]:
        """
        Get cache file paths based on scope and identifiers.
        
        Args:
            cache_key: Full cache key (SHA256)
            scope: Cache isolation scope
            project_id: Project identifier for scoping
            session_id: Session identifier for scoping
            run_id: Run identifier for scoping
            
        Returns:
            Tuple of (data_path, metadata_path)
        """
        # Create scope-based directory structure
        if scope == CacheScope.GLOBAL:
            cache_dir = self.cache_base_dir / "global"
        elif scope == CacheScope.PROJECT:
            cache_dir = self.cache_base_dir / "projects" / project_id
        elif scope == CacheScope.SESSION:
            cache_dir = self.cache_base_dir / "sessions" / session_id
        elif scope == CacheScope.RUN:
            cache_dir = self.cache_base_dir / "runs" / run_id
        else:
            cache_dir = self.cache_base_dir / "unknown"
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use first 2 characters of cache key for subdirectory distribution
        subdir = cache_dir / cache_key[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        
        data_path = subdir / f"{cache_key}.data"
        metadata_path = subdir / f"{cache_key}.meta.json"
        
        return data_path, metadata_path
    
    def _write_atomic(self, target_path: Path, data: bytes, 
                      compression: str = "") -> bool:
        """
        Write data atomically using temp file and rename.
        
        Args:
            target_path: Final target path for the data
            data: Data to write
            compression: Optional compression (gzip)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create temporary file in same directory as target
            temp_fd, temp_path = tempfile.mkstemp(
                dir=target_path.parent,
                prefix=f".tmp_{target_path.name}_",
                suffix=".tmp"
            )
            
            try:
                with os.fdopen(temp_fd, 'wb') as temp_file:
                    if compression == "gzip":
                        compressed_data = gzip.compress(data)
                        temp_file.write(compressed_data)
                    else:
                        temp_file.write(data)
                    
                    # Force write to disk
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                
                # Atomic rename
                os.rename(temp_path, target_path)
                
                self.logger.debug(
                    f"Atomic write successful",
                    target_path=str(target_path),
                    data_size=len(data),
                    compression=compression
                )
                
                return True
                
            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e
                
        except Exception as e:
            self.logger.error(
                f"Atomic write failed",
                target_path=str(target_path),
                error=str(e)
            )
            self.stats['atomic_write_failures'] += 1
            return False
    
    def _read_data(self, data_path: Path, compression: str = "") -> Optional[bytes]:
        """
        Read data from cache file with optional decompression.
        
        Args:
            data_path: Path to data file
            compression: Compression type if any
            
        Returns:
            Data bytes or None if failed
        """
        try:
            if not data_path.exists():
                return None
            
            with open(data_path, 'rb') as f:
                data = f.read()
            
            if compression == "gzip":
                data = gzip.decompress(data)
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Failed to read cache data",
                data_path=str(data_path),
                error=str(e)
            )
            return None
    
    def _load_metadata(self, metadata_path: Path) -> Optional[CacheMetadata]:
        """
        Load cache metadata from sidecar file.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            CacheMetadata object or None if failed
        """
        try:
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            return CacheMetadata.from_dict(metadata_dict)
            
        except Exception as e:
            self.logger.error(
                f"Failed to load cache metadata",
                metadata_path=str(metadata_path),
                error=str(e)
            )
            self.stats['metadata_errors'] += 1
            return None
    
    def _save_metadata(self, metadata_path: Path, metadata: CacheMetadata) -> bool:
        """
        Save cache metadata to sidecar file atomically.
        
        Args:
            metadata_path: Path to metadata file
            metadata: Metadata to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata_json = json.dumps(
                metadata.to_dict(), 
                indent=2, 
                sort_keys=True,
                ensure_ascii=True
            )
            
            return self._write_atomic(
                metadata_path, 
                metadata_json.encode('utf-8')
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to save cache metadata",
                metadata_path=str(metadata_path),
                error=str(e)
            )
            self.stats['metadata_errors'] += 1
            return False
    
    def cache_get(self, cache_key_obj: CacheKey, 
                  current_run_context,
                  format_hint: str = "pickle") -> Optional[Any]:
        """
        Get cached data with comprehensive validation.
        
        Args:
            cache_key_obj: CacheKey object for the data
            current_run_context: Current RunContext for validation
            format_hint: Data format hint (pickle, json, bytes)
            
        Returns:
            Cached data if valid, None if miss or invalid
        """
        with self._lock:
            self.stats['cache_reads'] += 1
            
            try:
                cache_key = cache_key_obj.generate_cache_key()
                data_path, metadata_path = self._get_cache_paths(
                    cache_key, 
                    cache_key_obj.scope,
                    cache_key_obj.project_id,
                    cache_key_obj.session_id,
                    cache_key_obj.run_id
                )
                
                # Load and validate metadata
                metadata = self._load_metadata(metadata_path)
                if metadata is None:
                    self.logger.debug(
                        "Cache miss: no metadata",
                        cache_key_short=cache_key[:12],
                        stage_name=cache_key_obj.stage_name,
                        component_name=cache_key_obj.component_name
                    )
                    self.stats['cache_misses'] += 1
                    return None
                
                # Validate cache key against current context
                stored_cache_key = CacheKey(**metadata.original_cache_key_dict)
                if not validate_cached_data(stored_cache_key, current_run_context):
                    self.logger.warning(
                        "Cache validation failed",
                        cache_key_short=cache_key[:12],
                        stage_name=cache_key_obj.stage_name,
                        component_name=cache_key_obj.component_name
                    )
                    self.stats['validation_failures'] += 1
                    self.stats['cache_misses'] += 1
                    return None
                
                # Read data file
                data_bytes = self._read_data(data_path, metadata.compression)
                if data_bytes is None:
                    self.logger.warning(
                        "Cache miss: failed to read data",
                        cache_key_short=cache_key[:12],
                        data_path=str(data_path)
                    )
                    self.stats['cache_misses'] += 1
                    return None
                
                # Deserialize data based on format
                try:
                    if format_hint == "pickle":
                        data = pickle.loads(data_bytes)
                    elif format_hint == "json":
                        data = json.loads(data_bytes.decode('utf-8'))
                    elif format_hint == "bytes":
                        data = data_bytes
                    else:
                        # Try pickle as fallback
                        data = pickle.loads(data_bytes)
                    
                except Exception as e:
                    self.logger.error(
                        "Failed to deserialize cached data",
                        cache_key_short=cache_key[:12],
                        format_hint=format_hint,
                        error=str(e)
                    )
                    self.stats['cache_misses'] += 1
                    return None
                
                # Update access tracking
                metadata.update_access()
                metadata.update_validation()
                self._save_metadata(metadata_path, metadata)
                
                # Cache hit!
                self.stats['cache_hits'] += 1
                self.logger.info(
                    "Cache hit",
                    cache_hit=True,
                    cache_key_short=cache_key[:12],
                    validated=True,
                    stage_name=cache_key_obj.stage_name,
                    component_name=cache_key_obj.component_name,
                    access_count=metadata.access_count,
                    data_size_bytes=len(data_bytes)
                )
                
                return data
                
            except Exception as e:
                self.logger.error(
                    "Cache get operation failed",
                    cache_key_short=cache_key_obj.generate_short_key(),
                    error=str(e)
                )
                self.stats['cache_misses'] += 1
                return None
    
    def cache_set(self, cache_key_obj: CacheKey, 
                  data: Any,
                  creating_component: str,
                  component_version: str = "1.0.0",
                  format_hint: str = "pickle",
                  compression: str = "") -> bool:
        """
        Store data in cache with atomic write and metadata.
        
        Args:
            cache_key_obj: CacheKey object for the data
            data: Data to cache
            creating_component: Component that created this data
            component_version: Version of creating component
            format_hint: Data format (pickle, json, bytes)
            compression: Compression type (gzip)
            
        Returns:
            True if successfully cached, False otherwise
        """
        with self._lock:
            self.stats['cache_writes'] += 1
            
            try:
                cache_key = cache_key_obj.generate_cache_key()
                data_path, metadata_path = self._get_cache_paths(
                    cache_key,
                    cache_key_obj.scope,
                    cache_key_obj.project_id,
                    cache_key_obj.session_id,
                    cache_key_obj.run_id
                )
                
                # Serialize data
                try:
                    if format_hint == "pickle":
                        data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                    elif format_hint == "json":
                        data_json = json.dumps(data, sort_keys=True, ensure_ascii=True)
                        data_bytes = data_json.encode('utf-8')
                    elif format_hint == "bytes":
                        data_bytes = data if isinstance(data, bytes) else bytes(data)
                    else:
                        # Default to pickle
                        data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                        
                except Exception as e:
                    self.logger.error(
                        "Failed to serialize data for caching",
                        cache_key_short=cache_key[:12],
                        format_hint=format_hint,
                        error=str(e)
                    )
                    return False
                
                # Write data atomically
                if not self._write_atomic(data_path, data_bytes, compression):
                    return False
                
                # Create metadata
                metadata = CacheMetadata(
                    cache_key=cache_key,
                    cache_key_short=cache_key[:12],
                    validation_fingerprint=cache_key_obj.get_validation_fingerprint(),
                    original_cache_key_dict=cache_key_obj.to_stable_dict(),
                    data_file_path=str(data_path),
                    byte_size=len(data_bytes),
                    file_format=format_hint,
                    compression=compression,
                    created_by_component=creating_component,
                    component_version=component_version,
                    producing_run_id=cache_key_obj.run_id
                )
                
                # Save metadata atomically
                if not self._save_metadata(metadata_path, metadata):
                    # Clean up data file if metadata save failed
                    try:
                        data_path.unlink()
                    except:
                        pass
                    return False
                
                self.logger.info(
                    "Cache write successful",
                    cache_key_short=cache_key[:12],
                    stage_name=cache_key_obj.stage_name,
                    component_name=cache_key_obj.component_name,
                    data_size_bytes=len(data_bytes),
                    compression=compression,
                    format_hint=format_hint
                )
                
                return True
                
            except Exception as e:
                self.logger.error(
                    "Cache set operation failed",
                    cache_key_short=cache_key_obj.generate_short_key(),
                    error=str(e)
                )
                return False
    
    def cache_invalidate(self, cache_key_obj: CacheKey) -> bool:
        """
        Invalidate cached data by removing files.
        
        Args:
            cache_key_obj: CacheKey object for the data
            
        Returns:
            True if successfully invalidated, False otherwise
        """
        with self._lock:
            try:
                cache_key = cache_key_obj.generate_cache_key()
                data_path, metadata_path = self._get_cache_paths(
                    cache_key,
                    cache_key_obj.scope,
                    cache_key_obj.project_id,
                    cache_key_obj.session_id,
                    cache_key_obj.run_id
                )
                
                removed_files = 0
                
                # Remove data file
                if data_path.exists():
                    data_path.unlink()
                    removed_files += 1
                
                # Remove metadata file
                if metadata_path.exists():
                    metadata_path.unlink()
                    removed_files += 1
                
                if removed_files > 0:
                    self.logger.info(
                        "Cache invalidated",
                        cache_key_short=cache_key[:12],
                        stage_name=cache_key_obj.stage_name,
                        component_name=cache_key_obj.component_name,
                        files_removed=removed_files
                    )
                
                return True
                
            except Exception as e:
                self.logger.error(
                    "Cache invalidation failed",
                    cache_key_short=cache_key_obj.generate_short_key(),
                    error=str(e)
                )
                return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache operation statistics"""
        with self._lock:
            total_operations = self.stats['cache_reads'] + self.stats['cache_writes']
            hit_rate = self.stats['cache_hits'] / max(1, self.stats['cache_reads'])
            
            return {
                **self.stats,
                'total_operations': total_operations,
                'cache_hit_rate': round(hit_rate, 3),
                'validation_failure_rate': self.stats['validation_failures'] / max(1, self.stats['cache_reads']),
                'cache_directory': str(self.cache_base_dir)
            }
    
    def scan_cache_entries(self, scope: Optional[CacheScope] = None,
                          project_id: Optional[str] = None) -> List[Tuple[Path, CacheMetadata]]:
        """
        Scan cache entries for audit and cleanup operations.
        
        Args:
            scope: Optional scope filter
            project_id: Optional project filter
            
        Returns:
            List of (metadata_path, metadata) tuples
        """
        entries = []
        
        try:
            # Determine scan directories based on filters
            if scope == CacheScope.GLOBAL:
                scan_dirs = [self.cache_base_dir / "global"]
            elif scope == CacheScope.PROJECT and project_id:
                scan_dirs = [self.cache_base_dir / "projects" / project_id]
            else:
                # Scan all directories
                scan_dirs = []
                for subdir in self.cache_base_dir.iterdir():
                    if subdir.is_dir():
                        scan_dirs.append(subdir)
                        if subdir.name in ("projects", "sessions", "runs"):
                            for project_dir in subdir.iterdir():
                                if project_dir.is_dir():
                                    scan_dirs.append(project_dir)
            
            # Scan for metadata files
            for scan_dir in scan_dirs:
                if not scan_dir.exists():
                    continue
                
                for metadata_file in scan_dir.rglob("*.meta.json"):
                    metadata = self._load_metadata(metadata_file)
                    if metadata is not None:
                        entries.append((metadata_file, metadata))
            
            self.logger.debug(f"Scanned {len(entries)} cache entries")
            
        except Exception as e:
            self.logger.error(f"Cache scan failed: {e}")
        
        return entries


# Context manager for cache operations
@contextmanager
def cache_operation_context(cache_ops: Optional[AtomicCacheOperations] = None):
    """Context manager for cache operations with error handling"""
    if cache_ops is None:
        cache_ops = get_cache_operations()
    
    try:
        yield cache_ops
    except Exception as e:
        cache_ops.logger.error(f"Cache operation context error: {e}")
        raise


# Global cache operations instance
_global_cache_operations: Optional[AtomicCacheOperations] = None


def get_cache_operations() -> AtomicCacheOperations:
    """Get or create global cache operations instance"""
    global _global_cache_operations
    if _global_cache_operations is None:
        _global_cache_operations = AtomicCacheOperations()
    return _global_cache_operations


def cached_stage_operation(stage_name: str, 
                          component_name: str,
                          stage_params: Optional[Dict[str, Any]] = None,
                          scope: CacheScope = CacheScope.PROJECT,
                          format_hint: str = "pickle"):
    """
    Decorator for automatic caching of stage operations.
    
    This decorator automatically handles cache key generation, validation,
    and atomic storage for processing stage operations.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Get run context from self or kwargs
            run_context = getattr(self, 'run_context', None)
            if run_context is None:
                run_context = kwargs.get('run_context')
            
            if run_context is None:
                # No run context available, execute without caching
                return func(self, *args, **kwargs)
            
            # Generate cache key
            cache_key_str, cache_key_obj = generate_cache_key_for_stage(
                stage_name=stage_name,
                component_name=component_name,
                run_context=run_context,
                stage_params=stage_params,
                scope=scope
            )
            
            # Try to get from cache
            cache_ops = get_cache_operations()
            cached_result = cache_ops.cache_get(
                cache_key_obj, 
                run_context, 
                format_hint=format_hint
            )
            
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(self, *args, **kwargs)
            
            # Cache the result
            component_version = getattr(self, 'version', '1.0.0')
            cache_ops.cache_set(
                cache_key_obj,
                result,
                creating_component=component_name,
                component_version=component_version,
                format_hint=format_hint
            )
            
            return result
        
        return wrapper
    return decorator