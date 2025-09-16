"""
Atomic I/O and Temp Hygiene System for Ensemble Transcription

This module provides bulletproof file operations that prevent partial file corruption
and ensures clean temporary directory management with comprehensive error recovery.

Key Features:
- Atomic file writes with temp files and fsync
- Per-run temporary directory isolation
- Filename collision prevention with deterministic suffixes
- Automatic cleanup with breadcrumb system for failures
- Thread-safe operations for concurrent processing
- Comprehensive telemetry and observability integration
"""

import os
import time
import json
import hashlib
import threading
import tempfile
import shutil
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

from utils.enhanced_structured_logger import create_enhanced_logger
from core.run_context import get_global_run_context


class TempDirectoryScope(Enum):
    """Scopes for temporary directory organization"""
    STEMS = "stems"           # Audio source separation stems
    CHUNKS = "chunks"         # Audio chunks for processing
    ASR = "asr"              # ASR temporary outputs
    DIARIZATION = "diarization"  # Diarization temporary outputs
    TRANSCRIPTS = "transcripts"  # Transcript temporary files
    CAPTIONS = "captions"     # Caption format temporary files
    ARTIFACTS = "artifacts"   # General processing artifacts
    CACHE = "cache"          # Temporary cache files


@dataclass
class AtomicOperationStats:
    """Statistics for atomic I/O operations"""
    atomic_commits_count: int = 0
    atomic_commit_failures: int = 0
    temp_file_creations: int = 0
    safe_removals: int = 0
    orphan_temp_files_count: int = 0
    total_bytes_written: int = 0
    total_operations_duration_ms: float = 0.0


@dataclass
class TempDirectoryInfo:
    """Information about a temporary directory"""
    run_id: str
    path: Path
    created_at: datetime
    stage_name: str = ""
    project_id: str = ""
    session_id: str = ""
    is_aborted: bool = False
    abort_reason: str = ""
    abort_stage: str = ""


class AtomicIOManager:
    """
    Thread-safe atomic I/O manager with comprehensive temp hygiene.
    
    Provides atomic file operations that prevent partial writes and corruption,
    along with per-run temporary directory management and automatic cleanup.
    """
    
    def __init__(self, base_temp_dir: Optional[str] = None, 
                 retention_hours: int = 24,
                 enable_telemetry: bool = True):
        """
        Initialize atomic I/O manager.
        
        Args:
            base_temp_dir: Base directory for temporary files (defaults to /data/tmp)
            retention_hours: Hours to retain abandoned temp directories
            enable_telemetry: Whether to enable telemetry tracking
        """
        # Setup base temp directory
        if base_temp_dir is None:
            base_temp_dir = "/tmp/ensemble_atomic_io"
        
        self.base_temp_dir = Path(base_temp_dir)
        self.base_temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.retention_hours = retention_hours
        self.enable_telemetry = enable_telemetry
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics tracking
        self.stats = AtomicOperationStats()
        
        # Active temp directories by run_id
        self._active_temp_dirs: Dict[str, TempDirectoryInfo] = {}
        
        # Logger
        self.logger = create_enhanced_logger("atomic_io_manager")
        
        self.logger.info(f"Initialized atomic I/O manager at {self.base_temp_dir}")
    
    def _generate_collision_suffix(self, cache_key: Optional[str] = None, 
                                 identifier: Optional[str] = None) -> str:
        """
        Generate deterministic 4-hex suffix to prevent filename collisions.
        
        Args:
            cache_key: Cache key for deterministic suffix
            identifier: Additional identifier for uniqueness
            
        Returns:
            4-character hex suffix
        """
        if cache_key:
            # Use first 4 chars of cache key hash for deterministic suffix
            suffix_source = cache_key
        else:
            # Fallback to timestamp + thread ID for uniqueness
            suffix_source = f"{time.time()}_{threading.get_ident()}_{identifier or ''}"
        
        hash_object = hashlib.md5(suffix_source.encode('utf-8'))
        return hash_object.hexdigest()[:4]
    
    def open_temp_for(self, final_path: Union[str, Path], 
                     cache_key: Optional[str] = None,
                     mode: str = 'w',
                     encoding: Optional[str] = 'utf-8',
                     create_parents: bool = True) -> Tuple[Any, str]:
        """
        Create a temporary file for atomic writing to final_path.
        
        Args:
            final_path: Final destination path for the file
            cache_key: Optional cache key for collision prevention
            mode: File open mode
            encoding: File encoding (for text modes)
            create_parents: Whether to create parent directories
            
        Returns:
            Tuple of (file_handle, temp_path)
        """
        final_path = Path(final_path)
        
        # Create parent directories if needed
        if create_parents:
            final_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate collision-safe temp filename
        collision_suffix = self._generate_collision_suffix(cache_key, str(final_path))
        temp_filename = f"{final_path.name}.tmp.{collision_suffix}"
        temp_path = final_path.parent / temp_filename
        
        try:
            # Open temp file
            if 'b' in mode:
                file_handle = open(temp_path, mode)
            else:
                file_handle = open(temp_path, mode, encoding=encoding)
            
            with self._lock:
                self.stats.temp_file_creations += 1
            
            self.logger.debug(f"Created temp file: {temp_path} -> {final_path}")
            
            return file_handle, str(temp_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create temp file for {final_path}: {e}")
            raise
    
    @contextmanager
    def atomic_write(self, final_path: Union[str, Path],
                    cache_key: Optional[str] = None,
                    mode: str = 'w',
                    encoding: Optional[str] = 'utf-8'):
        """
        Context manager for atomic file writing.
        
        Usage:
            with atomic_io.atomic_write('/path/to/file.json') as f:
                json.dump(data, f)
        
        Args:
            final_path: Final destination path
            cache_key: Optional cache key for collision prevention
            mode: File open mode
            encoding: File encoding
        """
        start_time = time.monotonic()
        final_path = Path(final_path)
        temp_file = None
        temp_path = None
        
        try:
            # Create temp file
            temp_file, temp_path = self.open_temp_for(
                final_path, cache_key, mode, encoding
            )
            
            yield temp_file
            
            # Ensure data is written to disk
            temp_file.flush()
            if hasattr(temp_file, 'fileno'):
                os.fsync(temp_file.fileno())
            
            temp_file.close()
            temp_file = None
            
            # Atomic commit
            self.commit_temp(temp_path, final_path)
            
            # Update telemetry
            duration_ms = (time.monotonic() - start_time) * 1000
            file_size = final_path.stat().st_size if final_path.exists() else 0
            
            if self.enable_telemetry:
                self._log_commit_telemetry(final_path, file_size, duration_ms)
            
        except Exception as e:
            # Cleanup on failure
            if temp_file:
                try:
                    temp_file.close()
                except:
                    pass
            
            if temp_path:
                self.safe_remove(temp_path)
            
            with self._lock:
                self.stats.atomic_commit_failures += 1
            
            self.logger.error(f"Atomic write failed for {final_path}: {e}")
            raise
    
    def commit_temp(self, temp_path: Union[str, Path], 
                   final_path: Union[str, Path]) -> None:
        """
        Atomically commit temporary file to final location.
        
        Args:
            temp_path: Path to temporary file
            final_path: Final destination path
        """
        temp_path = Path(temp_path)
        final_path = Path(final_path)
        
        try:
            # Ensure parent directory exists
            final_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic rename (works across filesystems on modern systems)
            os.replace(str(temp_path), str(final_path))
            
            with self._lock:
                self.stats.atomic_commits_count += 1
            
            self.logger.debug(f"Committed temp file: {temp_path} -> {final_path}")
            
        except Exception as e:
            with self._lock:
                self.stats.atomic_commit_failures += 1
            
            self.logger.error(f"Failed to commit temp file {temp_path} -> {final_path}: {e}")
            raise
    
    def safe_remove(self, file_path: Union[str, Path]) -> bool:
        """
        Safely remove a file with error handling.
        
        Args:
            file_path: Path to file to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        file_path = Path(file_path)
        
        try:
            if file_path.exists():
                file_path.unlink()
                
                with self._lock:
                    self.stats.safe_removals += 1
                
                self.logger.debug(f"Safely removed: {file_path}")
                return True
            
            return True  # Already doesn't exist
            
        except Exception as e:
            self.logger.warning(f"Failed to remove {file_path}: {e}")
            return False
    
    def create_run_temp_directory(self, run_id: str, 
                                stage_name: str = "",
                                project_id: str = "",
                                session_id: str = "") -> Path:
        """
        Create a temporary directory for a specific run.
        
        Args:
            run_id: Unique run identifier
            stage_name: Processing stage name
            project_id: Project identifier
            session_id: Session identifier
            
        Returns:
            Path to created temporary directory
        """
        run_temp_dir = self.base_temp_dir / run_id
        run_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        for scope in TempDirectoryScope:
            subdir = run_temp_dir / scope.value
            subdir.mkdir(exist_ok=True)
        
        # Register temp directory
        with self._lock:
            temp_info = TempDirectoryInfo(
                run_id=run_id,
                path=run_temp_dir,
                created_at=datetime.now(timezone.utc),
                stage_name=stage_name,
                project_id=project_id,
                session_id=session_id
            )
            self._active_temp_dirs[run_id] = temp_info
        
        self.logger.info(f"Created run temp directory: {run_temp_dir}")
        return run_temp_dir
    
    def get_run_temp_subdir(self, run_id: str, scope: TempDirectoryScope) -> Path:
        """
        Get subdirectory path within run temp directory.
        
        Args:
            run_id: Run identifier
            scope: Temporary directory scope
            
        Returns:
            Path to subdirectory
        """
        if run_id not in self._active_temp_dirs:
            raise ValueError(f"No temp directory registered for run_id: {run_id}")
        
        temp_info = self._active_temp_dirs[run_id]
        subdir = temp_info.path / scope.value
        subdir.mkdir(exist_ok=True)
        
        return subdir
    
    def mark_run_aborted(self, run_id: str, stage_name: str, error_message: str) -> None:
        """
        Mark a run as aborted and create breadcrumb file.
        
        Args:
            run_id: Run identifier
            stage_name: Stage where abortion occurred
            error_message: Error message
        """
        if run_id not in self._active_temp_dirs:
            self.logger.warning(f"Cannot mark run {run_id} as aborted - not registered")
            return
        
        temp_info = self._active_temp_dirs[run_id]
        temp_info.is_aborted = True
        temp_info.abort_stage = stage_name
        temp_info.abort_reason = error_message
        
        # Create ABORTED breadcrumb file
        breadcrumb_path = temp_info.path / "ABORTED"
        breadcrumb_data = {
            "run_id": run_id,
            "stage_name": stage_name,
            "error_message": error_message,
            "aborted_at": datetime.now(timezone.utc).isoformat(),
            "project_id": temp_info.project_id,
            "session_id": temp_info.session_id
        }
        
        try:
            with self.atomic_write(breadcrumb_path) as f:
                json.dump(breadcrumb_data, f, indent=2)
            
            self.logger.error(f"Marked run {run_id} as aborted in stage {stage_name}: {error_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to create ABORTED breadcrumb for {run_id}: {e}")
    
    def cleanup_run_temp_directory(self, run_id: str, force: bool = False) -> bool:
        """
        Clean up temporary directory for a completed run.
        
        Args:
            run_id: Run identifier
            force: Force cleanup even if marked as aborted
            
        Returns:
            True if cleaned up successfully
        """
        if run_id not in self._active_temp_dirs:
            self.logger.warning(f"Cannot cleanup run {run_id} - not registered")
            return False
        
        temp_info = self._active_temp_dirs[run_id]
        
        # Don't cleanup aborted runs unless forced
        if temp_info.is_aborted and not force:
            self.logger.info(f"Skipping cleanup of aborted run {run_id}")
            return False
        
        try:
            if temp_info.path.exists():
                shutil.rmtree(temp_info.path)
            
            # Remove from active registry
            with self._lock:
                del self._active_temp_dirs[run_id]
            
            self.logger.info(f"Cleaned up temp directory for run {run_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp directory for run {run_id}: {e}")
            return False
    
    def run_janitor_cleanup(self) -> Dict[str, Any]:
        """
        Run janitor process to clean up old temporary directories.
        
        Returns:
            Cleanup statistics
        """
        cleanup_stats = {
            'directories_scanned': 0,
            'directories_removed': 0,
            'directories_failed': 0,
            'bytes_freed': 0,
            'errors': []
        }
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        
        self.logger.info(f"Starting janitor cleanup (retention: {self.retention_hours}h)")
        
        try:
            for item in self.base_temp_dir.iterdir():
                if not item.is_dir():
                    continue
                
                cleanup_stats['directories_scanned'] += 1
                
                try:
                    # Get directory creation time
                    stat_info = item.stat()
                    created_time = datetime.fromtimestamp(stat_info.st_ctime, timezone.utc)
                    
                    # Check if directory is old enough for cleanup
                    if created_time < cutoff_time:
                        # Calculate size before removal
                        dir_size = sum(
                            f.stat().st_size for f in item.rglob('*') if f.is_file()
                        )
                        
                        # Remove directory
                        shutil.rmtree(item)
                        
                        cleanup_stats['directories_removed'] += 1
                        cleanup_stats['bytes_freed'] += dir_size
                        
                        self.logger.info(f"Cleaned up old temp directory: {item}")
                
                except Exception as e:
                    cleanup_stats['directories_failed'] += 1
                    cleanup_stats['errors'].append(f"{item}: {str(e)}")
                    self.logger.error(f"Failed to cleanup {item}: {e}")
        
        except Exception as e:
            self.logger.error(f"Janitor cleanup failed: {e}")
            cleanup_stats['errors'].append(f"General failure: {str(e)}")
        
        self.logger.info(
            f"Janitor cleanup completed: {cleanup_stats['directories_removed']} removed, "
            f"{cleanup_stats['directories_failed']} failed, "
            f"{cleanup_stats['bytes_freed']} bytes freed"
        )
        
        return cleanup_stats
    
    def scan_orphan_temp_files(self) -> List[Dict[str, Any]]:
        """
        Scan for orphaned temporary files that weren't cleaned up.
        
        Returns:
            List of orphan file information
        """
        orphans = []
        
        try:
            for temp_dir_path in self.base_temp_dir.rglob("*.tmp.*"):
                if temp_dir_path.is_file():
                    try:
                        stat_info = temp_dir_path.stat()
                        orphan_info = {
                            'path': str(temp_dir_path),
                            'size_bytes': stat_info.st_size,
                            'created_at': datetime.fromtimestamp(
                                stat_info.st_ctime, timezone.utc
                            ).isoformat(),
                            'modified_at': datetime.fromtimestamp(
                                stat_info.st_mtime, timezone.utc
                            ).isoformat()
                        }
                        orphans.append(orphan_info)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to stat orphan file {temp_dir_path}: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to scan for orphan temp files: {e}")
        
        with self._lock:
            self.stats.orphan_temp_files_count = len(orphans)
        
        return orphans
    
    def _log_commit_telemetry(self, final_path: Path, file_size: int, duration_ms: float):
        """
        Log telemetry data for atomic commit operations.
        
        Args:
            final_path: Final file path
            file_size: Size of file in bytes
            duration_ms: Operation duration in milliseconds
        """
        try:
            # Calculate SHA256 for integrity verification
            sha256_hash = hashlib.sha256()
            with open(final_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            file_sha256 = sha256_hash.hexdigest()
            
            # Log structured telemetry
            telemetry_data = {
                'final_path': str(final_path),
                'bytes': file_size,
                'sha256': file_sha256,
                'duration_ms': duration_ms,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.log_event("atomic_commit", telemetry_data)
            
            # Update stats
            with self._lock:
                self.stats.total_bytes_written += file_size
                self.stats.total_operations_duration_ms += duration_ms
        
        except Exception as e:
            self.logger.warning(f"Failed to log commit telemetry for {final_path}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current atomic I/O statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            stats_dict = {
                'atomic_commits_count': self.stats.atomic_commits_count,
                'atomic_commit_failures': self.stats.atomic_commit_failures,
                'temp_file_creations': self.stats.temp_file_creations,
                'safe_removals': self.stats.safe_removals,
                'orphan_temp_files_count': self.stats.orphan_temp_files_count,
                'total_bytes_written': self.stats.total_bytes_written,
                'total_operations_duration_ms': self.stats.total_operations_duration_ms,
                'active_temp_directories': len(self._active_temp_dirs),
                'base_temp_dir': str(self.base_temp_dir),
                'retention_hours': self.retention_hours
            }
        
        return stats_dict


# Global atomic I/O manager instance
_global_atomic_io_manager: Optional[AtomicIOManager] = None
_manager_lock = threading.Lock()


def get_atomic_io_manager(**kwargs) -> AtomicIOManager:
    """
    Get global atomic I/O manager instance (singleton).
    
    Args:
        **kwargs: Arguments for AtomicIOManager initialization
        
    Returns:
        Global AtomicIOManager instance
    """
    global _global_atomic_io_manager
    
    with _manager_lock:
        if _global_atomic_io_manager is None:
            _global_atomic_io_manager = AtomicIOManager(**kwargs)
        
        return _global_atomic_io_manager


def reset_atomic_io_manager():
    """Reset global atomic I/O manager (for testing)."""
    global _global_atomic_io_manager
    
    with _manager_lock:
        _global_atomic_io_manager = None


# Convenience functions for direct use
def open_temp_for(final_path: Union[str, Path], **kwargs):
    """Convenience function for creating temp file."""
    manager = get_atomic_io_manager()
    return manager.open_temp_for(final_path, **kwargs)


def commit_temp(temp_path: Union[str, Path], final_path: Union[str, Path]):
    """Convenience function for committing temp file."""
    manager = get_atomic_io_manager()
    return manager.commit_temp(temp_path, final_path)


def safe_remove(file_path: Union[str, Path]) -> bool:
    """Convenience function for safe file removal."""
    manager = get_atomic_io_manager()
    return manager.safe_remove(file_path)


@contextmanager
def atomic_write(final_path: Union[str, Path], **kwargs):
    """Convenience context manager for atomic writing."""
    manager = get_atomic_io_manager()
    with manager.atomic_write(final_path, **kwargs) as f:
        yield f


def create_run_temp_directory(run_id: Optional[str] = None, **kwargs) -> Path:
    """Convenience function for creating run temp directory."""
    if run_id is None:
        # Try to get run_id from global context
        run_context = get_global_run_context()
        if run_context:
            run_id = run_context.run_id
        else:
            raise ValueError("No run_id provided and no global run context available")
    
    manager = get_atomic_io_manager()
    return manager.create_run_temp_directory(run_id, **kwargs)


def get_run_temp_subdir(run_id: Optional[str] = None, 
                       scope: TempDirectoryScope = TempDirectoryScope.ARTIFACTS) -> Path:
    """Convenience function for getting run temp subdirectory."""
    if run_id is None:
        # Try to get run_id from global context
        run_context = get_global_run_context()
        if run_context:
            run_id = run_context.run_id
        else:
            raise ValueError("No run_id provided and no global run context available")
    
    manager = get_atomic_io_manager()
    return manager.get_run_temp_subdir(run_id, scope)