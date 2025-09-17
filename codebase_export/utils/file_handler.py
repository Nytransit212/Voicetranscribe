import os
import json
import tempfile
import shutil
import time
from typing import Dict, Any, Optional
from pathlib import Path

from utils.atomic_io import (
    get_atomic_io_manager, 
    atomic_write, 
    TempDirectoryScope,
    create_run_temp_directory,
    get_run_temp_subdir
)
from utils.enhanced_structured_logger import create_enhanced_logger
from core.run_context import get_global_run_context

class FileHandler:
    """
    Handles file operations for the ensemble transcription system with atomic I/O support.
    
    Now provides atomic file operations to prevent partial writes and corruption,
    with integrated temp directory management and automatic cleanup.
    """
    
    def __init__(self, base_dir: Optional[str] = None, 
                 use_atomic_io: bool = True,
                 run_id: Optional[str] = None,
                 session_id: Optional[str] = None):
        """
        Initialize FileHandler with atomic I/O support.
        
        Args:
            base_dir: Base directory for files (defaults to system temp)
            use_atomic_io: Whether to use atomic I/O operations
            run_id: Run identifier for temp directory management
            session_id: Session identifier
        """
        self.base_dir = base_dir or tempfile.gettempdir()
        self.session_dir = None
        self.use_atomic_io = use_atomic_io
        self.run_id = run_id
        self.session_id = session_id
        
        # Initialize atomic I/O manager if enabled
        if self.use_atomic_io:
            self.atomic_io = get_atomic_io_manager()
            self.logger = create_enhanced_logger("file_handler", run_id=run_id)
        else:
            self.atomic_io = None
            self.logger = None
        
        # Cache key for collision prevention
        self.cache_key = None
        
        # Auto-create run temp directory if run_id provided
        if self.use_atomic_io and self.run_id:
            try:
                self.create_run_temp_directory()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to auto-create run temp directory: {e}")
    
    def create_session_directory(self, session_id: Optional[str] = None) -> str:
        """
        Create a unique session directory for processing files.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Path to created session directory
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.session_dir = os.path.join(self.base_dir, session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['audio', 'diarization', 'asr', 'transcripts', 'captions']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.session_dir, subdir), exist_ok=True)
        
        return self.session_dir
    
    def save_json(self, data: Dict[str, Any], filename: str, subdir: str = '', 
                  cache_key: Optional[str] = None) -> str:
        """
        Save data as JSON file using atomic operations.
        
        Args:
            data: Data to save
            filename: Name of file (without extension)
            subdir: Subdirectory within session directory
            cache_key: Optional cache key for collision prevention
            
        Returns:
            Path to saved file
        """
        if not self.session_dir:
            raise ValueError("Session directory not created")
        
        file_path = os.path.join(self.session_dir, subdir, f"{filename}.json")
        
        # Use atomic write if available
        if self.use_atomic_io and self.atomic_io:
            try:
                with atomic_write(file_path, cache_key=cache_key or self.cache_key) as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                if self.logger:
                    self.logger.debug(f"Atomically saved JSON: {file_path}")
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Atomic JSON save failed for {file_path}: {e}")
                raise
        else:
            # Fallback to legacy method
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return file_path
    
    def save_text(self, text: str, filename: str, subdir: str = '', 
                  cache_key: Optional[str] = None) -> str:
        """
        Save text to file using atomic operations.
        
        Args:
            text: Text content to save
            filename: Name of file (with extension)
            subdir: Subdirectory within session directory
            cache_key: Optional cache key for collision prevention
            
        Returns:
            Path to saved file
        """
        if not self.session_dir:
            raise ValueError("Session directory not created")
        
        file_path = os.path.join(self.session_dir, subdir, filename)
        
        # Use atomic write if available
        if self.use_atomic_io and self.atomic_io:
            try:
                with atomic_write(file_path, cache_key=cache_key or self.cache_key) as f:
                    f.write(text)
                
                if self.logger:
                    self.logger.debug(f"Atomically saved text: {file_path}")
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Atomic text save failed for {file_path}: {e}")
                raise
        else:
            # Fallback to legacy method
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
        
        return file_path
    
    def load_json(self, filepath: str) -> Dict[str, Any]:
        """
        Load JSON data from file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_text(self, filepath: str) -> str:
        """
        Load text from file.
        
        Args:
            filepath: Path to text file
            
        Returns:
            File content as string
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def copy_file(self, source_path: str, dest_filename: str, subdir: str = '', 
                  use_atomic: bool = True) -> str:
        """
        Copy file to session directory with optional atomic operations.
        
        Args:
            source_path: Source file path
            dest_filename: Destination filename
            subdir: Subdirectory within session directory
            use_atomic: Whether to use atomic copy operations
            
        Returns:
            Path to copied file
        """
        if not self.session_dir:
            raise ValueError("Session directory not created")
        
        dest_path = os.path.join(self.session_dir, subdir, dest_filename)
        
        # Use atomic copy if available and requested
        if use_atomic and self.use_atomic_io and self.atomic_io:
            try:
                # Read source file and write atomically to destination
                with open(source_path, 'rb') as src:
                    with atomic_write(dest_path, mode='wb') as dst:
                        shutil.copyfileobj(src, dst)
                
                # Copy metadata
                shutil.copystat(source_path, dest_path)
                
                if self.logger:
                    self.logger.debug(f"Atomically copied file: {source_path} -> {dest_path}")
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Atomic file copy failed {source_path} -> {dest_path}: {e}")
                raise
        else:
            # Fallback to legacy method
            shutil.copy2(source_path, dest_path)
        
        return dest_path
    
    def get_file_path(self, filename: str, subdir: str = '') -> str:
        """
        Get full path for a file in the session directory.
        
        Args:
            filename: Name of file
            subdir: Subdirectory within session directory
            
        Returns:
            Full file path
        """
        if not self.session_dir:
            raise ValueError("Session directory not created")
        
        return os.path.join(self.session_dir, subdir, filename)
    
    def list_files(self, subdir: str = '', pattern: str = '*') -> list:
        """
        List files in session directory or subdirectory.
        
        Args:
            subdir: Subdirectory to search
            pattern: File pattern to match
            
        Returns:
            List of matching file paths
        """
        if not self.session_dir:
            return []
        
        search_dir = os.path.join(self.session_dir, subdir)
        if not os.path.exists(search_dir):
            return []
        
        path_obj = Path(search_dir)
        return list(path_obj.glob(pattern))
    
    def cleanup_session(self, force: bool = False):
        """
        Remove session directory and all contents with atomic I/O integration.
        
        Args:
            force: Force cleanup even if run is marked as aborted
        """
        if self.session_dir and os.path.exists(self.session_dir):
            try:
                # If using atomic I/O and we have a run_id, use atomic cleanup
                if self.use_atomic_io and self.atomic_io and self.run_id:
                    success = self.atomic_io.cleanup_run_temp_directory(self.run_id, force=force)
                    if success:
                        self.session_dir = None
                        if self.logger:
                            self.logger.info(f"Cleaned up session via atomic I/O: {self.run_id}")
                    else:
                        if self.logger:
                            self.logger.warning(f"Atomic cleanup failed for run {self.run_id}")
                else:
                    # Fallback to legacy cleanup
                    shutil.rmtree(self.session_dir)
                    self.session_dir = None
                    
            except Exception as e:
                error_msg = f"Warning: Could not clean up session directory: {e}"
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(error_msg)
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about current session.
        
        Returns:
            Session information dictionary
        """
        if not self.session_dir:
            return {'session_active': False}
        
        info = {
            'session_active': True,
            'session_directory': self.session_dir,
            'subdirectories': [],
            'total_files': 0,
            'total_size_bytes': 0
        }
        
        # Scan directory structure
        try:
            for root, dirs, files in os.walk(self.session_dir):
                relative_root = os.path.relpath(root, self.session_dir)
                if relative_root != '.':
                    info['subdirectories'].append(relative_root)
                
                for file in files:
                    file_path = os.path.join(root, file)
                    info['total_files'] += 1
                    info['total_size_bytes'] += os.path.getsize(file_path)
        
        except Exception as e:
            info['error'] = f"Could not scan directory: {e}"
        
        return info
    
    def create_run_temp_directory(self, run_id: Optional[str] = None, 
                                stage_name: str = "",
                                project_id: str = "",
                                session_id: Optional[str] = None) -> str:
        """
        Create a run-specific temporary directory using atomic I/O system.
        
        Args:
            run_id: Run identifier (uses instance run_id if not provided)
            stage_name: Processing stage name
            project_id: Project identifier
            session_id: Session identifier (uses instance session_id if not provided)
            
        Returns:
            Path to created run temp directory
        """
        if not self.use_atomic_io or not self.atomic_io:
            # Fallback to create_session_directory for legacy mode
            return self.create_session_directory(session_id)
        
        # Use provided or instance run_id
        actual_run_id = run_id or self.run_id
        if not actual_run_id:
            # Try to get from global context
            run_context = get_global_run_context()
            if run_context:
                actual_run_id = run_context.run_id
            else:
                raise ValueError("No run_id available for temp directory creation")
        
        # Use provided or instance session_id
        actual_session_id = session_id or self.session_id or f"session_{int(time.time())}"
        
        # Create run temp directory
        run_temp_path = self.atomic_io.create_run_temp_directory(
            run_id=actual_run_id,
            stage_name=stage_name,
            project_id=project_id,
            session_id=actual_session_id
        )
        
        # Set as session directory for compatibility
        self.session_dir = str(run_temp_path)
        self.run_id = actual_run_id
        
        if self.logger:
            self.logger.info(f"Created run temp directory: {run_temp_path}")
        
        return str(run_temp_path)
    
    def get_temp_subdir(self, scope: TempDirectoryScope, 
                       run_id: Optional[str] = None) -> str:
        """
        Get path to temporary subdirectory for specific scope.
        
        Args:
            scope: Temporary directory scope
            run_id: Run identifier (uses instance run_id if not provided)
            
        Returns:
            Path to temp subdirectory
        """
        if not self.use_atomic_io or not self.atomic_io:
            # Fallback for legacy mode
            if not self.session_dir:
                raise ValueError("No session directory available")
            subdir_path = os.path.join(self.session_dir, scope.value)
            os.makedirs(subdir_path, exist_ok=True)
            return subdir_path
        
        # Use provided or instance run_id
        actual_run_id = run_id or self.run_id
        if not actual_run_id:
            # Try to get from global context
            run_context = get_global_run_context()
            if run_context:
                actual_run_id = run_context.run_id
            else:
                raise ValueError("No run_id available for temp subdirectory")
        
        # Get temp subdirectory via atomic I/O
        subdir_path = get_run_temp_subdir(actual_run_id, scope)
        return str(subdir_path)
    
    def mark_run_aborted(self, stage_name: str, error_message: str, 
                        run_id: Optional[str] = None) -> None:
        """
        Mark current run as aborted with breadcrumb file.
        
        Args:
            stage_name: Stage where abortion occurred
            error_message: Error message
            run_id: Run identifier (uses instance run_id if not provided)
        """
        if not self.use_atomic_io or not self.atomic_io:
            if self.logger:
                self.logger.warning(f"Cannot mark run aborted - atomic I/O not available")
            return
        
        # Use provided or instance run_id
        actual_run_id = run_id or self.run_id
        if not actual_run_id:
            # Try to get from global context
            run_context = get_global_run_context()
            if run_context:
                actual_run_id = run_context.run_id
            else:
                if self.logger:
                    self.logger.error("No run_id available to mark as aborted")
                return
        
        # Mark run as aborted
        self.atomic_io.mark_run_aborted(actual_run_id, stage_name, error_message)
        
        if self.logger:
            self.logger.error(f"Marked run {actual_run_id} as aborted in stage {stage_name}")
    
    def set_cache_key(self, cache_key: str) -> None:
        """
        Set cache key for collision prevention in atomic operations.
        
        Args:
            cache_key: Cache key to use for collision prevention
        """
        self.cache_key = cache_key
        if self.logger:
            self.logger.debug(f"Set cache key: {cache_key[:12]}...")
    
    def get_atomic_io_stats(self) -> Dict[str, Any]:
        """
        Get atomic I/O statistics if available.
        
        Returns:
            Statistics dictionary or empty dict if atomic I/O not enabled
        """
        if self.use_atomic_io and self.atomic_io:
            return self.atomic_io.get_statistics()
        else:
            return {}
    
    def create_download_package(self, results: Dict[str, Any], 
                              cache_key: Optional[str] = None) -> str:
        """
        Create a downloadable package with all results using atomic operations.
        
        Args:
            results: Processing results dictionary
            cache_key: Optional cache key for collision prevention
            
        Returns:
            Path to created ZIP file
        """
        if not self.session_dir:
            raise ValueError("Session directory not created")
        
        # Use provided cache key or instance cache key
        actual_cache_key = cache_key or self.cache_key
        
        try:
            # Save all results to files atomically
            self.save_json(results['winner_transcript'], 'transcript', 'transcripts', actual_cache_key)
            self.save_text(results['winner_transcript_txt'], 'transcript.txt', 'transcripts', actual_cache_key)
            self.save_text(results['captions_vtt'], 'captions.vtt', 'captions', actual_cache_key)
            self.save_text(results['captions_srt'], 'captions.srt', 'captions', actual_cache_key)
            self.save_text(results['captions_ass'], 'captions.ass', 'captions', actual_cache_key)
            self.save_json(results['ensemble_audit'], 'ensemble_audit', 'transcripts', actual_cache_key)
            
            # Create ZIP package with atomic operations
            zip_filename = f"transcription_results_{int(time.time())}.zip"
            zip_path = os.path.join(self.base_dir, zip_filename)
            
            # Create ZIP atomically
            if self.use_atomic_io and self.atomic_io:
                # Create temp ZIP file
                temp_zip_path = zip_path + ".tmp"
                
                try:
                    shutil.make_archive(
                        temp_zip_path.replace('.zip.tmp', ''),
                        'zip',
                        self.session_dir
                    )
                    
                    # Add .tmp extension if shutil didn't
                    if not temp_zip_path.endswith('.tmp'):
                        temp_zip_path = temp_zip_path.replace('.zip', '') + '.zip'
                        final_temp_path = temp_zip_path + '.tmp'
                        os.rename(temp_zip_path, final_temp_path)
                        temp_zip_path = final_temp_path
                    
                    # Atomic commit
                    self.atomic_io.commit_temp(temp_zip_path, zip_path)
                    
                    if self.logger:
                        self.logger.info(f"Created download package atomically: {zip_path}")
                    
                except Exception as e:
                    # Cleanup temp file on failure
                    if os.path.exists(temp_zip_path):
                        self.atomic_io.safe_remove(temp_zip_path)
                    raise
            else:
                # Fallback to legacy method
                shutil.make_archive(
                    zip_path.replace('.zip', ''),
                    'zip',
                    self.session_dir
                )
            
            return zip_path
            
        except Exception as e:
            error_msg = f"Could not create download package: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise Exception(error_msg)
