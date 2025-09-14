import os
import json
import tempfile
import shutil
import time
from typing import Dict, Any, Optional
from pathlib import Path

class FileHandler:
    """Handles file operations for the ensemble transcription system"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or tempfile.gettempdir()
        self.session_dir = None
    
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
    
    def save_json(self, data: Dict[str, Any], filename: str, subdir: str = '') -> str:
        """
        Save data as JSON file.
        
        Args:
            data: Data to save
            filename: Name of file (without extension)
            subdir: Subdirectory within session directory
            
        Returns:
            Path to saved file
        """
        if not self.session_dir:
            raise ValueError("Session directory not created")
        
        file_path = os.path.join(self.session_dir, subdir, f"{filename}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return file_path
    
    def save_text(self, text: str, filename: str, subdir: str = '') -> str:
        """
        Save text to file.
        
        Args:
            text: Text content to save
            filename: Name of file (with extension)
            subdir: Subdirectory within session directory
            
        Returns:
            Path to saved file
        """
        if not self.session_dir:
            raise ValueError("Session directory not created")
        
        file_path = os.path.join(self.session_dir, subdir, filename)
        
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
    
    def copy_file(self, source_path: str, dest_filename: str, subdir: str = '') -> str:
        """
        Copy file to session directory.
        
        Args:
            source_path: Source file path
            dest_filename: Destination filename
            subdir: Subdirectory within session directory
            
        Returns:
            Path to copied file
        """
        if not self.session_dir:
            raise ValueError("Session directory not created")
        
        dest_path = os.path.join(self.session_dir, subdir, dest_filename)
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
    
    def cleanup_session(self):
        """Remove session directory and all contents"""
        if self.session_dir and os.path.exists(self.session_dir):
            try:
                shutil.rmtree(self.session_dir)
                self.session_dir = None
            except Exception as e:
                print(f"Warning: Could not clean up session directory: {e}")
    
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
    
    def create_download_package(self, results: Dict[str, Any]) -> str:
        """
        Create a downloadable package with all results.
        
        Args:
            results: Processing results dictionary
            
        Returns:
            Path to created ZIP file
        """
        if not self.session_dir:
            raise ValueError("Session directory not created")
        
        # Save all results to files
        self.save_json(results['winner_transcript'], 'transcript', 'transcripts')
        self.save_text(results['winner_transcript_txt'], 'transcript.txt', 'transcripts')
        self.save_text(results['captions_vtt'], 'captions.vtt', 'captions')
        self.save_text(results['captions_srt'], 'captions.srt', 'captions')
        self.save_text(results['captions_ass'], 'captions.ass', 'captions')
        self.save_json(results['ensemble_audit'], 'ensemble_audit', 'transcripts')
        
        # Create ZIP package
        zip_path = os.path.join(self.base_dir, f"transcription_results_{int(time.time())}.zip")
        
        try:
            shutil.make_archive(
                zip_path.replace('.zip', ''),
                'zip',
                self.session_dir
            )
            return zip_path
        except Exception as e:
            raise Exception(f"Could not create download package: {e}")
