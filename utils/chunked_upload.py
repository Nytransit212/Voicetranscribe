import os
import tempfile
import hashlib
import time
import json
from typing import Optional, Callable, Tuple, Dict, Any, List
from pathlib import Path
import streamlit as st


class ChunkedUploadManager:
    """
    Manages chunked file uploads to bypass nginx proxy size limitations.
    Breaks large files into smaller chunks, uploads them sequentially, and reassembles them.
    """
    
    def __init__(self, chunk_size_kb: int = 512, max_file_size_gb: int = 2):
        """
        Initialize chunked upload manager.
        
        Args:
            chunk_size_kb: Size of each chunk in KB (default 512KB, well under 1MB nginx limit)
            max_file_size_gb: Maximum allowed file size in GB
        """
        self.chunk_size_bytes = chunk_size_kb * 1024
        self.max_file_size_bytes = max_file_size_gb * 1024 * 1024 * 1024
        self.temp_dir = tempfile.mkdtemp(prefix='chunked_upload_')
        self.upload_session_id = None
        
    def validate_file(self, uploaded_file) -> Tuple[bool, str]:
        """
        Validate uploaded file size and type.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not uploaded_file:
            return False, "No file uploaded"
            
        # Check file type
        if not uploaded_file.name.lower().endswith('.mp4'):
            return False, "Only MP4 video files are supported"
            
        # Check file size
        file_size = uploaded_file.size
        if file_size > self.max_file_size_bytes:
            size_gb = file_size / (1024 * 1024 * 1024)
            return False, f"File too large ({size_gb:.1f}GB). Maximum allowed: {self.max_file_size_bytes // (1024*1024*1024)}GB"
            
        if file_size == 0:
            return False, "File appears to be empty"
            
        return True, f"File validation passed ({file_size / (1024*1024):.1f}MB)"
    
    def calculate_file_hash(self, file_data: bytes) -> str:
        """Calculate SHA-256 hash of file data for integrity verification."""
        return hashlib.sha256(file_data).hexdigest()
    
    def get_chunk_info(self, file_size: int) -> Dict[str, Any]:
        """
        Calculate chunk information for a given file size.
        
        Args:
            file_size: Size of file in bytes
            
        Returns:
            Dictionary with chunk count and size information
        """
        total_chunks = (file_size + self.chunk_size_bytes - 1) // self.chunk_size_bytes
        
        return {
            'total_chunks': total_chunks,
            'chunk_size_bytes': self.chunk_size_bytes,
            'chunk_size_kb': self.chunk_size_bytes // 1024,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'estimated_upload_time_seconds': max(total_chunks * 0.5, 10)  # Rough estimate
        }
    
    def create_upload_session(self, uploaded_file) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Create a new chunked upload session.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (success, message, session_info)
        """
        # Validate file first
        is_valid, validation_msg = self.validate_file(uploaded_file)
        if not is_valid:
            return False, validation_msg, {}
        
        # Generate unique session ID
        self.upload_session_id = f"upload_{int(time.time())}_{hash(uploaded_file.name) % 10000}"
        
        # Get file data and calculate hash
        file_data = uploaded_file.read()
        file_hash = self.calculate_file_hash(file_data)
        
        # Reset file pointer for chunking
        uploaded_file.seek(0)
        
        # Get chunk information
        chunk_info = self.get_chunk_info(len(file_data))
        
        # Create session info
        session_info = {
            'session_id': self.upload_session_id,
            'filename': uploaded_file.name,
            'file_hash': file_hash,
            'chunks_uploaded': 0,
            'chunks_failed': 0,
            'temp_dir': self.temp_dir,
            **chunk_info
        }
        
        return True, f"Upload session created for {uploaded_file.name}", session_info
    
    def upload_file_chunks(
        self, 
        uploaded_file, 
        session_info: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Upload file in chunks and reassemble.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            session_info: Session information from create_upload_session
            progress_callback: Optional callback for progress updates (chunk_num, total_chunks, message)
            
        Returns:
            Tuple of (success, message, assembled_file_path)
        """
        try:
            total_chunks = session_info['total_chunks']
            chunk_size = session_info['chunk_size_bytes']
            session_id = session_info['session_id']
            
            # Create session directory
            session_dir = os.path.join(self.temp_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Read entire file data
            file_data = uploaded_file.read()
            original_hash = self.calculate_file_hash(file_data)
            
            if progress_callback:
                progress_callback(0, total_chunks, f"Starting chunked upload ({total_chunks} chunks)")
            
            # Split into chunks and save
            chunk_paths = []
            chunks_uploaded = 0
            
            for chunk_num in range(total_chunks):
                start_byte = chunk_num * chunk_size
                end_byte = min(start_byte + chunk_size, len(file_data))
                chunk_data = file_data[start_byte:end_byte]
                
                # Save chunk to temporary file
                chunk_filename = f"chunk_{chunk_num:04d}.bin"
                chunk_path = os.path.join(session_dir, chunk_filename)
                
                try:
                    with open(chunk_path, 'wb') as chunk_file:
                        chunk_file.write(chunk_data)
                    
                    chunk_paths.append(chunk_path)
                    chunks_uploaded += 1
                    
                    if progress_callback:
                        progress_percent = int((chunks_uploaded / total_chunks) * 90)  # Reserve 10% for reassembly
                        progress_callback(
                            chunks_uploaded, 
                            total_chunks, 
                            f"Uploaded chunk {chunks_uploaded}/{total_chunks} ({len(chunk_data):,} bytes)"
                        )
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)
                    
                except Exception as e:
                    return False, f"Failed to save chunk {chunk_num}: {str(e)}", None
            
            if progress_callback:
                progress_callback(total_chunks, total_chunks, "Reassembling file...")
            
            # Reassemble chunks into final file
            final_filename = f"{session_id}_{uploaded_file.name}"
            final_path = os.path.join(session_dir, final_filename)
            
            try:
                with open(final_path, 'wb') as final_file:
                    for chunk_path in chunk_paths:
                        with open(chunk_path, 'rb') as chunk_file:
                            final_file.write(chunk_file.read())
                        # Clean up chunk file immediately
                        os.unlink(chunk_path)
                
                # Verify file integrity
                with open(final_path, 'rb') as verify_file:
                    reassembled_data = verify_file.read()
                    reassembled_hash = self.calculate_file_hash(reassembled_data)
                
                if reassembled_hash != original_hash:
                    return False, "File integrity check failed - upload corrupted", None
                
                if progress_callback:
                    progress_callback(total_chunks, total_chunks, f"✅ Upload complete: {uploaded_file.name}")
                
                return True, f"File uploaded successfully ({len(reassembled_data):,} bytes)", final_path
                
            except Exception as e:
                return False, f"Failed to reassemble file: {str(e)}", None
                
        except Exception as e:
            return False, f"Chunked upload failed: {str(e)}", None
    
    def cleanup_session(self, session_info: Optional[Dict[str, Any]] = None):
        """
        Clean up temporary files from upload session.
        
        Args:
            session_info: Optional session information. If not provided, cleans entire temp directory.
        """
        try:
            if session_info and 'session_id' in session_info:
                session_dir = os.path.join(self.temp_dir, session_info['session_id'])
                if os.path.exists(session_dir):
                    import shutil
                    shutil.rmtree(session_dir)
            else:
                # Clean up entire temp directory
                if os.path.exists(self.temp_dir):
                    import shutil
                    shutil.rmtree(self.temp_dir)
        except Exception:
            # Ignore cleanup errors - they're not critical
            pass
    
    def create_chunked_upload_interface(
        self, 
        help_text: str = "Upload your MP4 video file for ensemble transcription processing"
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Create Streamlit interface for chunked file upload.
        
        Args:
            help_text: Help text to display with the uploader
            
        Returns:
            Tuple of (upload_complete, file_path, session_info)
        """
        st.markdown("### 📁 File Upload")
        
        # File uploader (for file selection only, not actual upload)
        uploaded_file = st.file_uploader(
            "Choose an MP4 video file (up to 2GB)",
            type=['mp4'],
            help=help_text,
            key="chunked_file_uploader"
        )
        
        if uploaded_file is None:
            return False, None, None
        
        # Validate file
        is_valid, validation_msg = self.validate_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {validation_msg}")
            return False, None, None
        
        # Show file information
        file_size_mb = uploaded_file.size / (1024 * 1024)
        file_size_gb = file_size_mb / 1024
        
        if file_size_gb >= 1.0:
            size_display = f"{file_size_gb:.1f} GB"
        else:
            size_display = f"{file_size_mb:.1f} MB"
        
        st.info(f"📄 **{uploaded_file.name}** ({size_display})")
        
        # Show chunking information
        chunk_info = self.get_chunk_info(uploaded_file.size)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", chunk_info['total_chunks'])
        with col2:
            st.metric("Chunk Size", f"{chunk_info['chunk_size_kb']} KB")
        with col3:
            est_time = chunk_info['estimated_upload_time_seconds']
            if est_time >= 60:
                time_display = f"{est_time//60:.0f}m {est_time%60:.0f}s"
            else:
                time_display = f"{est_time:.0f}s"
            st.metric("Est. Upload Time", time_display)
        
        # Upload button and progress
        if st.button("🚀 Start Chunked Upload", key="start_chunked_upload"):
            # Create upload session
            success, message, session_info = self.create_upload_session(uploaded_file)
            
            if not success:
                st.error(f"❌ {message}")
                return False, None, None
            
            # Show upload progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(chunk_num: int, total_chunks: int, message: str):
                progress = min(int((chunk_num / total_chunks) * 100), 100)
                progress_bar.progress(progress)
                status_text.text(f"Progress: {chunk_num}/{total_chunks} chunks - {message}")
            
            # Perform chunked upload
            upload_success, upload_message, file_path = self.upload_file_chunks(
                uploaded_file, session_info, update_progress
            )
            
            if upload_success:
                progress_bar.progress(100)
                status_text.text("✅ Upload completed successfully!")
                st.success(f"🎉 {upload_message}")
                return True, file_path, session_info
            else:
                st.error(f"❌ Upload failed: {upload_message}")
                self.cleanup_session(session_info)
                return False, None, None
        
        return False, None, None


def create_chunked_uploader(
    chunk_size_kb: int = 512, 
    max_file_size_gb: int = 2,
    help_text: str = "Upload your MP4 video file for ensemble transcription processing"
) -> Tuple[bool, Optional[str], Optional[ChunkedUploadManager]]:
    """
    Convenience function to create a chunked uploader interface.
    
    Args:
        chunk_size_kb: Size of each chunk in KB
        max_file_size_gb: Maximum allowed file size in GB
        help_text: Help text for the uploader
        
    Returns:
        Tuple of (upload_complete, file_path, upload_manager)
    """
    # Initialize or get upload manager from session state
    if 'chunked_upload_manager' not in st.session_state:
        st.session_state.chunked_upload_manager = ChunkedUploadManager(chunk_size_kb, max_file_size_gb)
    
    upload_manager = st.session_state.chunked_upload_manager
    
    # Create interface
    upload_complete, file_path, session_info = upload_manager.create_chunked_upload_interface(help_text)
    
    # Store session info if upload is complete
    if upload_complete and session_info:
        st.session_state.chunked_upload_session = session_info
    
    return upload_complete, file_path, upload_manager