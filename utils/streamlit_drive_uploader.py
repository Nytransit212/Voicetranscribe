"""
Streamlit Google Drive Uploader Component
Replaces st.file_uploader with Google Drive upload functionality for large files.
"""

import os
import tempfile
import threading
import time
from typing import Optional, Dict, Any, Callable
import streamlit as st
from utils.google_drive_handler import get_drive_handler, verify_drive_setup

class StreamlitDriveUploader:
    """Custom Streamlit component for Google Drive uploads"""
    
    def __init__(self):
        """Initialize the uploader component"""
        self.drive_handler = None
        self._initialize_drive()
    
    def _initialize_drive(self):
        """Initialize Google Drive handler and verify setup"""
        try:
            self.drive_handler = get_drive_handler()
            return True
        except Exception as e:
            st.error(f"❌ Google Drive setup error: {e}")
            return False
    
    def render_upload_interface(
        self, 
        accept_types: list = ['.mp4', '.avi', '.mov', '.mkv'],
        max_size_mb: float = 200.0
    ) -> Optional[Dict[str, Any]]:
        """
        Render Google Drive upload interface
        
        Args:
            accept_types: List of accepted file extensions
            max_size_mb: Maximum file size in MB (server limit: 200MB)
            
        Returns:
            Upload result dictionary if successful, None otherwise
        """
        st.header("📁 Upload Video File to Google Drive")
        
        # Verify Google Drive setup first
        setup_status = verify_drive_setup()
        
        if setup_status['status'] == 'error':
            st.error(f"❌ Google Drive not configured: {setup_status['error']}")
            st.markdown("""
            **Required Configuration:**
            1. Set `GOOGLE_SERVICE_ACCOUNT_KEY` environment variable with service account JSON
            2. Set `GOOGLE_DRIVE_FOLDER_ID` environment variable with target folder ID
            3. Share the Google Drive folder with the service account email
            """)
            return None
        
        # Show setup status
        with st.expander("🔧 Google Drive Configuration Status", expanded=False):
            st.success("✅ Google Drive API authenticated successfully")
            
            if setup_status.get('service_account_email'):
                st.info(f"📧 Service Account: {setup_status['service_account_email']}")
            
            if setup_status.get('folder_access'):
                st.success(f"📁 Folder Access: ✅ {setup_status.get('folder_name', 'Configured')}")
            elif 'folder_error' in setup_status:
                st.warning(f"📁 Folder Access: ⚠️ {setup_status['folder_error']}")
            else:
                st.info("📁 Folder: Will upload to root directory")
            
            quota = setup_status.get('storage_quota', {})
            if quota:
                used_gb = int(quota.get('usage', 0)) / (1024**3)
                limit_gb = int(quota.get('limit', 0)) / (1024**3) if quota.get('limit') else None
                
                if limit_gb:
                    st.info(f"💾 Storage: {used_gb:.1f}GB / {limit_gb:.1f}GB used")
                else:
                    st.info(f"💾 Storage: {used_gb:.1f}GB used")
        
        # File upload section
        st.markdown("---")
        
        # Upload method selection with clear guidance
        st.markdown("""
        **📊 Choose Upload Method Based on File Size:**
        
        - **Files ≤ 200MB**: Use local upload (faster, direct from your computer)
        - **Files > 200MB**: Use Google Drive URL (bypasses server size limits)
        """)
        
        upload_method = st.radio(
            "🚀 Upload Method",
            options=["local_file", "drive_url"],
            format_func=lambda x: {
                "local_file": "📁 Upload Local File (≤ 200MB) - Direct Upload",
                "drive_url": "🔗 Use Google Drive File (> 200MB) - For Large Files"
            }[x],
            help="Select based on your file size. Local uploads are limited to 200MB by server configuration."
        )
        
        if upload_method == "local_file":
            return self._render_local_upload(accept_types, max_size_mb)
        else:
            return self._render_drive_url_input()
    
    def _render_local_upload(
        self, 
        accept_types: list, 
        max_size_mb: float
    ) -> Optional[Dict[str, Any]]:
        """Render local file upload interface"""
        
        # File uploader with corrected messaging
        st.markdown(f"""
        **📤 Upload Small to Medium Files (≤ 200MB)**
        - Supported formats: {', '.join(accept_types)}
        - **Maximum size: 200MB** (server limitation)
        - Files are uploaded to your Google Drive folder
        - For files > 200MB, use the Drive URL method instead
        
        ⚠️ **Important**: Server configuration limits uploads to 200MB. For larger files, upload to Google Drive first and use the Drive URL option.
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=[ext.lstrip('.') for ext in accept_types],
            help=f"Upload video files up to {max_size_mb}MB. Files will be uploaded directly to Google Drive."
        )
        
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            file_size_gb = file_size_mb / 1024
            
            # Show file info
            st.success(f"✅ File selected: **{uploaded_file.name}** ({file_size_mb:.1f}MB)")
            
            # Check file size against server limits
            if file_size_mb > 200:
                st.error(f"❌ File too large ({file_size_mb:.1f}MB). Server limit is 200MB.")
                st.warning("""
                💡 **For files > 200MB:**
                1. Upload your file to Google Drive manually
                2. Share it with the service account email
                3. Use the "Google Drive URL" option above
                """)
                return None
            
            # Upload button
            col1, col2 = st.columns([1, 3])
            
            with col1:
                upload_button = st.button(
                    "🚀 Upload to Google Drive",
                    type="primary",
                    help="Upload file directly to Google Drive using resumable upload"
                )
            
            with col2:
                if file_size_mb > 100:
                    st.warning(f"⚠️ Large file ({file_size_mb:.1f}MB) - consider using Drive URL for files > 200MB")
            
            if upload_button:
                return self._perform_upload(uploaded_file)
        
        return None
    
    def _render_drive_url_input(self) -> Optional[Dict[str, Any]]:
        """Render Google Drive URL input interface"""
        
        st.markdown("""
        **🔗 Use Large Files from Google Drive (Recommended for > 200MB)**
        - **Best for files > 200MB** (bypasses server upload limits)
        - Upload your large file to Google Drive first
        - Paste the Google Drive sharing URL or file ID below
        - File must be shared with the service account
        - Supports various Google Drive URL formats
        
        💡 **Tip**: This method works for files of any size (GB+ files are fine)
        """)
        
        drive_url = st.text_input(
            "Google Drive URL or File ID",
            placeholder="https://drive.google.com/file/d/1ABC123xyz/view?usp=sharing",
            help="Paste a Google Drive sharing URL or just the file ID"
        )
        
        if drive_url.strip():
            if self.drive_handler is None:
                st.error("❌ Google Drive handler not initialized. Please check configuration.")
                return None
            
            # Extract file ID and validate
            file_id = self.drive_handler.extract_file_id_from_url(drive_url.strip())
            
            if not file_id:
                st.error("❌ Invalid Google Drive URL or file ID format")
                return None
            
            # Get file info
            with st.spinner("🔍 Checking file accessibility..."):
                file_info = self.drive_handler.get_file_info(file_id)
            
            if not file_info:
                st.error("❌ File not found or not accessible to service account")
                st.markdown("""
                **Troubleshooting:**
                1. Ensure the file is shared with the service account email
                2. Check that the file ID/URL is correct
                3. Verify the file exists and isn't deleted
                """)
                return None
            
            # Show file info and confirm
            st.success(f"✅ File found: **{file_info['name']}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📊 Size: {file_info['size'] / (1024*1024):.1f}MB")
            with col2:
                st.info(f"📅 Created: {file_info['created_time'][:10]}")
            
            if st.button("✅ Use This File", type="primary"):
                return {
                    'status': 'success',
                    'source': 'existing_drive_file',
                    'file_id': file_id,
                    'filename': file_info['name'],
                    'size': file_info['size'],
                    'download_link': file_info['download_link'],
                    'web_view_link': file_info['web_view_link']
                }
        
        return None
    
    def _perform_upload(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """Perform the actual upload to Google Drive"""
        
        # Create progress indicators
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            try:
                start_time = time.time()
                
                def progress_callback(uploaded_bytes: int, total_bytes: int):
                    """Update progress indicators"""
                    if total_bytes > 0:
                        progress = uploaded_bytes / total_bytes
                        progress_bar.progress(progress)
                        
                        # Calculate speed and ETA
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 0 and uploaded_bytes > 0:
                            speed_mbps = (uploaded_bytes / (1024*1024)) / elapsed_time
                            remaining_bytes = total_bytes - uploaded_bytes
                            eta_seconds = remaining_bytes / (uploaded_bytes / elapsed_time) if uploaded_bytes > 0 else 0
                            
                            status_text.text(
                                f"📤 Uploading: {uploaded_bytes:,} / {total_bytes:,} bytes ({progress:.1%})"
                            )
                            time_text.text(
                                f"⚡ Speed: {speed_mbps:.1f} MB/s | ⏱️ ETA: {eta_seconds:.0f}s"
                            )
                
                # Perform upload
                status_text.text("🚀 Starting upload to Google Drive...")
                
                if self.drive_handler is None:
                    raise Exception("Google Drive handler not initialized. Please check configuration.")
                
                result = self.drive_handler.upload_file(
                    temp_path,
                    uploaded_file.name,
                    progress_callback
                )
                
                if result['status'] == 'success':
                    progress_bar.progress(1.0)
                    status_text.text("✅ Upload completed successfully!")
                    time_text.text(f"⏱️ Total time: {time.time() - start_time:.1f}s")
                    
                    # Show success info
                    st.success(f"🎉 **{result['filename']}** uploaded successfully to Google Drive!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"📊 Size: {result['size'] / (1024*1024):.1f}MB")
                    with col2:
                        if st.button("🔗 View in Google Drive"):
                            st.markdown(f"[Open in Google Drive]({result['web_view_link']})")
                    
                    # Add source info for processing
                    result['source'] = 'uploaded_to_drive'
                    return result
                
                else:
                    status_text.text(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                    st.error(f"Upload failed: {result.get('error', 'Unknown error')}")
                    return None
                    
            except Exception as e:
                status_text.text(f"❌ Upload error: {str(e)}")
                st.error(f"Upload error: {str(e)}")
                return None
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def show_recent_uploads(self, max_files: int = 10):
        """Show recent uploads in the configured folder"""
        if not self.drive_handler:
            return
        
        with st.expander("📂 Recent Files in Google Drive Folder", expanded=False):
            with st.spinner("Loading recent files..."):
                files = self.drive_handler.list_files_in_folder(max_files)
            
            if not files:
                st.info("No files found in the configured folder")
                return
            
            st.markdown(f"**{len(files)} recent files:**")
            
            for i, file_info in enumerate(files[:max_files]):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(f"📄 {file_info['name']}")
                
                with col2:
                    st.text(f"{file_info['size'] / (1024*1024):.1f}MB")
                
                with col3:
                    if st.button("Use", key=f"use_file_{i}"):
                        return {
                            'status': 'success',
                            'source': 'existing_drive_file',
                            'file_id': file_info['file_id'],
                            'filename': file_info['name'],
                            'size': file_info['size'],
                            'download_link': file_info['download_link'],
                            'web_view_link': file_info['web_view_link']
                        }

# Singleton instance
_drive_uploader = None

def get_drive_uploader() -> StreamlitDriveUploader:
    """Get singleton drive uploader instance"""
    global _drive_uploader
    if _drive_uploader is None:
        _drive_uploader = StreamlitDriveUploader()
    return _drive_uploader