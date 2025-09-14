"""
Google Drive Handler for Service Account Authentication and Resumable Uploads
Handles large file uploads directly to Google Drive using service account credentials.
"""

import os
import io
import json
import time
import tempfile
import logging
from typing import Optional, Dict, Any, Callable, Tuple, Union, List
from pathlib import Path

import streamlit as st
from google.auth.credentials import Credentials
from google.auth import default
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
import google.auth.transport.requests

# Configure logging
logger = logging.getLogger(__name__)

class GoogleDriveHandler:
    """Handles Google Drive operations with service account authentication"""
    
    # Google Drive API scopes
    SCOPES = [
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.metadata'
    ]
    
    def __init__(self):
        """Initialize Google Drive client with service account credentials"""
        self.service = None
        self.credentials = None
        self.folder_id = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Google Drive API client using service account credentials"""
        try:
            # Get service account credentials from environment variable
            service_account_key = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY', '').strip()
            
            # Check if environment variable is properly set
            if not service_account_key:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_KEY environment variable is not set or is empty")
            
            # Validate that it looks like JSON
            if not service_account_key.startswith('{') or not service_account_key.endswith('}'):
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_KEY does not appear to contain valid JSON (should start with '{' and end with '}')")
            
            # Parse JSON credentials
            try:
                service_account_info = json.loads(service_account_key)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed. First 100 chars of key: {service_account_key[:100]}")
                raise ValueError(f"Invalid JSON in GOOGLE_SERVICE_ACCOUNT_KEY: {e}")
            
            # Validate required fields in service account JSON
            required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in service_account_info]
            if missing_fields:
                raise ValueError(f"Service account JSON is missing required fields: {missing_fields}")
            
            # Create credentials from service account info
            self.credentials = service_account.Credentials.from_service_account_info(
                service_account_info, 
                scopes=self.SCOPES
            )
            
            # Build Drive API service
            self.service = build('drive', 'v3', credentials=self.credentials)
            
            # Get target folder ID from environment variable
            self.folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID', '').strip()
            if not self.folder_id:
                logger.warning("GOOGLE_DRIVE_FOLDER_ID not set - files will be uploaded to root")
            else:
                logger.info(f"Google Drive folder ID configured: {self.folder_id}")
            
            logger.info("Google Drive client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive client: {e}")
            # Don't re-raise the exception - allow the app to continue without Google Drive
            self.service = None
            self.credentials = None
            self.folder_id = None
    
    def verify_credentials(self) -> Dict[str, Any]:
        """Verify service account credentials and folder access"""
        try:
            if self.service is None:
                return {
                    'status': 'error',
                    'error': 'Google Drive service not initialized'
                }
            
            # Test API access by getting about info
            about = self.service.about().get(fields="user,storageQuota").execute()
            
            result = {
                'status': 'success',
                'service_account_email': about.get('user', {}).get('emailAddress'),
                'storage_quota': about.get('storageQuota', {})
            }
            
            # Test folder access if folder_id is set
            if self.folder_id:
                try:
                    folder_info = self.service.files().get(
                        fileId=self.folder_id,
                        fields="id,name,parents,permissions"
                    ).execute()
                    
                    result['folder_access'] = True
                    result['folder_name'] = folder_info.get('name')
                    
                except HttpError as e:
                    if e.resp.status == 404:
                        result['folder_access'] = False
                        result['folder_error'] = 'Folder not found or no access'
                    else:
                        result['folder_access'] = False
                        result['folder_error'] = f'Access error: {e}'
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def upload_file(
        self, 
        file_path: Union[str, Path], 
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to Google Drive using resumable upload
        
        Args:
            file_path: Local path to file to upload
            filename: Optional custom filename for Google Drive
            progress_callback: Optional callback for progress updates (uploaded_bytes, total_bytes)
            
        Returns:
            Dictionary with upload results including file_id and shareable_link
        """
        try:
            if self.service is None:
                return {
                    'status': 'error',
                    'error': 'Google Drive service not initialized'
                }
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Use provided filename or file's actual name
            drive_filename = filename or file_path.name
            file_size = file_path.stat().st_size
            
            # Prepare file metadata
            file_metadata = {
                'name': drive_filename,
                'parents': [self.folder_id] if self.folder_id else []
            }
            
            # Determine MIME type based on file extension
            mime_type = self._get_mime_type(file_path.suffix.lower())
            
            logger.info(f"Starting upload: {drive_filename} ({file_size:,} bytes)")
            
            # Use resumable upload for large files
            if file_size > 10 * 1024 * 1024:  # 10MB threshold
                return self._resumable_upload(
                    file_path, file_metadata, mime_type, progress_callback
                )
            else:
                return self._simple_upload(
                    file_path, file_metadata, mime_type, progress_callback
                )
                
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _simple_upload(
        self,
        file_path: Path,
        file_metadata: Dict[str, Any],
        mime_type: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Simple upload for smaller files"""
        try:
            if self.service is None:
                raise ValueError("Google Drive service not initialized")
            
            with open(file_path, 'rb') as file_data:
                media = MediaIoBaseUpload(
                    file_data,
                    mimetype=mime_type,
                    resumable=False
                )
                
                # Execute upload
                file_obj = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id,name,size,webViewLink'
                ).execute()
                
                if progress_callback:
                    progress_callback(file_path.stat().st_size, file_path.stat().st_size)
                
                # File remains private - no public sharing
                
                return {
                    'status': 'success',
                    'file_id': file_obj['id'],
                    'filename': file_obj['name'],
                    'size': int(file_obj.get('size', 0)),
                    'web_view_link': file_obj.get('webViewLink'),
                    'download_link': None  # Private file - use authenticated API download
                }
                
        except Exception as e:
            logger.error(f"Simple upload failed: {e}")
            raise
    
    def _resumable_upload(
        self,
        file_path: Path,
        file_metadata: Dict[str, Any],
        mime_type: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Resumable upload for large files"""
        try:
            if self.service is None:
                raise ValueError("Google Drive service not initialized")
            
            file_size = file_path.stat().st_size
            chunk_size = 1024 * 1024 * 5  # 5MB chunks
            
            with open(file_path, 'rb') as file_data:
                media = MediaIoBaseUpload(
                    file_data,
                    mimetype=mime_type,
                    resumable=True,
                    chunksize=chunk_size
                )
                
                # Create resumable upload request
                request = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id,name,size,webViewLink'
                )
                
                # Execute resumable upload with progress tracking
                response = None
                uploaded_bytes = 0
                
                while response is None:
                    try:
                        status, response = request.next_chunk()
                        
                        if status:
                            uploaded_bytes = status.resumable_progress
                            if progress_callback:
                                progress_callback(uploaded_bytes, file_size)
                            
                            logger.debug(f"Upload progress: {uploaded_bytes:,} / {file_size:,} bytes")
                            
                    except HttpError as e:
                        if e.resp.status in [502, 503, 504]:
                            # Retry on server errors
                            time.sleep(2)
                            continue
                        else:
                            raise
                
                # File remains private - no public sharing
                
                logger.info(f"Upload completed: {response['name']}")
                
                return {
                    'status': 'success',
                    'file_id': response['id'],
                    'filename': response['name'],
                    'size': int(response.get('size', 0)),
                    'web_view_link': response.get('webViewLink'),
                    'download_link': None  # Private file - use authenticated API download
                }
                
        except Exception as e:
            logger.error(f"Resumable upload failed: {e}")
            raise
    
    
    def download_file(
        self, 
        file_id: str, 
        local_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """
        Download a file from Google Drive
        
        Args:
            file_id: Google Drive file ID
            local_path: Local path to save the file
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            if self.service is None:
                logger.error("Cannot download file: Google Drive service not initialized")
                return False
            
            # Get file metadata to check size
            file_metadata = self.service.files().get(
                fileId=file_id,
                fields='id,name,size'
            ).execute()
            
            file_size = int(file_metadata.get('size', 0))
            logger.info(f"Downloading {file_metadata['name']} ({file_size:,} bytes)")
            
            # Create download request
            request = self.service.files().get_media(fileId=file_id)
            
            # Download with progress tracking
            with open(local_path, 'wb') as local_file:
                downloader = MediaIoBaseDownload(local_file, request)
                
                done = False
                downloaded_bytes = 0
                
                while not done:
                    status, done = downloader.next_chunk()
                    
                    if status:
                        downloaded_bytes = int(status.resumable_progress)
                        if progress_callback:
                            progress_callback(downloaded_bytes, file_size)
                        
                        logger.debug(f"Download progress: {downloaded_bytes:,} / {file_size:,} bytes")
            
            logger.info(f"Download completed: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a file on Google Drive"""
        try:
            if self.service is None:
                logger.error("Cannot get file info: Google Drive service not initialized")
                return None
            
            file_metadata = self.service.files().get(
                fileId=file_id,
                fields='id,name,size,mimeType,createdTime,webViewLink'
            ).execute()
            
            return {
                'file_id': file_metadata['id'],
                'name': file_metadata['name'],
                'size': int(file_metadata.get('size', 0)),
                'mime_type': file_metadata['mimeType'],
                'created_time': file_metadata['createdTime'],
                'web_view_link': file_metadata.get('webViewLink'),
                'download_link': None  # Private file - use authenticated API download
            }
            
        except HttpError as e:
            logger.error(f"Failed to get file info for {file_id}: {e}")
            return None
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file from Google Drive"""
        try:
            if self.service is None:
                logger.error("Cannot delete file: Google Drive service not initialized")
                return False
            
            self.service.files().delete(fileId=file_id).execute()
            logger.info(f"Deleted file {file_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    def list_files_in_folder(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """List files in the configured folder"""
        try:
            if self.service is None:
                logger.error("Cannot list files: Google Drive service not initialized")
                return []
            
            query = f"parents='{self.folder_id}'" if self.folder_id else ""
            
            results = self.service.files().list(
                q=query,
                pageSize=max_results,
                fields="files(id,name,size,mimeType,createdTime,webViewLink)"
            ).execute()
            
            files = results.get('files', [])
            
            return [
                {
                    'file_id': f['id'],
                    'name': f['name'],
                    'size': int(f.get('size', 0)),
                    'mime_type': f['mimeType'],
                    'created_time': f['createdTime'],
                    'web_view_link': f.get('webViewLink'),
                    'download_link': None  # Private file - use authenticated API download
                }
                for f in files
            ]
            
        except HttpError as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    @staticmethod
    def _get_mime_type(file_extension: str) -> str:
        """Get MIME type for file extension"""
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.flac': 'audio/flac',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.json': 'application/json'
        }
        
        return mime_types.get(file_extension.lower(), 'application/octet-stream')
    
    @staticmethod
    def extract_file_id_from_url(drive_url: str) -> Optional[str]:
        """Extract file ID from various Google Drive URL formats"""
        import re
        
        # Common Google Drive URL patterns
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',  # Standard sharing URL
            r'id=([a-zA-Z0-9-_]+)',       # Direct download URL
            r'/d/([a-zA-Z0-9-_]+)',       # Short URL format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drive_url)
            if match:
                return match.group(1)
        
        # If it's already just a file ID
        if re.match(r'^[a-zA-Z0-9-_]+$', drive_url):
            return drive_url
        
        return None


# Singleton instance for easy access
_drive_handler = None

def get_drive_handler() -> Optional[GoogleDriveHandler]:
    """Get singleton Google Drive handler instance, returns None if not available"""
    global _drive_handler
    if _drive_handler is None:
        try:
            _drive_handler = GoogleDriveHandler()
            # Verify that the handler was properly initialized
            if _drive_handler.service is None:
                logger.warning("Google Drive handler created but service is not available")
                return None
        except Exception as e:
            logger.warning(f"Google Drive handler not available: {e}")
            return None
    return _drive_handler

# Convenience functions for Streamlit integration
def upload_file_to_drive(
    file_path: str,
    filename: Optional[str] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Upload file to Google Drive with Streamlit progress tracking
    
    Args:
        file_path: Local file path
        filename: Optional custom filename
        show_progress: Whether to show Streamlit progress bar
        
    Returns:
        Upload result dictionary
    """
    drive_handler = get_drive_handler()
    
    if drive_handler is None:
        error_msg = "Google Drive is not available - please check your configuration"
        if show_progress:
            st.error(f"❌ {error_msg}")
        return {
            'status': 'error',
            'error': error_msg
        }
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(uploaded_bytes: int, total_bytes: int):
            progress = uploaded_bytes / total_bytes if total_bytes > 0 else 0
            progress_bar.progress(progress)
            status_text.text(
                f"Uploading: {uploaded_bytes:,} / {total_bytes:,} bytes "
                f"({progress:.1%})"
            )
        
        result = drive_handler.upload_file(file_path, filename, progress_callback)
        
        if result['status'] == 'success':
            progress_bar.progress(1.0)
            status_text.text("✅ Upload completed successfully!")
        else:
            status_text.text(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
            
        return result
    else:
        return drive_handler.upload_file(file_path, filename)

def download_file_from_drive(
    file_id_or_url: str,
    local_path: str,
    show_progress: bool = True
) -> bool:
    """
    Download file from Google Drive with Streamlit progress tracking
    
    Args:
        file_id_or_url: Google Drive file ID or sharing URL
        local_path: Local path to save file
        show_progress: Whether to show Streamlit progress bar
        
    Returns:
        True if download successful
    """
    drive_handler = get_drive_handler()
    
    if drive_handler is None:
        if show_progress:
            st.error("❌ Google Drive is not available - please check your configuration")
        return False
    
    # Extract file ID if URL provided
    file_id = drive_handler.extract_file_id_from_url(file_id_or_url)
    if not file_id:
        if show_progress:
            st.error("❌ Invalid Google Drive URL or file ID")
        return False
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(downloaded_bytes: int, total_bytes: int):
            progress = downloaded_bytes / total_bytes if total_bytes > 0 else 0
            progress_bar.progress(progress)
            status_text.text(
                f"Downloading: {downloaded_bytes:,} / {total_bytes:,} bytes "
                f"({progress:.1%})"
            )
        
        success = drive_handler.download_file(file_id, local_path, progress_callback)
        
        if success:
            progress_bar.progress(1.0)
            status_text.text("✅ Download completed successfully!")
        else:
            status_text.text("❌ Download failed!")
            
        return success
    else:
        return drive_handler.download_file(file_id, local_path)

def verify_drive_setup() -> Dict[str, Any]:
    """Verify Google Drive setup and return status"""
    try:
        drive_handler = get_drive_handler()
        if drive_handler is None:
            return {
                'status': 'error',
                'error': 'Google Drive handler not available - check service account configuration'
            }
        return drive_handler.verify_credentials()
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }