import os
import uuid
import json
import time
import tempfile
import requests
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import streamlit as st
from utils.enhanced_structured_logger import create_enhanced_logger

# Create logger for S3 operations
s3_logger = create_enhanced_logger("s3_direct_upload")


class S3DirectUploader:
    """
    Manages direct-to-S3 uploads using pre-signed URLs to bypass nginx proxy limitations.
    Supports files up to 5GB with browser-based progress tracking.
    """
    
    def __init__(
        self, 
        bucket_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = 'us-east-1'
    ):
        """
        Initialize S3 direct uploader.
        
        Args:
            bucket_name: S3 bucket name (defaults to env var S3_BUCKET_NAME)
            aws_access_key_id: AWS access key (defaults to env var AWS_ACCESS_KEY_ID)  
            aws_secret_access_key: AWS secret key (defaults to env var AWS_SECRET_ACCESS_KEY)
            aws_region: AWS region (defaults to us-east-1)
        """
        # Get credentials from parameters or environment variables
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME')
        self.aws_access_key_id = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = aws_region
        
        # Validate required credentials
        if not all([self.bucket_name, self.aws_access_key_id, self.aws_secret_access_key]):
            missing = []
            if not self.bucket_name:
                missing.append('S3_BUCKET_NAME')
            if not self.aws_access_key_id:
                missing.append('AWS_ACCESS_KEY_ID')
            if not self.aws_secret_access_key:
                missing.append('AWS_SECRET_ACCESS_KEY')
            
            raise ValueError(f"Missing required S3 credentials: {', '.join(missing)}")
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            s3_logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            raise ValueError("Invalid AWS credentials provided")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"S3 bucket '{self.bucket_name}' not found")
            elif error_code == '403':
                raise ValueError(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                raise ValueError(f"S3 connection error: {e}")
    
    def validate_file_for_upload(self, file_name: str, file_size: int) -> Tuple[bool, str]:
        """
        Validate file for S3 upload.
        
        Args:
            file_name: Name of the file
            file_size: Size of file in bytes
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check file extension
        if not file_name.lower().endswith('.mp4'):
            return False, "Only MP4 video files are supported"
        
        # Check file size limits
        max_size_gb = 5
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        if file_size <= 0:
            return False, "File appears to be empty"
        
        if file_size > max_size_bytes:
            size_gb = file_size / (1024 * 1024 * 1024)
            return False, f"File too large ({size_gb:.1f}GB). Maximum allowed: {max_size_gb}GB"
        
        return True, f"File validation passed ({file_size / (1024*1024):.1f}MB)"
    
    def generate_upload_key(self, original_filename: str) -> str:
        """
        Generate unique S3 key for upload.
        
        Args:
            original_filename: Original name of the file
            
        Returns:
            Unique S3 key for the upload
        """
        # Generate unique identifier
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean filename (keep only alphanumeric, dots, dashes, underscores)
        clean_filename = "".join(c for c in original_filename if c.isalnum() or c in '.-_')
        
        # Construct S3 key with organized structure
        s3_key = f"uploads/{timestamp}/{unique_id}_{clean_filename}"
        
        return s3_key
    
    def generate_presigned_upload_url(
        self, 
        file_name: str, 
        file_size: int,
        expiration_minutes: int = 60
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Generate pre-signed URL for direct browser upload to S3.
        
        Args:
            file_name: Name of the file to upload
            file_size: Size of file in bytes
            expiration_minutes: URL expiration time in minutes (default 60)
            
        Returns:
            Tuple of (success, message, upload_info)
        """
        try:
            # Validate file first
            is_valid, validation_msg = self.validate_file_for_upload(file_name, file_size)
            if not is_valid:
                return False, validation_msg, {}
            
            # Generate unique S3 key
            s3_key = self.generate_upload_key(file_name)
            
            # Set upload conditions
            conditions = [
                {"bucket": self.bucket_name},
                {"key": s3_key},
                {"Content-Type": "video/mp4"},
                ["content-length-range", 1, file_size + 1024]  # Allow small buffer
            ]
            
            # Generate presigned POST
            presigned_post = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=s3_key,
                Fields={
                    "Content-Type": "video/mp4"
                },
                Conditions=conditions,
                ExpiresIn=expiration_minutes * 60
            )
            
            # Create upload info
            upload_info = {
                'upload_url': presigned_post['url'],
                'fields': presigned_post['fields'],
                's3_key': s3_key,
                'bucket_name': self.bucket_name,
                'file_name': file_name,
                'file_size': file_size,
                'expires_at': (datetime.now() + timedelta(minutes=expiration_minutes)).isoformat(),
                'upload_id': str(uuid.uuid4())
            }
            
            s3_logger.info(f"Generated presigned upload URL for {file_name} -> {s3_key}")
            
            return True, f"Upload URL generated for {file_name}", upload_info
            
        except Exception as e:
            s3_logger.error(f"Error generating presigned URL: {e}")
            return False, f"Failed to generate upload URL: {str(e)}", {}
    
    def generate_generic_presigned_url(
        self,
        expiration_minutes: int = 60,
        max_file_size_gb: int = 5
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Generate pre-signed URL without specific file metadata for client-side uploads.
        The client will provide file details when uploading.
        
        Args:
            expiration_minutes: URL expiration time in minutes (default 60)
            max_file_size_gb: Maximum file size in GB (default 5)
            
        Returns:
            Tuple of (success, message, upload_config)
        """
        try:
            # Generate unique session ID for this upload opportunity
            session_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a template key that will be used by client
            key_template = f"uploads/{timestamp}/{session_id}_{{filename}}"
            
            # Set maximum file size conditions  
            max_file_size_bytes = max_file_size_gb * 1024 * 1024 * 1024
            
            # Create upload configuration for client-side use
            upload_config = {
                'bucket_name': self.bucket_name,
                'aws_region': self.aws_region,
                'key_template': key_template,
                'session_id': session_id,
                'max_file_size_bytes': max_file_size_bytes,
                'max_file_size_gb': max_file_size_gb,
                'expires_at': (datetime.now() + timedelta(minutes=expiration_minutes)).isoformat(),
                'expiration_minutes': expiration_minutes,
                'allowed_content_types': ['video/mp4'],
                'upload_endpoint': '/generate_upload_url'  # Endpoint for client to request specific URLs
            }
            
            s3_logger.info(f"Generated generic upload config with session {session_id}")
            
            return True, f"Upload configuration generated (expires in {expiration_minutes}m)", upload_config
            
        except Exception as e:
            s3_logger.error(f"Error generating generic upload config: {e}")
            return False, f"Failed to generate upload configuration: {str(e)}", {}
    
    def generate_specific_presigned_url_for_client(
        self,
        session_id: str,
        file_name: str,
        file_size: int,
        expiration_minutes: int = 60
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Generate specific pre-signed URL for a client-selected file.
        This is called via API when client provides file details.
        
        Args:
            session_id: Session ID from initial upload config
            file_name: Name of the file to upload
            file_size: Size of file in bytes
            expiration_minutes: URL expiration time in minutes
            
        Returns:
            Tuple of (success, message, upload_info)
        """
        try:
            # Validate file
            is_valid, validation_msg = self.validate_file_for_upload(file_name, file_size)
            if not is_valid:
                return False, validation_msg, {}
            
            # Generate S3 key using session ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_filename = "".join(c for c in file_name if c.isalnum() or c in '.-_')
            s3_key = f"uploads/{timestamp}/{session_id}_{clean_filename}"
            
            # Set upload conditions
            conditions = [
                {"bucket": self.bucket_name},
                {"key": s3_key},
                {"Content-Type": "video/mp4"},
                ["content-length-range", 1, file_size + 1024]  # Allow small buffer
            ]
            
            # Generate presigned POST
            presigned_post = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=s3_key,
                Fields={
                    "Content-Type": "video/mp4"
                },
                Conditions=conditions,
                ExpiresIn=expiration_minutes * 60
            )
            
            # Create upload info
            upload_info = {
                'upload_url': presigned_post['url'],
                'fields': presigned_post['fields'],
                's3_key': s3_key,
                'bucket_name': self.bucket_name,
                'file_name': file_name,
                'file_size': file_size,
                'session_id': session_id,
                'expires_at': (datetime.now() + timedelta(minutes=expiration_minutes)).isoformat(),
                'upload_id': str(uuid.uuid4())
            }
            
            s3_logger.info(f"Generated specific presigned URL for {file_name} -> {s3_key} (session {session_id})")
            
            return True, f"Upload URL generated for {file_name}", upload_info
            
        except Exception as e:
            s3_logger.error(f"Error generating specific presigned URL: {e}")
            return False, f"Failed to generate upload URL: {str(e)}", {}
    
    def verify_upload_completion(self, s3_key: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Verify that file was successfully uploaded to S3.
        
        Args:
            s3_key: S3 key of the uploaded file
            
        Returns:
            Tuple of (success, message, file_info)
        """
        try:
            # Check if object exists and get metadata
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            
            file_info = {
                's3_key': s3_key,
                'bucket_name': self.bucket_name,
                'file_size': response['ContentLength'],
                'content_type': response.get('ContentType', 'unknown'),
                'last_modified': response['LastModified'].isoformat(),
                'etag': response['ETag'].strip('"'),
                's3_url': f"s3://{self.bucket_name}/{s3_key}"
            }
            
            s3_logger.info(f"Upload verified: {s3_key} ({file_info['file_size']:,} bytes)")
            
            return True, "Upload verification successful", file_info
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False, "File not found in S3 - upload may have failed", {}
            else:
                return False, f"Error verifying upload: {e}", {}
        except Exception as e:
            return False, f"Unexpected error during verification: {e}", {}
    
    def download_from_s3(self, s3_key: str, local_path: Optional[str] = None) -> Tuple[bool, str, str]:
        """
        Download file from S3 to local filesystem for processing.
        
        Args:
            s3_key: S3 key of the file to download
            local_path: Local path to save file (optional, uses temp file if not provided)
            
        Returns:
            Tuple of (success, message, local_file_path)
        """
        try:
            # If no local path provided, create temporary file
            if not local_path:
                temp_dir = tempfile.mkdtemp(prefix='s3_download_')
                filename = os.path.basename(s3_key)
                local_path = os.path.join(temp_dir, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            s3_logger.info(f"Downloading {s3_key} to {local_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            # Verify download
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                s3_logger.info(f"Download completed: {local_path} ({file_size:,} bytes)")
                return True, f"File downloaded successfully ({file_size:,} bytes)", local_path
            else:
                return False, "Download failed - file not found locally", ""
                
        except Exception as e:
            s3_logger.error(f"Error downloading from S3: {e}")
            return False, f"Download failed: {str(e)}", ""
    
    def cleanup_uploaded_file(self, s3_key: str) -> Tuple[bool, str]:
        """
        Delete uploaded file from S3 after processing.
        
        Args:
            s3_key: S3 key of the file to delete
            
        Returns:
            Tuple of (success, message)
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            s3_logger.info(f"Cleaned up S3 file: {s3_key}")
            return True, f"File {s3_key} deleted from S3"
            
        except Exception as e:
            s3_logger.error(f"Error cleaning up S3 file {s3_key}: {e}")
            return False, f"Failed to delete {s3_key}: {str(e)}"
    
    def get_file_url(self, s3_key: str, expiration_hours: int = 24) -> Tuple[bool, str, str]:
        """
        Generate temporary download URL for processed results.
        
        Args:
            s3_key: S3 key of the file
            expiration_hours: URL expiration time in hours
            
        Returns:
            Tuple of (success, message, download_url)
        """
        try:
            download_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration_hours * 3600
            )
            
            return True, f"Download URL generated (expires in {expiration_hours}h)", download_url
            
        except Exception as e:
            s3_logger.error(f"Error generating download URL: {e}")
            return False, f"Failed to generate download URL: {str(e)}", ""


def create_s3_uploader() -> Optional[S3DirectUploader]:
    """
    Factory function to create S3 uploader with error handling.
    
    Returns:
        S3DirectUploader instance or None if credentials not available
    """
    try:
        return S3DirectUploader()
    except ValueError as e:
        s3_logger.warning(f"S3 uploader not available: {e}")
        return None
    except Exception as e:
        s3_logger.error(f"Unexpected error creating S3 uploader: {e}")
        return None


def check_s3_credentials() -> Tuple[bool, str, Dict[str, str]]:
    """
    Check if S3 credentials are properly configured.
    
    Returns:
        Tuple of (configured, message, credential_status)
    """
    credential_status = {}
    
    required_vars = ['S3_BUCKET_NAME', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    
    for var in required_vars:
        value = os.getenv(var)
        credential_status[var] = "✅ Set" if value else "❌ Missing"
    
    all_configured = all(os.getenv(var) for var in required_vars)
    
    if all_configured:
        return True, "All S3 credentials are configured", credential_status
    else:
        missing = [var for var in required_vars if not os.getenv(var)]
        return False, f"Missing S3 credentials: {', '.join(missing)}", credential_status


# JavaScript component for pure client-side S3 upload
def get_streamlit_compatible_s3_upload_component(s3_uploader: 'S3DirectUploader') -> str:
    """
    Generate Streamlit-compatible pure client-side JavaScript component for direct browser-to-S3 upload.
    This component uses Streamlit session state for communication instead of API endpoints.
    
    Args:
        s3_uploader: S3DirectUploader instance for generating URLs
        
    Returns:
        HTML/JavaScript component as string
    """
    # Generate generic upload config
    success, msg, upload_config = s3_uploader.generate_generic_presigned_url()
    
    if not success:
        return f"""
        <div style="padding: 20px; border: 2px solid #dc3545; border-radius: 10px; text-align: center; background-color: #f8d7da; color: #721c24;">
            <h3>❌ S3 Upload Configuration Error</h3>
            <p>{msg}</p>
        </div>
        """
    
    component_html = f"""
    <div id="s3-upload-container" style="padding: 20px; border: 2px dashed #ccc; border-radius: 10px; text-align: center; background-color: #f9f9f9; font-family: sans-serif;">
        <h3>🚀 Direct S3 Upload (Pure Client-Side)</h3>
        <p>Select your MP4 video file for processing (up to {upload_config.get('max_file_size_gb', 5)}GB)</p>
        
        <input type="file" id="s3-file-input" accept=".mp4,video/mp4" style="margin: 10px; padding: 10px; display: block; margin: 0 auto;">
        <br>
        <button id="s3-upload-btn" onclick="startUpload()" style="margin: 10px; padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;" disabled>
            📤 Select File to Upload
        </button>
        
        <div id="s3-file-info" style="margin-top: 10px; display: none; padding: 10px; background-color: #e8f4f8; border-radius: 5px;">
            <p id="s3-file-details"></p>
        </div>
        
        <div id="s3-upload-progress" style="margin-top: 20px; display: none;">
            <div style="background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0;">
                <div id="s3-progress-bar" style="height: 25px; background-color: #4CAF50; width: 0%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;"></div>
            </div>
            <p id="s3-progress-text">Preparing upload...</p>
        </div>
        
        <div id="s3-upload-result" style="margin-top: 20px; display: none;"></div>
    </div>

    <script>
    let selectedFile = null;
    let uploadConfig = {json.dumps(upload_config)};
    let uploadInProgress = false;
    
    // File input change handler
    document.getElementById('s3-file-input').addEventListener('change', function(e) {{
        selectedFile = e.target.files[0];
        const uploadBtn = document.getElementById('s3-upload-btn');
        const fileInfoDiv = document.getElementById('s3-file-info');
        const fileDetailsP = document.getElementById('s3-file-details');
        
        if (selectedFile) {{
            // Validate file type
            if (selectedFile.type !== 'video/mp4' && !selectedFile.name.toLowerCase().endsWith('.mp4')) {{
                showError('Please select an MP4 video file');
                selectedFile = null;
                uploadBtn.disabled = true;
                uploadBtn.innerHTML = '📤 Select File to Upload';
                fileInfoDiv.style.display = 'none';
                return;
            }}
            
            // Validate file size
            const maxSizeBytes = uploadConfig.max_file_size_bytes || (5 * 1024 * 1024 * 1024);
            if (selectedFile.size > maxSizeBytes) {{
                const maxSizeGB = uploadConfig.max_file_size_gb || 5;
                const fileSizeGB = (selectedFile.size / (1024 * 1024 * 1024)).toFixed(1);
                showError(`File too large (${{fileSizeGB}}GB). Maximum size is ${{maxSizeGB}}GB`);
                selectedFile = null;
                uploadBtn.disabled = true;
                uploadBtn.innerHTML = '📤 Select File to Upload';
                fileInfoDiv.style.display = 'none';
                return;
            }}
            
            // File is valid - show info and enable upload
            const fileSizeMB = (selectedFile.size / (1024 * 1024)).toFixed(1);
            const fileSizeGB = (selectedFile.size / (1024 * 1024 * 1024)).toFixed(2);
            
            const sizeDisplay = selectedFile.size >= 1024 * 1024 * 1024 ? 
                `${{fileSizeGB}}GB` : `${{fileSizeMB}}MB`;
            
            fileDetailsP.innerHTML = `
                <strong>📄 ${{selectedFile.name}}</strong><br>
                Size: ${{sizeDisplay}}<br>
                Type: ${{selectedFile.type}}
            `;
            fileInfoDiv.style.display = 'block';
            
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = `🚀 Upload ${{selectedFile.name}} (${{sizeDisplay}})`;
            uploadBtn.style.backgroundColor = '#4CAF50';
            
            // Clear any previous errors
            const resultDiv = document.getElementById('s3-upload-result');
            resultDiv.style.display = 'none';
            
        }} else {{
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '📤 Select File to Upload';
            uploadBtn.style.backgroundColor = '#cccccc';
            fileInfoDiv.style.display = 'none';
        }}
    }});
    
    // Error display function
    function showError(message) {{
        const resultDiv = document.getElementById('s3-upload-result');
        resultDiv.innerHTML = `
            <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <strong>❌ Error</strong><br>
                ${{message}}
            </div>
        `;
        resultDiv.style.display = 'block';
    }}
    
    // Success display function
    function showSuccess(s3Key, fileName, fileSize) {{
        const resultDiv = document.getElementById('s3-upload-result');
        const fileSizeMB = (fileSize / (1024 * 1024)).toFixed(1);
        
        resultDiv.innerHTML = `
            <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <strong>✅ Upload Successful!</strong><br>
                File: ${{fileName}}<br>
                Size: ${{fileSizeMB}}MB<br>
                S3 Key: ${{s3Key}}<br>
                <small>🎯 File is ready for ensemble transcription processing!</small>
            </div>
        `;
        resultDiv.style.display = 'block';
    }}
    
    // Main upload function
    async function startUpload() {{
        if (!selectedFile || uploadInProgress) {{
            if (!selectedFile) {{
                showError('Please select a file first');
            }}
            return;
        }}
        
        uploadInProgress = true;
        
        const progressContainer = document.getElementById('s3-upload-progress');
        const progressBar = document.getElementById('s3-progress-bar');
        const progressText = document.getElementById('s3-progress-text');
        const resultDiv = document.getElementById('s3-upload-result');
        const uploadBtn = document.getElementById('s3-upload-btn');
        
        // Show progress UI
        progressContainer.style.display = 'block';
        resultDiv.style.display = 'none';
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '⏳ Uploading...';
        uploadBtn.style.backgroundColor = '#ffc107';
        
        try {{
            // Step 1: Request file upload info from parent window
            progressText.innerHTML = '🔗 Requesting upload authorization...';
            progressBar.style.width = '10%';
            progressBar.innerHTML = '10%';
            
            // Send file info to Streamlit for pre-signed URL generation
            if (window.parent) {{
                window.parent.postMessage({{
                    type: 'S3_UPLOAD_REQUEST',
                    session_id: uploadConfig.session_id,
                    file_name: selectedFile.name,
                    file_size: selectedFile.size,
                    content_type: selectedFile.type
                }}, '*');
            }}
            
            // Wait for upload info from Streamlit
            const uploadInfo = await waitForUploadInfo();
            
            // Step 2: Upload file to S3
            progressText.innerHTML = '📤 Starting upload to S3...';
            progressBar.style.width = '20%';
            progressBar.innerHTML = '20%';
            
            await uploadFileToS3(uploadInfo, selectedFile, progressBar, progressText);
            
            // Step 3: Complete
            progressBar.style.width = '100%';
            progressBar.innerHTML = '100%';
            progressText.innerHTML = '✅ Upload completed successfully!';
            
            // Show success message
            showSuccess(uploadInfo.s3_key, selectedFile.name, selectedFile.size);
            
            // Notify Streamlit that upload is complete
            if (window.parent) {{
                window.parent.postMessage({{
                    type: 'S3_UPLOAD_COMPLETE',
                    s3_key: uploadInfo.s3_key,
                    file_name: selectedFile.name,
                    file_size: selectedFile.size,
                    bucket_name: uploadInfo.bucket_name,
                    session_id: uploadConfig.session_id
                }}, '*');
            }}
            
            // Reset button
            uploadBtn.innerHTML = '✅ Upload Complete';
            uploadBtn.style.backgroundColor = '#28a745';
            
        }} catch (error) {{
            console.error('Upload error:', error);
            progressBar.style.width = '0%';
            progressBar.innerHTML = '';
            progressText.innerHTML = '❌ Upload failed';
            
            showError(error.message || 'Upload failed due to an unexpected error');
            
            // Reset button
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = `🚀 Upload ${{selectedFile.name}}`;
            uploadBtn.style.backgroundColor = '#4CAF50';
        }} finally {{
            uploadInProgress = false;
        }}
    }}
    
    // Wait for upload info from Streamlit
    function waitForUploadInfo() {{
        return new Promise((resolve, reject) => {{
            const timeout = setTimeout(() => {{
                reject(new Error('Timeout waiting for upload authorization'));
            }}, 30000); // 30 second timeout
            
            function messageHandler(event) {{
                if (event.data.type === 'S3_UPLOAD_INFO') {{
                    clearTimeout(timeout);
                    window.removeEventListener('message', messageHandler);
                    
                    if (event.data.success) {{
                        resolve(event.data.upload_info);
                    }} else {{
                        reject(new Error(event.data.message || 'Failed to get upload authorization'));
                    }}
                }}
            }}
            
            window.addEventListener('message', messageHandler);
        }});
    }}
    
    // Upload file to S3 with progress tracking
    async function uploadFileToS3(uploadInfo, file, progressBar, progressText) {{
        return new Promise((resolve, reject) => {{
            const formData = new FormData();
            
            // Add all required fields from presigned post
            Object.keys(uploadInfo.fields).forEach(key => {{
                formData.append(key, uploadInfo.fields[key]);
            }});
            
            // Add the file last
            formData.append('file', file);
            
            // Upload with progress tracking
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', function(e) {{
                if (e.lengthComputable) {{
                    // Progress from 20% to 95% for upload
                    const uploadProgress = (e.loaded / e.total) * 75; // 75% of total
                    const totalProgress = 20 + uploadProgress; // Start at 20%
                    
                    progressBar.style.width = totalProgress + '%';
                    progressBar.innerHTML = totalProgress.toFixed(0) + '%';
                    
                    const loadedMB = (e.loaded / 1024 / 1024).toFixed(1);
                    const totalMB = (e.total / 1024 / 1024).toFixed(1);
                    const uploadPercent = ((e.loaded / e.total) * 100).toFixed(1);
                    
                    progressText.innerHTML = `📤 Uploading to S3... ${{uploadPercent}}% (${{loadedMB}}MB / ${{totalMB}}MB)`;
                }}
            }});
            
            xhr.addEventListener('load', function() {{
                if (xhr.status === 204) {{
                    resolve();
                }} else {{
                    reject(new Error(`S3 upload failed with status: ${{xhr.status}}`));
                }}
            }});
            
            xhr.addEventListener('error', function() {{
                reject(new Error('Network error during S3 upload'));
            }});
            
            xhr.addEventListener('timeout', function() {{
                reject(new Error('Upload timeout - file may be too large or network too slow'));
            }});
            
            // Set timeout to 15 minutes for large files
            xhr.timeout = 15 * 60 * 1000;
            
            xhr.open('POST', uploadInfo.upload_url);
            xhr.send(formData);
        }});
    }}
    
    // Initialize component
    console.log('S3 Direct Upload Component Initialized', uploadConfig);
    </script>
    """
    
    return component_html


def get_pure_s3_upload_component(upload_config: Dict[str, Any]) -> str:
    """
    Generate pure client-side JavaScript component for direct browser-to-S3 upload.
    This component handles file selection, validation, pre-signed URL generation, and upload.
    
    Args:
        upload_config: Upload configuration from generate_generic_presigned_url
        
    Returns:
        HTML/JavaScript component as string
    """
    component_html = f"""
    <div id="s3-upload-container" style="padding: 20px; border: 2px dashed #ccc; border-radius: 10px; text-align: center; background-color: #f9f9f9; font-family: sans-serif;">
        <h3>🚀 Direct S3 Upload (Pure Client-Side)</h3>
        <p>Select your MP4 video file for processing (up to {upload_config.get('max_file_size_gb', 5)}GB)</p>
        
        <input type="file" id="s3-file-input" accept=".mp4,video/mp4" style="margin: 10px; padding: 10px; display: block; margin: 0 auto;">
        <br>
        <button id="s3-upload-btn" onclick="startUpload()" style="margin: 10px; padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;" disabled>
            📤 Select File to Upload
        </button>
        
        <div id="s3-file-info" style="margin-top: 10px; display: none; padding: 10px; background-color: #e8f4f8; border-radius: 5px;">
            <p id="s3-file-details"></p>
        </div>
        
        <div id="s3-upload-progress" style="margin-top: 20px; display: none;">
            <div style="background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0;">
                <div id="s3-progress-bar" style="height: 25px; background-color: #4CAF50; width: 0%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;"></div>
            </div>
            <p id="s3-progress-text">Preparing upload...</p>
        </div>
        
        <div id="s3-upload-result" style="margin-top: 20px; display: none;"></div>
    </div>

    <script>
    let selectedFile = null;
    let uploadConfig = {json.dumps(upload_config)};
    let uploadInProgress = false;
    
    // File input change handler
    document.getElementById('s3-file-input').addEventListener('change', function(e) {{
        selectedFile = e.target.files[0];
        const uploadBtn = document.getElementById('s3-upload-btn');
        const fileInfoDiv = document.getElementById('s3-file-info');
        const fileDetailsP = document.getElementById('s3-file-details');
        
        if (selectedFile) {{
            // Validate file type
            if (selectedFile.type !== 'video/mp4' && !selectedFile.name.toLowerCase().endsWith('.mp4')) {{
                showError('Please select an MP4 video file');
                selectedFile = null;
                uploadBtn.disabled = true;
                uploadBtn.innerHTML = '📤 Select File to Upload';
                fileInfoDiv.style.display = 'none';
                return;
            }}
            
            // Validate file size
            const maxSizeBytes = uploadConfig.max_file_size_bytes || (5 * 1024 * 1024 * 1024);
            if (selectedFile.size > maxSizeBytes) {{
                const maxSizeGB = uploadConfig.max_file_size_gb || 5;
                const fileSizeGB = (selectedFile.size / (1024 * 1024 * 1024)).toFixed(1);
                showError(`File too large (${{fileSizeGB}}GB). Maximum size is ${{maxSizeGB}}GB`);
                selectedFile = null;
                uploadBtn.disabled = true;
                uploadBtn.innerHTML = '📤 Select File to Upload';
                fileInfoDiv.style.display = 'none';
                return;
            }}
            
            // File is valid - show info and enable upload
            const fileSizeMB = (selectedFile.size / (1024 * 1024)).toFixed(1);
            const fileSizeGB = (selectedFile.size / (1024 * 1024 * 1024)).toFixed(2);
            
            const sizeDisplay = selectedFile.size >= 1024 * 1024 * 1024 ? 
                `${{fileSizeGB}}GB` : `${{fileSizeMB}}MB`;
            
            fileDetailsP.innerHTML = `
                <strong>📄 ${{selectedFile.name}}</strong><br>
                Size: ${{sizeDisplay}}<br>
                Type: ${{selectedFile.type}}
            `;
            fileInfoDiv.style.display = 'block';
            
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = `🚀 Upload ${{selectedFile.name}} (${{sizeDisplay}})`;
            uploadBtn.style.backgroundColor = '#4CAF50';
            
            // Clear any previous errors
            const resultDiv = document.getElementById('s3-upload-result');
            resultDiv.style.display = 'none';
            
        }} else {{
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '📤 Select File to Upload';
            uploadBtn.style.backgroundColor = '#cccccc';
            fileInfoDiv.style.display = 'none';
        }}
    }});
    
    // Error display function
    function showError(message) {{
        const resultDiv = document.getElementById('s3-upload-result');
        resultDiv.innerHTML = `
            <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <strong>❌ Error</strong><br>
                ${{message}}
            </div>
        `;
        resultDiv.style.display = 'block';
    }}
    
    // Success display function
    function showSuccess(s3Key, fileName, fileSize) {{
        const resultDiv = document.getElementById('s3-upload-result');
        const fileSizeMB = (fileSize / (1024 * 1024)).toFixed(1);
        
        resultDiv.innerHTML = `
            <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <strong>✅ Upload Successful!</strong><br>
                File: ${{fileName}}<br>
                Size: ${{fileSizeMB}}MB<br>
                S3 Key: ${{s3Key}}<br>
                <small>🎯 File is ready for ensemble transcription processing!</small>
            </div>
        `;
        resultDiv.style.display = 'block';
    }}
    
    // Main upload function
    async function startUpload() {{
        if (!selectedFile || uploadInProgress) {{
            if (!selectedFile) {{
                showError('Please select a file first');
            }}
            return;
        }}
        
        uploadInProgress = true;
        
        const progressContainer = document.getElementById('s3-upload-progress');
        const progressBar = document.getElementById('s3-progress-bar');
        const progressText = document.getElementById('s3-progress-text');
        const resultDiv = document.getElementById('s3-upload-result');
        const uploadBtn = document.getElementById('s3-upload-btn');
        
        // Show progress UI
        progressContainer.style.display = 'block';
        resultDiv.style.display = 'none';
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '⏳ Uploading...';
        uploadBtn.style.backgroundColor = '#ffc107';
        
        try {{
            // Step 1: Request specific pre-signed URL for this file
            progressText.innerHTML = '🔗 Generating upload URL...';
            progressBar.style.width = '10%';
            progressBar.innerHTML = '10%';
            
            const urlResponse = await requestPresignedUrl(selectedFile.name, selectedFile.size);
            if (!urlResponse.success) {{
                throw new Error(urlResponse.message);
            }}
            
            // Step 2: Upload file to S3
            progressText.innerHTML = '📤 Starting upload to S3...';
            progressBar.style.width = '20%';
            progressBar.innerHTML = '20%';
            
            await uploadFileToS3(urlResponse.upload_info, selectedFile, progressBar, progressText);
            
            // Step 3: Verify upload
            progressText.innerHTML = '🔍 Verifying upload...';
            progressBar.style.width = '95%';
            progressBar.innerHTML = '95%';
            
            const verifySuccess = await verifyUpload(urlResponse.upload_info.s3_key);
            if (!verifySuccess) {{
                throw new Error('Upload verification failed');
            }}
            
            // Step 4: Complete
            progressBar.style.width = '100%';
            progressBar.innerHTML = '100%';
            progressText.innerHTML = '✅ Upload completed successfully!';
            
            // Show success message
            showSuccess(urlResponse.upload_info.s3_key, selectedFile.name, selectedFile.size);
            
            // Notify Streamlit
            if (window.parent) {{
                window.parent.postMessage({{
                    type: 'S3_UPLOAD_COMPLETE',
                    s3_key: urlResponse.upload_info.s3_key,
                    file_name: selectedFile.name,
                    file_size: selectedFile.size,
                    bucket_name: urlResponse.upload_info.bucket_name
                }}, '*');
            }}
            
            // Reset button
            uploadBtn.innerHTML = '✅ Upload Complete';
            uploadBtn.style.backgroundColor = '#28a745';
            
        }} catch (error) {{
            console.error('Upload error:', error);
            progressBar.style.width = '0%';
            progressBar.innerHTML = '';
            progressText.innerHTML = '❌ Upload failed';
            
            showError(error.message || 'Upload failed due to an unexpected error');
            
            // Reset button
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = `🚀 Upload ${{selectedFile.name}}`;
            uploadBtn.style.backgroundColor = '#4CAF50';
        }} finally {{
            uploadInProgress = false;
        }}
    }}
    
    // Request pre-signed URL from Streamlit backend
    async function requestPresignedUrl(fileName, fileSize) {{
        try {{
            // Use Streamlit's built-in method to call Python function
            const response = await fetch('/component/s3_upload.get_presigned_url', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    session_id: uploadConfig.session_id,
                    file_name: fileName,
                    file_size: fileSize
                }})
            }});
            
            if (!response.ok) {{
                throw new Error(`Failed to get upload URL: ${{response.status}}`);
            }}
            
            return await response.json();
            
        }} catch (error) {{
            console.error('Error requesting presigned URL:', error);
            throw new Error('Failed to generate upload URL. Please try again.');
        }}
    }}
    
    // Upload file to S3 with progress tracking
    async function uploadFileToS3(uploadInfo, file, progressBar, progressText) {{
        return new Promise((resolve, reject) => {{
            const formData = new FormData();
            
            // Add all required fields from presigned post
            Object.keys(uploadInfo.fields).forEach(key => {{
                formData.append(key, uploadInfo.fields[key]);
            }});
            
            // Add the file last
            formData.append('file', file);
            
            // Upload with progress tracking
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', function(e) {{
                if (e.lengthComputable) {{
                    // Progress from 20% to 90% for upload
                    const uploadProgress = (e.loaded / e.total) * 70; // 70% of total
                    const totalProgress = 20 + uploadProgress; // Start at 20%
                    
                    progressBar.style.width = totalProgress + '%';
                    progressBar.innerHTML = totalProgress.toFixed(0) + '%';
                    
                    const loadedMB = (e.loaded / 1024 / 1024).toFixed(1);
                    const totalMB = (e.total / 1024 / 1024).toFixed(1);
                    const uploadPercent = ((e.loaded / e.total) * 100).toFixed(1);
                    
                    progressText.innerHTML = `📤 Uploading to S3... ${{uploadPercent}}% (${{loadedMB}}MB / ${{totalMB}}MB)`;
                }}
            }});
            
            xhr.addEventListener('load', function() {{
                if (xhr.status === 204) {{
                    resolve();
                }} else {{
                    reject(new Error(`S3 upload failed with status: ${{xhr.status}}`));
                }}
            }});
            
            xhr.addEventListener('error', function() {{
                reject(new Error('Network error during S3 upload'));
            }});
            
            xhr.addEventListener('timeout', function() {{
                reject(new Error('Upload timeout - file may be too large or network too slow'));
            }});
            
            // Set timeout to 10 minutes
            xhr.timeout = 10 * 60 * 1000;
            
            xhr.open('POST', uploadInfo.upload_url);
            xhr.send(formData);
        }});
    }}
    
    // Verify upload completion
    async function verifyUpload(s3Key) {{
        try {{
            const response = await fetch('/component/s3_upload.verify_upload', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    s3_key: s3Key
                }})
            }});
            
            if (!response.ok) {{
                return false;
            }}
            
            const result = await response.json();
            return result.success || false;
            
        }} catch (error) {{
            console.error('Error verifying upload:', error);
            return false;
        }}
    }}
    
    // Initialize component
    console.log('S3 Direct Upload Component Initialized', uploadConfig);
    </script>
    """
    
    return component_html


# Legacy component for backward compatibility (deprecated)
def get_s3_upload_component(upload_info: Dict[str, Any]) -> str:
    """
    DEPRECATED: Legacy component that requires pre-existing upload info.
    Use get_pure_s3_upload_component instead for pure client-side uploads.
    
    Args:
        upload_info: Upload information from generate_presigned_upload_url
        
    Returns:
        HTML/JavaScript component as string
    """
    s3_logger.warning("Using deprecated get_s3_upload_component. Consider migrating to get_pure_s3_upload_component.")
    
    component_html = f"""
    <div id="s3-upload-container" style="padding: 20px; border: 2px dashed #ccc; border-radius: 10px; text-align: center; background-color: #f9f9f9;">
        <h3>🚀 Direct S3 Upload (Legacy)</h3>
        <p>Upload your pre-selected MP4 video file</p>
        
        <button id="s3-upload-btn" onclick="uploadToS3()" style="margin: 10px; padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">
            📤 Upload {upload_info.get('file_name', 'File')} ({(upload_info.get('file_size', 0) / 1024 / 1024):.1f}MB)
        </button>
        
        <div id="s3-upload-progress" style="margin-top: 20px; display: none;">
            <div style="background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                <div id="s3-progress-bar" style="height: 20px; background-color: #4CAF50; width: 0%; transition: width 0.3s;"></div>
            </div>
            <p id="s3-progress-text">Preparing upload...</p>
        </div>
        
        <div id="s3-upload-result" style="margin-top: 20px; display: none;"></div>
    </div>

    <script>
    let uploadInfo = {json.dumps(upload_info)};
    
    async function uploadToS3() {{
        const progressContainer = document.getElementById('s3-upload-progress');
        const progressBar = document.getElementById('s3-progress-bar');
        const progressText = document.getElementById('s3-progress-text');
        const resultDiv = document.getElementById('s3-upload-result');
        const uploadBtn = document.getElementById('s3-upload-btn');
        
        // Show progress UI
        progressContainer.style.display = 'block';
        resultDiv.style.display = 'none';
        uploadBtn.disabled = true;
        
        // Note: This legacy component cannot actually upload without a file from st.file_uploader
        progressText.innerHTML = '❌ Legacy component requires migration to pure client-side version';
        
        resultDiv.innerHTML = `
            <div style="background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <strong>⚠️ Component Update Required</strong><br>
                This upload method requires migration to the new pure client-side component.<br>
                <small>Please contact the developer to update the implementation.</small>
            </div>
        `;
        resultDiv.style.display = 'block';
        uploadBtn.disabled = false;
    }}
    </script>
    """
    
    return component_html