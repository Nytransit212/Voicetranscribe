#!/usr/bin/env python3
"""
Test script for chunked upload functionality.
Creates test files of various sizes to verify chunked upload works correctly.
"""

import os
import tempfile
import time
import hashlib
from utils.chunked_upload import ChunkedUploadManager


class MockUploadedFile:
    """Mock Streamlit uploaded file for testing"""
    
    def __init__(self, file_path: str, name: str):
        self.file_path = file_path
        self.name = name
        with open(file_path, 'rb') as f:
            self.data = f.read()
        self.size = len(self.data)
        self.position = 0
    
    def read(self, size: int = -1):
        if size == -1:
            result = self.data[self.position:]
            self.position = len(self.data)
        else:
            result = self.data[self.position:self.position + size]
            self.position += len(result)
        return result
    
    def seek(self, position: int):
        self.position = position


def create_test_mp4_file(file_path: str, size_mb: int):
    """
    Create a test MP4 file of specified size.
    Creates a minimal MP4 file structure with dummy data.
    """
    # Minimal MP4 header (ftyp + mdat boxes)
    ftyp_box = b'\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41'
    mdat_header = b'\x00\x00\x00\x08mdat'
    
    # Calculate size of dummy data needed
    header_size = len(ftyp_box) + len(mdat_header)
    target_size = size_mb * 1024 * 1024
    dummy_data_size = target_size - header_size
    
    # Create dummy data (pattern to make it compressible/realistic)
    pattern = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F' * 64  # 1KB pattern
    
    with open(file_path, 'wb') as f:
        # Write MP4 headers
        f.write(ftyp_box)
        f.write(mdat_header)
        
        # Write dummy data
        full_patterns = dummy_data_size // len(pattern)
        remainder = dummy_data_size % len(pattern)
        
        for _ in range(full_patterns):
            f.write(pattern)
        
        if remainder > 0:
            f.write(pattern[:remainder])
    
    print(f"Created test MP4 file: {file_path} ({size_mb}MB)")


def test_chunked_upload_system():
    """Test the chunked upload system with various file sizes"""
    
    print("=" * 60)
    print("TESTING CHUNKED UPLOAD SYSTEM")
    print("=" * 60)
    
    # Test cases: file sizes in MB
    test_cases = [
        0.5,   # Small file (should work with regular upload too)
        2,     # Medium file (larger than 1MB nginx limit)
        10,    # Large file
        50,    # Very large file
        # 100,   # Extremely large file (comment out if too slow)
    ]
    
    temp_dir = tempfile.mkdtemp(prefix='chunked_upload_test_')
    
    try:
        for size_mb in test_cases:
            print(f"\n--- Testing {size_mb}MB file ---")
            
            # Create test file
            test_file_path = os.path.join(temp_dir, f"test_{size_mb}mb.mp4")
            create_test_mp4_file(test_file_path, int(size_mb))
            
            # Create mock uploaded file
            mock_uploaded_file = MockUploadedFile(test_file_path, f"test_{size_mb}mb.mp4")
            
            # Initialize chunked upload manager
            upload_manager = ChunkedUploadManager(chunk_size_kb=512, max_file_size_gb=2)
            
            # Test file validation
            print("Testing file validation...")
            is_valid, validation_msg = upload_manager.validate_file(mock_uploaded_file)
            print(f"Validation result: {is_valid} - {validation_msg}")
            
            if not is_valid:
                print(f"❌ File validation failed for {size_mb}MB file")
                continue
            
            # Test chunk info calculation
            chunk_info = upload_manager.get_chunk_info(mock_uploaded_file.size)
            print(f"Chunk info: {chunk_info['total_chunks']} chunks of {chunk_info['chunk_size_kb']}KB each")
            
            # Test upload session creation
            print("Testing upload session creation...")
            session_success, session_msg, session_info = upload_manager.create_upload_session(mock_uploaded_file)
            print(f"Session creation: {session_success} - {session_msg}")
            
            if not session_success:
                print(f"❌ Session creation failed for {size_mb}MB file")
                continue
            
            # Test chunked upload
            print("Testing chunked upload...")
            start_time = time.time()
            
            def progress_callback(chunk_num, total_chunks, message):
                progress_percent = int((chunk_num / total_chunks) * 100)
                print(f"  Progress: {progress_percent}% ({chunk_num}/{total_chunks}) - {message}")
            
            upload_success, upload_msg, final_path = upload_manager.upload_file_chunks(
                mock_uploaded_file, session_info, progress_callback
            )
            
            upload_time = time.time() - start_time
            print(f"Upload result: {upload_success} - {upload_msg}")
            print(f"Upload time: {upload_time:.2f} seconds")
            
            if upload_success:
                # Verify file integrity
                print("Verifying file integrity...")
                
                # Compare original and reassembled files
                with open(test_file_path, 'rb') as orig_file:
                    original_data = orig_file.read()
                    original_hash = hashlib.sha256(original_data).hexdigest()
                
                if final_path and os.path.exists(final_path):
                    with open(final_path, 'rb') as final_file:
                        final_data = final_file.read()
                        final_hash = hashlib.sha256(final_data).hexdigest()
                else:
                    final_data = b''
                    final_hash = 'invalid'
                
                integrity_check = original_hash == final_hash
                print(f"Integrity check: {'✅ PASSED' if integrity_check else '❌ FAILED'}")
                print(f"Original size: {len(original_data):,} bytes")
                print(f"Final size: {len(final_data):,} bytes")
                
                if integrity_check:
                    print(f"✅ {size_mb}MB file upload test PASSED")
                else:
                    print(f"❌ {size_mb}MB file upload test FAILED (integrity check)")
            else:
                print(f"❌ {size_mb}MB file upload test FAILED")
            
            # Clean up session
            upload_manager.cleanup_session(session_info)
    
    finally:
        # Clean up test directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up test directory: {temp_dir}")
    
    print("\n" + "=" * 60)
    print("CHUNKED UPLOAD TESTING COMPLETE")
    print("=" * 60)


def test_error_scenarios():
    """Test error handling scenarios"""
    
    print("\n--- Testing Error Scenarios ---")
    
    upload_manager = ChunkedUploadManager()
    
    # Test with None file
    print("Testing with None file...")
    is_valid, msg = upload_manager.validate_file(None)
    print(f"None file validation: {is_valid} - {msg}")
    
    # Test with non-MP4 file
    print("Testing with non-MP4 file...")
    temp_dir = tempfile.mkdtemp()
    txt_file_path = os.path.join(temp_dir, "test.txt")
    with open(txt_file_path, 'w') as f:
        f.write("This is not an MP4 file")
    
    mock_txt_file = MockUploadedFile(txt_file_path, "test.txt")
    is_valid, msg = upload_manager.validate_file(mock_txt_file)
    print(f"TXT file validation: {is_valid} - {msg}")
    
    # Test with empty file
    print("Testing with empty file...")
    empty_file_path = os.path.join(temp_dir, "empty.mp4")
    with open(empty_file_path, 'wb') as f:
        pass  # Create empty file
    
    mock_empty_file = MockUploadedFile(empty_file_path, "empty.mp4")
    is_valid, msg = upload_manager.validate_file(mock_empty_file)
    print(f"Empty file validation: {is_valid} - {msg}")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Starting chunked upload system tests...")
    
    try:
        test_error_scenarios()
        test_chunked_upload_system()
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()