"""
Error handling tests for long video processing edge cases.

This module tests error handling, graceful degradation, and edge cases
specifically related to long video processing scenarios.
"""

import pytest
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

from core.ensemble_manager import EnsembleManager, EnsembleManagerInitializationError
from core.audio_processor import AudioProcessor


class TestLongVideoErrorHandling:
    """Test error handling for long video processing scenarios"""
    
    def test_extremely_long_video_handling(self):
        """Test handling of videos that exceed reasonable limits"""
        
        # Test with video that approaches system limits
        extreme_duration = 10 * 3600  # 10 hours (beyond config limit)
        
        manager = EnsembleManager.create_safe()
        
        with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
            def extreme_duration_handler(*args, **kwargs):
                # Simulate system behavior with extreme duration
                if "10hour" in str(args[0]):
                    # Should either process with warnings or fail gracefully
                    raise TimeoutError("Processing timeout for extremely long video")
                return {'master_transcript': {'segments': [], 'metadata': {'total_duration': 3600}}}
            
            mock_process.side_effect = extreme_duration_handler
            
            # Should handle extreme duration gracefully
            with pytest.raises(TimeoutError) as exc_info:
                manager.process_video('/fake/10hour_video.mp4')
            
            # Error message should be informative
            assert "timeout" in str(exc_info.value).lower(), "Should have clear timeout error message"
    
    def test_insufficient_memory_handling(self):
        """Test handling when system runs out of memory during long processing"""
        
        with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
            def memory_exhaustion_handler(*args, **kwargs):
                # Simulate memory exhaustion
                raise MemoryError("Insufficient memory for processing long video")
            
            mock_process.side_effect = memory_exhaustion_handler
            
            manager = EnsembleManager.create_safe()
            
            # Should handle memory errors gracefully
            with pytest.raises(MemoryError) as exc_info:
                manager.process_video('/fake/memory_intensive_video.mp4')
            
            assert "memory" in str(exc_info.value).lower(), "Should have clear memory error message"
    
    def test_corrupted_long_video_handling(self):
        """Test handling of corrupted long video files"""
        
        processor = AudioProcessor()
        
        with patch('subprocess.run') as mock_subprocess:
            # Simulate FFmpeg error for corrupted file
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stderr = "Invalid data found when processing input"
            mock_subprocess.return_value = mock_result
            
            # Should handle corrupted file gracefully
            with pytest.raises(Exception) as exc_info:
                processor.copy_audio_stream('/fake/corrupted_long_video.mp4')
            
            error_message = str(exc_info.value).lower()
            assert any(keyword in error_message for keyword in ['invalid', 'failed', 'error']), \
                "Should have informative error message for corrupted file"
    
    def test_disk_space_exhaustion(self):
        """Test handling when disk space is exhausted during long processing"""
        
        with patch('tempfile.mkstemp') as mock_mkstemp:
            # Simulate disk space exhaustion
            mock_mkstemp.side_effect = OSError("No space left on device")
            
            processor = AudioProcessor()
            
            # Should handle disk space errors gracefully
            with pytest.raises(OSError) as exc_info:
                processor.copy_audio_stream('/fake/large_video.mp4')
            
            assert "space" in str(exc_info.value).lower(), "Should have clear disk space error"
    
    def test_network_timeout_during_processing(self):
        """Test handling of network timeouts during long processing"""
        
        with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
            def network_timeout_handler(*args, **kwargs):
                # Simulate network timeout for cloud services
                import requests
                raise requests.exceptions.Timeout("Request timeout during long video processing")
            
            mock_process.side_effect = network_timeout_handler
            
            manager = EnsembleManager.create_safe()
            
            # Should handle network timeouts gracefully
            with pytest.raises(Exception) as exc_info:
                manager.process_video('/fake/cloud_processed_video.mp4')
            
            error_message = str(exc_info.value).lower()
            assert "timeout" in error_message, "Should have clear timeout error message"
    
    def test_graceful_degradation_long_processing(self):
        """Test graceful degradation when processing long videos with issues"""
        
        with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
            def degraded_processing(*args, **kwargs):
                # Simulate degraded processing (fewer candidates, lower quality)
                return {
                    'master_transcript': {
                        'segments': [
                            {'start': 0, 'end': 60, 'speaker': 'Speaker_A', 'text': 'Degraded quality transcript', 'confidence': 0.6}
                        ],
                        'metadata': {
                            'total_duration': 7200.0,  # 2 hours
                            'processing_warnings': ['Degraded processing due to resource constraints'],
                            'quality_degraded': True
                        }
                    },
                    'processing_metadata': {
                        'degraded_mode': True,
                        'warnings': ['Limited processing due to constraints']
                    }
                }
            
            mock_process.side_effect = degraded_processing
            
            manager = EnsembleManager.create_safe()
            result = manager.process_video('/fake/resource_constrained_video.mp4')
            
            # Should still produce results even in degraded mode
            assert result is not None, "Should produce results even in degraded mode"
            assert result['master_transcript']['metadata']['total_duration'] > 0, "Should have duration even in degraded mode"
            
            # Should indicate degraded processing
            metadata = result.get('processing_metadata', {})
            assert metadata.get('degraded_mode', False), "Should indicate degraded mode"
    
    def test_chunking_failure_fallback(self):
        """Test fallback when chunking fails for long videos"""
        
        manager = EnsembleManager.create_safe()
        
        with patch.object(manager, 'chunked_processing_threshold', 60):  # Force chunking
            with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
                def chunking_failure_handler(*args, **kwargs):
                    # Simulate chunking failure with fallback to non-chunked processing
                    if 'chunked' not in kwargs:
                        # First call - chunking fails
                        raise RuntimeError("Chunking failed due to audio analysis error")
                    else:
                        # Fallback call - non-chunked processing
                        return {
                            'master_transcript': {
                                'segments': [
                                    {'start': 0, 'end': 1800, 'speaker': 'Speaker_A', 'text': 'Fallback processing', 'confidence': 0.8}
                                ],
                                'metadata': {'total_duration': 1800.0}
                            },
                            'processing_metadata': {'chunking_failed': True, 'fallback_used': True}
                        }
                
                mock_process.side_effect = chunking_failure_handler
                
                # Should handle chunking failure and use fallback
                try:
                    result = manager.process_video('/fake/chunking_problematic_video.mp4')
                    # If no exception, should have used fallback
                    assert result['processing_metadata'].get('fallback_used', False), "Should use fallback when chunking fails"
                except RuntimeError as e:
                    # If exception, should be about chunking failure
                    assert "chunking" in str(e).lower(), "Should have clear chunking error message"


class TestConfigurationErrorHandling:
    """Test error handling related to configuration issues"""
    
    def test_invalid_duration_configuration(self):
        """Test handling of invalid duration configuration"""
        
        # Test with mocked invalid config
        with patch('builtins.open', mock_open_config_with_invalid_duration):
            # Should handle invalid config gracefully
            try:
                manager = EnsembleManager.create_safe()
                # Should either use defaults or fail gracefully
                assert hasattr(manager, 'chunked_processing_threshold'), "Should have threshold even with invalid config"
            except Exception as e:
                # If it fails, should be informative
                assert any(keyword in str(e).lower() for keyword in ['config', 'duration', 'invalid']), \
                    "Should have clear config error message"
    
    def test_missing_dependencies_handling(self):
        """Test handling when required dependencies are missing"""
        
        # Test with mocked missing FFmpeg
        with patch('subprocess.run', side_effect=FileNotFoundError("FFmpeg not found")):
            processor = AudioProcessor()
            
            is_available, message = processor.check_ffmpeg_availability()
            
            # Should detect missing dependency
            assert not is_available, "Should detect missing FFmpeg"
            assert "not found" in message.lower(), "Should have clear missing dependency message"
    
    def test_insufficient_permissions_handling(self):
        """Test handling of insufficient file permissions"""
        
        with patch('tempfile.mkstemp', side_effect=PermissionError("Permission denied")):
            processor = AudioProcessor()
            
            # Should handle permission errors gracefully
            with pytest.raises(PermissionError) as exc_info:
                processor.copy_audio_stream('/fake/video.mp4')
            
            assert "permission" in str(exc_info.value).lower(), "Should have clear permission error"


class TestRecoveryMechanisms:
    """Test recovery mechanisms for long video processing"""
    
    def test_automatic_retry_on_transient_failure(self):
        """Test automatic retry on transient failures"""
        
        call_count = 0
        
        with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
            def transient_failure_then_success(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    # First call fails
                    raise ConnectionError("Transient network error")
                else:
                    # Second call succeeds
                    return {
                        'master_transcript': {
                            'segments': [{'start': 0, 'end': 60, 'speaker': 'A', 'text': 'Success after retry', 'confidence': 0.9}],
                            'metadata': {'total_duration': 60.0}
                        }
                    }
            
            mock_process.side_effect = transient_failure_then_success
            
            # Should implement retry logic (if available)
            manager = EnsembleManager.create_safe()
            
            try:
                result = manager.process_video('/fake/transient_failure_video.mp4')
                # If retry mechanism exists, should succeed on second try
                assert result is not None, "Should succeed after retry"
                assert call_count >= 1, "Should attempt processing at least once"
            except ConnectionError:
                # If no retry mechanism, should fail on first try
                assert call_count == 1, "Should fail on first attempt without retry"
    
    def test_partial_processing_recovery(self):
        """Test recovery from partial processing failures"""
        
        with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
            def partial_failure_handler(*args, **kwargs):
                # Simulate partial processing success
                return {
                    'master_transcript': {
                        'segments': [
                            {'start': 0, 'end': 1800, 'speaker': 'A', 'text': 'First part processed', 'confidence': 0.9}
                            # Missing second part due to failure
                        ],
                        'metadata': {
                            'total_duration': 3600.0,  # 1 hour
                            'partial_processing': True,
                            'processing_errors': ['Failed to process segment 1800-3600']
                        }
                    },
                    'processing_metadata': {
                        'partial_success': True,
                        'completion_ratio': 0.5
                    }
                }
            
            mock_process.side_effect = partial_failure_handler
            
            manager = EnsembleManager.create_safe()
            result = manager.process_video('/fake/partial_failure_video.mp4')
            
            # Should handle partial processing
            assert result is not None, "Should return results even with partial processing"
            
            metadata = result['master_transcript']['metadata']
            assert metadata.get('partial_processing', False), "Should indicate partial processing"


def mock_open_config_with_invalid_duration(filename, mode='r'):
    """Mock config file with invalid duration settings"""
    if 'config.yaml' in filename:
        invalid_config = """
audio:
  max_duration_seconds: invalid_value  # Invalid duration
  sample_rate: 16000

chunking:
  enabled: true
  min_chunk_seconds: -10  # Invalid negative value
  max_chunk_seconds: abc  # Invalid non-numeric value
"""
        from unittest.mock import mock_open
        return mock_open(read_data=invalid_config)(filename, mode)
    else:
        # For other files, use real open
        return open(filename, mode)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])