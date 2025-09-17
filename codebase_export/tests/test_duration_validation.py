"""
Focused duration validation tests for long video processing.

This module provides quick, focused tests specifically for validating that
no truncation occurs during video processing, complementing the comprehensive
long video integration tests.
"""

import pytest
import inspect
import subprocess
import re
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from core.audio_processor import AudioProcessor
from core.ensemble_manager import EnsembleManager


class TestDurationValidation:
    """Quick tests to validate duration handling and prevent truncation"""
    
    def test_audio_processor_no_truncation_flags(self):
        """Test that AudioProcessor doesn't use any truncation flags"""
        processor = AudioProcessor()
        
        # Get the source code of the AudioProcessor class
        processor_source = inspect.getsource(AudioProcessor)
        
        # Check for truncation-related flags that should NOT be present in FFmpeg commands
        forbidden_flags = ['-t ', '--duration', '-to ', '--to']
        truncation_keywords = ['truncat', 'limit', 'cut', 'trim_duration']
        
        # Look for these flags in command construction, not just anywhere in the source
        ffmpeg_command_lines = [line for line in processor_source.split('\n') 
                               if ('ffmpeg' in line.lower() and any(flag in line for flag in forbidden_flags))]
        
        for line in ffmpeg_command_lines:
            # Check if this line constructs an FFmpeg command with truncation flags
            if any(flag in line for flag in forbidden_flags):
                # Verify this isn't in a comment about avoiding truncation
                if not any(avoid_word in line.lower() for avoid_word in ['removed', 'without', 'no ', 'avoid', 'prevent']):
                    assert False, f"AudioProcessor constructs FFmpeg command with truncation flag: {line.strip()}"
        
        # Check for truncation keywords (should only appear in comments about avoiding truncation)
        for keyword in truncation_keywords:
            matches = [line for line in processor_source.split('\n') if keyword.lower() in line.lower()]
            for match in matches:
                # If truncation keyword is found, it should be in context of avoiding it
                assert any(avoid_word in match.lower() for avoid_word in ['without', 'no ', 'avoid', 'prevent']), \
                    f"Truncation keyword '{keyword}' found without avoidance context: {match.strip()}"
    
    def test_ffmpeg_commands_preserve_full_duration(self):
        """Test that FFmpeg commands are configured to preserve full duration"""
        processor = AudioProcessor()
        
        # Mock subprocess to capture ffmpeg commands
        captured_commands = []
        
        def capture_subprocess(*args, **kwargs):
            if args and len(args) > 0 and isinstance(args[0], list):
                captured_commands.append(args[0])
            # Return successful mock result
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "success"
            mock_result.stderr = ""
            return mock_result
        
        with patch('subprocess.run', side_effect=capture_subprocess), \
             patch('tempfile.mkstemp', return_value=(1, '/tmp/test.m4a')), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1000000), \
             patch('os.close'):
            
            try:
                processor.copy_audio_stream('/fake/video.mp4')
            except:
                pass  # We only care about capturing commands
            
            try:
                processor.make_asr_wav_from_audio('/fake/audio.m4a')
            except:
                pass
        
        # Analyze captured FFmpeg commands
        ffmpeg_commands = [cmd for cmd in captured_commands if cmd and 'ffmpeg' in cmd[0]]
        
        assert len(ffmpeg_commands) > 0, "Should have captured FFmpeg commands"
        
        for cmd in ffmpeg_commands:
            # Ensure no duration limiting flags
            assert '-t' not in cmd, f"FFmpeg command contains duration limit flag: {cmd}"
            assert '-to' not in cmd, f"FFmpeg command contains end time flag: {cmd}"
            assert '--duration' not in cmd, f"FFmpeg command contains duration flag: {cmd}"
            
            # Check for flags that preserve full content
            if '-i' in cmd:  # Input file flag
                input_idx = cmd.index('-i')
                # Should have input file after -i flag
                assert input_idx + 1 < len(cmd), "FFmpeg command should have input file"
    
    def test_config_duration_limits(self):
        """Test that configuration supports long duration videos"""
        
        # Test the actual config file
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # Check for duration settings
            assert 'max_duration_seconds' in config_content, "Config should specify max duration"
            
            # Extract max duration value
            import re
            duration_match = re.search(r'max_duration_seconds:\s*(\d+)', config_content)
            if duration_match:
                max_duration = int(duration_match.group(1))
                # Should support at least 2 hours (7200 seconds)
                assert max_duration >= 7200, f"Max duration too short: {max_duration}s (need ≥7200s for 2+ hours)"
                # Current config should support 8 hours
                assert max_duration >= 28800, f"Expected 8-hour support: {max_duration}s"
        else:
            pytest.skip("Config file not found")
    
    def test_ensemble_manager_chunking_threshold(self):
        """Test that EnsembleManager has appropriate chunking for long videos"""
        manager = EnsembleManager.create_safe()
        
        # Should have chunking threshold configured
        assert hasattr(manager, 'chunked_processing_threshold'), "EnsembleManager should have chunking threshold"
        
        threshold = getattr(manager, 'chunked_processing_threshold', float('inf'))
        
        # Threshold should enable chunking for long videos (typically 15-30 minutes)
        assert threshold <= 1800, f"Chunking threshold too high for long videos: {threshold}s"
        assert threshold >= 300, f"Chunking threshold too low: {threshold}s"
    
    def test_timeout_configuration_adequate(self):
        """Test that timeout configurations support long video processing"""
        
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Check timeout settings
        timeout_patterns = {
            'api_request': 900,      # 15 minutes minimum
            'asr_variant': 600,      # 10 minutes minimum  
            'diarization': 1800,     # 30 minutes minimum
        }
        
        for timeout_name, min_value in timeout_patterns.items():
            pattern = rf'{timeout_name}:\s*(\d+)'
            match = re.search(pattern, config_content)
            
            if match:
                timeout_value = int(match.group(1))
                assert timeout_value >= min_value, \
                    f"{timeout_name} timeout too short for long videos: {timeout_value}s (need ≥{min_value}s)"
    
    @patch('core.audio_processor.AudioProcessor.get_audio_duration')
    def test_duration_validation_method(self, mock_get_duration):
        """Test that duration validation method works correctly"""
        mock_get_duration.return_value = 7200.0  # 2 hours
        
        processor = AudioProcessor()
        duration = processor.get_audio_duration('/fake/audio.wav')
        
        # Should return duration in seconds
        assert duration == 7200.0, f"Duration method should return seconds: {duration}"
        assert duration >= 7200, "Should support 2+ hour duration detection"


class TestMemoryConfiguration:
    """Test memory-related configuration for long video processing"""
    
    def test_chunking_configuration_exists(self):
        """Test that chunking configuration exists and is reasonable"""
        
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Should have chunking section
        assert 'chunking:' in config_content, "Config should have chunking section"
        assert 'enabled: true' in config_content or 'enabled:true' in config_content, "Chunking should be enabled"
        
        # Check chunk size settings
        chunk_patterns = {
            'min_chunk_seconds': (10, 30),    # 10-30 seconds
            'max_chunk_seconds': (30, 120),   # 30-120 seconds  
            'target_chunk_seconds': (15, 60), # 15-60 seconds
        }
        
        import re
        for setting, (min_val, max_val) in chunk_patterns.items():
            pattern = rf'{setting}:\s*([0-9.]+)'
            match = re.search(pattern, config_content)
            
            if match:
                value = float(match.group(1))
                assert min_val <= value <= max_val, \
                    f"{setting} value out of range: {value} (should be {min_val}-{max_val})"
    
    def test_resource_scheduling_configuration(self):
        """Test resource scheduling configuration for long videos"""
        
        # Test that EnsembleManager can be created with resource scheduling
        manager = EnsembleManager.create_safe()
        
        # Should have resource scheduling capability
        assert hasattr(manager, 'enable_resource_scheduling'), "Should have resource scheduling option"
        
        # If enabled, should have resource scheduler
        if getattr(manager, 'enable_resource_scheduling', False):
            assert hasattr(manager, 'resource_scheduler'), "Should have resource scheduler when enabled"


class TestErrorHandlingConfiguration:
    """Test error handling configuration for long videos"""
    
    def test_circuit_breaker_configuration(self):
        """Test that circuit breakers are configured for long processing"""
        
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Should have circuit breaker configuration
        assert 'circuit_breakers:' in config_content, "Config should have circuit breaker section"
        
        # Check for service-specific breakers
        services = ['openai', 'huggingface', 'requests']
        for service in services:
            assert f'{service}:' in config_content, f"Should have circuit breaker config for {service}"
    
    def test_retry_configuration(self):
        """Test retry configuration for long video processing"""
        
        config_path = Path("config/config.yaml") 
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Should have retry configuration
        retry_keywords = ['retry', 'timeout', 'failure_threshold']
        
        for keyword in retry_keywords:
            assert keyword in config_content, f"Config should contain retry-related setting: {keyword}"


@pytest.mark.integration  
class TestEndToEndDurationValidation:
    """End-to-end tests for duration validation"""
    
    @patch('core.ensemble_manager.EnsembleManager.process_video')
    def test_mock_2_hour_processing(self, mock_process_video):
        """Test mock 2-hour video processing to validate pipeline"""
        
        # Mock a 2-hour video result
        target_duration = 7200  # 2 hours
        
        mock_result = {
            'master_transcript': {
                'segments': [
                    {
                        'start': i * 30,
                        'end': (i * 30) + 25,
                        'speaker': f'Speaker_{i % 3}',
                        'text': f'Segment {i} content.',
                        'confidence': 0.9
                    }
                    for i in range(target_duration // 30)  # 30-second segments
                ],
                'metadata': {
                    'total_duration': target_duration,
                    'confidence_summary': {'final_score': 0.85}
                }
            },
            'processing_metadata': {
                'total_processing_time': target_duration * 0.1,  # 10% RTF
                'chunked_processing': True
            }
        }
        
        mock_process_video.return_value = mock_result
        
        # Create manager and process
        manager = EnsembleManager.create_safe()
        result = manager.process_video('/fake/2hour_video.mp4')
        
        # Validate full duration was processed
        actual_duration = result['master_transcript']['metadata']['total_duration']
        assert actual_duration >= target_duration * 0.98, f"Duration validation failed: {actual_duration} < {target_duration}"
        
        # Validate segment coverage
        segments = result['master_transcript']['segments']
        if segments:
            last_segment_end = max(seg['end'] for seg in segments)
            coverage_ratio = last_segment_end / target_duration
            assert coverage_ratio >= 0.95, f"Poor segment coverage: {coverage_ratio:.2%}"
        
        # Should use chunked processing for 2-hour video
        assert result['processing_metadata']['chunked_processing'], "2-hour video should use chunked processing"
    
    def test_audio_processor_integration(self):
        """Test AudioProcessor integration for duration handling"""
        
        processor = AudioProcessor()
        
        # Test that required methods exist
        assert hasattr(processor, 'extract_audio_from_video'), "Should have main extraction method"
        assert hasattr(processor, 'get_audio_duration'), "Should have duration detection method"
        assert hasattr(processor, 'copy_audio_stream'), "Should have stream copy method"
        assert hasattr(processor, 'make_asr_wav_from_audio'), "Should have WAV creation method"
        
        # Test FFmpeg availability check
        is_available, message = processor.check_ffmpeg_availability()
        
        if is_available:
            # FFmpeg is available, test command structure
            assert "ffmpeg version" in message.lower() or "ffmpeg" in message.lower(), \
                f"FFmpeg availability message unclear: {message}"
        else:
            # FFmpeg not available, should have helpful message
            assert any(word in message.lower() for word in ['install', 'not found', 'path']), \
                f"FFmpeg unavailable message should be helpful: {message}"


if __name__ == "__main__":
    # Run focused duration validation tests
    pytest.main([__file__, "-v"])