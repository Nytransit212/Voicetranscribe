"""
Audio Format Validator and Converter

Ensures all audio inputs are in proper format for processing:
- Converts to 16 kHz mono PCM WAV if needed
- Validates audio file integrity 
- Handles multiple input formats (MP4, MOV, MP3, WAV, etc.)
- Provides format normalization for integration tests

Uses ffmpeg-python for robust format conversion with fallbacks.
"""

import os
import tempfile
import librosa
import soundfile as sf
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
from pathlib import Path
import hashlib
import json
from dataclasses import dataclass

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.capability_manager import get_capability_manager, require_feature

@dataclass
class AudioFormat:
    """Audio format specification"""
    sample_rate: int
    channels: int
    duration: float
    format: str
    bit_depth: Optional[int] = None
    codec: Optional[str] = None

@dataclass 
class ValidationResult:
    """Audio validation result"""
    valid: bool
    original_format: AudioFormat
    normalized_path: Optional[str] = None
    conversion_needed: bool = False
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None

class AudioFormatValidator:
    """Validates and normalizes audio formats for ensemble processing"""
    
    TARGET_SAMPLE_RATE = 16000  # Standard rate for speech processing
    TARGET_CHANNELS = 1  # Mono audio
    TARGET_FORMAT = 'wav'
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.logger = create_enhanced_logger("audio_format_validator")
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.capability_manager = get_capability_manager()
        
        # Check conversion capabilities
        self.ffmpeg_available = self.capability_manager.is_dependency_available('ffmpeg')
        self.librosa_available = self.capability_manager.is_dependency_available('librosa')
        self.soundfile_available = self.capability_manager.is_dependency_available('soundfile')
        
        self.logger.info("Audio format validator initialized", context={
            'target_sample_rate': self.TARGET_SAMPLE_RATE,
            'target_channels': self.TARGET_CHANNELS,
            'target_format': self.TARGET_FORMAT,
            'ffmpeg_available': self.ffmpeg_available,
            'librosa_available': self.librosa_available,
            'soundfile_available': self.soundfile_available
        })
    
    def validate_and_normalize(self, audio_path: Union[str, Path]) -> ValidationResult:
        """
        Validate audio file and normalize if needed
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            ValidationResult with validation status and normalized path
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            return ValidationResult(
                valid=False,
                original_format=AudioFormat(0, 0, 0.0, 'unknown'),
                error_message=f"Audio file not found: {audio_path}"
            )
        
        try:
            # Try to get audio info using librosa (most robust)
            original_format = self._get_audio_info(audio_path)
            
            # Check if conversion is needed
            conversion_needed = self._needs_conversion(original_format)
            
            if not conversion_needed:
                # File is already in correct format
                return ValidationResult(
                    valid=True,
                    original_format=original_format,
                    normalized_path=str(audio_path),
                    conversion_needed=False
                )
            
            # Convert file to target format
            normalized_path = self._convert_audio(audio_path, original_format)
            
            if normalized_path:
                return ValidationResult(
                    valid=True,
                    original_format=original_format,
                    normalized_path=normalized_path,
                    conversion_needed=True
                )
            else:
                return ValidationResult(
                    valid=False,
                    original_format=original_format,
                    error_message="Audio conversion failed"
                )
        
        except Exception as e:
            self.logger.error(f"Audio validation failed: {e}")
            return ValidationResult(
                valid=False,
                original_format=AudioFormat(0, 0, 0.0, 'unknown'),
                error_message=f"Validation error: {str(e)}"
            )
    
    def _get_audio_info(self, audio_path: Path) -> AudioFormat:
        """Get audio file information"""
        try:
            if self.librosa_available:
                # Use librosa for robust audio info extraction
                y, sr = librosa.load(str(audio_path), sr=None, mono=False)
                
                if len(y.shape) == 1:
                    channels = 1
                else:
                    channels = y.shape[0]
                
                duration = len(y) / sr if channels == 1 else y.shape[1] / sr
                
                return AudioFormat(
                    sample_rate=int(sr),
                    channels=channels,
                    duration=duration,
                    format=audio_path.suffix.lower().lstrip('.')
                )
            
            elif self.soundfile_available:
                # Fallback to soundfile
                import soundfile as sf
                info = sf.info(str(audio_path))
                
                return AudioFormat(
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    duration=info.duration,
                    format=info.format.lower()
                )
            
            else:
                # No audio libraries available - use basic file info
                self.logger.warning("No audio libraries available, using basic file info")
                return AudioFormat(
                    sample_rate=self.TARGET_SAMPLE_RATE,  # Assume target rate
                    channels=self.TARGET_CHANNELS,  # Assume mono
                    duration=0.0,  # Unknown duration
                    format=audio_path.suffix.lower().lstrip('.')
                )
        
        except Exception as e:
            self.logger.error(f"Failed to get audio info: {e}")
            raise
    
    def _needs_conversion(self, audio_format: AudioFormat) -> bool:
        """Check if audio needs format conversion"""
        return (
            audio_format.sample_rate != self.TARGET_SAMPLE_RATE or
            audio_format.channels != self.TARGET_CHANNELS or
            audio_format.format.lower() not in ['wav', 'wave']
        )
    
    def _convert_audio(self, audio_path: Path, original_format: AudioFormat) -> Optional[str]:
        """Convert audio to target format"""
        
        # Generate output filename
        output_filename = f"normalized_{hashlib.md5(str(audio_path).encode()).hexdigest()[:8]}.wav"
        output_path = Path(self.temp_dir) / output_filename
        
        conversion_success = False
        
        # Try ffmpeg-python first (most robust)
        if self.ffmpeg_available and not conversion_success:
            conversion_success = self._convert_with_ffmpeg(audio_path, output_path)
        
        # Try librosa as fallback
        if self.librosa_available and not conversion_success:
            conversion_success = self._convert_with_librosa(audio_path, output_path)
        
        # Try soundfile as last resort
        if self.soundfile_available and not conversion_success:
            conversion_success = self._convert_with_soundfile(audio_path, output_path)
        
        if conversion_success and output_path.exists():
            self.logger.info(f"Audio converted successfully: {audio_path} -> {output_path}")
            return str(output_path)
        else:
            self.logger.error(f"All conversion methods failed for: {audio_path}")
            return None
    
    def _convert_with_ffmpeg(self, input_path: Path, output_path: Path) -> bool:
        """Convert audio using ffmpeg-python"""
        try:
            import ffmpeg
            
            # Use ffmpeg for robust conversion
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                ar=self.TARGET_SAMPLE_RATE,  # Sample rate
                ac=self.TARGET_CHANNELS,     # Channels (mono)
                acodec='pcm_s16le',          # 16-bit PCM
                f='wav'                      # WAV format
            )
            
            # Run conversion with error handling
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            return output_path.exists()
        
        except Exception as e:
            self.logger.warning(f"ffmpeg conversion failed: {e}")
            return False
    
    def _convert_with_librosa(self, input_path: Path, output_path: Path) -> bool:
        """Convert audio using librosa + soundfile"""
        try:
            # Load audio with librosa (handles many formats)
            y, sr = librosa.load(
                str(input_path),
                sr=self.TARGET_SAMPLE_RATE,
                mono=True
            )
            
            # Save with soundfile if available
            if self.soundfile_available:
                sf.write(str(output_path), y, self.TARGET_SAMPLE_RATE, format='WAV', subtype='PCM_16')
            else:
                # Fallback - save as numpy array and hope for the best
                np.save(str(output_path).replace('.wav', '.npy'), y)
                return False  # Not really WAV format
            
            return output_path.exists()
        
        except Exception as e:
            self.logger.warning(f"librosa conversion failed: {e}")
            return False
    
    def _convert_with_soundfile(self, input_path: Path, output_path: Path) -> bool:
        """Convert audio using soundfile only (limited format support)"""
        try:
            import soundfile as sf
            
            # Read with soundfile
            data, samplerate = sf.read(str(input_path))
            
            # Resample if needed (basic resampling)
            if samplerate != self.TARGET_SAMPLE_RATE:
                # Simple resampling (not ideal but better than nothing)
                ratio = self.TARGET_SAMPLE_RATE / samplerate
                new_length = int(len(data) * ratio)
                data = np.interp(np.linspace(0, len(data), new_length), np.arange(len(data)), data)
            
            # Convert to mono if needed
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Save as WAV
            sf.write(str(output_path), data, self.TARGET_SAMPLE_RATE, format='WAV', subtype='PCM_16')
            
            return output_path.exists()
        
        except Exception as e:
            self.logger.warning(f"soundfile conversion failed: {e}")
            return False
    
    def validate_test_audio_files(self, test_files: list) -> Dict[str, ValidationResult]:
        """Validate multiple test audio files"""
        results = {}
        
        self.logger.info(f"Validating {len(test_files)} test audio files")
        
        for file_path in test_files:
            file_path = Path(file_path)
            result = self.validate_and_normalize(file_path)
            results[str(file_path)] = result
            
            if result.valid:
                self.logger.info(f"✅ {file_path.name}: Valid (converted: {result.conversion_needed})")
            else:
                self.logger.error(f"❌ {file_path.name}: Invalid - {result.error_message}")
        
        return results
    
    def cleanup_temp_files(self):
        """Clean up temporary normalized files"""
        try:
            temp_path = Path(self.temp_dir)
            for file in temp_path.glob("normalized_*.wav"):
                file.unlink()
                self.logger.debug(f"Cleaned up temp file: {file}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {e}")

# Global validator instance
_audio_validator = None

def get_audio_validator() -> AudioFormatValidator:
    """Get global audio format validator instance"""
    global _audio_validator
    if _audio_validator is None:
        _audio_validator = AudioFormatValidator()
    return _audio_validator

def validate_audio_file(audio_path: Union[str, Path]) -> ValidationResult:
    """Convenience function to validate and normalize audio file"""
    return get_audio_validator().validate_and_normalize(audio_path)

def ensure_audio_format(audio_path: Union[str, Path]) -> str:
    """
    Ensure audio file is in correct format, return path to normalized file
    
    Raises:
        ValueError: If audio validation/conversion fails
    """
    result = validate_audio_file(audio_path)
    
    if not result.valid:
        raise ValueError(f"Audio validation failed: {result.error_message}")
    
    return result.normalized_path or str(audio_path)