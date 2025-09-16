"""
OpenAI Whisper ASR provider for API-based transcription
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from .base import ASRProvider, ASRResult, ASRSegment, DecodeMode

logger = logging.getLogger(__name__)

class OpenAIProvider(ASRProvider):
    """
    OpenAI Whisper provider for API-based transcription
    
    Uses OpenAI's hosted Whisper API for reliable transcription
    with good general quality across various audio types.
    """
    
    def __init__(self, provider_name: str = "openai", model_name: str = "whisper-1", config: Dict[str, Any] = None):
        super().__init__(provider_name, model_name, config)
        self.api_key = self._get_api_key()
        self.client = None
        self._initialize_client()
    
    def _get_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment or config"""
        api_key = (
            self.config.get('api_key') or
            os.getenv('OPENAI_API_KEY')
        )
        
        if not api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        return api_key
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client"""
        if not self.api_key:
            logger.error("Cannot initialize OpenAI client: no API key")
            return
        
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=self.api_key)
            
            logger.info("OpenAI client initialized successfully")
            
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        decode_mode: DecodeMode = DecodeMode.DETERMINISTIC,
        language: str = "en",
        prompt: Optional[str] = None,
        **kwargs
    ) -> ASRResult:
        """
        Transcribe audio with OpenAI Whisper API
        
        Args:
            audio_path: Path to audio file
            decode_mode: Quality vs speed tradeoff mode (limited effect on API)
            language: Language code (en for English)
            prompt: Optional context prompt for better accuracy
            **kwargs: Additional OpenAI parameters
            
        Returns:
            ASRResult with segments and confidence scores
        """
        if not self.is_available():
            raise RuntimeError("OpenAI client not available")
        
        start_time = time.time()
        
        # Configure parameters
        transcribe_params = self._get_transcribe_params(decode_mode, language, prompt)
        
        # Override with kwargs
        transcribe_params.update(kwargs)
        
        logger.info(f"Transcribing with OpenAI Whisper in {decode_mode.value} mode")
        
        try:
            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size > self.get_max_file_size():
                raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB > 25MB")
            
            # Call OpenAI API
            with open(audio_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.model_name,
                    **transcribe_params
                )
            
            # Parse response based on format
            asr_segments = []
            full_text = ""
            
            if transcribe_params.get('response_format') == 'verbose_json':
                # Detailed response with segments
                if hasattr(transcript, 'segments') and transcript.segments:
                    for segment in transcript.segments:
                        # OpenAI doesn't provide confidence, estimate from context
                        estimated_confidence = self._estimate_confidence(segment, transcript)
                        calibrated_confidence = self.calibrate_confidence(
                            estimated_confidence,
                            segment.end - segment.start
                        )
                        
                        asr_segment = ASRSegment(
                            start=segment.start,
                            end=segment.end,
                            text=segment.text.strip(),
                            confidence=calibrated_confidence,
                            words=None  # OpenAI API doesn't provide word-level timestamps
                        )
                        
                        asr_segments.append(asr_segment)
                
                full_text = transcript.text if hasattr(transcript, 'text') else ""
                
            else:
                # Simple text response
                full_text = transcript.text if hasattr(transcript, 'text') else str(transcript)
                
                # Create single segment
                duration = self._estimate_duration(audio_path)
                estimated_confidence = 0.85  # Default for simple format
                calibrated_confidence = self.calibrate_confidence(estimated_confidence, duration)
                
                asr_segment = ASRSegment(
                    start=0.0,
                    end=duration,
                    text=full_text.strip(),
                    confidence=calibrated_confidence
                )
                
                asr_segments.append(asr_segment)
            
            processing_time = time.time() - start_time
            
            # Calculate overall confidence
            if asr_segments:
                avg_confidence = sum(seg.confidence for seg in asr_segments) / len(asr_segments)
            else:
                avg_confidence = 0.0
            
            # Create result
            result = ASRResult(
                segments=asr_segments,
                full_text=full_text,
                language=getattr(transcript, 'language', language),
                confidence=avg_confidence,
                calibrated_confidence=avg_confidence,
                processing_time=processing_time,
                provider=self.provider_name,
                decode_mode=decode_mode,
                model_name=self.model_name,
                metadata={
                    'response_format': transcribe_params.get('response_format', 'text'),
                    'temperature': transcribe_params.get('temperature', 0.0),
                    'file_size_mb': file_size / 1024 / 1024
                }
            )
            
            logger.info(f"OpenAI transcription complete: {len(asr_segments)} segments, "
                       f"{avg_confidence:.3f} confidence, {processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
            raise
    
    def _get_transcribe_params(self, decode_mode: DecodeMode, language: str, prompt: Optional[str]) -> Dict[str, Any]:
        """Get transcription parameters for decode mode"""
        base_params = {
            'language': language,
            'response_format': 'verbose_json',
            'timestamp_granularities': ['segment']
        }
        
        # Mode-specific parameters (OpenAI API has limited control)
        mode_params = {
            DecodeMode.CAREFUL: {
                'temperature': 0.0,
                'response_format': 'verbose_json'
            },
            DecodeMode.DETERMINISTIC: {
                'temperature': 0.2,
                'response_format': 'verbose_json'
            },
            DecodeMode.EXPLORATORY: {
                'temperature': 0.7,
                'response_format': 'verbose_json'
            },
            DecodeMode.FAST: {
                'temperature': 0.0,
                'response_format': 'text'  # Faster without detailed response
            },
            DecodeMode.ENHANCED: {
                'temperature': 0.1,
                'response_format': 'verbose_json'
            }
        }
        
        params = base_params.copy()
        params.update(mode_params.get(decode_mode, mode_params[DecodeMode.DETERMINISTIC]))
        
        # Add prompt if provided
        if prompt:
            params['prompt'] = prompt[:224]  # OpenAI limit
        
        return params
    
    def _estimate_confidence(self, segment: Any, transcript: Any) -> float:
        """Estimate confidence for OpenAI segment (no direct confidence provided)"""
        # Heuristic confidence estimation based on segment characteristics
        base_confidence = 0.85
        
        # Adjust based on segment length
        duration = segment.end - segment.start
        if duration < 0.5:
            base_confidence *= 0.9  # Short segments less reliable
        elif duration > 10.0:
            base_confidence *= 0.95  # Longer segments more context
        
        # Adjust based on text characteristics
        text = segment.text.strip()
        if len(text) < 10:
            base_confidence *= 0.9  # Very short text
        
        # Check for common transcription artifacts
        artifacts = ['[MUSIC]', '[NOISE]', '[INAUDIBLE]', '...', 'um', 'uh']
        if any(artifact.lower() in text.lower() for artifact in artifacts):
            base_confidence *= 0.8
        
        return min(0.95, max(0.3, base_confidence))
    
    def _estimate_duration(self, audio_path: Union[str, Path]) -> float:
        """Estimate audio duration"""
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', str(audio_path)
            ], capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 60.0  # Default fallback
    
    def is_available(self) -> bool:
        """Check if OpenAI client is available"""
        return self.client is not None and self.api_key is not None
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats"""
        return ['wav', 'mp3', 'mp4', 'flac', 'm4a', 'ogg', 'webm']
    
    def get_max_file_size(self) -> int:
        """Get maximum file size for OpenAI API"""
        return 25 * 1024 * 1024  # 25MB
    
    def estimate_processing_time(self, audio_duration: float, decode_mode: DecodeMode) -> float:
        """Estimate processing time for OpenAI API"""
        # OpenAI API is typically 0.1-0.3x real-time
        base_ratio = 0.2
        
        # Mode doesn't significantly affect API processing time
        return audio_duration * base_ratio
    
    def _load_confidence_calibration(self) -> Dict[str, float]:
        """Load OpenAI-specific confidence calibration"""
        # OpenAI confidence needs estimation, so we're conservative
        return {
            'scale': 1.0,
            'offset': 0.0,
            'min_confidence': 0.3,
            'max_confidence': 0.95
        }