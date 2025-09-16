"""
Faster-Whisper ASR provider for local GPU-accelerated transcription
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from .base import ASRProvider, ASRResult, ASRSegment, DecodeMode

logger = logging.getLogger(__name__)

class FasterWhisperProvider(ASRProvider):
    """
    Faster-Whisper provider for local GPU transcription
    
    Provides high-quality Whisper inference with GPU acceleration
    and multiple decode modes for quality vs speed tradeoffs.
    """
    
    def __init__(self, provider_name: str = "faster-whisper", model_name: str = "large-v3", config: Dict[str, Any] = None):
        super().__init__(provider_name, model_name, config)
        self.model = None
        self.device = config.get('device', 'auto')
        self.compute_type = config.get('compute_type', 'float16')
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize Faster-Whisper model with GPU acceleration"""
        try:
            # Try to import faster-whisper
            from faster_whisper import WhisperModel
            
            # Configure device
            if self.device == 'auto':
                # Try CUDA first, fall back to CPU
                try:
                    import torch
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                except ImportError:
                    self.device = 'cpu'
            
            # Adjust compute type for device
            if self.device == 'cpu':
                self.compute_type = 'int8'
            
            logger.info(f"Initializing Faster-Whisper {self.model_name} on {self.device} with {self.compute_type}")
            
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.config.get('model_cache_dir'),
                local_files_only=self.config.get('local_files_only', False)
            )
            
            logger.info(f"Faster-Whisper model {self.model_name} loaded successfully")
            
        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize Faster-Whisper: {e}")
            self.model = None
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        decode_mode: DecodeMode = DecodeMode.DETERMINISTIC,
        language: str = "en",
        prompt: Optional[str] = None,
        **kwargs
    ) -> ASRResult:
        """
        Transcribe audio with Faster-Whisper
        
        Args:
            audio_path: Path to audio file
            decode_mode: Quality vs speed tradeoff mode
            language: Language code (en for US English)
            prompt: Optional context prompt for better accuracy
            **kwargs: Additional parameters
            
        Returns:
            ASRResult with segments and confidence scores
        """
        if not self.is_available():
            raise RuntimeError("Faster-Whisper model not available")
        
        start_time = time.time()
        
        # Configure parameters based on decode mode
        decode_params = self._get_decode_params(decode_mode)
        
        # Override with kwargs
        decode_params.update(kwargs)
        
        # Add prompt if provided
        if prompt:
            decode_params['initial_prompt'] = prompt
        
        logger.info(f"Transcribing with Faster-Whisper {decode_mode.value} mode")
        
        try:
            # Run transcription
            segments, info = self.model.transcribe(
                str(audio_path),
                language=language,
                **decode_params
            )
            
            # Convert segments to our format
            asr_segments = []
            full_text_parts = []
            
            for segment in segments:
                # Calculate confidence (Faster-Whisper provides avg_logprob)
                raw_confidence = self._logprob_to_confidence(segment.avg_logprob)
                calibrated_confidence = self.calibrate_confidence(
                    raw_confidence, 
                    segment.end - segment.start
                )
                
                asr_segment = ASRSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=calibrated_confidence,
                    words=[{
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    } for word in segment.words] if hasattr(segment, 'words') and segment.words else None
                )
                
                asr_segments.append(asr_segment)
                full_text_parts.append(segment.text.strip())
            
            processing_time = time.time() - start_time
            
            # Calculate overall confidence
            if asr_segments:
                avg_confidence = sum(seg.confidence for seg in asr_segments) / len(asr_segments)
            else:
                avg_confidence = 0.0
            
            # Create result
            result = ASRResult(
                segments=asr_segments,
                full_text=' '.join(full_text_parts),
                language=info.language if hasattr(info, 'language') else language,
                confidence=avg_confidence,
                calibrated_confidence=avg_confidence,
                processing_time=processing_time,
                provider=self.provider_name,
                decode_mode=decode_mode,
                model_name=self.model_name,
                metadata={
                    'device': self.device,
                    'compute_type': self.compute_type,
                    'language_probability': getattr(info, 'language_probability', 0.0),
                    'duration': getattr(info, 'duration', 0.0),
                    'decode_params': decode_params
                }
            )
            
            logger.info(f"Faster-Whisper transcription complete: {len(asr_segments)} segments, "
                       f"{avg_confidence:.3f} confidence, {processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Faster-Whisper transcription failed: {e}")
            raise
    
    def _get_decode_params(self, decode_mode: DecodeMode) -> Dict[str, Any]:
        """Get decode parameters for specific mode"""
        base_params = {
            'word_timestamps': True,
            'hallucination_silence_threshold': 1.0,
            'compression_ratio_threshold': 2.4,
            'log_prob_threshold': -1.0,
            'no_speech_threshold': 0.6
        }
        
        mode_params = {
            DecodeMode.CAREFUL: {
                'beam_size': 5,
                'best_of': 5,
                'temperature': 0.0,
                'patience': 2.0,
                'length_penalty': 1.0,
                'repetition_penalty': 1.0
            },
            DecodeMode.DETERMINISTIC: {
                'beam_size': 5,
                'best_of': 1,
                'temperature': 0.2,
                'patience': 1.0,
                'length_penalty': 1.0,
                'repetition_penalty': 1.0
            },
            DecodeMode.EXPLORATORY: {
                'beam_size': 3,
                'best_of': 3,
                'temperature': 0.7,
                'patience': 1.0,
                'length_penalty': 0.1,
                'repetition_penalty': 1.1
            },
            DecodeMode.FAST: {
                'beam_size': 1,
                'best_of': 1,
                'temperature': 0.0,
                'patience': 1.0,
                'length_penalty': 1.0,
                'repetition_penalty': 1.0
            },
            DecodeMode.ENHANCED: {
                'beam_size': 5,
                'best_of': 3,
                'temperature': 0.1,
                'patience': 1.5,
                'length_penalty': 1.0,
                'repetition_penalty': 1.0
            }
        }
        
        params = base_params.copy()
        params.update(mode_params.get(decode_mode, mode_params[DecodeMode.DETERMINISTIC]))
        
        return params
    
    def _logprob_to_confidence(self, avg_logprob: float) -> float:
        """Convert average log probability to confidence score"""
        # Empirical mapping from logprob to confidence
        # Based on typical Whisper logprob ranges
        if avg_logprob >= -0.5:
            return 0.95
        elif avg_logprob >= -1.0:
            return 0.85 + (avg_logprob + 1.0) * 0.2
        elif avg_logprob >= -2.0:
            return 0.65 + (avg_logprob + 2.0) * 0.2
        elif avg_logprob >= -3.0:
            return 0.45 + (avg_logprob + 3.0) * 0.2
        else:
            return max(0.1, 0.45 + (avg_logprob + 3.0) * 0.1)
    
    def is_available(self) -> bool:
        """Check if Faster-Whisper is available"""
        return self.model is not None
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats"""
        return ['wav', 'mp3', 'mp4', 'flac', 'm4a', 'ogg', 'webm']
    
    def get_max_file_size(self) -> int:
        """Get maximum file size (local processing, no strict limit)"""
        return 2 * 1024 * 1024 * 1024  # 2GB practical limit
    
    def estimate_processing_time(self, audio_duration: float, decode_mode: DecodeMode) -> float:
        """Estimate processing time based on mode and hardware"""
        # Base ratios for different modes
        base_ratios = {
            DecodeMode.FAST: 0.05,
            DecodeMode.DETERMINISTIC: 0.1,
            DecodeMode.CAREFUL: 0.3,
            DecodeMode.EXPLORATORY: 0.15,
            DecodeMode.ENHANCED: 0.2
        }
        
        base_ratio = base_ratios.get(decode_mode, 0.1)
        
        # Adjust for device
        if self.device == 'cuda':
            device_multiplier = 1.0
        elif self.device == 'cpu':
            device_multiplier = 3.0  # CPU is much slower
        else:
            device_multiplier = 2.0
        
        return audio_duration * base_ratio * device_multiplier
    
    def _load_confidence_calibration(self) -> Dict[str, float]:
        """Load Faster-Whisper specific confidence calibration"""
        return {
            'scale': 1.0,
            'offset': 0.0,
            'min_confidence': 0.1,
            'max_confidence': 0.95
        }