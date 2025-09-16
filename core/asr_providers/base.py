"""
Base ASR Provider interface for multi-engine transcription system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

class DecodeMode(Enum):
    """ASR decode modes for different quality vs speed tradeoffs"""
    CAREFUL = "careful"          # High beam, best_of=5, temp=0.0 - highest quality
    DETERMINISTIC = "deterministic"  # beam=5, temp=0.2 - balanced
    EXPLORATORY = "exploratory"     # temp=0.7, length_penalty=0.1 - diverse
    FAST = "fast"               # Low beam, temp=0.0 - fastest
    ENHANCED = "enhanced"       # Provider-specific enhanced model

@dataclass
class ASRSegment:
    """Individual transcription segment with timing and confidence"""
    start: float
    end: float
    text: str
    confidence: float
    words: Optional[List[Dict[str, Any]]] = None
    speaker_id: Optional[str] = None
    language: Optional[str] = None

@dataclass
class ASRResult:
    """Complete ASR result with metadata and calibrated confidence"""
    segments: List[ASRSegment]
    full_text: str
    language: str
    confidence: float
    calibrated_confidence: float  # Provider-specific calibration
    processing_time: float
    provider: str
    decode_mode: DecodeMode
    model_name: str
    metadata: Dict[str, Any]
    
    @property
    def duration(self) -> float:
        """Total duration of transcribed audio"""
        if not self.segments:
            return 0.0
        return self.segments[-1].end - self.segments[0].start
    
    @property
    def word_count(self) -> int:
        """Approximate word count"""
        return len(self.full_text.split())
    
    def get_text_at_time(self, start_time: float, end_time: float) -> str:
        """Extract text within specified time range"""
        text_parts = []
        for segment in self.segments:
            if segment.start >= end_time:
                break
            if segment.end <= start_time:
                continue
            text_parts.append(segment.text)
        return ' '.join(text_parts)

class ASRProvider(ABC):
    """Abstract base class for ASR providers"""
    
    def __init__(self, provider_name: str, model_name: str, config: Dict[str, Any] = None):
        self.provider_name = provider_name
        self.model_name = model_name
        self.config = config or {}
        self.confidence_calibration = self._load_confidence_calibration()
        
    @abstractmethod
    def transcribe(
        self, 
        audio_path: Union[str, Path], 
        decode_mode: DecodeMode = DecodeMode.DETERMINISTIC,
        language: str = "en",
        prompt: Optional[str] = None,
        **kwargs
    ) -> ASRResult:
        """
        Transcribe audio file and return structured result
        
        Args:
            audio_path: Path to audio file
            decode_mode: Quality vs speed tradeoff
            language: Language code (en for US English)
            prompt: Optional context prompt for better accuracy
            **kwargs: Provider-specific parameters
            
        Returns:
            ASRResult with segments, confidence, and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats"""
        pass
    
    @abstractmethod
    def get_max_file_size(self) -> int:
        """Get maximum file size in bytes"""
        pass
    
    def calibrate_confidence(self, raw_confidence: float, segment_length: float = 1.0) -> float:
        """
        Apply provider-specific confidence calibration
        
        Args:
            raw_confidence: Raw confidence from provider
            segment_length: Segment duration for length-based adjustment
            
        Returns:
            Calibrated confidence score (0.0 to 1.0)
        """
        # Default linear calibration - override in providers
        calibrated = raw_confidence * self.confidence_calibration.get('scale', 1.0)
        calibrated += self.confidence_calibration.get('offset', 0.0)
        
        # Length penalty for very short segments
        if segment_length < 0.5:
            calibrated *= 0.9
        elif segment_length < 0.2:
            calibrated *= 0.8
            
        return max(0.0, min(1.0, calibrated))
    
    def _load_confidence_calibration(self) -> Dict[str, float]:
        """Load provider-specific confidence calibration parameters"""
        # Default calibration - should be overridden with measured values
        return {
            'scale': 1.0,
            'offset': 0.0,
            'min_confidence': 0.1,
            'max_confidence': 0.95
        }
    
    def estimate_processing_time(self, audio_duration: float, decode_mode: DecodeMode) -> float:
        """Estimate processing time for given audio duration and mode"""
        # Default estimate - override in providers
        base_ratio = {
            DecodeMode.FAST: 0.1,
            DecodeMode.DETERMINISTIC: 0.3,
            DecodeMode.CAREFUL: 0.8,
            DecodeMode.EXPLORATORY: 0.5,
            DecodeMode.ENHANCED: 0.4
        }.get(decode_mode, 0.3)
        
        return audio_duration * base_ratio
    
    def __str__(self) -> str:
        return f"{self.provider_name}({self.model_name})"
    
    def __repr__(self) -> str:
        return f"ASRProvider(provider='{self.provider_name}', model='{self.model_name}')"