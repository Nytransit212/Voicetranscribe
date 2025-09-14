"""
Base Diarization Provider Interface

Defines the abstract interface that all diarization providers must implement.
Provides consistent API for external diarization services with standardized
error handling, health checking, and result formatting.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class DiarizationError(Exception):
    """Base exception for diarization provider errors"""
    pass


class ProviderStatus(Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass
class DiarizationResult:
    """
    Standardized diarization result container.
    
    This provides a consistent interface between different provider APIs
    and the existing Annotation/Segment system.
    """
    segments: List[Dict[str, Any]]  # List of {start, end, speaker_id, confidence} dicts
    total_speakers: int
    confidence_score: float  # Overall confidence 0.0-1.0
    processing_time: float  # Seconds
    provider_metadata: Dict[str, Any]  # Provider-specific data
    variant_id: Optional[int] = None
    variant_name: Optional[str] = None
    
    def to_annotation(self):
        """Convert to Annotation object compatible with existing system"""
        # Import here to avoid circular dependencies
        from core.diarization_engine import Annotation, Segment
        
        annotation = Annotation()
        for segment in self.segments:
            start = float(segment['start'])
            end = float(segment['end'])
            speaker_id = str(segment['speaker_id'])
            annotation[Segment(start, end)] = speaker_id
        
        return annotation


class DiarizationProvider(ABC):
    """
    Abstract base class for all diarization providers.
    
    Providers implement speaker diarization using external APIs or local models,
    with standardized interfaces for error handling, health checking, and result formatting.
    """
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        self.provider_name = provider_name
        self.config = config
        self.last_health_check = 0.0
        self.health_check_interval = config.get('health_check_interval', 300)  # 5 minutes
        self._status = ProviderStatus.HEALTHY
        
    @abstractmethod
    def diarize(
        self, 
        audio_path: str, 
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        **kwargs
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum expected speakers (optional)
            max_speakers: Maximum expected speakers (optional)
            **kwargs: Provider-specific parameters
            
        Returns:
            DiarizationResult with speaker segments and metadata
            
        Raises:
            DiarizationError: If processing fails
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate provider configuration and credentials.
        
        Returns:
            True if configuration is valid and provider is accessible
        """
        pass
    
    @abstractmethod
    def health_check(self) -> ProviderStatus:
        """
        Check provider health and availability.
        
        Returns:
            Current provider status
        """
        pass
    
    def get_status(self) -> ProviderStatus:
        """Get current provider status with cache"""
        current_time = time.time()
        if current_time - self.last_health_check > self.health_check_interval:
            self._status = self.health_check()
            self.last_health_check = current_time
        return self._status
    
    def supports_file_size(self, file_size_mb: float) -> bool:
        """Check if provider supports given file size"""
        max_size = self.config.get('max_file_size_mb', 100)  # Default 100MB
        return file_size_mb <= max_size
    
    def supports_duration(self, duration_seconds: float) -> bool:
        """Check if provider supports given audio duration"""
        max_duration = self.config.get('max_duration_seconds', 3600)  # Default 1 hour
        return duration_seconds <= max_duration
    
    def get_estimated_cost(self, duration_seconds: float) -> Optional[float]:
        """Get estimated cost for processing given duration"""
        cost_per_second = self.config.get('cost_per_second', None)
        if cost_per_second is None:
            return None
        return cost_per_second * duration_seconds
    
    def create_variants(
        self, 
        audio_path: str, 
        base_params: Dict[str, Any], 
        num_variants: int = 5
    ) -> List[DiarizationResult]:
        """
        Create multiple diarization variants with different parameters.
        
        Default implementation calls diarize() multiple times with parameter variations.
        Providers can override for more sophisticated variant generation.
        
        Args:
            audio_path: Path to audio file
            base_params: Base parameters for diarization
            num_variants: Number of variants to generate
            
        Returns:
            List of DiarizationResult variants
        """
        variants = []
        
        # Define parameter variations for different variants
        variant_configs = self._get_variant_configurations(base_params, num_variants)
        
        for i, variant_config in enumerate(variant_configs, 1):
            try:
                result = self.diarize(audio_path, **variant_config)
                result.variant_id = i
                result.variant_name = variant_config.get('variant_name', f"Variant_{i}")
                variants.append(result)
            except Exception as e:
                # Log error but continue with other variants
                print(f"⚠ Warning: Failed to generate variant {i}: {e}")
                continue
        
        return variants
    
    def _get_variant_configurations(
        self, 
        base_params: Dict[str, Any], 
        num_variants: int
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter configurations for variants.
        
        Default implementation provides conservative parameter variations.
        Providers should override to provide service-specific optimizations.
        """
        variants = []
        
        # Extract base parameters
        min_speakers = base_params.get('min_speakers', 2)
        max_speakers = base_params.get('max_speakers', 10)
        
        # Variant 1: Conservative (fewer speakers expected)
        variants.append({
            **base_params,
            'min_speakers': max(2, min_speakers - 1),
            'max_speakers': max_speakers - 1,
            'variant_name': 'Conservative',
            'sensitivity': 'low'
        })
        
        # Variant 2: Balanced (expected speaker count)  
        variants.append({
            **base_params,
            'min_speakers': min_speakers,
            'max_speakers': max_speakers,
            'variant_name': 'Balanced',
            'sensitivity': 'medium'
        })
        
        # Variant 3: Liberal (more speakers expected)
        variants.append({
            **base_params,
            'min_speakers': min_speakers,
            'max_speakers': max_speakers + 2,
            'variant_name': 'Liberal', 
            'sensitivity': 'high'
        })
        
        # Variant 4: Multi-resolution
        variants.append({
            **base_params,
            'min_speakers': min_speakers,
            'max_speakers': max_speakers + 1,
            'variant_name': 'Multi_Resolution',
            'sensitivity': 'medium',
            'use_multi_resolution': True
        })
        
        # Variant 5: Ensemble optimized
        variants.append({
            **base_params,
            'min_speakers': min_speakers,
            'max_speakers': max_speakers,
            'variant_name': 'Ensemble_Optimized',
            'sensitivity': 'medium',
            'optimize_for_ensemble': True
        })
        
        return variants[:num_variants]