"""
Unified Overlap Detection Module

This module provides shared overlap detection logic used by both TurnStabilizer 
and SourceSeparationEngine to ensure consistency and eliminate code duplication.

Key Features:
- Probabilistic overlap scoring with multiple factors
- Configurable thresholds and parameters  
- Support for both basic overlap detection and source separation triggering
- Comprehensive overlap statistics and reporting

Author: Advanced Ensemble Transcription System
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class OverlapDetectionConfig:
    """Configuration for overlap detection parameters"""
    
    # Basic detection thresholds
    overlap_threshold: float = 0.25  # Minimum overlap probability to trigger separation
    min_overlap_duration: float = 0.1  # Minimum overlap duration to consider (seconds)
    confidence_weight: float = 0.3  # Weight for confidence in probability calculation
    
    # Probability calculation factors
    duration_normalization: float = 0.5  # Duration to normalize overlap ratios (seconds)
    speaker_difference_factor: float = 1.0  # Bonus for different speakers overlapping
    same_speaker_penalty: float = 0.7  # Penalty for same speaker "overlaps"
    center_proximity_weight: float = 0.2  # Weight for overlap position within segments
    
    # Advanced options
    enable_center_proximity_bonus: bool = True  # Enable bonus for overlaps near segment centers
    enable_confidence_adjustment: bool = True   # Enable confidence-based adjustments
    enable_duration_scaling: bool = True        # Enable duration-based scaling
    
    # Reporting and metrics
    track_detailed_statistics: bool = True
    generate_overlap_report: bool = True

@dataclass 
class OverlapFrame:
    """Represents a detected overlap frame with metadata"""
    start_time: float
    end_time: float
    duration: float
    overlap_probability: float
    speakers_involved: List[str]
    confidence_score: float
    segment_indices: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OverlapDetectionResult:
    """Complete result from overlap detection analysis"""
    overlap_frames: List[OverlapFrame] = field(default_factory=list)
    overlap_statistics: Dict[str, Any] = field(default_factory=dict)
    requires_source_separation: bool = False
    detection_metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedOverlapDetector:
    """
    Unified overlap detection engine used by multiple components
    
    This class consolidates overlap detection logic to ensure consistency
    between TurnStabilizer and SourceSeparationEngine while providing
    flexible configuration for different use cases.
    """
    
    def __init__(self, config: Optional[OverlapDetectionConfig] = None):
        """
        Initialize unified overlap detector
        
        Args:
            config: Optional configuration. Uses defaults if None.
        """
        self.config = config or OverlapDetectionConfig()
        self.logger = create_enhanced_logger("unified_overlap_detector")
        
        # Processing statistics
        self.detection_history = []
        self.last_detection_result = None
    
    def detect_overlap_frames(self, 
                             segments: List[Dict[str, Any]], 
                             audio_duration: float,
                             source_separation_mode: bool = False) -> OverlapDetectionResult:
        """
        Detect and analyze overlap frames in diarization segments
        
        Args:
            segments: Diarization segments with start, end, speaker_id, confidence
            audio_duration: Total audio duration for normalization
            source_separation_mode: If True, focuses on frames suitable for source separation
            
        Returns:
            Complete overlap detection results with frames and statistics
        """
        if not segments:
            return OverlapDetectionResult()
        
        self.logger.info(f"Starting overlap detection", 
                        context={
                            'segments_count': len(segments),
                            'audio_duration': audio_duration,
                            'overlap_threshold': self.config.overlap_threshold,
                            'source_separation_mode': source_separation_mode
                        })
        
        overlap_frames = []
        total_overlap_duration = 0.0
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        # Find overlapping segment pairs
        for i in range(len(sorted_segments)):
            for j in range(i + 1, len(sorted_segments)):
                seg1 = sorted_segments[i]
                seg2 = sorted_segments[j]
                
                # Check for temporal overlap
                overlap_start = max(seg1['start'], seg2['start'])
                overlap_end = min(seg1['end'], seg2['end'])
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    
                    # Check minimum duration threshold
                    if overlap_duration >= self.config.min_overlap_duration:
                        
                        # Calculate overlap probability
                        overlap_prob = self._calculate_overlap_probability(
                            seg1, seg2, overlap_duration, audio_duration
                        )
                        
                        # Create overlap frame
                        overlap_frame = OverlapFrame(
                            start_time=overlap_start,
                            end_time=overlap_end,
                            duration=overlap_duration,
                            overlap_probability=overlap_prob,
                            speakers_involved=[seg1['speaker_id'], seg2['speaker_id']],
                            confidence_score=min(
                                seg1.get('confidence', 1.0),
                                seg2.get('confidence', 1.0)
                            ),
                            segment_indices=[i, j],
                            metadata={
                                'seg1_duration': seg1['end'] - seg1['start'],
                                'seg2_duration': seg2['end'] - seg2['start'],
                                'overlap_ratio_seg1': overlap_duration / (seg1['end'] - seg1['start']),
                                'overlap_ratio_seg2': overlap_duration / (seg2['end'] - seg2['start']),
                                'seg1_confidence': seg1.get('confidence', 1.0),
                                'seg2_confidence': seg2.get('confidence', 1.0)
                            }
                        )
                        
                        # Add frame if it meets criteria
                        include_frame = True
                        
                        if source_separation_mode:
                            # In source separation mode, only include high-probability overlaps
                            include_frame = overlap_prob >= self.config.overlap_threshold
                        
                        if include_frame:
                            overlap_frames.append(overlap_frame)
                            total_overlap_duration += overlap_duration
        
        # Calculate detection statistics
        high_prob_overlaps = [
            frame for frame in overlap_frames 
            if frame.overlap_probability >= self.config.overlap_threshold
        ]
        
        overlap_statistics = {
            'total_overlap_frames': len(overlap_frames),
            'high_probability_frames': len(high_prob_overlaps),
            'total_overlap_duration': total_overlap_duration,
            'overlap_ratio': total_overlap_duration / audio_duration if audio_duration > 0 else 0.0,
            'avg_overlap_probability': np.mean([f.overlap_probability for f in overlap_frames]) if overlap_frames else 0.0,
            'max_overlap_probability': max([f.overlap_probability for f in overlap_frames]) if overlap_frames else 0.0,
            'avg_overlap_duration': total_overlap_duration / len(overlap_frames) if overlap_frames else 0.0,
            'segments_analyzed': len(segments),
            'audio_duration': audio_duration
        }
        
        # Determine if source separation is needed
        requires_separation = len(high_prob_overlaps) > 0 if source_separation_mode else False
        
        detection_result = OverlapDetectionResult(
            overlap_frames=overlap_frames,
            overlap_statistics=overlap_statistics,
            requires_source_separation=requires_separation,
            detection_metadata={
                'config': {
                    'overlap_threshold': self.config.overlap_threshold,
                    'min_overlap_duration': self.config.min_overlap_duration,
                    'confidence_weight': self.config.confidence_weight
                },
                'source_separation_mode': source_separation_mode,
                'detection_timestamp': np.datetime64('now')
            }
        )
        
        # Store result for history
        self.last_detection_result = detection_result
        if self.config.track_detailed_statistics:
            self.detection_history.append(detection_result)
        
        self.logger.info(f"Overlap detection completed",
                        context={
                            'total_frames': len(overlap_frames),
                            'high_prob_frames': len(high_prob_overlaps),
                            'requires_separation': requires_separation,
                            'total_overlap_duration': total_overlap_duration,
                            'overlap_ratio': overlap_statistics['overlap_ratio']
                        })
        
        return detection_result
    
    def _calculate_overlap_probability(self, 
                                     seg1: Dict[str, Any], 
                                     seg2: Dict[str, Any], 
                                     overlap_duration: float,
                                     audio_duration: float) -> float:
        """
        Calculate overlap probability using multiple factors
        
        Args:
            seg1, seg2: Overlapping segments
            overlap_duration: Duration of overlap
            audio_duration: Total audio duration
            
        Returns:
            Overlap probability score (0.0 to 1.0)
        """
        # Calculate segment durations
        seg1_duration = seg1['end'] - seg1['start']
        seg2_duration = seg2['end'] - seg2['start']
        
        # Base probability from overlap ratios
        overlap_ratio_1 = overlap_duration / seg1_duration if seg1_duration > 0 else 0
        overlap_ratio_2 = overlap_duration / seg2_duration if seg2_duration > 0 else 0
        base_prob = max(overlap_ratio_1, overlap_ratio_2)
        
        # Confidence adjustment
        confidence_factor = 1.0
        if self.config.enable_confidence_adjustment:
            avg_confidence = (seg1.get('confidence', 1.0) + seg2.get('confidence', 1.0)) / 2
            confidence_factor = 1.0 - (1.0 - avg_confidence) * self.config.confidence_weight
        
        # Duration scaling factor
        duration_factor = 1.0
        if self.config.enable_duration_scaling:
            duration_factor = min(1.0, overlap_duration / self.config.duration_normalization)
        
        # Speaker difference factor
        speaker_factor = (
            self.config.speaker_difference_factor 
            if seg1['speaker_id'] != seg2['speaker_id'] 
            else self.config.same_speaker_penalty
        )
        
        # Center proximity factor
        center_factor = 1.0
        if self.config.enable_center_proximity_bonus:
            center_factor = self._calculate_center_proximity_factor(
                seg1, seg2, overlap_duration
            )
        
        # Combined probability
        overlap_prob = (
            base_prob * 
            confidence_factor * 
            duration_factor * 
            speaker_factor * 
            center_factor
        )
        
        return min(1.0, max(0.0, overlap_prob))
    
    def _calculate_center_proximity_factor(self, 
                                         seg1: Dict[str, Any], 
                                         seg2: Dict[str, Any], 
                                         overlap_duration: float) -> float:
        """
        Calculate bonus factor for overlaps occurring near segment centers
        
        Args:
            seg1, seg2: Overlapping segments
            overlap_duration: Duration of overlap
            
        Returns:
            Center proximity factor (0.5 to 1.0)
        """
        try:
            # Calculate segment centers
            seg1_center = (seg1['start'] + seg1['end']) / 2
            seg2_center = (seg2['start'] + seg2['end']) / 2
            
            # Calculate overlap center
            overlap_start = max(seg1['start'], seg2['start'])
            overlap_center = overlap_start + overlap_duration / 2
            
            # Calculate proximity to segment centers
            seg1_duration = seg1['end'] - seg1['start']
            seg2_duration = seg2['end'] - seg2['start']
            
            center_distance_1 = abs(seg1_center - overlap_center)
            center_distance_2 = abs(seg2_center - overlap_center)
            
            # Normalize distances to segment half-lengths
            normalized_distance_1 = center_distance_1 / (seg1_duration / 2) if seg1_duration > 0 else 1.0
            normalized_distance_2 = center_distance_2 / (seg2_duration / 2) if seg2_duration > 0 else 1.0
            
            # Calculate proximity factors (closer to center = higher factor)
            proximity_1 = 1.0 - min(1.0, normalized_distance_1)
            proximity_2 = 1.0 - min(1.0, normalized_distance_2)
            
            # Average proximity with weight
            avg_proximity = (proximity_1 + proximity_2) / 2
            center_factor = 1.0 - self.config.center_proximity_weight + (self.config.center_proximity_weight * avg_proximity)
            
            return max(0.5, min(1.0, center_factor))
            
        except (ZeroDivisionError, ValueError):
            return 1.0  # Neutral factor on calculation errors
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of recent detection activity"""
        if not self.last_detection_result:
            return {'status': 'no_detections'}
        
        result = self.last_detection_result
        return {
            'total_frames_detected': len(result.overlap_frames),
            'high_probability_frames': len([
                f for f in result.overlap_frames 
                if f.overlap_probability >= self.config.overlap_threshold
            ]),
            'total_overlap_duration': result.overlap_statistics.get('total_overlap_duration', 0.0),
            'overlap_ratio': result.overlap_statistics.get('overlap_ratio', 0.0),
            'requires_source_separation': result.requires_source_separation,
            'avg_overlap_probability': result.overlap_statistics.get('avg_overlap_probability', 0.0)
        }
    
    def clear_history(self):
        """Clear detection history for memory management"""
        self.detection_history.clear()
        self.last_detection_result = None

# Convenience function for creating configured detector instances
def create_overlap_detector(overlap_threshold: float = 0.25,
                          min_overlap_duration: float = 0.1,
                          confidence_weight: float = 0.3,
                          **kwargs) -> UnifiedOverlapDetector:
    """
    Create a configured overlap detector instance
    
    Args:
        overlap_threshold: Minimum probability to trigger separation
        min_overlap_duration: Minimum overlap duration to consider
        confidence_weight: Weight for confidence adjustment
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured UnifiedOverlapDetector instance
    """
    config = OverlapDetectionConfig(
        overlap_threshold=overlap_threshold,
        min_overlap_duration=min_overlap_duration,
        confidence_weight=confidence_weight,
        **kwargs
    )
    return UnifiedOverlapDetector(config)