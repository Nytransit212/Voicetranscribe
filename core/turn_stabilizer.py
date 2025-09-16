"""
Turn Stabilization Module with Median Filtering

This module implements advanced turn stabilization algorithms to eliminate rapid speaker 
transitions and smooth speaker label oscillations in diarization results. It uses median 
filtering and minimum duration constraints to improve speaker boundary quality.

Key Features:
- Rapid transition detection (<100ms threshold)
- Median filtering for speaker label sequence smoothing
- Minimum turn duration enforcement
- Comprehensive stabilization metrics and reporting
- Seamless integration with existing diarization pipeline

Author: Advanced Ensemble Transcription System
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import copy
from collections import Counter
import warnings
from dataclasses import dataclass, field
from utils.structured_logger import StructuredLogger

# Import for enhanced overlap detection
@dataclass
class OverlapDetectionResult:
    """Result from enhanced overlap detection with probability scores"""
    overlap_frames: List[Dict[str, Any]] = field(default_factory=list)
    overlap_statistics: Dict[str, Any] = field(default_factory=dict)
    requires_source_separation: bool = False


@dataclass
class StabilizationConfig:
    """Configuration parameters for turn stabilization"""
    
    # Rapid transition detection
    rapid_transition_threshold: float = 0.1  # 100ms threshold for rapid transitions
    
    # Median filtering parameters
    median_window_size: int = 5  # Window size for median filtering (must be odd)
    min_consensus_ratio: float = 0.6  # Minimum ratio for speaker consensus in window
    
    # Minimum turn duration enforcement
    min_turn_duration: float = 0.5  # Minimum turn duration (500ms)
    preserve_boundary_threshold: float = 0.05  # 50ms threshold for boundary preservation
    
    # Processing options
    enable_median_filtering: bool = True
    enable_min_duration_enforcement: bool = True
    enable_boundary_preservation: bool = True
    
    # Metrics and reporting
    track_detailed_metrics: bool = True
    generate_stability_report: bool = True
    
    # Enhanced overlap detection parameters
    enable_overlap_probability_detection: bool = True
    overlap_probability_threshold: float = 0.25  # Threshold for source separation trigger
    min_overlap_duration_for_separation: float = 0.1  # Minimum overlap duration to consider


@dataclass
class StabilizationMetrics:
    """Comprehensive metrics for turn stabilization analysis"""
    
    # Before stabilization
    original_segments_count: int = 0
    original_transitions_count: int = 0
    rapid_transitions_count: int = 0
    
    # After stabilization
    stabilized_segments_count: int = 0
    stabilized_transitions_count: int = 0
    remaining_rapid_transitions: int = 0
    
    # Improvement metrics
    transitions_eliminated: int = 0
    rapid_transitions_eliminated: int = 0
    segments_merged: int = 0
    stability_improvement_ratio: float = 0.0
    
    # Detailed analysis
    speaker_turn_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    duration_distribution: Dict[str, float] = field(default_factory=dict)
    boundary_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metrics storage
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    segments_merged: int = 0
    
    # Quality metrics
    boundary_preservation_score: float = 0.0
    speaker_consistency_score: float = 0.0
    temporal_smoothness_score: float = 0.0


class TurnStabilizer:
    """
    Advanced turn stabilization system with median filtering and minimum duration enforcement.
    
    This class implements sophisticated algorithms to eliminate rapid speaker transitions
    and smooth speaker label oscillations while preserving natural speaker boundaries
    and content integrity.
    """
    
    def __init__(self, config: Optional[StabilizationConfig] = None):
        """
        Initialize the TurnStabilizer with configuration parameters.
        
        Args:
            config: Optional configuration object. Uses defaults if None.
        """
        self.config = config or StabilizationConfig()
        self.logger = StructuredLogger("turn_stabilizer")
        
        # Validate configuration
        self._validate_config()
        
        # Processing state
        self.last_stabilization_metrics = None
        self.processing_history = []
        
    def _validate_config(self) -> None:
        """Validate configuration parameters and issue warnings for potential issues"""
        
        if self.config.median_window_size % 2 == 0:
            warnings.warn(
                f"Median window size should be odd, got {self.config.median_window_size}. "
                f"Adjusting to {self.config.median_window_size + 1}."
            )
            self.config.median_window_size += 1
            
        if self.config.median_window_size < 3:
            warnings.warn("Median window size too small, setting to minimum value of 3.")
            self.config.median_window_size = 3
            
        if self.config.min_consensus_ratio < 0.5:
            warnings.warn("Min consensus ratio should be >= 0.5 for effective filtering.")
            
        if self.config.min_turn_duration < 0.1:
            warnings.warn("Minimum turn duration is very small, may affect speech quality.")
    
    def stabilize_segments(self, segments: List[Dict[str, Any]], 
                          variant_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], StabilizationMetrics]:
        """
        Apply turn stabilization to a list of diarization segments.
        
        Args:
            segments: List of diarization segments with start, end, speaker_id, confidence
            variant_id: Optional identifier for this variant (for logging)
            
        Returns:
            Tuple of (stabilized_segments, metrics)
        """
        if not segments:
            return segments, StabilizationMetrics()
            
        self.logger.info(f"stabilization_start: variant_id={variant_id}, original_segments={len(segments)}")
        
        # Deep copy to avoid modifying original segments
        working_segments = copy.deepcopy(segments)
        
        # Sort segments by start time
        working_segments.sort(key=lambda x: x['start'])
        
        # Initialize metrics
        metrics = self._initialize_metrics(working_segments)
        
        # Step 1: Detect rapid transitions
        rapid_transitions = self._detect_rapid_transitions(working_segments)
        metrics.rapid_transitions_count = len(rapid_transitions)
        
        # Step 2: Apply median filtering if enabled
        if self.config.enable_median_filtering:
            working_segments, median_metrics = self._apply_median_filtering(
                working_segments, rapid_transitions
            )
            metrics = self._merge_metrics(metrics, median_metrics)
        
        # Step 3: Enforce minimum turn duration if enabled
        if self.config.enable_min_duration_enforcement:
            working_segments, duration_metrics = self._enforce_minimum_duration(
                working_segments
            )
            metrics = self._merge_metrics(metrics, duration_metrics)
        
        # Step 4: Final metrics calculation
        metrics = self._finalize_metrics(metrics, working_segments)
        
        # Add stabilization flags to segments
        self._add_stabilization_flags(working_segments, metrics)
        
        # Store metrics for reporting
        self.last_stabilization_metrics = metrics
        
        self.logger.info(f"stabilization_complete: variant_id={variant_id}, "
                        f"original_segments={metrics.original_segments_count}, "
                        f"stabilized_segments={metrics.stabilized_segments_count}, "
                        f"transitions_eliminated={metrics.transitions_eliminated}")
        
        return working_segments, metrics
    
    def detect_overlap_frames_with_probability(self, segments: List[Dict[str, Any]], 
                                             audio_duration: float) -> OverlapDetectionResult:
        """
        Enhanced overlap detection with probability calculation for source separation
        
        Args:
            segments: Diarization segments with start, end, speaker_id, confidence
            audio_duration: Total audio duration for normalization
            
        Returns:
            OverlapDetectionResult with overlap frames and statistics
        """
        if not self.config.enable_overlap_probability_detection:
            return OverlapDetectionResult()
        
        overlap_frames = []
        total_overlap_duration = 0.0
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        # Find overlapping segments and calculate probabilities
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
                    if overlap_duration >= self.config.min_overlap_duration_for_separation:
                        # Calculate overlap probability
                        overlap_prob = self._calculate_overlap_probability_score(
                            seg1, seg2, overlap_duration, audio_duration
                        )
                        
                        overlap_frame = {
                            'start_time': overlap_start,
                            'end_time': overlap_end,
                            'duration': overlap_duration,
                            'overlap_probability': overlap_prob,
                            'speakers_involved': [seg1['speaker_id'], seg2['speaker_id']],
                            'confidence_score': min(
                                seg1.get('confidence', 1.0),
                                seg2.get('confidence', 1.0)
                            ),
                            'segment_indices': [i, j],
                            'metadata': {
                                'seg1_duration': seg1['end'] - seg1['start'],
                                'seg2_duration': seg2['end'] - seg2['start'],
                                'overlap_ratio_seg1': overlap_duration / (seg1['end'] - seg1['start']),
                                'overlap_ratio_seg2': overlap_duration / (seg2['end'] - seg2['start'])
                            }
                        }
                        
                        overlap_frames.append(overlap_frame)
                        total_overlap_duration += overlap_duration
        
        # Determine if source separation is needed
        high_prob_overlaps = [
            frame for frame in overlap_frames 
            if frame['overlap_probability'] >= self.config.overlap_probability_threshold
        ]
        requires_separation = len(high_prob_overlaps) > 0
        
        # Calculate statistics
        overlap_statistics = {
            'total_overlap_frames': len(overlap_frames),
            'high_probability_frames': len(high_prob_overlaps),
            'total_overlap_duration': total_overlap_duration,
            'overlap_ratio': total_overlap_duration / audio_duration if audio_duration > 0 else 0.0,
            'avg_overlap_probability': np.mean([f['overlap_probability'] for f in overlap_frames]) if overlap_frames else 0.0,
            'max_overlap_probability': max([f['overlap_probability'] for f in overlap_frames]) if overlap_frames else 0.0
        }
        
        self.logger.info(f"Overlap detection: {len(overlap_frames)} total frames, "
                        f"{len(high_prob_overlaps)} requiring separation (≥{self.config.overlap_probability_threshold})")
        
        return OverlapDetectionResult(
            overlap_frames=overlap_frames,
            overlap_statistics=overlap_statistics,
            requires_source_separation=requires_separation
        )
    
    def _calculate_overlap_probability_score(self, 
                                           seg1: Dict[str, Any], 
                                           seg2: Dict[str, Any], 
                                           overlap_duration: float,
                                           audio_duration: float) -> float:
        """
        Calculate overlap probability based on multiple factors
        
        Args:
            seg1, seg2: Overlapping segments
            overlap_duration: Duration of overlap
            audio_duration: Total audio duration
            
        Returns:
            Overlap probability (0.0 to 1.0)
        """
        # Base probability from overlap duration
        seg1_duration = seg1['end'] - seg1['start']
        seg2_duration = seg2['end'] - seg2['start']
        
        # Overlap ratio relative to each segment
        overlap_ratio_1 = overlap_duration / seg1_duration
        overlap_ratio_2 = overlap_duration / seg2_duration
        
        # Base probability from maximum overlap ratio
        base_prob = max(overlap_ratio_1, overlap_ratio_2)
        
        # Confidence adjustment
        avg_confidence = (seg1.get('confidence', 1.0) + seg2.get('confidence', 1.0)) / 2
        confidence_factor = 1.0 - (1.0 - avg_confidence) * 0.3  # 30% weight for confidence
        
        # Duration adjustment (longer overlaps are more likely to be real)
        duration_factor = min(1.0, overlap_duration / 0.5)  # Normalize to 0.5 seconds
        
        # Speaker difference factor (different speakers more likely to overlap)
        speaker_factor = 1.0 if seg1['speaker_id'] != seg2['speaker_id'] else 0.7
        
        # Temporal position factor (overlaps in middle of segments more likely)
        seg1_center = (seg1['start'] + seg1['end']) / 2
        seg2_center = (seg2['start'] + seg2['end']) / 2
        overlap_center = (overlap_duration) / 2
        
        # Check if overlap is near segment centers
        center_proximity_1 = 1.0 - abs(seg1_center - overlap_center) / (seg1_duration / 2) if seg1_duration > 0 else 0.5
        center_proximity_2 = 1.0 - abs(seg2_center - overlap_center) / (seg2_duration / 2) if seg2_duration > 0 else 0.5
        center_factor = (center_proximity_1 + center_proximity_2) / 2
        
        # Combined probability
        overlap_prob = base_prob * confidence_factor * duration_factor * speaker_factor * center_factor
        
        return min(1.0, max(0.0, overlap_prob))
    
    def _detect_rapid_transitions(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect rapid speaker transitions below the threshold.
        
        Args:
            segments: List of segments to analyze
            
        Returns:
            List of rapid transition descriptors
        """
        rapid_transitions = []
        
        for i in range(len(segments) - 1):
            current_seg = segments[i]
            next_seg = segments[i + 1]
            
            # Calculate gap between segments
            gap_duration = next_seg['start'] - current_seg['end']
            
            # Check for rapid transition (including overlaps)
            if gap_duration < self.config.rapid_transition_threshold:
                # Check if speakers are different
                if current_seg['speaker_id'] != next_seg['speaker_id']:
                    rapid_transitions.append({
                        'index': i,
                        'current_speaker': current_seg['speaker_id'],
                        'next_speaker': next_seg['speaker_id'],
                        'gap_duration': gap_duration,
                        'current_end': current_seg['end'],
                        'next_start': next_seg['start'],
                        'is_overlap': gap_duration < 0
                    })
        
        return rapid_transitions
    
    def _apply_median_filtering(self, segments: List[Dict[str, Any]], 
                               rapid_transitions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply median filtering to smooth speaker label oscillations.
        
        Args:
            segments: List of segments to filter
            rapid_transitions: List of detected rapid transitions
            
        Returns:
            Tuple of (filtered_segments, filtering_metrics)
        """
        if len(segments) < self.config.median_window_size:
            return segments, {'transitions_smoothed': 0, 'labels_changed': 0}
        
        # Create temporal sequence for analysis
        temporal_sequence = self._create_temporal_sequence(segments)
        
        # Apply median filtering to speaker labels
        smoothed_sequence = self._median_filter_speaker_sequence(temporal_sequence)
        
        # Convert back to segments
        filtered_segments = self._sequence_to_segments(smoothed_sequence, segments)
        
        # Calculate filtering metrics
        filtering_metrics = {
            'transitions_smoothed': len(rapid_transitions),
            'labels_changed': self._count_label_changes(segments, filtered_segments),
            'sequence_length': len(temporal_sequence)
        }
        
        return filtered_segments, filtering_metrics
    
    def _create_temporal_sequence(self, segments: List[Dict[str, Any]], 
                                 resolution: float = 0.01) -> List[Dict[str, Any]]:
        """
        Create a high-resolution temporal sequence for median filtering.
        
        Args:
            segments: Input segments
            resolution: Temporal resolution in seconds (default 10ms)
            
        Returns:
            List of temporal points with speaker labels
        """
        if not segments:
            return []
        
        # Find temporal bounds
        start_time = min(seg['start'] for seg in segments)
        end_time = max(seg['end'] for seg in segments)
        
        # Create temporal grid
        temporal_points = []
        current_time = start_time
        
        while current_time <= end_time:
            # Find which segment(s) are active at this time
            active_speakers = []
            for seg in segments:
                if seg['start'] <= current_time <= seg['end']:
                    active_speakers.append({
                        'speaker_id': seg['speaker_id'],
                        'confidence': seg.get('confidence', 1.0)
                    })
            
            # Handle overlaps by choosing highest confidence speaker
            if active_speakers:
                best_speaker = max(active_speakers, key=lambda x: x['confidence'])
                temporal_points.append({
                    'time': current_time,
                    'speaker_id': best_speaker['speaker_id'],
                    'confidence': best_speaker['confidence']
                })
            
            current_time += resolution
        
        return temporal_points
    
    def _median_filter_speaker_sequence(self, temporal_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply median filtering to the temporal speaker sequence.
        
        Args:
            temporal_sequence: High-resolution temporal sequence
            
        Returns:
            Smoothed temporal sequence
        """
        if len(temporal_sequence) < self.config.median_window_size:
            return temporal_sequence
        
        smoothed_sequence = []
        half_window = self.config.median_window_size // 2
        
        for i in range(len(temporal_sequence)):
            # Define window bounds
            window_start = max(0, i - half_window)
            window_end = min(len(temporal_sequence), i + half_window + 1)
            
            # Extract window speaker labels
            window_labels = [temporal_sequence[j]['speaker_id'] for j in range(window_start, window_end)]
            
            # Find consensus speaker using modified median
            consensus_speaker = self._find_consensus_speaker(
                window_labels, temporal_sequence[i]['speaker_id']
            )
            
            # Create smoothed point
            smoothed_point = temporal_sequence[i].copy()
            smoothed_point['speaker_id'] = consensus_speaker
            smoothed_point['original_speaker'] = temporal_sequence[i]['speaker_id']
            smoothed_sequence.append(smoothed_point)
        
        return smoothed_sequence
    
    def _find_consensus_speaker(self, window_labels: List[str], current_speaker: str) -> str:
        """
        Find consensus speaker in a window using intelligent voting.
        
        Args:
            window_labels: Speaker labels in the current window
            current_speaker: Current speaker label
            
        Returns:
            Consensus speaker ID
        """
        if not window_labels:
            return current_speaker
        
        # Count speaker occurrences
        speaker_counts = Counter(window_labels)
        total_labels = len(window_labels)
        
        # Find most frequent speaker
        most_frequent_speaker, max_count = speaker_counts.most_common(1)[0]
        consensus_ratio = max_count / total_labels
        
        # Use consensus speaker if it meets the threshold
        if consensus_ratio >= self.config.min_consensus_ratio:
            return most_frequent_speaker
        else:
            # Keep original speaker if no strong consensus
            return current_speaker
    
    def _sequence_to_segments(self, temporal_sequence: List[Dict[str, Any]], 
                             original_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert smoothed temporal sequence back to segments.
        
        Args:
            temporal_sequence: Smoothed temporal sequence
            original_segments: Original segments for reference
            
        Returns:
            List of smoothed segments
        """
        if not temporal_sequence:
            return original_segments
        
        # Group consecutive points with same speaker
        filtered_segments = []
        current_speaker = None
        segment_start = None
        
        for point in temporal_sequence:
            if point['speaker_id'] != current_speaker:
                # End previous segment
                if current_speaker is not None:
                    filtered_segments.append({
                        'start': segment_start,
                        'end': point['time'],
                        'speaker_id': current_speaker,
                        'confidence': self._estimate_segment_confidence(
                            segment_start if segment_start is not None else 0.0, 
                            point['time'] if point['time'] is not None else 0.0, 
                            original_segments
                        ),
                        'filtered': True
                    })
                
                # Start new segment
                current_speaker = point['speaker_id']
                segment_start = point['time']
        
        # Close final segment
        if current_speaker is not None and temporal_sequence:
            filtered_segments.append({
                'start': segment_start,
                'end': temporal_sequence[-1]['time'],
                'speaker_id': current_speaker,
                'confidence': self._estimate_segment_confidence(
                    segment_start if segment_start is not None else 0.0, 
                    temporal_sequence[-1]['time'] if temporal_sequence[-1]['time'] is not None else 0.0, 
                    original_segments
                ),
                'filtered': True
            })
        
        return filtered_segments
    
    def _estimate_segment_confidence(self, start: float, end: float, 
                                   original_segments: List[Dict[str, Any]]) -> float:
        """
        Estimate confidence for a filtered segment based on overlapping original segments.
        
        Args:
            start: Segment start time
            end: Segment end time
            original_segments: Original segments for reference
            
        Returns:
            Estimated confidence score
        """
        overlapping_confidences = []
        
        for seg in original_segments:
            # Check for temporal overlap
            overlap_start = max(start, seg['start'])
            overlap_end = min(end, seg['end'])
            
            if overlap_start < overlap_end:
                # Weight confidence by overlap duration
                overlap_duration = overlap_end - overlap_start
                segment_duration = end - start
                if segment_duration > 0:
                    weight = overlap_duration / segment_duration
                    overlapping_confidences.append(seg.get('confidence', 1.0) * weight)
        
        return float(np.mean(overlapping_confidences)) if overlapping_confidences else 0.8
    
    def _enforce_minimum_duration(self, segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Enforce minimum turn duration by merging short segments.
        
        Args:
            segments: List of segments to process
            
        Returns:
            Tuple of (processed_segments, duration_metrics)
        """
        if not segments:
            return segments, {'segments_merged': 0, 'duration_violations': 0}
        
        processed_segments = []
        segments_merged = 0
        duration_violations = 0
        
        i = 0
        while i < len(segments):
            current_seg = segments[i]
            duration = current_seg['end'] - current_seg['start']
            
            if duration < self.config.min_turn_duration:
                duration_violations += 1
                
                # Try to merge with adjacent segments
                merged_seg = self._merge_short_segment(segments, i)
                if merged_seg:
                    processed_segments.append(merged_seg)
                    segments_merged += 1
                    # Skip merged segments
                    i = merged_seg.get('last_merged_index', i) + 1
                else:
                    # Keep original if merging fails
                    processed_segments.append(current_seg)
                    i += 1
            else:
                processed_segments.append(current_seg)
                i += 1
        
        duration_metrics = {
            'segments_merged': segments_merged,
            'duration_violations': duration_violations
        }
        
        return processed_segments, duration_metrics
    
    def _merge_short_segment(self, segments: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
        """
        Merge a short segment with adjacent segments intelligently.
        
        Args:
            segments: All segments
            index: Index of the short segment to merge
            
        Returns:
            Merged segment or None if merging fails
        """
        current_seg = segments[index]
        
        # Consider adjacent segments for merging
        merge_candidates = []
        
        # Previous segment
        if index > 0:
            prev_seg = segments[index - 1]
            merge_candidates.append({
                'segment': prev_seg,
                'type': 'previous',
                'index': index - 1,
                'speaker_match': prev_seg['speaker_id'] == current_seg['speaker_id']
            })
        
        # Next segment
        if index < len(segments) - 1:
            next_seg = segments[index + 1]
            merge_candidates.append({
                'segment': next_seg,
                'type': 'next',
                'index': index + 1,
                'speaker_match': next_seg['speaker_id'] == current_seg['speaker_id']
            })
        
        if not merge_candidates:
            return None
        
        # Prefer merging with same speaker
        same_speaker_candidates = [c for c in merge_candidates if c['speaker_match']]
        if same_speaker_candidates:
            best_candidate = same_speaker_candidates[0]  # Use first same-speaker match
        else:
            # Merge with highest confidence adjacent segment
            best_candidate = max(merge_candidates, 
                               key=lambda c: c['segment'].get('confidence', 0.0))
        
        # Create merged segment
        merge_target = best_candidate['segment']
        
        merged_segment = {
            'start': min(current_seg['start'], merge_target['start']),
            'end': max(current_seg['end'], merge_target['end']),
            'speaker_id': merge_target['speaker_id'],  # Use target speaker
            'confidence': float(np.mean([
                current_seg.get('confidence', 1.0),
                merge_target.get('confidence', 1.0)
            ])),
            'merged': True,
            'merged_segments': [current_seg['speaker_id'], merge_target['speaker_id']],
            'last_merged_index': max(index, best_candidate['index'])
        }
        
        return merged_segment
    
    def _initialize_metrics(self, segments: List[Dict[str, Any]]) -> StabilizationMetrics:
        """Initialize metrics object with original segment statistics"""
        
        metrics = StabilizationMetrics()
        metrics.original_segments_count = len(segments)
        
        # Count original transitions
        for i in range(len(segments) - 1):
            if segments[i]['speaker_id'] != segments[i + 1]['speaker_id']:
                metrics.original_transitions_count += 1
        
        return metrics
    
    def _merge_metrics(self, base_metrics: StabilizationMetrics, 
                      new_metrics: Dict[str, Any]) -> StabilizationMetrics:
        """Merge new metrics into base metrics object"""
        
        # Add new metric fields dynamically
        for key, value in new_metrics.items():
            if hasattr(base_metrics, key):
                setattr(base_metrics, key, getattr(base_metrics, key) + value)
            else:
                # Store in a generic metrics dict if field doesn't exist
                if not hasattr(base_metrics, 'additional_metrics'):
                    base_metrics.additional_metrics = {}
                base_metrics.additional_metrics[key] = value
        
        return base_metrics
    
    def _finalize_metrics(self, metrics: StabilizationMetrics, 
                         final_segments: List[Dict[str, Any]]) -> StabilizationMetrics:
        """Calculate final metrics and improvement ratios"""
        
        metrics.stabilized_segments_count = len(final_segments)
        
        # Count final transitions
        for i in range(len(final_segments) - 1):
            if final_segments[i]['speaker_id'] != final_segments[i + 1]['speaker_id']:
                metrics.stabilized_transitions_count += 1
        
        # Calculate improvements
        metrics.transitions_eliminated = (
            metrics.original_transitions_count - metrics.stabilized_transitions_count
        )
        
        # Calculate remaining rapid transitions
        final_rapid_transitions = self._detect_rapid_transitions(final_segments)
        metrics.remaining_rapid_transitions = len(final_rapid_transitions)
        metrics.rapid_transitions_eliminated = (
            metrics.rapid_transitions_count - metrics.remaining_rapid_transitions
        )
        
        # Calculate improvement ratio
        if metrics.original_transitions_count > 0:
            metrics.stability_improvement_ratio = (
                metrics.transitions_eliminated / metrics.original_transitions_count
            )
        
        # Calculate quality scores
        metrics.boundary_preservation_score = self._calculate_boundary_preservation_score(final_segments)
        metrics.speaker_consistency_score = self._calculate_speaker_consistency_score(final_segments)
        metrics.temporal_smoothness_score = self._calculate_temporal_smoothness_score(final_segments)
        
        return metrics
    
    def _add_stabilization_flags(self, segments: List[Dict[str, Any]], 
                                metrics: StabilizationMetrics) -> None:
        """Add stabilization metadata to segments"""
        
        for segment in segments:
            segment['stabilized_turns'] = {
                'processed': True,
                'original_transitions': metrics.original_transitions_count,
                'stabilized_transitions': metrics.stabilized_transitions_count,
                'transitions_eliminated': metrics.transitions_eliminated,
                'rapid_transitions_eliminated': metrics.rapid_transitions_eliminated,
                'stability_improvement_ratio': metrics.stability_improvement_ratio
            }
    
    def _count_label_changes(self, original_segments: List[Dict[str, Any]], 
                            filtered_segments: List[Dict[str, Any]]) -> int:
        """Count how many speaker labels were changed during filtering"""
        
        # This is a simplified count - in practice you'd do temporal alignment
        changes = 0
        
        # Create a rough mapping by comparing segment counts per speaker
        orig_speakers = Counter(seg['speaker_id'] for seg in original_segments)
        filt_speakers = Counter(seg['speaker_id'] for seg in filtered_segments)
        
        # Count differences
        for speaker in set(orig_speakers.keys()) | set(filt_speakers.keys()):
            changes += abs(orig_speakers.get(speaker, 0) - filt_speakers.get(speaker, 0))
        
        return changes
    
    def _calculate_boundary_preservation_score(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate how well original speaker boundaries were preserved"""
        
        # Simplified scoring based on segment continuity
        if len(segments) <= 1:
            return 1.0
        
        natural_boundaries = 0
        total_boundaries = len(segments) - 1
        
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            # Check if boundary looks natural (reasonable gap/overlap)
            gap = next_seg['start'] - current['end']
            if -0.1 <= gap <= 2.0:  # Allow small overlaps and reasonable pauses
                natural_boundaries += 1
        
        return natural_boundaries / total_boundaries if total_boundaries > 0 else 1.0
    
    def _calculate_speaker_consistency_score(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate speaker consistency (fewer very short segments)"""
        
        if not segments:
            return 1.0
        
        # Count segments that meet minimum duration
        acceptable_segments = sum(
            1 for seg in segments 
            if (seg['end'] - seg['start']) >= self.config.min_turn_duration
        )
        
        return acceptable_segments / len(segments)
    
    def _calculate_temporal_smoothness_score(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate temporal smoothness (fewer rapid transitions)"""
        
        if len(segments) <= 1:
            return 1.0
        
        rapid_transitions = self._detect_rapid_transitions(segments)
        total_transitions = len(segments) - 1
        
        if total_transitions == 0:
            return 1.0
        
        smooth_transitions = total_transitions - len(rapid_transitions)
        return smooth_transitions / total_transitions
    
    def generate_stabilization_report(self, metrics: Optional[StabilizationMetrics] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive stabilization report.
        
        Args:
            metrics: Optional metrics object. Uses last processed if None.
            
        Returns:
            Detailed stabilization report
        """
        if metrics is None:
            metrics = self.last_stabilization_metrics
        
        if metrics is None:
            return {"error": "No stabilization metrics available"}
        
        report = {
            "stabilization_summary": {
                "original_segments": metrics.original_segments_count,
                "stabilized_segments": metrics.stabilized_segments_count,
                "transitions_eliminated": metrics.transitions_eliminated,
                "rapid_transitions_eliminated": metrics.rapid_transitions_eliminated,
                "stability_improvement_ratio": f"{metrics.stability_improvement_ratio:.3f}",
                "segments_merged": metrics.segments_merged if hasattr(metrics, 'segments_merged') else 0
            },
            "quality_scores": {
                "boundary_preservation": f"{metrics.boundary_preservation_score:.3f}",
                "speaker_consistency": f"{metrics.speaker_consistency_score:.3f}",
                "temporal_smoothness": f"{metrics.temporal_smoothness_score:.3f}"
            },
            "transition_analysis": {
                "original_transitions": metrics.original_transitions_count,
                "original_rapid_transitions": metrics.rapid_transitions_count,
                "final_transitions": metrics.stabilized_transitions_count,
                "final_rapid_transitions": metrics.remaining_rapid_transitions,
                "rapid_transition_reduction": f"{((metrics.rapid_transitions_count - metrics.remaining_rapid_transitions) / max(metrics.rapid_transitions_count, 1) * 100):.1f}%"
            },
            "configuration": {
                "rapid_transition_threshold": f"{self.config.rapid_transition_threshold:.3f}s",
                "median_window_size": self.config.median_window_size,
                "min_turn_duration": f"{self.config.min_turn_duration:.3f}s",
                "median_filtering_enabled": self.config.enable_median_filtering,
                "min_duration_enforcement_enabled": self.config.enable_min_duration_enforcement
            }
        }
        
        return report
    
    def batch_stabilize_variants(self, variant_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply turn stabilization to multiple diarization variants.
        
        Args:
            variant_list: List of diarization variant dictionaries
            
        Returns:
            List of stabilized variants with metrics
        """
        stabilized_variants = []
        
        for i, variant in enumerate(variant_list):
            variant_id = variant.get('variant_id', f"variant_{i}")
            
            try:
                # Extract segments from variant
                segments = variant.get('segments', [])
                
                # Apply stabilization
                stabilized_segments, metrics = self.stabilize_segments(segments, variant_id)
                
                # Create new variant with stabilized data
                stabilized_variant = copy.deepcopy(variant)
                stabilized_variant['segments'] = stabilized_segments
                stabilized_variant['stabilization_metrics'] = metrics
                stabilized_variant['stabilization_report'] = self.generate_stabilization_report(metrics)
                
                # Update variant metadata
                stabilized_variant['processed_with_stabilization'] = True
                stabilized_variant['original_segment_count'] = len(segments)
                stabilized_variant['stabilized_segment_count'] = len(stabilized_segments)
                
                stabilized_variants.append(stabilized_variant)
                
            except Exception as e:
                self.logger.error(f"stabilization_error: variant_id={variant_id}, error={str(e)}")
                
                # Add original variant with error flag
                error_variant = copy.deepcopy(variant)
                error_variant['stabilization_error'] = str(e)
                error_variant['processed_with_stabilization'] = False
                stabilized_variants.append(error_variant)
        
        return stabilized_variants