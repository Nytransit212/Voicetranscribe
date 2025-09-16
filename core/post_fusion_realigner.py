"""
Post-Fusion Alignment Realigner

Fixes micro boundary drift that produces insert/delete errors by realigning
word boundaries to voice activity detection (VAD) energy edges using DTW-style
dynamic programming. Only adjusts timing within strict constraints (80ms max) 
while preserving all word content and sequence order.

Key Features:
- DTW-style boundary alignment with energy edge preferences
- VAD energy curve integration for optimal boundary placement
- Constrained search within configurable time windows
- Thread-safe concurrent processing
- Comprehensive telemetry and performance metrics
- Graceful fallback on realignment failures

Author: Advanced Ensemble Transcription System
"""

import numpy as np
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import math

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import trace_stage, track_cost
from config.hydra_settings import settings

@dataclass
class WordTiming:
    """Individual word with timing information"""
    word: str
    start_time: float
    end_time: float
    confidence: float
    speaker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnergyFrame:
    """Energy information for a time frame"""
    timestamp: float
    energy_level: float
    is_voiced: bool
    is_boundary_candidate: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlignmentCost:
    """Cost components for boundary alignment"""
    energy_cost: float
    distance_cost: float
    smoothness_cost: float
    total_cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BoundaryShift:
    """Information about a boundary adjustment"""
    word_index: int
    original_time: float
    adjusted_time: float
    shift_ms: float
    energy_improvement: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RealignmentResult:
    """Complete result from realignment process"""
    realigned_words: List[WordTiming]
    boundary_shifts: List[BoundaryShift]
    energy_alignment_score: float
    total_shift_ms: float
    mean_shift_ms: float
    max_shift_ms: float
    processing_time: float
    realignment_applied: bool
    fallback_reason: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RealignerConfig:
    """Configuration for post-fusion realigner"""
    enabled: bool = True
    max_boundary_shift_ms: float = 80.0
    energy_window_ms: float = 50.0
    cost_weight_energy: float = 0.6
    cost_weight_distance: float = 0.4
    cost_weight_smoothness: float = 0.2
    min_word_duration_ms: float = 50.0
    energy_threshold: float = 0.3
    boundary_search_radius_ms: float = 100.0
    dtw_constraint_window: float = 0.15  # 15% of sequence length
    enable_fallback: bool = True
    validate_timing_constraints: bool = True

class EnergyBoundaryDetector:
    """Detects optimal word boundaries based on energy transitions"""
    
    def __init__(self, config: RealignerConfig):
        self.config = config
        self.logger = create_enhanced_logger("energy_boundary_detector")
        
    def detect_boundary_candidates(self, 
                                 energy_frames: List[EnergyFrame],
                                 search_center_time: float,
                                 search_radius_ms: float) -> List[Tuple[float, float]]:
        """
        Detect potential word boundary positions based on energy transitions
        
        Args:
            energy_frames: VAD energy curve data
            search_center_time: Center time for boundary search
            search_radius_ms: Search radius in milliseconds
            
        Returns:
            List of (timestamp, boundary_score) tuples
        """
        search_radius_s = search_radius_ms / 1000.0
        search_start = search_center_time - search_radius_s
        search_end = search_center_time + search_radius_s
        
        candidates = []
        
        # Filter frames within search window
        relevant_frames = [
            frame for frame in energy_frames
            if search_start <= frame.timestamp <= search_end
        ]
        
        if len(relevant_frames) < 2:
            return [(search_center_time, 0.5)]  # Fallback to center
        
        # Detect energy transitions (voice activity changes)
        for i in range(1, len(relevant_frames) - 1):
            prev_frame = relevant_frames[i-1]
            current_frame = relevant_frames[i]
            next_frame = relevant_frames[i+1]
            
            # Calculate energy gradient
            energy_gradient = abs(next_frame.energy_level - prev_frame.energy_level)
            
            # Voice activity transition bonus
            voice_transition_bonus = 0.0
            if prev_frame.is_voiced != next_frame.is_voiced:
                voice_transition_bonus = 0.5
            
            # Energy edge strength
            edge_strength = energy_gradient + voice_transition_bonus
            
            # Proximity to center (prefer closer boundaries when scores are equal)
            distance_from_center = abs(current_frame.timestamp - search_center_time)
            proximity_score = 1.0 - (distance_from_center / search_radius_s)
            
            # Combined boundary score
            boundary_score = (edge_strength * 0.7) + (proximity_score * 0.3)
            
            candidates.append((current_frame.timestamp, boundary_score))
        
        # Sort by boundary score (descending) and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:5]  # Return top 5 candidates

class DTWBoundaryAligner:
    """DTW-style dynamic programming for optimal boundary alignment"""
    
    def __init__(self, config: RealignerConfig):
        self.config = config
        self.logger = create_enhanced_logger("dtw_boundary_aligner")
        self.boundary_detector = EnergyBoundaryDetector(config)
        
    def align_boundaries(self,
                        words: List[WordTiming],
                        energy_frames: List[EnergyFrame]) -> List[BoundaryShift]:
        """
        Use DTW-style alignment to find optimal boundary adjustments
        
        Args:
            words: List of words with original timing
            energy_frames: VAD energy curve data
            
        Returns:
            List of boundary shifts to apply
        """
        if not words or not energy_frames:
            return []
        
        boundary_shifts = []
        
        # Process each word boundary (end times, as start of next word)
        for i, word in enumerate(words[:-1]):  # Skip last word (no next boundary)
            original_boundary_time = word.end_time
            next_word = words[i + 1]
            
            # Find optimal boundary position
            optimal_shift = self._find_optimal_boundary_shift(
                original_boundary_time,
                word,
                next_word, 
                energy_frames,
                i
            )
            
            if optimal_shift:
                boundary_shifts.append(optimal_shift)
        
        # Apply smoothness constraints
        boundary_shifts = self._apply_smoothness_constraints(boundary_shifts, words)
        
        return boundary_shifts
    
    def _find_optimal_boundary_shift(self,
                                   original_time: float,
                                   current_word: WordTiming,
                                   next_word: WordTiming,
                                   energy_frames: List[EnergyFrame],
                                   word_index: int) -> Optional[BoundaryShift]:
        """Find optimal shift for a single boundary"""
        
        # Get boundary candidates from energy detector
        candidates = self.boundary_detector.detect_boundary_candidates(
            energy_frames,
            original_time,
            self.config.boundary_search_radius_ms
        )
        
        if not candidates:
            return None
        
        best_candidate = None
        best_cost = float('inf')
        
        for candidate_time, boundary_score in candidates:
            # Calculate shift amount
            shift_ms = abs(candidate_time - original_time) * 1000.0
            
            # Skip if shift exceeds maximum allowed
            if shift_ms > self.config.max_boundary_shift_ms:
                continue
            
            # Calculate timing constraints
            if not self._validate_timing_constraints(
                candidate_time, current_word, next_word, word_index
            ):
                continue
            
            # Calculate alignment costs
            cost = self._calculate_alignment_cost(
                original_time,
                candidate_time,
                boundary_score,
                shift_ms
            )
            
            if cost.total_cost < best_cost:
                best_cost = cost.total_cost
                best_candidate = BoundaryShift(
                    word_index=word_index,
                    original_time=original_time,
                    adjusted_time=candidate_time,
                    shift_ms=shift_ms,
                    energy_improvement=boundary_score,
                    confidence_score=1.0 - (cost.total_cost / 10.0),  # Normalize
                    metadata={
                        'energy_cost': cost.energy_cost,
                        'distance_cost': cost.distance_cost,
                        'smoothness_cost': cost.smoothness_cost,
                        'boundary_score': boundary_score
                    }
                )
        
        return best_candidate
    
    def _calculate_alignment_cost(self,
                                original_time: float,
                                candidate_time: float,
                                boundary_score: float,
                                shift_ms: float) -> AlignmentCost:
        """Calculate alignment cost components"""
        
        # Energy cost (lower is better for high boundary scores)
        energy_cost = (1.0 - boundary_score) * self.config.cost_weight_energy
        
        # Distance cost (prefer smaller movements)
        max_shift = self.config.max_boundary_shift_ms
        distance_cost = (shift_ms / max_shift) * self.config.cost_weight_distance
        
        # Smoothness cost (will be calculated globally)
        smoothness_cost = 0.0
        
        total_cost = energy_cost + distance_cost + smoothness_cost
        
        return AlignmentCost(
            energy_cost=energy_cost,
            distance_cost=distance_cost,
            smoothness_cost=smoothness_cost,
            total_cost=total_cost
        )
    
    def _validate_timing_constraints(self,
                                   candidate_time: float,
                                   current_word: WordTiming,
                                   next_word: WordTiming,
                                   word_index: int) -> bool:
        """Validate that boundary adjustment maintains valid timing constraints"""
        
        if not self.config.validate_timing_constraints:
            return True
        
        min_duration_s = self.config.min_word_duration_ms / 1000.0
        
        # Check current word doesn't become too short
        current_duration = candidate_time - current_word.start_time
        if current_duration < min_duration_s:
            return False
        
        # Check next word doesn't become too short  
        next_duration = next_word.end_time - candidate_time
        if next_duration < min_duration_s:
            return False
        
        # Check for negative durations or time reversals
        if candidate_time <= current_word.start_time:
            return False
        if candidate_time >= next_word.end_time:
            return False
        
        return True
    
    def _apply_smoothness_constraints(self,
                                    boundary_shifts: List[BoundaryShift],
                                    words: List[WordTiming]) -> List[BoundaryShift]:
        """Apply smoothness constraints to reduce erratic boundary movements"""
        
        if len(boundary_shifts) <= 1:
            return boundary_shifts
        
        # Calculate smoothness penalties
        for i, shift in enumerate(boundary_shifts):
            smoothness_penalty = 0.0
            
            # Check consistency with neighboring shifts
            for j in range(max(0, i-2), min(len(boundary_shifts), i+3)):
                if i == j:
                    continue
                
                neighbor_shift = boundary_shifts[j]
                shift_diff = abs(shift.shift_ms - neighbor_shift.shift_ms)
                distance_weight = 1.0 / (abs(i - j) + 1)  # Closer neighbors have more weight
                
                smoothness_penalty += shift_diff * distance_weight * 0.1
            
            # Update shift confidence with smoothness penalty
            shift.confidence_score = max(0.1, shift.confidence_score - smoothness_penalty)
            shift.metadata['smoothness_penalty'] = smoothness_penalty
        
        # Filter out shifts with very low confidence after smoothness adjustment
        filtered_shifts = [
            shift for shift in boundary_shifts
            if shift.confidence_score >= 0.3
        ]
        
        return filtered_shifts

class PostFusionRealigner:
    """Main post-fusion alignment realigner"""
    
    def __init__(self, config: Optional[RealignerConfig] = None):
        """Initialize realigner with configuration"""
        self.config = config or self._load_config_from_settings()
        self.logger = create_enhanced_logger("post_fusion_realigner")
        self.dtw_aligner = DTWBoundaryAligner(self.config)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Telemetry tracking
        self.telemetry = {
            'total_realignments': 0,
            'successful_realignments': 0,
            'fallback_count': 0,
            'total_boundary_shifts': 0,
            'mean_shift_ms_history': [],
            'max_shift_ms_history': [],
            'processing_times': [],
            'energy_improvements': []
        }
    
    def _load_config_from_settings(self) -> RealignerConfig:
        """Load configuration from Hydra settings"""
        try:
            realigner_config = getattr(settings.config, 'realigner', {})
            return RealignerConfig(
                enabled=realigner_config.get('enabled', True),
                max_boundary_shift_ms=realigner_config.get('max_boundary_shift_ms', 80.0),
                energy_window_ms=realigner_config.get('energy_window_ms', 50.0),
                cost_weight_energy=realigner_config.get('cost_weight_energy', 0.6),
                cost_weight_distance=realigner_config.get('cost_weight_distance', 0.4),
                cost_weight_smoothness=realigner_config.get('cost_weight_smoothness', 0.2)
            )
        except Exception as e:
            self.logger.warning(f"Failed to load realigner config from settings, using defaults: {e}")
            return RealignerConfig()
    
    @trace_stage("post_fusion_realignment")
    def realign_boundaries(self,
                          words: List[Dict[str, Any]],
                          energy_frames: List[Dict[str, Any]],
                          audio_duration: float) -> RealignmentResult:
        """
        Realign word boundaries based on VAD energy curve
        
        Args:
            words: List of word dictionaries with timing info
            energy_frames: VAD energy curve data 
            audio_duration: Total audio duration for validation
            
        Returns:
            RealignmentResult with adjusted boundaries and telemetry
        """
        start_time = time.time()
        
        with self._lock:
            self.telemetry['total_realignments'] += 1
        
        if not self.config.enabled:
            self.logger.info("Post-fusion realigner disabled by configuration")
            return self._create_passthrough_result(words, start_time, "disabled")
        
        try:
            # Convert input format to internal format
            word_timings = self._convert_words_to_timing_objects(words)
            energy_curve = self._convert_energy_frames(energy_frames)
            
            # Validate inputs
            validation_error = self._validate_inputs(word_timings, energy_curve, audio_duration)
            if validation_error:
                return self._create_passthrough_result(words, start_time, validation_error)
            
            self.logger.info(
                f"Starting boundary realignment for {len(word_timings)} words",
                context={
                    'max_shift_ms': self.config.max_boundary_shift_ms,
                    'energy_frames': len(energy_curve),
                    'audio_duration': audio_duration
                }
            )
            
            # Perform DTW-style boundary alignment
            boundary_shifts = self.dtw_aligner.align_boundaries(word_timings, energy_curve)
            
            if not boundary_shifts:
                self.logger.info("No beneficial boundary shifts found")
                return self._create_passthrough_result(words, start_time, "no_shifts_found")
            
            # Apply boundary shifts to create realigned transcript
            realigned_words = self._apply_boundary_shifts(word_timings, boundary_shifts)
            
            # Calculate performance metrics
            metrics = self._calculate_realignment_metrics(boundary_shifts, realigned_words)
            
            # Create result
            result = RealignmentResult(
                realigned_words=realigned_words,
                boundary_shifts=boundary_shifts,
                energy_alignment_score=metrics['energy_alignment_score'],
                total_shift_ms=metrics['total_shift_ms'],
                mean_shift_ms=metrics['mean_shift_ms'],
                max_shift_ms=metrics['max_shift_ms'],
                processing_time=time.time() - start_time,
                realignment_applied=True,
                telemetry=metrics
            )
            
            # Update global telemetry
            self._update_telemetry(result)
            
            self.logger.info(
                f"Boundary realignment completed successfully",
                context={
                    'boundary_shifts': len(boundary_shifts),
                    'mean_shift_ms': metrics['mean_shift_ms'],
                    'max_shift_ms': metrics['max_shift_ms'],
                    'processing_time': result.processing_time
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Boundary realignment failed: {str(e)}")
            with self._lock:
                self.telemetry['fallback_count'] += 1
            return self._create_passthrough_result(words, start_time, f"error: {str(e)}")
    
    def _convert_words_to_timing_objects(self, words: List[Dict[str, Any]]) -> List[WordTiming]:
        """Convert word dictionaries to WordTiming objects"""
        word_timings = []
        for word_dict in words:
            word_timing = WordTiming(
                word=word_dict.get('word', ''),
                start_time=word_dict.get('start_time', 0.0),
                end_time=word_dict.get('end_time', 0.0),
                confidence=word_dict.get('confidence', 1.0),
                speaker_id=word_dict.get('speaker_id'),
                metadata=word_dict.get('metadata', {})
            )
            word_timings.append(word_timing)
        return word_timings
    
    def _convert_energy_frames(self, energy_frames: List[Dict[str, Any]]) -> List[EnergyFrame]:
        """Convert energy frame dictionaries to EnergyFrame objects"""
        frames = []
        for frame_dict in energy_frames:
            frame = EnergyFrame(
                timestamp=frame_dict.get('timestamp', 0.0),
                energy_level=frame_dict.get('energy_level', 0.0),
                is_voiced=frame_dict.get('is_voiced', False),
                is_boundary_candidate=frame_dict.get('is_boundary_candidate', False),
                metadata=frame_dict.get('metadata', {})
            )
            frames.append(frame)
        return frames
    
    def _validate_inputs(self,
                        word_timings: List[WordTiming],
                        energy_frames: List[EnergyFrame],
                        audio_duration: float) -> Optional[str]:
        """Validate inputs for realignment"""
        
        if not word_timings:
            return "empty_word_list"
        
        if not energy_frames:
            return "empty_energy_frames"
        
        if len(word_timings) < 2:
            return "insufficient_words"
        
        # Check timing consistency
        for i, word in enumerate(word_timings[:-1]):
            next_word = word_timings[i + 1]
            if word.end_time > next_word.start_time:
                return f"overlapping_words_at_index_{i}"
            if word.start_time >= word.end_time:
                return f"invalid_word_duration_at_index_{i}"
        
        # Validate audio duration consistency
        max_word_time = max(word.end_time for word in word_timings)
        if max_word_time > audio_duration * 1.1:  # 10% tolerance
            return "word_times_exceed_audio_duration"
        
        return None
    
    def _apply_boundary_shifts(self,
                             word_timings: List[WordTiming], 
                             boundary_shifts: List[BoundaryShift]) -> List[WordTiming]:
        """Apply boundary shifts to create realigned word sequence"""
        
        # Create copy of original words
        realigned_words = [
            WordTiming(
                word=w.word,
                start_time=w.start_time,
                end_time=w.end_time,
                confidence=w.confidence,
                speaker_id=w.speaker_id,
                metadata=w.metadata.copy()
            ) for w in word_timings
        ]
        
        # Apply each boundary shift
        for shift in boundary_shifts:
            word_idx = shift.word_index
            
            if 0 <= word_idx < len(realigned_words) - 1:
                # Update current word end time
                realigned_words[word_idx].end_time = shift.adjusted_time
                
                # Update next word start time
                realigned_words[word_idx + 1].start_time = shift.adjusted_time
                
                # Add realignment metadata
                realigned_words[word_idx].metadata['realignment_applied'] = True
                realigned_words[word_idx].metadata['boundary_shift_ms'] = shift.shift_ms
                realigned_words[word_idx + 1].metadata['realignment_applied'] = True
                realigned_words[word_idx + 1].metadata['boundary_shift_ms'] = shift.shift_ms
        
        return realigned_words
    
    def _calculate_realignment_metrics(self,
                                     boundary_shifts: List[BoundaryShift],
                                     realigned_words: List[WordTiming]) -> Dict[str, Any]:
        """Calculate performance and quality metrics for realignment"""
        
        if not boundary_shifts:
            return {
                'energy_alignment_score': 0.0,
                'total_shift_ms': 0.0,
                'mean_shift_ms': 0.0,
                'max_shift_ms': 0.0,
                'boundary_shifts_applied': 0
            }
        
        shift_amounts = [abs(shift.shift_ms) for shift in boundary_shifts]
        energy_improvements = [shift.energy_improvement for shift in boundary_shifts]
        
        return {
            'energy_alignment_score': np.mean(energy_improvements) if energy_improvements else 0.0,
            'total_shift_ms': sum(shift_amounts),
            'mean_shift_ms': np.mean(shift_amounts),
            'max_shift_ms': max(shift_amounts) if shift_amounts else 0.0,
            'boundary_shifts_applied': len(boundary_shifts),
            'shift_distribution': {
                'p50_ms': np.percentile(shift_amounts, 50) if shift_amounts else 0.0,
                'p95_ms': np.percentile(shift_amounts, 95) if shift_amounts else 0.0,
                'p99_ms': np.percentile(shift_amounts, 99) if shift_amounts else 0.0
            }
        }
    
    def _create_passthrough_result(self,
                                 original_words: List[Dict[str, Any]],
                                 start_time: float,
                                 fallback_reason: str) -> RealignmentResult:
        """Create passthrough result when realignment is not applied"""
        
        # Convert back to WordTiming objects for consistency
        word_timings = self._convert_words_to_timing_objects(original_words)
        
        return RealignmentResult(
            realigned_words=word_timings,
            boundary_shifts=[],
            energy_alignment_score=0.0,
            total_shift_ms=0.0,
            mean_shift_ms=0.0,
            max_shift_ms=0.0,
            processing_time=time.time() - start_time,
            realignment_applied=False,
            fallback_reason=fallback_reason,
            telemetry={'fallback_reason': fallback_reason}
        )
    
    def _update_telemetry(self, result: RealignmentResult) -> None:
        """Update global telemetry with result data"""
        
        with self._lock:
            if result.realignment_applied:
                self.telemetry['successful_realignments'] += 1
                self.telemetry['total_boundary_shifts'] += len(result.boundary_shifts)
                
                if result.mean_shift_ms > 0:
                    self.telemetry['mean_shift_ms_history'].append(result.mean_shift_ms)
                    self.telemetry['max_shift_ms_history'].append(result.max_shift_ms)
                
                self.telemetry['energy_improvements'].append(result.energy_alignment_score)
            else:
                self.telemetry['fallback_count'] += 1
            
            self.telemetry['processing_times'].append(result.processing_time)
            
            # Maintain sliding window of recent metrics (last 100 entries)
            for key in ['mean_shift_ms_history', 'max_shift_ms_history', 'energy_improvements', 'processing_times']:
                if len(self.telemetry[key]) > 100:
                    self.telemetry[key] = self.telemetry[key][-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance and telemetry summary"""
        
        with self._lock:
            summary = {
                'total_realignments': self.telemetry['total_realignments'],
                'successful_realignments': self.telemetry['successful_realignments'],
                'fallback_count': self.telemetry['fallback_count'],
                'success_rate': (
                    self.telemetry['successful_realignments'] / max(1, self.telemetry['total_realignments'])
                ),
                'total_boundary_shifts': self.telemetry['total_boundary_shifts'],
            }
            
            # Calculate aggregate statistics
            if self.telemetry['mean_shift_ms_history']:
                summary['aggregate_metrics'] = {
                    'mean_boundary_shift_ms': np.mean(self.telemetry['mean_shift_ms_history']),
                    'p95_boundary_shift_ms': np.percentile(self.telemetry['max_shift_ms_history'], 95) if self.telemetry['max_shift_ms_history'] else 0.0,
                    'mean_energy_improvement': np.mean(self.telemetry['energy_improvements']) if self.telemetry['energy_improvements'] else 0.0,
                    'mean_processing_time': np.mean(self.telemetry['processing_times']) if self.telemetry['processing_times'] else 0.0
                }
            
            return summary

# Factory function for easy instantiation
def create_post_fusion_realigner(config: Optional[RealignerConfig] = None) -> PostFusionRealigner:
    """Create configured PostFusionRealigner instance"""
    return PostFusionRealigner(config)

# Helper functions for transcript format conversion
def convert_transcript_to_realigner_format(transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert fused transcript format to realigner word format"""
    words = []
    
    # Handle different transcript formats
    if 'fused_segments' in transcript_data:
        for segment in transcript_data['fused_segments']:
            if 'words' in segment:
                for word_data in segment['words']:
                    words.append({
                        'word': word_data.get('word', ''),
                        'start_time': word_data.get('start_time', 0.0),
                        'end_time': word_data.get('end_time', 0.0),
                        'confidence': word_data.get('confidence', 1.0),
                        'speaker_id': segment.get('speaker_id'),
                        'metadata': word_data.get('metadata', {})
                    })
    elif 'words' in transcript_data:
        words = transcript_data['words']
    
    return words

def convert_realigner_result_to_transcript_format(
    realignment_result: RealignmentResult,
    original_transcript: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert realigner result back to original transcript format"""
    
    # Create updated transcript with realigned boundaries
    updated_transcript = original_transcript.copy()
    
    # Update word-level timing in the transcript
    if 'fused_segments' in updated_transcript:
        word_idx = 0
        for segment in updated_transcript['fused_segments']:
            if 'words' in segment:
                for i, word_data in enumerate(segment['words']):
                    if word_idx < len(realignment_result.realigned_words):
                        realigned_word = realignment_result.realigned_words[word_idx]
                        segment['words'][i]['start_time'] = realigned_word.start_time
                        segment['words'][i]['end_time'] = realigned_word.end_time
                        segment['words'][i]['metadata'] = realigned_word.metadata
                        word_idx += 1
    
    # Add realignment metadata to transcript
    updated_transcript['realignment_applied'] = realignment_result.realignment_applied
    updated_transcript['realignment_metadata'] = {
        'boundary_shifts_applied': len(realignment_result.boundary_shifts),
        'mean_shift_ms': realignment_result.mean_shift_ms,
        'max_shift_ms': realignment_result.max_shift_ms,
        'energy_alignment_score': realignment_result.energy_alignment_score,
        'processing_time': realignment_result.processing_time,
        'fallback_reason': realignment_result.fallback_reason
    }
    
    return updated_transcript