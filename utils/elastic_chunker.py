"""
Elastic Chunking System

This module provides intelligent audio chunking that adapts chunk sizes based on 
overlap density and speech patterns to reduce boundary errors and improve transcription accuracy.

Key Features:
- Dynamic chunk sizing (15-60s) based on audio characteristics
- Speech pattern analysis with VAD, overlap detection, and pause detection
- Boundary error reduction through word/sentence/speaker turn alignment
- Integration with existing overlap detection and audio processing systems
- Comprehensive metrics and telemetry for performance monitoring

Author: Advanced Ensemble Transcription System
"""

import librosa
import numpy as np
import soundfile as sf
import tempfile
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.overlap_detector import UnifiedOverlapDetector, OverlapDetectionConfig, OverlapDetectionResult
from utils.intelligent_cache import cached_operation, get_cache_manager
from utils.observability import trace_stage, track_cost


@dataclass
class ChunkingConfig:
    """Configuration for elastic chunking parameters"""
    
    # Core chunking parameters
    enabled: bool = True
    min_chunk_seconds: float = 15.0      # Minimum chunk size
    max_chunk_seconds: float = 60.0      # Maximum chunk size
    target_chunk_seconds: float = 30.0   # Default target size
    overlap_threshold: float = 0.3       # Overlap threshold for size decisions
    
    # Speech analysis parameters
    vad_frame_length_ms: float = 25.0    # VAD frame length
    vad_hop_length_ms: float = 10.0      # VAD hop length
    energy_smoothing_window: int = 5     # Energy smoothing window
    voice_activity_threshold: float = 0.015  # Voice activity threshold
    silence_threshold_db: float = -40.0  # Silence detection threshold (dB)
    
    # Boundary detection parameters
    pause_min_duration: float = 0.3      # Minimum pause for boundary consideration
    pause_max_duration: float = 2.0      # Maximum pause to consider (avoid long silences)
    speaker_turn_weight: float = 1.5     # Weight bonus for speaker turn boundaries
    energy_valley_weight: float = 1.2    # Weight for energy valley boundaries
    word_boundary_preference: float = 0.8 # Preference for word boundaries
    
    # Adaptive logic parameters
    high_overlap_threshold: float = 0.4   # Threshold for high overlap density
    low_overlap_threshold: float = 0.15   # Threshold for low overlap density
    complexity_analysis_window: float = 10.0  # Window for complexity analysis
    boundary_search_window: float = 5.0   # Search window around target boundary
    
    # Performance and caching
    enable_parallel_analysis: bool = True
    cache_audio_analysis: bool = True
    max_analysis_workers: int = 3
    chunk_overlap_seconds: float = 1.0    # Overlap between chunks for continuity


@dataclass
class SpeechCharacteristics:
    """Analysis results for speech characteristics in an audio segment"""
    start_time: float
    end_time: float
    duration: float
    
    # VAD analysis
    voice_activity_ratio: float
    speech_segments: List[Tuple[float, float]]  # (start, end) of speech regions
    silence_segments: List[Tuple[float, float]]  # (start, end) of silence regions
    
    # Energy analysis
    average_energy: float
    energy_variance: float
    energy_valleys: List[float]  # Timestamps of energy valleys
    
    # Overlap analysis
    overlap_density: float
    overlap_regions: List[Tuple[float, float, float]]  # (start, end, probability)
    speaker_change_points: List[float]  # Timestamps of likely speaker changes
    
    # Complexity metrics
    complexity_score: float  # Overall complexity (0-1, higher = more complex)
    recommended_chunk_size: float  # Recommended chunk size for this region
    boundary_quality_score: float  # Quality of potential boundaries (0-1)


@dataclass
class ChunkBoundary:
    """Represents a chunk boundary with quality metrics"""
    timestamp: float
    boundary_type: str  # 'pause', 'speaker_turn', 'energy_valley', 'forced'
    quality_score: float  # 0-1, higher is better
    confidence: float  # Confidence in this boundary
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingResult:
    """Result of elastic chunking analysis"""
    boundaries: List[ChunkBoundary]
    chunk_durations: List[float]
    total_chunks: int
    average_chunk_size: float
    quality_metrics: Dict[str, Any]
    processing_time: float
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


class ElasticChunker:
    """
    Intelligent audio chunking system that adapts chunk sizes based on 
    overlap density and speech patterns to minimize boundary errors.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize elastic chunker
        
        Args:
            config: Chunking configuration. Uses defaults if None.
        """
        self.config = config or ChunkingConfig()
        self.logger = create_enhanced_logger("elastic_chunker")
        
        # Initialize overlap detector for density analysis
        overlap_config = OverlapDetectionConfig()
        self.overlap_detector = UnifiedOverlapDetector(overlap_config)
        
        # Performance tracking
        self.processing_stats = {
            'total_chunks_created': 0,
            'total_processing_time': 0.0,
            'boundary_quality_scores': [],
            'chunk_size_distribution': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Threading for parallel analysis
        self._analysis_executor = None
        if self.config.enable_parallel_analysis:
            self._analysis_executor = ThreadPoolExecutor(
                max_workers=self.config.max_analysis_workers,
                thread_name_prefix="elastic_chunker"
            )
        
        self.logger.info("ElasticChunker initialized", 
                        context={
                            'config': {
                                'min_chunk_seconds': self.config.min_chunk_seconds,
                                'max_chunk_seconds': self.config.max_chunk_seconds,
                                'target_chunk_seconds': self.config.target_chunk_seconds,
                                'overlap_threshold': self.config.overlap_threshold
                            },
                            'parallel_analysis': self.config.enable_parallel_analysis
                        })
    
    def __del__(self):
        """Cleanup resources"""
        if self._analysis_executor:
            self._analysis_executor.shutdown(wait=False)
    
    @trace_stage("elastic_chunking")
    def chunk_audio(self, audio_path: str, 
                   segments: Optional[List[Dict[str, Any]]] = None,
                   force_rechunk: bool = False) -> ChunkingResult:
        """
        Perform intelligent chunking of audio based on speech characteristics
        
        Args:
            audio_path: Path to audio file to chunk
            segments: Optional pre-existing segments for analysis
            force_rechunk: Force re-analysis even if cached
            
        Returns:
            ChunkingResult with boundaries and quality metrics
        """
        if not self.config.enabled:
            return self._fallback_to_fixed_chunking(audio_path)
        
        start_time = time.time()
        
        self.logger.info(f"Starting elastic chunking analysis", 
                        context={
                            'audio_path': audio_path,
                            'has_segments': segments is not None,
                            'force_rechunk': force_rechunk
                        })
        
        try:
            # Load and analyze audio
            audio_analysis = self._analyze_audio_characteristics(
                audio_path, segments, force_rechunk
            )
            
            # Determine optimal chunk boundaries
            boundaries = self._determine_optimal_boundaries(audio_analysis)
            
            # Validate and refine boundaries
            refined_boundaries = self._refine_boundaries(boundaries, audio_analysis)
            
            # Calculate metrics
            chunk_durations = self._calculate_chunk_durations(refined_boundaries)
            quality_metrics = self._calculate_quality_metrics(
                refined_boundaries, audio_analysis, chunk_durations
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.processing_stats['total_chunks_created'] += len(refined_boundaries)
            self.processing_stats['total_processing_time'] += processing_time
            self.processing_stats['chunk_size_distribution'].extend(chunk_durations)
            self.processing_stats['boundary_quality_scores'].extend([b.quality_score for b in refined_boundaries])
            
            result = ChunkingResult(
                boundaries=refined_boundaries,
                chunk_durations=chunk_durations,
                total_chunks=len(refined_boundaries),
                average_chunk_size=float(np.mean(chunk_durations)) if chunk_durations else 0.0,
                quality_metrics=quality_metrics,
                processing_time=processing_time,
                analysis_metadata={
                    'audio_duration': audio_analysis[-1].end_time if audio_analysis else 0,
                    'complexity_scores': [a.complexity_score for a in audio_analysis],
                    'overlap_densities': [a.overlap_density for a in audio_analysis]
                }
            )
            
            self.logger.info(f"Elastic chunking completed", 
                            context={
                                'total_chunks': result.total_chunks,
                                'average_chunk_size': result.average_chunk_size,
                                'processing_time': processing_time,
                                'overall_quality': quality_metrics.get('overall_quality_score', 0)
                            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Elastic chunking failed, falling back to fixed chunking: {e}")
            return self._fallback_to_fixed_chunking(audio_path)
    
    @cached_operation("audio_analysis")
    def _analyze_audio_characteristics(self, audio_path: str, 
                                     segments: Optional[List[Dict[str, Any]]] = None,
                                     force_rechunk: bool = False) -> List[SpeechCharacteristics]:
        """
        Analyze audio characteristics for chunking decisions
        
        Args:
            audio_path: Path to audio file
            segments: Optional pre-existing segments
            force_rechunk: Force re-analysis
            
        Returns:
            List of SpeechCharacteristics for different audio regions
        """
        self.logger.info("Analyzing audio characteristics for chunking")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        # Analyze in windows for detailed characteristics
        analysis_window = self.config.complexity_analysis_window
        characteristics = []
        
        current_time = 0.0
        window_idx = 0
        
        while current_time < duration:
            end_time = min(current_time + analysis_window, duration)
            window_idx += 1
            
            # Extract window audio
            start_sample = int(current_time * sr)
            end_sample = int(end_time * sr)
            window_audio = y[start_sample:end_sample]
            
            if len(window_audio) < sr * 0.1:  # Skip very short windows
                break
            
            # Analyze this window
            char = self._analyze_audio_window(
                window_audio, int(sr), current_time, end_time, 
                segments, window_idx
            )
            characteristics.append(char)
            
            current_time = end_time
        
        self.logger.info(f"Completed audio analysis", 
                        context={
                            'total_windows': len(characteristics),
                            'analysis_duration': duration,
                            'average_complexity': np.mean([c.complexity_score for c in characteristics])
                        })
        
        return characteristics
    
    def _analyze_audio_window(self, window_audio: np.ndarray, sr: int,
                            start_time: float, end_time: float,
                            segments: Optional[List[Dict[str, Any]]],
                            window_idx: int) -> SpeechCharacteristics:
        """
        Analyze characteristics of a specific audio window
        
        Args:
            window_audio: Audio samples for this window
            sr: Sample rate
            start_time: Window start time
            end_time: Window end time
            segments: Optional pre-existing segments
            window_idx: Window index for logging
            
        Returns:
            SpeechCharacteristics for this window
        """
        # VAD analysis
        vad_result = self._perform_vad_analysis(window_audio, sr, start_time)
        
        # Energy analysis
        energy_result = self._perform_energy_analysis(window_audio, sr, start_time)
        
        # Overlap analysis (if segments provided)
        overlap_result = self._perform_overlap_analysis(
            segments, start_time, end_time
        ) if segments else self._empty_overlap_result()
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            vad_result, energy_result, overlap_result
        )
        
        # Recommend chunk size based on characteristics
        recommended_size = self._recommend_chunk_size(complexity_score, overlap_result['overlap_density'])
        
        # Calculate boundary quality score
        boundary_quality = self._calculate_boundary_quality_score(
            vad_result, energy_result, overlap_result
        )
        
        return SpeechCharacteristics(
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            voice_activity_ratio=vad_result['voice_activity_ratio'],
            speech_segments=vad_result['speech_segments'],
            silence_segments=vad_result['silence_segments'],
            average_energy=energy_result['average_energy'],
            energy_variance=energy_result['energy_variance'],
            energy_valleys=energy_result['energy_valleys'],
            overlap_density=overlap_result['overlap_density'],
            overlap_regions=overlap_result['overlap_regions'],
            speaker_change_points=overlap_result['speaker_change_points'],
            complexity_score=complexity_score,
            recommended_chunk_size=recommended_size,
            boundary_quality_score=boundary_quality
        )
    
    def _perform_vad_analysis(self, audio: np.ndarray, sr: int, 
                            offset_time: float) -> Dict[str, Any]:
        """
        Perform Voice Activity Detection analysis
        
        Args:
            audio: Audio samples
            sr: Sample rate  
            offset_time: Time offset for absolute timestamps
            
        Returns:
            Dictionary with VAD analysis results
        """
        # Calculate frame parameters
        frame_length = int(self.config.vad_frame_length_ms * sr / 1000)
        hop_length = int(self.config.vad_hop_length_ms * sr / 1000)
        
        # Compute RMS energy
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Compute spectral centroid for voice characteristics
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=hop_length
        )[0]
        
        # Voice activity detection based on energy and spectral features
        energy_threshold = np.max(rms) * self.config.voice_activity_threshold
        spectral_threshold = np.mean(spectral_centroid)
        
        is_voiced = (rms > energy_threshold) & (spectral_centroid > spectral_threshold * 0.5)
        
        # Convert frame indices to time stamps
        frame_times = librosa.frames_to_time(
            np.arange(len(is_voiced)), sr=sr, hop_length=hop_length
        ) + offset_time
        
        # Find speech and silence segments
        speech_segments = []
        silence_segments = []
        
        in_speech = False
        speech_start = None
        silence_start = None
        
        for i, (time_stamp, voiced) in enumerate(zip(frame_times, is_voiced)):
            if voiced and not in_speech:
                # Start of speech
                if silence_start is not None:
                    silence_segments.append((silence_start, time_stamp))
                    silence_start = None
                speech_start = time_stamp
                in_speech = True
            elif not voiced and in_speech:
                # End of speech
                if speech_start is not None:
                    speech_segments.append((speech_start, time_stamp))
                    speech_start = None
                silence_start = time_stamp
                in_speech = False
        
        # Handle final segment
        final_time = frame_times[-1] if len(frame_times) > 0 else offset_time
        if in_speech and speech_start is not None:
            speech_segments.append((speech_start, final_time))
        elif not in_speech and silence_start is not None:
            silence_segments.append((silence_start, final_time))
        
        voice_activity_ratio = np.sum(is_voiced) / len(is_voiced) if len(is_voiced) > 0 else 0
        
        return {
            'voice_activity_ratio': voice_activity_ratio,
            'speech_segments': speech_segments,
            'silence_segments': silence_segments,
            'frame_energy': rms,
            'spectral_centroid': spectral_centroid,
            'is_voiced_frames': is_voiced
        }
    
    def _perform_energy_analysis(self, audio: np.ndarray, sr: int,
                               offset_time: float) -> Dict[str, Any]:
        """
        Perform energy-based analysis for boundary detection
        
        Args:
            audio: Audio samples
            sr: Sample rate
            offset_time: Time offset for absolute timestamps
            
        Returns:
            Dictionary with energy analysis results
        """
        # Calculate short-time energy
        hop_length = int(self.config.vad_hop_length_ms * sr / 1000)
        frame_length = int(self.config.vad_frame_length_ms * sr / 1000)
        
        # RMS energy
        rms_energy = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms_energy, ref=np.max)
        
        # Smooth energy curve
        if len(rms_db) > self.config.energy_smoothing_window:
            smooth_energy = np.convolve(
                rms_db, 
                np.ones(self.config.energy_smoothing_window) / self.config.energy_smoothing_window, 
                mode='same'
            )
        else:
            smooth_energy = rms_db
        
        # Find energy valleys (local minima)
        energy_valleys = []
        if len(smooth_energy) > 3:
            # Find local minima
            from scipy.signal import find_peaks
            
            # Invert signal to find valleys as peaks
            inverted_energy = -smooth_energy
            valley_indices, _ = find_peaks(
                inverted_energy,
                height=-self.config.silence_threshold_db,  # Only consider significant valleys
                distance=int(0.3 * sr / hop_length)  # Minimum 0.3s between valleys
            )
            
            # Convert to timestamps
            frame_times = librosa.frames_to_time(
                valley_indices, sr=sr, hop_length=hop_length
            ) + offset_time
            
            energy_valleys = frame_times.tolist()
        
        # Calculate statistics
        average_energy = np.mean(rms_db)
        energy_variance = np.var(rms_db)
        
        return {
            'average_energy': average_energy,
            'energy_variance': energy_variance,
            'energy_valleys': energy_valleys,
            'rms_energy': rms_energy,
            'rms_db': rms_db,
            'smooth_energy': smooth_energy
        }
    
    def _perform_overlap_analysis(self, segments: List[Dict[str, Any]],
                                start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Perform overlap analysis for the given time window
        
        Args:
            segments: Diarization segments
            start_time: Window start time
            end_time: Window end time
            
        Returns:
            Dictionary with overlap analysis results
        """
        if not segments:
            return self._empty_overlap_result()
        
        # Filter segments to window
        window_segments = [
            seg for seg in segments
            if seg['start'] < end_time and seg['end'] > start_time
        ]
        
        if len(window_segments) < 2:
            return self._empty_overlap_result()
        
        # Use overlap detector
        audio_duration = end_time - start_time
        overlap_result = self.overlap_detector.detect_overlap_frames(
            window_segments, audio_duration
        )
        
        # Calculate overlap density
        total_overlap_duration = sum(
            frame.duration for frame in overlap_result.overlap_frames
        )
        overlap_density = total_overlap_duration / audio_duration if audio_duration > 0 else 0
        
        # Extract overlap regions
        overlap_regions = [
            (frame.start_time, frame.end_time, frame.overlap_probability)
            for frame in overlap_result.overlap_frames
        ]
        
        # Find speaker change points
        speaker_changes = []
        for i in range(len(window_segments) - 1):
            current_seg = window_segments[i]
            next_seg = window_segments[i + 1]
            
            # Check for speaker change
            if (current_seg['speaker_id'] != next_seg['speaker_id'] and 
                next_seg['start'] > current_seg['end'] - 0.1):  # Allow small gap
                speaker_changes.append(next_seg['start'])
        
        return {
            'overlap_density': overlap_density,
            'overlap_regions': overlap_regions,
            'speaker_change_points': speaker_changes,
            'overlap_frames': overlap_result.overlap_frames,
            'total_overlap_duration': total_overlap_duration
        }
    
    def _empty_overlap_result(self) -> Dict[str, Any]:
        """Return empty overlap analysis result"""
        return {
            'overlap_density': 0.0,
            'overlap_regions': [],
            'speaker_change_points': [],
            'overlap_frames': [],
            'total_overlap_duration': 0.0
        }
    
    def _calculate_complexity_score(self, vad_result: Dict[str, Any],
                                  energy_result: Dict[str, Any],
                                  overlap_result: Dict[str, Any]) -> float:
        """
        Calculate complexity score for audio segment (0-1, higher = more complex)
        
        Args:
            vad_result: VAD analysis results
            energy_result: Energy analysis results
            overlap_result: Overlap analysis results
            
        Returns:
            Complexity score between 0 and 1
        """
        # Factor 1: Speech density (more speech = more complex)
        speech_density_score = vad_result['voice_activity_ratio']
        
        # Factor 2: Energy variance (more dynamic = more complex)
        energy_variance_score = min(energy_result['energy_variance'] / 100, 1.0)
        
        # Factor 3: Overlap density (more overlaps = more complex)
        overlap_score = min(overlap_result['overlap_density'] * 2, 1.0)
        
        # Factor 4: Speaker changes (more changes = more complex)
        speaker_change_score = min(len(overlap_result['speaker_change_points']) * 0.1, 1.0)
        
        # Factor 5: Silence fragmentation (more fragments = more complex)
        silence_fragments = len(vad_result['silence_segments'])
        fragmentation_score = min(silence_fragments * 0.05, 1.0)
        
        # Weighted combination
        complexity_score = (
            0.25 * speech_density_score +
            0.15 * energy_variance_score +
            0.30 * overlap_score +
            0.20 * speaker_change_score +
            0.10 * fragmentation_score
        )
        
        return min(complexity_score, 1.0)
    
    def _recommend_chunk_size(self, complexity_score: float, overlap_density: float) -> float:
        """
        Recommend chunk size based on complexity and overlap density
        
        Args:
            complexity_score: Complexity score (0-1)
            overlap_density: Overlap density (0-1+)
            
        Returns:
            Recommended chunk size in seconds
        """
        # Start with target size
        base_size = self.config.target_chunk_seconds
        
        # Adjust based on overlap density
        if overlap_density >= self.config.high_overlap_threshold:
            # High overlap -> shorter chunks
            overlap_factor = 0.5 + (0.3 * (1 - min(overlap_density, 1.0)))
        elif overlap_density <= self.config.low_overlap_threshold:
            # Low overlap -> longer chunks
            overlap_factor = 1.3 + (0.5 * (1 - overlap_density))
        else:
            # Medium overlap -> slight adjustment
            overlap_factor = 1.0 - (0.2 * (overlap_density - self.config.low_overlap_threshold))
        
        # Adjust based on complexity
        if complexity_score >= 0.7:
            # High complexity -> shorter chunks
            complexity_factor = 0.7 + (0.3 * (1 - complexity_score))
        elif complexity_score <= 0.3:
            # Low complexity -> longer chunks  
            complexity_factor = 1.2 + (0.3 * (1 - complexity_score))
        else:
            # Medium complexity -> minimal adjustment
            complexity_factor = 1.0
        
        # Combine factors
        recommended_size = base_size * overlap_factor * complexity_factor
        
        # Clamp to configured bounds
        return max(
            self.config.min_chunk_seconds,
            min(recommended_size, self.config.max_chunk_seconds)
        )
    
    def _calculate_boundary_quality_score(self, vad_result: Dict[str, Any],
                                        energy_result: Dict[str, Any],
                                        overlap_result: Dict[str, Any]) -> float:
        """
        Calculate quality score for potential boundaries in this region (0-1)
        
        Args:
            vad_result: VAD analysis results
            energy_result: Energy analysis results
            overlap_result: Overlap analysis results
            
        Returns:
            Boundary quality score between 0 and 1
        """
        # More silence segments = better boundary opportunities
        silence_score = min(len(vad_result['silence_segments']) * 0.1, 1.0)
        
        # More energy valleys = better boundary opportunities
        valley_score = min(len(energy_result['energy_valleys']) * 0.15, 1.0)
        
        # Fewer overlaps = cleaner boundaries
        overlap_penalty = max(0, 1.0 - overlap_result['overlap_density'])
        
        # More speaker changes = more boundary opportunities
        speaker_change_score = min(len(overlap_result['speaker_change_points']) * 0.2, 1.0)
        
        # Weighted combination
        quality_score = (
            0.30 * silence_score +
            0.25 * valley_score +
            0.25 * overlap_penalty +
            0.20 * speaker_change_score
        )
        
        return min(quality_score, 1.0)
    
    def _determine_optimal_boundaries(self, characteristics: List[SpeechCharacteristics]) -> List[ChunkBoundary]:
        """
        Determine optimal chunk boundaries based on speech characteristics
        
        Args:
            characteristics: List of speech characteristics for audio regions
            
        Returns:
            List of chunk boundaries with quality scores
        """
        if not characteristics:
            return []
        
        boundaries = []
        total_duration = characteristics[-1].end_time
        
        current_pos = 0.0
        chunk_idx = 0
        
        while current_pos < total_duration:
            chunk_idx += 1
            
            # Find the characteristics window containing current position
            current_char = self._find_characteristics_at_time(characteristics, current_pos)
            
            if current_char is None:
                # Fallback: add boundary at target interval
                next_boundary = min(current_pos + self.config.target_chunk_seconds, total_duration)
                boundaries.append(ChunkBoundary(
                    timestamp=next_boundary,
                    boundary_type='forced',
                    quality_score=0.1,
                    confidence=0.1
                ))
                current_pos = next_boundary
                continue
            
            # Get recommended chunk size for this region
            recommended_size = current_char.recommended_chunk_size
            
            # Define search window around target boundary
            target_boundary = current_pos + recommended_size
            search_start = target_boundary - self.config.boundary_search_window
            search_end = min(target_boundary + self.config.boundary_search_window, total_duration)
            
            # Find best boundary in search window
            best_boundary = self._find_best_boundary_in_window(
                characteristics, search_start, search_end, target_boundary
            )
            
            if best_boundary:
                boundaries.append(best_boundary)
                current_pos = best_boundary.timestamp
            else:
                # Fallback to target boundary
                boundaries.append(ChunkBoundary(
                    timestamp=min(target_boundary, total_duration),
                    boundary_type='forced',
                    quality_score=0.2,
                    confidence=0.2
                ))
                current_pos = min(target_boundary, total_duration)
        
        # Remove the last boundary if it's at the end of the audio
        if boundaries and abs(boundaries[-1].timestamp - total_duration) < 0.1:
            boundaries.pop()
        
        self.logger.info(f"Determined {len(boundaries)} optimal boundaries")
        return boundaries
    
    def _find_characteristics_at_time(self, characteristics: List[SpeechCharacteristics],
                                    timestamp: float) -> Optional[SpeechCharacteristics]:
        """Find characteristics window that contains the given timestamp"""
        for char in characteristics:
            if char.start_time <= timestamp < char.end_time:
                return char
        return characteristics[-1] if characteristics else None
    
    def _find_best_boundary_in_window(self, characteristics: List[SpeechCharacteristics],
                                    window_start: float, window_end: float,
                                    target_time: float) -> Optional[ChunkBoundary]:
        """
        Find the best boundary within a search window
        
        Args:
            characteristics: Speech characteristics
            window_start: Start of search window
            window_end: End of search window  
            target_time: Target boundary time
            
        Returns:
            Best boundary or None if no good boundary found
        """
        candidates = []
        
        # Collect all potential boundaries in window
        for char in characteristics:
            if not (char.start_time < window_end and char.end_time > window_start):
                continue
            
            # Add silence boundaries
            for silence_start, silence_end in char.silence_segments:
                if window_start <= silence_start <= window_end:
                    # Prefer middle of silence regions
                    silence_duration = silence_end - silence_start
                    if (self.config.pause_min_duration <= silence_duration <= self.config.pause_max_duration):
                        boundary_time = silence_start + silence_duration / 2
                        distance_penalty = abs(boundary_time - target_time) / self.config.boundary_search_window
                        quality = (1.0 - distance_penalty) * char.boundary_quality_score
                        
                        candidates.append(ChunkBoundary(
                            timestamp=boundary_time,
                            boundary_type='pause',
                            quality_score=quality * 0.9,  # High quality for pauses
                            confidence=0.8,
                            metadata={'silence_duration': silence_duration}
                        ))
            
            # Add energy valley boundaries  
            for valley_time in char.energy_valleys:
                if window_start <= valley_time <= window_end:
                    distance_penalty = abs(valley_time - target_time) / self.config.boundary_search_window
                    quality = (1.0 - distance_penalty) * char.boundary_quality_score * self.config.energy_valley_weight
                    
                    candidates.append(ChunkBoundary(
                        timestamp=valley_time,
                        boundary_type='energy_valley',
                        quality_score=quality * 0.7,
                        confidence=0.6
                    ))
            
            # Add speaker change boundaries
            for change_time in char.speaker_change_points:
                if window_start <= change_time <= window_end:
                    distance_penalty = abs(change_time - target_time) / self.config.boundary_search_window
                    quality = (1.0 - distance_penalty) * char.boundary_quality_score * self.config.speaker_turn_weight
                    
                    candidates.append(ChunkBoundary(
                        timestamp=change_time,
                        boundary_type='speaker_turn',
                        quality_score=quality * 0.85,  # High quality for speaker turns
                        confidence=0.75
                    ))
        
        # Select best candidate
        if not candidates:
            return None
        
        # Sort by quality score
        candidates.sort(key=lambda x: x.quality_score, reverse=True)
        best_candidate = candidates[0]
        
        # Only return if quality is above threshold
        if best_candidate.quality_score > 0.3:
            return best_candidate
        
        return None
    
    def _refine_boundaries(self, boundaries: List[ChunkBoundary],
                         characteristics: List[SpeechCharacteristics]) -> List[ChunkBoundary]:
        """
        Refine boundaries to ensure quality and avoid issues
        
        Args:
            boundaries: Initial boundary list
            characteristics: Speech characteristics
            
        Returns:
            Refined boundary list
        """
        if not boundaries:
            return boundaries
        
        refined = []
        
        for i, boundary in enumerate(boundaries):
            # Check minimum chunk size constraint
            if i > 0:
                prev_boundary = refined[-1] if refined else ChunkBoundary(timestamp=0.0, boundary_type='start', quality_score=1.0, confidence=1.0)
                chunk_duration = boundary.timestamp - prev_boundary.timestamp
                
                if chunk_duration < self.config.min_chunk_seconds:
                    # Merge with next boundary or extend
                    if i < len(boundaries) - 1:
                        # Try to merge with next
                        next_boundary = boundaries[i + 1]
                        merged_time = (boundary.timestamp + next_boundary.timestamp) / 2
                        merged_boundary = ChunkBoundary(
                            timestamp=merged_time,
                            boundary_type='merged',
                            quality_score=min(boundary.quality_score, next_boundary.quality_score),
                            confidence=min(boundary.confidence, next_boundary.confidence),
                            metadata={'merged_from': [boundary.timestamp, next_boundary.timestamp]}
                        )
                        refined.append(merged_boundary)
                        boundaries.pop(i + 1)  # Skip next boundary
                        continue
                    else:
                        # Extend current boundary
                        extended_time = prev_boundary.timestamp + self.config.min_chunk_seconds
                        boundary.timestamp = extended_time
                        boundary.boundary_type = 'extended'
                        boundary.quality_score *= 0.8  # Reduce quality for forced extension
            
            refined.append(boundary)
        
        self.logger.info(f"Refined boundaries: {len(boundaries)} -> {len(refined)}")
        return refined
    
    def _calculate_chunk_durations(self, boundaries: List[ChunkBoundary]) -> List[float]:
        """Calculate duration of each chunk from boundaries"""
        if not boundaries:
            return []
        
        durations = []
        prev_time = 0.0
        
        for boundary in boundaries:
            duration = boundary.timestamp - prev_time
            durations.append(duration)
            prev_time = boundary.timestamp
        
        return durations
    
    def _calculate_quality_metrics(self, boundaries: List[ChunkBoundary],
                                 characteristics: List[SpeechCharacteristics],
                                 chunk_durations: List[float]) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics for the chunking result
        
        Args:
            boundaries: Final boundaries
            characteristics: Speech characteristics
            chunk_durations: Chunk durations
            
        Returns:
            Dictionary with quality metrics
        """
        if not boundaries or not chunk_durations:
            return {'overall_quality_score': 0.0}
        
        # Boundary quality metrics
        avg_boundary_quality = np.mean([b.quality_score for b in boundaries])
        boundary_type_distribution = {}
        for b in boundaries:
            boundary_type_distribution[b.boundary_type] = boundary_type_distribution.get(b.boundary_type, 0) + 1
        
        # Chunk size metrics
        avg_chunk_size = np.mean(chunk_durations)
        chunk_size_variance = np.var(chunk_durations)
        size_deviation_from_target = abs(avg_chunk_size - self.config.target_chunk_seconds) / self.config.target_chunk_seconds
        
        # Adaptive sizing effectiveness
        complexity_scores = [c.complexity_score for c in characteristics]
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0
        
        # Boundary error reduction estimate
        natural_boundaries = len([b for b in boundaries if b.boundary_type in ['pause', 'speaker_turn']])
        natural_boundary_ratio = natural_boundaries / len(boundaries) if boundaries else 0
        
        # Overall quality score
        overall_quality = (
            0.30 * avg_boundary_quality +
            0.25 * (1.0 - size_deviation_from_target) +
            0.20 * natural_boundary_ratio +
            0.15 * (1.0 - min(float(chunk_size_variance) / 100, 1.0)) +
            0.10 * avg_complexity  # Higher complexity handled well = good
        )
        
        return {
            'overall_quality_score': overall_quality,
            'average_boundary_quality': avg_boundary_quality,
            'boundary_type_distribution': boundary_type_distribution,
            'average_chunk_size': avg_chunk_size,
            'chunk_size_variance': chunk_size_variance,
            'size_deviation_from_target': size_deviation_from_target,
            'natural_boundary_ratio': natural_boundary_ratio,
            'average_complexity': avg_complexity,
            'total_boundaries': len(boundaries),
            'natural_boundaries': natural_boundaries
        }
    
    def _fallback_to_fixed_chunking(self, audio_path: str) -> ChunkingResult:
        """
        Fallback to simple fixed-interval chunking when elastic chunking fails
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            ChunkingResult with fixed-size chunks
        """
        self.logger.warning("Falling back to fixed chunking")
        
        try:
            # Get audio duration
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            # Create fixed boundaries
            boundaries = []
            current_time = self.config.target_chunk_seconds
            
            while current_time < duration:
                boundaries.append(ChunkBoundary(
                    timestamp=current_time,
                    boundary_type='fixed',
                    quality_score=0.5,  # Neutral quality for fixed chunks
                    confidence=0.5
                ))
                current_time += self.config.target_chunk_seconds
            
            chunk_durations = self._calculate_chunk_durations(boundaries)
            
            return ChunkingResult(
                boundaries=boundaries,
                chunk_durations=chunk_durations,
                total_chunks=len(boundaries),
                average_chunk_size=self.config.target_chunk_seconds,
                quality_metrics={'overall_quality_score': 0.5, 'fallback_used': True},
                processing_time=0.1,
                analysis_metadata={'fallback_chunking': True}
            )
            
        except Exception as e:
            self.logger.error(f"Even fallback chunking failed: {e}")
            return ChunkingResult(
                boundaries=[],
                chunk_durations=[],
                total_chunks=0,
                average_chunk_size=0,
                quality_metrics={'overall_quality_score': 0.0, 'failed': True},
                processing_time=0.0
            )
    
    def create_audio_chunks(self, audio_path: str, boundaries: List[ChunkBoundary]) -> List[str]:
        """
        Create physical audio chunk files from boundaries
        
        Args:
            audio_path: Original audio file path
            boundaries: Chunk boundaries
            
        Returns:
            List of paths to created chunk files
        """
        if not boundaries:
            return [audio_path]  # Return original file if no chunking needed
        
        self.logger.info(f"Creating {len(boundaries)} audio chunks")
        
        chunk_paths = []
        
        try:
            # Load original audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Create chunks
            start_time = 0.0
            for i, boundary in enumerate(boundaries):
                end_time = boundary.timestamp
                
                # Convert times to sample indices
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Add small overlap for continuity if configured
                if self.config.chunk_overlap_seconds > 0 and i > 0:
                    overlap_samples = int(self.config.chunk_overlap_seconds * sr)
                    start_sample = max(0, start_sample - overlap_samples)
                
                # Extract chunk audio
                chunk_audio = y[start_sample:end_sample]
                
                # Create temporary file for this chunk
                chunk_file = tempfile.NamedTemporaryFile(
                    suffix=f'_elastic_chunk_{i}.wav', delete=False
                )
                chunk_path = chunk_file.name
                chunk_file.close()
                
                # Save chunk audio
                sf.write(chunk_path, chunk_audio, sr)
                chunk_paths.append(chunk_path)
                
                self.logger.debug(f"Created chunk {i}: {start_time:.1f}s - {end_time:.1f}s ({chunk_path})")
                
                start_time = end_time
            
            self.logger.info(f"Successfully created {len(chunk_paths)} audio chunks")
            return chunk_paths
            
        except Exception as e:
            # Clean up any created files on error
            for path in chunk_paths:
                try:
                    import os
                    os.unlink(path)
                except:
                    pass
            raise Exception(f"Failed to create audio chunks: {e}")
    
    def cleanup_chunks(self, chunk_paths: List[str]) -> None:
        """
        Clean up temporary chunk files
        
        Args:
            chunk_paths: List of chunk file paths to delete
        """
        import os
        cleaned_count = 0
        
        for path in chunk_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to clean up chunk file {path}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} temporary chunk files")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the chunker
        
        Returns:
            Dictionary with performance metrics
        """
        stats = dict(self.processing_stats)
        
        if stats['chunk_size_distribution']:
            stats['chunk_size_statistics'] = {
                'mean': np.mean(stats['chunk_size_distribution']),
                'std': np.std(stats['chunk_size_distribution']),
                'min': np.min(stats['chunk_size_distribution']),
                'max': np.max(stats['chunk_size_distribution'])
            }
        
        if stats['boundary_quality_scores']:
            stats['boundary_quality_statistics'] = {
                'mean': np.mean(stats['boundary_quality_scores']),
                'std': np.std(stats['boundary_quality_scores']),
                'min': np.min(stats['boundary_quality_scores']),
                'max': np.max(stats['boundary_quality_scores'])
            }
        
        # Add cache performance
        cache_manager = get_cache_manager()
        if hasattr(cache_manager, 'get_statistics'):
            stats['cache_statistics'] = cache_manager.get_statistics()
        
        return stats


# Factory functions for integration

def create_elastic_chunker(config: Optional[Dict[str, Any]] = None) -> ElasticChunker:
    """
    Factory function to create ElasticChunker instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ElasticChunker instance
    """
    if config:
        chunking_config = ChunkingConfig(**config)
    else:
        chunking_config = ChunkingConfig()
    
    return ElasticChunker(chunking_config)


# Global instance for singleton pattern
_elastic_chunker_instance = None
_elastic_chunker_lock = threading.Lock()

def get_elastic_chunker() -> ElasticChunker:
    """
    Get singleton ElasticChunker instance (thread-safe)
    
    Returns:
        ElasticChunker singleton instance
    """
    global _elastic_chunker_instance
    
    if _elastic_chunker_instance is None:
        with _elastic_chunker_lock:
            if _elastic_chunker_instance is None:
                _elastic_chunker_instance = create_elastic_chunker()
    
    return _elastic_chunker_instance