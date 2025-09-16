"""
Source Separation Engine for Overlap Frames

This module implements selective source separation for overlapping speech segments
using Demucs or Conv-TasNet models. It activates when overlap probability ≥ 0.25
and separates audio into individual speaker stems for improved transcription quality.

Key Features:
- Automatic overlap probability detection and thresholding
- Demucs-based audio source separation
- Multi-stem transcription using existing ASR providers
- Viterbi attribution algorithm for speaker-stem matching
- Seamless integration with existing diarization pipeline

Author: Advanced Ensemble Transcription System
"""

import os
import torch
import torchaudio
import numpy as np
import tempfile
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

from core.asr_providers.base import ASRProvider, ASRResult, DecodeMode
from core.asr_providers.factory import ASRProviderFactory
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.resilient_api import subprocess_retry
from core.circuit_breaker import CircuitBreakerOpenException

# Import Demucs components
try:
    import demucs
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    DEMUCS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Demucs not available: {e}")
    DEMUCS_AVAILABLE = False

@dataclass
class OverlapFrame:
    """Represents a frame with detected speaker overlap"""
    start_time: float
    end_time: float
    duration: float
    overlap_probability: float
    speakers_involved: List[str]
    confidence_score: float
    segment_indices: List[int]  # Which diarization segments are involved
    audio_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SeparatedStem:
    """Represents a separated audio stem for a single speaker"""
    speaker_id: str
    stem_path: str
    confidence: float
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StemTranscription:
    """Transcription result for a separated stem"""
    stem: SeparatedStem
    asr_result: ASRResult
    attribution_confidence: float
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SourceSeparationResult:
    """Complete result from source separation processing"""
    overlap_frame: OverlapFrame
    separated_stems: List[SeparatedStem]
    stem_transcriptions: List[StemTranscription]
    attribution_results: Dict[str, Any]
    final_segments: List[Dict[str, Any]]
    processing_time: float
    separation_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class OverlapDetector:
    """Detects and quantifies speaker overlaps in diarization results"""
    
    def __init__(self, 
                 overlap_threshold: float = 0.25,
                 min_overlap_duration: float = 0.1,
                 confidence_weight: float = 0.3):
        """
        Initialize overlap detector
        
        Args:
            overlap_threshold: Minimum overlap probability to trigger separation
            min_overlap_duration: Minimum overlap duration to consider (seconds)
            confidence_weight: Weight for confidence in overlap probability calculation
        """
        self.overlap_threshold = overlap_threshold
        self.min_overlap_duration = min_overlap_duration
        self.confidence_weight = confidence_weight
        self.logger = create_enhanced_logger("overlap_detector")
    
    def detect_overlap_frames(self, 
                             segments: List[Dict[str, Any]], 
                             audio_duration: float) -> List[OverlapFrame]:
        """
        Detect and quantify overlap frames from diarization segments
        
        Args:
            segments: Diarization segments with start, end, speaker_id, confidence
            audio_duration: Total audio duration for normalization
            
        Returns:
            List of overlap frames meeting the threshold criteria
        """
        overlap_frames = []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        # Find overlapping segments
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
                    if overlap_duration >= self.min_overlap_duration:
                        # Calculate overlap probability
                        overlap_prob = self._calculate_overlap_probability(
                            seg1, seg2, overlap_duration, audio_duration
                        )
                        
                        # Check overlap probability threshold
                        if overlap_prob >= self.overlap_threshold:
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
                                    'overlap_ratio_seg2': overlap_duration / (seg2['end'] - seg2['start'])
                                }
                            )
                            overlap_frames.append(overlap_frame)
        
        self.logger.info(f"Detected {len(overlap_frames)} overlap frames meeting threshold {self.overlap_threshold}")
        return overlap_frames
    
    def _calculate_overlap_probability(self, 
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
        confidence_factor = 1.0 - (1.0 - avg_confidence) * self.confidence_weight
        
        # Duration adjustment (longer overlaps are more likely to be real)
        duration_factor = min(1.0, overlap_duration / 0.5)  # Normalize to 0.5 seconds
        
        # Speaker difference factor (different speakers more likely to overlap)
        speaker_factor = 1.0 if seg1['speaker_id'] != seg2['speaker_id'] else 0.7
        
        # Combined probability
        overlap_prob = base_prob * confidence_factor * duration_factor * speaker_factor
        
        return min(1.0, max(0.0, overlap_prob))

class SourceSeparationEngine:
    """
    Core engine for audio source separation using Demucs models
    """
    
    def __init__(self, 
                 model_name: str = "htdemucs_ft",
                 device: Optional[str] = None,
                 overlap_threshold: float = 0.25,
                 max_stems: int = 4,
                 enable_caching: bool = True,
                 # Performance controls
                 max_concurrent_jobs: int = 2,
                 batch_size: int = 4,
                 preferred_asr_providers: Optional[List[str]] = None,
                 enable_fallback: bool = True,
                 # Budget controls
                 max_separation_ratio: float = 0.1,  # 10% of audio
                 max_separation_duration: float = 300.0,  # 5 minutes absolute max
                 enable_budget_controls: bool = True):
        """
        Initialize source separation engine with performance and budget controls
        
        Args:
            model_name: Demucs model to use (htdemucs_ft, hdemucs_mmi, etc.)
            device: Processing device ('cpu', 'cuda', 'auto')
            overlap_threshold: Minimum overlap probability to trigger separation
            max_stems: Maximum number of stems to separate
            enable_caching: Enable result caching for performance
            
            # Performance controls
            max_concurrent_jobs: Maximum concurrent Demucs separation jobs
            batch_size: Number of overlap frames to process in batch
            preferred_asr_providers: Preferred ASR providers for stems (best 2)
            enable_fallback: Enable fallback to original segments on failures
            
            # Budget controls  
            max_separation_ratio: Maximum ratio of audio to separate (0.1 = 10%)
            max_separation_duration: Absolute maximum duration to separate (seconds)
            enable_budget_controls: Enable budget enforcement
        """
        # Core configuration
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.overlap_threshold = overlap_threshold
        self.max_stems = max_stems
        self.enable_caching = enable_caching
        
        # Performance controls
        self.max_concurrent_jobs = max_concurrent_jobs
        self.batch_size = batch_size
        self.preferred_asr_providers = preferred_asr_providers or ['faster-whisper', 'deepgram']
        self.enable_fallback = enable_fallback
        
        # Budget controls
        self.max_separation_ratio = max_separation_ratio
        self.max_separation_duration = max_separation_duration
        self.enable_budget_controls = enable_budget_controls
        
        # Processing state
        self.total_separated_duration = 0.0
        self.active_jobs = 0
        self.fallback_count = 0
        
        self.logger = create_enhanced_logger("source_separation_engine")
        
        # Initialize components
        self.overlap_detector = OverlapDetector(overlap_threshold=overlap_threshold)
        self.asr_factory = ASRProviderFactory()
        
        # Initialize model
        self.model = None
        self._initialize_model()
        
        # Processing cache and job queue
        self._cache = {} if enable_caching else None
        self._job_semaphore = None
        
        self.logger.info("Source separation engine initialized with controls",
                        context={
                            'max_concurrent_jobs': max_concurrent_jobs,
                            'batch_size': batch_size,
                            'preferred_asr_providers': self.preferred_asr_providers,
                            'max_separation_ratio': max_separation_ratio,
                            'max_separation_duration': max_separation_duration,
                            'budget_controls_enabled': enable_budget_controls
                        })
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup processing device with fallback logic"""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _initialize_model(self) -> None:
        """Initialize Demucs model with error handling"""
        if not DEMUCS_AVAILABLE:
            self.logger.warning("Demucs not available, source separation disabled")
            self.model = None
            return
        
        try:
            from demucs import pretrained
            self.logger.info(f"Loading Demucs model: {self.model_name}")
            self.model = pretrained.get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load Demucs model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if source separation is available"""
        return DEMUCS_AVAILABLE and self.model is not None
    
    def process_audio_with_overlaps(self, 
                                   audio_path: str, 
                                   diarization_segments: List[Dict[str, Any]],
                                   asr_providers: Optional[List[str]] = None) -> List[SourceSeparationResult]:
        """
        Process audio with source separation for detected overlap frames
        
        Args:
            audio_path: Path to input audio file
            diarization_segments: Speaker diarization results
            asr_providers: ASR providers to use for transcription
            
        Returns:
            List of source separation results for each overlap frame
        """
        if not self.is_available():
            self.logger.warning("Source separation not available, skipping")
            return []
        
        # Load audio metadata
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_duration = waveform.shape[1] / sample_rate
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            return []
        
        # Detect overlap frames
        overlap_frames = self.overlap_detector.detect_overlap_frames(
            diarization_segments, audio_duration
        )
        
        if not overlap_frames:
            self.logger.info("No overlap frames detected meeting threshold")
            return []
        
        self.logger.info(f"Processing {len(overlap_frames)} overlap frames")
        
        # Apply budget controls before processing
        if self.enable_budget_controls:
            overlap_frames = self._apply_budget_controls(overlap_frames, audio_duration)
        
        # Process overlap frames with performance controls
        results = self._process_overlap_frames_with_controls(
            overlap_frames, audio_path, waveform, sample_rate, asr_providers
        )
        
        return results
    
    def _process_overlap_frame(self, 
                              overlap_frame: OverlapFrame,
                              audio_path: str,
                              waveform: torch.Tensor,
                              sample_rate: int,
                              asr_providers: Optional[List[str]] = None) -> Optional[SourceSeparationResult]:
        """
        Process a single overlap frame with source separation
        
        Args:
            overlap_frame: Overlap frame to process
            audio_path: Original audio path
            waveform: Audio waveform tensor
            sample_rate: Audio sample rate
            asr_providers: ASR providers to use
            
        Returns:
            Source separation result or None if processing fails
        """
        start_time = overlap_frame.start_time
        end_time = overlap_frame.end_time
        
        self.logger.info(f"Processing overlap frame: {start_time:.3f}s - {end_time:.3f}s")
        
        # Extract overlap segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        overlap_segment = waveform[:, start_sample:end_sample]
        
        # Perform source separation
        separated_stems = self._separate_sources(overlap_segment, sample_rate, overlap_frame)
        
        if not separated_stems:
            self.logger.warning(f"Source separation failed for frame {start_time}-{end_time}")
            return None
        
        # Transcribe each stem
        stem_transcriptions = self._transcribe_stems(separated_stems, asr_providers)
        
        # Perform Viterbi attribution
        attribution_results = self._perform_viterbi_attribution(
            stem_transcriptions, overlap_frame
        )
        
        # Generate final segments
        final_segments = self._generate_final_segments(
            stem_transcriptions, attribution_results, overlap_frame
        )
        
        return SourceSeparationResult(
            overlap_frame=overlap_frame,
            separated_stems=separated_stems,
            stem_transcriptions=stem_transcriptions,
            attribution_results=attribution_results,
            final_segments=final_segments,
            processing_time=0.0,  # TODO: Add timing
            separation_confidence=self._calculate_separation_confidence(separated_stems),
            metadata={
                'model_name': self.model_name,
                'device': self.device,
                'num_stems': len(separated_stems)
            }
        )
    
    def _separate_sources(self, 
                         waveform: torch.Tensor, 
                         sample_rate: int,
                         overlap_frame: OverlapFrame) -> List[SeparatedStem]:
        """
        Separate audio sources using Demucs
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate
            overlap_frame: Overlap frame metadata
            
        Returns:
            List of separated stems
        """
        try:
            # Convert audio format for Demucs
            if waveform.shape[0] == 1:  # Mono to stereo
                waveform = waveform.repeat(2, 1)
            
            # Normalize and convert to model sample rate  
            if self.model is None:
                raise ValueError("Model not available for source separation")
                
            from demucs.audio import convert_audio
            model_sample_rate = getattr(self.model, 'sample_rate', 44100)
            model_channels = getattr(self.model, 'audio_channels', 2)
            
            if sample_rate != model_sample_rate:
                waveform = convert_audio(waveform, sample_rate, model_sample_rate, model_channels)
            
            # Apply source separation
            from demucs.apply import apply_model
            with torch.no_grad():
                sources = apply_model(self.model, waveform.unsqueeze(0).to(self.device), device=self.device)[0]
            
            # Create separated stems
            separated_stems = []
            
            # Get source names from model
            source_names = getattr(self.model, 'sources', ['vocals', 'drums', 'bass', 'other'])
            
            for i, source_name in enumerate(source_names[:self.max_stems]):
                if i < sources.shape[0]:
                    # Save stem to temporary file
                    stem_path = self._save_stem_to_file(
                        sources[i], int(model_sample_rate), f"{source_name}_{overlap_frame.start_time:.3f}"
                    )
                    
                    if stem_path:
                        # Calculate stem confidence based on energy
                        stem_confidence = self._calculate_stem_confidence(sources[i])
                        
                        separated_stem = SeparatedStem(
                            speaker_id=f"stem_{i}_{source_name}",
                            stem_path=stem_path,
                            confidence=stem_confidence,
                            processing_metadata={
                                'source_name': source_name,
                                'source_index': i,
                                'model_sample_rate': model_sample_rate,
                                'original_sample_rate': sample_rate
                            }
                        )
                        separated_stems.append(separated_stem)
            
            return separated_stems
            
        except Exception as e:
            self.logger.error(f"Source separation failed: {e}")
            return []
    
    def _save_stem_to_file(self, 
                          stem_tensor: torch.Tensor, 
                          sample_rate: int, 
                          stem_name: str) -> Optional[str]:
        """Save separated stem to temporary file"""
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f"_{stem_name}.wav", 
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # Save audio
            torchaudio.save(temp_path, stem_tensor.cpu(), sample_rate)
            
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Failed to save stem {stem_name}: {e}")
            return None
    
    def _calculate_stem_confidence(self, stem_tensor: torch.Tensor) -> float:
        """Calculate confidence score for a separated stem based on energy"""
        try:
            # Calculate RMS energy
            rms_energy = torch.sqrt(torch.mean(stem_tensor ** 2))
            
            # Normalize to 0-1 range (heuristic)
            confidence = min(1.0, float(rms_energy) * 10.0)
            
            return confidence
            
        except Exception:
            return 0.5  # Default confidence
    
    def _transcribe_stems(self, 
                         stems: List[SeparatedStem], 
                         asr_providers: Optional[List[str]] = None) -> List[StemTranscription]:
        """
        Transcribe separated stems using multiple ASR providers
        
        Args:
            stems: List of separated stems
            asr_providers: ASR providers to use
            
        Returns:
            List of stem transcriptions
        """
        if asr_providers is None:
            asr_providers = ['faster-whisper', 'deepgram', 'openai']
        
        transcriptions = []
        
        for stem in stems:
            # Use the highest quality ASR provider available
            best_transcription = None
            best_confidence = 0.0
            
            for provider_name in asr_providers:
                try:
                    provider = self.asr_factory.create_provider(provider_name)
                    
                    if provider.is_available():
                        asr_result = provider.transcribe(
                            stem.stem_path,
                            decode_mode=DecodeMode.CAREFUL,
                            language="en"
                        )
                        
                        # Select best transcription based on confidence
                        if asr_result.calibrated_confidence > best_confidence:
                            best_confidence = asr_result.calibrated_confidence
                            best_transcription = asr_result
                            
                except Exception as e:
                    self.logger.warning(f"ASR provider {provider_name} failed for stem {stem.speaker_id}: {e}")
            
            if best_transcription:
                stem_transcription = StemTranscription(
                    stem=stem,
                    asr_result=best_transcription,
                    attribution_confidence=0.0,  # Will be set by Viterbi
                    processing_metadata={
                        'best_provider': best_transcription.provider,
                        'providers_tried': asr_providers
                    }
                )
                transcriptions.append(stem_transcription)
        
        return transcriptions
    
    def _perform_viterbi_attribution(self, 
                                   stem_transcriptions: List[StemTranscription],
                                   overlap_frame: OverlapFrame) -> Dict[str, Any]:
        """
        Perform Viterbi attribution to match stems to speakers using dynamic programming
        
        This algorithm finds the optimal assignment of separated stems to speaker identities
        by considering ASR confidence, audio energy, consistency, and prior probabilities.
        
        Args:
            stem_transcriptions: Transcribed stems with ASR results
            overlap_frame: Original overlap frame with speaker information
            
        Returns:
            Attribution results with optimal speaker-stem mappings
        """
        if not stem_transcriptions or not overlap_frame.speakers_involved:
            return self._fallback_attribution(stem_transcriptions, overlap_frame)
        
        speakers = overlap_frame.speakers_involved
        n_stems = len(stem_transcriptions)
        n_speakers = len(speakers)
        
        # If we have more stems than speakers, use only the best stems
        if n_stems > n_speakers:
            stem_transcriptions = sorted(
                stem_transcriptions,
                key=lambda x: x.asr_result.calibrated_confidence,
                reverse=True
            )[:n_speakers]
            n_stems = len(stem_transcriptions)
        
        # Calculate emission probabilities (how likely each stem belongs to each speaker)
        emission_probs = self._calculate_emission_probabilities(
            stem_transcriptions, speakers, overlap_frame
        )
        
        # Calculate transition probabilities (speaker consistency)
        transition_probs = self._calculate_transition_probabilities(
            speakers, overlap_frame
        )
        
        # Apply Viterbi algorithm for optimal assignment
        optimal_assignment, path_probability = self._viterbi_decode(
            emission_probs, transition_probs, speakers
        )
        
        # Create attribution results
        attribution_map = {}
        attribution_scores = {}
        
        for i, transcription in enumerate(stem_transcriptions):
            if i < len(optimal_assignment):
                assigned_speaker = optimal_assignment[i]
                attribution_map[transcription.stem.speaker_id] = assigned_speaker
                
                # Calculate attribution confidence
                confidence = emission_probs[i][speakers.index(assigned_speaker)]
                attribution_scores[transcription.stem.speaker_id] = confidence
                transcription.attribution_confidence = confidence
        
        return {
            'attribution_map': attribution_map,
            'attribution_scores': attribution_scores,
            'algorithm': 'viterbi_dynamic_programming',
            'total_confidence': path_probability,
            'emission_probabilities': emission_probs,
            'optimal_path_probability': path_probability,
            'num_stems_attributed': len(attribution_map),
            'num_speakers_involved': n_speakers
        }
    
    def _generate_final_segments(self, 
                               stem_transcriptions: List[StemTranscription],
                               attribution_results: Dict[str, Any],
                               overlap_frame: OverlapFrame) -> List[Dict[str, Any]]:
        """
        Generate final transcription segments from attributed stems with proper timeline integration
        
        This method creates segments that are designed to replace the overlapped interval
        in the original diarization timeline, ensuring proper temporal alignment and
        speaker attribution consistency.
        
        Args:
            stem_transcriptions: Transcribed and attributed stems
            attribution_results: Viterbi attribution results
            overlap_frame: Original overlap frame information
            
        Returns:
            List of final segments ready for timeline integration
        """
        if not stem_transcriptions or not attribution_results.get('attribution_map'):
            self.logger.warning(f"No valid transcriptions or attributions for overlap frame {overlap_frame.start_time}-{overlap_frame.end_time}")
            return self._create_fallback_segments(overlap_frame)
        
        attribution_map = attribution_results.get('attribution_map', {})
        
        # Group stem segments by attributed speaker
        speaker_segments = {}
        for transcription in stem_transcriptions:
            stem_id = transcription.stem.speaker_id
            attributed_speaker = attribution_map.get(stem_id)
            
            if attributed_speaker and transcription.asr_result.segments:
                if attributed_speaker not in speaker_segments:
                    speaker_segments[attributed_speaker] = []
                
                for segment in transcription.asr_result.segments:
                    # Adjust timestamps to global timeline
                    adjusted_segment = {
                        'start': overlap_frame.start_time + segment.start,
                        'end': overlap_frame.start_time + segment.end,
                        'speaker_id': attributed_speaker,
                        'text': segment.text.strip(),
                        'confidence': segment.confidence,
                        'source_separated': True,
                        'separation_confidence': transcription.stem.confidence,
                        'attribution_confidence': transcription.attribution_confidence,
                        'original_overlap_prob': overlap_frame.overlap_probability,
                        'overlap_frame_bounds': {
                            'start': overlap_frame.start_time,
                            'end': overlap_frame.end_time
                        },
                        'processing_metadata': {
                            'stem_id': stem_id,
                            'asr_provider': transcription.asr_result.provider,
                            'decode_mode': transcription.asr_result.decode_mode.value if hasattr(transcription.asr_result.decode_mode, 'value') else str(transcription.asr_result.decode_mode),
                            'attribution_algorithm': attribution_results.get('algorithm', 'unknown'),
                            'viterbi_path_confidence': attribution_results.get('total_confidence', 0.0)
                        }
                    }
                    
                    # Ensure segment stays within overlap frame bounds
                    adjusted_segment['start'] = max(adjusted_segment['start'], overlap_frame.start_time)
                    adjusted_segment['end'] = min(adjusted_segment['end'], overlap_frame.end_time)
                    
                    # Skip segments that are too short after adjustment
                    if adjusted_segment['end'] - adjusted_segment['start'] >= 0.05:  # 50ms minimum
                        speaker_segments[attributed_speaker].append(adjusted_segment)
        
        # Merge and optimize segments per speaker
        final_segments = []
        for speaker_id, segments in speaker_segments.items():
            if segments:
                # Sort segments by start time
                segments.sort(key=lambda x: x['start'])
                
                # Merge overlapping or adjacent segments from same speaker
                merged_segments = self._merge_speaker_segments(segments, speaker_id)
                final_segments.extend(merged_segments)
        
        # Sort all final segments by start time
        final_segments.sort(key=lambda x: x['start'])
        
        # Validate temporal consistency
        final_segments = self._validate_temporal_consistency(final_segments, overlap_frame)
        
        # Add timeline replacement metadata
        for segment in final_segments:
            segment['timeline_replacement'] = {
                'replaces_overlap': True,
                'original_bounds': {
                    'start': overlap_frame.start_time,
                    'end': overlap_frame.end_time
                },
                'original_speakers': overlap_frame.speakers_involved,
                'replacement_confidence': (
                    segment['confidence'] * 0.4 +
                    segment['separation_confidence'] * 0.3 +
                    segment['attribution_confidence'] * 0.3
                )
            }
        
        self.logger.info(f"Generated {len(final_segments)} final segments for overlap frame {overlap_frame.start_time:.3f}-{overlap_frame.end_time:.3f}s")
        return final_segments
    
    def _merge_speaker_segments(self, segments: List[Dict[str, Any]], speaker_id: str) -> List[Dict[str, Any]]:
        """
        Merge overlapping or adjacent segments from the same speaker
        
        Args:
            segments: Sorted list of segments from same speaker
            speaker_id: Speaker identifier
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for next_segment in segments[1:]:
            # Check if segments can be merged (overlap or small gap)
            gap = next_segment['start'] - current['end']
            
            if gap <= 0.1:  # 100ms merge threshold
                # Merge segments
                current['end'] = max(current['end'], next_segment['end'])
                
                # Combine text with appropriate spacing
                if current['text'] and next_segment['text']:
                    current['text'] = f"{current['text']} {next_segment['text']}"
                elif next_segment['text']:
                    current['text'] = next_segment['text']
                
                # Update confidence as weighted average
                current_duration = current['end'] - current['start']
                next_duration = next_segment['end'] - next_segment['start']
                total_duration = current_duration + next_duration
                
                if total_duration > 0:
                    current['confidence'] = (
                        current['confidence'] * current_duration +
                        next_segment['confidence'] * next_duration
                    ) / total_duration
                
                # Update other confidence scores similarly
                current['separation_confidence'] = (
                    current['separation_confidence'] * current_duration +
                    next_segment['separation_confidence'] * next_duration
                ) / total_duration
                
                current['attribution_confidence'] = (
                    current['attribution_confidence'] * current_duration +
                    next_segment['attribution_confidence'] * next_duration
                ) / total_duration
                
            else:
                # Cannot merge, add current and start new
                merged.append(current)
                current = next_segment.copy()
        
        # Add the last segment
        merged.append(current)
        
        return merged
    
    def _validate_temporal_consistency(self, segments: List[Dict[str, Any]], 
                                     overlap_frame: OverlapFrame) -> List[Dict[str, Any]]:
        """
        Validate and fix temporal consistency issues in final segments
        
        Args:
            segments: List of final segments
            overlap_frame: Original overlap frame
            
        Returns:
            Validated and corrected segments
        """
        if not segments:
            return segments
        
        validated = []
        
        for segment in segments:
            # Ensure segment is within overlap frame bounds
            segment['start'] = max(segment['start'], overlap_frame.start_time)
            segment['end'] = min(segment['end'], overlap_frame.end_time)
            
            # Skip segments that became invalid
            if segment['end'] <= segment['start']:
                continue
            
            # Ensure minimum duration
            if segment['end'] - segment['start'] < 0.05:  # 50ms minimum
                continue
            
            # Ensure text is not empty
            if not segment.get('text', '').strip():
                continue
            
            validated.append(segment)
        
        # Resolve any remaining overlaps between different speakers
        if len(validated) > 1:
            validated = self._resolve_inter_speaker_overlaps(validated)
        
        return validated
    
    def _resolve_inter_speaker_overlaps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve overlaps between segments from different speakers
        
        Args:
            segments: List of segments sorted by start time
            
        Returns:
            Segments with resolved overlaps
        """
        if len(segments) <= 1:
            return segments
        
        resolved = [segments[0]]
        
        for current_segment in segments[1:]:
            previous_segment = resolved[-1]
            
            # Check for overlap with previous segment
            if (current_segment['start'] < previous_segment['end'] and 
                current_segment['speaker_id'] != previous_segment['speaker_id']):
                
                # Resolve overlap by splitting at midpoint
                overlap_midpoint = (current_segment['start'] + previous_segment['end']) / 2
                
                # Adjust previous segment end
                previous_segment['end'] = overlap_midpoint
                
                # Adjust current segment start
                current_segment['start'] = overlap_midpoint
                
                # Ensure segments remain valid
                if previous_segment['end'] <= previous_segment['start']:
                    resolved.pop()  # Remove invalid previous segment
                
                if current_segment['end'] <= current_segment['start']:
                    continue  # Skip invalid current segment
            
            resolved.append(current_segment)
        
        return resolved
    
    def _create_fallback_segments(self, overlap_frame: OverlapFrame) -> List[Dict[str, Any]]:
        """
        Create fallback segments when source separation fails
        
        Args:
            overlap_frame: Original overlap frame
            
        Returns:
            Fallback segments preserving original overlap
        """
        # Create a single segment marking the overlap as unresolved
        fallback_segment = {
            'start': overlap_frame.start_time,
            'end': overlap_frame.end_time,
            'speaker_id': overlap_frame.speakers_involved[0] if overlap_frame.speakers_involved else 'unknown',
            'text': '[OVERLAP - Source separation failed]',
            'confidence': 0.1,  # Very low confidence
            'source_separated': False,
            'separation_confidence': 0.0,
            'attribution_confidence': 0.0,
            'original_overlap_prob': overlap_frame.overlap_probability,
            'overlap_frame_bounds': {
                'start': overlap_frame.start_time,
                'end': overlap_frame.end_time
            },
            'processing_metadata': {
                'fallback_reason': 'source_separation_failed',
                'original_speakers': overlap_frame.speakers_involved
            },
            'timeline_replacement': {
                'replaces_overlap': True,
                'original_bounds': {
                    'start': overlap_frame.start_time,
                    'end': overlap_frame.end_time
                },
                'original_speakers': overlap_frame.speakers_involved,
                'replacement_confidence': 0.1
            }
        }
        
        return [fallback_segment]
    
    def _apply_budget_controls(self, 
                             overlap_frames: List[OverlapFrame], 
                             audio_duration: float) -> List[OverlapFrame]:
        """
        Apply budget controls to limit source separation processing
        
        Args:
            overlap_frames: Detected overlap frames
            audio_duration: Total audio duration
            
        Returns:
            Filtered overlap frames within budget constraints
        """
        if not self.enable_budget_controls or not overlap_frames:
            return overlap_frames
        
        # Calculate budget limits
        ratio_limit = audio_duration * self.max_separation_ratio
        duration_limit = min(ratio_limit, self.max_separation_duration)
        remaining_budget = duration_limit - self.total_separated_duration
        
        self.logger.info(f"Applying budget controls",
                        context={
                            'audio_duration': audio_duration,
                            'ratio_limit': ratio_limit,
                            'duration_limit': duration_limit,
                            'already_separated': self.total_separated_duration,
                            'remaining_budget': remaining_budget,
                            'overlap_frames_candidate': len(overlap_frames)
                        })
        
        if remaining_budget <= 0:
            self.logger.warning("Source separation budget exhausted - skipping all overlap frames")
            return []
        
        # Sort overlap frames by priority (highest probability first)
        prioritized_frames = sorted(
            overlap_frames,
            key=lambda f: f.overlap_probability,
            reverse=True
        )
        
        # Select frames within budget
        selected_frames = []
        accumulated_duration = 0.0
        
        for frame in prioritized_frames:
            if accumulated_duration + frame.duration <= remaining_budget:
                selected_frames.append(frame)
                accumulated_duration += frame.duration
            else:
                self.logger.debug(f"Skipping overlap frame {frame.start_time:.3f}-{frame.end_time:.3f} - would exceed budget")
        
        budget_utilization = accumulated_duration / duration_limit if duration_limit > 0 else 0
        
        self.logger.info(f"Budget control applied",
                        context={
                            'frames_selected': len(selected_frames),
                            'frames_rejected': len(overlap_frames) - len(selected_frames),
                            'selected_duration': accumulated_duration,
                            'budget_utilization': budget_utilization
                        })
        
        return selected_frames
    
    def _process_overlap_frames_with_controls(self,
                                            overlap_frames: List[OverlapFrame],
                                            audio_path: str,
                                            waveform: torch.Tensor,
                                            sample_rate: int,
                                            asr_providers: Optional[List[str]] = None) -> List[SourceSeparationResult]:
        """
        Process overlap frames with performance controls (batching, concurrency limits)
        
        Args:
            overlap_frames: Overlap frames to process
            audio_path: Audio file path
            waveform: Audio waveform tensor
            sample_rate: Audio sample rate
            asr_providers: ASR providers to use
            
        Returns:
            List of source separation results
        """
        if not overlap_frames:
            return []
        
        # Initialize job control
        if self._job_semaphore is None:
            import threading
            self._job_semaphore = threading.Semaphore(self.max_concurrent_jobs)
        
        # Apply ASR provider restrictions
        effective_asr_providers = self._get_effective_asr_providers(asr_providers)
        
        results = []
        fallback_count = 0
        
        # Process frames in batches
        for i in range(0, len(overlap_frames), self.batch_size):
            batch = overlap_frames[i:i + self.batch_size]
            
            self.logger.info(f"Processing batch {i//self.batch_size + 1} with {len(batch)} overlap frames")
            
            batch_results = []
            
            for frame in batch:
                try:
                    # Acquire job semaphore (limits concurrent processing)
                    with self._job_semaphore:
                        self.active_jobs += 1
                        
                        try:
                            result = self._process_overlap_frame(
                                frame, audio_path, waveform, sample_rate, effective_asr_providers
                            )
                            
                            if result and result.final_segments:
                                batch_results.append(result)
                                # Update budget tracking
                                self.total_separated_duration += frame.duration
                            else:
                                # Apply fallback
                                if self.enable_fallback:
                                    fallback_result = self._create_fallback_result(frame)
                                    batch_results.append(fallback_result)
                                    fallback_count += 1
                        
                        finally:
                            self.active_jobs -= 1
                
                except Exception as e:
                    self.logger.error(f"Failed to process overlap frame {frame.start_time}-{frame.end_time}: {e}")
                    
                    # Apply fallback on error
                    if self.enable_fallback:
                        try:
                            fallback_result = self._create_fallback_result(frame)
                            batch_results.append(fallback_result)
                            fallback_count += 1
                        except Exception as fallback_error:
                            self.logger.error(f"Fallback also failed: {fallback_error}")
            
            results.extend(batch_results)
        
        self.fallback_count += fallback_count
        
        self.logger.info(f"Overlap frame processing completed",
                        context={
                            'total_frames_processed': len(overlap_frames),
                            'successful_separations': len(results) - fallback_count,
                            'fallback_applied': fallback_count,
                            'total_separated_duration': self.total_separated_duration,
                            'asr_providers_used': effective_asr_providers
                        })
        
        return results
    
    def _get_effective_asr_providers(self, asr_providers: Optional[List[str]]) -> List[str]:
        """
        Get effective ASR providers applying performance restrictions
        
        Args:
            asr_providers: Requested ASR providers
            
        Returns:
            Filtered list of ASR providers (limited to best 2)
        """
        if asr_providers is None:
            asr_providers = self.preferred_asr_providers
        
        # Ensure we use only our preferred providers and limit to 2
        effective_providers = []
        
        for provider in self.preferred_asr_providers:
            if provider in asr_providers:
                effective_providers.append(provider)
                if len(effective_providers) >= 2:  # Limit to best 2
                    break
        
        # If no preferred providers found, use first 2 from requested
        if not effective_providers and asr_providers:
            effective_providers = asr_providers[:2]
        
        # Fallback to defaults if still empty
        if not effective_providers:
            effective_providers = ['faster-whisper', 'deepgram']
        
        return effective_providers
    
    def _create_fallback_result(self, overlap_frame: OverlapFrame) -> SourceSeparationResult:
        """
        Create fallback result when source separation fails
        
        Args:
            overlap_frame: The overlap frame that failed
            
        Returns:
            Fallback source separation result
        """
        # Create basic fallback segments preserving original overlap
        fallback_segments = [{
            'start': overlap_frame.start_time,
            'end': overlap_frame.end_time,
            'speaker_id': overlap_frame.speakers_involved[0] if overlap_frame.speakers_involved else 'unknown',
            'text': '[OVERLAP - Fallback applied]',
            'confidence': 0.1,
            'source_separated': False,
            'separation_confidence': 0.0,
            'attribution_confidence': 0.0,
            'original_overlap_prob': overlap_frame.overlap_probability,
            'processing_metadata': {
                'fallback_reason': 'source_separation_failed',
                'original_speakers': overlap_frame.speakers_involved
            },
            'timeline_replacement': {
                'replaces_overlap': True,
                'original_bounds': {
                    'start': overlap_frame.start_time,
                    'end': overlap_frame.end_time
                },
                'original_speakers': overlap_frame.speakers_involved,
                'replacement_confidence': 0.1
            }
        }]
        
        return SourceSeparationResult(
            overlap_frame=overlap_frame,
            separated_stems=[],
            stem_transcriptions=[],
            attribution_results={'algorithm': 'fallback', 'total_confidence': 0.0},
            final_segments=fallback_segments,
            processing_time=0.001,  # Minimal processing time
            separation_confidence=0.0,
            metadata={
                'fallback_applied': True,
                'fallback_reason': 'processing_failure'
            }
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance and budget statistics"""
        return {
            'total_separated_duration': self.total_separated_duration,
            'active_jobs': self.active_jobs,
            'fallback_count': self.fallback_count,
            'budget_utilization': (
                self.total_separated_duration / self.max_separation_duration 
                if self.max_separation_duration > 0 else 0
            ),
            'preferred_asr_providers': self.preferred_asr_providers,
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'batch_size': self.batch_size,
            'budget_controls_enabled': self.enable_budget_controls
        }
    
    def reset_performance_counters(self):
        """Reset performance counters for new processing session"""
        self.total_separated_duration = 0.0
        self.active_jobs = 0
        self.fallback_count = 0
    
    def _calculate_emission_probabilities(self,
                                        stem_transcriptions: List[StemTranscription],
                                        speakers: List[str],
                                        overlap_frame: OverlapFrame) -> List[List[float]]:
        """
        Calculate emission probabilities for Viterbi algorithm
        
        Args:
            stem_transcriptions: List of transcribed stems
            speakers: List of speaker IDs involved in overlap
            overlap_frame: Overlap frame information
            
        Returns:
            Matrix of emission probabilities [stem_index][speaker_index]
        """
        emission_probs = []
        
        for transcription in stem_transcriptions:
            stem_probs = []
            
            for speaker in speakers:
                # Base probability from ASR confidence
                asr_confidence = transcription.asr_result.calibrated_confidence
                
                # Audio quality factor from stem separation
                audio_quality = transcription.stem.confidence
                
                # Text content quality (length, linguistic features)
                text_quality = self._assess_text_quality(transcription.asr_result.full_text)
                
                # Speaker consistency (if we have prior segments for this speaker)
                consistency_bonus = self._calculate_speaker_consistency(
                    transcription, speaker, overlap_frame
                )
                
                # Combine factors with weights
                emission_prob = (
                    asr_confidence * 0.4 +  # ASR confidence is primary
                    audio_quality * 0.3 +   # Audio separation quality
                    text_quality * 0.2 +    # Text linguistic quality
                    consistency_bonus * 0.1  # Speaker consistency bonus
                )
                
                # Ensure probability is in valid range
                emission_prob = max(0.01, min(0.99, emission_prob))
                stem_probs.append(emission_prob)
            
            # Normalize probabilities to sum to 1
            prob_sum = sum(stem_probs)
            if prob_sum > 0:
                stem_probs = [p / prob_sum for p in stem_probs]
            else:
                # Uniform distribution as fallback
                stem_probs = [1.0 / len(speakers)] * len(speakers)
            
            emission_probs.append(stem_probs)
        
        return emission_probs
    
    def _calculate_transition_probabilities(self,
                                          speakers: List[str],
                                          overlap_frame: OverlapFrame) -> List[List[float]]:
        """
        Calculate transition probabilities between speaker assignments
        
        Args:
            speakers: List of speaker IDs
            overlap_frame: Overlap frame information
            
        Returns:
            Transition probability matrix [from_speaker][to_speaker]
        """
        n_speakers = len(speakers)
        
        # Initialize with uniform distribution but bias toward different speakers
        # (overlaps are more likely between different speakers)
        transition_probs = []
        
        for i in range(n_speakers):
            row = []
            for j in range(n_speakers):
                if i == j:
                    # Same speaker continuation (less likely in overlaps)
                    prob = 0.2
                else:
                    # Different speaker (more likely in overlaps)
                    prob = 0.8 / (n_speakers - 1) if n_speakers > 1 else 0.8
                row.append(prob)
            transition_probs.append(row)
        
        return transition_probs
    
    def _viterbi_decode(self,
                       emission_probs: List[List[float]],
                       transition_probs: List[List[float]],
                       speakers: List[str]) -> Tuple[List[str], float]:
        """
        Apply Viterbi algorithm for optimal speaker assignment
        
        Args:
            emission_probs: Emission probability matrix
            transition_probs: Transition probability matrix
            speakers: List of speaker IDs
            
        Returns:
            Tuple of (optimal_assignment, path_probability)
        """
        if not emission_probs or not speakers:
            return [], 0.0
        
        n_steps = len(emission_probs)
        n_states = len(speakers)
        
        # Initialize Viterbi tables
        viterbi_prob = np.zeros((n_steps, n_states))
        viterbi_path = np.zeros((n_steps, n_states), dtype=int)
        
        # Initialize first step
        for s in range(n_states):
            viterbi_prob[0][s] = emission_probs[0][s]
            viterbi_path[0][s] = s
        
        # Forward pass
        for t in range(1, n_steps):
            for s in range(n_states):
                # Find best previous state
                best_prob = 0.0
                best_prev = 0
                
                for prev_s in range(n_states):
                    prob = (viterbi_prob[t-1][prev_s] * 
                           transition_probs[prev_s][s] * 
                           emission_probs[t][s])
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = prev_s
                
                viterbi_prob[t][s] = best_prob
                viterbi_path[t][s] = best_prev
        
        # Backward pass - find optimal path
        optimal_assignment = []
        
        # Find best final state
        best_final_prob = np.max(viterbi_prob[n_steps-1])
        best_final_state = np.argmax(viterbi_prob[n_steps-1])
        
        # Trace back optimal path
        current_state = best_final_state
        for t in range(n_steps-1, -1, -1):
            optimal_assignment.insert(0, speakers[current_state])
            if t > 0:
                current_state = viterbi_path[t][current_state]
        
        return optimal_assignment, float(best_final_prob)
    
    def _assess_text_quality(self, text: str) -> float:
        """
        Assess the linguistic quality of transcribed text
        
        Args:
            text: Transcribed text
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.1
        
        # Basic quality indicators
        word_count = len(text.split())
        char_count = len(text.strip())
        
        # Length factor (reasonable length is better)
        length_factor = min(1.0, word_count / 5.0)  # Normalize to 5 words
        
        # Character diversity (more diverse is usually better)
        unique_chars = len(set(text.lower()))
        diversity_factor = min(1.0, unique_chars / 10.0)  # Normalize to 10 unique chars
        
        # Avoid empty or single-character results
        if char_count < 2:
            return 0.2
        
        # Combine factors
        quality = (length_factor * 0.6 + diversity_factor * 0.4)
        return max(0.1, min(1.0, quality))
    
    def _calculate_speaker_consistency(self,
                                     transcription: StemTranscription,
                                     speaker: str,
                                     overlap_frame: OverlapFrame) -> float:
        """
        Calculate bonus for speaker consistency with prior segments
        
        Args:
            transcription: Current stem transcription
            speaker: Candidate speaker ID
            overlap_frame: Overlap frame information
            
        Returns:
            Consistency bonus between 0.0 and 1.0
        """
        # For now, return base consistency
        # This could be enhanced with speaker embedding similarity
        # or acoustic feature consistency in future versions
        
        # Give slight bonus if speaker is primary in overlap
        if speaker in overlap_frame.speakers_involved:
            speaker_index = overlap_frame.speakers_involved.index(speaker)
            if speaker_index == 0:  # Primary speaker
                return 0.8
            else:
                return 0.6
        
        return 0.5  # Neutral
    
    def _fallback_attribution(self,
                             stem_transcriptions: List[StemTranscription],
                             overlap_frame: OverlapFrame) -> Dict[str, Any]:
        """
        Fallback attribution method when Viterbi fails
        
        Args:
            stem_transcriptions: Stem transcriptions
            overlap_frame: Overlap frame
            
        Returns:
            Simple attribution results
        """
        attribution_map = {}
        attribution_scores = {}
        
        if not stem_transcriptions:
            return {
                'attribution_map': attribution_map,
                'attribution_scores': attribution_scores,
                'algorithm': 'fallback_empty',
                'total_confidence': 0.0
            }
        
        # Simple confidence-based assignment
        speakers = overlap_frame.speakers_involved if overlap_frame.speakers_involved else ['unknown']
        
        for i, transcription in enumerate(stem_transcriptions):
            speaker = speakers[i % len(speakers)]  # Round-robin assignment
            confidence = transcription.asr_result.calibrated_confidence * 0.5  # Reduced confidence for fallback
            
            attribution_map[transcription.stem.speaker_id] = speaker
            attribution_scores[transcription.stem.speaker_id] = confidence
            transcription.attribution_confidence = confidence
        
        return {
            'attribution_map': attribution_map,
            'attribution_scores': attribution_scores,
            'algorithm': 'fallback_round_robin',
            'total_confidence': sum(attribution_scores.values()) / len(attribution_scores) if attribution_scores else 0.0
        }
    
    def _calculate_separation_confidence(self, stems: List[SeparatedStem]) -> float:
        """Calculate overall confidence in source separation quality"""
        if not stems:
            return 0.0
        
        # Average stem confidences weighted by energy
        total_confidence = sum(stem.confidence for stem in stems)
        avg_confidence = total_confidence / len(stems)
        
        # Penalty for too few or too many stems
        stem_count_penalty = 1.0
        if len(stems) < 2:
            stem_count_penalty = 0.7
        elif len(stems) > 3:
            stem_count_penalty = 0.9
        
        return avg_confidence * stem_count_penalty
    
    def cleanup_temp_files(self, results: List[SourceSeparationResult]) -> None:
        """Clean up temporary stem files"""
        for result in results:
            for stem in result.separated_stems:
                try:
                    if os.path.exists(stem.stem_path):
                        os.remove(stem.stem_path)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file {stem.stem_path}: {e}")