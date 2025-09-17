"""
Stem Manifest System for Deterministic Stitching and Quality Tracking

This module manages manifests that map chunk start/end times, stem IDs, ASR engine IDs, 
and offsets for deterministic stitching of separated audio stems. It includes per-stem 
SNR (Signal-to-Noise Ratio) and leakage scores to track separation quality and enable 
quality-based processing decisions.

Key Features:
- Chunk-level manifest generation with precise timing and offsets
- Per-stem quality metrics (SNR, leakage, separation confidence)
- Deterministic stitching support for distributed processing
- Integration with existing cache and processing systems
- Comprehensive quality tracking and reporting

Author: Advanced Ensemble Transcription System
"""

import os
import json
import time
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
import numpy as np

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import trace_stage, track_cost
from utils.intelligent_cache import cached_operation

@dataclass
class StemQualityMetrics:
    """Quality metrics for a separated stem"""
    stem_id: str
    
    # Core quality metrics
    snr_db: float  # Signal-to-noise ratio in decibels
    leakage_score: float  # Cross-stem leakage score (0-1, lower is better)
    separation_confidence: float  # Overall separation confidence (0-1)
    
    # Audio characteristics
    rms_energy: float  # Root mean square energy level
    spectral_centroid: float  # Spectral centroid (brightness indicator)
    zero_crossing_rate: float  # Zero crossing rate (spectral characteristic)
    
    # Quality assessments
    silence_ratio: float  # Ratio of silent frames (0-1)
    speech_activity_score: float  # Speech activity detection score (0-1)
    audio_quality_score: float  # Composite audio quality score (0-1)
    
    # Processing metadata
    processing_time: float = 0.0
    calculation_method: str = "rms_based"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StemChunk:
    """Individual chunk within a separated stem"""
    chunk_id: str
    stem_id: str
    
    # Timing information
    start_time: float
    end_time: float
    duration: float
    
    # File and offset information
    stem_file_path: str
    chunk_offset_samples: int
    chunk_length_samples: int
    sample_rate: int
    
    # ASR processing information
    asr_engine_id: str
    cache_key: str
    processing_status: str = "pending"  # pending, processing, completed, failed
    
    # Quality metrics for this chunk
    chunk_quality: Optional[StemQualityMetrics] = None
    
    # Processing metadata
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StemManifestEntry:
    """Complete manifest entry for a separated stem"""
    stem_id: str
    original_overlap_frame_id: str
    
    # File information
    stem_file_path: str
    stem_duration: float
    stem_sample_rate: int
    
    # Overall quality metrics
    overall_quality: StemQualityMetrics
    
    # Chunk breakdown
    chunks: List[StemChunk]
    total_chunks: int
    
    # Processing information
    source_separation_engine: str
    creation_timestamp: float
    processing_status: str = "active"  # active, archived, failed
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class StemManifest:
    """Complete manifest for all stems from source separation"""
    manifest_id: str
    overlap_frame_id: str
    
    # Overall timing
    start_time: float
    end_time: float
    duration: float
    
    # Stem entries
    stem_entries: List[StemManifestEntry]
    total_stems: int
    
    # Overall quality assessment
    average_snr_db: float
    average_leakage_score: float
    separation_success_rate: float  # Percentage of successful separations
    
    # Processing information
    source_separation_config: Dict[str, Any]
    chunking_config: Dict[str, Any]
    creation_timestamp: float
    processing_duration: float
    
    # Cache and retrieval information
    cache_base_path: str
    manifest_file_path: str
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class StemQualityAnalyzer:
    """Analyzes quality metrics for separated stems"""
    
    def __init__(self,
                 min_snr_threshold: float = 6.0,
                 max_leakage_threshold: float = 0.3,
                 min_speech_activity: float = 0.1):
        """
        Initialize stem quality analyzer
        
        Args:
            min_snr_threshold: Minimum SNR (dB) for acceptable quality
            max_leakage_threshold: Maximum leakage score for acceptable quality
            min_speech_activity: Minimum speech activity for valid stems
        """
        self.min_snr_threshold = min_snr_threshold
        self.max_leakage_threshold = max_leakage_threshold
        self.min_speech_activity = min_speech_activity
        
        self.logger = create_enhanced_logger("stem_quality_analyzer")
    
    @trace_stage("stem_quality_analysis")
    def analyze_stem_quality(self,
                           stem_id: str,
                           stem_audio_path: str,
                           reference_audio_paths: Optional[List[str]] = None) -> StemQualityMetrics:
        """
        Analyze quality metrics for a separated stem
        
        Args:
            stem_id: Unique identifier for the stem
            stem_audio_path: Path to the separated stem audio file
            reference_audio_paths: Paths to other stems for leakage calculation
            
        Returns:
            Complete quality metrics for the stem
        """
        start_time = time.time()
        
        try:
            # Load stem audio
            import librosa
            stem_audio, sample_rate = librosa.load(stem_audio_path, sr=None)
            
            # Calculate core quality metrics
            snr_db = self._calculate_snr(stem_audio)
            leakage_score = self._calculate_leakage_score(
                stem_audio, stem_audio_path, reference_audio_paths or []
            )
            
            # Calculate audio characteristics
            rms_energy = self._calculate_rms_energy(stem_audio)
            spectral_centroid = self._calculate_spectral_centroid(stem_audio, sample_rate)
            zero_crossing_rate = self._calculate_zero_crossing_rate(stem_audio)
            
            # Calculate activity and quality scores
            silence_ratio = self._calculate_silence_ratio(stem_audio)
            speech_activity_score = self._estimate_speech_activity(stem_audio, sample_rate)
            
            # Calculate separation confidence based on individual metrics
            separation_confidence = self._calculate_separation_confidence(
                snr_db, leakage_score, speech_activity_score, silence_ratio
            )
            
            # Calculate composite audio quality score
            audio_quality_score = self._calculate_audio_quality_score(
                snr_db, spectral_centroid, zero_crossing_rate, speech_activity_score
            )
            
            processing_time = time.time() - start_time
            
            quality_metrics = StemQualityMetrics(
                stem_id=stem_id,
                snr_db=snr_db,
                leakage_score=leakage_score,
                separation_confidence=separation_confidence,
                rms_energy=rms_energy,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=zero_crossing_rate,
                silence_ratio=silence_ratio,
                speech_activity_score=speech_activity_score,
                audio_quality_score=audio_quality_score,
                processing_time=processing_time,
                calculation_method="librosa_based",
                metadata={
                    'sample_rate': sample_rate,
                    'audio_duration': len(stem_audio) / sample_rate,
                    'audio_samples': len(stem_audio)
                }
            )
            
            self.logger.info(f"Quality analysis completed for stem {stem_id}",
                           context={
                               'stem_id': stem_id,
                               'snr_db': snr_db,
                               'leakage_score': leakage_score,
                               'separation_confidence': separation_confidence,
                               'processing_time': processing_time
                           })
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze quality for stem {stem_id}: {e}")
            
            # Return default metrics on failure
            return StemQualityMetrics(
                stem_id=stem_id,
                snr_db=0.0,
                leakage_score=1.0,
                separation_confidence=0.0,
                rms_energy=0.0,
                spectral_centroid=0.0,
                zero_crossing_rate=0.0,
                silence_ratio=1.0,
                speech_activity_score=0.0,
                audio_quality_score=0.0,
                processing_time=time.time() - start_time,
                calculation_method="failed",
                metadata={'error': str(e)}
            )
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio using spectral analysis"""
        try:
            if len(audio) == 0:
                return 0.0
            
            # Calculate RMS of the entire signal
            signal_rms = np.sqrt(np.mean(audio ** 2))
            
            # Estimate noise floor from quietest 10% of frames
            frame_size = 2048
            frame_rmses = []
            
            for i in range(0, len(audio) - frame_size, frame_size // 2):
                frame = audio[i:i + frame_size]
                frame_rms = np.sqrt(np.mean(frame ** 2))
                frame_rmses.append(frame_rms)
            
            if not frame_rmses:
                return 0.0
            
            # Use 10th percentile as noise floor estimate
            noise_floor = np.percentile(frame_rmses, 10)
            
            # Avoid division by zero
            if noise_floor <= 1e-8:
                noise_floor = 1e-8
            
            # Calculate SNR in dB
            snr_linear = signal_rms / noise_floor
            snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else 0.0
            
            return max(0.0, min(60.0, snr_db))  # Clamp to reasonable range
            
        except Exception:
            return 0.0
    
    def _calculate_leakage_score(self,
                               stem_audio: np.ndarray,
                               stem_path: str,
                               reference_paths: List[str]) -> float:
        """Calculate cross-stem leakage score"""
        
        if not reference_paths or len(reference_paths) == 0:
            # No references available, return neutral leakage score
            return 0.5
        
        try:
            import librosa
            
            # Calculate spectral features of current stem
            stem_stft = librosa.stft(stem_audio, n_fft=2048, hop_length=512)
            stem_magnitude = np.abs(stem_stft)
            
            # Calculate cross-correlation with reference stems
            total_correlation = 0.0
            valid_references = 0
            
            for ref_path in reference_paths:
                if ref_path != stem_path and os.path.exists(ref_path):
                    try:
                        ref_audio, _ = librosa.load(ref_path, sr=None)
                        
                        # Ensure same length for comparison
                        min_len = min(len(stem_audio), len(ref_audio))
                        if min_len > 1024:  # Minimum length for meaningful comparison
                            stem_segment = stem_audio[:min_len]
                            ref_segment = ref_audio[:min_len]
                            
                            # Calculate normalized cross-correlation
                            correlation = np.corrcoef(stem_segment, ref_segment)[0, 1]
                            
                            if not np.isnan(correlation):
                                total_correlation += abs(correlation)
                                valid_references += 1
                    
                    except Exception:
                        continue  # Skip problematic reference files
            
            if valid_references == 0:
                return 0.5  # Neutral score when no valid references
            
            # Average correlation as leakage indicator
            average_correlation = total_correlation / valid_references
            
            # Convert correlation to leakage score (higher correlation = more leakage)
            leakage_score = min(1.0, max(0.0, average_correlation))
            
            return leakage_score
            
        except Exception:
            return 0.5  # Neutral score on calculation failure
    
    def _calculate_rms_energy(self, audio: np.ndarray) -> float:
        """Calculate RMS energy level"""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def _calculate_spectral_centroid(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate spectral centroid (brightness indicator)"""
        try:
            import librosa
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            return float(np.mean(spectral_centroids))
        except Exception:
            return 0.0
    
    def _calculate_zero_crossing_rate(self, audio: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        try:
            import librosa
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            return float(np.mean(zcr))
        except Exception:
            return 0.0
    
    def _calculate_silence_ratio(self, audio: np.ndarray, silence_threshold: float = 0.01) -> float:
        """Calculate ratio of silent frames"""
        if len(audio) == 0:
            return 1.0
        
        frame_size = 2048
        silent_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio) - frame_size, frame_size // 2):
            frame = audio[i:i + frame_size]
            frame_energy = np.sqrt(np.mean(frame ** 2))
            
            if frame_energy < silence_threshold:
                silent_frames += 1
            total_frames += 1
        
        return silent_frames / total_frames if total_frames > 0 else 1.0
    
    def _estimate_speech_activity(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate speech activity using energy and spectral features"""
        try:
            import librosa
            
            # Calculate energy-based VAD
            frame_size = int(0.025 * sample_rate)  # 25ms frames
            hop_size = int(0.010 * sample_rate)    # 10ms hop
            
            energy_vad_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                frame_energy = np.sqrt(np.mean(frame ** 2))
                
                # Simple energy-based VAD threshold
                if frame_energy > 0.01:
                    energy_vad_frames += 1
                total_frames += 1
            
            energy_vad_ratio = energy_vad_frames / total_frames if total_frames > 0 else 0.0
            
            # Calculate spectral-based features
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            spectral_activity = np.mean(spectral_rolloff > 0.02)  # Threshold for spectral activity
            
            # Combine energy and spectral features
            speech_activity_score = (energy_vad_ratio * 0.7 + spectral_activity * 0.3)
            
            return float(max(0.0, min(1.0, speech_activity_score)))
            
        except Exception:
            return 0.0
    
    def _calculate_separation_confidence(self,
                                       snr_db: float,
                                       leakage_score: float,
                                       speech_activity_score: float,
                                       silence_ratio: float) -> float:
        """Calculate overall separation confidence score"""
        
        # SNR component (normalized to 0-1)
        snr_component = min(1.0, max(0.0, snr_db / 20.0))  # 20 dB = excellent
        
        # Leakage component (inverted - lower leakage is better)
        leakage_component = 1.0 - min(1.0, max(0.0, leakage_score))
        
        # Speech activity component
        activity_component = speech_activity_score
        
        # Silence penalty (too much silence indicates poor separation)
        silence_penalty = 1.0 if silence_ratio < 0.8 else (1.0 - silence_ratio)
        
        # Weighted combination
        confidence = (
            snr_component * 0.3 +
            leakage_component * 0.3 +
            activity_component * 0.2 +
            silence_penalty * 0.2
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_audio_quality_score(self,
                                     snr_db: float,
                                     spectral_centroid: float,
                                     zero_crossing_rate: float,
                                     speech_activity_score: float) -> float:
        """Calculate composite audio quality score"""
        
        # SNR quality component
        snr_quality = min(1.0, max(0.0, snr_db / 25.0))  # 25 dB = high quality
        
        # Spectral balance (reasonable spectral centroid indicates good quality)
        spectral_balance = 1.0 - abs(spectral_centroid - 2000) / 5000  # Centered around 2kHz
        spectral_balance = max(0.0, min(1.0, spectral_balance))
        
        # ZCR quality (moderate ZCR indicates speech-like content)
        zcr_quality = 1.0 - abs(zero_crossing_rate - 0.1)  # Target around 0.1
        zcr_quality = max(0.0, min(1.0, zcr_quality))
        
        # Activity quality
        activity_quality = speech_activity_score
        
        # Weighted combination
        quality_score = (
            snr_quality * 0.4 +
            spectral_balance * 0.2 +
            zcr_quality * 0.2 +
            activity_quality * 0.2
        )
        
        return max(0.0, min(1.0, quality_score))

class StemChunker:
    """Creates chunk breakdowns for separated stems"""
    
    def __init__(self,
                 chunk_duration: float = 30.0,
                 overlap_seconds: float = 1.0,
                 min_chunk_duration: float = 5.0):
        """
        Initialize stem chunker
        
        Args:
            chunk_duration: Target chunk duration in seconds
            overlap_seconds: Overlap between adjacent chunks
            min_chunk_duration: Minimum chunk duration to avoid tiny chunks
        """
        self.chunk_duration = chunk_duration
        self.overlap_seconds = overlap_seconds
        self.min_chunk_duration = min_chunk_duration
        
        self.logger = create_enhanced_logger("stem_chunker")
    
    def create_stem_chunks(self,
                         stem_id: str,
                         stem_file_path: str,
                         stem_duration: float,
                         sample_rate: int,
                         asr_engine_ids: List[str]) -> List[StemChunk]:
        """
        Create chunk breakdown for a separated stem
        
        Args:
            stem_id: Unique identifier for the stem
            stem_file_path: Path to the stem audio file
            stem_duration: Duration of the stem in seconds
            sample_rate: Sample rate of the stem audio
            asr_engine_ids: List of ASR engines that will process chunks
            
        Returns:
            List of stem chunks with proper timing and cache keys
        """
        if stem_duration < self.min_chunk_duration:
            # Single chunk for short stems
            return self._create_single_chunk(stem_id, stem_file_path, stem_duration, sample_rate, asr_engine_ids)
        
        chunks = []
        current_time = 0.0
        chunk_index = 0
        
        while current_time < stem_duration:
            # Calculate chunk boundaries
            chunk_start = max(0.0, current_time - self.overlap_seconds if chunk_index > 0 else 0.0)
            chunk_end = min(stem_duration, current_time + self.chunk_duration)
            chunk_duration = chunk_end - chunk_start
            
            # Skip tiny chunks at the end
            if chunk_duration < self.min_chunk_duration and chunk_index > 0:
                # Extend the previous chunk to include the remainder
                if chunks:
                    chunks[-1].end_time = chunk_end
                    chunks[-1].duration = chunk_end - chunks[-1].start_time
                    chunks[-1].chunk_length_samples = int((chunk_end - chunks[-1].start_time) * sample_rate)
                break
            
            # Create chunks for each ASR engine
            for asr_engine_id in asr_engine_ids:
                chunk_id = f"{stem_id}_chunk_{chunk_index:03d}_{asr_engine_id}"
                cache_key = self._generate_cache_key(stem_id, chunk_index, asr_engine_id, chunk_start, chunk_end)
                
                chunk = StemChunk(
                    chunk_id=chunk_id,
                    stem_id=stem_id,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    duration=chunk_duration,
                    stem_file_path=stem_file_path,
                    chunk_offset_samples=int(chunk_start * sample_rate),
                    chunk_length_samples=int(chunk_duration * sample_rate),
                    sample_rate=sample_rate,
                    asr_engine_id=asr_engine_id,
                    cache_key=cache_key,
                    processing_metadata={
                        'chunk_index': chunk_index,
                        'overlap_applied': chunk_start < current_time if chunk_index > 0 else False
                    }
                )
                chunks.append(chunk)
            
            # Move to next chunk
            current_time += self.chunk_duration
            chunk_index += 1
        
        self.logger.info(f"Created {len(chunks)} chunks for stem {stem_id}",
                        context={
                            'stem_id': stem_id,
                            'stem_duration': stem_duration,
                            'chunks_created': len(chunks),
                            'asr_engines': len(asr_engine_ids),
                            'chunk_duration': self.chunk_duration
                        })
        
        return chunks
    
    def _create_single_chunk(self,
                           stem_id: str,
                           stem_file_path: str,
                           stem_duration: float,
                           sample_rate: int,
                           asr_engine_ids: List[str]) -> List[StemChunk]:
        """Create a single chunk for short stems"""
        
        chunks = []
        
        for asr_engine_id in asr_engine_ids:
            chunk_id = f"{stem_id}_chunk_000_{asr_engine_id}"
            cache_key = self._generate_cache_key(stem_id, 0, asr_engine_id, 0.0, stem_duration)
            
            chunk = StemChunk(
                chunk_id=chunk_id,
                stem_id=stem_id,
                start_time=0.0,
                end_time=stem_duration,
                duration=stem_duration,
                stem_file_path=stem_file_path,
                chunk_offset_samples=0,
                chunk_length_samples=int(stem_duration * sample_rate),
                sample_rate=sample_rate,
                asr_engine_id=asr_engine_id,
                cache_key=cache_key,
                processing_metadata={
                    'chunk_index': 0,
                    'single_chunk': True
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_cache_key(self,
                          stem_id: str,
                          chunk_index: int,
                          asr_engine_id: str,
                          start_time: float,
                          end_time: float) -> str:
        """Generate deterministic cache key for chunk processing"""
        
        # Create reproducible cache key
        key_components = [
            stem_id,
            f"chunk_{chunk_index:03d}",
            asr_engine_id,
            f"start_{start_time:.3f}",
            f"end_{end_time:.3f}"
        ]
        
        key_string = "_".join(key_components)
        
        # Hash for consistent length
        cache_key = hashlib.md5(key_string.encode()).hexdigest()[:16]
        
        return f"stem_chunk_{cache_key}"

class StemManifestManager:
    """Main manager for stem manifests and processing coordination"""
    
    def __init__(self,
                 cache_base_path: str = "/tmp/stem_cache",
                 enable_quality_analysis: bool = True,
                 chunking_config: Optional[Dict[str, Any]] = None):
        """
        Initialize stem manifest manager
        
        Args:
            cache_base_path: Base directory for stem caching
            enable_quality_analysis: Enable automatic quality analysis
            chunking_config: Configuration for chunk creation
        """
        self.cache_base_path = Path(cache_base_path)
        self.cache_base_path.mkdir(exist_ok=True)
        
        self.enable_quality_analysis = enable_quality_analysis
        
        # Initialize components
        self.quality_analyzer = StemQualityAnalyzer() if enable_quality_analysis else None
        
        # Chunking configuration
        chunking_config = chunking_config or {}
        self.chunker = StemChunker(
            chunk_duration=chunking_config.get('chunk_duration', 30.0),
            overlap_seconds=chunking_config.get('overlap_seconds', 1.0),
            min_chunk_duration=chunking_config.get('min_chunk_duration', 5.0)
        )
        
        self.logger = create_enhanced_logger("stem_manifest_manager")
        
        # Processing state
        self._active_manifests: Dict[str, StemManifest] = {}
    
    @trace_stage("stem_manifest_creation")
    def create_stem_manifest(self,
                           overlap_frame_id: str,
                           separated_stems: List[Any],  # List[SeparatedStem]
                           asr_engine_ids: List[str],
                           source_separation_config: Dict[str, Any],
                           start_time: float,
                           end_time: float) -> StemManifest:
        """
        Create complete stem manifest for separated stems
        
        Args:
            overlap_frame_id: ID of the original overlap frame
            separated_stems: List of separated stems from source separation
            asr_engine_ids: List of ASR engines to process stems
            source_separation_config: Configuration used for separation
            start_time: Start time of the overlap frame
            end_time: End time of the overlap frame
            
        Returns:
            Complete stem manifest with all processing information
        """
        creation_start_time = time.time()
        
        manifest_id = f"stem_manifest_{overlap_frame_id}_{int(creation_start_time)}"
        
        self.logger.info(f"Creating stem manifest for {len(separated_stems)} stems",
                        context={
                            'manifest_id': manifest_id,
                            'overlap_frame_id': overlap_frame_id,
                            'num_stems': len(separated_stems),
                            'asr_engines': len(asr_engine_ids)
                        })
        
        # Create manifest entries for each stem
        stem_entries = []
        quality_metrics_list = []
        
        for stem in separated_stems:
            try:
                # Analyze stem quality if enabled
                overall_quality = None
                if self.enable_quality_analysis and self.quality_analyzer:
                    reference_paths = [s.stem_path for s in separated_stems if s.stem_path != stem.stem_path]
                    overall_quality = self.quality_analyzer.analyze_stem_quality(
                        stem.speaker_id, stem.stem_path, reference_paths
                    )
                    quality_metrics_list.append(overall_quality)
                else:
                    # Create default quality metrics
                    overall_quality = StemQualityMetrics(
                        stem_id=stem.speaker_id,
                        snr_db=10.0,  # Default reasonable SNR
                        leakage_score=0.3,  # Default moderate leakage
                        separation_confidence=0.7,  # Default reasonable confidence
                        rms_energy=0.1,
                        spectral_centroid=2000.0,
                        zero_crossing_rate=0.1,
                        silence_ratio=0.2,
                        speech_activity_score=0.8,
                        audio_quality_score=0.7,
                        calculation_method="default"
                    )
                    quality_metrics_list.append(overall_quality)
                
                # Get stem duration
                stem_duration = self._get_stem_duration(stem.stem_path)
                stem_sample_rate = self._get_stem_sample_rate(stem.stem_path)
                
                # Create chunks for this stem
                chunks = self.chunker.create_stem_chunks(
                    stem.speaker_id, stem.stem_path, stem_duration, stem_sample_rate, asr_engine_ids
                )
                
                # Create manifest entry
                stem_entry = StemManifestEntry(
                    stem_id=stem.speaker_id,
                    original_overlap_frame_id=overlap_frame_id,
                    stem_file_path=stem.stem_path,
                    stem_duration=stem_duration,
                    stem_sample_rate=stem_sample_rate,
                    overall_quality=overall_quality,
                    chunks=chunks,
                    total_chunks=len(chunks),
                    source_separation_engine="demucs",  # or from config
                    creation_timestamp=creation_start_time,
                    metadata={
                        'original_stem_metadata': stem.processing_metadata,
                        'stem_confidence': stem.confidence
                    }
                )
                
                stem_entries.append(stem_entry)
                
                self.logger.info(f"Created manifest entry for stem {stem.speaker_id}",
                               context={
                                   'stem_id': stem.speaker_id,
                                   'chunks_created': len(chunks),
                                   'stem_duration': stem_duration,
                                   'quality_analyzed': self.enable_quality_analysis
                               })
                
            except Exception as e:
                self.logger.error(f"Failed to create manifest entry for stem {stem.speaker_id}: {e}")
                continue
        
        if not stem_entries:
            raise ValueError("No valid stem entries could be created")
        
        # Calculate overall quality metrics
        if quality_metrics_list:
            average_snr_db = np.mean([q.snr_db for q in quality_metrics_list])
            average_leakage_score = np.mean([q.leakage_score for q in quality_metrics_list])
            separation_success_rate = len(stem_entries) / len(separated_stems) if separated_stems else 0.0
        else:
            average_snr_db = 0.0
            average_leakage_score = 1.0
            separation_success_rate = 0.0
        
        processing_duration = time.time() - creation_start_time
        
        # Create manifest file path
        manifest_file_path = self.cache_base_path / f"{manifest_id}.json"
        
        # Create complete manifest
        manifest = StemManifest(
            manifest_id=manifest_id,
            overlap_frame_id=overlap_frame_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            stem_entries=stem_entries,
            total_stems=len(stem_entries),
            average_snr_db=average_snr_db,
            average_leakage_score=average_leakage_score,
            separation_success_rate=separation_success_rate,
            source_separation_config=source_separation_config,
            chunking_config={
                'chunk_duration': self.chunker.chunk_duration,
                'overlap_seconds': self.chunker.overlap_seconds,
                'min_chunk_duration': self.chunker.min_chunk_duration
            },
            creation_timestamp=creation_start_time,
            processing_duration=processing_duration,
            cache_base_path=str(self.cache_base_path),
            manifest_file_path=str(manifest_file_path),
            metadata={
                'total_chunks': sum(entry.total_chunks for entry in stem_entries),
                'quality_analysis_enabled': self.enable_quality_analysis
            }
        )
        
        # Save manifest to file
        self._save_manifest(manifest)
        
        # Store in active manifests
        self._active_manifests[manifest_id] = manifest
        
        self.logger.info(f"Stem manifest created successfully",
                        context={
                            'manifest_id': manifest_id,
                            'total_stems': len(stem_entries),
                            'total_chunks': sum(entry.total_chunks for entry in stem_entries),
                            'average_snr_db': average_snr_db,
                            'separation_success_rate': separation_success_rate,
                            'processing_duration': processing_duration
                        })
        
        return manifest
    
    def _get_stem_duration(self, stem_path: str) -> float:
        """Get duration of a stem audio file"""
        try:
            import librosa
            duration = librosa.get_duration(path=stem_path)
            return duration
        except Exception:
            return 0.0
    
    def _get_stem_sample_rate(self, stem_path: str) -> int:
        """Get sample rate of a stem audio file"""
        try:
            import librosa
            _, sample_rate = librosa.load(stem_path, sr=None, duration=0.1)  # Load just a small portion
            return int(sample_rate)
        except Exception:
            return 44100  # Default sample rate
    
    def _save_manifest(self, manifest: StemManifest) -> None:
        """Save manifest to JSON file"""
        try:
            manifest_dict = asdict(manifest)
            
            with open(manifest.manifest_file_path, 'w') as f:
                json.dump(manifest_dict, f, indent=2, default=str)
                
            self.logger.info(f"Manifest saved to {manifest.manifest_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save manifest {manifest.manifest_id}: {e}")
    
    def load_manifest(self, manifest_id: str) -> Optional[StemManifest]:
        """Load manifest from file"""
        
        # Check active manifests first
        if manifest_id in self._active_manifests:
            return self._active_manifests[manifest_id]
        
        # Try to load from file
        manifest_file_path = self.cache_base_path / f"{manifest_id}.json"
        
        if not manifest_file_path.exists():
            self.logger.warning(f"Manifest file not found: {manifest_file_path}")
            return None
        
        try:
            with open(manifest_file_path, 'r') as f:
                manifest_dict = json.load(f)
            
            # Reconstruct manifest object (simplified for now)
            # In a full implementation, this would properly reconstruct all nested objects
            self.logger.info(f"Manifest {manifest_id} loaded from file")
            
            return None  # TODO: Implement full deserialization
            
        except Exception as e:
            self.logger.error(f"Failed to load manifest {manifest_id}: {e}")
            return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        
        active_manifests = len(self._active_manifests)
        total_stems = sum(manifest.total_stems for manifest in self._active_manifests.values())
        total_chunks = sum(
            sum(entry.total_chunks for entry in manifest.stem_entries)
            for manifest in self._active_manifests.values()
        )
        
        if self._active_manifests:
            average_snr = np.mean([manifest.average_snr_db for manifest in self._active_manifests.values()])
            average_leakage = np.mean([manifest.average_leakage_score for manifest in self._active_manifests.values()])
            average_success_rate = np.mean([manifest.separation_success_rate for manifest in self._active_manifests.values()])
        else:
            average_snr = 0.0
            average_leakage = 0.0
            average_success_rate = 0.0
        
        return {
            'active_manifests': active_manifests,
            'total_stems': total_stems,
            'total_chunks': total_chunks,
            'average_snr_db': average_snr,
            'average_leakage_score': average_leakage,
            'average_separation_success_rate': average_success_rate,
            'cache_base_path': str(self.cache_base_path),
            'quality_analysis_enabled': self.enable_quality_analysis
        }
    
    def cleanup_old_manifests(self, max_age_hours: float = 24.0) -> None:
        """Clean up old manifest files and cached data"""
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_count = 0
        
        # Clean up active manifests
        manifests_to_remove = []
        for manifest_id, manifest in self._active_manifests.items():
            if current_time - manifest.creation_timestamp > max_age_seconds:
                manifests_to_remove.append(manifest_id)
        
        for manifest_id in manifests_to_remove:
            del self._active_manifests[manifest_id]
            cleaned_count += 1
        
        # Clean up manifest files
        if self.cache_base_path.exists():
            for manifest_file in self.cache_base_path.glob("*.json"):
                if current_time - manifest_file.stat().st_mtime > max_age_seconds:
                    try:
                        manifest_file.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to delete old manifest file {manifest_file}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old manifests")