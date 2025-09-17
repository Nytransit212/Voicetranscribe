"""
Overlap-Aware Diarization Engine

This module performs speaker diarization on individual separated stems from source separation,
with special handling for overlap detection and dual-speaker labeling. It runs diarization 
on each stem separately and produces turn segments with speaker labels, stem IDs, and overlap 
tags when segments coincide across stems.

Key Features:
- Per-stem diarization with speaker identification
- Cross-stem overlap detection and tagging
- Speaker consistency tracking across stems
- Integration with existing diarization providers
- Overlap metadata generation for fusion

Author: Advanced Ensemble Transcription System
"""

import os
import time
import tempfile
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from core.diarization_engine import DiarizationEngine
from core.source_separation_engine import SeparatedStem, SourceSeparationResult, OverlapFrame
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import trace_stage, track_cost
from utils.intelligent_cache import cached_operation

@dataclass
class StemDiarizationSegment:
    """Diarization segment for a specific stem with overlap metadata"""
    start_time: float
    end_time: float
    duration: float
    speaker_id: str
    confidence: float
    stem_id: str
    stem_path: str
    
    # Overlap detection fields
    is_overlapped: bool = False
    overlap_probability: float = 0.0
    overlapping_stems: List[str] = field(default_factory=list)
    overlapping_speakers: List[str] = field(default_factory=list)
    
    # Quality and consistency metrics
    stem_quality: float = 1.0
    speaker_consistency: float = 1.0
    temporal_coherence: float = 1.0
    
    # Processing metadata
    diarization_provider: str = "unknown"
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StemDiarizationResult:
    """Complete diarization result for a single stem"""
    stem: SeparatedStem
    segments: List[StemDiarizationSegment]
    total_speech_duration: float
    speaker_count: int
    speakers_detected: List[str]
    diarization_confidence: float
    processing_time: float
    provider_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossStemOverlapRegion:
    """Represents an overlap region detected across multiple stems"""
    start_time: float
    end_time: float
    duration: float
    stems_involved: List[str]
    speakers_involved: List[str]
    overlap_confidence: float
    segments_involved: List[StemDiarizationSegment]
    
    # Reconciliation metadata
    dominant_stem: Optional[str] = None
    dominant_speaker: Optional[str] = None
    reconciliation_strategy: str = "confidence_based"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OverlapDiarizationResult:
    """Complete overlap-aware diarization result across all stems"""
    original_overlap_frame: OverlapFrame
    stem_results: List[StemDiarizationResult]
    cross_stem_overlaps: List[CrossStemOverlapRegion]
    unified_speaker_map: Dict[str, str]  # stem_speaker_id -> unified_speaker_id
    processing_metrics: Dict[str, Any]
    total_processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class StemDiarizer:
    """Performs diarization on individual separated stems"""
    
    def __init__(self,
                 base_diarization_engine: Optional[DiarizationEngine] = None,
                 enable_speaker_mapping: bool = True,
                 overlap_detection_threshold: float = 0.1,
                 min_segment_duration: float = 0.5,
                 enable_caching: bool = True):
        """
        Initialize stem diarizer
        
        Args:
            base_diarization_engine: Base diarization engine to use for stems
            enable_speaker_mapping: Enable cross-stem speaker identity mapping
            overlap_detection_threshold: Minimum temporal overlap to consider
            min_segment_duration: Minimum segment duration for valid segments
            enable_caching: Enable result caching
        """
        self.base_engine = base_diarization_engine or DiarizationEngine(
            expected_speakers=2,  # Conservative for separated stems
            noise_level='low',  # Stems should be cleaner
            enable_turn_stabilization=True
        )
        
        self.enable_speaker_mapping = enable_speaker_mapping
        self.overlap_detection_threshold = overlap_detection_threshold
        self.min_segment_duration = min_segment_duration
        self.enable_caching = enable_caching
        
        self.logger = create_enhanced_logger("stem_diarizer")
        
        # Processing state
        self._stem_cache = {} if enable_caching else None
        self._speaker_embedding_cache = {}
        
    @trace_stage("stem_diarization")
    def diarize_stems(self, 
                     stems: List[SeparatedStem],
                     original_overlap_frame: OverlapFrame) -> OverlapDiarizationResult:
        """
        Perform diarization on all separated stems
        
        Args:
            stems: List of separated stems to diarize
            original_overlap_frame: Original overlap frame context
            
        Returns:
            Complete overlap-aware diarization result
        """
        start_time = time.time()
        
        self.logger.info(f"Starting stem diarization for {len(stems)} stems",
                        context={
                            'num_stems': len(stems),
                            'overlap_frame_duration': original_overlap_frame.duration,
                            'expected_speakers': len(original_overlap_frame.speakers_involved)
                        })
        
        # Diarize each stem individually
        stem_results = []
        for i, stem in enumerate(stems):
            self.logger.info(f"Diarizing stem {i+1}/{len(stems)}: {stem.speaker_id}")
            
            try:
                stem_result = self._diarize_single_stem(stem, original_overlap_frame)
                if stem_result:
                    stem_results.append(stem_result)
                    self.logger.info(f"Stem {stem.speaker_id} diarization complete: "
                                   f"{len(stem_result.segments)} segments, "
                                   f"{stem_result.speaker_count} speakers")
                else:
                    self.logger.warning(f"Failed to diarize stem {stem.speaker_id}")
                    
            except Exception as e:
                self.logger.error(f"Error diarizing stem {stem.speaker_id}: {e}")
                continue
        
        if not stem_results:
            self.logger.error("No stems were successfully diarized")
            return self._create_empty_result(stems, original_overlap_frame, time.time() - start_time)
        
        # Detect cross-stem overlaps
        cross_stem_overlaps = self._detect_cross_stem_overlaps(stem_results)
        
        # Create unified speaker mapping
        unified_speaker_map = self._create_unified_speaker_map(stem_results, original_overlap_frame)
        
        # Apply overlap tags to segments
        self._apply_overlap_tags(stem_results, cross_stem_overlaps)
        
        processing_time = time.time() - start_time
        
        # Calculate processing metrics
        processing_metrics = self._calculate_processing_metrics(stem_results, cross_stem_overlaps, processing_time)
        
        result = OverlapDiarizationResult(
            original_overlap_frame=original_overlap_frame,
            stem_results=stem_results,
            cross_stem_overlaps=cross_stem_overlaps,
            unified_speaker_map=unified_speaker_map,
            processing_metrics=processing_metrics,
            total_processing_time=processing_time,
            metadata={
                'stems_processed': len(stems),
                'stems_successful': len(stem_results),
                'cross_stem_overlaps_detected': len(cross_stem_overlaps),
                'unified_speakers': len(set(unified_speaker_map.values()))
            }
        )
        
        self.logger.info("Stem diarization completed",
                        context={
                            'total_processing_time': processing_time,
                            'stems_successful': len(stem_results),
                            'cross_stem_overlaps': len(cross_stem_overlaps),
                            'unified_speakers': len(set(unified_speaker_map.values()))
                        })
        
        return result
    
    @cached_operation
    def _diarize_single_stem(self, 
                           stem: SeparatedStem,
                           original_overlap_frame: OverlapFrame) -> Optional[StemDiarizationResult]:
        """
        Diarize a single separated stem
        
        Args:
            stem: Separated stem to diarize
            original_overlap_frame: Original overlap frame context
            
        Returns:
            Diarization result for the stem or None if failed
        """
        if not os.path.exists(stem.stem_path):
            self.logger.error(f"Stem audio file not found: {stem.stem_path}")
            return None
        
        try:
            start_time = time.time()
            
            # Run base diarization on the stem
            # We need to create a temporary speaker count estimate for the stem
            # Most stems should have 1-2 speakers max after separation
            estimated_speakers = min(2, len(original_overlap_frame.speakers_involved))
            
            # Configure diarization for single stem (expect cleaner audio)
            # Use the appropriate method based on DiarizationEngine interface
            if hasattr(self.base_engine, 'run_diarization_variants'):
                diarization_variants = self.base_engine.run_diarization_variants(
                    audio_path=stem.stem_path,
                    expected_speakers=estimated_speakers,
                    enable_voting_fusion=False  # Single variant for stems
                )
            else:
                # Fallback to basic diarization if variants method not available
                basic_result = self.base_engine.diarize_audio(stem.stem_path)
                diarization_variants = [basic_result] if basic_result else []
            
            if not diarization_variants:
                self.logger.warning(f"No diarization results for stem {stem.speaker_id}")
                return None
            
            # Use the best (first) diarization variant
            best_diarization = diarization_variants[0]
            raw_segments = best_diarization.get('segments', [])
            
            if not raw_segments:
                self.logger.warning(f"No segments detected in stem {stem.speaker_id}")
                return None
            
            # Convert to stem diarization segments
            stem_segments = []
            speakers_detected = set()
            total_speech_duration = 0.0
            
            for seg in raw_segments:
                if seg['end'] - seg['start'] >= self.min_segment_duration:
                    stem_segment = StemDiarizationSegment(
                        start_time=seg['start'],
                        end_time=seg['end'],
                        duration=seg['end'] - seg['start'],
                        speaker_id=seg['speaker_id'],
                        confidence=seg.get('confidence', 1.0),
                        stem_id=stem.speaker_id,
                        stem_path=stem.stem_path,
                        stem_quality=stem.confidence,
                        diarization_provider=getattr(self.base_engine, 'active_provider', 'unknown'),
                        processing_metadata={
                            'original_segment': seg,
                            'stem_metadata': stem.processing_metadata
                        }
                    )
                    
                    stem_segments.append(stem_segment)
                    speakers_detected.add(seg['speaker_id'])
                    total_speech_duration += stem_segment.duration
            
            processing_time = time.time() - start_time
            
            # Calculate overall diarization confidence
            if stem_segments:
                diarization_confidence = float(np.mean([seg.confidence for seg in stem_segments]))
            else:
                diarization_confidence = 0.0
            
            return StemDiarizationResult(
                stem=stem,
                segments=stem_segments,
                total_speech_duration=total_speech_duration,
                speaker_count=len(speakers_detected),
                speakers_detected=list(speakers_detected),
                diarization_confidence=diarization_confidence,
                processing_time=processing_time,
                provider_used=getattr(self.base_engine, 'active_provider', 'unknown'),
                metadata={
                    'original_diarization_variant': best_diarization,
                    'segments_filtered': len(raw_segments) - len(stem_segments),
                    'min_segment_duration_applied': self.min_segment_duration
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error during stem diarization for {stem.speaker_id}: {e}")
            return None
    
    def _detect_cross_stem_overlaps(self, 
                                  stem_results: List[StemDiarizationResult]) -> List[CrossStemOverlapRegion]:
        """
        Detect overlapping regions across multiple stems
        
        Args:
            stem_results: Diarization results for all stems
            
        Returns:
            List of detected cross-stem overlap regions
        """
        if len(stem_results) < 2:
            return []
        
        cross_stem_overlaps = []
        
        # Compare segments across all stem pairs
        for i in range(len(stem_results)):
            for j in range(i + 1, len(stem_results)):
                stem1_result = stem_results[i]
                stem2_result = stem_results[j]
                
                # Find overlapping segments between stems
                for seg1 in stem1_result.segments:
                    for seg2 in stem2_result.segments:
                        # Calculate temporal overlap
                        overlap_start = max(seg1.start_time, seg2.start_time)
                        overlap_end = min(seg1.end_time, seg2.end_time)
                        
                        if overlap_start < overlap_end:
                            overlap_duration = overlap_end - overlap_start
                            
                            # Check if overlap meets threshold
                            if overlap_duration >= self.overlap_detection_threshold:
                                # Calculate overlap confidence
                                overlap_confidence = self._calculate_overlap_confidence(
                                    seg1, seg2, overlap_duration
                                )
                                
                                cross_stem_overlap = CrossStemOverlapRegion(
                                    start_time=overlap_start,
                                    end_time=overlap_end,
                                    duration=overlap_duration,
                                    stems_involved=[seg1.stem_id, seg2.stem_id],
                                    speakers_involved=[seg1.speaker_id, seg2.speaker_id],
                                    overlap_confidence=overlap_confidence,
                                    segments_involved=[seg1, seg2],
                                    dominant_stem=seg1.stem_id if seg1.confidence > seg2.confidence else seg2.stem_id,
                                    dominant_speaker=seg1.speaker_id if seg1.confidence > seg2.confidence else seg2.speaker_id,
                                    metadata={
                                        'overlap_ratio_seg1': overlap_duration / seg1.duration,
                                        'overlap_ratio_seg2': overlap_duration / seg2.duration,
                                        'confidence_diff': abs(seg1.confidence - seg2.confidence)
                                    }
                                )
                                
                                cross_stem_overlaps.append(cross_stem_overlap)
        
        # Merge nearby overlaps to avoid fragmentation
        merged_overlaps = self._merge_nearby_overlaps(cross_stem_overlaps)
        
        self.logger.info(f"Detected {len(cross_stem_overlaps)} raw cross-stem overlaps, "
                        f"merged to {len(merged_overlaps)} final regions")
        
        return merged_overlaps
    
    def _calculate_overlap_confidence(self,
                                    seg1: StemDiarizationSegment,
                                    seg2: StemDiarizationSegment,
                                    overlap_duration: float) -> float:
        """Calculate confidence score for a cross-stem overlap"""
        
        # Base confidence from segment confidences
        base_confidence = (seg1.confidence + seg2.confidence) / 2
        
        # Duration factor (longer overlaps are more reliable)
        duration_factor = min(1.0, overlap_duration / 1.0)  # Normalize to 1 second
        
        # Stem quality factor
        quality_factor = (seg1.stem_quality + seg2.stem_quality) / 2
        
        # Different speakers are more likely to overlap
        speaker_difference_bonus = 1.2 if seg1.speaker_id != seg2.speaker_id else 0.8
        
        # Combine all factors
        overlap_confidence = (base_confidence * 0.4 + 
                            duration_factor * 0.3 + 
                            quality_factor * 0.3) * speaker_difference_bonus
        
        return max(0.1, min(1.0, overlap_confidence))
    
    def _merge_nearby_overlaps(self, 
                             overlaps: List[CrossStemOverlapRegion],
                             merge_threshold: float = 0.2) -> List[CrossStemOverlapRegion]:
        """Merge nearby overlap regions to reduce fragmentation"""
        if not overlaps:
            return []
        
        # Sort by start time
        sorted_overlaps = sorted(overlaps, key=lambda x: x.start_time)
        merged = []
        current = sorted_overlaps[0]
        
        for next_overlap in sorted_overlaps[1:]:
            # Check if overlaps should be merged
            gap = next_overlap.start_time - current.end_time
            
            if (gap <= merge_threshold and 
                set(current.stems_involved) == set(next_overlap.stems_involved)):
                
                # Merge the overlaps
                current = CrossStemOverlapRegion(
                    start_time=current.start_time,
                    end_time=max(current.end_time, next_overlap.end_time),
                    duration=max(current.end_time, next_overlap.end_time) - current.start_time,
                    stems_involved=current.stems_involved,
                    speakers_involved=list(set(current.speakers_involved + next_overlap.speakers_involved)),
                    overlap_confidence=max(current.overlap_confidence, next_overlap.overlap_confidence),
                    segments_involved=current.segments_involved + next_overlap.segments_involved,
                    dominant_stem=current.dominant_stem if current.overlap_confidence > next_overlap.overlap_confidence else next_overlap.dominant_stem,
                    dominant_speaker=current.dominant_speaker if current.overlap_confidence > next_overlap.overlap_confidence else next_overlap.dominant_speaker,
                    reconciliation_strategy="merged",
                    metadata={
                        'merged_from': [current.metadata, next_overlap.metadata],
                        'merge_gap': gap
                    }
                )
            else:
                # No merge, add current to results and move to next
                merged.append(current)
                current = next_overlap
        
        merged.append(current)
        return merged
    
    def _create_unified_speaker_map(self,
                                  stem_results: List[StemDiarizationResult],
                                  original_overlap_frame: OverlapFrame) -> Dict[str, str]:
        """
        Create unified speaker mapping across stems
        
        Args:
            stem_results: Diarization results for all stems
            original_overlap_frame: Original overlap frame with known speakers
            
        Returns:
            Mapping from stem_speaker_id to unified_speaker_id
        """
        unified_map = {}
        
        if not self.enable_speaker_mapping:
            # Simple pass-through mapping
            for result in stem_results:
                for speaker in result.speakers_detected:
                    stem_speaker_key = f"{result.stem.speaker_id}_{speaker}"
                    unified_map[stem_speaker_key] = speaker
            return unified_map
        
        # More sophisticated mapping based on original speakers and consistency
        original_speakers = original_overlap_frame.speakers_involved
        used_unified_speakers = set()
        
        for result in stem_results:
            for speaker in result.speakers_detected:
                stem_speaker_key = f"{result.stem.speaker_id}_{speaker}"
                
                # Try to map to original speakers first
                best_match = None
                best_score = 0.0
                
                for original_speaker in original_speakers:
                    if original_speaker not in used_unified_speakers:
                        # Calculate similarity score (placeholder - could use embeddings)
                        similarity_score = self._calculate_speaker_similarity(
                            result, speaker, original_speaker
                        )
                        
                        if similarity_score > best_score:
                            best_score = similarity_score
                            best_match = original_speaker
                
                if best_match and best_score > 0.5:
                    unified_map[stem_speaker_key] = best_match
                    used_unified_speakers.add(best_match)
                else:
                    # Fall back to creating new speaker ID
                    new_speaker_id = f"speaker_{len(used_unified_speakers) + 1}"
                    unified_map[stem_speaker_key] = new_speaker_id
                    used_unified_speakers.add(new_speaker_id)
        
        self.logger.info(f"Created unified speaker mapping: {len(unified_map)} stem-speakers -> {len(used_unified_speakers)} unified speakers")
        
        return unified_map
    
    def _calculate_speaker_similarity(self,
                                    stem_result: StemDiarizationResult,
                                    stem_speaker: str,
                                    original_speaker: str) -> float:
        """Calculate similarity between stem speaker and original speaker"""
        
        # Placeholder similarity calculation
        # In a full implementation, this would use speaker embeddings
        
        # Simple heuristic based on stem index and speaker patterns
        if stem_speaker == original_speaker:
            return 0.9
        
        # Check if speaker ID patterns suggest similarity
        if original_speaker in stem_speaker or stem_speaker in original_speaker:
            return 0.7
        
        # Default similarity based on stem confidence
        return stem_result.diarization_confidence * 0.5
    
    def _apply_overlap_tags(self,
                          stem_results: List[StemDiarizationResult],
                          cross_stem_overlaps: List[CrossStemOverlapRegion]) -> None:
        """Apply overlap tags to segments based on detected overlaps"""
        
        for overlap_region in cross_stem_overlaps:
            for segment in overlap_region.segments_involved:
                # Find the segment in the stem results and tag it
                for stem_result in stem_results:
                    if stem_result.stem.speaker_id == segment.stem_id:
                        for seg in stem_result.segments:
                            if (seg.start_time == segment.start_time and 
                                seg.end_time == segment.end_time and
                                seg.speaker_id == segment.speaker_id):
                                
                                # Apply overlap tags
                                seg.is_overlapped = True
                                seg.overlap_probability = overlap_region.overlap_confidence
                                seg.overlapping_stems = overlap_region.stems_involved
                                seg.overlapping_speakers = overlap_region.speakers_involved
                                break
    
    def _calculate_processing_metrics(self,
                                    stem_results: List[StemDiarizationResult],
                                    cross_stem_overlaps: List[CrossStemOverlapRegion],
                                    total_processing_time: float) -> Dict[str, Any]:
        """Calculate comprehensive processing metrics"""
        
        if not stem_results:
            return {}
        
        # Basic metrics
        total_segments = sum(len(result.segments) for result in stem_results)
        total_speech_duration = sum(result.total_speech_duration for result in stem_results)
        total_overlapped_duration = sum(overlap.duration for overlap in cross_stem_overlaps)
        
        # Per-stem metrics
        per_stem_metrics = {}
        for result in stem_results:
            per_stem_metrics[result.stem.speaker_id] = {
                'segments_count': len(result.segments),
                'speech_duration': result.total_speech_duration,
                'speaker_count': result.speaker_count,
                'diarization_confidence': result.diarization_confidence,
                'processing_time': result.processing_time
            }
        
        # Overlap metrics
        overlap_metrics = {
            'total_overlap_regions': len(cross_stem_overlaps),
            'total_overlapped_duration': total_overlapped_duration,
            'average_overlap_duration': total_overlapped_duration / len(cross_stem_overlaps) if cross_stem_overlaps else 0.0,
            'average_overlap_confidence': np.mean([o.overlap_confidence for o in cross_stem_overlaps]) if cross_stem_overlaps else 0.0
        }
        
        # Processing efficiency metrics
        efficiency_metrics = {
            'total_processing_time': total_processing_time,
            'average_processing_time_per_stem': total_processing_time / len(stem_results),
            'segments_per_second': total_segments / total_processing_time if total_processing_time > 0 else 0
        }
        
        return {
            'basic_metrics': {
                'total_segments': total_segments,
                'total_speech_duration': total_speech_duration,
                'stems_processed': len(stem_results)
            },
            'per_stem_metrics': per_stem_metrics,
            'overlap_metrics': overlap_metrics,
            'efficiency_metrics': efficiency_metrics
        }
    
    def _create_empty_result(self, 
                           stems: List[SeparatedStem],
                           original_overlap_frame: OverlapFrame,
                           processing_time: float) -> OverlapDiarizationResult:
        """Create an empty result when diarization fails completely"""
        
        return OverlapDiarizationResult(
            original_overlap_frame=original_overlap_frame,
            stem_results=[],
            cross_stem_overlaps=[],
            unified_speaker_map={},
            processing_metrics={
                'error': 'no_stems_diarized',
                'stems_attempted': len(stems),
                'processing_time': processing_time
            },
            total_processing_time=processing_time,
            metadata={'error': 'diarization_failed'}
        )

class OverlapDiarizationEngine:
    """
    Main engine orchestrating overlap-aware diarization across separated stems
    """
    
    def __init__(self,
                 stem_diarizer: Optional[StemDiarizer] = None,
                 enable_cross_stem_analysis: bool = True,
                 enable_performance_optimizations: bool = True):
        """
        Initialize overlap diarization engine
        
        Args:
            stem_diarizer: Stem diarizer instance (will create default if None)
            enable_cross_stem_analysis: Enable cross-stem overlap analysis
            enable_performance_optimizations: Enable performance optimizations
        """
        self.stem_diarizer = stem_diarizer or StemDiarizer()
        self.enable_cross_stem_analysis = enable_cross_stem_analysis
        self.enable_performance_optimizations = enable_performance_optimizations
        
        self.logger = create_enhanced_logger("overlap_diarization_engine")
        
    @trace_stage("overlap_diarization_processing")
    def process_source_separation_result(self,
                                       separation_result: SourceSeparationResult) -> OverlapDiarizationResult:
        """
        Process a source separation result through overlap-aware diarization
        
        Args:
            separation_result: Result from source separation engine
            
        Returns:
            Complete overlap-aware diarization result
        """
        self.logger.info("Starting overlap-aware diarization processing",
                        context={
                            'overlap_frame_duration': separation_result.overlap_frame.duration,
                            'num_stems': len(separation_result.separated_stems),
                            'separation_confidence': separation_result.separation_confidence
                        })
        
        if not separation_result.separated_stems:
            self.logger.warning("No separated stems provided for diarization")
            return self.stem_diarizer._create_empty_result(
                [], separation_result.overlap_frame, 0.0
            )
        
        # Perform stem diarization
        diarization_result = self.stem_diarizer.diarize_stems(
            separation_result.separated_stems,
            separation_result.overlap_frame
        )
        
        # Add source separation context to metadata
        diarization_result.metadata.update({
            'source_separation_result': {
                'separation_confidence': separation_result.separation_confidence,
                'processing_time': separation_result.processing_time,
                'attribution_algorithm': separation_result.attribution_results.get('algorithm', 'unknown')
            }
        })
        
        self.logger.info("Overlap-aware diarization processing completed",
                        context={
                            'total_processing_time': diarization_result.total_processing_time,
                            'stems_successful': len(diarization_result.stem_results),
                            'cross_stem_overlaps': len(diarization_result.cross_stem_overlaps)
                        })
        
        return diarization_result