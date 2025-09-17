"""
Speaker Relabeling Engine for Global ID Assignment and Swap Detection

This module handles the final stage of long-horizon speaker tracking by:
1. Remapping local diarization labels to consistent global speaker IDs
2. Detecting and correcting speaker label swaps through embedding analysis  
3. Preserving overlap regions with dual-speaker labels
4. Computing consistency metrics and validation

Key Features:
- Global speaker ID assignment from clustering results
- Swap detection using embedding neighborhood analysis
- Contiguous block analysis for systematic label errors
- Overlap preservation for multi-speaker regions
- Human-friendly speaker name management
- Comprehensive validation and error correction

Process:
1. Apply global speaker mapping to fusion results
2. Detect swap candidates through embedding discontinuities
3. Validate and correct swaps using cluster cohesion analysis
4. Preserve overlap regions with multiple global speakers
5. Generate human-readable speaker labels

Author: Advanced Ensemble Transcription System
"""

import numpy as np
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter, deque
import copy

from core.overlap_fusion import OverlapFusionResult, FusedTranscriptSegment, WordAlignment, OverlapRegion
from core.global_speaker_linker import ClusteringResult, SpeakerCluster, TurnEmbedding
from utils.embedding_cache import get_embedding_cache, CacheEntry
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import trace_stage, track_cost
from utils.intelligent_cache import cached_operation
from scipy.spatial.distance import cosine, euclidean

@dataclass
class SwapCandidate:
    """Candidate speaker swap detected in analysis"""
    start_time: float
    end_time: float
    duration: float
    
    # Swap information
    original_speaker: str
    proposed_speaker: str
    local_speaker_id: str
    
    # Detection metrics
    embedding_discontinuity_score: float
    neighborhood_change_score: float
    contiguity_score: float
    
    # Validation metrics
    cluster_cohesion_improvement: float = 0.0
    confidence_score: float = 0.0
    
    # Context information
    affected_segments: List[int] = field(default_factory=list)
    surrounding_speakers: List[str] = field(default_factory=list)
    
    @property
    def swap_score(self) -> float:
        """Overall swap likelihood score"""
        base_score = (
            self.embedding_discontinuity_score * 0.4 +
            self.neighborhood_change_score * 0.3 +
            self.contiguity_score * 0.3
        )
        
        # Boost if cluster cohesion improves
        cohesion_boost = max(0, self.cluster_cohesion_improvement) * 0.5
        
        return min(base_score + cohesion_boost, 1.0)

@dataclass
class RelabelingResult:
    """Result of speaker relabeling and swap correction"""
    session_id: str
    
    # Input and output
    original_fusion_result: OverlapFusionResult
    relabeled_segments: List[FusedTranscriptSegment]
    global_speaker_mapping: Dict[str, str]
    
    # Swap detection and correction
    swap_candidates_detected: List[SwapCandidate]
    swaps_corrected: List[SwapCandidate]
    swap_detection_time: float
    
    # Consistency metrics
    speaker_consistency_score: float
    temporal_consistency_score: float
    overlap_preservation_score: float
    
    # Error correction metrics
    segments_relabeled: int
    swaps_corrected_count: int
    der_improvement_estimate: float
    
    # Quality validation
    validation_passed: bool = True
    validation_issues: List[str] = field(default_factory=list)
    
    # Human-friendly labels
    speaker_display_names: Dict[str, str] = field(default_factory=dict)
    speaker_roles: Dict[str, str] = field(default_factory=dict)
    
    # Processing metadata
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class SwapDetector:
    """Detects speaker label swaps through embedding neighborhood analysis"""
    
    def __init__(self,
                 neighborhood_size: int = 5,
                 discontinuity_threshold: float = 0.4,
                 min_block_duration: float = 5.0,
                 max_block_duration: float = 300.0,
                 contiguity_threshold: float = 0.7):
        """
        Initialize swap detector
        
        Args:
            neighborhood_size: Number of segments to consider for neighborhood analysis
            discontinuity_threshold: Threshold for embedding discontinuity detection
            min_block_duration: Minimum duration for swap block consideration
            max_block_duration: Maximum duration for swap block consideration  
            contiguity_threshold: Threshold for contiguous block detection
        """
        self.neighborhood_size = neighborhood_size
        self.discontinuity_threshold = discontinuity_threshold
        self.min_block_duration = min_block_duration
        self.max_block_duration = max_block_duration
        self.contiguity_threshold = contiguity_threshold
        
        self.logger = create_enhanced_logger("swap_detector")
    
    def detect_swaps(self, 
                    segments: List[FusedTranscriptSegment],
                    turn_embeddings: List[TurnEmbedding],
                    clustering_result: ClusteringResult) -> List[SwapCandidate]:
        """
        Detect speaker swap candidates using embedding analysis
        
        Args:
            segments: Fused transcript segments
            turn_embeddings: Turn embeddings for analysis
            clustering_result: Global speaker clustering result
            
        Returns:
            List of detected swap candidates
        """
        self.logger.info(f"Analyzing {len(segments)} segments for speaker swaps")
        
        swap_candidates = []
        
        # Create embedding lookup by segment
        embedding_lookup = self._create_embedding_lookup(segments, turn_embeddings)
        
        # Analyze each segment for potential swaps
        for i, segment in enumerate(segments):
            # Skip short segments and overlap regions
            if segment.duration < self.min_block_duration or segment.has_overlap:
                continue
            
            # Get embedding for this segment
            if i not in embedding_lookup:
                continue
            
            current_embedding = embedding_lookup[i]
            
            # Analyze embedding neighborhood
            discontinuity_score = self._compute_embedding_discontinuity(
                i, segments, embedding_lookup
            )
            
            if discontinuity_score > self.discontinuity_threshold:
                # Potential swap detected - analyze further
                swap_candidate = self._analyze_swap_candidate(
                    i, segment, segments, embedding_lookup, clustering_result, discontinuity_score
                )
                
                if swap_candidate and swap_candidate.swap_score > 0.5:
                    swap_candidates.append(swap_candidate)
        
        self.logger.info(f"Detected {len(swap_candidates)} swap candidates")
        return swap_candidates
    
    def _create_embedding_lookup(self, 
                               segments: List[FusedTranscriptSegment],
                               turn_embeddings: List[TurnEmbedding]) -> Dict[int, np.ndarray]:
        """Create lookup from segment index to embedding"""
        embedding_lookup = {}
        
        # Map turn embeddings to segments by time overlap
        for i, segment in enumerate(segments):
            best_embedding = None
            best_overlap = 0.0
            
            for turn in turn_embeddings:
                # Calculate time overlap
                overlap_start = max(segment.start_time, turn.turn_start)
                overlap_end = min(segment.end_time, turn.turn_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap and overlap > 0.1:  # At least 100ms overlap
                    best_embedding = turn.embedding
                    best_overlap = overlap
            
            if best_embedding is not None:
                embedding_lookup[i] = best_embedding
        
        return embedding_lookup
    
    def _compute_embedding_discontinuity(self,
                                       segment_idx: int,
                                       segments: List[FusedTranscriptSegment],
                                       embedding_lookup: Dict[int, np.ndarray]) -> float:
        """Compute embedding discontinuity score for a segment"""
        if segment_idx not in embedding_lookup:
            return 0.0
        
        current_embedding = embedding_lookup[segment_idx]
        current_speaker = segments[segment_idx].speaker_id
        
        # Get neighborhood embeddings
        neighborhood_distances = []
        
        # Look backward
        for i in range(max(0, segment_idx - self.neighborhood_size), segment_idx):
            if (i in embedding_lookup and 
                segments[i].speaker_id == current_speaker and
                not segments[i].has_overlap):
                
                distance = cosine(current_embedding, embedding_lookup[i])
                neighborhood_distances.append(distance)
        
        # Look forward
        for i in range(segment_idx + 1, min(len(segments), segment_idx + self.neighborhood_size + 1)):
            if (i in embedding_lookup and 
                segments[i].speaker_id == current_speaker and
                not segments[i].has_overlap):
                
                distance = cosine(current_embedding, embedding_lookup[i])
                neighborhood_distances.append(distance)
        
        if len(neighborhood_distances) < 2:
            return 0.0
        
        # Compute discontinuity as relative increase in distance
        avg_distance = np.mean(neighborhood_distances)
        max_distance = np.max(neighborhood_distances)
        
        # Discontinuity score is normalized deviation from typical distance
        discontinuity = (max_distance - avg_distance) / (avg_distance + 1e-8)
        
        return np.clip(discontinuity, 0.0, 1.0)
    
    def _analyze_swap_candidate(self,
                              segment_idx: int,
                              segment: FusedTranscriptSegment,
                              segments: List[FusedTranscriptSegment],
                              embedding_lookup: Dict[int, np.ndarray],
                              clustering_result: ClusteringResult,
                              discontinuity_score: float) -> Optional[SwapCandidate]:
        """Analyze a potential swap candidate in detail"""
        
        # Find contiguous block with same local speaker
        block_start, block_end = self._find_contiguous_block(
            segment_idx, segments, segment.speaker_id
        )
        
        block_duration = segments[block_end].end_time - segments[block_start].start_time
        
        # Check if block meets duration criteria
        if block_duration < self.min_block_duration or block_duration > self.max_block_duration:
            return None
        
        # Compute neighborhood change score
        neighborhood_score = self._compute_neighborhood_change_score(
            block_start, block_end, segments, embedding_lookup, clustering_result
        )
        
        # Compute contiguity score
        contiguity_score = self._compute_contiguity_score(
            block_start, block_end, segments
        )
        
        # Find best alternative speaker assignment
        proposed_speaker = self._find_best_alternative_speaker(
            block_start, block_end, segments, embedding_lookup, clustering_result
        )
        
        if proposed_speaker is None:
            return None
        
        # Compute cluster cohesion improvement
        cohesion_improvement = self._estimate_cohesion_improvement(
            block_start, block_end, segments, embedding_lookup, 
            clustering_result, segment.speaker_id, proposed_speaker
        )
        
        swap_candidate = SwapCandidate(
            start_time=segments[block_start].start_time,
            end_time=segments[block_end].end_time,
            duration=block_duration,
            original_speaker=segment.speaker_id,
            proposed_speaker=proposed_speaker,
            local_speaker_id=segment.speaker_id,
            embedding_discontinuity_score=discontinuity_score,
            neighborhood_change_score=neighborhood_score,
            contiguity_score=contiguity_score,
            cluster_cohesion_improvement=cohesion_improvement,
            affected_segments=list(range(block_start, block_end + 1))
        )
        
        return swap_candidate
    
    def _find_contiguous_block(self,
                             start_idx: int,
                             segments: List[FusedTranscriptSegment],
                             speaker_id: str) -> Tuple[int, int]:
        """Find contiguous block of segments with same speaker"""
        
        # Expand backward
        block_start = start_idx
        while (block_start > 0 and 
               segments[block_start - 1].speaker_id == speaker_id and
               not segments[block_start - 1].has_overlap):
            block_start -= 1
        
        # Expand forward
        block_end = start_idx
        while (block_end < len(segments) - 1 and
               segments[block_end + 1].speaker_id == speaker_id and
               not segments[block_end + 1].has_overlap):
            block_end += 1
        
        return block_start, block_end
    
    def _compute_neighborhood_change_score(self,
                                         block_start: int,
                                         block_end: int,
                                         segments: List[FusedTranscriptSegment],
                                         embedding_lookup: Dict[int, np.ndarray],
                                         clustering_result: ClusteringResult) -> float:
        """Compute how much the speaker's neighborhood changes in this block"""
        
        # Get embeddings for the block
        block_embeddings = []
        for i in range(block_start, block_end + 1):
            if i in embedding_lookup:
                block_embeddings.append(embedding_lookup[i])
        
        if len(block_embeddings) < 2:
            return 0.0
        
        # Compare with typical embeddings for this speaker
        speaker_id = segments[block_start].speaker_id
        typical_embeddings = []
        
        for i, segment in enumerate(segments):
            if (i < block_start or i > block_end) and segment.speaker_id == speaker_id:
                if i in embedding_lookup and not segment.has_overlap:
                    typical_embeddings.append(embedding_lookup[i])
        
        if len(typical_embeddings) < 2:
            return 0.0
        
        # Compute average distance within block vs. to typical embeddings
        block_centroid = np.mean(block_embeddings, axis=0)
        typical_centroid = np.mean(typical_embeddings, axis=0)
        
        centroid_distance = cosine(block_centroid, typical_centroid)
        
        # Compute intra-block vs. cross-block consistency
        intra_block_distances = []
        for emb in block_embeddings:
            intra_block_distances.append(cosine(emb, block_centroid))
        
        cross_block_distances = []
        for emb in block_embeddings:
            cross_block_distances.append(cosine(emb, typical_centroid))
        
        intra_block_consistency = 1.0 - np.mean(intra_block_distances)
        cross_block_consistency = 1.0 - np.mean(cross_block_distances)
        
        # Neighborhood change score is relative difference
        neighborhood_score = max(0, cross_block_consistency - intra_block_consistency)
        
        return np.clip(neighborhood_score + centroid_distance * 0.5, 0.0, 1.0)
    
    def _compute_contiguity_score(self,
                                block_start: int,
                                block_end: int,
                                segments: List[FusedTranscriptSegment]) -> float:
        """Compute how contiguous and consistent the block is"""
        
        block_segments = segments[block_start:block_end + 1]
        
        # Check temporal contiguity (gaps between segments)
        gaps = []
        for i in range(len(block_segments) - 1):
            gap = block_segments[i + 1].start_time - block_segments[i].end_time
            gaps.append(max(0, gap))
        
        avg_gap = np.mean(gaps) if gaps else 0.0
        gap_penalty = min(avg_gap / 5.0, 1.0)  # Penalize gaps > 5 seconds
        
        # Check speaker consistency
        speaker_id = block_segments[0].speaker_id
        consistent_segments = sum(1 for seg in block_segments if seg.speaker_id == speaker_id)
        speaker_consistency = consistent_segments / len(block_segments)
        
        # Check confidence consistency
        confidences = [seg.confidence for seg in block_segments]
        confidence_std = np.std(confidences)
        confidence_consistency = 1.0 - min(confidence_std, 1.0)
        
        contiguity_score = (
            (1.0 - gap_penalty) * 0.4 +
            speaker_consistency * 0.4 +
            confidence_consistency * 0.2
        )
        
        return np.clip(contiguity_score, 0.0, 1.0)
    
    def _find_best_alternative_speaker(self,
                                     block_start: int,
                                     block_end: int,
                                     segments: List[FusedTranscriptSegment],
                                     embedding_lookup: Dict[int, np.ndarray],
                                     clustering_result: ClusteringResult) -> Optional[str]:
        """Find best alternative global speaker for the block"""
        
        # Get block embeddings
        block_embeddings = []
        for i in range(block_start, block_end + 1):
            if i in embedding_lookup:
                block_embeddings.append(embedding_lookup[i])
        
        if not block_embeddings:
            return None
        
        block_centroid = np.mean(block_embeddings, axis=0)
        
        # Compare with each global speaker cluster
        best_speaker = None
        best_distance = float('inf')
        
        for cluster in clustering_result.clusters:
            distance = cosine(block_centroid, cluster.centroid_embedding)
            
            # Skip if this is the current speaker
            current_speaker = segments[block_start].speaker_id
            if cluster.global_speaker_id in clustering_result.local_to_global_mapping.values():
                mapped_locals = [k for k, v in clustering_result.local_to_global_mapping.items() 
                               if v == cluster.global_speaker_id]
                if current_speaker in mapped_locals:
                    continue
            
            if distance < best_distance:
                best_distance = distance
                best_speaker = cluster.global_speaker_id
        
        # Only return if distance is reasonable
        if best_distance < 0.6:  # Configurable threshold
            return best_speaker
        
        return None
    
    def _estimate_cohesion_improvement(self,
                                     block_start: int,
                                     block_end: int,
                                     segments: List[FusedTranscriptSegment],
                                     embedding_lookup: Dict[int, np.ndarray],
                                     clustering_result: ClusteringResult,
                                     original_speaker: str,
                                     proposed_speaker: str) -> float:
        """Estimate cluster cohesion improvement from proposed swap"""
        
        # Get block embeddings
        block_embeddings = []
        for i in range(block_start, block_end + 1):
            if i in embedding_lookup:
                block_embeddings.append(embedding_lookup[i])
        
        if not block_embeddings:
            return 0.0
        
        # Find original and proposed clusters
        original_cluster = None
        proposed_cluster = None
        
        for cluster in clustering_result.clusters:
            if cluster.global_speaker_id == clustering_result.local_to_global_mapping.get(original_speaker):
                original_cluster = cluster
            if cluster.global_speaker_id == proposed_speaker:
                proposed_cluster = cluster
        
        if not original_cluster or not proposed_cluster:
            return 0.0
        
        # Compute current cohesion (distance from block to original cluster)
        current_cohesion = []
        for emb in block_embeddings:
            current_cohesion.append(cosine(emb, original_cluster.centroid_embedding))
        
        # Compute proposed cohesion (distance from block to proposed cluster)
        proposed_cohesion = []
        for emb in block_embeddings:
            proposed_cohesion.append(cosine(emb, proposed_cluster.centroid_embedding))
        
        current_avg_distance = np.mean(current_cohesion)
        proposed_avg_distance = np.mean(proposed_cohesion)
        
        # Improvement is reduction in average distance
        improvement = max(0, current_avg_distance - proposed_avg_distance)
        
        return improvement

class SpeakerRelabeler:
    """
    Speaker relabeling engine for global ID assignment and swap correction
    
    Handles the final stage of long-horizon speaker tracking by applying
    global speaker mappings and detecting/correcting speaker label swaps.
    """
    
    def __init__(self,
                 swap_detector: Optional[SwapDetector] = None,
                 swap_correction_threshold: float = 0.6,
                 enable_overlap_preservation: bool = True,
                 enable_human_friendly_names: bool = True,
                 confidence_threshold: float = 0.5):
        """
        Initialize speaker relabeler
        
        Args:
            swap_detector: Swap detector instance
            swap_correction_threshold: Threshold for applying swap corrections
            enable_overlap_preservation: Preserve overlap regions with multiple speakers
            enable_human_friendly_names: Generate human-friendly speaker names
            confidence_threshold: Minimum confidence for relabeling operations
        """
        self.swap_detector = swap_detector or SwapDetector()
        self.swap_correction_threshold = swap_correction_threshold
        self.enable_overlap_preservation = enable_overlap_preservation
        self.enable_human_friendly_names = enable_human_friendly_names
        self.confidence_threshold = confidence_threshold
        
        # Human-friendly naming
        self.speaker_name_templates = [
            "Speaker A", "Speaker B", "Speaker C", "Speaker D", "Speaker E",
            "Speaker F", "Speaker G", "Speaker H", "Speaker I", "Speaker J"
        ]
        
        self.logger = create_enhanced_logger("speaker_relabeler")
        
        self.logger.info("Speaker relabeler initialized",
                        context={
                            'swap_correction_threshold': swap_correction_threshold,
                            'overlap_preservation': enable_overlap_preservation,
                            'human_friendly_names': enable_human_friendly_names
                        })
    
    @trace_stage("relabel_speakers")
    def relabel_speakers(self,
                        fusion_result: OverlapFusionResult,
                        clustering_result: ClusteringResult,
                        turn_embeddings: List[TurnEmbedding],
                        session_id: str) -> RelabelingResult:
        """
        Complete speaker relabeling pipeline
        
        Args:
            fusion_result: Overlap fusion result to relabel
            clustering_result: Global speaker clustering result
            turn_embeddings: Turn embeddings for swap detection
            session_id: Session identifier
            
        Returns:
            Relabeling result with corrected speaker assignments
        """
        start_time = time.time()
        
        self.logger.info(f"Starting speaker relabeling for session {session_id}")
        
        # Step 1: Apply global speaker mapping
        relabeled_segments = self._apply_global_mapping(
            fusion_result.fused_segments, clustering_result.local_to_global_mapping
        )
        
        # Step 2: Detect potential swaps
        swap_start_time = time.time()
        swap_candidates = self.swap_detector.detect_swaps(
            relabeled_segments, turn_embeddings, clustering_result
        )
        swap_detection_time = time.time() - swap_start_time
        
        # Step 3: Validate and apply swap corrections
        swaps_corrected = self._apply_swap_corrections(
            relabeled_segments, swap_candidates, clustering_result
        )
        
        # Step 4: Preserve overlap regions
        if self.enable_overlap_preservation:
            self._preserve_overlap_regions(relabeled_segments, clustering_result)
        
        # Step 5: Generate human-friendly names
        speaker_display_names, speaker_roles = self._generate_speaker_names(
            clustering_result.clusters
        )
        
        # Step 6: Compute consistency metrics
        consistency_metrics = self._compute_consistency_metrics(
            relabeled_segments, clustering_result
        )
        
        # Step 7: Validate results
        validation_result = self._validate_relabeling_result(
            relabeled_segments, clustering_result, consistency_metrics
        )
        
        processing_time = time.time() - start_time
        
        result = RelabelingResult(
            session_id=session_id,
            original_fusion_result=fusion_result,
            relabeled_segments=relabeled_segments,
            global_speaker_mapping=clustering_result.local_to_global_mapping,
            swap_candidates_detected=swap_candidates,
            swaps_corrected=swaps_corrected,
            swap_detection_time=swap_detection_time,
            speaker_consistency_score=consistency_metrics['speaker_consistency'],
            temporal_consistency_score=consistency_metrics['temporal_consistency'],
            overlap_preservation_score=consistency_metrics['overlap_preservation'],
            segments_relabeled=len(relabeled_segments),
            swaps_corrected_count=len(swaps_corrected),
            der_improvement_estimate=consistency_metrics.get('der_improvement', 0.0),
            validation_passed=validation_result['passed'],
            validation_issues=validation_result['issues'],
            speaker_display_names=speaker_display_names,
            speaker_roles=speaker_roles,
            processing_time=processing_time,
            metadata={
                'consistency_metrics': consistency_metrics,
                'validation_result': validation_result,
                'swap_detection_stats': {
                    'candidates_detected': len(swap_candidates),
                    'swaps_applied': len(swaps_corrected),
                    'detection_time': swap_detection_time
                }
            }
        )
        
        self.logger.info(f"Speaker relabeling completed in {processing_time:.2f}s",
                        context={
                            'segments_processed': len(relabeled_segments),
                            'swaps_detected': len(swap_candidates),
                            'swaps_corrected': len(swaps_corrected),
                            'validation_passed': validation_result['passed']
                        })
        
        return result
    
    def _apply_global_mapping(self,
                            segments: List[FusedTranscriptSegment],
                            local_to_global_mapping: Dict[str, str]) -> List[FusedTranscriptSegment]:
        """Apply global speaker mapping to segments"""
        
        relabeled_segments = []
        
        for segment in segments:
            # Create copy of segment
            new_segment = copy.deepcopy(segment)
            
            # Map speaker ID
            if segment.speaker_id in local_to_global_mapping:
                new_segment.speaker_id = local_to_global_mapping[segment.speaker_id]
            else:
                # Keep original if no mapping found
                self.logger.warning(f"No global mapping found for local speaker: {segment.speaker_id}")
            
            # Map overlap region speakers if present
            if segment.has_overlap and segment.overlap_regions:
                for overlap_region in new_segment.overlap_regions:
                    if overlap_region.primary_speaker in local_to_global_mapping:
                        overlap_region.primary_speaker = local_to_global_mapping[overlap_region.primary_speaker]
                    
                    if overlap_region.secondary_speaker in local_to_global_mapping:
                        overlap_region.secondary_speaker = local_to_global_mapping[overlap_region.secondary_speaker]
                
                # Update overlapping speakers list
                new_segment.overlapping_speakers = [
                    local_to_global_mapping.get(spk, spk) 
                    for spk in segment.overlapping_speakers
                ]
            
            # Map word-level speaker information
            for word in new_segment.words:
                if hasattr(word, 'unified_speaker_id') and word.unified_speaker_id in local_to_global_mapping:
                    word.unified_speaker_id = local_to_global_mapping[word.unified_speaker_id]
            
            relabeled_segments.append(new_segment)
        
        return relabeled_segments
    
    def _apply_swap_corrections(self,
                              segments: List[FusedTranscriptSegment],
                              swap_candidates: List[SwapCandidate],
                              clustering_result: ClusteringResult) -> List[SwapCandidate]:
        """Apply validated swap corrections to segments"""
        
        swaps_corrected = []
        
        # Sort swap candidates by confidence score (highest first)
        validated_swaps = [
            swap for swap in swap_candidates 
            if swap.swap_score > self.swap_correction_threshold
        ]
        validated_swaps.sort(key=lambda x: x.swap_score, reverse=True)
        
        for swap in validated_swaps:
            # Apply the swap
            swapped = False
            
            for seg_idx in swap.affected_segments:
                if seg_idx < len(segments):
                    segment = segments[seg_idx]
                    
                    if segment.speaker_id == swap.original_speaker:
                        # Apply swap
                        segment.speaker_id = swap.proposed_speaker
                        
                        # Update word-level assignments
                        for word in segment.words:
                            if hasattr(word, 'unified_speaker_id') and word.unified_speaker_id == swap.original_speaker:
                                word.unified_speaker_id = swap.proposed_speaker
                        
                        # Mark as relabeled in metadata
                        if 'relabeling_metadata' not in segment.metadata:
                            segment.metadata['relabeling_metadata'] = {}
                        
                        segment.metadata['relabeling_metadata']['swap_corrected'] = True
                        segment.metadata['relabeling_metadata']['original_speaker'] = swap.original_speaker
                        segment.metadata['relabeling_metadata']['swap_score'] = swap.swap_score
                        
                        swapped = True
            
            if swapped:
                swaps_corrected.append(swap)
                self.logger.info(f"Applied swap correction: {swap.original_speaker} -> {swap.proposed_speaker} "
                               f"({swap.start_time:.1f}s-{swap.end_time:.1f}s, score={swap.swap_score:.3f})")
        
        return swaps_corrected
    
    def _preserve_overlap_regions(self,
                                segments: List[FusedTranscriptSegment],
                                clustering_result: ClusteringResult):
        """Preserve overlap regions with multiple global speakers"""
        
        for segment in segments:
            if not segment.has_overlap or not segment.overlap_regions:
                continue
            
            # Ensure overlap region speakers are properly mapped
            for overlap_region in segment.overlap_regions:
                # Validate that speakers are different global speakers
                if overlap_region.primary_speaker == overlap_region.secondary_speaker:
                    # Same global speaker - this might be a mapping error
                    self.logger.warning(f"Overlap region has same primary and secondary speaker: "
                                      f"{overlap_region.primary_speaker}")
                    
                    # Try to preserve original distinction if possible
                    # This is a complex case that might require special handling
                    pass
                
                # Update overlap region metadata to reflect global mapping
                overlap_region.metadata['global_speakers_preserved'] = True
                overlap_region.metadata['primary_global'] = overlap_region.primary_speaker
                overlap_region.metadata['secondary_global'] = overlap_region.secondary_speaker
            
            # Update segment's overlapping speakers list to use global IDs
            unique_overlap_speakers = set()
            for overlap_region in segment.overlap_regions:
                unique_overlap_speakers.add(overlap_region.primary_speaker)
                unique_overlap_speakers.add(overlap_region.secondary_speaker)
            
            segment.overlapping_speakers = list(unique_overlap_speakers)
    
    def _generate_speaker_names(self, clusters: List[SpeakerCluster]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Generate human-friendly speaker names and roles"""
        
        display_names = {}
        speaker_roles = {}
        
        if not self.enable_human_friendly_names:
            return display_names, speaker_roles
        
        # Sort clusters by total speaking time for consistent naming
        sorted_clusters = sorted(clusters, key=lambda c: c.total_duration, reverse=True)
        
        for i, cluster in enumerate(sorted_clusters):
            global_id = cluster.global_speaker_id
            
            # Assign display name
            if i < len(self.speaker_name_templates):
                display_names[global_id] = self.speaker_name_templates[i]
            else:
                display_names[global_id] = f"Speaker {i + 1}"
            
            # Assign role based on speaking patterns (basic heuristic)
            if cluster.total_duration > 300:  # More than 5 minutes
                if i == 0:
                    speaker_roles[global_id] = "Primary Speaker"
                else:
                    speaker_roles[global_id] = "Frequent Participant"
            elif cluster.total_duration > 60:  # More than 1 minute
                speaker_roles[global_id] = "Active Participant"
            else:
                speaker_roles[global_id] = "Occasional Participant"
        
        return display_names, speaker_roles
    
    def _compute_consistency_metrics(self,
                                   segments: List[FusedTranscriptSegment],
                                   clustering_result: ClusteringResult) -> Dict[str, float]:
        """Compute consistency metrics for relabeling result"""
        
        metrics = {}
        
        # Speaker consistency: How consistent speaker assignments are
        speaker_transitions = 0
        total_transitions = 0
        
        for i in range(1, len(segments)):
            if segments[i].speaker_id != segments[i-1].speaker_id:
                speaker_transitions += 1
            total_transitions += 1
        
        if total_transitions > 0:
            transition_rate = speaker_transitions / total_transitions
            metrics['speaker_consistency'] = 1.0 - min(transition_rate, 1.0)
        else:
            metrics['speaker_consistency'] = 1.0
        
        # Temporal consistency: How well speakers maintain identity over time
        speaker_time_spans = defaultdict(list)
        
        for segment in segments:
            speaker_time_spans[segment.speaker_id].append((segment.start_time, segment.end_time))
        
        temporal_consistencies = []
        for speaker_id, time_spans in speaker_time_spans.items():
            if len(time_spans) > 1:
                # Sort by start time
                time_spans.sort()
                
                # Compute gaps between appearances
                gaps = []
                for i in range(1, len(time_spans)):
                    gap = time_spans[i][0] - time_spans[i-1][1]
                    gaps.append(max(0, gap))
                
                # Temporal consistency is inverse of average gap
                avg_gap = np.mean(gaps) if gaps else 0.0
                consistency = 1.0 / (1.0 + avg_gap / 60.0)  # Normalize by minutes
                temporal_consistencies.append(consistency)
        
        if temporal_consistencies:
            metrics['temporal_consistency'] = np.mean(temporal_consistencies)
        else:
            metrics['temporal_consistency'] = 1.0
        
        # Overlap preservation: How well overlap regions are preserved
        overlap_segments = [seg for seg in segments if seg.has_overlap]
        if overlap_segments:
            preserved_overlaps = sum(
                1 for seg in overlap_segments
                if seg.overlap_regions and len(set(
                    [region.primary_speaker for region in seg.overlap_regions] +
                    [region.secondary_speaker for region in seg.overlap_regions]
                )) > 1
            )
            
            metrics['overlap_preservation'] = preserved_overlaps / len(overlap_segments)
        else:
            metrics['overlap_preservation'] = 1.0
        
        # Estimate DER improvement (rough heuristic)
        # Based on cluster quality and consistency scores
        cluster_quality = clustering_result.overall_silhouette_score
        consistency_score = (metrics['speaker_consistency'] + metrics['temporal_consistency']) / 2.0
        
        # Simple heuristic: better clustering and consistency -> better DER
        der_improvement = (cluster_quality + consistency_score) * 0.2  # Up to 20% improvement
        metrics['der_improvement'] = der_improvement
        
        return metrics
    
    def _validate_relabeling_result(self,
                                  segments: List[FusedTranscriptSegment],
                                  clustering_result: ClusteringResult,
                                  consistency_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate relabeling result quality"""
        
        issues = []
        passed = True
        
        # Check for missing speaker assignments
        unassigned_segments = [seg for seg in segments if not seg.speaker_id]
        if unassigned_segments:
            issues.append(f"Found {len(unassigned_segments)} segments without speaker assignments")
            if len(unassigned_segments) > len(segments) * 0.1:  # More than 10%
                passed = False
        
        # Check consistency scores
        if consistency_metrics['speaker_consistency'] < 0.5:
            issues.append(f"Low speaker consistency: {consistency_metrics['speaker_consistency']:.3f}")
            if consistency_metrics['speaker_consistency'] < 0.3:
                passed = False
        
        if consistency_metrics['temporal_consistency'] < 0.4:
            issues.append(f"Low temporal consistency: {consistency_metrics['temporal_consistency']:.3f}")
        
        # Check for degenerate global speakers (too few segments)
        speaker_segment_counts = Counter(seg.speaker_id for seg in segments)
        min_segments_per_speaker = max(2, len(segments) // (len(clustering_result.clusters) * 10))
        
        sparse_speakers = [
            spk for spk, count in speaker_segment_counts.items()
            if count < min_segments_per_speaker
        ]
        
        if sparse_speakers:
            issues.append(f"Found {len(sparse_speakers)} speakers with very few segments")
        
        # Check overlap preservation
        if (self.enable_overlap_preservation and 
            consistency_metrics['overlap_preservation'] < 0.7):
            issues.append(f"Low overlap preservation: {consistency_metrics['overlap_preservation']:.3f}")
        
        return {
            'passed': passed,
            'issues': issues,
            'unassigned_segments': len(unassigned_segments),
            'sparse_speakers': len(sparse_speakers),
            'overall_quality_score': np.mean(list(consistency_metrics.values()))
        }
    
    def update_speaker_names(self,
                           relabeling_result: RelabelingResult,
                           name_mapping: Dict[str, str]) -> RelabelingResult:
        """Update speaker display names with user-provided mappings"""
        
        # Update display names
        for global_id, new_name in name_mapping.items():
            if global_id in relabeling_result.speaker_display_names:
                relabeling_result.speaker_display_names[global_id] = new_name
        
        # Update metadata
        relabeling_result.metadata['name_updates'] = name_mapping
        relabeling_result.metadata['name_update_time'] = time.time()
        
        self.logger.info(f"Updated speaker names for {len(name_mapping)} speakers")
        
        return relabeling_result
    
    def assign_speaker_roles(self,
                           relabeling_result: RelabelingResult,
                           role_mapping: Dict[str, str]) -> RelabelingResult:
        """Assign roles to speakers (e.g., Moderator, Participant, etc.)"""
        
        # Update speaker roles
        for global_id, role in role_mapping.items():
            relabeling_result.speaker_roles[global_id] = role
        
        # Update metadata
        relabeling_result.metadata['role_assignments'] = role_mapping
        relabeling_result.metadata['role_assignment_time'] = time.time()
        
        self.logger.info(f"Assigned roles for {len(role_mapping)} speakers")
        
        return relabeling_result