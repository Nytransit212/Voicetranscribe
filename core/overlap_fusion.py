"""
Overlap Fusion Engine for Timeline Alignment and Cross-Channel Reconciliation

This module performs timeline alignment of ASR outputs from multiple separated stems 
back to a unified timeline with explicit overlap regions. It handles cross-channel
reconciliation using acoustic confidence, cross-engine agreement, and fusion priors.
When stems disagree on words, it uses the highest joint score and retains dual-speaker
labels where word timing collides.

Key Features:
- Multi-stem ASR output alignment to unified timeline
- Cross-channel word-level reconciliation with confidence scoring
- Dual-speaker label retention for simultaneous speech
- Acoustic confidence-based conflict resolution
- Fusion priors and glossary-based scoring
- Overlap region metadata preservation

Author: Advanced Ensemble Transcription System
"""

import numpy as np
import time
import re
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path

from core.overlap_diarizer import OverlapDiarizationResult, StemDiarizationSegment, CrossStemOverlapRegion
from core.asr_providers.base import ASRResult, ASRSegment
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import trace_stage, track_cost
from utils.intelligent_cache import cached_operation

@dataclass
class WordAlignment:
    """Word-level alignment across stems with timing and confidence"""
    word: str
    start_time: float
    end_time: float
    duration: float
    
    # Source information
    stem_id: str
    speaker_id: str
    unified_speaker_id: str
    
    # Confidence and scoring
    acoustic_confidence: float
    fusion_score: float
    joint_score: float
    cross_engine_agreement: float = 0.0
    
    # Overlap metadata
    is_overlapped: bool = False
    competing_words: List[str] = field(default_factory=list)
    competing_stems: List[str] = field(default_factory=list)
    
    # Processing metadata
    asr_provider: str = "unknown"
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OverlapRegion:
    """Explicit overlap region with dual-speaker information"""
    start_time: float
    end_time: float
    duration: float
    
    # Speaker and content information
    primary_speaker: str
    secondary_speaker: str
    primary_words: List[WordAlignment]
    secondary_words: List[WordAlignment]
    
    # Confidence and quality metrics
    overlap_confidence: float
    reconciliation_confidence: float
    conflict_resolution_method: str
    
    # Metadata
    stems_involved: List[str]
    cross_stem_overlap_regions: List[CrossStemOverlapRegion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusedTranscriptSegment:
    """Unified transcript segment with overlap handling"""
    start_time: float
    end_time: float
    duration: float
    
    # Primary content
    text: str
    speaker_id: str
    confidence: float
    words: List[WordAlignment]
    
    # Overlap information
    has_overlap: bool = False
    overlap_regions: List[OverlapRegion] = field(default_factory=list)
    overlapping_speakers: List[str] = field(default_factory=list)
    
    # Processing metadata
    fusion_method: str = "single_stem"
    reconciliation_conflicts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OverlapFusionResult:
    """Complete result from overlap-aware fusion processing"""
    original_overlap_frame: Any  # OverlapFrame from source separation
    diarization_result: OverlapDiarizationResult
    
    # Fusion outputs
    fused_segments: List[FusedTranscriptSegment]
    explicit_overlap_regions: List[OverlapRegion]
    unified_transcript: str
    
    # Processing metrics
    total_words_processed: int
    reconciliation_conflicts: int
    overlap_regions_count: int
    dual_speaker_regions_count: int
    
    # Quality metrics
    overall_confidence: float
    fusion_confidence: float
    reconciliation_conflict_rate: float
    
    # Timing
    processing_time: float
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class GlossaryPriorScorer:
    """Provides glossary-based prior scoring for word fusion"""
    
    def __init__(self,
                 domain_glossary: Optional[Dict[str, float]] = None,
                 enable_meeting_vocabulary: bool = True,
                 enable_technical_terms: bool = True):
        """
        Initialize glossary prior scorer
        
        Args:
            domain_glossary: Domain-specific vocabulary with prior scores
            enable_meeting_vocabulary: Enable meeting-specific vocabulary
            enable_technical_terms: Enable technical terminology recognition
        """
        self.domain_glossary = domain_glossary or {}
        self.enable_meeting_vocabulary = enable_meeting_vocabulary
        self.enable_technical_terms = enable_technical_terms
        
        self.logger = create_enhanced_logger("glossary_prior_scorer")
        
        # Initialize built-in vocabularies
        self.meeting_vocabulary = self._load_meeting_vocabulary()
        self.technical_terms = self._load_technical_terms()
        
    def _load_meeting_vocabulary(self) -> Dict[str, float]:
        """Load meeting-specific vocabulary with prior scores"""
        return {
            # Meeting actions and processes
            'meeting': 1.5, 'agenda': 1.4, 'minutes': 1.3, 'action': 1.3,
            'decision': 1.4, 'follow': 1.2, 'followup': 1.2, 'next': 1.1,
            'schedule': 1.3, 'calendar': 1.2, 'deadline': 1.3,
            
            # Discussion terms
            'discuss': 1.2, 'review': 1.2, 'consider': 1.1, 'propose': 1.3,
            'suggest': 1.2, 'recommend': 1.2, 'approve': 1.3, 'agree': 1.2,
            'disagree': 1.2, 'concern': 1.2, 'issue': 1.1, 'problem': 1.1,
            
            # Business terms
            'project': 1.2, 'budget': 1.3, 'timeline': 1.3, 'resource': 1.2,
            'stakeholder': 1.4, 'client': 1.2, 'customer': 1.2, 'vendor': 1.2,
            'deliverable': 1.4, 'milestone': 1.3, 'phase': 1.2, 'scope': 1.3,
            
            # Communication terms
            'email': 1.1, 'call': 1.1, 'presentation': 1.2, 'report': 1.2,
            'document': 1.1, 'share': 1.1, 'update': 1.1, 'status': 1.2,
        }
    
    def _load_technical_terms(self) -> Dict[str, float]:
        """Load technical terminology with prior scores"""
        return {
            # Technology terms
            'api': 1.4, 'database': 1.3, 'server': 1.2, 'cloud': 1.2,
            'security': 1.3, 'authentication': 1.4, 'authorization': 1.4,
            'encryption': 1.4, 'backup': 1.2, 'restore': 1.2,
            
            # Development terms
            'development': 1.2, 'testing': 1.2, 'deployment': 1.3, 'release': 1.2,
            'version': 1.1, 'feature': 1.1, 'bug': 1.2, 'fix': 1.1,
            'update': 1.1, 'upgrade': 1.2, 'maintenance': 1.2,
            
            # Process terms
            'requirement': 1.2, 'specification': 1.3, 'design': 1.1, 'architecture': 1.3,
            'implementation': 1.3, 'integration': 1.3, 'interface': 1.2, 'workflow': 1.2,
        }
    
    def calculate_word_prior(self, word: str, context: Optional[List[str]] = None) -> float:
        """
        Calculate glossary prior score for a word
        
        Args:
            word: Word to score
            context: Surrounding words for context
            
        Returns:
            Prior score (1.0 = neutral, >1.0 = positive prior, <1.0 = negative prior)
        """
        if not word or not word.strip():
            return 1.0
        
        normalized_word = word.lower().strip()
        prior_score = 1.0
        
        # Check domain glossary first
        if normalized_word in self.domain_glossary:
            prior_score *= self.domain_glossary[normalized_word]
        
        # Check meeting vocabulary
        if self.enable_meeting_vocabulary and normalized_word in self.meeting_vocabulary:
            prior_score *= self.meeting_vocabulary[normalized_word]
        
        # Check technical terms
        if self.enable_technical_terms and normalized_word in self.technical_terms:
            prior_score *= self.technical_terms[normalized_word]
        
        # Context-based adjustments
        if context:
            context_bonus = self._calculate_context_bonus(normalized_word, context)
            prior_score *= context_bonus
        
        return prior_score
    
    def _calculate_context_bonus(self, word: str, context: List[str]) -> float:
        """Calculate context-based bonus for word prior"""
        
        # Simple context patterns
        context_lower = [w.lower() for w in context if w]
        
        # Meeting context patterns
        if word in ['minutes', 'agenda', 'action'] and any(w in ['meeting', 'call'] for w in context_lower):
            return 1.2
        
        # Technical context patterns
        if word in ['api', 'database', 'server'] and any(w in ['system', 'application', 'platform'] for w in context_lower):
            return 1.2
        
        # Business context patterns
        if word in ['budget', 'timeline', 'resource'] and any(w in ['project', 'plan', 'management'] for w in context_lower):
            return 1.2
        
        return 1.0

class CrossEngineAgreementScorer:
    """Calculates cross-engine agreement scores for word-level fusion"""
    
    def __init__(self,
                 agreement_threshold: float = 0.7,
                 partial_agreement_bonus: float = 0.3):
        """
        Initialize cross-engine agreement scorer
        
        Args:
            agreement_threshold: Threshold for considering words as agreeing
            partial_agreement_bonus: Bonus for partial agreement (similar words)
        """
        self.agreement_threshold = agreement_threshold
        self.partial_agreement_bonus = partial_agreement_bonus
        
        self.logger = create_enhanced_logger("cross_engine_agreement_scorer")
    
    def calculate_agreement_score(self,
                                 word: str,
                                 competing_words: List[str],
                                 word_confidences: List[float]) -> float:
        """
        Calculate cross-engine agreement score for a word
        
        Args:
            word: Primary word to score
            competing_words: Alternative words from other engines/stems
            word_confidences: Confidence scores for competing words
            
        Returns:
            Agreement score (0.0 to 1.0)
        """
        if not competing_words:
            return 0.5  # Neutral when no alternatives
        
        # Exact matches
        exact_matches = sum(1 for w in competing_words if w.lower() == word.lower())
        if exact_matches >= len(competing_words) * self.agreement_threshold:
            return 1.0
        
        # Partial matches (phonetic/edit distance similarity)
        partial_matches = 0
        for competing_word in competing_words:
            similarity = self._calculate_word_similarity(word, competing_word)
            if similarity >= self.agreement_threshold:
                partial_matches += 1
        
        if partial_matches >= len(competing_words) * self.agreement_threshold:
            return 0.8 + self.partial_agreement_bonus
        
        # Weighted agreement based on confidences
        total_confidence = sum(word_confidences) if word_confidences else 1.0
        agreement_score = 0.0
        
        for i, competing_word in enumerate(competing_words):
            similarity = self._calculate_word_similarity(word, competing_word)
            confidence_weight = word_confidences[i] / total_confidence if word_confidences and i < len(word_confidences) else 1.0 / len(competing_words)
            agreement_score += similarity * confidence_weight
        
        return max(0.0, min(1.0, agreement_score))
    
    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words using edit distance"""
        if not word1 or not word2:
            return 0.0
        
        if word1.lower() == word2.lower():
            return 1.0
        
        # Simple edit distance calculation
        w1, w2 = word1.lower(), word2.lower()
        
        # Create matrix for dynamic programming
        rows = len(w1) + 1
        cols = len(w2) + 1
        dist = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Initialize first row and column
        for i in range(1, rows):
            dist[i][0] = i
        for j in range(1, cols):
            dist[0][j] = j
        
        # Fill the matrix
        for i in range(1, rows):
            for j in range(1, cols):
                if w1[i-1] == w2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                dist[i][j] = min(
                    dist[i-1][j] + 1,      # deletion
                    dist[i][j-1] + 1,      # insertion
                    dist[i-1][j-1] + cost  # substitution
                )
        
        # Convert edit distance to similarity
        max_len = max(len(w1), len(w2))
        if max_len == 0:
            return 1.0
        
        edit_distance = dist[rows-1][cols-1]
        similarity = 1.0 - (edit_distance / max_len)
        
        return max(0.0, similarity)

class TimelineAligner:
    """Aligns multi-stem ASR outputs to unified timeline"""
    
    def __init__(self,
                 time_tolerance: float = 0.1,
                 overlap_merge_threshold: float = 0.05):
        """
        Initialize timeline aligner
        
        Args:
            time_tolerance: Tolerance for aligning overlapping segments (seconds)
            overlap_merge_threshold: Threshold for merging nearby overlaps (seconds)
        """
        self.time_tolerance = time_tolerance
        self.overlap_merge_threshold = overlap_merge_threshold
        
        self.logger = create_enhanced_logger("timeline_aligner")
    
    @trace_stage("timeline_alignment")
    def align_stem_transcriptions(self,
                                stem_transcriptions: List[Any],  # List[StemTranscription]
                                unified_speaker_map: Dict[str, str]) -> List[WordAlignment]:
        """
        Align ASR transcriptions from multiple stems to unified timeline
        
        Args:
            stem_transcriptions: Transcription results from all stems
            unified_speaker_map: Mapping from stem speakers to unified speakers
            
        Returns:
            List of word alignments across all stems
        """
        if not stem_transcriptions:
            return []
        
        self.logger.info(f"Aligning transcriptions from {len(stem_transcriptions)} stems")
        
        word_alignments = []
        
        # Extract word-level alignments from each stem
        for stem_transcription in stem_transcriptions:
            stem_words = self._extract_stem_words(stem_transcription, unified_speaker_map)
            word_alignments.extend(stem_words)
        
        # Sort by start time for timeline processing
        word_alignments.sort(key=lambda w: w.start_time)
        
        self.logger.info(f"Extracted {len(word_alignments)} word alignments for timeline processing")
        
        return word_alignments
    
    def _extract_stem_words(self,
                          stem_transcription: Any,  # StemTranscription
                          unified_speaker_map: Dict[str, str]) -> List[WordAlignment]:
        """Extract word-level alignments from stem transcription"""
        
        stem_words = []
        
        # Get the ASR result from the stem transcription
        asr_result = stem_transcription.asr_result
        stem_id = stem_transcription.stem.speaker_id
        
        # Extract unified speaker ID
        stem_speaker_key = f"{stem_id}_{getattr(stem_transcription, 'speaker_id', 'unknown')}"
        unified_speaker_id = unified_speaker_map.get(stem_speaker_key, 'unknown')
        
        # Process word-level alignments if available
        if hasattr(asr_result, 'word_segments') and asr_result.word_segments:
            for word_segment in asr_result.word_segments:
                if hasattr(word_segment, 'word') and hasattr(word_segment, 'start') and hasattr(word_segment, 'end'):
                    word_alignment = WordAlignment(
                        word=word_segment.word,
                        start_time=word_segment.start,
                        end_time=word_segment.end,
                        duration=word_segment.end - word_segment.start,
                        stem_id=stem_id,
                        speaker_id=getattr(stem_transcription, 'speaker_id', 'unknown'),
                        unified_speaker_id=unified_speaker_id,
                        acoustic_confidence=getattr(word_segment, 'confidence', asr_result.calibrated_confidence),
                        fusion_score=0.0,  # Will be calculated later
                        joint_score=0.0,   # Will be calculated later
                        asr_provider=asr_result.provider,
                        processing_metadata={
                            'stem_confidence': stem_transcription.stem.confidence,
                            'attribution_confidence': stem_transcription.attribution_confidence
                        }
                    )
                    stem_words.append(word_alignment)
        else:
            # Fall back to sentence-level processing
            # This is a simplified approach when word-level timing isn't available
            full_text = asr_result.full_text
            if full_text and full_text.strip():
                # Estimate word timing by distributing duration across words
                words = full_text.split()
                if words and hasattr(asr_result, 'segments') and asr_result.segments:
                    # Use first segment for timing estimate
                    segment = asr_result.segments[0]
                    segment_duration = segment.end - segment.start
                    word_duration = segment_duration / len(words)
                    
                    for i, word in enumerate(words):
                        start_time = segment.start + (i * word_duration)
                        end_time = start_time + word_duration
                        
                        word_alignment = WordAlignment(
                            word=word,
                            start_time=start_time,
                            end_time=end_time,
                            duration=word_duration,
                            stem_id=stem_id,
                            speaker_id=getattr(stem_transcription, 'speaker_id', 'unknown'),
                            unified_speaker_id=unified_speaker_id,
                            acoustic_confidence=asr_result.calibrated_confidence,
                            fusion_score=0.0,
                            joint_score=0.0,
                            asr_provider=asr_result.provider,
                            processing_metadata={
                                'estimated_timing': True,
                                'stem_confidence': stem_transcription.stem.confidence,
                                'attribution_confidence': stem_transcription.attribution_confidence
                            }
                        )
                        stem_words.append(word_alignment)
        
        return stem_words

class OverlapFusionEngine:
    """
    Main engine for overlap-aware fusion of multi-stem ASR outputs
    """
    
    def __init__(self,
                 glossary_scorer: Optional[GlossaryPriorScorer] = None,
                 agreement_scorer: Optional[CrossEngineAgreementScorer] = None,
                 timeline_aligner: Optional[TimelineAligner] = None,
                 fusion_score_weights: Optional[Dict[str, float]] = None,
                 conflict_resolution_strategy: str = "highest_joint_score"):
        """
        Initialize overlap fusion engine
        
        Args:
            glossary_scorer: Glossary prior scorer instance
            agreement_scorer: Cross-engine agreement scorer instance
            timeline_aligner: Timeline aligner instance
            fusion_score_weights: Weights for fusion score components
            conflict_resolution_strategy: Strategy for resolving word conflicts
        """
        self.glossary_scorer = glossary_scorer or GlossaryPriorScorer()
        self.agreement_scorer = agreement_scorer or CrossEngineAgreementScorer()
        self.timeline_aligner = timeline_aligner or TimelineAligner()
        
        # Default fusion score weights
        self.fusion_score_weights = fusion_score_weights or {
            'acoustic_confidence': 0.4,
            'cross_engine_agreement': 0.3,
            'glossary_prior': 0.2,
            'attribution_confidence': 0.1
        }
        
        self.conflict_resolution_strategy = conflict_resolution_strategy
        
        self.logger = create_enhanced_logger("overlap_fusion_engine")
    
    @trace_stage("overlap_fusion_processing")
    def fuse_overlap_transcriptions(self,
                                  diarization_result: OverlapDiarizationResult,
                                  stem_transcriptions: List[Any]) -> OverlapFusionResult:  # List[StemTranscription]
        """
        Fuse multi-stem transcriptions with overlap-aware processing
        
        Args:
            diarization_result: Result from overlap-aware diarization
            stem_transcriptions: ASR transcriptions from all stems
            
        Returns:
            Complete overlap fusion result
        """
        start_time = time.time()
        
        self.logger.info("Starting overlap-aware fusion processing",
                        context={
                            'num_stems': len(stem_transcriptions),
                            'cross_stem_overlaps': len(diarization_result.cross_stem_overlaps),
                            'unified_speakers': len(set(diarization_result.unified_speaker_map.values()))
                        })
        
        if not stem_transcriptions:
            return self._create_empty_fusion_result(diarization_result, time.time() - start_time)
        
        # Step 1: Align transcriptions to unified timeline
        word_alignments = self.timeline_aligner.align_stem_transcriptions(
            stem_transcriptions, diarization_result.unified_speaker_map
        )
        
        if not word_alignments:
            return self._create_empty_fusion_result(diarization_result, time.time() - start_time)
        
        # Step 2: Calculate fusion scores for all words
        self._calculate_fusion_scores(word_alignments)
        
        # Step 3: Detect and resolve word-level conflicts
        reconciliation_conflicts = self._resolve_word_conflicts(word_alignments)
        
        # Step 4: Create explicit overlap regions
        explicit_overlap_regions = self._create_explicit_overlap_regions(
            word_alignments, diarization_result.cross_stem_overlaps
        )
        
        # Step 5: Generate fused transcript segments
        fused_segments = self._generate_fused_segments(
            word_alignments, explicit_overlap_regions, reconciliation_conflicts
        )
        
        # Step 6: Create unified transcript text
        unified_transcript = self._generate_unified_transcript(fused_segments)
        
        processing_time = time.time() - start_time
        
        # Calculate quality metrics
        quality_metrics = self._calculate_fusion_quality_metrics(
            word_alignments, fused_segments, explicit_overlap_regions, reconciliation_conflicts
        )
        
        result = OverlapFusionResult(
            original_overlap_frame=diarization_result.original_overlap_frame,
            diarization_result=diarization_result,
            fused_segments=fused_segments,
            explicit_overlap_regions=explicit_overlap_regions,
            unified_transcript=unified_transcript,
            total_words_processed=len(word_alignments),
            reconciliation_conflicts=reconciliation_conflicts,
            overlap_regions_count=len(explicit_overlap_regions),
            dual_speaker_regions_count=sum(1 for r in explicit_overlap_regions if len(set([r.primary_speaker, r.secondary_speaker])) == 2),
            overall_confidence=quality_metrics['overall_confidence'],
            fusion_confidence=quality_metrics['fusion_confidence'],
            reconciliation_conflict_rate=quality_metrics['reconciliation_conflict_rate'],
            processing_time=processing_time,
            metadata={
                'fusion_score_weights': self.fusion_score_weights,
                'conflict_resolution_strategy': self.conflict_resolution_strategy,
                'quality_metrics': quality_metrics
            }
        )
        
        self.logger.info("Overlap-aware fusion processing completed",
                        context={
                            'processing_time': processing_time,
                            'total_words_processed': len(word_alignments),
                            'reconciliation_conflicts': reconciliation_conflicts,
                            'overlap_regions_count': len(explicit_overlap_regions),
                            'overall_confidence': quality_metrics['overall_confidence']
                        })
        
        return result
    
    def _calculate_fusion_scores(self, word_alignments: List[WordAlignment]) -> None:
        """Calculate fusion scores for all word alignments"""
        
        for word_alignment in word_alignments:
            # Extract context words
            word_index = word_alignments.index(word_alignment)
            context_words = []
            context_range = 3  # 3 words before and after
            
            for i in range(max(0, word_index - context_range), 
                          min(len(word_alignments), word_index + context_range + 1)):
                if i != word_index:
                    context_words.append(word_alignments[i].word)
            
            # Calculate glossary prior
            glossary_prior = self.glossary_scorer.calculate_word_prior(
                word_alignment.word, context_words
            )
            
            # Calculate attribution confidence component
            attribution_confidence = word_alignment.processing_metadata.get('attribution_confidence', 1.0)
            
            # Calculate fusion score as weighted combination
            fusion_score = (
                word_alignment.acoustic_confidence * self.fusion_score_weights['acoustic_confidence'] +
                word_alignment.cross_engine_agreement * self.fusion_score_weights['cross_engine_agreement'] +
                (glossary_prior - 1.0 + 1.0) * self.fusion_score_weights['glossary_prior'] +  # Normalize glossary prior
                attribution_confidence * self.fusion_score_weights['attribution_confidence']
            )
            
            word_alignment.fusion_score = max(0.0, min(1.0, fusion_score))
            word_alignment.joint_score = word_alignment.fusion_score  # Joint score equals fusion score for now
    
    def _resolve_word_conflicts(self, word_alignments: List[WordAlignment]) -> int:
        """Resolve conflicts between competing words at same timeline positions"""
        
        conflicts_resolved = 0
        
        # Group words by time overlaps
        time_groups = self._group_words_by_time_overlap(word_alignments)
        
        for time_group in time_groups:
            if len(time_group) > 1:
                # Multiple words at same time - potential conflict
                conflicts_resolved += self._resolve_time_group_conflicts(time_group)
        
        return conflicts_resolved
    
    def _group_words_by_time_overlap(self, 
                                   word_alignments: List[WordAlignment],
                                   overlap_threshold: float = 0.1) -> List[List[WordAlignment]]:
        """Group word alignments by temporal overlap"""
        
        if not word_alignments:
            return []
        
        # Sort by start time
        sorted_words = sorted(word_alignments, key=lambda w: w.start_time)
        
        groups = []
        current_group = [sorted_words[0]]
        
        for word in sorted_words[1:]:
            # Check if word overlaps with any word in current group
            overlaps_with_group = any(
                self._words_overlap(word, group_word, overlap_threshold)
                for group_word in current_group
            )
            
            if overlaps_with_group:
                current_group.append(word)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [word]
        
        groups.append(current_group)
        return groups
    
    def _words_overlap(self, 
                      word1: WordAlignment, 
                      word2: WordAlignment, 
                      threshold: float) -> bool:
        """Check if two words overlap in time within threshold"""
        
        overlap_start = max(word1.start_time, word2.start_time)
        overlap_end = min(word1.end_time, word2.end_time)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        return overlap_duration >= threshold
    
    def _resolve_time_group_conflicts(self, time_group: List[WordAlignment]) -> int:
        """Resolve conflicts within a time group using configured strategy"""
        
        if len(time_group) <= 1:
            return 0
        
        if self.conflict_resolution_strategy == "highest_joint_score":
            # Select word with highest joint score
            winner = max(time_group, key=lambda w: w.joint_score)
        elif self.conflict_resolution_strategy == "highest_acoustic_confidence":
            # Select word with highest acoustic confidence
            winner = max(time_group, key=lambda w: w.acoustic_confidence)
        elif self.conflict_resolution_strategy == "cross_engine_agreement":
            # Select word with highest cross-engine agreement
            winner = max(time_group, key=lambda w: w.cross_engine_agreement)
        else:
            # Default to highest joint score
            winner = max(time_group, key=lambda w: w.joint_score)
        
        # Mark the winner and competing words
        for word in time_group:
            if word != winner:
                # This word lost the conflict resolution
                winner.competing_words.append(word.word)
                winner.competing_stems.append(word.stem_id)
                winner.is_overlapped = True
                
                # Calculate cross-engine agreement for winner
                competing_words = [w.word for w in time_group if w != winner]
                competing_confidences = [w.acoustic_confidence for w in time_group if w != winner]
                
                winner.cross_engine_agreement = self.agreement_scorer.calculate_agreement_score(
                    winner.word, competing_words, competing_confidences
                )
        
        return len(time_group) - 1  # Number of conflicts resolved
    
    def _create_explicit_overlap_regions(self,
                                       word_alignments: List[WordAlignment],
                                       cross_stem_overlaps: List[CrossStemOverlapRegion]) -> List[OverlapRegion]:
        """Create explicit overlap regions from word alignments and cross-stem overlaps"""
        
        overlap_regions = []
        
        for cross_stem_overlap in cross_stem_overlaps:
            # Find words that fall within this cross-stem overlap region
            overlapping_words = [
                word for word in word_alignments
                if (word.start_time < cross_stem_overlap.end_time and 
                    word.end_time > cross_stem_overlap.start_time)
            ]
            
            if not overlapping_words:
                continue
            
            # Group words by speaker
            speaker_words = defaultdict(list)
            for word in overlapping_words:
                speaker_words[word.unified_speaker_id].append(word)
            
            if len(speaker_words) >= 2:
                # Create overlap region with dual speakers
                speakers = list(speaker_words.keys())
                primary_speaker = speakers[0]
                secondary_speaker = speakers[1] if len(speakers) > 1 else speakers[0]
                
                # Determine primary/secondary based on confidence
                primary_confidence = np.mean([w.joint_score for w in speaker_words[primary_speaker]])
                secondary_confidence = np.mean([w.joint_score for w in speaker_words[secondary_speaker]]) if len(speakers) > 1 else 0.0
                
                if secondary_confidence > primary_confidence:
                    primary_speaker, secondary_speaker = secondary_speaker, primary_speaker
                
                overlap_region = OverlapRegion(
                    start_time=cross_stem_overlap.start_time,
                    end_time=cross_stem_overlap.end_time,
                    duration=cross_stem_overlap.duration,
                    primary_speaker=primary_speaker,
                    secondary_speaker=secondary_speaker,
                    primary_words=speaker_words[primary_speaker],
                    secondary_words=speaker_words[secondary_speaker],
                    overlap_confidence=cross_stem_overlap.overlap_confidence,
                    reconciliation_confidence=np.mean([w.joint_score for w in overlapping_words]),
                    conflict_resolution_method=self.conflict_resolution_strategy,
                    stems_involved=cross_stem_overlap.stems_involved,
                    cross_stem_overlap_regions=[cross_stem_overlap],
                    metadata={
                        'words_in_overlap': len(overlapping_words),
                        'speakers_in_overlap': len(speaker_words)
                    }
                )
                overlap_regions.append(overlap_region)
        
        return overlap_regions
    
    def _generate_fused_segments(self,
                               word_alignments: List[WordAlignment],
                               explicit_overlap_regions: List[OverlapRegion],
                               reconciliation_conflicts: int) -> List[FusedTranscriptSegment]:
        """Generate fused transcript segments from word alignments and overlap regions"""
        
        if not word_alignments:
            return []
        
        segments = []
        
        # Group words by speaker and temporal continuity
        speaker_segments = self._group_words_by_speaker_continuity(word_alignments)
        
        for segment_words in speaker_segments:
            if not segment_words:
                continue
            
            # Calculate segment boundaries
            start_time = min(w.start_time for w in segment_words)
            end_time = max(w.end_time for w in segment_words)
            duration = end_time - start_time
            
            # Determine primary speaker
            speaker_counts = Counter(w.unified_speaker_id for w in segment_words)
            primary_speaker = speaker_counts.most_common(1)[0][0]
            
            # Build segment text
            segment_text = ' '.join(w.word for w in sorted(segment_words, key=lambda w: w.start_time))
            
            # Calculate segment confidence
            segment_confidence = np.mean([w.joint_score for w in segment_words]) if segment_words else 0.0
            
            # Find overlapping regions for this segment
            overlapping_regions = [
                region for region in explicit_overlap_regions
                if (region.start_time < end_time and region.end_time > start_time)
            ]
            
            # Count reconciliation conflicts in this segment
            segment_conflicts = sum(1 for w in segment_words if w.is_overlapped)
            
            fused_segment = FusedTranscriptSegment(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                text=segment_text,
                speaker_id=primary_speaker,
                confidence=segment_confidence,
                words=sorted(segment_words, key=lambda w: w.start_time),
                has_overlap=len(overlapping_regions) > 0,
                overlap_regions=overlapping_regions,
                overlapping_speakers=list(set(w.unified_speaker_id for w in segment_words)),
                fusion_method="multi_stem" if len(set(w.stem_id for w in segment_words)) > 1 else "single_stem",
                reconciliation_conflicts=segment_conflicts,
                metadata={
                    'word_count': len(segment_words),
                    'unique_stems': len(set(w.stem_id for w in segment_words)),
                    'unique_speakers': len(set(w.unified_speaker_id for w in segment_words))
                }
            )
            segments.append(fused_segment)
        
        return segments
    
    def _group_words_by_speaker_continuity(self,
                                         word_alignments: List[WordAlignment],
                                         max_gap: float = 0.5) -> List[List[WordAlignment]]:
        """Group words by speaker continuity with gap tolerance"""
        
        if not word_alignments:
            return []
        
        # Sort words by time
        sorted_words = sorted(word_alignments, key=lambda w: w.start_time)
        
        segments = []
        current_segment = [sorted_words[0]]
        current_speaker = sorted_words[0].unified_speaker_id
        
        for word in sorted_words[1:]:
            # Check if word continues with same speaker within gap tolerance
            time_gap = word.start_time - current_segment[-1].end_time
            
            if (word.unified_speaker_id == current_speaker and time_gap <= max_gap):
                # Continue current segment
                current_segment.append(word)
            else:
                # Start new segment
                segments.append(current_segment)
                current_segment = [word]
                current_speaker = word.unified_speaker_id
        
        segments.append(current_segment)
        return segments
    
    def _generate_unified_transcript(self, fused_segments: List[FusedTranscriptSegment]) -> str:
        """Generate unified transcript text from fused segments"""
        
        if not fused_segments:
            return ""
        
        # Sort segments by time
        sorted_segments = sorted(fused_segments, key=lambda s: s.start_time)
        
        transcript_parts = []
        
        for segment in sorted_segments:
            # Add speaker label for clarity
            if segment.has_overlap and len(segment.overlapping_speakers) > 1:
                # Multiple speakers - show overlap
                speaker_label = f"[{segment.speaker_id}+{'+'.join(s for s in segment.overlapping_speakers if s != segment.speaker_id)}]"
            else:
                speaker_label = f"[{segment.speaker_id}]"
            
            transcript_parts.append(f"{speaker_label}: {segment.text}")
        
        return "\n".join(transcript_parts)
    
    def _calculate_fusion_quality_metrics(self,
                                        word_alignments: List[WordAlignment],
                                        fused_segments: List[FusedTranscriptSegment],
                                        explicit_overlap_regions: List[OverlapRegion],
                                        reconciliation_conflicts: int) -> Dict[str, float]:
        """Calculate quality metrics for fusion result"""
        
        if not word_alignments:
            return {
                'overall_confidence': 0.0,
                'fusion_confidence': 0.0,
                'reconciliation_conflict_rate': 0.0
            }
        
        # Overall confidence from word alignments
        overall_confidence = np.mean([w.joint_score for w in word_alignments])
        
        # Fusion confidence from segments
        if fused_segments:
            fusion_confidence = np.mean([s.confidence for s in fused_segments])
        else:
            fusion_confidence = 0.0
        
        # Reconciliation conflict rate
        reconciliation_conflict_rate = reconciliation_conflicts / len(word_alignments) if word_alignments else 0.0
        
        return {
            'overall_confidence': overall_confidence,
            'fusion_confidence': fusion_confidence,
            'reconciliation_conflict_rate': reconciliation_conflict_rate
        }
    
    def _create_empty_fusion_result(self,
                                  diarization_result: OverlapDiarizationResult,
                                  processing_time: float) -> OverlapFusionResult:
        """Create empty fusion result when processing fails"""
        
        return OverlapFusionResult(
            original_overlap_frame=diarization_result.original_overlap_frame,
            diarization_result=diarization_result,
            fused_segments=[],
            explicit_overlap_regions=[],
            unified_transcript="",
            total_words_processed=0,
            reconciliation_conflicts=0,
            overlap_regions_count=0,
            dual_speaker_regions_count=0,
            overall_confidence=0.0,
            fusion_confidence=0.0,
            reconciliation_conflict_rate=0.0,
            processing_time=processing_time,
            metadata={'error': 'no_transcriptions_to_fuse'}
        )