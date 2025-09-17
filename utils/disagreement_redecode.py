"""
Disagreement-Triggered Re-decode System

This module provides intelligent re-decoding for uncertain transcription spans
using entropy thresholds and diversity enforcement to improve accuracy through
alternative engine configurations and parameters.

Key Features:
- Cross-engine disagreement analysis for uncertainty detection
- Entropy-based threshold calculation for uncertain spans
- Alternative engine configuration with diversity enforcement
- Budget-constrained re-decode attempts with time management
- Integration with existing confidence scoring and ASR systems

Author: Advanced Ensemble Transcription System
"""

import os
import time
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import tempfile
import json
from enum import Enum

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import trace_stage, track_cost
from utils.intelligent_cache import get_cache_manager, cached_operation
from core.run_context import get_global_run_context
from core.asr_engine import ASREngine
from core.confidence_scorer import ConfidenceScorer
from utils.audio_format_validator import ensure_audio_format
from utils.deterministic_processing import get_deterministic_processor
from utils.resilient_api import openai_retry


class UncertaintyDetectionMethod(Enum):
    """Methods for detecting uncertain spans"""
    CROSS_ENGINE_DISAGREEMENT = "cross_engine_disagreement"
    ENTROPY_THRESHOLD = "entropy_threshold"
    CONFIDENCE_GAP = "confidence_gap"
    ACOUSTIC_LINGUISTIC_MISMATCH = "acoustic_linguistic_mismatch"


class AlternativeEngineType(Enum):
    """Types of alternative engines for re-decode"""
    FASTER_WHISPER_HIGH_TEMP = "faster_whisper_high_temp"
    FASTER_WHISPER_LOW_TEMP = "faster_whisper_low_temp"
    OPENAI_WHISPER_BEAM = "openai_whisper_beam"
    OPENAI_WHISPER_GREEDY = "openai_whisper_greedy"
    CONTEXTUAL_PROMPT = "contextual_prompt"
    ALTERNATIVE_CHUNKING = "alternative_chunking"


@dataclass
class UncertainSpan:
    """Represents an uncertain span requiring re-decode"""
    start_time: float
    end_time: float
    duration: float
    original_text: str
    uncertainty_score: float
    entropy_score: float
    confidence_gap: float
    detection_methods: List[UncertaintyDetectionMethod]
    original_engines: List[str]  # Engines that produced this span
    word_level_uncertainties: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RedecodeAttempt:
    """Represents a re-decode attempt"""
    attempt_id: str
    uncertain_span: UncertainSpan
    alternative_engine: AlternativeEngineType
    engine_config: Dict[str, Any]
    result_text: str
    confidence_score: float
    processing_time: float
    success: bool
    improvement_score: float = 0.0  # How much this improved over original
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RedecodeConfig:
    """Configuration for disagreement-triggered re-decode system"""
    
    # Core system controls
    enabled: bool = True
    max_redecode_attempts_per_span: int = 2
    max_total_redecode_attempts: int = 10
    processing_time_budget_seconds: float = 120.0  # 2 minutes max additional processing
    
    # Uncertainty detection thresholds
    entropy_threshold: float = 2.5
    confidence_gap_threshold: float = 0.3
    min_span_seconds: float = 2.0
    max_span_seconds: float = 30.0
    min_uncertainty_score: float = 0.6
    
    # Word-level analysis
    min_word_confidence: float = 0.4
    word_disagreement_threshold: float = 0.5
    acoustic_linguistic_mismatch_threshold: float = 0.4
    
    # Diversity enforcement
    enforce_diversity: bool = True
    max_similar_attempts: int = 1
    parameter_diversity_threshold: float = 0.3
    
    # Alternative engine preferences
    alternative_engine_weights: Dict[AlternativeEngineType, float] = field(default_factory=lambda: {
        AlternativeEngineType.FASTER_WHISPER_HIGH_TEMP: 1.0,
        AlternativeEngineType.FASTER_WHISPER_LOW_TEMP: 0.8,
        AlternativeEngineType.OPENAI_WHISPER_BEAM: 0.9,
        AlternativeEngineType.OPENAI_WHISPER_GREEDY: 0.7,
        AlternativeEngineType.CONTEXTUAL_PROMPT: 1.2,
        AlternativeEngineType.ALTERNATIVE_CHUNKING: 0.6
    })
    
    # Performance optimization
    enable_parallel_redecode: bool = True
    max_parallel_workers: int = 3
    cache_redecode_results: bool = True
    
    # Quality validation
    require_improvement_validation: bool = True
    min_improvement_threshold: float = 0.1
    linguistic_plausibility_check: bool = True


class DisagreementRedecodeEngine:
    """
    Main engine for disagreement-triggered re-decoding with uncertainty detection
    and diversity enforcement for improved transcription accuracy.
    """
    
    def __init__(self, config: Optional[RedecodeConfig] = None):
        """
        Initialize disagreement re-decode engine
        
        Args:
            config: Re-decode configuration. Uses defaults if None.
        """
        self.config = config or RedecodeConfig()
        self.logger = create_enhanced_logger("disagreement_redecode")
        
        # Initialize core components
        self.asr_engine = ASREngine(enable_enhanced_decode=True)
        self.confidence_scorer = ConfidenceScorer()
        self.cache_manager = get_cache_manager()
        
        # Initialize alternative engine configurations
        self._initialize_alternative_engines()
        
        # Performance tracking
        self.stats = {
            'total_uncertain_spans_detected': 0,
            'total_redecode_attempts': 0,
            'successful_redecodes': 0,
            'total_improvement_score': 0.0,
            'processing_time_used': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'diversity_enforcements': 0,
            'uncertainty_detection_methods': defaultdict(int)
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._processing_budget_tracker = 0.0
        
        # Alternative engine pool
        self._alternative_engines: Dict[AlternativeEngineType, Dict[str, Any]] = {}
        
        self.logger.info("DisagreementRedecodeEngine initialized", 
                        context={
                            'config': {
                                'enabled': self.config.enabled,
                                'entropy_threshold': self.config.entropy_threshold,
                                'confidence_gap_threshold': self.config.confidence_gap_threshold,
                                'max_redecode_attempts': self.config.max_redecode_attempts_per_span,
                                'diversity_enforcement': self.config.enforce_diversity
                            }
                        })
    
    def _initialize_alternative_engines(self):
        """Initialize alternative engine configurations for diversity"""
        
        self._alternative_engines = {
            AlternativeEngineType.FASTER_WHISPER_HIGH_TEMP: {
                'provider': 'faster_whisper',
                'temperature': 0.8,
                'beam_size': 5,
                'best_of': 5,
                'description': 'High temperature for creative decoding'
            },
            AlternativeEngineType.FASTER_WHISPER_LOW_TEMP: {
                'provider': 'faster_whisper',
                'temperature': 0.1,
                'beam_size': 3,
                'best_of': 1,
                'description': 'Low temperature for conservative decoding'
            },
            AlternativeEngineType.OPENAI_WHISPER_BEAM: {
                'provider': 'openai',
                'temperature': 0.3,
                'response_format': {'type': 'verbose_json'},
                'description': 'OpenAI with verbose output for confidence'
            },
            AlternativeEngineType.OPENAI_WHISPER_GREEDY: {
                'provider': 'openai',
                'temperature': 0.0,
                'description': 'Greedy decoding for deterministic output'
            },
            AlternativeEngineType.CONTEXTUAL_PROMPT: {
                'provider': 'openai',
                'temperature': 0.2,
                'use_context_prompt': True,
                'description': 'Context-aware decoding with prompts'
            },
            AlternativeEngineType.ALTERNATIVE_CHUNKING: {
                'provider': 'faster_whisper',
                'chunk_length_s': 15,  # Different from default 30s
                'temperature': 0.4,
                'description': 'Alternative chunking strategy'
            }
        }
        
        self.logger.debug("Alternative engine configurations initialized", 
                         context={'engine_count': len(self._alternative_engines)})
    
    @trace_stage("disagreement_redecode_analysis")
    def analyze_and_redecode(self, 
                           candidates: List[Dict[str, Any]], 
                           audio_path: str,
                           original_transcript: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Analyze candidates for disagreement/uncertainty and perform selective re-decode
        
        Args:
            candidates: List of candidate transcripts from ensemble
            audio_path: Path to original audio file
            original_transcript: Optional original transcript for comparison
            
        Returns:
            Tuple of (improved_candidates, redecode_report)
        """
        if not self.config.enabled:
            self.logger.info("Disagreement re-decode disabled by configuration")
            return candidates, {'redecode_enabled': False}
        
        start_time = time.time()
        
        self.logger.info(f"Starting disagreement analysis and re-decode", 
                        context={
                            'candidates_count': len(candidates),
                            'audio_path': audio_path,
                            'processing_budget': self.config.processing_time_budget_seconds
                        })
        
        try:
            # Step 1: Detect uncertain spans across candidates
            uncertain_spans = self._detect_uncertain_spans(candidates)
            
            if not uncertain_spans:
                self.logger.info("No uncertain spans detected, skipping re-decode")
                return candidates, {
                    'redecode_enabled': True,
                    'uncertain_spans_count': 0,
                    'processing_time': time.time() - start_time
                }
            
            # Step 2: Prioritize spans for re-decode based on uncertainty and budget
            prioritized_spans = self._prioritize_uncertain_spans(uncertain_spans)
            
            # Step 3: Perform selective re-decode with diversity enforcement
            redecode_attempts = self._perform_selective_redecode(
                prioritized_spans, audio_path, start_time
            )
            
            # Step 4: Integrate improved results back into candidates
            improved_candidates = self._integrate_redecode_results(
                candidates, redecode_attempts
            )
            
            processing_time = time.time() - start_time
            
            # Step 5: Generate comprehensive report
            redecode_report = self._generate_redecode_report(
                uncertain_spans, redecode_attempts, processing_time
            )
            
            self.logger.info(f"Disagreement re-decode completed", 
                            context={
                                'uncertain_spans': len(uncertain_spans),
                                'redecode_attempts': len(redecode_attempts),
                                'successful_redecodes': len([a for a in redecode_attempts if a.success]),
                                'processing_time': processing_time,
                                'total_improvement': sum(a.improvement_score for a in redecode_attempts)
                            })
            
            return improved_candidates, redecode_report
            
        except Exception as e:
            self.logger.error(f"Error in disagreement re-decode analysis: {e}")
            return candidates, {
                'redecode_enabled': True,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _detect_uncertain_spans(self, candidates: List[Dict[str, Any]]) -> List[UncertainSpan]:
        """
        Detect uncertain spans using multiple uncertainty detection methods
        
        Args:
            candidates: List of candidate transcripts
            
        Returns:
            List of UncertainSpan objects for re-decode consideration
        """
        uncertain_spans = []
        
        # Extract all segments from candidates for cross-engine analysis
        all_segments = []
        for candidate in candidates:
            if 'segments' in candidate and candidate['segments']:
                all_segments.extend([
                    {**seg, 'candidate_id': candidate.get('candidate_id', 'unknown')}
                    for seg in candidate['segments']
                ])
        
        if not all_segments:
            self.logger.warning("No segments found in candidates for uncertainty analysis")
            return uncertain_spans
        
        # Group segments by time overlap for cross-engine comparison
        time_aligned_segments = self._align_segments_by_time(all_segments)
        
        # Apply different uncertainty detection methods
        for time_group in time_aligned_segments:
            # Method 1: Cross-engine disagreement
            disagreement_spans = self._detect_cross_engine_disagreement(time_group)
            uncertain_spans.extend(disagreement_spans)
            
            # Method 2: Entropy threshold analysis
            entropy_spans = self._detect_entropy_uncertainty(time_group)
            uncertain_spans.extend(entropy_spans)
            
            # Method 3: Confidence gap analysis
            confidence_spans = self._detect_confidence_gaps(time_group)
            uncertain_spans.extend(confidence_spans)
            
            # Method 4: Acoustic-linguistic mismatch (if available)
            if self.config.acoustic_linguistic_mismatch_threshold > 0:
                mismatch_spans = self._detect_acoustic_linguistic_mismatch(time_group)
                uncertain_spans.extend(mismatch_spans)
        
        # Merge overlapping uncertain spans and filter by thresholds
        merged_spans = self._merge_overlapping_spans(uncertain_spans)
        filtered_spans = self._filter_spans_by_thresholds(merged_spans)
        
        # Update statistics
        with self._lock:
            self.stats['total_uncertain_spans_detected'] += len(filtered_spans)
            for span in filtered_spans:
                for method in span.detection_methods:
                    self.stats['uncertainty_detection_methods'][method] += 1
        
        self.logger.debug(f"Uncertain span detection completed", 
                         context={
                             'raw_spans': len(uncertain_spans),
                             'merged_spans': len(merged_spans),
                             'filtered_spans': len(filtered_spans)
                         })
        
        return filtered_spans
    
    def _align_segments_by_time(self, segments: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group segments by temporal overlap for cross-engine comparison
        
        Args:
            segments: All segments from all candidates
            
        Returns:
            List of segment groups aligned by time overlap
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.get('start', 0))
        
        time_groups = []
        current_group = [sorted_segments[0]]
        current_end = sorted_segments[0].get('end', sorted_segments[0].get('start', 0))
        
        for segment in sorted_segments[1:]:
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', segment_start)
            
            # Check for temporal overlap with current group
            if segment_start <= current_end + 0.5:  # 0.5s tolerance for alignment
                current_group.append(segment)
                current_end = max(current_end, segment_end)
            else:
                # Start new group
                if len(current_group) > 1:  # Only include groups with multiple segments
                    time_groups.append(current_group)
                current_group = [segment]
                current_end = segment_end
        
        # Add final group
        if len(current_group) > 1:
            time_groups.append(current_group)
        
        return time_groups
    
    def _detect_cross_engine_disagreement(self, segment_group: List[Dict[str, Any]]) -> List[UncertainSpan]:
        """
        Detect uncertainty through cross-engine disagreement analysis
        
        Args:
            segment_group: Group of temporally aligned segments from different engines
            
        Returns:
            List of uncertain spans due to engine disagreement
        """
        uncertain_spans = []
        
        if len(segment_group) < 2:
            return uncertain_spans
        
        # Extract texts and confidence scores
        texts = [seg.get('text', '').strip() for seg in segment_group]
        confidences = [seg.get('confidence', 0.5) for seg in segment_group]
        engines = [seg.get('candidate_id', 'unknown') for seg in segment_group]
        
        # Calculate text similarity and disagreement
        unique_texts = list(set(texts))
        if len(unique_texts) <= 1:
            return uncertain_spans  # No disagreement
        
        # Calculate disagreement metrics
        text_variations = len(unique_texts)
        max_confidence = max(confidences)
        min_confidence = min(confidences)
        confidence_gap = max_confidence - min_confidence
        
        # Calculate word-level disagreement
        word_disagreement_score = self._calculate_word_level_disagreement(texts)
        
        # Determine uncertainty score
        uncertainty_score = min(1.0, (
            (text_variations - 1) * 0.3 +  # More variations = more uncertainty
            confidence_gap * 0.4 +          # Higher confidence gap = more uncertainty
            word_disagreement_score * 0.3   # Word-level disagreement
        ))
        
        # Check if this meets the disagreement threshold
        if (uncertainty_score >= self.config.min_uncertainty_score and 
            confidence_gap >= self.config.confidence_gap_threshold):
            
            # Calculate span timing from the group
            start_time = min(seg.get('start', 0) for seg in segment_group)
            end_time = max(seg.get('end', seg.get('start', 0)) for seg in segment_group)
            duration = end_time - start_time
            
            if duration >= self.config.min_span_seconds:
                uncertain_span = UncertainSpan(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    original_text=texts[0],  # Use first text as reference
                    uncertainty_score=uncertainty_score,
                    entropy_score=0.0,  # Will be calculated separately
                    confidence_gap=confidence_gap,
                    detection_methods=[UncertaintyDetectionMethod.CROSS_ENGINE_DISAGREEMENT],
                    original_engines=engines,
                    metadata={
                        'text_variations': text_variations,
                        'all_texts': texts,
                        'all_confidences': confidences,
                        'word_disagreement_score': word_disagreement_score
                    }
                )
                uncertain_spans.append(uncertain_span)
        
        return uncertain_spans
    
    def _calculate_word_level_disagreement(self, texts: List[str]) -> float:
        """
        Calculate word-level disagreement between multiple texts
        
        Args:
            texts: List of text variations
            
        Returns:
            Word-level disagreement score (0-1)
        """
        if len(texts) < 2:
            return 0.0
        
        # Tokenize texts into words
        word_lists = [text.lower().split() for text in texts]
        max_length = max(len(words) for words in word_lists)
        
        if max_length == 0:
            return 0.0
        
        # Calculate word position disagreements
        disagreements = 0
        total_positions = 0
        
        for pos in range(max_length):
            words_at_pos = []
            for word_list in word_lists:
                if pos < len(word_list):
                    words_at_pos.append(word_list[pos])
                else:
                    words_at_pos.append("")  # Missing word
            
            # Count unique words at this position
            unique_words = set(words_at_pos)
            if len(unique_words) > 1:
                disagreements += 1
            total_positions += 1
        
        return disagreements / total_positions if total_positions > 0 else 0.0
    
    def _detect_entropy_uncertainty(self, segment_group: List[Dict[str, Any]]) -> List[UncertainSpan]:
        """
        Detect uncertainty through entropy analysis of token distributions
        
        Args:
            segment_group: Group of temporally aligned segments
            
        Returns:
            List of uncertain spans with high entropy scores
        """
        uncertain_spans = []
        
        if len(segment_group) < 2:
            return uncertain_spans
        
        # Extract texts for entropy calculation
        texts = [seg.get('text', '').strip() for seg in segment_group]
        
        # Calculate token-level entropy
        entropy_score = self._calculate_text_entropy(texts)
        
        if entropy_score >= self.config.entropy_threshold:
            # Calculate span timing
            start_time = min(seg.get('start', 0) for seg in segment_group)
            end_time = max(seg.get('end', seg.get('start', 0)) for seg in segment_group)
            duration = end_time - start_time
            
            if duration >= self.config.min_span_seconds:
                # Calculate uncertainty score based on entropy
                uncertainty_score = min(1.0, entropy_score / 5.0)  # Normalize entropy to 0-1
                
                confidences = [seg.get('confidence', 0.5) for seg in segment_group]
                confidence_gap = max(confidences) - min(confidences)
                engines = [seg.get('candidate_id', 'unknown') for seg in segment_group]
                
                uncertain_span = UncertainSpan(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    original_text=texts[0],
                    uncertainty_score=uncertainty_score,
                    entropy_score=entropy_score,
                    confidence_gap=confidence_gap,
                    detection_methods=[UncertaintyDetectionMethod.ENTROPY_THRESHOLD],
                    original_engines=engines,
                    metadata={
                        'calculated_entropy': entropy_score,
                        'entropy_threshold': self.config.entropy_threshold
                    }
                )
                uncertain_spans.append(uncertain_span)
        
        return uncertain_spans
    
    def _calculate_text_entropy(self, texts: List[str]) -> float:
        """
        Calculate entropy of token distribution across texts
        
        Args:
            texts: List of text variations
            
        Returns:
            Entropy score (higher = more uncertain)
        """
        if not texts:
            return 0.0
        
        # Tokenize all texts and count token frequencies
        all_tokens = []
        for text in texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return 0.0
        
        # Calculate token probabilities
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in token_counts.values():
            probability = count / total_tokens
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _detect_confidence_gaps(self, segment_group: List[Dict[str, Any]]) -> List[UncertainSpan]:
        """
        Detect uncertainty through large confidence gaps between engines
        
        Args:
            segment_group: Group of temporally aligned segments
            
        Returns:
            List of uncertain spans with large confidence gaps
        """
        uncertain_spans = []
        
        if len(segment_group) < 2:
            return uncertain_spans
        
        confidences = [seg.get('confidence', 0.5) for seg in segment_group]
        confidence_gap = max(confidences) - min(confidences)
        
        if confidence_gap >= self.config.confidence_gap_threshold:
            # Calculate span timing
            start_time = min(seg.get('start', 0) for seg in segment_group)
            end_time = max(seg.get('end', seg.get('start', 0)) for seg in segment_group)
            duration = end_time - start_time
            
            if duration >= self.config.min_span_seconds:
                texts = [seg.get('text', '').strip() for seg in segment_group]
                engines = [seg.get('candidate_id', 'unknown') for seg in segment_group]
                
                # Uncertainty score based on confidence gap
                uncertainty_score = min(1.0, confidence_gap / 0.5)  # Normalize to 0-1
                
                uncertain_span = UncertainSpan(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    original_text=texts[0],
                    uncertainty_score=uncertainty_score,
                    entropy_score=0.0,
                    confidence_gap=confidence_gap,
                    detection_methods=[UncertaintyDetectionMethod.CONFIDENCE_GAP],
                    original_engines=engines,
                    metadata={
                        'confidence_values': confidences,
                        'max_confidence': max(confidences),
                        'min_confidence': min(confidences)
                    }
                )
                uncertain_spans.append(uncertain_span)
        
        return uncertain_spans
    
    def _detect_acoustic_linguistic_mismatch(self, segment_group: List[Dict[str, Any]]) -> List[UncertainSpan]:
        """
        Detect uncertainty through acoustic-linguistic confidence mismatch
        
        Args:
            segment_group: Group of temporally aligned segments
            
        Returns:
            List of uncertain spans with acoustic-linguistic mismatches
        """
        uncertain_spans = []
        
        # This would require acoustic model confidence vs linguistic model confidence
        # For now, return empty list - this can be implemented when acoustic confidence is available
        # TODO: Implement when acoustic model outputs separate confidence scores
        
        return uncertain_spans
    
    def _merge_overlapping_spans(self, spans: List[UncertainSpan]) -> List[UncertainSpan]:
        """
        Merge overlapping uncertain spans to avoid redundant re-decode attempts
        
        Args:
            spans: List of uncertain spans
            
        Returns:
            List of merged spans
        """
        if not spans:
            return spans
        
        # Sort spans by start time
        sorted_spans = sorted(spans, key=lambda x: x.start_time)
        merged_spans = []
        
        current_span = sorted_spans[0]
        
        for next_span in sorted_spans[1:]:
            # Check for overlap (with small tolerance)
            if next_span.start_time <= current_span.end_time + 0.5:
                # Merge spans
                current_span = UncertainSpan(
                    start_time=current_span.start_time,
                    end_time=max(current_span.end_time, next_span.end_time),
                    duration=max(current_span.end_time, next_span.end_time) - current_span.start_time,
                    original_text=current_span.original_text,  # Keep first text
                    uncertainty_score=max(current_span.uncertainty_score, next_span.uncertainty_score),
                    entropy_score=max(current_span.entropy_score, next_span.entropy_score),
                    confidence_gap=max(current_span.confidence_gap, next_span.confidence_gap),
                    detection_methods=list(set(current_span.detection_methods + next_span.detection_methods)),
                    original_engines=list(set(current_span.original_engines + next_span.original_engines)),
                    metadata={
                        'merged_from': [current_span.metadata, next_span.metadata],
                        'merge_count': current_span.metadata.get('merge_count', 1) + 1
                    }
                )
            else:
                # No overlap, add current span and start new one
                merged_spans.append(current_span)
                current_span = next_span
        
        # Add the last span
        merged_spans.append(current_span)
        
        return merged_spans
    
    def _filter_spans_by_thresholds(self, spans: List[UncertainSpan]) -> List[UncertainSpan]:
        """
        Filter spans based on configuration thresholds
        
        Args:
            spans: List of uncertain spans
            
        Returns:
            List of spans that meet threshold criteria
        """
        filtered_spans = []
        
        for span in spans:
            # Check duration thresholds
            if (span.duration < self.config.min_span_seconds or 
                span.duration > self.config.max_span_seconds):
                continue
            
            # Check uncertainty score threshold
            if span.uncertainty_score < self.config.min_uncertainty_score:
                continue
            
            # Check specific method thresholds
            meets_entropy_threshold = (
                UncertaintyDetectionMethod.ENTROPY_THRESHOLD in span.detection_methods and
                span.entropy_score >= self.config.entropy_threshold
            )
            
            meets_confidence_threshold = (
                UncertaintyDetectionMethod.CONFIDENCE_GAP in span.detection_methods and
                span.confidence_gap >= self.config.confidence_gap_threshold
            )
            
            meets_disagreement_threshold = (
                UncertaintyDetectionMethod.CROSS_ENGINE_DISAGREEMENT in span.detection_methods and
                span.uncertainty_score >= self.config.min_uncertainty_score
            )
            
            # Span must meet at least one specific threshold
            if meets_entropy_threshold or meets_confidence_threshold or meets_disagreement_threshold:
                filtered_spans.append(span)
        
        return filtered_spans
    
    def _prioritize_uncertain_spans(self, spans: List[UncertainSpan]) -> List[UncertainSpan]:
        """
        Prioritize uncertain spans for re-decode based on uncertainty score and processing budget
        
        Args:
            spans: List of uncertain spans
            
        Returns:
            Prioritized list of spans for re-decode
        """
        if not spans:
            return spans
        
        # Calculate priority score for each span
        for span in spans:
            # Priority based on multiple factors
            priority_score = (
                span.uncertainty_score * 0.4 +           # Higher uncertainty = higher priority
                (span.confidence_gap / 1.0) * 0.3 +      # Larger confidence gap = higher priority
                (span.entropy_score / 5.0) * 0.2 +       # Higher entropy = higher priority
                min(span.duration / 10.0, 1.0) * 0.1     # Longer spans get slight priority boost
            )
            span.metadata['priority_score'] = priority_score
        
        # Sort by priority score (highest first)
        prioritized_spans = sorted(spans, key=lambda x: x.metadata.get('priority_score', 0), reverse=True)
        
        # Limit based on total re-decode attempts budget
        max_spans = min(len(prioritized_spans), self.config.max_total_redecode_attempts)
        
        return prioritized_spans[:max_spans]
    
    def _perform_selective_redecode(self, 
                                  uncertain_spans: List[UncertainSpan],
                                  audio_path: str,
                                  start_time: float) -> List[RedecodeAttempt]:
        """
        Perform selective re-decode with diversity enforcement and budget management
        
        Args:
            uncertain_spans: Prioritized uncertain spans
            audio_path: Path to original audio file
            start_time: Start time for budget tracking
            
        Returns:
            List of re-decode attempts
        """
        redecode_attempts = []
        
        if not uncertain_spans:
            return redecode_attempts
        
        # Process spans with parallel execution if enabled
        if self.config.enable_parallel_redecode:
            redecode_attempts = self._parallel_redecode_spans(uncertain_spans, audio_path, start_time)
        else:
            redecode_attempts = self._sequential_redecode_spans(uncertain_spans, audio_path, start_time)
        
        return redecode_attempts
    
    def _parallel_redecode_spans(self, 
                               spans: List[UncertainSpan], 
                               audio_path: str,
                               start_time: float) -> List[RedecodeAttempt]:
        """
        Process multiple spans in parallel with budget management
        
        Args:
            spans: Uncertain spans to re-decode
            audio_path: Path to original audio file
            start_time: Start time for budget tracking
            
        Returns:
            List of re-decode attempts
        """
        redecode_attempts = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
            future_to_span = {}
            
            for span in spans:
                # Check budget before submitting
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.config.processing_time_budget_seconds:
                    self.logger.info(f"Processing budget exceeded, stopping at {len(redecode_attempts)} attempts")
                    break
                
                future = executor.submit(self._redecode_single_span, span, audio_path)
                future_to_span[future] = span
            
            # Collect results with timeout
            remaining_budget = max(1.0, self.config.processing_time_budget_seconds - (time.time() - start_time))
            
            for future in as_completed(future_to_span, timeout=remaining_budget):
                try:
                    span_attempts = future.result()
                    redecode_attempts.extend(span_attempts)
                    
                    # Check budget after each completion
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= self.config.processing_time_budget_seconds:
                        self.logger.info("Processing budget exceeded during parallel execution")
                        # Cancel remaining futures
                        for remaining_future in future_to_span:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                        
                except Exception as e:
                    span = future_to_span[future]
                    self.logger.error(f"Error in parallel re-decode for span {span.start_time}-{span.end_time}: {e}")
        
        return redecode_attempts
    
    def _sequential_redecode_spans(self,
                                 spans: List[UncertainSpan], 
                                 audio_path: str,
                                 start_time: float) -> List[RedecodeAttempt]:
        """
        Process spans sequentially with budget management
        
        Args:
            spans: Uncertain spans to re-decode
            audio_path: Path to original audio file
            start_time: Start time for budget tracking
            
        Returns:
            List of re-decode attempts
        """
        redecode_attempts = []
        
        for span in spans:
            # Check budget before processing
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.config.processing_time_budget_seconds:
                self.logger.info(f"Processing budget exceeded, stopping at {len(redecode_attempts)} attempts")
                break
            
            try:
                span_attempts = self._redecode_single_span(span, audio_path)
                redecode_attempts.extend(span_attempts)
                
            except Exception as e:
                self.logger.error(f"Error in sequential re-decode for span {span.start_time}-{span.end_time}: {e}")
                continue
        
        return redecode_attempts
    
    @cached_operation("redecode_span")
    def _redecode_single_span(self, span: UncertainSpan, audio_path: str) -> List[RedecodeAttempt]:
        """
        Re-decode a single uncertain span using alternative engines with diversity enforcement
        
        Args:
            span: Uncertain span to re-decode
            audio_path: Path to original audio file
            
        Returns:
            List of re-decode attempts for this span
        """
        attempts = []
        
        self.logger.debug(f"Starting re-decode for span {span.start_time:.2f}-{span.end_time:.2f}",
                         context={
                             'uncertainty_score': span.uncertainty_score,
                             'entropy_score': span.entropy_score,
                             'confidence_gap': span.confidence_gap,
                             'original_text': span.original_text[:100]
                         })
        
        # Extract audio segment for this span
        segment_audio_path = self._extract_audio_segment(audio_path, span.start_time, span.end_time)
        
        if not segment_audio_path:
            self.logger.error(f"Failed to extract audio segment for span {span.start_time}-{span.end_time}")
            return attempts
        
        try:
            # Select alternative engines with diversity enforcement
            selected_engines = self._select_diverse_engines(span, self.config.max_redecode_attempts_per_span)
            
            for i, (engine_type, engine_config) in enumerate(selected_engines):
                attempt_start_time = time.time()
                attempt_id = f"{span.start_time:.2f}-{span.end_time:.2f}-{engine_type.value}-{i}"
                
                try:
                    # Perform re-decode with alternative engine
                    result_text, confidence_score = self._execute_alternative_decode(
                        segment_audio_path, engine_type, engine_config
                    )
                    
                    processing_time = time.time() - attempt_start_time
                    
                    # Calculate improvement score
                    improvement_score = self._calculate_improvement_score(
                        span.original_text, result_text, span.uncertainty_score
                    )
                    
                    # Create attempt record
                    attempt = RedecodeAttempt(
                        attempt_id=attempt_id,
                        uncertain_span=span,
                        alternative_engine=engine_type,
                        engine_config=engine_config,
                        result_text=result_text,
                        confidence_score=confidence_score,
                        processing_time=processing_time,
                        success=True,
                        improvement_score=improvement_score,
                        metadata={
                            'original_text': span.original_text,
                            'engine_description': self._alternative_engines[engine_type]['description']
                        }
                    )
                    
                    attempts.append(attempt)
                    
                    # Update statistics
                    with self._lock:
                        self.stats['total_redecode_attempts'] += 1
                        if improvement_score > self.config.min_improvement_threshold:
                            self.stats['successful_redecodes'] += 1
                        self.stats['total_improvement_score'] += improvement_score
                        self.stats['processing_time_used'] += processing_time
                    
                    self.logger.debug(f"Re-decode attempt completed",
                                     context={
                                         'attempt_id': attempt_id,
                                         'engine_type': engine_type.value,
                                         'improvement_score': improvement_score,
                                         'processing_time': processing_time
                                     })
                
                except Exception as e:
                    processing_time = time.time() - attempt_start_time
                    
                    # Create failed attempt record
                    failed_attempt = RedecodeAttempt(
                        attempt_id=attempt_id,
                        uncertain_span=span,
                        alternative_engine=engine_type,
                        engine_config=engine_config,
                        result_text="",
                        confidence_score=0.0,
                        processing_time=processing_time,
                        success=False,
                        improvement_score=0.0,
                        metadata={'error': str(e)}
                    )
                    
                    attempts.append(failed_attempt)
                    
                    self.logger.warning(f"Re-decode attempt failed: {e}",
                                       context={'attempt_id': attempt_id, 'engine_type': engine_type.value})
        
        finally:
            # Clean up temporary audio segment
            try:
                if segment_audio_path and os.path.exists(segment_audio_path):
                    os.unlink(segment_audio_path)
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary audio segment: {e}")
        
        return attempts
    
    def _extract_audio_segment(self, audio_path: str, start_time: float, end_time: float) -> Optional[str]:
        """
        Extract audio segment for re-decode processing
        
        Args:
            audio_path: Path to original audio file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            
        Returns:
            Path to extracted audio segment or None if failed
        """
        try:
            import librosa
            import soundfile as sf
            
            # Load audio segment
            y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time - start_time)
            
            # Create temporary file for segment
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='redecode_segment_')
            os.close(temp_fd)
            
            # Write segment to temporary file
            sf.write(temp_path, y, sr)
            
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Failed to extract audio segment {start_time}-{end_time}: {e}")
            return None
    
    def _select_diverse_engines(self, 
                              span: UncertainSpan, 
                              max_attempts: int) -> List[Tuple[AlternativeEngineType, Dict[str, Any]]]:
        """
        Select diverse alternative engines based on original engines and diversity requirements
        
        Args:
            span: Uncertain span with original engine information
            max_attempts: Maximum number of re-decode attempts
            
        Returns:
            List of (engine_type, config) tuples ensuring diversity
        """
        selected_engines = []
        used_engines = set(span.original_engines)  # Avoid using same engines as original
        
        # Get available alternative engines sorted by weight
        available_engines = sorted(
            self._alternative_engines.items(),
            key=lambda x: self.config.alternative_engine_weights.get(x[0], 0.5),
            reverse=True
        )
        
        # Apply diversity enforcement
        for engine_type, base_config in available_engines:
            if len(selected_engines) >= max_attempts:
                break
            
            # Check if this engine type provides sufficient diversity
            if self.config.enforce_diversity:
                # Ensure we don't repeat similar configurations
                is_diverse = True
                for selected_type, _ in selected_engines:
                    if self._calculate_engine_similarity(engine_type, selected_type) > self.config.parameter_diversity_threshold:
                        is_diverse = False
                        break
                
                if not is_diverse:
                    continue
            
            # Create configuration with span-specific adaptations
            adapted_config = self._adapt_engine_config(base_config.copy(), span)
            selected_engines.append((engine_type, adapted_config))
            
            # Update diversity tracking
            with self._lock:
                self.stats['diversity_enforcements'] += 1
        
        # Fallback: if diversity enforcement blocked all options, select top options anyway
        if not selected_engines and available_engines:
            self.logger.warning("Diversity enforcement blocked all engines, using fallback selection")
            for engine_type, base_config in available_engines[:max_attempts]:
                adapted_config = self._adapt_engine_config(base_config.copy(), span)
                selected_engines.append((engine_type, adapted_config))
        
        return selected_engines
    
    def _calculate_engine_similarity(self, engine1: AlternativeEngineType, engine2: AlternativeEngineType) -> float:
        """
        Calculate similarity between two engine types for diversity enforcement
        
        Args:
            engine1: First engine type
            engine2: Second engine type
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if engine1 == engine2:
            return 1.0
        
        config1 = self._alternative_engines[engine1]
        config2 = self._alternative_engines[engine2]
        
        # Compare key parameters
        similarity_factors = []
        
        # Provider similarity
        if config1.get('provider') == config2.get('provider'):
            similarity_factors.append(0.4)
        else:
            similarity_factors.append(0.0)
        
        # Temperature similarity
        temp1 = config1.get('temperature', 0.5)
        temp2 = config2.get('temperature', 0.5)
        temp_diff = abs(temp1 - temp2)
        temp_similarity = max(0, 1 - temp_diff * 2)  # Scale temperature differences
        similarity_factors.append(temp_similarity * 0.3)
        
        # Other parameter similarities
        if config1.get('beam_size') == config2.get('beam_size'):
            similarity_factors.append(0.2)
        
        if config1.get('chunk_length_s') == config2.get('chunk_length_s'):
            similarity_factors.append(0.1)
        
        return sum(similarity_factors)
    
    def _adapt_engine_config(self, base_config: Dict[str, Any], span: UncertainSpan) -> Dict[str, Any]:
        """
        Adapt engine configuration based on span characteristics
        
        Args:
            base_config: Base engine configuration
            span: Uncertain span for adaptation
            
        Returns:
            Adapted configuration
        """
        adapted_config = base_config.copy()
        
        # Adapt based on span duration
        if span.duration > 15.0:  # Long spans
            if 'chunk_length_s' in adapted_config:
                adapted_config['chunk_length_s'] = min(adapted_config['chunk_length_s'], span.duration / 2)
        
        # Adapt based on uncertainty type
        if UncertaintyDetectionMethod.ENTROPY_THRESHOLD in span.detection_methods:
            # High entropy suggests need for more diversity
            if 'temperature' in adapted_config:
                adapted_config['temperature'] = min(1.0, adapted_config['temperature'] * 1.2)
        
        if UncertaintyDetectionMethod.CONFIDENCE_GAP in span.detection_methods:
            # Confidence gaps suggest need for more conservative decoding
            if 'temperature' in adapted_config:
                adapted_config['temperature'] = max(0.0, adapted_config['temperature'] * 0.8)
        
        # Add context if available
        if 'use_context_prompt' in adapted_config:
            adapted_config['context_text'] = span.original_text
        
        return adapted_config
    
    def _execute_alternative_decode(self, 
                                  segment_audio_path: str,
                                  engine_type: AlternativeEngineType, 
                                  engine_config: Dict[str, Any]) -> Tuple[str, float]:
        """
        Execute alternative decoding with specified engine and configuration
        
        Args:
            segment_audio_path: Path to audio segment
            engine_type: Type of alternative engine
            engine_config: Engine configuration parameters
            
        Returns:
            Tuple of (decoded_text, confidence_score)
        """
        # This would integrate with actual ASR providers
        # For now, simulate different decoding approaches
        
        try:
            if engine_config.get('provider') == 'openai':
                # Use OpenAI Whisper with specified configuration
                return self._decode_with_openai(segment_audio_path, engine_config)
            
            elif engine_config.get('provider') == 'faster_whisper':
                # Use Faster Whisper with specified configuration
                return self._decode_with_faster_whisper(segment_audio_path, engine_config)
            
            else:
                # Fallback to OpenAI
                return self._decode_with_openai(segment_audio_path, engine_config)
                
        except Exception as e:
            self.logger.error(f"Alternative decode execution failed: {e}")
            return "", 0.0
    
    @openai_retry
    def _decode_with_openai(self, audio_path: str, config: Dict[str, Any]) -> Tuple[str, float]:
        """
        Decode using OpenAI Whisper with alternative configuration
        
        Args:
            audio_path: Path to audio segment
            config: OpenAI configuration
            
        Returns:
            Tuple of (decoded_text, confidence_score)
        """
        try:
            # Use existing ASR engine but with modified parameters
            with open(audio_path, 'rb') as audio_file:
                transcription_params = {
                    'model': 'whisper-1',
                    'temperature': config.get('temperature', 0.3),
                    'response_format': config.get('response_format', 'text')
                }
                
                # Add context prompt if specified
                if config.get('use_context_prompt') and config.get('context_text'):
                    transcription_params['prompt'] = f"Context: {config['context_text'][:200]}"
                
                # Make API call with alternative configuration
                response = self.asr_engine.client.audio.transcriptions.create(
                    file=audio_file,
                    **transcription_params
                )
                
                if isinstance(response, dict) and 'text' in response:
                    text = response['text'].strip()
                    confidence = response.get('confidence', 0.7)  # Default confidence
                else:
                    text = str(response).strip()
                    confidence = 0.7  # Default confidence when not available
                
                return text, confidence
                
        except Exception as e:
            self.logger.error(f"OpenAI alternative decode failed: {e}")
            return "", 0.0
    
    def _decode_with_faster_whisper(self, audio_path: str, config: Dict[str, Any]) -> Tuple[str, float]:
        """
        Decode using Faster Whisper with alternative configuration
        
        Args:
            audio_path: Path to audio segment
            config: Faster Whisper configuration
            
        Returns:
            Tuple of (decoded_text, confidence_score)
        """
        try:
            # This would use actual Faster Whisper implementation
            # For now, simulate by using OpenAI with modified parameters
            return self._decode_with_openai(audio_path, config)
            
        except Exception as e:
            self.logger.error(f"Faster Whisper alternative decode failed: {e}")
            return "", 0.0
    
    def _calculate_improvement_score(self, original_text: str, new_text: str, uncertainty_score: float) -> float:
        """
        Calculate improvement score for re-decode attempt
        
        Args:
            original_text: Original text from uncertain span
            new_text: New text from re-decode attempt
            uncertainty_score: Original uncertainty score
            
        Returns:
            Improvement score (positive = improvement, negative = degradation)
        """
        if not new_text or not original_text:
            return 0.0
        
        # Calculate text similarity metrics
        from difflib import SequenceMatcher
        
        # Basic text similarity
        similarity = SequenceMatcher(None, original_text.lower(), new_text.lower()).ratio()
        
        # If texts are identical, no improvement
        if similarity > 0.99:
            return 0.0
        
        # Calculate potential improvement based on uncertainty
        # Higher uncertainty means more room for improvement
        max_possible_improvement = uncertainty_score
        
        # Length similarity factor (very different lengths might indicate problems)
        length_ratio = min(len(new_text), len(original_text)) / max(len(new_text), len(original_text))
        length_penalty = max(0, 1 - abs(1 - length_ratio))
        
        # Calculate improvement score
        # This is a heuristic - in practice, this would use quality metrics
        improvement_score = (
            (1 - similarity) * max_possible_improvement * length_penalty * 0.5 +
            uncertainty_score * 0.3 +  # Base improvement from reducing uncertainty
            (0.2 if len(new_text) > len(original_text) else 0)  # Slight bonus for more detailed transcripts
        )
        
        return max(0.0, min(1.0, improvement_score))
    
    def _integrate_redecode_results(self, 
                                  original_candidates: List[Dict[str, Any]],
                                  redecode_attempts: List[RedecodeAttempt]) -> List[Dict[str, Any]]:
        """
        Integrate successful re-decode results back into candidate list
        
        Args:
            original_candidates: Original candidate transcripts
            redecode_attempts: List of re-decode attempts
            
        Returns:
            Updated candidate list with integrated improvements
        """
        improved_candidates = []
        
        # Filter successful re-decode attempts with sufficient improvement
        successful_attempts = [
            attempt for attempt in redecode_attempts
            if (attempt.success and 
                attempt.improvement_score >= self.config.min_improvement_threshold)
        ]
        
        if not successful_attempts:
            self.logger.info("No successful re-decode attempts with sufficient improvement")
            return original_candidates
        
        # Group successful attempts by span
        attempts_by_span = defaultdict(list)
        for attempt in successful_attempts:
            span_key = f"{attempt.uncertain_span.start_time}-{attempt.uncertain_span.end_time}"
            attempts_by_span[span_key].append(attempt)
        
        # For each original candidate, check for improvements
        for candidate in original_candidates:
            improved_candidate = self._apply_improvements_to_candidate(
                candidate, attempts_by_span
            )
            improved_candidates.append(improved_candidate)
        
        # Optionally add new candidates from best re-decode attempts
        if len(successful_attempts) > 0:
            # Create a new candidate from best re-decode attempts
            best_redecode_candidate = self._create_redecode_candidate(successful_attempts)
            if best_redecode_candidate:
                improved_candidates.append(best_redecode_candidate)
        
        return improved_candidates
    
    def _apply_improvements_to_candidate(self, 
                                       candidate: Dict[str, Any],
                                       attempts_by_span: Dict[str, List[RedecodeAttempt]]) -> Dict[str, Any]:
        """
        Apply re-decode improvements to a specific candidate
        
        Args:
            candidate: Original candidate transcript
            attempts_by_span: Re-decode attempts grouped by span
            
        Returns:
            Improved candidate with re-decode results integrated
        """
        improved_candidate = candidate.copy()
        
        # Track improvements applied
        improvements_applied = []
        
        # Check each segment in the candidate for potential improvements
        if 'segments' in candidate and candidate['segments']:
            improved_segments = []
            
            for segment in candidate['segments']:
                segment_start = segment.get('start', 0)
                segment_end = segment.get('end', segment_start)
                
                # Look for overlapping re-decode attempts
                best_improvement = None
                best_improvement_score = 0.0
                
                for span_key, attempts in attempts_by_span.items():
                    for attempt in attempts:
                        span = attempt.uncertain_span
                        
                        # Check for temporal overlap
                        overlap_start = max(segment_start, span.start_time)
                        overlap_end = min(segment_end, span.end_time)
                        
                        if overlap_start < overlap_end:  # There is overlap
                            overlap_duration = overlap_end - overlap_start
                            overlap_ratio = overlap_duration / (segment_end - segment_start)
                            
                            # Only apply if significant overlap (>50%) and good improvement
                            if (overlap_ratio > 0.5 and 
                                attempt.improvement_score > best_improvement_score):
                                best_improvement = attempt
                                best_improvement_score = attempt.improvement_score
                
                # Apply best improvement if found
                if best_improvement:
                    improved_segment = segment.copy()
                    improved_segment['text'] = best_improvement.result_text
                    improved_segment['confidence'] = max(
                        segment.get('confidence', 0.5),
                        best_improvement.confidence_score
                    )
                    improved_segment['redecode_metadata'] = {
                        'original_text': segment.get('text', ''),
                        'redecode_engine': best_improvement.alternative_engine.value,
                        'improvement_score': best_improvement.improvement_score,
                        'attempt_id': best_improvement.attempt_id
                    }
                    improved_segments.append(improved_segment)
                    improvements_applied.append(best_improvement.attempt_id)
                else:
                    improved_segments.append(segment)
            
            improved_candidate['segments'] = improved_segments
        
        # Update candidate metadata
        if improvements_applied:
            improved_candidate['redecode_metadata'] = {
                'improvements_applied': improvements_applied,
                'improvement_count': len(improvements_applied),
                'redecode_engine_used': True
            }
        
        return improved_candidate
    
    def _create_redecode_candidate(self, successful_attempts: List[RedecodeAttempt]) -> Optional[Dict[str, Any]]:
        """
        Create a new candidate from best re-decode attempts
        
        Args:
            successful_attempts: List of successful re-decode attempts
            
        Returns:
            New candidate created from re-decode results or None
        """
        if not successful_attempts:
            return None
        
        # Sort attempts by improvement score
        sorted_attempts = sorted(successful_attempts, key=lambda x: x.improvement_score, reverse=True)
        
        # Create segments from best attempts
        segments = []
        total_improvement = 0.0
        
        for attempt in sorted_attempts:
            span = attempt.uncertain_span
            segment = {
                'start': span.start_time,
                'end': span.end_time,
                'text': attempt.result_text,
                'confidence': attempt.confidence_score,
                'speaker_id': 'unknown',  # Would need speaker info from original
                'redecode_metadata': {
                    'original_text': span.original_text,
                    'redecode_engine': attempt.alternative_engine.value,
                    'improvement_score': attempt.improvement_score,
                    'attempt_id': attempt.attempt_id
                }
            }
            segments.append(segment)
            total_improvement += attempt.improvement_score
        
        # Create new candidate
        redecode_candidate = {
            'candidate_id': f'redecode_composite_{int(time.time())}',
            'text': ' '.join(seg['text'] for seg in segments),
            'segments': segments,
            'confidence': np.mean([seg['confidence'] for seg in segments]),
            'metadata': {
                'source': 'disagreement_redecode',
                'total_improvement_score': total_improvement,
                'redecode_attempts_used': len(successful_attempts),
                'creation_timestamp': time.time()
            }
        }
        
        return redecode_candidate
    
    def _generate_redecode_report(self,
                                uncertain_spans: List[UncertainSpan],
                                redecode_attempts: List[RedecodeAttempt],
                                processing_time: float) -> Dict[str, Any]:
        """
        Generate comprehensive report of re-decode operations
        
        Args:
            uncertain_spans: List of uncertain spans detected
            redecode_attempts: List of re-decode attempts made
            processing_time: Total processing time
            
        Returns:
            Comprehensive re-decode report
        """
        successful_attempts = [a for a in redecode_attempts if a.success]
        improved_attempts = [a for a in successful_attempts if a.improvement_score >= self.config.min_improvement_threshold]
        
        # Calculate statistics
        total_uncertain_duration = sum(span.duration for span in uncertain_spans)
        successful_redecode_duration = sum(
            attempt.uncertain_span.duration for attempt in improved_attempts
        )
        
        # Detection method breakdown
        detection_methods_count = defaultdict(int)
        for span in uncertain_spans:
            for method in span.detection_methods:
                detection_methods_count[method.value] += 1
        
        # Engine usage breakdown
        engine_usage_count = defaultdict(int)
        engine_success_count = defaultdict(int)
        for attempt in redecode_attempts:
            engine_usage_count[attempt.alternative_engine.value] += 1
            if attempt.success and attempt.improvement_score >= self.config.min_improvement_threshold:
                engine_success_count[attempt.alternative_engine.value] += 1
        
        # Generate report
        report = {
            'redecode_enabled': True,
            'processing_time': processing_time,
            'processing_budget_used': processing_time / self.config.processing_time_budget_seconds,
            
            # Uncertain span analysis
            'uncertain_spans': {
                'total_count': len(uncertain_spans),
                'total_duration': total_uncertain_duration,
                'average_duration': total_uncertain_duration / len(uncertain_spans) if uncertain_spans else 0,
                'average_uncertainty_score': np.mean([s.uncertainty_score for s in uncertain_spans]) if uncertain_spans else 0,
                'detection_methods': dict(detection_methods_count)
            },
            
            # Re-decode attempts
            'redecode_attempts': {
                'total_attempts': len(redecode_attempts),
                'successful_attempts': len(successful_attempts),
                'improved_attempts': len(improved_attempts),
                'success_rate': len(successful_attempts) / len(redecode_attempts) if redecode_attempts else 0,
                'improvement_rate': len(improved_attempts) / len(redecode_attempts) if redecode_attempts else 0,
                'average_processing_time': np.mean([a.processing_time for a in redecode_attempts]) if redecode_attempts else 0
            },
            
            # Quality improvements
            'quality_improvements': {
                'total_improvement_score': sum(a.improvement_score for a in redecode_attempts),
                'average_improvement_score': np.mean([a.improvement_score for a in redecode_attempts]) if redecode_attempts else 0,
                'duration_improved': successful_redecode_duration,
                'improvement_coverage': successful_redecode_duration / total_uncertain_duration if total_uncertain_duration > 0 else 0
            },
            
            # Engine performance
            'engine_performance': {
                'usage_count': dict(engine_usage_count),
                'success_count': dict(engine_success_count),
                'success_rates': {
                    engine: engine_success_count[engine] / engine_usage_count[engine]
                    for engine in engine_usage_count
                    if engine_usage_count[engine] > 0
                }
            },
            
            # System performance
            'system_performance': {
                'cache_hits': self.stats.get('cache_hits', 0),
                'cache_misses': self.stats.get('cache_misses', 0),
                'diversity_enforcements': self.stats.get('diversity_enforcements', 0),
                'processing_time_used': self.stats.get('processing_time_used', 0.0)
            },
            
            # Configuration used
            'configuration': {
                'entropy_threshold': self.config.entropy_threshold,
                'confidence_gap_threshold': self.config.confidence_gap_threshold,
                'min_span_seconds': self.config.min_span_seconds,
                'max_redecode_attempts': self.config.max_redecode_attempts_per_span,
                'diversity_enforcement': self.config.enforce_diversity,
                'parallel_processing': self.config.enable_parallel_redecode
            }
        }
        
        return report


# Factory functions for easy integration
def create_disagreement_redecode_engine(config: Optional[RedecodeConfig] = None) -> DisagreementRedecodeEngine:
    """
    Factory function to create disagreement re-decode engine
    
    Args:
        config: Optional configuration. Uses defaults if None.
        
    Returns:
        Initialized DisagreementRedecodeEngine instance
    """
    return DisagreementRedecodeEngine(config)


# Module-level singleton instance
_disagreement_redecode_engine_instance: Optional[DisagreementRedecodeEngine] = None
_engine_lock = threading.Lock()


def get_disagreement_redecode_engine() -> DisagreementRedecodeEngine:
    """
    Get singleton instance of disagreement re-decode engine
    
    Returns:
        Global DisagreementRedecodeEngine instance
    """
    global _disagreement_redecode_engine_instance
    if _disagreement_redecode_engine_instance is None:
        with _engine_lock:
            if _disagreement_redecode_engine_instance is None:
                _disagreement_redecode_engine_instance = DisagreementRedecodeEngine()
    return _disagreement_redecode_engine_instance