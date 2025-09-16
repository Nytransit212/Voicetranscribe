"""
Confusion Network Fusion Engine for Multi-Engine ASR Results

Implements sophisticated fusion using confusion networks, calibrated confidence scores,
temporal coherence penalties, and Minimum Bayes Risk (MBR) path selection for 
optimal transcript generation from multiple ASR engine candidates.
"""

import numpy as np
import re
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import logging

from .alignment_fusion import TemporalAligner, WordAlignment, AlignmentFusionResult
from .intelligent_controller import SegmentAnalysis, SegmentCandidate, IntelligentControllerResult
from .asr_providers.base import ASRResult, ASRSegment, DecodeMode
from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class TokenPosterior:
    """Individual token hypothesis with posterior probability"""
    token: str
    start_time: float
    end_time: float
    posterior: float  # calibrated_confidence × engine_weight × temporal_coherence
    source_engine: str
    source_candidate_id: str
    raw_confidence: float
    calibrated_confidence: float
    engine_weight: float
    temporal_coherence: float
    temporal_penalty: float = 0.0
    is_entity: bool = False
    entity_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ConfusionNetwork:
    """Confusion network representing competing hypotheses at temporal positions"""
    start_time: float
    end_time: float
    token_posteriors: List[TokenPosterior]  # All competing tokens at this position
    consensus_token: Optional[str] = None
    consensus_posterior: float = 0.0
    total_mass: float = 0.0  # Sum of all posteriors (should ≈ 1.0)
    entropy: float = 0.0  # Confusion entropy
    num_engines: int = 0  # Number of engines contributing
    temporal_spread: float = 0.0  # Time span covered by tokens
    confusion_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MBRPath:
    """Minimum Bayes Risk path through confusion networks"""
    tokens: List[str]
    total_score: float
    average_posterior: float
    path_confidence: float
    temporal_coherence_score: float
    entity_consistency_score: float
    path_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusedSegment:
    """Fused transcript segment with confidence and timing"""
    start_time: float
    end_time: float
    text: str
    confidence: float
    words: List[Dict[str, Any]]  # Word-level fusion results
    speaker_id: Optional[str] = None
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusionResult:
    """Complete fusion result for a segment or full transcript"""
    fused_segments: List[FusedSegment]
    fused_transcript: str
    overall_confidence: float
    confusion_networks: List[ConfusionNetwork]
    mbr_path: MBRPath
    fusion_metrics: Dict[str, Any]
    processing_time: float
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)

class EntityDetector:
    """Detects and classifies entities for special fusion handling"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("entity_detector")
        
        # Entity patterns
        self.patterns = {
            'number': re.compile(r'\b\d+(?:\.\d+)?\b'),
            'date': re.compile(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})\b', re.IGNORECASE),
            'time': re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?\b', re.IGNORECASE),
            'money': re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+\s*dollars?\b', re.IGNORECASE),
            'percentage': re.compile(r'\b\d+(?:\.\d+)?%\b'),
            'phone': re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),
            'proper_name': re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        }
    
    def detect_entities(self, token: str, context: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
        """
        Detect if a token is an entity and classify its type
        
        Args:
            token: Token to analyze
            context: Surrounding tokens for context
            
        Returns:
            (is_entity, entity_type) tuple
        """
        for entity_type, pattern in self.patterns.items():
            if pattern.search(token):
                return True, entity_type
        
        return False, None

class TemporalCoherenceScorer:
    """Calculates temporal coherence scores and penalties"""
    
    def __init__(self, 
                 baseline_offset: float = 0.15,  # 150ms baseline
                 penalty_per_100ms: float = 0.10):
        """
        Initialize temporal coherence scorer
        
        Args:
            baseline_offset: Baseline acceptable offset in seconds
            penalty_per_100ms: Penalty per 100ms beyond baseline
        """
        self.baseline_offset = baseline_offset
        self.penalty_per_100ms = penalty_per_100ms
        self.logger = create_enhanced_logger("temporal_coherence")
    
    def calculate_coherence_score(self, 
                                token_start: float,
                                expected_start: float,
                                token_duration: float) -> Tuple[float, float]:
        """
        Calculate temporal coherence score and penalty
        
        Args:
            token_start: Actual token start time
            expected_start: Expected start time based on context
            token_duration: Duration of the token
            
        Returns:
            (coherence_score, temporal_penalty) tuple
        """
        offset = abs(token_start - expected_start)
        
        if offset <= self.baseline_offset:
            return 1.0, 0.0
        
        # Calculate penalty for offset beyond baseline
        excess_offset = offset - self.baseline_offset
        penalty_units = excess_offset / 0.1  # 100ms units
        temporal_penalty = penalty_units * self.penalty_per_100ms
        
        # Cap penalty at 0.8 (coherence score minimum 0.2)
        temporal_penalty = min(temporal_penalty, 0.8)
        coherence_score = 1.0 - temporal_penalty
        
        return max(coherence_score, 0.2), temporal_penalty

class MBRPathSelector:
    """Minimum Bayes Risk path selection through confusion networks"""
    
    def __init__(self, 
                 entity_boost: float = 1.2,
                 consistency_weight: float = 0.15,
                 temporal_weight: float = 0.10):
        """
        Initialize MBR path selector
        
        Args:
            entity_boost: Boost factor for entity tokens
            consistency_weight: Weight for cross-engine consistency
            temporal_weight: Weight for temporal coherence
        """
        self.entity_boost = entity_boost
        self.consistency_weight = consistency_weight
        self.temporal_weight = temporal_weight
        self.logger = create_enhanced_logger("mbr_path_selector")
    
    def select_optimal_path(self, 
                          confusion_networks: List[ConfusionNetwork]) -> MBRPath:
        """
        Select optimal path through confusion networks using MBR
        
        Args:
            confusion_networks: List of confusion networks to traverse
            
        Returns:
            MBRPath with optimal token sequence
        """
        if not confusion_networks:
            return MBRPath([], 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Dynamic programming for MBR path selection
        path_scores = self._calculate_path_scores(confusion_networks)
        optimal_path = self._extract_optimal_path(confusion_networks, path_scores)
        
        return optimal_path
    
    def _calculate_path_scores(self, 
                             confusion_networks: List[ConfusionNetwork]) -> List[Dict[str, float]]:
        """Calculate cumulative scores for all possible paths"""
        n_networks = len(confusion_networks)
        dp_table = [{} for _ in range(n_networks)]
        
        # Initialize first network
        for token_posterior in confusion_networks[0].token_posteriors:
            token = token_posterior.token
            score = self._calculate_token_score(token_posterior, None, 0)
            dp_table[0][token] = score
        
        # Fill DP table
        for i in range(1, n_networks):
            current_network = confusion_networks[i]
            prev_scores = dp_table[i-1]
            
            for token_posterior in current_network.token_posteriors:
                token = token_posterior.token
                best_score = float('-inf')
                
                # Try all previous tokens
                for prev_token, prev_score in prev_scores.items():
                    transition_score = self._calculate_transition_score(
                        prev_token, token_posterior, i
                    )
                    total_score = prev_score + transition_score
                    
                    if total_score > best_score:
                        best_score = total_score
                
                dp_table[i][token] = best_score
        
        return dp_table
    
    def _calculate_token_score(self, 
                             token_posterior: TokenPosterior,
                             prev_token: Optional[str],
                             position: int) -> float:
        """Calculate score for a single token"""
        base_score = token_posterior.posterior
        
        # Apply entity boost
        if token_posterior.is_entity:
            base_score *= self.entity_boost
        
        # Apply temporal coherence
        coherence_bonus = token_posterior.temporal_coherence * self.temporal_weight
        
        return base_score + coherence_bonus
    
    def _calculate_transition_score(self,
                                  prev_token: str,
                                  current_token_posterior: TokenPosterior,
                                  position: int) -> float:
        """Calculate transition score between tokens"""
        # Base token score
        token_score = self._calculate_token_score(current_token_posterior, prev_token, position)
        
        # Linguistic consistency bonus (simplified)
        consistency_bonus = 0.0
        if prev_token and current_token_posterior.token:
            # Simple bigram consistency check
            if self._are_linguistically_consistent(prev_token, current_token_posterior.token):
                consistency_bonus = self.consistency_weight
        
        return token_score + consistency_bonus
    
    def _are_linguistically_consistent(self, prev_token: str, current_token: str) -> bool:
        """Check if two tokens are linguistically consistent"""
        # Simplified linguistic consistency check
        # In practice, this could use language models or n-gram statistics
        
        # Basic rules
        if prev_token.lower() in ['the', 'a', 'an'] and current_token.lower() in ['and', 'or', 'but']:
            return False
        
        if prev_token.endswith('.') and not current_token[0].isupper():
            return False
        
        return True
    
    def _extract_optimal_path(self, 
                            confusion_networks: List[ConfusionNetwork],
                            path_scores: List[Dict[str, float]]) -> MBRPath:
        """Extract the optimal path from DP table"""
        if not path_scores or not path_scores[-1]:
            return MBRPath([], 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Find best final token
        final_scores = path_scores[-1]
        best_final_token = max(final_scores.keys(), key=lambda k: final_scores[k])
        best_score = final_scores[best_final_token]
        
        # Backtrack to find optimal path
        path_tokens = []
        current_token = best_final_token
        
        for i in range(len(confusion_networks) - 1, -1, -1):
            path_tokens.append(current_token)
            
            if i == 0:
                break
            
            # Find best previous token
            best_prev_score = float('-inf')
            best_prev_token = None
            
            for token_posterior in confusion_networks[i].token_posteriors:
                if token_posterior.token == current_token:
                    # Check all possible previous tokens
                    for prev_token in path_scores[i-1].keys():
                        prev_score = path_scores[i-1][prev_token]
                        if prev_score > best_prev_score:
                            best_prev_score = prev_score
                            best_prev_token = prev_token
                    break
            
            current_token = best_prev_token
        
        path_tokens.reverse()
        
        # Calculate path metrics
        average_posterior = best_score / len(path_tokens) if path_tokens else 0.0
        path_confidence = self._calculate_path_confidence(path_tokens, confusion_networks)
        temporal_coherence = self._calculate_temporal_coherence_score(confusion_networks)
        entity_consistency = self._calculate_entity_consistency_score(path_tokens, confusion_networks)
        
        return MBRPath(
            tokens=path_tokens,
            total_score=best_score,
            average_posterior=average_posterior,
            path_confidence=path_confidence,
            temporal_coherence_score=temporal_coherence,
            entity_consistency_score=entity_consistency,
            path_metadata={
                'networks_traversed': len(confusion_networks),
                'total_candidates_considered': sum(len(cn.token_posteriors) for cn in confusion_networks)
            }
        )
    
    def _calculate_path_confidence(self, 
                                 tokens: List[str],
                                 networks: List[ConfusionNetwork]) -> float:
        """Calculate overall confidence for the selected path"""
        if not tokens or not networks:
            return 0.0
        
        total_confidence = 0.0
        for i, token in enumerate(tokens):
            if i < len(networks):
                # Find the posterior for this token in this network
                for token_posterior in networks[i].token_posteriors:
                    if token_posterior.token == token:
                        total_confidence += token_posterior.posterior
                        break
        
        return total_confidence / len(tokens)
    
    def _calculate_temporal_coherence_score(self, networks: List[ConfusionNetwork]) -> float:
        """Calculate temporal coherence score for the path"""
        if len(networks) < 2:
            return 1.0
        
        coherence_scores = []
        for i in range(1, len(networks)):
            prev_network = networks[i-1]
            curr_network = networks[i]
            
            # Calculate expected timing vs actual timing
            expected_gap = 0.1  # Expected 100ms between words
            actual_gap = curr_network.start_time - prev_network.end_time
            
            if abs(actual_gap - expected_gap) <= 0.05:  # Within 50ms
                coherence_scores.append(1.0)
            else:
                penalty = abs(actual_gap - expected_gap) * 2.0  # 2.0 penalty per second
                coherence_scores.append(max(0.0, 1.0 - penalty))
        
        return float(np.mean(coherence_scores)) if coherence_scores else 1.0
    
    def _calculate_entity_consistency_score(self, 
                                          tokens: List[str],
                                          networks: List[ConfusionNetwork]) -> float:
        """Calculate entity consistency score"""
        # Simplified entity consistency check
        # Could be enhanced with more sophisticated entity linking
        return 0.8  # Placeholder

class FusionEngine:
    """
    Main confusion network fusion engine for multi-engine ASR results
    
    Integrates multiple ASR engine outputs using confusion networks,
    calibrated confidence scores, temporal coherence analysis, and
    Minimum Bayes Risk path selection for optimal transcript generation.
    """
    
    def __init__(self,
                 engine_weights: Optional[Dict[str, float]] = None,
                 temporal_coherence_config: Optional[Dict[str, Any]] = None,
                 entity_detection_enabled: bool = True,
                 mbr_config: Optional[Dict[str, Any]] = None):
        """
        Initialize fusion engine
        
        Args:
            engine_weights: Weight per ASR engine (defaults to equal weights)
            temporal_coherence_config: Temporal coherence parameters
            entity_detection_enabled: Enable entity-aware fusion
            mbr_config: MBR path selection configuration
        """
        self.logger = create_enhanced_logger("fusion_engine")
        
        # Engine weights (default: equal weights)
        self.engine_weights = engine_weights or {
            'faster-whisper': 1.0,
            'deepgram': 1.0,
            'openai': 1.0
        }
        
        # Normalize weights
        total_weight = sum(self.engine_weights.values())
        self.engine_weights = {k: v/total_weight for k, v in self.engine_weights.items()}
        
        # Initialize components
        temporal_config = temporal_coherence_config or {}
        self.temporal_scorer = TemporalCoherenceScorer(
            baseline_offset=temporal_config.get('baseline_offset', 0.15),
            penalty_per_100ms=temporal_config.get('penalty_per_100ms', 0.10)
        )
        
        self.entity_detector = EntityDetector() if entity_detection_enabled else None
        
        mbr_params = mbr_config or {}
        self.mbr_selector = MBRPathSelector(
            entity_boost=mbr_params.get('entity_boost', 1.2),
            consistency_weight=mbr_params.get('consistency_weight', 0.15),
            temporal_weight=mbr_params.get('temporal_weight', 0.10)
        )
        
        self.logger.info("FusionEngine initialized", 
                        context={
                            'engine_weights': self.engine_weights,
                            'entity_detection': entity_detection_enabled,
                            'temporal_config': temporal_config
                        })
    
    def fuse_segment_candidates(self, 
                              segment_analysis: SegmentAnalysis) -> FusionResult:
        """
        Fuse multiple ASR candidates for a single segment using confusion networks
        
        Args:
            segment_analysis: SegmentAnalysis with candidates and word alignments
            
        Returns:
            FusionResult with fused transcript and metadata
        """
        start_time = time.time()
        
        self.logger.info(f"Fusing segment with {len(segment_analysis.candidates)} candidates",
                        context={
                            'segment_start': segment_analysis.segment_start,
                            'segment_end': segment_analysis.segment_end,
                            'candidates': len(segment_analysis.candidates)
                        })
        
        # Build confusion networks from word alignments
        confusion_networks = self._build_confusion_networks(
            segment_analysis.word_alignments,
            segment_analysis.candidates
        )
        
        # Select optimal path using MBR
        mbr_path = self.mbr_selector.select_optimal_path(confusion_networks)
        
        # Build fused segments
        fused_segments = self._build_fused_segments(
            mbr_path, 
            confusion_networks,
            segment_analysis
        )
        
        # Generate final transcript
        fused_transcript = ' '.join(segment.text for segment in fused_segments)
        
        # Apply normalization
        fused_transcript = self._apply_normalization(fused_transcript)
        
        # Calculate metrics
        overall_confidence = self._calculate_overall_confidence(fused_segments)
        fusion_metrics = self._calculate_fusion_metrics(
            confusion_networks, 
            mbr_path,
            segment_analysis
        )
        
        processing_time = time.time() - start_time
        
        result = FusionResult(
            fused_segments=fused_segments,
            fused_transcript=fused_transcript,
            overall_confidence=overall_confidence,
            confusion_networks=confusion_networks,
            mbr_path=mbr_path,
            fusion_metrics=fusion_metrics,
            processing_time=processing_time,
            fusion_metadata={
                'segment_start': segment_analysis.segment_start,
                'segment_end': segment_analysis.segment_end,
                'original_candidates': len(segment_analysis.candidates),
                'confusion_networks': len(confusion_networks)
            }
        )
        
        self.logger.info("Segment fusion complete",
                        context={
                            'fused_transcript': fused_transcript[:100] + "..." if len(fused_transcript) > 100 else fused_transcript,
                            'confidence': overall_confidence,
                            'processing_time': processing_time,
                            'networks': len(confusion_networks)
                        })
        
        return result
    
    def fuse_controller_result(self, 
                             controller_result: IntelligentControllerResult) -> List[FusionResult]:
        """
        Fuse all segments from IntelligentController result
        
        Args:
            controller_result: Complete result from IntelligentController
            
        Returns:
            List of FusionResult objects for each segment
        """
        self.logger.info(f"Fusing complete controller result with {len(controller_result.segments)} segments")
        
        fusion_results = []
        for i, segment_analysis in enumerate(controller_result.segments):
            self.logger.info(f"Processing segment {i+1}/{len(controller_result.segments)}")
            
            fusion_result = self.fuse_segment_candidates(segment_analysis)
            fusion_results.append(fusion_result)
        
        self.logger.info(f"Controller result fusion complete: {len(fusion_results)} segments fused")
        return fusion_results
    
    def _build_confusion_networks(self,
                                word_alignments: List[WordAlignment],
                                candidates: List[SegmentCandidate]) -> List[ConfusionNetwork]:
        """Build confusion networks from word alignments"""
        confusion_networks = []
        
        for alignment in word_alignments:
            # Create TokenPosterior objects for each word variant
            token_posteriors = []
            
            for word_variant in alignment.word_variants:
                token_posterior = self._create_token_posterior(
                    word_variant, 
                    alignment,
                    candidates
                )
                token_posteriors.append(token_posterior)
            
            # Create confusion network
            confusion_network = self._create_confusion_network(
                alignment, 
                token_posteriors
            )
            confusion_networks.append(confusion_network)
        
        self.logger.info(f"Built {len(confusion_networks)} confusion networks from {len(word_alignments)} alignments")
        return confusion_networks
    
    def _create_token_posterior(self,
                              word_variant: Dict[str, Any],
                              alignment: WordAlignment,
                              candidates: List[SegmentCandidate]) -> TokenPosterior:
        """Create a TokenPosterior from word variant and alignment data"""
        # Extract basic token information
        token = word_variant.get('word', '').strip()
        start_time = word_variant.get('start', alignment.timestamp_start)
        end_time = word_variant.get('end', alignment.timestamp_end)
        raw_confidence = word_variant.get('confidence', 0.0)
        
        # Find the candidate this word variant belongs to
        candidate_idx = word_variant.get('candidate_idx', 0)
        candidate_id = word_variant.get('candidate_id', f'unknown_{candidate_idx}')
        
        # Get engine information
        source_engine = 'unknown'
        calibrated_confidence = raw_confidence
        
        if candidate_idx < len(candidates):
            candidate = candidates[candidate_idx]
            source_engine = candidate.provider
            # Use the candidate's calibrated confidence as base
            calibrated_confidence = candidate.calibrated_confidence
            
            # Apply word-level confidence if available
            if raw_confidence > 0:
                # Blend candidate-level and word-level confidence
                calibrated_confidence = (calibrated_confidence + raw_confidence) / 2.0
        
        # Get engine weight
        engine_weight = self.engine_weights.get(source_engine, 1.0)
        
        # Calculate temporal coherence
        expected_start = alignment.timestamp_start
        token_duration = end_time - start_time
        temporal_coherence, temporal_penalty = self.temporal_scorer.calculate_coherence_score(
            start_time, expected_start, token_duration
        )
        
        # Detect entities
        is_entity = False
        entity_type = None
        if self.entity_detector:
            is_entity, entity_type = self.entity_detector.detect_entities(token)
        
        # Calculate final posterior
        posterior = calibrated_confidence * engine_weight * temporal_coherence
        
        return TokenPosterior(
            token=token,
            start_time=start_time,
            end_time=end_time,
            posterior=posterior,
            source_engine=source_engine,
            source_candidate_id=candidate_id,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            engine_weight=engine_weight,
            temporal_coherence=temporal_coherence,
            temporal_penalty=temporal_penalty,
            is_entity=is_entity,
            entity_type=entity_type,
            metadata={
                'candidate_idx': candidate_idx,
                'alignment_quality': alignment.alignment_quality
            }
        )
    
    def _create_confusion_network(self,
                                alignment: WordAlignment,
                                token_posteriors: List[TokenPosterior]) -> ConfusionNetwork:
        """Create a ConfusionNetwork from alignment and token posteriors"""
        if not token_posteriors:
            return ConfusionNetwork(
                start_time=alignment.timestamp_start,
                end_time=alignment.timestamp_end,
                token_posteriors=[]
            )
        
        # Calculate network statistics
        total_mass = sum(tp.posterior for tp in token_posteriors)
        
        # Normalize posteriors to sum to 1.0
        if total_mass > 0:
            for tp in token_posteriors:
                tp.posterior = tp.posterior / total_mass
        
        # Find consensus token (highest posterior)
        best_token_posterior = max(token_posteriors, key=lambda tp: tp.posterior)
        consensus_token = best_token_posterior.token
        consensus_posterior = best_token_posterior.posterior
        
        # Calculate entropy (measure of confusion)
        entropy = 0.0
        for tp in token_posteriors:
            if tp.posterior > 0:
                entropy -= tp.posterior * math.log2(tp.posterior)
        
        # Calculate temporal spread
        all_starts = [tp.start_time for tp in token_posteriors]
        all_ends = [tp.end_time for tp in token_posteriors]
        temporal_spread = max(all_ends) - min(all_starts)
        
        # Count unique engines
        unique_engines = len(set(tp.source_engine for tp in token_posteriors))
        
        return ConfusionNetwork(
            start_time=alignment.timestamp_start,
            end_time=alignment.timestamp_end,
            token_posteriors=token_posteriors,
            consensus_token=consensus_token,
            consensus_posterior=consensus_posterior,
            total_mass=1.0,  # Normalized
            entropy=entropy,
            num_engines=unique_engines,
            temporal_spread=temporal_spread,
            confusion_metadata={
                'original_alignment_quality': alignment.alignment_quality,
                'is_confusion_set': alignment.is_confusion_set,
                'original_consensus': alignment.consensus_word
            }
        )
    
    def _build_fused_segments(self,
                            mbr_path: MBRPath,
                            confusion_networks: List[ConfusionNetwork],
                            segment_analysis: SegmentAnalysis) -> List[FusedSegment]:
        """Build fused segments from MBR path"""
        if not mbr_path.tokens or not confusion_networks:
            return []
        
        # Group consecutive tokens into segments
        # For simplicity, create one segment for the entire path
        # In practice, this could be more sophisticated
        
        words = []
        total_confidence = 0.0
        start_time = confusion_networks[0].start_time if confusion_networks else segment_analysis.segment_start
        end_time = confusion_networks[-1].end_time if confusion_networks else segment_analysis.segment_end
        
        # Build word-level fusion results
        for i, token in enumerate(mbr_path.tokens):
            if i < len(confusion_networks):
                network = confusion_networks[i]
                
                # Find the token posterior for this token
                token_posterior = None
                for tp in network.token_posteriors:
                    if tp.token == token:
                        token_posterior = tp
                        break
                
                if token_posterior:
                    word_info = {
                        'word': token,
                        'start': token_posterior.start_time,
                        'end': token_posterior.end_time,
                        'confidence': token_posterior.posterior,
                        'source_engine': token_posterior.source_engine,
                        'is_entity': token_posterior.is_entity,
                        'entity_type': token_posterior.entity_type,
                        'temporal_coherence': token_posterior.temporal_coherence,
                        'temporal_penalty': token_posterior.temporal_penalty
                    }
                    words.append(word_info)
                    total_confidence += token_posterior.posterior
        
        # Calculate segment confidence
        segment_confidence = total_confidence / len(words) if words else 0.0
        
        # Build text
        segment_text = ' '.join(word['word'] for word in words)
        
        fused_segment = FusedSegment(
            start_time=start_time,
            end_time=end_time,
            text=segment_text,
            confidence=segment_confidence,
            words=words,
            speaker_id=None,  # Could be extracted from diarization if available
            fusion_metadata={
                'mbr_total_score': mbr_path.total_score,
                'mbr_average_posterior': mbr_path.average_posterior,
                'path_confidence': mbr_path.path_confidence,
                'temporal_coherence_score': mbr_path.temporal_coherence_score,
                'entity_consistency_score': mbr_path.entity_consistency_score,
                'confusion_networks_count': len(confusion_networks),
                'original_candidates_count': len(segment_analysis.candidates)
            }
        )
        
        return [fused_segment]
    
    def _apply_normalization(self, transcript: str) -> str:
        """Apply punctuation and casing normalization"""
        if not transcript:
            return ""
        
        # Apply basic text normalization
        normalized = transcript.strip()
        
        # Fix spacing around punctuation
        normalized = re.sub(r'\s+([,.!?;:])', r'\1', normalized)  # Remove space before punctuation
        normalized = re.sub(r'([,.!?;:])\s*', r'\1 ', normalized)  # Ensure space after punctuation
        
        # Fix multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Capitalize first letter of sentences
        sentences = re.split(r'([.!?]+)', normalized)
        capitalized_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Text parts (not punctuation)
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            capitalized_sentences.append(sentence)
        
        normalized = ''.join(capitalized_sentences)
        
        # Final cleanup
        normalized = normalized.strip()
        
        return normalized
    
    def _calculate_overall_confidence(self, segments: List[FusedSegment]) -> float:
        """Calculate overall confidence for fused segments"""
        if not segments:
            return 0.0
        
        total_confidence = sum(segment.confidence for segment in segments)
        return total_confidence / len(segments)
    
    def _calculate_fusion_metrics(self,
                                networks: List[ConfusionNetwork],
                                mbr_path: MBRPath,
                                segment_analysis: SegmentAnalysis) -> Dict[str, Any]:
        """Calculate comprehensive fusion metrics"""
        metrics = {}
        
        # Basic network statistics
        metrics['total_confusion_networks'] = len(networks)
        metrics['total_candidate_sources'] = len(segment_analysis.candidates)
        
        # Confusion analysis
        if networks:
            entropies = [network.entropy for network in networks]
            metrics['average_confusion_entropy'] = np.mean(entropies)
            metrics['max_confusion_entropy'] = np.max(entropies)
            metrics['min_confusion_entropy'] = np.min(entropies)
            
            # Consensus statistics
            consensus_posteriors = [network.consensus_posterior for network in networks]
            metrics['average_consensus_confidence'] = np.mean(consensus_posteriors)
            metrics['min_consensus_confidence'] = np.min(consensus_posteriors)
            
            # Temporal spread analysis
            temporal_spreads = [network.temporal_spread for network in networks]
            metrics['average_temporal_spread'] = np.mean(temporal_spreads)
            metrics['max_temporal_spread'] = np.max(temporal_spreads)
            
            # Engine diversity
            all_engines = set()
            for network in networks:
                for tp in network.token_posteriors:
                    all_engines.add(tp.source_engine)
            metrics['unique_engines_used'] = len(all_engines)
            metrics['engines_list'] = list(all_engines)
        
        # MBR path analysis
        metrics['mbr_total_score'] = mbr_path.total_score
        metrics['mbr_average_posterior'] = mbr_path.average_posterior
        metrics['mbr_path_confidence'] = mbr_path.path_confidence
        metrics['mbr_temporal_coherence'] = mbr_path.temporal_coherence_score
        metrics['mbr_entity_consistency'] = mbr_path.entity_consistency_score
        metrics['mbr_tokens_selected'] = len(mbr_path.tokens)
        
        # Entity analysis
        entity_counts = Counter()
        total_entities = 0
        total_temporal_penalties = 0.0
        
        for network in networks:
            for tp in network.token_posteriors:
                if tp.is_entity:
                    total_entities += 1
                    if tp.entity_type:
                        entity_counts[tp.entity_type] += 1
                total_temporal_penalties += tp.temporal_penalty
        
        metrics['total_entities_detected'] = total_entities
        metrics['entity_type_distribution'] = dict(entity_counts)
        metrics['average_temporal_penalty'] = total_temporal_penalties / max(1, sum(len(n.token_posteriors) for n in networks))
        
        # Fusion effectiveness
        original_best_confidence = segment_analysis.confidence_score
        fused_confidence = mbr_path.path_confidence
        metrics['fusion_confidence_improvement'] = fused_confidence - original_best_confidence
        metrics['fusion_effectiveness_ratio'] = fused_confidence / max(0.01, original_best_confidence)
        
        # Agreement analysis
        metrics['original_agreement_score'] = segment_analysis.agreement_score
        metrics['original_best_candidate_provider'] = segment_analysis.best_candidate.provider if segment_analysis.best_candidate else None
        
        return metrics