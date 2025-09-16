"""
Consensus Module for Advanced Ensemble Transcription System

Provides multiple consensus strategies for combining 15 candidate transcripts 
into a final result. Supports both "best single candidate" and "fused consensus" approaches.
"""

import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from difflib import SequenceMatcher
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.deterministic_parallel import StableTieBreaker
from core.alignment_fusion import AlignmentAwareFusionEngine, AlignmentFusionResult

@dataclass
class ConsensusResult:
    """Result from consensus processing"""
    winner_candidate: Dict[str, Any]
    consensus_method: str
    consensus_confidence: float
    consensus_metadata: Dict[str, Any]
    alternative_candidates: Optional[List[Dict[str, Any]]] = None
    fused_segments: Optional[List[Dict[str, Any]]] = None

class ConsensusStrategy(ABC):
    """Abstract base class for consensus strategies"""
    
    @abstractmethod
    def name(self) -> str:
        """Return strategy name"""
        pass
    
    @abstractmethod
    def select_consensus(self, candidates: List[Dict[str, Any]]) -> ConsensusResult:
        """Select consensus result from candidates"""
        pass
    
    def apply_session_bias(self, session_bias_list) -> None:
        """Apply session bias to consensus strategy (default no-op implementation)"""
        # Default implementation - strategies can override for specific bias application
        pass
    
    def supports_session_bias(self) -> bool:
        """Return whether this strategy supports session bias application"""
        # Default assumes basic support via no-op - strategies can override
        return True

class BestSingleCandidateStrategy(ConsensusStrategy):
    """Baseline strategy: select best single candidate (current system)"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("consensus_best_single")
    
    def name(self) -> str:
        return "best_single_candidate"
    
    def select_consensus(self, candidates: List[Dict[str, Any]]) -> ConsensusResult:
        """Select best single candidate using current scoring logic"""
        if not candidates:
            raise ValueError("No candidates provided for winner selection")
        
        # Sort by final score (descending) - current system logic
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x['confidence_scores']['final_score'],
            reverse=True
        )
        
        winner = sorted_candidates[0]
        
        # Check for ties and apply tie-breakers (current system logic)
        final_score = winner['confidence_scores']['final_score']
        tied_candidates = [c for c in sorted_candidates 
                          if abs(c['confidence_scores']['final_score'] - final_score) < 0.001]
        
        if len(tied_candidates) > 1:
            winner = self._apply_tie_breakers(tied_candidates)
        
        self.logger.info(f"Selected winner with score {final_score:.3f}", 
                        context={'method': 'best_single_candidate', 'tied_count': len(tied_candidates)})
        
        return ConsensusResult(
            winner_candidate=winner,
            consensus_method=self.name(),
            consensus_confidence=final_score,
            consensus_metadata={
                'selection_method': 'highest_score_with_tiebreakers',
                'tied_candidates_count': len(tied_candidates),
                'winner_score': final_score,
                'score_gap': (final_score - sorted_candidates[1]['confidence_scores']['final_score']) if len(sorted_candidates) > 1 else 0.0
            },
            alternative_candidates=sorted_candidates[1:6]  # Top 5 alternatives
        )
    
    def _apply_tie_breakers(self, tied_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply deterministic tie-breaking rules to select winner"""
        
        # Use stable tie-breaker for deterministic results
        return StableTieBreaker.break_ties_by_score_then_lexical(
            tied_candidates, 
            score_key='confidence_scores',
            fallback_key='candidate_id'
        )
    
    def apply_session_bias(self, session_bias_list) -> None:
        """Best single candidate doesn't modify selection based on session bias"""
        # This strategy uses existing confidence scores which may already incorporate bias
        # No additional bias application needed at consensus level
        self.logger.debug("Session bias received but not applied - confidence scores may already include bias effects")
    
    def supports_session_bias(self) -> bool:
        """Returns True since bias effects are handled via confidence scoring"""
        return True

class WeightedVotingStrategy(ConsensusStrategy):
    """Weighted voting strategy based on confidence scores"""
    
    def __init__(self, confidence_threshold: float = 0.7, top_k: int = 5):
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.logger = create_enhanced_logger("consensus_weighted_voting")
    
    def name(self) -> str:
        return "weighted_voting"
    
    def select_consensus(self, candidates: List[Dict[str, Any]]) -> ConsensusResult:
        """Select consensus using weighted voting of top candidates"""
        if not candidates:
            raise ValueError("No candidates provided for consensus")
        
        # Filter candidates above confidence threshold
        viable_candidates = [c for c in candidates 
                           if c['confidence_scores']['final_score'] >= self.confidence_threshold]
        
        if not viable_candidates:
            # Fall back to top 3 if none meet threshold
            viable_candidates = sorted(candidates, 
                                     key=lambda x: x['confidence_scores']['final_score'], 
                                     reverse=True)[:3]
        
        # Take top-k viable candidates for voting
        top_candidates = sorted(viable_candidates, 
                              key=lambda x: x['confidence_scores']['final_score'], 
                              reverse=True)[:self.top_k]
        
        # Weight votes by confidence scores
        total_weight = sum(c['confidence_scores']['final_score'] for c in top_candidates)
        weights = [c['confidence_scores']['final_score'] / total_weight for c in top_candidates]
        
        # Create weighted consensus by fusing segments
        fused_segments = self._create_weighted_fusion(top_candidates, weights)
        
        # Select best candidate as representative
        winner = top_candidates[0]
        
        # Calculate consensus confidence as weighted average
        consensus_confidence = sum(w * c['confidence_scores']['final_score'] 
                                 for w, c in zip(weights, top_candidates))
        
        self.logger.info(f"Weighted voting consensus from {len(top_candidates)} candidates", 
                        context={'method': 'weighted_voting', 'consensus_confidence': consensus_confidence})
        
        return ConsensusResult(
            winner_candidate=winner,
            consensus_method=self.name(),
            consensus_confidence=consensus_confidence,
            consensus_metadata={
                'voting_candidates': len(top_candidates),
                'total_candidates': len(candidates),
                'confidence_threshold': self.confidence_threshold,
                'weights': weights,
                'fusion_applied': True
            },
            alternative_candidates=top_candidates[1:],
            fused_segments=fused_segments
        )
    
    def _create_weighted_fusion(self, candidates: List[Dict[str, Any]], weights: List[float]) -> List[Dict[str, Any]]:
        """Create weighted fusion of segment alignments"""
        # For now, return segments from highest weighted candidate
        # TODO: Implement proper CTM-level fusion
        return candidates[0].get('aligned_segments', [])

class MultiDimensionalConsensusStrategy(ConsensusStrategy):
    """Consensus based on multi-dimensional agreement patterns"""
    
    def __init__(self, dimension_weights: Optional[Dict[str, float]] = None):
        self.dimension_weights = dimension_weights or {
            'D': 0.25, 'A': 0.30, 'L': 0.20, 'R': 0.15, 'O': 0.10
        }
        self.logger = create_enhanced_logger("consensus_multidimensional")
    
    def name(self) -> str:
        return "multidimensional_consensus"
    
    def select_consensus(self, candidates: List[Dict[str, Any]]) -> ConsensusResult:
        """Select consensus based on dimensional agreement patterns"""
        if not candidates:
            raise ValueError("No candidates provided for consensus")
        
        # Analyze dimensional performance patterns
        dimension_leaders = self._find_dimensional_leaders(candidates)
        
        # Calculate consensus scores based on dimensional excellence
        consensus_scores = self._calculate_consensus_scores(candidates, dimension_leaders)
        
        # Select winner based on consensus scoring
        winner_idx = np.argmax(consensus_scores)
        winner = candidates[winner_idx]
        consensus_confidence = consensus_scores[winner_idx]
        
        # Sort by consensus scores for alternatives
        sorted_indices = np.argsort(consensus_scores)[::-1]
        alternatives = [candidates[i] for i in sorted_indices[1:6]]
        
        self.logger.info(f"Multi-dimensional consensus selected candidate {winner_idx}", 
                        context={'method': 'multidimensional_consensus', 
                                'consensus_confidence': consensus_confidence,
                                'dimension_leaders': dimension_leaders})
        
        return ConsensusResult(
            winner_candidate=winner,
            consensus_method=self.name(),
            consensus_confidence=consensus_confidence,
            consensus_metadata={
                'dimension_leaders': dimension_leaders,
                'consensus_scores': consensus_scores.tolist(),
                'dimensional_analysis': True,
                'dimension_weights': self.dimension_weights
            },
            alternative_candidates=alternatives
        )
    
    def _find_dimensional_leaders(self, candidates: List[Dict[str, Any]]) -> Dict[str, int]:
        """Find which candidates lead in each dimension"""
        leaders = {}
        dimensions = ['D', 'A', 'L', 'R', 'O']
        
        for dim in dimensions:
            scores = [c['confidence_scores'][f'{dim}_{"diarization" if dim=="D" else "asr_alignment" if dim=="A" else "linguistic" if dim=="L" else "agreement" if dim=="R" else "overlap"}'] 
                     for c in candidates]
            leaders[dim] = int(np.argmax(scores))
        
        return leaders
    
    def _calculate_consensus_scores(self, candidates: List[Dict[str, Any]], 
                                  dimension_leaders: Dict[str, int]) -> np.ndarray:
        """Calculate consensus scores based on dimensional leadership"""
        scores = np.zeros(len(candidates))
        
        # Award points for dimensional leadership
        for dim, leader_idx in dimension_leaders.items():
            weight = self.dimension_weights[dim]
            scores[leader_idx] += weight * 0.5  # Leadership bonus
        
        # Add base scores weighted by dimensional performance
        for i, candidate in enumerate(candidates):
            base_score = 0.0
            conf_scores = candidate['confidence_scores']
            
            for dim, weight in self.dimension_weights.items():
                dim_key = f'{dim}_{"diarization" if dim=="D" else "asr_alignment" if dim=="A" else "linguistic" if dim=="L" else "agreement" if dim=="R" else "overlap"}'
                base_score += weight * conf_scores[dim_key]
            
            scores[i] += base_score * 0.5  # Base performance
        
        return scores

class ConfidenceBasedStrategy(ConsensusStrategy):
    """Consensus based on confidence distribution analysis"""
    
    def __init__(self, confidence_variance_threshold: float = 0.1):
        self.confidence_variance_threshold = confidence_variance_threshold
        self.logger = create_enhanced_logger("consensus_confidence_based")
    
    def name(self) -> str:
        return "confidence_based"
    
    def select_consensus(self, candidates: List[Dict[str, Any]]) -> ConsensusResult:
        """Select consensus based on confidence distribution patterns"""
        if not candidates:
            raise ValueError("No candidates provided for consensus")
        
        # Analyze confidence score distribution
        final_scores = [c['confidence_scores']['final_score'] for c in candidates]
        confidence_variance = np.var(final_scores)
        confidence_mean = np.mean(final_scores)
        
        # If low variance, select based on stability
        if confidence_variance < self.confidence_variance_threshold:
            winner = self._select_most_stable_candidate(candidates)
            selection_reason = "stability_based_low_variance"
        else:
            # High variance: select clear leader
            winner = max(candidates, key=lambda x: x['confidence_scores']['final_score'])
            selection_reason = "clear_leader_high_variance"
        
        winner_score = winner['confidence_scores']['final_score']
        
        # Calculate consensus confidence based on distribution
        confidence_spread = max(final_scores) - min(final_scores)
        consensus_confidence = winner_score * (1.0 - min(confidence_spread, 0.5))
        
        self.logger.info(f"Confidence-based selection: {selection_reason}", 
                        context={'method': 'confidence_based', 
                                'variance': confidence_variance,
                                'spread': confidence_spread})
        
        return ConsensusResult(
            winner_candidate=winner,
            consensus_method=self.name(),
            consensus_confidence=consensus_confidence,
            consensus_metadata={
                'selection_reason': selection_reason,
                'confidence_variance': confidence_variance,
                'confidence_mean': confidence_mean,
                'confidence_spread': confidence_spread,
                'stability_analysis': True
            },
            alternative_candidates=sorted(candidates, 
                                        key=lambda x: x['confidence_scores']['final_score'], 
                                        reverse=True)[1:6]
        )
    
    def _select_most_stable_candidate(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select candidate with most stable dimensional scores"""
        stability_scores = []
        
        for candidate in candidates:
            scores = candidate['confidence_scores']
            dimension_scores = [
                scores['D_diarization'],
                scores['A_asr_alignment'], 
                scores['L_linguistic'],
                scores['R_agreement'],
                scores['O_overlap']
            ]
            
            # Lower variance = higher stability
            variance = np.var(dimension_scores)
            mean_score = np.mean(dimension_scores)
            stability = mean_score / (1.0 + variance)
            stability_scores.append(stability)
        
        most_stable_idx = np.argmax(stability_scores)
        return candidates[most_stable_idx]

class AlignmentAwareFusionStrategy(ConsensusStrategy):
    """Advanced consensus strategy using alignment-aware fusion with confusion sets"""
    
    def __init__(self, 
                 timestamp_tolerance: float = 0.3,
                 confidence_threshold: float = 0.1,
                 fusion_strategy: str = "confidence_weighted"):
        """
        Initialize alignment-aware fusion strategy
        
        Args:
            timestamp_tolerance: Maximum time difference for word alignment (seconds)
            confidence_threshold: Minimum confidence difference to consider significant
            fusion_strategy: Strategy for fusion ('confidence_weighted', 'majority_vote', 'hybrid')
        """
        self.timestamp_tolerance = timestamp_tolerance
        self.confidence_threshold = confidence_threshold
        self.fusion_strategy = fusion_strategy
        
        # Initialize fusion engine
        self.fusion_engine = AlignmentAwareFusionEngine(
            timestamp_tolerance=timestamp_tolerance,
            confidence_threshold=confidence_threshold,
            fusion_strategy=fusion_strategy
        )
        
        self.logger = create_enhanced_logger("consensus_alignment_fusion")
    
    def name(self) -> str:
        return "alignment_aware_fusion"
    
    def select_consensus(self, candidates: List[Dict[str, Any]]) -> ConsensusResult:
        """
        Select consensus using alignment-aware fusion with confusion sets
        
        Args:
            candidates: List of scored candidate transcripts
            
        Returns:
            ConsensusResult with fused transcript and alignment metadata
        """
        if not candidates:
            raise ValueError("No candidates provided for alignment-aware fusion")
        
        self.logger.info(f"Starting alignment-aware fusion of {len(candidates)} candidates", 
                        context={'candidates_count': len(candidates)})
        
        try:
            # Perform alignment-aware fusion
            fusion_result = self.fusion_engine.fuse_candidates_with_alignment(candidates)
            
            # Create a synthetic winner candidate from the fused result
            winner_candidate = self._create_winner_from_fusion(fusion_result, candidates)
            
            # Calculate consensus confidence based on fusion metrics
            consensus_confidence = fusion_result.confidence_weighted_score
            
            # Prepare metadata
            consensus_metadata = {
                'fusion_strategy': self.fusion_strategy,
                'timestamp_tolerance': self.timestamp_tolerance,
                'alignment_metrics': fusion_result.alignment_metrics.__dict__,
                'word_alignments_count': len(fusion_result.word_alignments),
                'confusion_sets_count': len(fusion_result.confusion_sets),
                'fusion_effectiveness': fusion_result.alignment_metrics.fusion_effectiveness,
                'token_oscillation_reduction': fusion_result.alignment_metrics.token_oscillation_reduction,
                'numeric_consistency_score': fusion_result.alignment_metrics.numeric_consistency_score,
                'alignment_coverage': fusion_result.alignment_metrics.alignment_coverage
            }
            
            self.logger.info(f"Alignment-aware fusion completed", 
                            context={
                                'consensus_confidence': consensus_confidence,
                                'word_alignments': len(fusion_result.word_alignments),
                                'confusion_sets': len(fusion_result.confusion_sets),
                                'fusion_effectiveness': fusion_result.alignment_metrics.fusion_effectiveness
                            })
            
            return ConsensusResult(
                winner_candidate=winner_candidate,
                consensus_method=self.name(),
                consensus_confidence=consensus_confidence,
                consensus_metadata=consensus_metadata,
                alternative_candidates=self._get_alternative_candidates(candidates),
                fused_segments=fusion_result.fused_segments
            )
            
        except Exception as e:
            self.logger.error(f"Alignment-aware fusion failed: {e}")
            
            # Fallback to best single candidate
            self.logger.warning("Falling back to best single candidate selection")
            return self._fallback_to_best_candidate(candidates)
    
    def _create_winner_from_fusion(self, 
                                 fusion_result: AlignmentFusionResult,
                                 original_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a winner candidate from fusion result"""
        
        # Use the highest scoring original candidate as base
        best_candidate = max(original_candidates, 
                           key=lambda x: x.get('confidence_scores', {}).get('final_score', 0.0))
        
        # Create new candidate with fused data
        winner_candidate = best_candidate.copy()
        
        # Update with fused transcript data
        winner_candidate['candidate_id'] = f"fusion_{len(original_candidates)}_candidates"
        winner_candidate['aligned_segments'] = fusion_result.fused_segments
        
        # Update ASR data with fused transcript
        if 'asr_data' in winner_candidate:
            winner_candidate['asr_data'] = winner_candidate['asr_data'].copy()
            winner_candidate['asr_data']['text'] = fusion_result.fused_transcript
            
            # Extract words from fused segments
            fused_words = []
            for segment in fusion_result.fused_segments:
                if 'words' in segment:
                    fused_words.extend(segment['words'])
            
            winner_candidate['asr_data']['words'] = fused_words
        
        # Update confidence scores to reflect fusion quality
        if 'confidence_scores' in winner_candidate:
            confidence_scores = winner_candidate['confidence_scores'].copy()
            
            # Boost final score based on fusion effectiveness
            fusion_boost = fusion_result.alignment_metrics.fusion_effectiveness * 0.1
            confidence_scores['final_score'] = min(1.0, confidence_scores['final_score'] + fusion_boost)
            
            # Add fusion-specific scores
            confidence_scores['alignment_quality'] = fusion_result.alignment_metrics.fusion_effectiveness
            confidence_scores['confusion_resolution_rate'] = (
                fusion_result.alignment_metrics.confusion_sets_resolved / 
                max(fusion_result.alignment_metrics.confusion_sets_created, 1)
            )
            
            winner_candidate['confidence_scores'] = confidence_scores
        
        # Add fusion metadata
        winner_candidate['fusion_metadata'] = {
            'fusion_applied': True,
            'alignment_metrics': fusion_result.alignment_metrics.__dict__,
            'original_candidates_count': len(original_candidates),
            'fusion_confidence': fusion_result.confidence_weighted_score
        }
        
        return winner_candidate
    
    def _get_alternative_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get alternative candidates sorted by confidence"""
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('confidence_scores', {}).get('final_score', 0.0),
            reverse=True
        )
        return sorted_candidates[:5]  # Top 5 alternatives
    
    def _fallback_to_best_candidate(self, candidates: List[Dict[str, Any]]) -> ConsensusResult:
        """Fallback to best single candidate selection"""
        if not candidates:
            raise ValueError("No candidates available for fallback")
        
        best_candidate = max(candidates, 
                           key=lambda x: x.get('confidence_scores', {}).get('final_score', 0.0))
        
        final_score = best_candidate.get('confidence_scores', {}).get('final_score', 0.0)
        
        return ConsensusResult(
            winner_candidate=best_candidate,
            consensus_method=f"{self.name()}_fallback",
            consensus_confidence=final_score,
            consensus_metadata={
                'fallback_used': True,
                'fallback_reason': 'alignment_fusion_failed',
                'original_strategy': self.name()
            },
            alternative_candidates=self._get_alternative_candidates(candidates)
        )

class ConsensusModule:
    """Main consensus processing module"""
    
    def __init__(self, default_strategy: str = "best_single_candidate"):
        self.strategies = {
            "best_single_candidate": BestSingleCandidateStrategy(),
            "weighted_voting": WeightedVotingStrategy(),
            "multidimensional_consensus": MultiDimensionalConsensusStrategy(),
            "confidence_based": ConfidenceBasedStrategy(),
            "alignment_aware_fusion": AlignmentAwareFusionStrategy()
        }
        self.default_strategy = default_strategy
        self.logger = create_enhanced_logger("consensus_module")
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available consensus strategies"""
        return list(self.strategies.keys())
    
    def process_consensus(self, 
                         candidates: List[Dict[str, Any]], 
                         strategy: Optional[str] = None,
                         strategy_params: Optional[Dict[str, Any]] = None,
                         session_bias_list = None) -> ConsensusResult:
        """
        Process consensus using specified strategy
        
        Args:
            candidates: List of scored candidate transcripts
            strategy: Strategy name to use (defaults to instance default)
            strategy_params: Optional parameters for strategy
            session_bias_list: Optional session bias list for adaptive biasing
            
        Returns:
            ConsensusResult with winner and metadata
        """
        if not candidates:
            raise ValueError("No candidates provided for consensus processing")
        
        strategy_name = strategy or self.default_strategy
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(self.strategies.keys())}")
        
        consensus_strategy = self.strategies[strategy_name]
        
        # Apply strategy parameters if provided
        if strategy_params:
            self._apply_strategy_params(consensus_strategy, strategy_params)
        
        self.logger.info(f"Processing consensus with strategy: {strategy_name}", 
                        context={'candidates_count': len(candidates), 'strategy': strategy_name})
        
        try:
            # Apply adaptive biasing if available
            if session_bias_list and hasattr(consensus_strategy, 'apply_session_bias'):
                consensus_strategy.apply_session_bias(session_bias_list)
                self.logger.info("Applied session bias list to consensus strategy",
                               context={'bias_terms': session_bias_list.total_bias_terms if session_bias_list else 0})
            
            result = consensus_strategy.select_consensus(candidates)
            
            self.logger.info(f"Consensus processing completed successfully", 
                           context={'strategy': strategy_name, 
                                   'consensus_confidence': result.consensus_confidence})
            
            return result
            
        except Exception as e:
            self.logger.error(f"Consensus processing failed: {e}", 
                            strategy=strategy_name, error=str(e))
            
            # Fallback to best single candidate if strategy fails
            if strategy_name != "best_single_candidate":
                self.logger.warning("Falling back to best_single_candidate strategy")
                fallback_strategy = self.strategies["best_single_candidate"]
                return fallback_strategy.select_consensus(candidates)
            else:
                raise
    
    def _apply_strategy_params(self, strategy: ConsensusStrategy, params: Dict[str, Any]):
        """Apply parameters to strategy instance"""
        for key, value in params.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
    
    def compare_strategies(self, 
                          candidates: List[Dict[str, Any]], 
                          strategies: Optional[List[str]] = None) -> Dict[str, ConsensusResult]:
        """
        Compare multiple consensus strategies
        
        Args:
            candidates: List of scored candidates
            strategies: List of strategy names to compare (defaults to all)
            
        Returns:
            Dictionary mapping strategy names to consensus results
        """
        if strategies is None:
            strategies = list(self.strategies.keys())
        
        results = {}
        
        for strategy_name in strategies:
            if strategy_name in self.strategies:
                try:
                    result = self.process_consensus(candidates, strategy=strategy_name)
                    results[strategy_name] = result
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy_name} failed during comparison: {e}")
        
        return results