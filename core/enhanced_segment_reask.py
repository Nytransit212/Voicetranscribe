"""
Enhanced Segment-Level Re-Ask System for Ensemble Transcription
Implements per-segment agreement scoring across multiple ASR voters with targeted reprocessing.
"""

import os
import tempfile
import time
import json
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import librosa
import soundfile as sf
from difflib import SequenceMatcher
import logging

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.segment_worklist import SegmentWorklistManager, SegmentFlag, get_worklist_manager
from utils.selective_asr import SelectiveASRProcessor, get_selective_asr_processor
from utils.intelligent_cache import get_cache_manager, cached_operation
from utils.deterministic_processing import get_deterministic_processor
from core.asr_engine import ASREngine
from core.audio_processor import AudioProcessor

# Configure logging
reask_logger = logging.getLogger(__name__)

@dataclass
class SegmentAgreementScore:
    """Agreement score for a specific segment across multiple voters"""
    segment_id: str
    start_time: float
    end_time: float
    voter_count: int
    agreement_metrics: Dict[str, float]
    overall_agreement: float
    consensus_text: str
    voter_texts: List[str]
    voter_confidences: List[float]
    needs_reprocessing: bool
    reprocessing_priority: int

@dataclass
class ChunkingStrategy:
    """Configuration for alternative chunking approaches"""
    strategy_id: str
    strategy_name: str
    chunk_overlap_seconds: float
    chunk_size_seconds: float
    context_window_seconds: float
    decode_temperature: float
    prompt_template: str
    max_attempts: int

@dataclass
class BudgetConfiguration:
    """Budget management configuration for selective reprocessing"""
    max_cost_usd: float
    max_segments_per_file: int
    max_api_calls_per_hour: int
    cost_per_second_audio: float
    priority_threshold: float
    enable_cost_tracking: bool

@dataclass
class ReprocessingResult:
    """Result from segment reprocessing with alternative strategies"""
    segment_id: str
    original_agreement: float
    improved_agreement: float
    original_text: str
    improved_text: str
    strategy_used: str
    cost_incurred: float
    processing_time: float
    improvement_gained: float

class EnhancedSegmentReaskSystem:
    """
    Main system for enhanced segment-level re-ask with agreement scoring and targeted reprocessing.
    """
    
    def __init__(self, 
                 agreement_threshold: float = 0.65,
                 enable_budget_management: bool = True,
                 budget_config: Optional[BudgetConfiguration] = None,
                 target_language: Optional[str] = None):
        """
        Initialize enhanced segment re-ask system.
        
        Args:
            agreement_threshold: Threshold below which segments are flagged for reprocessing
            enable_budget_management: Whether to enable budget-aware processing
            budget_config: Budget configuration for cost management
            target_language: Target language for ASR
        """
        self.agreement_threshold = agreement_threshold
        self.enable_budget_management = enable_budget_management
        self.target_language = target_language
        
        # Initialize budget configuration
        self.budget_config = budget_config or BudgetConfiguration(
            max_cost_usd=5.0,
            max_segments_per_file=20,
            max_api_calls_per_hour=100,
            cost_per_second_audio=0.006,  # OpenAI Whisper pricing
            priority_threshold=0.7,
            enable_cost_tracking=True
        )
        
        # Initialize components
        self.worklist_manager = get_worklist_manager()
        self.selective_asr_processor = get_selective_asr_processor()
        self.cache_manager = get_cache_manager()
        self.deterministic_processor = get_deterministic_processor()
        self.asr_engine = ASREngine()
        self.audio_processor = AudioProcessor()
        
        # Enhanced logging
        self.logger = create_enhanced_logger("enhanced_segment_reask")
        
        # Define alternative chunking strategies
        self.chunking_strategies = self._initialize_chunking_strategies()
        
        # Processing statistics
        self.stats = {
            'segments_analyzed': 0,
            'segments_flagged_low_agreement': 0,
            'segments_reprocessed': 0,
            'segments_improved': 0,
            'total_cost_incurred': 0.0,
            'total_processing_time': 0.0,
            'average_agreement_improvement': 0.0,
            'budget_savings_from_targeting': 0.0
        }
        
        reask_logger.info("Enhanced segment re-ask system initialized")
        reask_logger.info(f"Agreement threshold: {agreement_threshold}")
        reask_logger.info(f"Budget management: {enable_budget_management}")
        if enable_budget_management:
            reask_logger.info(f"Budget limit: ${self.budget_config.max_cost_usd}")
    
    def _initialize_chunking_strategies(self) -> List[ChunkingStrategy]:
        """Initialize alternative chunking strategies for reprocessing."""
        return [
            # Strategy 1: High overlap, smaller chunks for precision
            ChunkingStrategy(
                strategy_id="precision_overlap",
                strategy_name="Precision with High Overlap",
                chunk_overlap_seconds=2.0,
                chunk_size_seconds=8.0,
                context_window_seconds=1.0,
                decode_temperature=0.0,
                prompt_template="Transcribe this audio segment with high precision, focusing on speaker transitions.",
                max_attempts=2
            ),
            # Strategy 2: Larger context window for better understanding
            ChunkingStrategy(
                strategy_id="context_aware",
                strategy_name="Context-Aware Processing",
                chunk_overlap_seconds=1.5,
                chunk_size_seconds=12.0,
                context_window_seconds=3.0,
                decode_temperature=0.1,
                prompt_template="Transcribe this audio with attention to context and speaker patterns.",
                max_attempts=2
            ),
            # Strategy 3: Conservative approach with minimal temperature
            ChunkingStrategy(
                strategy_id="conservative_stable",
                strategy_name="Conservative Stable Processing",
                chunk_overlap_seconds=1.0,
                chunk_size_seconds=6.0,
                context_window_seconds=0.5,
                decode_temperature=0.0,
                prompt_template="Provide accurate transcription focusing on clear speech patterns.",
                max_attempts=3
            ),
            # Strategy 4: Aggressive approach for difficult segments
            ChunkingStrategy(
                strategy_id="aggressive_resolve",
                strategy_name="Aggressive Resolution",
                chunk_overlap_seconds=3.0,
                chunk_size_seconds=10.0,
                context_window_seconds=2.0,
                decode_temperature=0.2,
                prompt_template="Resolve difficult audio segments with multiple speaker overlaps.",
                max_attempts=1
            )
        ]
    
    def analyze_candidates_for_agreement(self, candidates: List[Dict[str, Any]], 
                                       file_path: str, run_id: str) -> List[SegmentAgreementScore]:
        """
        Analyze candidate transcripts to calculate per-segment agreement scores.
        
        Args:
            candidates: List of candidate transcripts from ensemble processing
            file_path: Path to original audio file
            run_id: Processing run identifier
            
        Returns:
            List of segment agreement scores
        """
        start_time = time.time()
        
        self.logger.info(f"Analyzing {len(candidates)} candidates for agreement scoring")
        
        # Extract all unique time segments across candidates
        time_segments = self._extract_unified_time_segments(candidates)
        
        # Calculate agreement scores for each time segment
        agreement_scores = []
        
        for segment_info in time_segments:
            segment_id = f"seg_{segment_info['start']:.1f}_{segment_info['end']:.1f}"
            
            # Collect voter information for this time window
            voter_data = self._collect_voter_data_for_segment(
                candidates, segment_info['start'], segment_info['end']
            )
            
            if len(voter_data['texts']) < 2:
                # Skip segments with insufficient voters
                continue
            
            # Calculate agreement metrics
            agreement_metrics = self._calculate_agreement_metrics(voter_data)
            
            # Determine overall agreement score
            overall_agreement = self._calculate_overall_agreement(agreement_metrics)
            
            # Determine if reprocessing is needed
            needs_reprocessing = overall_agreement < self.agreement_threshold
            
            # Calculate priority based on disagreement severity and segment importance
            priority = self._calculate_reprocessing_priority(
                overall_agreement, segment_info, voter_data
            )
            
            agreement_score = SegmentAgreementScore(
                segment_id=segment_id,
                start_time=segment_info['start'],
                end_time=segment_info['end'],
                voter_count=len(voter_data['texts']),
                agreement_metrics=agreement_metrics,
                overall_agreement=overall_agreement,
                consensus_text=self._generate_consensus_text(voter_data['texts']),
                voter_texts=voter_data['texts'],
                voter_confidences=voter_data['confidences'],
                needs_reprocessing=needs_reprocessing,
                reprocessing_priority=priority
            )
            
            agreement_scores.append(agreement_score)
        
        # Update statistics
        self.stats['segments_analyzed'] += len(agreement_scores)
        flagged_count = sum(1 for score in agreement_scores if score.needs_reprocessing)
        self.stats['segments_flagged_low_agreement'] += flagged_count
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"Agreement analysis complete: {len(agreement_scores)} segments analyzed, "
                        f"{flagged_count} flagged for reprocessing in {processing_time:.2f}s")
        
        return agreement_scores
    
    def _extract_unified_time_segments(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Extract unified time segments across all candidates for consistent analysis."""
        all_segments = []
        
        # Collect all segment boundaries
        boundaries = set()
        for candidate in candidates:
            segments = candidate.get('aligned_segments', [])
            for segment in segments:
                boundaries.add(segment.get('start', 0.0))
                boundaries.add(segment.get('end', 0.0))
        
        # Sort boundaries and create unified segments
        sorted_boundaries = sorted(boundaries)
        
        for i in range(len(sorted_boundaries) - 1):
            start = sorted_boundaries[i]
            end = sorted_boundaries[i + 1]
            
            # Skip very short segments (< 0.5 seconds)
            if end - start >= 0.5:
                all_segments.append({
                    'start': start,
                    'end': end,
                    'duration': end - start
                })
        
        return all_segments
    
    def _collect_voter_data_for_segment(self, candidates: List[Dict[str, Any]], 
                                      start_time: float, end_time: float) -> Dict[str, List]:
        """Collect text and confidence data from all voters for a specific time segment."""
        voter_data = {
            'texts': [],
            'confidences': [],
            'candidate_ids': []
        }
        
        for candidate in candidates:
            segments = candidate.get('aligned_segments', [])
            
            # Find overlapping segments
            overlapping_text = ""
            segment_confidences = []
            
            for segment in segments:
                seg_start = segment.get('start', 0.0)
                seg_end = segment.get('end', 0.0)
                
                # Check for overlap with target time window
                overlap_start = max(start_time, seg_start)
                overlap_end = min(end_time, seg_end)
                
                if overlap_end > overlap_start:
                    # There's an overlap
                    segment_text = segment.get('text', '').strip()
                    
                    # Calculate overlap ratio
                    overlap_duration = overlap_end - overlap_start
                    segment_duration = seg_end - seg_start
                    overlap_ratio = overlap_duration / max(segment_duration, 0.1)
                    
                    if overlap_ratio > 0.3:  # At least 30% overlap
                        overlapping_text += " " + segment_text
                        
                        # Collect word-level confidences if available
                        words = segment.get('words', [])
                        for word in words:
                            word_start = word.get('start', 0.0)
                            word_end = word.get('end', 0.0)
                            
                            if word_start >= start_time and word_end <= end_time:
                                word_conf = word.get('confidence', 0.5)
                                segment_confidences.append(word_conf)
            
            if overlapping_text.strip():
                voter_data['texts'].append(overlapping_text.strip())
                
                # Calculate average confidence for this voter's contribution
                avg_confidence = np.mean(segment_confidences) if segment_confidences else 0.5
                voter_data['confidences'].append(avg_confidence)
                voter_data['candidate_ids'].append(candidate.get('candidate_id', 'unknown'))
        
        return voter_data
    
    def _calculate_agreement_metrics(self, voter_data: Dict[str, List]) -> Dict[str, float]:
        """Calculate various agreement metrics between voters."""
        texts = voter_data['texts']
        confidences = voter_data['confidences']
        
        if len(texts) < 2:
            return {'lexical_similarity': 0.0, 'length_consistency': 0.0, 
                   'confidence_agreement': 0.0, 'edit_distance_agreement': 0.0}
        
        # 1. Lexical similarity using TF-IDF
        lexical_similarity = self._calculate_lexical_similarity(texts)
        
        # 2. Length consistency
        lengths = [len(text.split()) for text in texts]
        length_std = np.std(lengths) if len(lengths) > 1 else 0.0
        length_consistency = max(0.0, 1.0 - (length_std / max(float(np.mean(lengths)), 1.0)))
        
        # 3. Confidence agreement
        confidence_agreement = 1.0 - np.std(confidences) if len(confidences) > 1 else 1.0
        
        # 4. Edit distance agreement
        edit_distance_agreement = self._calculate_edit_distance_agreement(texts)
        
        return {
            'lexical_similarity': float(lexical_similarity),
            'length_consistency': float(length_consistency),
            'confidence_agreement': float(confidence_agreement),
            'edit_distance_agreement': float(edit_distance_agreement)
        }
    
    def _calculate_lexical_similarity(self, texts: List[str]) -> float:
        """Calculate lexical similarity between texts using word overlap."""
        if len(texts) < 2:
            return 1.0
        
        # Convert to word sets
        word_sets = [set(text.lower().split()) for text in texts]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                set1, set2 = word_sets[i], word_sets[j]
                
                if not set1 and not set2:
                    similarities.append(1.0)
                elif not set1 or not set2:
                    similarities.append(0.0)
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = intersection / union if union > 0 else 0.0
                    similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _calculate_edit_distance_agreement(self, texts: List[str]) -> float:
        """Calculate agreement based on normalized edit distances."""
        if len(texts) < 2:
            return 1.0
        
        # Calculate pairwise edit distance agreements
        agreements = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                text1, text2 = texts[i], texts[j]
                
                # Use SequenceMatcher for normalized similarity
                similarity = SequenceMatcher(None, text1, text2).ratio()
                agreements.append(similarity)
        
        return float(np.mean(agreements)) if agreements else 0.0
    
    def _calculate_overall_agreement(self, agreement_metrics: Dict[str, float]) -> float:
        """Calculate overall agreement score from individual metrics."""
        # Weighted combination of different agreement metrics
        weights = {
            'lexical_similarity': 0.3,
            'length_consistency': 0.2,
            'confidence_agreement': 0.2,
            'edit_distance_agreement': 0.3
        }
        
        overall_score = 0.0
        for metric, weight in weights.items():
            score = agreement_metrics.get(metric, 0.0)
            overall_score += weight * score
        
        return overall_score
    
    def _generate_consensus_text(self, texts: List[str]) -> str:
        """Generate consensus text from multiple voter texts."""
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]
        
        # Simple approach: use the most common text or longest if all different
        text_counts = Counter(texts)
        most_common = text_counts.most_common(1)[0]
        
        if most_common[1] > 1:  # Multiple voters agree
            return most_common[0]
        else:
            # No consensus, return longest text
            return max(texts, key=len)
    
    def _calculate_reprocessing_priority(self, agreement_score: float, 
                                       segment_info: Dict[str, float],
                                       voter_data: Dict[str, List]) -> int:
        """Calculate priority for reprocessing (1-10, higher = more urgent)."""
        # Base priority from disagreement severity
        disagreement = 1.0 - agreement_score
        base_priority = int(disagreement * 10)
        
        # Adjust for segment duration (longer segments get higher priority)
        duration_factor = min(segment_info['duration'] / 10.0, 1.0)  # Cap at 10 seconds
        
        # Adjust for confidence variance (higher variance = higher priority)
        confidences = voter_data['confidences']
        confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0
        variance_factor = min(float(confidence_variance) * 5, 1.0)
        
        # Calculate final priority
        final_priority = base_priority + duration_factor * 2 + variance_factor * 2
        
        return max(1, min(10, int(final_priority)))
    
    def process_segments_with_enhanced_reask(self, agreement_scores: List[SegmentAgreementScore],
                                           file_path: str, run_id: str,
                                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process flagged segments using enhanced re-ask with alternative chunking strategies.
        
        Args:
            agreement_scores: List of segment agreement scores
            file_path: Path to original audio file
            run_id: Processing run identifier
            progress_callback: Optional progress callback
            
        Returns:
            Processing results with improvements and metrics
        """
        start_time = time.time()
        
        # Filter segments that need reprocessing
        segments_to_reprocess = [score for score in agreement_scores if score.needs_reprocessing]
        
        if not segments_to_reprocess:
            self.logger.info("No segments require reprocessing")
            return {'segments_reprocessed': 0, 'segments_improved': 0, 'cost_incurred': 0.0}
        
        # Sort by priority (highest first)
        segments_to_reprocess.sort(key=lambda x: x.reprocessing_priority, reverse=True)
        
        # Apply budget constraints
        if self.enable_budget_management:
            segments_to_reprocess = self._apply_budget_constraints(segments_to_reprocess, file_path)
        
        self.logger.info(f"Reprocessing {len(segments_to_reprocess)} segments with enhanced strategies")
        
        # Load original audio for segment extraction
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        except Exception as e:
            self.logger.error(f"Failed to load audio file {file_path}: {e}")
            return {'error': f"Failed to load audio: {e}"}
        
        # Process segments with alternative chunking strategies
        reprocessing_results = []
        total_cost = 0.0
        
        for i, segment_score in enumerate(segments_to_reprocess):
            if progress_callback:
                progress = int((i / len(segments_to_reprocess)) * 100)
                progress_callback(f"Reprocessing segment {i+1}/{len(segments_to_reprocess)}", progress)
            
            try:
                result = self._reprocess_segment_with_strategies(
                    segment_score, audio_data, int(sample_rate), file_path
                )
                
                if result:
                    reprocessing_results.append(result)
                    total_cost += result.cost_incurred
                    
                    # Check budget limits
                    if (self.enable_budget_management and 
                        total_cost >= self.budget_config.max_cost_usd):
                        self.logger.warning(f"Budget limit reached: ${total_cost:.4f}")
                        break
                
            except Exception as e:
                self.logger.error(f"Failed to reprocess segment {segment_score.segment_id}: {e}")
                continue
        
        # Calculate summary metrics
        processing_time = time.time() - start_time
        improved_count = sum(1 for r in reprocessing_results if r.improvement_gained > 0.05)
        
        avg_improvement = (sum(r.improvement_gained for r in reprocessing_results) / 
                          max(len(reprocessing_results), 1))
        
        # Update statistics
        self.stats['segments_reprocessed'] += len(reprocessing_results)
        self.stats['segments_improved'] += improved_count
        self.stats['total_cost_incurred'] += total_cost
        self.stats['total_processing_time'] += processing_time
        self.stats['average_agreement_improvement'] = (
            (self.stats['average_agreement_improvement'] * (self.stats['segments_reprocessed'] - len(reprocessing_results)) + 
             avg_improvement * len(reprocessing_results)) / self.stats['segments_reprocessed']
        )
        
        summary = {
            'segments_analyzed': len(agreement_scores),
            'segments_flagged': len([s for s in agreement_scores if s.needs_reprocessing]),
            'segments_reprocessed': len(reprocessing_results),
            'segments_improved': improved_count,
            'total_cost_incurred': total_cost,
            'average_improvement': avg_improvement,
            'processing_time': processing_time,
            'reprocessing_results': reprocessing_results
        }
        
        self.logger.info(f"Enhanced re-ask complete: {improved_count}/{len(reprocessing_results)} segments improved")
        return summary
    
    def _apply_budget_constraints(self, segments: List[SegmentAgreementScore], 
                                file_path: str) -> List[SegmentAgreementScore]:
        """Apply budget constraints to limit reprocessing."""
        if not self.enable_budget_management:
            return segments
        
        # Estimate cost for each segment
        segments_with_cost = []
        for segment in segments:
            duration = segment.end_time - segment.start_time
            estimated_cost = duration * self.budget_config.cost_per_second_audio
            segments_with_cost.append((segment, estimated_cost))
        
        # Select segments within budget, prioritizing high-priority ones
        selected_segments = []
        total_estimated_cost = 0.0
        
        for segment, cost in segments_with_cost:
            if (total_estimated_cost + cost <= self.budget_config.max_cost_usd and
                len(selected_segments) < self.budget_config.max_segments_per_file):
                selected_segments.append(segment)
                total_estimated_cost += cost
            else:
                break
        
        saved_cost = sum(cost for _, cost in segments_with_cost[len(selected_segments):])
        self.stats['budget_savings_from_targeting'] += saved_cost
        
        if len(selected_segments) < len(segments):
            self.logger.info(f"Budget constraints applied: selected {len(selected_segments)}/{len(segments)} segments, "
                           f"estimated savings: ${saved_cost:.4f}")
        
        return selected_segments
    
    def _reprocess_segment_with_strategies(self, segment_score: SegmentAgreementScore,
                                         audio_data: np.ndarray, sample_rate: int,
                                         original_file_path: str) -> Optional[ReprocessingResult]:
        """Reprocess a single segment using alternative chunking strategies."""
        start_time = time.time()
        
        # Extract audio segment with context
        segment_audio_path = self._extract_segment_audio(
            audio_data, sample_rate, segment_score, original_file_path
        )
        
        if not segment_audio_path:
            return None
        
        try:
            # Try different chunking strategies
            best_result = None
            best_improvement = 0.0
            
            for strategy in self.chunking_strategies:
                try:
                    result = self._apply_chunking_strategy(
                        segment_audio_path, segment_score, strategy
                    )
                    
                    if result and result['improvement'] > best_improvement:
                        best_improvement = result['improvement']
                        best_result = result
                        best_result['strategy_used'] = strategy.strategy_id
                
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy.strategy_id} failed for segment {segment_score.segment_id}: {e}")
                    continue
            
            # Clean up temporary file
            try:
                os.unlink(segment_audio_path)
            except:
                pass
            
            if not best_result:
                return None
            
            # Create reprocessing result
            processing_time = time.time() - start_time
            segment_duration = segment_score.end_time - segment_score.start_time
            cost_incurred = segment_duration * self.budget_config.cost_per_second_audio
            
            return ReprocessingResult(
                segment_id=segment_score.segment_id,
                original_agreement=segment_score.overall_agreement,
                improved_agreement=best_result['new_agreement'],
                original_text=segment_score.consensus_text,
                improved_text=best_result['improved_text'],
                strategy_used=best_result['strategy_used'],
                cost_incurred=cost_incurred,
                processing_time=processing_time,
                improvement_gained=best_result['improvement']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to reprocess segment {segment_score.segment_id}: {e}")
            return None
    
    def _extract_segment_audio(self, audio_data: np.ndarray, sample_rate: int,
                             segment_score: SegmentAgreementScore, 
                             original_file_path: str) -> Optional[str]:
        """Extract audio segment to temporary file with context padding."""
        try:
            # Calculate sample indices with context padding
            context_padding = 1.0  # 1 second padding on each side
            start_time = max(0.0, segment_score.start_time - context_padding)
            end_time = min(len(audio_data) / sample_rate, segment_score.end_time + context_padding)
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Extract segment
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) == 0:
                return None
            
            # Create temporary file
            temp_dir = tempfile.mkdtemp(prefix='enhanced_reask_')
            segment_filename = f"{segment_score.segment_id}_{int(start_time)}_{int(end_time)}.wav"
            segment_path = os.path.join(temp_dir, segment_filename)
            
            # Save segment
            sf.write(segment_path, segment_audio, sample_rate)
            
            return segment_path
            
        except Exception as e:
            self.logger.error(f"Failed to extract segment audio: {e}")
            return None
    
    def _apply_chunking_strategy(self, audio_path: str, segment_score: SegmentAgreementScore,
                               strategy: ChunkingStrategy) -> Optional[Dict[str, Any]]:
        """Apply a specific chunking strategy to reprocess a segment."""
        try:
            # Prepare ASR parameters based on strategy
            asr_params = {
                'model': 'whisper-1',
                'response_format': 'verbose_json',
                'temperature': strategy.decode_temperature,
                'prompt': strategy.prompt_template,
                'language': self.target_language
            }
            
            # Remove None values
            asr_params = {k: v for k, v in asr_params.items() if v is not None}
            
            # Make ASR call
            response = self.asr_engine._make_transcription_api_call(audio_path, **asr_params)
            
            if not response or not hasattr(response, 'text'):
                return None
            
            improved_text = response.text.strip()
            
            # Calculate improvement by comparing with original voter texts
            new_agreement = self._calculate_improvement_agreement(
                improved_text, segment_score.voter_texts
            )
            
            improvement = new_agreement - segment_score.overall_agreement
            
            return {
                'improved_text': improved_text,
                'new_agreement': new_agreement,
                'improvement': improvement
            }
            
        except Exception as e:
            self.logger.error(f"Failed to apply chunking strategy {strategy.strategy_id}: {e}")
            return None
    
    def _calculate_improvement_agreement(self, improved_text: str, original_texts: List[str]) -> float:
        """Calculate agreement between improved text and original voter texts."""
        if not original_texts:
            return 0.0
        
        # Calculate similarities with each original text
        similarities = []
        for original_text in original_texts:
            similarity = SequenceMatcher(None, improved_text, original_text).ratio()
            similarities.append(similarity)
        
        # Return average similarity as agreement score
        return float(np.mean(similarities))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current status and statistics of the enhanced re-ask system."""
        return {
            'configuration': {
                'agreement_threshold': self.agreement_threshold,
                'budget_management_enabled': self.enable_budget_management,
                'budget_limit_usd': self.budget_config.max_cost_usd if self.enable_budget_management else None,
                'max_segments_per_file': self.budget_config.max_segments_per_file,
                'target_language': self.target_language,
                'chunking_strategies_count': len(self.chunking_strategies)
            },
            'statistics': self.stats.copy(),
            'performance_metrics': {
                'improvement_rate': (self.stats['segments_improved'] / 
                                   max(self.stats['segments_reprocessed'], 1)),
                'cost_per_improvement': (self.stats['total_cost_incurred'] / 
                                       max(self.stats['segments_improved'], 1)),
                'processing_efficiency': (self.stats['segments_improved'] / 
                                        max(self.stats['segments_analyzed'], 1)),
                'average_processing_time_per_segment': (self.stats['total_processing_time'] / 
                                                      max(self.stats['segments_reprocessed'], 1))
            },
            'chunking_strategies': [
                {
                    'strategy_id': strategy.strategy_id,
                    'strategy_name': strategy.strategy_name,
                    'chunk_size': strategy.chunk_size_seconds,
                    'overlap': strategy.chunk_overlap_seconds
                }
                for strategy in self.chunking_strategies
            ]
        }

# Global instance for easy access
_enhanced_reask_system = None

def get_enhanced_reask_system(agreement_threshold: float = 0.65,
                             enable_budget_management: bool = True,
                             budget_config: Optional[BudgetConfiguration] = None,
                             target_language: Optional[str] = None) -> EnhancedSegmentReaskSystem:
    """Get or create the global enhanced re-ask system instance."""
    global _enhanced_reask_system
    
    if _enhanced_reask_system is None:
        _enhanced_reask_system = EnhancedSegmentReaskSystem(
            agreement_threshold=agreement_threshold,
            enable_budget_management=enable_budget_management,
            budget_config=budget_config,
            target_language=target_language
        )
    
    return _enhanced_reask_system