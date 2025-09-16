"""
Intelligent Controller for Confidence-Based Decode Expansion

Implements adaptive ASR strategy with 3-7 decodes per segment based on 
confidence thresholds and inter-candidate agreement analysis.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import Counter

from .asr_providers.base import ASRProvider, ASRResult, ASRSegment, DecodeMode
from .asr_providers.factory import ASRProviderFactory
from .alignment_fusion import TemporalAligner, WordAlignment
from .fusion_engine import FusionEngine, FusionResult
from utils.enhanced_structured_logger import create_enhanced_logger

logger = logging.getLogger(__name__)

@dataclass
class SegmentCandidate:
    """Single ASR candidate for a segment"""
    provider: str
    decode_mode: DecodeMode
    model_name: str
    result: ASRResult
    calibrated_confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SegmentAnalysis:
    """Analysis results for a segment including agreement metrics"""
    segment_start: float
    segment_end: float
    segment_duration: float
    candidates: List[SegmentCandidate]
    word_alignments: List[WordAlignment]
    agreement_score: float
    confidence_score: float
    best_candidate: Optional[SegmentCandidate]
    expansion_decision: str  # "stop_early", "expand_standard", "expand_maximum"
    total_decodes_run: int
    fusion_result: Optional[FusionResult] = None
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntelligentControllerResult:
    """Final result from intelligent controller processing"""
    segments: List[SegmentAnalysis]
    overall_confidence: float
    total_processing_time: float
    total_decodes_run: int
    early_stops: int
    standard_expansions: int
    maximum_expansions: int
    controller_metadata: Dict[str, Any] = field(default_factory=dict)

class IntelligentController:
    """
    Intelligent ASR Controller implementing confidence-based decode expansion
    
    Strategy:
    1. Run 3 initial probes per segment (Whisper careful/deterministic, Deepgram base)
    2. Calculate calibrated confidence and token agreement
    3. If confidence ≥ 0.90 and agreement ≥ 0.85, stop early
    4. Otherwise expand with up to 2 more decodes
    5. Cap at maximum 7 decodes per segment
    6. Use word-level alignment for agreement calculation
    7. Select best candidate using confidence scoring and temporal coherence
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.90,
                 agreement_threshold: float = 0.85,
                 max_decodes_per_segment: int = 7,
                 segment_duration_range: Tuple[float, float] = (30.0, 45.0),
                 provider_config: Optional[Dict[str, Any]] = None,
                 enable_fusion: bool = True,
                 fusion_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Intelligent Controller
        
        Args:
            confidence_threshold: Minimum confidence to trigger early stop
            agreement_threshold: Minimum agreement to trigger early stop  
            max_decodes_per_segment: Maximum decodes allowed per segment
            segment_duration_range: Target segment duration range in seconds
            provider_config: Optional configuration for ASR providers
        """
        self.confidence_threshold = confidence_threshold
        self.agreement_threshold = agreement_threshold
        self.max_decodes_per_segment = max_decodes_per_segment
        self.segment_duration_range = segment_duration_range
        self.provider_config = provider_config or {}
        
        # Initialize logger
        self.logger = create_enhanced_logger("intelligent_controller")
        
        # Initialize ASR providers
        self.providers = self._initialize_providers()
        
        # Initialize temporal aligner for word-level alignment
        self.temporal_aligner = TemporalAligner(
            timestamp_tolerance=0.3,
            confidence_threshold=0.1,
            max_alignment_gap=1.0
        )
        
        # Initialize fusion engine if enabled
        self.enable_fusion = enable_fusion
        self.fusion_engine = None
        if self.enable_fusion:
            fusion_config = fusion_config or {}
            self.fusion_engine = FusionEngine(
                engine_weights=fusion_config.get('engine_weights'),
                temporal_coherence_config=fusion_config.get('temporal_coherence_config'),
                entity_detection_enabled=fusion_config.get('entity_detection_enabled', True),
                mbr_config=fusion_config.get('mbr_config')
            )
        
        # Decode strategy configuration
        self.initial_probe_configs = [
            ("faster-whisper", DecodeMode.CAREFUL),
            ("faster-whisper", DecodeMode.DETERMINISTIC), 
            ("deepgram", DecodeMode.DETERMINISTIC)
        ]
        
        self.expansion_configs = [
            ("deepgram", DecodeMode.ENHANCED),
            ("faster-whisper", DecodeMode.EXPLORATORY),
            ("openai", DecodeMode.DETERMINISTIC),
            ("deepgram", DecodeMode.FAST)
        ]
        
        self.logger.info("Intelligent Controller initialized", 
                        context={
                            'confidence_threshold': confidence_threshold,
                            'agreement_threshold': agreement_threshold,
                            'max_decodes': max_decodes_per_segment,
                            'available_providers': list(self.providers.keys()),
                            'fusion_enabled': self.enable_fusion,
                            'fusion_engine': self.fusion_engine is not None
                        })
    
    def _initialize_providers(self) -> Dict[str, ASRProvider]:
        """Initialize available ASR providers"""
        providers = {}
        
        try:
            # Get available providers from factory
            available_provider_names = ASRProviderFactory.get_available_providers()
            
            for provider_name in available_provider_names:
                try:
                    provider = ASRProviderFactory.create_provider(
                        provider_name, 
                        config=self.provider_config.get(provider_name, {})
                    )
                    if provider.is_available():
                        providers[provider_name] = provider
                        self.logger.info(f"Initialized provider: {provider_name}")
                    else:
                        self.logger.warning(f"Provider {provider_name} not available")
                except Exception as e:
                    self.logger.error(f"Failed to initialize provider {provider_name}: {e}")
            
            if not providers:
                raise RuntimeError("No ASR providers available")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize providers: {e}")
            raise
        
        return providers
    
    def process_audio_segments(self, 
                              audio_path: Union[str, Path],
                              segments: Optional[List[Dict[str, Any]]] = None) -> IntelligentControllerResult:
        """
        Process audio through intelligent controller with confidence-based expansion
        
        Args:
            audio_path: Path to audio file
            segments: Optional pre-defined segments (will auto-segment if None)
            
        Returns:
            IntelligentControllerResult with processed segments and metadata
        """
        start_time = time.time()
        
        self.logger.info("Starting intelligent controller processing", 
                        context={'audio_path': str(audio_path)})
        
        # Auto-segment if not provided
        if segments is None:
            segments = self._auto_segment_audio(audio_path)
        
        # Process each segment
        segment_analyses = []
        total_decodes_run = 0
        early_stops = 0
        standard_expansions = 0
        maximum_expansions = 0
        
        for i, segment in enumerate(segments):
            self.logger.info(f"Processing segment {i+1}/{len(segments)}", 
                           context={'segment_start': segment.get('start', 0),
                                  'segment_end': segment.get('end', 0)})
            
            # Process single segment with adaptive strategy
            segment_analysis = self._process_single_segment(audio_path, segment, i)
            segment_analyses.append(segment_analysis)
            
            # Update statistics
            total_decodes_run += segment_analysis.total_decodes_run
            
            if segment_analysis.expansion_decision == "stop_early":
                early_stops += 1
            elif segment_analysis.expansion_decision == "expand_standard":
                standard_expansions += 1
            elif segment_analysis.expansion_decision == "expand_maximum":
                maximum_expansions += 1
        
        # Calculate overall metrics
        overall_confidence = self._calculate_overall_confidence(segment_analyses)
        processing_time = time.time() - start_time
        
        result = IntelligentControllerResult(
            segments=segment_analyses,
            overall_confidence=overall_confidence,
            total_processing_time=processing_time,
            total_decodes_run=total_decodes_run,
            early_stops=early_stops,
            standard_expansions=standard_expansions,
            maximum_expansions=maximum_expansions,
            controller_metadata={
                'audio_path': str(audio_path),
                'total_segments': len(segments),
                'avg_decodes_per_segment': total_decodes_run / len(segments) if segments else 0,
                'early_stop_rate': early_stops / len(segments) if segments else 0
            }
        )
        
        self.logger.info("Intelligent controller processing complete", 
                        context={
                            'total_segments': len(segments),
                            'total_decodes': total_decodes_run,
                            'early_stops': early_stops,
                            'processing_time': processing_time,
                            'overall_confidence': overall_confidence
                        })
        
        return result
    
    def _auto_segment_audio(self, audio_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Auto-segment audio into 30-45s chunks"""
        # This is a simplified implementation - in practice you might use
        # voice activity detection or existing diarization boundaries
        
        # For now, create fixed-duration segments
        try:
            # Get audio duration (simplified - would use librosa or similar)
            import librosa
            duration = librosa.get_duration(filename=str(audio_path))
        except:
            # Fallback - assume 60 seconds for testing
            duration = 60.0
        
        segments = []
        segment_size = (self.segment_duration_range[0] + self.segment_duration_range[1]) / 2.0
        
        current_time = 0.0
        segment_id = 0
        
        while current_time < duration:
            end_time = min(current_time + segment_size, duration)
            
            segments.append({
                'segment_id': segment_id,
                'start': current_time,
                'end': end_time,
                'duration': end_time - current_time
            })
            
            current_time = end_time
            segment_id += 1
        
        self.logger.info(f"Auto-segmented audio into {len(segments)} segments",
                        context={'total_duration': duration, 'segment_size': segment_size})
        
        return segments
    
    def _process_single_segment(self, 
                               audio_path: Union[str, Path], 
                               segment: Dict[str, Any], 
                               segment_idx: int) -> SegmentAnalysis:
        """
        Process single segment with adaptive decode expansion strategy
        
        Args:
            audio_path: Path to audio file
            segment: Segment definition with start/end times
            segment_idx: Index of segment for logging
            
        Returns:
            SegmentAnalysis with candidates and decision metadata
        """
        segment_start = segment.get('start', 0.0)
        segment_end = segment.get('end', 0.0)
        segment_duration = segment_end - segment_start
        
        self.logger.info(f"Processing segment {segment_idx}: {segment_start:.1f}s - {segment_end:.1f}s",
                        context={'segment_duration': segment_duration})
        
        candidates = []
        
        # Phase 1: Initial 3 probes
        self.logger.info("Phase 1: Running initial 3 probes")
        for provider_name, decode_mode in self.initial_probe_configs:
            if provider_name in self.providers:
                candidate = self._run_single_decode(
                    audio_path, segment, provider_name, decode_mode, segment_idx
                )
                if candidate:
                    candidates.append(candidate)
        
        # Analyze initial results
        if len(candidates) >= 2:
            agreement_score = self._calculate_agreement_score(candidates)
            confidence_score = self._calculate_confidence_score(candidates)
            
            self.logger.info(f"Initial analysis: confidence={confidence_score:.3f}, agreement={agreement_score:.3f}")
            
            # Decision: Early stop?
            if (confidence_score >= self.confidence_threshold and 
                agreement_score >= self.agreement_threshold):
                
                expansion_decision = "stop_early"
                self.logger.info("Early stop triggered - high confidence and agreement")
                
            else:
                # Phase 2: Standard expansion (up to 2 more decodes)
                self.logger.info("Phase 2: Standard expansion - adding 2 more decodes")
                expansion_decision = "expand_standard"
                
                added_decodes = 0
                for provider_name, decode_mode in self.expansion_configs:
                    if (added_decodes < 2 and 
                        len(candidates) < self.max_decodes_per_segment and
                        provider_name in self.providers):
                        
                        # Avoid duplicate provider+mode combinations
                        existing_configs = {(c.provider, c.decode_mode) for c in candidates}
                        if (provider_name, decode_mode) not in existing_configs:
                            candidate = self._run_single_decode(
                                audio_path, segment, provider_name, decode_mode, segment_idx
                            )
                            if candidate:
                                candidates.append(candidate)
                                added_decodes += 1
                
                # Re-analyze after standard expansion
                if len(candidates) >= 2:
                    agreement_score = self._calculate_agreement_score(candidates)
                    confidence_score = self._calculate_confidence_score(candidates)
                    
                    self.logger.info(f"Post-expansion analysis: confidence={confidence_score:.3f}, agreement={agreement_score:.3f}")
                    
                    # Decision: Maximum expansion?
                    if (confidence_score < self.confidence_threshold * 0.8 or 
                        agreement_score < self.agreement_threshold * 0.8):
                        
                        # Phase 3: Maximum expansion (fill remaining decode slots)
                        self.logger.info("Phase 3: Maximum expansion - using remaining decode slots")
                        expansion_decision = "expand_maximum"
                        
                        # Add remaining decodes up to maximum
                        for provider_name, decode_mode in self.expansion_configs[2:]:  # Remaining configs
                            if (len(candidates) < self.max_decodes_per_segment and
                                provider_name in self.providers):
                                
                                existing_configs = {(c.provider, c.decode_mode) for c in candidates}
                                if (provider_name, decode_mode) not in existing_configs:
                                    candidate = self._run_single_decode(
                                        audio_path, segment, provider_name, decode_mode, segment_idx
                                    )
                                    if candidate:
                                        candidates.append(candidate)
        else:
            # Fallback if initial probes failed
            expansion_decision = "expand_maximum"
            confidence_score = 0.0
            agreement_score = 0.0
        
        # Final analysis and best candidate selection
        fusion_result = None
        if candidates:
            word_alignments = self._create_word_alignments(candidates)
            agreement_score = self._calculate_agreement_score(candidates)
            confidence_score = self._calculate_confidence_score(candidates)
            best_candidate = self._select_best_candidate(candidates, word_alignments)
            
            # Apply fusion if enabled
            if self.enable_fusion and self.fusion_engine and len(candidates) >= 2:
                try:
                    # Create temporary SegmentAnalysis for fusion
                    temp_analysis = SegmentAnalysis(
                        segment_start=segment_start,
                        segment_end=segment_end,
                        segment_duration=segment_duration,
                        candidates=candidates,
                        word_alignments=word_alignments,
                        agreement_score=agreement_score,
                        confidence_score=confidence_score,
                        best_candidate=best_candidate,
                        expansion_decision=expansion_decision,
                        total_decodes_run=len(candidates)
                    )
                    
                    # Apply fusion
                    fusion_result = self.fusion_engine.fuse_segment_candidates(temp_analysis)
                    
                    self.logger.info(f"Fusion applied: confidence={fusion_result.overall_confidence:.3f}, "
                                   f"networks={len(fusion_result.confusion_networks)}")
                    
                except Exception as e:
                    self.logger.error(f"Fusion failed for segment {segment_idx}: {e}")
                    fusion_result = None
        else:
            word_alignments = []
            agreement_score = 0.0
            confidence_score = 0.0
            best_candidate = None
        
        analysis = SegmentAnalysis(
            segment_start=segment_start,
            segment_end=segment_end,
            segment_duration=segment_duration,
            candidates=candidates,
            word_alignments=word_alignments,
            agreement_score=agreement_score,
            confidence_score=confidence_score,
            best_candidate=best_candidate,
            expansion_decision=expansion_decision,
            total_decodes_run=len(candidates),
            fusion_result=fusion_result,
            analysis_metadata={
                'segment_idx': segment_idx,
                'initial_probes': min(3, len(candidates)),
                'expansion_decodes': max(0, len(candidates) - 3),
                'final_confidence': confidence_score,
                'final_agreement': agreement_score,
                'fusion_enabled': self.enable_fusion,
                'fusion_applied': fusion_result is not None
            }
        )
        
        self.logger.info(f"Segment {segment_idx} complete: {len(candidates)} decodes, decision={expansion_decision}",
                        context={'confidence': confidence_score, 'agreement': agreement_score})
        
        return analysis
    
    def _run_single_decode(self, 
                          audio_path: Union[str, Path],
                          segment: Dict[str, Any],
                          provider_name: str,
                          decode_mode: DecodeMode,
                          segment_idx: int) -> Optional[SegmentCandidate]:
        """
        Run single ASR decode for a segment
        
        Args:
            audio_path: Path to audio file
            segment: Segment definition
            provider_name: ASR provider name
            decode_mode: Decode mode to use
            segment_idx: Segment index for logging
            
        Returns:
            SegmentCandidate or None if decode failed
        """
        provider = self.providers.get(provider_name)
        if not provider:
            self.logger.warning(f"Provider {provider_name} not available")
            return None
        
        try:
            start_time = time.time()
            
            # In a real implementation, you would extract the segment from the audio file
            # For now, we'll transcribe the full file and extract the relevant segments
            result = provider.transcribe(
                audio_path=audio_path,
                decode_mode=decode_mode,
                language="en"
            )
            
            processing_time = time.time() - start_time
            
            # Extract segments for the specified time range
            segment_start = segment.get('start', 0.0)
            segment_end = segment.get('end', result.duration)
            
            # Filter segments to time range
            filtered_segments = [
                seg for seg in result.segments
                if seg.start < segment_end and seg.end > segment_start
            ]
            
            # Adjust segment timestamps relative to segment start
            adjusted_segments = []
            for seg in filtered_segments:
                adjusted_seg = ASRSegment(
                    start=max(0, seg.start - segment_start),
                    end=min(segment_end - segment_start, seg.end - segment_start),
                    text=seg.text,
                    confidence=seg.confidence,
                    words=seg.words,
                    speaker_id=seg.speaker_id,
                    language=seg.language
                )
                adjusted_segments.append(adjusted_seg)
            
            # Create segment-specific result
            segment_text = ' '.join(seg.text for seg in adjusted_segments)
            segment_confidence = float(np.mean([seg.confidence for seg in adjusted_segments])) if adjusted_segments else 0.0
            
            # Apply confidence calibration
            calibrated_confidence = provider.calibrate_confidence(
                segment_confidence,
                segment_end - segment_start
            )
            
            candidate = SegmentCandidate(
                provider=provider_name,
                decode_mode=decode_mode,
                model_name=provider.model_name,
                result=ASRResult(
                    segments=adjusted_segments,
                    full_text=segment_text,
                    language=result.language,
                    confidence=float(segment_confidence),
                    calibrated_confidence=float(calibrated_confidence),
                    processing_time=processing_time,
                    provider=provider_name,
                    decode_mode=decode_mode,
                    model_name=provider.model_name,
                    metadata=result.metadata
                ),
                calibrated_confidence=calibrated_confidence,
                processing_time=processing_time,
                metadata={
                    'segment_start': segment_start,
                    'segment_end': segment_end,
                    'original_duration': result.duration,
                    'filtered_segments': len(adjusted_segments)
                }
            )
            
            self.logger.info(f"Decode complete: {provider_name}-{decode_mode.value}",
                           context={
                               'confidence': calibrated_confidence,
                               'processing_time': processing_time,
                               'text_length': len(segment_text)
                           })
            
            return candidate
            
        except Exception as e:
            self.logger.error(f"Decode failed: {provider_name}-{decode_mode.value}: {e}")
            return None
    
    def _create_word_alignments(self, candidates: List[SegmentCandidate]) -> List[WordAlignment]:
        """Create word-level alignments across candidates"""
        if len(candidates) < 2:
            return []
        
        # Convert candidates to format expected by temporal aligner
        candidate_dicts = []
        for i, candidate in enumerate(candidates):
            candidate_dict = {
                'candidate_id': f"{candidate.provider}_{candidate.decode_mode.value}_{i}",
                'asr_data': {
                    'words': []
                }
            }
            
            # Extract words from segments
            for segment in candidate.result.segments:
                if segment.words:
                    for word in segment.words:
                        candidate_dict['asr_data']['words'].append({
                            'word': word.get('word', ''),
                            'start': word.get('start', 0.0),
                            'end': word.get('end', 0.0),
                            'confidence': word.get('confidence', candidate.calibrated_confidence)
                        })
                else:
                    # Fallback: create word entries from segment text
                    words_in_segment = segment.text.split()
                    if words_in_segment:
                        word_duration = (segment.end - segment.start) / len(words_in_segment)
                        for j, word in enumerate(words_in_segment):
                            word_start = segment.start + j * word_duration
                            word_end = word_start + word_duration
                            candidate_dict['asr_data']['words'].append({
                                'word': word,
                                'start': word_start,
                                'end': word_end,
                                'confidence': segment.confidence
                            })
            
            candidate_dicts.append(candidate_dict)
        
        # Create alignments using temporal aligner
        try:
            alignments = self.temporal_aligner.align_words_across_candidates(candidate_dicts)
            return alignments
        except Exception as e:
            self.logger.error(f"Word alignment failed: {e}")
            return []
    
    def _calculate_agreement_score(self, candidates: List[SegmentCandidate]) -> float:
        """Calculate inter-candidate agreement score"""
        if len(candidates) < 2:
            return 1.0
        
        # Create word alignments
        alignments = self._create_word_alignments(candidates)
        
        if not alignments:
            # Fallback: simple text similarity
            texts = [candidate.result.full_text for candidate in candidates]
            return self._calculate_text_similarity(texts)
        
        # Calculate agreement based on word alignments
        total_alignments = len(alignments)
        perfect_agreements = sum(1 for alignment in alignments 
                               if not alignment.is_confusion_set)
        
        if total_alignments == 0:
            return 0.0
        
        agreement_score = perfect_agreements / total_alignments
        
        # Weight by alignment quality
        if alignments:
            avg_quality = np.mean([alignment.alignment_quality for alignment in alignments])
            agreement_score = 0.7 * agreement_score + 0.3 * avg_quality
        
        return float(agreement_score)
    
    def _calculate_confidence_score(self, candidates: List[SegmentCandidate]) -> float:
        """Calculate overall confidence score from candidates"""
        if not candidates:
            return 0.0
        
        # Use calibrated confidence scores
        confidences = [candidate.calibrated_confidence for candidate in candidates]
        
        # Calculate weighted average (weight higher confidences more)
        weights = np.array(confidences)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        
        weighted_confidence = np.average(confidences, weights=weights)
        
        # Penalize high variance (disagreement between providers)
        confidence_variance = np.var(confidences)
        variance_penalty = 1.0 / (1.0 + confidence_variance)
        
        final_confidence = float(weighted_confidence * variance_penalty)
        
        return final_confidence
    
    def _calculate_text_similarity(self, texts: List[str]) -> float:
        """Calculate simple text similarity between multiple texts"""
        if len(texts) < 2:
            return 1.0
        
        # Normalize texts
        normalized_texts = [text.lower().strip() for text in texts]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(normalized_texts)):
            for j in range(i + 1, len(normalized_texts)):
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, normalized_texts[i], normalized_texts[j]).ratio()
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _select_best_candidate(self, 
                              candidates: List[SegmentCandidate],
                              word_alignments: List[WordAlignment]) -> Optional[SegmentCandidate]:
        """
        Select best candidate using confidence scoring and temporal coherence
        
        Args:
            candidates: List of segment candidates
            word_alignments: Word alignments for temporal coherence analysis
            
        Returns:
            Best candidate or None if no valid candidates
        """
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Score each candidate
        candidate_scores = []
        
        for candidate in candidates:
            score = self._score_candidate(candidate, candidates, word_alignments)
            candidate_scores.append((candidate, score))
        
        # Sort by score (descending)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_candidate = candidate_scores[0][0]
        best_score = candidate_scores[0][1]
        
        self.logger.info(f"Best candidate selected: {best_candidate.provider}-{best_candidate.decode_mode.value}",
                        context={'score': best_score, 'confidence': best_candidate.calibrated_confidence})
        
        return best_candidate
    
    def _score_candidate(self, 
                        candidate: SegmentCandidate,
                        all_candidates: List[SegmentCandidate],
                        word_alignments: List[WordAlignment]) -> float:
        """Score a candidate for best selection"""
        
        # Base score from calibrated confidence
        confidence_score = candidate.calibrated_confidence
        
        # Temporal coherence score (based on word alignment quality)
        temporal_score = 0.8  # Default good score
        if word_alignments:
            # Find alignments where this candidate participates
            candidate_alignments = []
            for alignment in word_alignments:
                for variant in alignment.word_variants:
                    if variant.get('candidate_id', '').startswith(candidate.provider):
                        candidate_alignments.append(alignment)
                        break
            
            if candidate_alignments:
                temporal_score = np.mean([alignment.alignment_quality for alignment in candidate_alignments])
        
        # Processing efficiency score (favor faster processing)
        max_processing_time = max(c.processing_time for c in all_candidates)
        if max_processing_time > 0:
            efficiency_score = 1.0 - (candidate.processing_time / max_processing_time)
        else:
            efficiency_score = 1.0
        
        # Text length consistency (penalize outliers)
        text_lengths = [len(c.result.full_text) for c in all_candidates]
        avg_length = np.mean(text_lengths)
        length_deviation = abs(len(candidate.result.full_text) - avg_length) / (avg_length + 1)
        length_score = 1.0 / (1.0 + length_deviation)
        
        # Combined weighted score
        final_score = (
            0.5 * confidence_score +      # Primary: confidence
            0.3 * temporal_score +        # Important: temporal coherence  
            0.1 * efficiency_score +      # Minor: processing speed
            0.1 * length_score           # Minor: length consistency
        )
        
        return float(final_score)
    
    def _calculate_overall_confidence(self, segment_analyses: List[SegmentAnalysis]) -> float:
        """Calculate overall confidence across all segments"""
        if not segment_analyses:
            return 0.0
        
        # Weight by segment duration
        total_duration = sum(analysis.segment_duration for analysis in segment_analyses)
        if total_duration == 0:
            return 0.0
        
        weighted_confidence = 0.0
        for analysis in segment_analyses:
            weight = analysis.segment_duration / total_duration
            segment_confidence = analysis.confidence_score
            weighted_confidence += weight * segment_confidence
        
        return float(weighted_confidence)