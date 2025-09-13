import numpy as np
import json
from typing import List, Dict, Any, Optional, Set, Tuple, Union, cast
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from difflib import SequenceMatcher
import string
from scipy.sparse import spmatrix
import numpy.typing as npt

class ConfidenceScorer:
    """Calculates multi-dimensional confidence scores for transcript candidates"""
    
    def __init__(self, scoring_weights: Optional[Dict[str, float]] = None) -> None:
        # Use custom weights if provided, otherwise use defaults
        default_weights = {
            'D': 0.28,  # Diarization consistency
            'A': 0.32,  # ASR alignment and confidence  
            'L': 0.18,  # Linguistic quality
            'R': 0.12,  # Cross-run agreement
            'O': 0.10   # Overlap handling
        }
        
        if scoring_weights is not None:
            # Validate that all required keys are present
            required_keys = set(default_weights.keys())
            provided_keys = set(scoring_weights.keys())
            
            if required_keys != provided_keys:
                raise ValueError(f"Scoring weights must contain exactly these keys: {required_keys}")
            
            # Validate that weights sum to approximately 1.0
            total_weight = sum(scoring_weights.values())
            if abs(total_weight - 1.0) > 0.001:
                raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight:.6f}")
            
            self.score_weights = scoring_weights.copy()
        else:
            self.score_weights = default_weights
        
        # Vectorizer cache for performance optimization
        self._vectorizer_cache: Dict[Any, TfidfVectorizer] = {}
        self._fitted_cache: Dict[Any, Tuple[TfidfVectorizer, Any]] = {}
    
    def _get_cached_vectorizer(self, vectorizer_type: str, **kwargs) -> TfidfVectorizer:
        """
        Get or create a cached TfidfVectorizer instance.
        
        Args:
            vectorizer_type: Type identifier for the vectorizer (e.g., 'fine', 'coarse', 'agreement')
            **kwargs: TfidfVectorizer parameters
            
        Returns:
            Cached or new TfidfVectorizer instance
        """
        # Create cache key from type and parameters
        cache_key = (vectorizer_type, tuple(sorted(kwargs.items())))
        
        if cache_key not in self._vectorizer_cache:
            self._vectorizer_cache[cache_key] = TfidfVectorizer(**kwargs)
        
        return self._vectorizer_cache[cache_key]
    
    def _get_fitted_vectorizer(self, vectorizer_type: str, texts: List[str], **kwargs) -> Tuple[TfidfVectorizer, Any]:
        """
        Get a fitted vectorizer and its transform matrix, using cache when possible.
        
        Args:
            vectorizer_type: Type identifier for the vectorizer
            texts: Text corpus to fit/transform
            **kwargs: TfidfVectorizer parameters
            
        Returns:
            Tuple of (fitted_vectorizer, tfidf_matrix)
        """
        # Create cache key including text hash for fitted versions
        text_hash = hash(tuple(texts))
        cache_key = (vectorizer_type, text_hash, tuple(sorted(kwargs.items())))
        
        if cache_key in self._fitted_cache:
            return self._fitted_cache[cache_key]
        
        # Get vectorizer and fit it
        vectorizer = self._get_cached_vectorizer(vectorizer_type, **kwargs)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Cache the result
        self._fitted_cache[cache_key] = (vectorizer, tfidf_matrix)
        return vectorizer, tfidf_matrix
    
    def _clear_vectorizer_cache(self):
        """Clear vectorizer cache to free memory between batches."""
        self._vectorizer_cache.clear()
        self._fitted_cache.clear()
    
    def score_all_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score all 15 candidates across 5 confidence dimensions.
        
        Args:
            candidates: List of candidate transcripts
            
        Returns:
            List of candidates with added confidence scores
        """
        # Calculate raw scores for each dimension
        d_scores = self._calculate_diarization_scores(candidates)
        a_scores = self._calculate_asr_scores(candidates)
        l_scores = self._calculate_linguistic_scores(candidates)
        r_scores = self._calculate_agreement_scores(candidates)
        o_scores = self._calculate_overlap_scores(candidates)
        
        # Normalize scores to 0.00-1.00 range using calibrated absolute normalization
        d_scores_norm = self._normalize_scores(d_scores, dimension='D')
        a_scores_norm = self._normalize_scores(a_scores, dimension='A')
        l_scores_norm = self._normalize_scores(l_scores, dimension='L')
        r_scores_norm = self._normalize_scores(r_scores, dimension='R')
        o_scores_norm = self._normalize_scores(o_scores, dimension='O')
        
        # Add scores to candidates and calculate final scores
        scored_candidates = []
        for i, candidate in enumerate(candidates):
            # Individual dimension scores
            d_score = d_scores_norm[i]
            a_score = a_scores_norm[i]
            l_score = l_scores_norm[i]
            r_score = r_scores_norm[i]
            o_score = o_scores_norm[i]
            
            # Calculate weighted final score
            final_score = (
                self.score_weights['D'] * d_score +
                self.score_weights['A'] * a_score +
                self.score_weights['L'] * l_score +
                self.score_weights['R'] * r_score +
                self.score_weights['O'] * o_score
            )
            
            # Add scores to candidate
            candidate_with_scores = candidate.copy()
            candidate_with_scores['confidence_scores'] = {
                'D_diarization': float(d_score),
                'A_asr_alignment': float(a_score), 
                'L_linguistic': float(l_score),
                'R_agreement': float(r_score),
                'O_overlap': float(o_score),
                'final_score': float(final_score)
            }
            
            scored_candidates.append(candidate_with_scores)
        
        # Clear vectorizer cache to free memory after batch processing
        self._clear_vectorizer_cache()
        
        return scored_candidates
    
    def _calculate_diarization_scores(self, candidates: List[Dict[str, Any]]) -> List[float]:
        """Calculate D1 - Diarization consistency scores"""
        scores = []
        
        for candidate in candidates:
            segments = candidate.get('aligned_segments', [])
            if not segments:
                scores.append(0.0)
                continue
            
            # D1a: Turn boundary word-snap score
            boundary_score = self._calculate_boundary_snap_score(segments)
            
            # D1b: Speaker switch smoothness  
            smoothness_score = self._calculate_speaker_smoothness(segments)
            
            # D1c: Overlap plausibility
            overlap_score = self._calculate_overlap_plausibility(candidate)
            
            # D1d: Global speaker stability
            stability_score = self._calculate_speaker_stability(segments)
            
            # Aggregate D score
            d_score = (
                0.30 * boundary_score +
                0.25 * smoothness_score + 
                0.25 * overlap_score +
                0.20 * stability_score
            )
            
            scores.append(d_score)
        
        return scores
    
    def _calculate_boundary_snap_score(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate fraction of boundaries within ±120ms of word edges"""
        if len(segments) < 2:
            return 1.0
        
        snapped_boundaries = 0
        total_boundaries = 0
        
        for i in range(len(segments) - 1):
            current_end = segments[i]['end']
            next_start = segments[i + 1]['start']
            
            # Check if boundary aligns with word timestamps
            current_words = segments[i].get('words', [])
            next_words = segments[i + 1].get('words', [])
            
            if current_words and next_words:
                last_word_end = current_words[-1]['end']
                first_word_start = next_words[0]['start']
                
                # Check if diarization boundary is within 120ms of word boundary
                if abs(current_end - last_word_end) <= 0.12 and abs(next_start - first_word_start) <= 0.12:
                    snapped_boundaries += 1
                
                total_boundaries += 1
        
        return snapped_boundaries / max(total_boundaries, 1)
    
    def _calculate_speaker_smoothness(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate penalty for rapid A↔B↔A speaker flips"""
        if len(segments) < 3:
            return 1.0
        
        flip_penalty = 0.0
        total_transitions = 0
        
        # Look for A->B->A patterns within 3 second windows
        for i in range(len(segments) - 2):
            seg1 = segments[i]
            seg2 = segments[i + 1] 
            seg3 = segments[i + 2]
            
            # Check if this forms an A->B->A pattern within 3 seconds
            if (seg1['speaker_id'] == seg3['speaker_id'] and 
                seg1['speaker_id'] != seg2['speaker_id'] and
                seg3['end'] - seg1['start'] <= 3.0):
                
                flip_penalty += 1.0
            
            total_transitions += 1
        
        # Return 1.0 minus normalized penalty
        return max(0.0, 1.0 - (flip_penalty / max(total_transitions, 1)))
    
    def _calculate_overlap_plausibility(self, candidate: Dict[str, Any]) -> float:
        """Calculate plausibility of detected overlaps"""
        # For simplicity, return moderate score
        # In practice, this would analyze energy and ASR concurrency
        diar_data = candidate.get('diarization_data', {})
        overlaps = diar_data.get('overlaps', [])
        
        if not overlaps:
            return 0.8  # No overlaps detected - moderate score
        
        # Basic heuristic based on overlap duration and frequency
        total_overlap_time = sum(o.get('duration', 0) for o in overlaps)
        total_speech_time = diar_data.get('total_speech_time', 1.0)
        
        overlap_ratio = total_overlap_time / total_speech_time
        
        # Reasonable overlap ratio is 5-15% for meetings
        if 0.05 <= overlap_ratio <= 0.15:
            return 0.9
        elif overlap_ratio < 0.05:
            return 0.7  # Maybe missing some overlaps
        else:
            return 0.6  # Probably too many false overlaps
    
    def _calculate_speaker_stability(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate consistency of speaker embeddings across session"""
        if not segments:
            return 0.0
        
        # Count speaker appearances and durations
        speaker_stats = {}
        for segment in segments:
            speaker_id = segment['speaker_id']
            duration = segment['end'] - segment['start']
            
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {'count': 0, 'total_duration': 0.0}
            
            speaker_stats[speaker_id]['count'] += 1
            speaker_stats[speaker_id]['total_duration'] += duration
        
        # Calculate distribution evenness (higher is better for meetings)
        total_duration = sum(stats['total_duration'] for stats in speaker_stats.values())
        
        if total_duration == 0:
            return 0.0
        
        # Calculate entropy of speaker time distribution
        entropy = 0.0
        for stats in speaker_stats.values():
            proportion = stats['total_duration'] / total_duration
            if proportion > 0:
                entropy -= proportion * math.log2(proportion)
        
        # Normalize entropy (max entropy for uniform distribution)
        max_entropy = math.log2(len(speaker_stats)) if len(speaker_stats) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _calculate_asr_scores(self, candidates: List[Dict[str, Any]]) -> List[float]:
        """Calculate A1 - Enhanced ASR alignment and confidence scores"""
        scores = []
        
        for candidate in candidates:
            asr_data = candidate.get('asr_data', {})
            segments = candidate.get('aligned_segments', [])
            
            # A1a: Word confidence mean (trimmed at 10% tails)
            word_confidence = self._calculate_trimmed_word_confidence(asr_data)
            
            # A1b: Enhanced alignment quality metrics
            alignment_quality = self._calculate_enhanced_alignment_quality(segments)
            
            # A1c: Timestamp coherence and monotonicity
            temporal_coherence = self._calculate_enhanced_temporal_coherence(asr_data, segments)
            
            # A1d: Boundary precision and speaker consistency
            boundary_precision = self._calculate_enhanced_boundary_precision(segments)
            
            # A1e: Coverage and gap handling quality
            coverage_quality = self._calculate_coverage_quality(segments)
            
            # Aggregate A score with enhanced weighting
            a_score = (
                0.25 * word_confidence +
                0.25 * alignment_quality +
                0.20 * temporal_coherence +
                0.20 * boundary_precision +
                0.10 * coverage_quality
            )
            
            scores.append(a_score)
        
        return scores
    
    def _calculate_trimmed_word_confidence(self, asr_data: Dict[str, Any]) -> float:
        """Calculate trimmed mean of word confidences"""
        words = asr_data.get('words', [])
        if not words:
            return 0.0
        
        confidences = [w.get('confidence', 0.0) for w in words]
        
        # Trim 10% from each tail
        sorted_conf = sorted(confidences)
        trim_count = int(0.1 * len(sorted_conf))
        
        if trim_count > 0:
            trimmed_conf = sorted_conf[trim_count:-trim_count]
        else:
            trimmed_conf = sorted_conf
        
        return float(np.mean(trimmed_conf)) if trimmed_conf else 0.0
    
    def _calculate_timestamp_monotonicity(self, asr_data: Dict[str, Any]) -> float:
        """Calculate fraction of words with monotonic timestamps"""
        words = asr_data.get('words', [])
        if len(words) < 2:
            return 1.0
        
        monotonic_pairs = 0
        total_pairs = 0
        
        for i in range(len(words) - 1):
            current_end = words[i]['end']
            next_start = words[i + 1]['start']
            
            if current_end <= next_start:
                monotonic_pairs += 1
            
            total_pairs += 1
        
        return monotonic_pairs / max(total_pairs, 1)
    
    def _calculate_enhanced_alignment_quality(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate enhanced alignment quality using new metrics"""
        if not segments:
            return 0.0
        
        # Extract alignment metrics from segments
        alignment_metrics = []
        for segment in segments:
            metrics = segment.get('alignment_metrics', {})
            if metrics:
                alignment_metrics.append(metrics)
        
        if not alignment_metrics:
            return 0.0
        
        # Average the key alignment quality metrics
        coverage_ratios = [m.get('coverage_ratio', 0.0) for m in alignment_metrics]
        assignment_confidences = [m.get('avg_assignment_confidence', 0.0) for m in alignment_metrics]
        overlap_scores = [m.get('overlap_handling_score', 0.0) for m in alignment_metrics]
        
        avg_coverage = float(np.mean(coverage_ratios)) if coverage_ratios else 0.0
        avg_assignment = float(np.mean(assignment_confidences)) if assignment_confidences else 0.0
        avg_overlap = float(np.mean(overlap_scores)) if overlap_scores else 0.0
        
        # Weighted combination
        return 0.4 * avg_coverage + 0.35 * avg_assignment + 0.25 * avg_overlap
    
    def _calculate_enhanced_temporal_coherence(self, asr_data: Dict[str, Any], 
                                             segments: List[Dict[str, Any]]) -> float:
        """Calculate enhanced temporal coherence across ASR and alignment"""
        # Original timestamp monotonicity
        asr_monotonicity = self._calculate_timestamp_monotonicity(asr_data)
        
        # Segment-level temporal coherence
        if not segments:
            return asr_monotonicity
        
        temporal_scores = []
        for segment in segments:
            # Get temporal coherence from segment metadata
            segment_coherence = segment.get('temporal_coherence', 0.0)
            temporal_scores.append(segment_coherence)
        
        segment_coherence = float(np.mean(temporal_scores)) if temporal_scores else 0.0
        
        # Cross-segment temporal consistency
        cross_segment_consistency = self._calculate_cross_segment_consistency(segments)
        
        return 0.4 * asr_monotonicity + 0.35 * segment_coherence + 0.25 * cross_segment_consistency
    
    def _calculate_cross_segment_consistency(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate temporal consistency across segment boundaries"""
        if len(segments) < 2:
            return 1.0
        
        consistent_transitions = 0
        total_transitions = 0
        
        for i in range(len(segments) - 1):
            current_seg = segments[i]
            next_seg = segments[i + 1]
            
            # Check for reasonable time gap between segments
            time_gap = next_seg['start'] - current_seg['end']
            
            # Good transition: gap between -0.1s and 3.0s
            if -0.1 <= time_gap <= 3.0:
                consistent_transitions += 1
            
            total_transitions += 1
        
        return consistent_transitions / max(total_transitions, 1)
    
    def _calculate_enhanced_boundary_precision(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate enhanced boundary precision using alignment scores"""
        if not segments:
            return 0.0
        
        boundary_scores = []
        for segment in segments:
            # Use the boundary alignment score from the enhanced alignment
            boundary_score = segment.get('boundary_alignment_score', 0.0)
            boundary_scores.append(boundary_score)
        
        avg_boundary_score = float(np.mean(boundary_scores)) if boundary_scores else 0.0
        
        # Also consider speaker conflict penalties
        total_words = sum(segment.get('word_count', 0) for segment in segments)
        total_conflicts = sum(segment.get('speaker_conflicts', 0) for segment in segments)
        
        conflict_penalty = total_conflicts / max(total_words, 1)
        conflict_score = max(0.0, 1.0 - conflict_penalty * 2.0)  # Penalty up to 50%
        
        return 0.7 * avg_boundary_score + 0.3 * conflict_score
    
    def _calculate_coverage_quality(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate quality of word coverage and gap handling"""
        if not segments:
            return 0.0
        
        total_words = sum(segment.get('word_count', 0) for segment in segments)
        gap_assignments = sum(segment.get('gap_assignments', 0) for segment in segments)
        
        if total_words == 0:
            return 0.0
        
        # Coverage ratio (fewer gap assignments is better)
        coverage_ratio = 1.0 - (gap_assignments / total_words)
        
        # Segment distribution quality (avoid too many tiny segments)
        segment_sizes = [segment.get('word_count', 0) for segment in segments]
        avg_segment_size = np.mean(segment_sizes) if segment_sizes else 0.0
        min_segment_size = min(segment_sizes) if segment_sizes else 0.0
        
        # Penalty for very small segments (< 3 words)
        size_penalty = 0.0
        if avg_segment_size > 0:
            small_segments = sum(1 for size in segment_sizes if size < 3)
            size_penalty = small_segments / len(segment_sizes)
        
        size_quality = max(0.0, 1.0 - size_penalty)
        
        return 0.7 * coverage_ratio + 0.3 * size_quality
    
    def _calculate_boundary_fit(self, segments: List[Dict[str, Any]]) -> float:
        """Legacy boundary fit calculation for backwards compatibility"""
        if not segments:
            return 0.0
        
        total_words = 0
        words_in_bounds = 0
        
        for segment in segments:
            words = segment.get('words', [])
            segment_start = segment['start']
            segment_end = segment['end']
            
            for word in words:
                total_words += 1
                if segment_start <= word['start'] and word['end'] <= segment_end:
                    words_in_bounds += 1
        
        return words_in_bounds / max(total_words, 1)
    
    def _calculate_word_stability(self, asr_data: Dict[str, Any]) -> float:
        """Calculate consistency of proper noun recognition"""
        words = asr_data.get('words', [])
        if not words:
            return 0.0
        
        # Find potential proper nouns (capitalized words that aren't sentence starts)
        proper_nouns = []
        
        for i, word in enumerate(words):
            word_text = word.get('word', '').strip()
            
            # Simple heuristic for proper nouns
            if (word_text and word_text[0].isupper() and 
                len(word_text) > 2 and
                i > 0):  # Not sentence start
                proper_nouns.append(word_text.lower())
        
        if not proper_nouns:
            return 0.8  # No proper nouns found - neutral score
        
        # Count occurrences
        noun_counts = Counter(proper_nouns)
        
        # Calculate consistency (repeated nouns get higher score)
        total_occurrences = sum(noun_counts.values())
        repeated_occurrences = sum(count for count in noun_counts.values() if count > 1)
        
        return repeated_occurrences / max(total_occurrences, 1)
    
    def _calculate_linguistic_scores(self, candidates: List[Dict[str, Any]]) -> List[float]:
        """Calculate L1 - Linguistic quality scores"""
        scores = []
        
        for candidate in candidates:
            asr_data = candidate.get('asr_data', {})
            text = asr_data.get('text', '')
            
            # L1a: Language model plausibility (inverse perplexity)
            lm_score = self._calculate_language_model_score(text)
            
            # L1b: Punctuation and casing plausibility
            punct_score = self._calculate_punctuation_score(text)
            
            # L1c: Disfluency handling
            disfluency_score = self._calculate_disfluency_score(text)
            
            # Aggregate L score
            l_score = (
                0.40 * lm_score +
                0.35 * punct_score +
                0.25 * disfluency_score
            )
            
            scores.append(l_score)
        
        return scores
    
    def _calculate_language_model_score(self, text: str) -> float:
        """Calculate enhanced language model plausibility score with sophisticated heuristics"""
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) < 5:
            return 0.5
        
        # 1. Vocabulary sophistication analysis (20%)
        vocab_score = self._analyze_vocabulary_sophistication(words)
        
        # 2. Grammar pattern detection (25%)
        grammar_score = self._analyze_grammar_patterns(text, words)
        
        # 3. Unnatural word combination detection (20%)
        combination_score = self._detect_unnatural_combinations(words)
        
        # 4. Coherence checking (15%)
        coherence_score = self._check_coherence(text)
        
        # 5. Proper noun detection (10%)
        proper_noun_score = self._validate_proper_nouns(words)
        
        # 6. Function word ratio (10%)
        function_word_score = self._analyze_function_words(words)
        
        # Weighted combination
        final_score = (
            0.20 * vocab_score +
            0.25 * grammar_score +
            0.20 * combination_score +
            0.15 * coherence_score +
            0.10 * proper_noun_score +
            0.10 * function_word_score
        )
        
        return min(1.0, max(0.0, final_score))
    
    def _analyze_vocabulary_sophistication(self, words: List[str]) -> float:
        """Analyze vocabulary sophistication and variety"""
        if len(words) < 5:
            return 0.5
        
        # Clean words for analysis
        clean_words = [re.sub(r'[^\w]', '', word.lower()) for word in words if word.strip()]
        clean_words = [w for w in clean_words if len(w) > 0]
        
        if not clean_words:
            return 0.0
        
        # Type-token ratio (vocabulary diversity)
        unique_words = set(clean_words)
        ttr = len(unique_words) / len(clean_words)
        ttr_score = min(1.0, ttr * 2.0)  # Normalize, good TTR is ~0.5
        
        # Average word length sophistication
        avg_length = np.mean([len(w) for w in clean_words])
        length_score = 1.0 if 4.0 <= avg_length <= 7.0 else max(0.3, 1.0 - abs(float(avg_length) - 5.5) * 0.2)
        
        # Avoid excessive simple words
        simple_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        simple_ratio = sum(1 for w in clean_words if w in simple_words) / len(clean_words)
        simple_score = 1.0 if 0.15 <= simple_ratio <= 0.35 else max(0.4, 1.0 - abs(simple_ratio - 0.25) * 3)
        
        # Repetition penalty
        word_counts = Counter(clean_words)
        total_repetitions = sum(count - 1 for count in word_counts.values() if count > 1)
        repetition_penalty = total_repetitions / len(clean_words)
        repetition_score = max(0.0, 1.0 - repetition_penalty * 2)
        
        return (ttr_score + length_score + simple_score + repetition_score) / 4
    
    def _analyze_grammar_patterns(self, text: str, words: List[str]) -> float:
        """Analyze grammar patterns and structure"""
        if len(words) < 3:
            return 0.5
        
        # 1. Article usage patterns
        articles = {'a', 'an', 'the'}
        article_pattern_score = self._check_article_patterns(words, articles)
        
        # 2. Sentence structure variety
        sentences = re.split(r'[.!?]+', text)
        structure_score = self._analyze_sentence_structures(sentences)
        
        # 3. Word order patterns (very basic)
        word_order_score = self._check_word_order_patterns(words)
        
        # 4. Conjunction usage
        conjunctions = {'and', 'but', 'or', 'so', 'because', 'although', 'while', 'if', 'when', 'since'}
        conjunction_score = self._analyze_conjunction_usage(words, conjunctions)
        
        return (article_pattern_score + structure_score + word_order_score + conjunction_score) / 4
    
    def _check_article_patterns(self, words: List[str], articles: set) -> float:
        """Check for natural article usage patterns"""
        if len(words) < 10:
            return 0.7
        
        clean_words = [w.lower().strip('.,!?;:') for w in words]
        article_ratio = sum(1 for w in clean_words if w in articles) / len(clean_words)
        
        # Natural article ratio is 8-15% in English
        if 0.08 <= article_ratio <= 0.15:
            return 1.0
        elif 0.05 <= article_ratio <= 0.20:
            return 0.8
        else:
            return 0.5
    
    def _analyze_sentence_structures(self, sentences: List[str]) -> float:
        """Analyze variety in sentence structures"""
        valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]
        
        if len(valid_sentences) < 2:
            return 0.6
        
        # Check sentence length variety
        lengths = [len(s.split()) for s in valid_sentences]
        avg_length = np.mean(lengths)
        length_variety = np.std(lengths) if len(lengths) > 1 else 0
        
        # Good average sentence length for speech: 8-20 words
        length_score = 1.0 if 8 <= avg_length <= 20 else max(0.4, 1.0 - abs(float(avg_length) - 14) * 0.1)
        
        # Variety bonus
        variety_score = min(1.0, float(length_variety) / 5.0)  # Normalize standard deviation
        
        return (length_score + variety_score) / 2
    
    def _check_word_order_patterns(self, words: List[str]) -> float:
        """Check for basic English word order patterns"""
        if len(words) < 5:
            return 0.7
        
        # Very basic heuristics for English word order
        clean_words = [w.lower().strip('.,!?;:') for w in words]
        
        # Penalize sequences that violate basic English patterns
        violations = 0
        total_checks = 0
        
        for i in range(len(clean_words) - 1):
            word1, word2 = clean_words[i], clean_words[i + 1]
            
            # Basic patterns that would be unusual
            if word1 in {'a', 'an', 'the'} and word2 in {'a', 'an', 'the'}:
                violations += 1  # Double articles
            elif word1 in {'very', 'really', 'quite', 'extremely'} and word2 in {'and', 'or', 'but'}:
                violations += 1  # Adverb before conjunction
            
            total_checks += 1
        
        if total_checks == 0:
            return 0.7
        
        violation_rate = violations / total_checks
        return max(0.3, 1.0 - violation_rate * 3)
    
    def _analyze_conjunction_usage(self, words: List[str], conjunctions: set) -> float:
        """Analyze natural conjunction usage"""
        if len(words) < 10:
            return 0.7
        
        clean_words = [w.lower().strip('.,!?;:') for w in words]
        conjunction_ratio = sum(1 for w in clean_words if w in conjunctions) / len(clean_words)
        
        # Natural conjunction ratio is 3-8% in speech
        if 0.03 <= conjunction_ratio <= 0.08:
            return 1.0
        elif 0.01 <= conjunction_ratio <= 0.12:
            return 0.8
        else:
            return 0.5
    
    def _detect_unnatural_combinations(self, words: List[str]) -> float:
        """Detect unlikely consecutive word combinations"""
        if len(words) < 3:
            return 0.8
        
        clean_words = [w.lower().strip('.,!?;:') for w in words if w.strip()]
        
        # Define unlikely patterns
        unnatural_patterns = 0
        total_bigrams = 0
        
        for i in range(len(clean_words) - 1):
            word1, word2 = clean_words[i], clean_words[i + 1]
            
            # Check for unlikely combinations
            if self._is_unnatural_bigram(word1, word2):
                unnatural_patterns += 1
            
            total_bigrams += 1
        
        if total_bigrams == 0:
            return 0.8
        
        unnatural_ratio = unnatural_patterns / total_bigrams
        return max(0.0, 1.0 - unnatural_ratio * 5)  # Heavy penalty for unnatural combinations
    
    def _is_unnatural_bigram(self, word1: str, word2: str) -> bool:
        """Check if a word pair is unnatural"""
        # Articles followed by articles
        if word1 in {'a', 'an', 'the'} and word2 in {'a', 'an', 'the'}:
            return True
        
        # Prepositions followed by articles in unlikely ways
        if word1 in {'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'} and word2 in {'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}:
            return True
        
        # Multiple question words
        if word1 in {'what', 'where', 'when', 'why', 'how', 'who'} and word2 in {'what', 'where', 'when', 'why', 'how', 'who'}:
            return True
        
        # Very unlikely patterns specific to ASR errors
        unlikely_pairs = {
            ('i', 'are'), ('you', 'am'), ('they', 'is'), ('we', 'is'),
            ('and', 'and'), ('but', 'but'), ('or', 'or')
        }
        
        return (word1, word2) in unlikely_pairs
    
    def _check_coherence(self, text: str) -> float:
        """Check for basic coherence and logical flow"""
        if not text or len(text.split()) < 10:
            return 0.6
        
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]
        
        if len(valid_sentences) < 2:
            return 0.6
        
        # 1. Topic consistency (simple keyword overlap)
        topic_score = self._calculate_topic_consistency(valid_sentences)
        
        # 2. Pronoun reference coherence
        pronoun_score = self._check_pronoun_coherence(valid_sentences)
        
        # 3. Temporal coherence (tense consistency)
        temporal_score = self._check_temporal_coherence(text)
        
        return (topic_score + pronoun_score + temporal_score) / 3
    
    def _calculate_topic_consistency(self, sentences: List[str]) -> float:
        """Calculate topic consistency across sentences"""
        if len(sentences) < 2:
            return 0.7
        
        # Extract content words from each sentence
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
        
        sentence_words = []
        for sentence in sentences:
            words = [w.lower().strip('.,!?;:') for w in sentence.split()]
            content_words = [w for w in words if w not in stop_words and len(w) > 2]
            sentence_words.append(set(content_words))
        
        # Calculate overlap between adjacent sentences
        overlaps = []
        for i in range(len(sentence_words) - 1):
            words1 = sentence_words[i]
            words2 = sentence_words[i + 1]
            
            if not words1 or not words2:
                overlaps.append(0.0)
                continue
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            overlap = intersection / union if union > 0 else 0.0
            overlaps.append(overlap)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0.0
        return min(1.0, float(avg_overlap) * 4)  # Scale up since perfect overlap is rare
    
    def _check_pronoun_coherence(self, sentences: List[str]) -> float:
        """Check for basic pronoun reference coherence"""
        pronouns = {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their', 'theirs'}
        
        pronoun_sentences = []
        for sentence in sentences:
            words = [w.lower().strip('.,!?;:') for w in sentence.split()]
            if any(w in pronouns for w in words):
                pronoun_sentences.append(sentence)
        
        if len(pronoun_sentences) == 0:
            return 0.8  # No pronouns to evaluate
        
        # Simple heuristic: pronouns should have potential antecedents
        # This is a very basic check
        return 0.7  # Placeholder for more sophisticated analysis
    
    def _check_temporal_coherence(self, text: str) -> float:
        """Check for temporal coherence (basic tense consistency)"""
        # Count different tense markers
        past_markers = re.findall(r'\b(was|were|had|did|went|came|said|told|asked)\b', text.lower())
        present_markers = re.findall(r'\b(is|are|do|does|go|come|say|tell|ask)\b', text.lower())
        future_markers = re.findall(r'\b(will|would|going to|gonna)\b', text.lower())
        
        total_markers = len(past_markers) + len(present_markers) + len(future_markers)
        
        if total_markers == 0:
            return 0.7
        
        # Calculate tense distribution
        past_ratio = len(past_markers) / total_markers
        present_ratio = len(present_markers) / total_markers
        future_ratio = len(future_markers) / total_markers
        
        # Penalize if one tense dominates too much (indicates consistency)
        # or if tenses are too evenly mixed (indicates inconsistency)
        dominant_ratio = max(past_ratio, present_ratio, future_ratio)
        
        if dominant_ratio >= 0.7:
            return 0.9  # Good consistency
        elif dominant_ratio >= 0.5:
            return 0.8  # Reasonable consistency
        else:
            return 0.6  # Mixed tenses
    
    def _validate_proper_nouns(self, words: List[str]) -> float:
        """Validate that capitalized words are likely proper nouns"""
        if len(words) < 5:
            return 0.8
        
        # Find capitalized words that aren't sentence starts
        potential_proper_nouns = []
        
        # Simple sentence boundary detection
        sentence_starts = set()
        for i, word in enumerate(words):
            if i == 0:
                sentence_starts.add(i)
            elif i > 0 and words[i-1].rstrip('.,!?;:') != words[i-1]:
                sentence_starts.add(i)
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if (clean_word and len(clean_word) > 1 and 
                clean_word[0].isupper() and i not in sentence_starts):
                potential_proper_nouns.append(clean_word.lower())
        
        if not potential_proper_nouns:
            return 0.8  # No proper nouns to evaluate
        
        # Basic validation using common name patterns
        valid_count = 0
        for noun in potential_proper_nouns:
            if self._is_likely_proper_noun(noun):
                valid_count += 1
        
        validation_ratio = valid_count / len(potential_proper_nouns)
        return max(0.3, validation_ratio)
    
    def _is_likely_proper_noun(self, word: str) -> bool:
        """Check if a word is likely a valid proper noun"""
        # Common English name patterns and endings
        name_endings = {'son', 'sen', 'ton', 'ham', 'ford', 'burg', 'wick', 'worth', 'field', 'wood', 'land', 'stein', 'berg', 'man', 'er', 'ing', 'ly'}
        
        # Check for common name patterns
        if len(word) >= 3:
            if any(word.endswith(ending) for ending in name_endings):
                return True
            
            # Common place name patterns
            if word.endswith('ia') or word.endswith('ica') or word.endswith('land'):
                return True
        
        # Avoid obvious non-names
        function_words = {'and', 'the', 'but', 'for', 'not', 'you', 'are', 'can', 'will', 'was', 'has', 'had'}
        return word.lower() not in function_words
    
    def _analyze_function_words(self, words: List[str]) -> float:
        """Analyze the natural distribution of function words"""
        if len(words) < 10:
            return 0.7
        
        clean_words = [w.lower().strip('.,!?;:') for w in words]
        
        # Define function word categories
        articles = {'a', 'an', 'the'}
        prepositions = {'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among'}
        pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'}
        auxiliaries = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'might', 'may', 'can'}
        
        # Calculate ratios
        article_ratio = sum(1 for w in clean_words if w in articles) / len(clean_words)
        prep_ratio = sum(1 for w in clean_words if w in prepositions) / len(clean_words)
        pronoun_ratio = sum(1 for w in clean_words if w in pronouns) / len(clean_words)
        aux_ratio = sum(1 for w in clean_words if w in auxiliaries) / len(clean_words)
        
        # Score each category based on expected ranges for natural English
        article_score = 1.0 if 0.08 <= article_ratio <= 0.15 else max(0.4, 1.0 - abs(article_ratio - 0.12) * 5)
        prep_score = 1.0 if 0.10 <= prep_ratio <= 0.20 else max(0.4, 1.0 - abs(prep_ratio - 0.15) * 3)
        pronoun_score = 1.0 if 0.08 <= pronoun_ratio <= 0.18 else max(0.4, 1.0 - abs(pronoun_ratio - 0.13) * 3)
        aux_score = 1.0 if 0.08 <= aux_ratio <= 0.16 else max(0.4, 1.0 - abs(aux_ratio - 0.12) * 4)
        
        return (article_score + prep_score + pronoun_score + aux_score) / 4
    
    def _calculate_punctuation_score(self, text: str) -> float:
        """Calculate enhanced punctuation and casing plausibility with sophisticated validation"""
        if not text:
            return 0.0
        
        # 1. Enhanced sentence boundary detection (25%)
        boundary_score = self._analyze_sentence_boundaries(text)
        
        # 2. Missing punctuation detection (20%)
        missing_punct_score = self._detect_missing_punctuation(text)
        
        # 3. Punctuation placement validation (20%)
        placement_score = self._validate_punctuation_placement(text)
        
        # 4. Quote and parentheses matching (15%)
        matching_score = self._check_punctuation_matching(text)
        
        # 5. Enhanced capitalization context analysis (20%)
        capitalization_score = self._analyze_capitalization_context(text)
        
        # Weighted combination
        final_score = (
            0.25 * boundary_score +
            0.20 * missing_punct_score +
            0.20 * placement_score +
            0.15 * matching_score +
            0.20 * capitalization_score
        )
        
        return min(1.0, max(0.0, final_score))
    
    def _analyze_sentence_boundaries(self, text: str) -> float:
        """Analyze quality of sentence boundary detection and punctuation"""
        if not text.strip():
            return 0.0
        
        # 1. Check for proper sentence endings
        ending_patterns = re.findall(r'[.!?]+', text)
        if not ending_patterns:
            return 0.3  # No sentence endings found
        
        # 2. Analyze sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if s.strip()]
        
        if not valid_sentences:
            return 0.3
        
        # Check capitalization after punctuation
        proper_starts = 0
        for sentence in valid_sentences:
            if sentence and sentence[0].isupper():
                proper_starts += 1
        
        capitalization_ratio = proper_starts / len(valid_sentences)
        
        # Check for reasonable sentence lengths
        lengths = [len(s.split()) for s in valid_sentences]
        avg_length = np.mean(lengths) if lengths else 0
        
        # Good speech sentence length: 5-25 words
        length_score = 1.0 if 5 <= avg_length <= 25 else max(0.4, 1.0 - abs(float(avg_length) - 15) * 0.05)
        
        # Check for proper punctuation distribution
        punct_distribution_score = self._analyze_punctuation_distribution(text)
        
        return (0.5 * capitalization_ratio + 0.3 * length_score + 0.2 * punct_distribution_score)
    
    def _analyze_punctuation_distribution(self, text: str) -> float:
        """Analyze distribution of different punctuation marks"""
        # Count different punctuation types
        periods = text.count('.')
        questions = text.count('?')
        exclamations = text.count('!')
        commas = text.count(',')
        
        total_punct = periods + questions + exclamations + commas
        
        if total_punct == 0:
            return 0.5
        
        # Check for reasonable distribution
        period_ratio = periods / total_punct
        comma_ratio = commas / total_punct
        
        # For speech, periods should dominate (60-80%), commas moderate (15-30%)
        period_score = 1.0 if 0.6 <= period_ratio <= 0.8 else max(0.5, 1.0 - abs(period_ratio - 0.7) * 2)
        comma_score = 1.0 if 0.15 <= comma_ratio <= 0.3 else max(0.7, 1.0 - abs(comma_ratio - 0.22) * 3)
        
        return (period_score + comma_score) / 2
    
    def _detect_missing_punctuation(self, text: str) -> float:
        """Detect places where punctuation is likely missing"""
        if not text.strip():
            return 0.0
        
        words = text.split()
        if len(words) < 10:
            return 0.7
        
        # 1. Check for run-on sentences (very long sequences without punctuation)
        missing_penalty = 0.0
        
        # Find sequences of words without internal punctuation
        word_sequences = []
        current_sequence = []
        
        for word in words:
            if re.search(r'[.!?]', word):
                if current_sequence:
                    word_sequences.append(len(current_sequence))
                current_sequence = []
            else:
                current_sequence.append(word)
        
        if current_sequence:
            word_sequences.append(len(current_sequence))
        
        # Penalize very long sequences (>30 words likely missing punctuation)
        for seq_length in word_sequences:
            if seq_length > 30:
                missing_penalty += (seq_length - 30) * 0.02
        
        # 2. Check for missing commas in lists
        list_patterns = re.findall(r'\b\w+\s+\w+\s+and\s+\w+\b', text)
        potential_lists = len(list_patterns)
        actual_commas_in_lists = len(re.findall(r'\b\w+,\s+\w+\s+and\s+\w+\b', text))
        
        if potential_lists > 0:
            list_comma_ratio = actual_commas_in_lists / potential_lists
            list_score = list_comma_ratio
        else:
            list_score = 0.8
        
        # 3. Check for missing question marks
        question_words = ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose']
        potential_questions = 0
        actual_questions = text.count('?')
        
        for word in question_words:
            potential_questions += len(re.findall(rf'\b{word}\b', text.lower()))
        
        if potential_questions > 0:
            question_ratio = min(1.0, actual_questions / potential_questions)
        else:
            question_ratio = 0.8
        
        # Combine scores
        missing_score = max(0.0, 1.0 - missing_penalty)
        return (0.5 * missing_score + 0.3 * list_score + 0.2 * question_ratio)
    
    def _validate_punctuation_placement(self, text: str) -> float:
        """Validate that punctuation is placed correctly"""
        if not text.strip():
            return 0.0
        
        # 1. Check for punctuation placement errors
        placement_errors = 0
        total_checks = 0
        
        # Common placement errors
        error_patterns = [
            r'\s+[.!?]',  # Space before ending punctuation
            r'[.!?]\w',   # No space after ending punctuation
            r'\s+,',      # Space before comma
            r',[^\s]',    # No space after comma
            r'[.!?]{2,}', # Multiple consecutive punctuation
            r'\s+[;:]',   # Space before semicolon/colon
        ]
        
        for pattern in error_patterns:
            errors = len(re.findall(pattern, text))
            placement_errors += errors
            total_checks += 1
        
        # 2. Check for proper abbreviation handling
        abbrev_score = self._check_abbreviation_punctuation(text)
        
        # 3. Check for proper contractions
        contraction_score = self._check_contraction_punctuation(text)
        
        # Calculate placement score
        if total_checks > 0:
            words = len(text.split())
            error_rate = placement_errors / max(words, 1)
            placement_error_score = max(0.0, 1.0 - error_rate * 10)
        else:
            placement_error_score = 0.8
        
        return (0.5 * placement_error_score + 0.25 * abbrev_score + 0.25 * contraction_score)
    
    def _check_abbreviation_punctuation(self, text: str) -> float:
        """Check for proper abbreviation punctuation"""
        # Common abbreviations that should have periods
        abbrevs = ['dr', 'mr', 'mrs', 'ms', 'prof', 'inc', 'corp', 'ltd', 'etc', 'vs', 'ie', 'eg']
        
        abbrev_count = 0
        proper_abbrev_count = 0
        
        for abbrev in abbrevs:
            # Find potential abbreviations
            pattern = rf'\b{abbrev}\b'
            matches = re.findall(pattern, text.lower())
            abbrev_count += len(matches)
            
            # Check if they have proper punctuation
            pattern_with_period = rf'\b{abbrev}\.\b'
            proper_matches = re.findall(pattern_with_period, text.lower())
            proper_abbrev_count += len(proper_matches)
        
        if abbrev_count == 0:
            return 0.8
        
        return proper_abbrev_count / abbrev_count
    
    def _check_contraction_punctuation(self, text: str) -> float:
        """Check for proper contraction punctuation"""
        # Common contraction patterns
        contractions = ["n't", "'ll", "'re", "'ve", "'m", "'s", "'d"]
        
        contraction_count = 0
        proper_contractions = 0
        
        for contraction in contractions:
            # Count contractions with apostrophes
            proper_matches = re.findall(rf"\w+{re.escape(contraction)}\b", text)
            proper_contractions += len(proper_matches)
            
            # Count potential contractions without apostrophes
            no_apostrophe = contraction.replace("'", "")
            if no_apostrophe:
                potential_matches = re.findall(rf"\w+{re.escape(no_apostrophe)}\b", text)
                contraction_count += len(potential_matches)
        
        contraction_count += proper_contractions
        
        if contraction_count == 0:
            return 0.8
        
        return proper_contractions / contraction_count
    
    def _check_punctuation_matching(self, text: str) -> float:
        """Check for proper matching of quotes and parentheses"""
        if not text.strip():
            return 0.0
        
        # 1. Check parentheses matching
        open_parens = text.count('(')
        close_parens = text.count(')')
        paren_score = 1.0 if open_parens == close_parens else max(0.0, 1.0 - abs(open_parens - close_parens) * 0.2)
        
        # 2. Check square brackets matching
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        bracket_score = 1.0 if open_brackets == close_brackets else max(0.0, 1.0 - abs(open_brackets - close_brackets) * 0.2)
        
        # 3. Check quote matching (simple approach)
        double_quotes = text.count('"')
        single_quotes = text.count("'")
        
        # For speech transcripts, quotes should typically be paired
        double_quote_score = 1.0 if double_quotes % 2 == 0 else 0.7
        
        # Single quotes are trickier due to contractions
        # Simple heuristic: if there are many single quotes, they should be mostly in contractions
        if single_quotes > 0:
            # Count contractions
            contraction_quotes = len(re.findall(r"\w+'\w+", text))
            non_contraction_quotes = single_quotes - contraction_quotes
            single_quote_score = 1.0 if non_contraction_quotes % 2 == 0 else 0.8
        else:
            single_quote_score = 1.0
        
        # 4. Check for proper quote placement
        quote_placement_score = self._check_quote_placement(text)
        
        return (0.3 * paren_score + 0.2 * bracket_score + 0.2 * double_quote_score + 
                0.1 * single_quote_score + 0.2 * quote_placement_score)
    
    def _check_quote_placement(self, text: str) -> float:
        """Check for proper quote placement relative to punctuation"""
        # Look for quotes that should have punctuation inside/outside
        
        # Incorrect: ". or ,", should be "." or ,"
        incorrect_placement = len(re.findall(r'"[.!?,]', text))
        
        # Total quote sections
        quote_sections = len(re.findall(r'"[^"]*"', text))
        
        if quote_sections == 0:
            return 0.9  # No quotes to evaluate
        
        error_rate = incorrect_placement / max(quote_sections, 1)
        return max(0.3, 1.0 - error_rate)
    
    def _analyze_capitalization_context(self, text: str) -> float:
        """Analyze capitalization in context with sophisticated heuristics"""
        if not text.strip():
            return 0.0
        
        words = text.split()
        if len(words) < 5:
            return 0.6
        
        # 1. Sentence start capitalization
        sentence_start_score = self._check_sentence_start_capitalization(text)
        
        # 2. Proper noun capitalization
        proper_noun_score = self._analyze_proper_noun_capitalization(words)
        
        # 3. Title case validation
        title_case_score = self._check_title_case_patterns(words)
        
        # 4. Inappropriate capitalization detection
        inappropriate_caps_score = self._detect_inappropriate_capitalization(words)
        
        # 5. Acronym validation
        acronym_score = self._validate_acronym_capitalization(words)
        
        return (0.25 * sentence_start_score + 0.25 * proper_noun_score + 
                0.20 * title_case_score + 0.15 * inappropriate_caps_score + 
                0.15 * acronym_score)
    
    def _check_sentence_start_capitalization(self, text: str) -> float:
        """Check capitalization at sentence beginnings"""
        # Split into sentences and check first word of each
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if s.strip()]
        
        if not valid_sentences:
            return 0.5
        
        proper_starts = 0
        for sentence in valid_sentences:
            words = sentence.split()
            if words and words[0] and words[0][0].isupper():
                proper_starts += 1
        
        return proper_starts / len(valid_sentences)
    
    def _analyze_proper_noun_capitalization(self, words: List[str]) -> float:
        """Analyze whether capitalized words are appropriate proper nouns"""
        if len(words) < 5:
            return 0.7
        
        # Find words that should potentially be capitalized
        capitalized_words = []
        sentence_starts = self._find_sentence_starts(words)
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and len(clean_word) > 1 and clean_word[0].isupper() and i not in sentence_starts:
                capitalized_words.append(clean_word.lower())
        
        if not capitalized_words:
            return 0.8
        
        # Check if capitalized words are likely proper nouns
        valid_proper_nouns = 0
        for word in capitalized_words:
            if self._is_likely_proper_noun_context(word):
                valid_proper_nouns += 1
        
        return valid_proper_nouns / len(capitalized_words)
    
    def _find_sentence_starts(self, words: List[str]) -> set:
        """Find indices of words that start sentences"""
        sentence_starts = {0}  # First word always starts a sentence
        
        for i in range(len(words) - 1):
            word = words[i]
            if re.search(r'[.!?]', word):
                sentence_starts.add(i + 1)
        
        return sentence_starts
    
    def _is_likely_proper_noun_context(self, word: str) -> bool:
        """Enhanced check for likely proper noun context"""
        # Common proper noun patterns
        name_patterns = [
            r'.*son$', r'.*sen$', r'.*ton$', r'.*ham$', r'.*ford$',
            r'.*burg$', r'.*wick$', r'.*worth$', r'.*field$', r'.*wood$',
            r'.*land$', r'.*stein$', r'.*berg$', r'.*man$'
        ]
        
        # Check name patterns
        for pattern in name_patterns:
            if re.match(pattern, word):
                return True
        
        # Common place name endings
        place_endings = ['ia', 'ica', 'land', 'burg', 'ville', 'town', 'city', 'ford']
        if any(word.endswith(ending) for ending in place_endings):
            return True
        
        # Avoid common words that shouldn't be capitalized
        common_words = {'and', 'the', 'but', 'for', 'not', 'you', 'are', 'can', 'will', 'was', 'has', 'had', 'with', 'from', 'they', 'this', 'that', 'there', 'their'}
        return word not in common_words
    
    def _check_title_case_patterns(self, words: List[str]) -> float:
        """Check for appropriate title case usage"""
        # Look for sequences that might be titles
        title_sequences = []
        current_sequence = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper():
                current_sequence.append(word)
            else:
                if len(current_sequence) >= 2:
                    title_sequences.append(current_sequence)
                current_sequence = []
        
        if len(current_sequence) >= 2:
            title_sequences.append(current_sequence)
        
        if not title_sequences:
            return 0.8
        
        # Basic validation of title sequences
        valid_titles = 0
        for sequence in title_sequences:
            if self._is_likely_title_sequence(sequence):
                valid_titles += 1
        
        return valid_titles / len(title_sequences)
    
    def _is_likely_title_sequence(self, sequence: List[str]) -> bool:
        """Check if a sequence of capitalized words forms a likely title"""
        # Simple heuristics for titles
        if len(sequence) > 6:
            return False  # Too long for most titles
        
        # Check for title words vs function words
        function_words = {'The', 'A', 'An', 'And', 'Or', 'But', 'Of', 'In', 'On', 'At', 'To', 'For', 'With', 'By'}
        content_words = [w for w in sequence if w not in function_words]
        
        # Should have some content words
        return len(content_words) >= 1
    
    def _detect_inappropriate_capitalization(self, words: List[str]) -> float:
        """Detect words that are inappropriately capitalized"""
        if len(words) < 10:
            return 0.8
        
        inappropriate_count = 0
        total_caps = 0
        sentence_starts = self._find_sentence_starts(words)
        
        # Common words that should not be capitalized unless at sentence start
        should_not_cap = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may', 'might',
            'this', 'that', 'these', 'those', 'there', 'their', 'they', 'them',
            'you', 'your', 'we', 'our', 'us', 'i', 'me', 'my'
        }
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if word and word[0].isupper():
                total_caps += 1
                if clean_word in should_not_cap and i not in sentence_starts:
                    inappropriate_count += 1
        
        if total_caps == 0:
            return 0.8
        
        inappropriate_ratio = inappropriate_count / total_caps
        return max(0.0, 1.0 - inappropriate_ratio * 2)
    
    def _validate_acronym_capitalization(self, words: List[str]) -> float:
        """Validate that acronyms are properly capitalized"""
        # Find potential acronyms (all caps, 2-6 letters)
        potential_acronyms = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and 2 <= len(clean_word) <= 6 and clean_word.isupper():
                potential_acronyms.append(clean_word)
        
        if not potential_acronyms:
            return 0.8
        
        # Basic validation of acronyms
        valid_acronyms = 0
        common_acronyms = {'AI', 'API', 'CEO', 'CTO', 'USA', 'UK', 'EU', 'UN', 'NASA', 'FBI', 'CIA', 'IBM', 'IT', 'HR', 'PR', 'TV', 'CD', 'DVD', 'GPS', 'DNA', 'RNA'}
        
        for acronym in potential_acronyms:
            # Known acronyms or reasonable patterns
            if acronym in common_acronyms or self._is_reasonable_acronym_pattern(acronym):
                valid_acronyms += 1
        
        return valid_acronyms / len(potential_acronyms)
    
    def _is_reasonable_acronym_pattern(self, word: str) -> bool:
        """Check if a word follows reasonable acronym patterns"""
        # Very basic heuristics
        if len(word) == 2:
            return True  # Most 2-letter combos could be acronyms
        
        # Avoid patterns that are unlikely to be acronyms
        vowels = set('AEIOU')
        consonants = set('BCDFGHJKLMNPQRSTVWXYZ')
        
        # All vowels or all consonants are suspicious
        if all(c in vowels for c in word) or all(c in consonants for c in word):
            return False
        
        return True
    
    def _calculate_disfluency_score(self, text: str) -> float:
        """Calculate enhanced disfluency handling with sophisticated pattern detection"""
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) < 5:
            return 0.7
        
        # 1. Expanded filler detection (30%)
        filler_score = self._detect_expanded_fillers(text, words)
        
        # 2. Repetition pattern detection (25%)
        repetition_score = self._detect_repetition_patterns(words)
        
        # 3. Incomplete word detection (20%)
        incomplete_word_score = self._detect_incomplete_words(words)
        
        # 4. False start detection (15%)
        false_start_score = self._detect_false_starts(words)
        
        # 5. Pause marker handling (10%)
        pause_marker_score = self._handle_pause_markers(text, words)
        
        # Weighted combination
        final_score = (
            0.30 * filler_score +
            0.25 * repetition_score +
            0.20 * incomplete_word_score +
            0.15 * false_start_score +
            0.10 * pause_marker_score
        )
        
        return min(1.0, max(0.0, final_score))
    
    def _detect_expanded_fillers(self, text: str, words: List[str]) -> float:
        """Detect comprehensive list of filler words and phrases"""
        if not words:
            return 0.7
        
        # Comprehensive filler categories
        basic_fillers = {
            'um', 'uh', 'er', 'ah', 'eh', 'mm', 'hmm', 'hm', 'mhm',
            'umm', 'uhh', 'err', 'ahh', 'ohh', 'mmm'
        }
        
        discourse_markers = {
            'you know', 'i mean', 'like', 'so', 'well', 'right', 'okay', 
            'i guess', 'kind of', 'sort of', 'i think', 'actually', 
            'basically', 'literally', 'honestly', 'obviously'
        }
        
        false_starts = {
            'i was', 'we were', 'they are', 'it is', 'that was',
            'he said', 'she said', 'we had', 'i had', 'they had'
        }
        
        hesitation_sounds = {
            'erm', 'emm', 'mmhmm', 'uh-huh', 'mm-hmm', 'yeah', 'yes',
            'no', 'nah', 'yep', 'yup', 'sure', 'ok', 'alright'
        }
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        words_lower = [w.lower().strip('.,!?;:') for w in words]
        
        # Count different types of fillers
        basic_count = sum(1 for w in words_lower if w in basic_fillers)
        
        # Count discourse markers (phrases)
        discourse_count = 0
        for marker in discourse_markers:
            discourse_count += len(re.findall(rf'\b{re.escape(marker)}\b', text_lower))
        
        # Count false starts
        false_start_count = 0
        for start in false_starts:
            false_start_count += len(re.findall(rf'\b{re.escape(start)}\b', text_lower))
        
        # Count hesitation sounds
        hesitation_count = sum(1 for w in words_lower if w in hesitation_sounds)
        
        total_fillers = basic_count + discourse_count + false_start_count + hesitation_count
        filler_ratio = total_fillers / len(words)
        
        # Score based on filler ratio analysis
        return self._score_filler_ratio(filler_ratio, total_fillers, len(words))
    
    def _score_filler_ratio(self, filler_ratio: float, total_fillers: int, total_words: int) -> float:
        """Score filler ratio with nuanced analysis"""
        # Natural filler ratios for different speech types:
        # Formal speech: 1-3%
        # Casual conversation: 3-8%
        # Nervous/unprepared: 8-15%
        # Over-disfluent: >15%
        
        if filler_ratio == 0.0:
            return 0.6  # Suspiciously clean for natural speech
        elif 0.01 <= filler_ratio <= 0.08:
            return 0.9  # Natural range
        elif 0.08 < filler_ratio <= 0.12:
            return 0.8  # Slightly high but acceptable
        elif 0.12 < filler_ratio <= 0.18:
            return 0.6  # High disfluency
        else:
            return 0.3  # Excessive disfluency
    
    def _detect_repetition_patterns(self, words: List[str]) -> float:
        """Detect word and phrase repetition patterns that indicate disfluency"""
        if len(words) < 5:
            return 0.8
        
        clean_words = [w.lower().strip('.,!?;:') for w in words if w.strip()]
        
        # 1. Immediate word repetitions (most obvious disfluency)
        immediate_reps = self._count_immediate_repetitions(clean_words)
        
        # 2. Short phrase repetitions
        phrase_reps = self._count_phrase_repetitions(clean_words)
        
        # 3. Stutter patterns
        stutter_count = self._detect_stutter_patterns(words)
        
        # 4. Self-corrections
        correction_count = self._detect_self_corrections(clean_words)
        
        total_repetitions = immediate_reps + phrase_reps + stutter_count + correction_count
        repetition_ratio = total_repetitions / len(clean_words)
        
        # Score repetition patterns
        if repetition_ratio == 0.0:
            return 0.8  # No repetitions - could be natural or over-cleaned
        elif repetition_ratio <= 0.02:
            return 0.9  # Very few repetitions - good
        elif repetition_ratio <= 0.05:
            return 0.8  # Some repetitions - acceptable
        elif repetition_ratio <= 0.10:
            return 0.6  # Many repetitions - concerning
        else:
            return 0.3  # Excessive repetitions
    
    def _count_immediate_repetitions(self, words: List[str]) -> int:
        """Count immediate word repetitions (word word)"""
        repetitions = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:
                repetitions += 1
        return repetitions
    
    def _count_phrase_repetitions(self, words: List[str]) -> int:
        """Count short phrase repetitions"""
        repetitions = 0
        
        # Look for 2-3 word phrase repetitions
        for phrase_len in [2, 3]:
            for i in range(len(words) - 2 * phrase_len + 1):
                phrase1 = ' '.join(words[i:i + phrase_len])
                phrase2 = ' '.join(words[i + phrase_len:i + 2 * phrase_len])
                
                if phrase1 == phrase2:
                    repetitions += phrase_len
        
        return repetitions
    
    def _detect_stutter_patterns(self, words: List[str]) -> int:
        """Detect stutter patterns in individual words"""
        stutter_count = 0
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) >= 3:
                # Look for repeated initial sounds (stuttering)
                if self._is_stutter_pattern(clean_word):
                    stutter_count += 1
        
        return stutter_count
    
    def _is_stutter_pattern(self, word: str) -> bool:
        """Check if a word shows stutter patterns"""
        # Simple stutter detection
        if len(word) < 3:
            return False
        
        # Look for patterns like "b-b-but", "w-w-we"
        if '-' in word:
            parts = word.split('-')
            if len(parts) >= 2 and parts[0] == parts[1]:
                return True
        
        # Look for repeated initial syllables
        if len(word) >= 4:
            first_two = word[:2]
            if word.startswith(first_two + first_two):
                return True
        
        return False
    
    def _detect_self_corrections(self, words: List[str]) -> int:
        """Detect self-correction patterns"""
        corrections = 0
        
        # Look for correction patterns: "word, no, other_word" or "word, I mean, other_word"
        correction_markers = {'no', 'i mean', 'sorry', 'wait', 'actually', 'rather'}
        
        for i in range(len(words) - 2):
            if words[i + 1] in correction_markers:
                corrections += 1
        
        return corrections
    
    def _detect_incomplete_words(self, words: List[str]) -> float:
        """Detect incomplete or cut-off words"""
        if not words:
            return 0.8
        
        incomplete_count = 0
        
        for word in words:
            clean_word = re.sub(r'[^\w\-]', '', word)
            
            if self._is_incomplete_word(clean_word):
                incomplete_count += 1
        
        incomplete_ratio = incomplete_count / len(words)
        
        # Score incomplete words
        if incomplete_ratio == 0.0:
            return 0.9  # No incomplete words
        elif incomplete_ratio <= 0.02:
            return 0.8  # Very few incomplete words
        elif incomplete_ratio <= 0.05:
            return 0.6  # Some incomplete words
        else:
            return 0.3  # Many incomplete words
    
    def _is_incomplete_word(self, word: str) -> bool:
        """Check if a word appears to be incomplete"""
        if len(word) < 2:
            return False
        
        # Look for truncation markers
        if word.endswith('-') or word.endswith('--'):
            return True
        
        # Look for very short "words" that might be cut-off
        if len(word) == 1 and word.isalpha():
            return True
        
        # Look for unusual consonant clusters at the end
        if len(word) >= 3:
            last_three = word[-3:]
            consonants = 'bcdfghjklmnpqrstvwxyz'
            if all(c in consonants for c in last_three):
                return True
        
        # Look for words ending in uncommon letter combinations
        uncommon_endings = ['tch', 'dge', 'ght', 'sch', 'pth']
        if any(word.endswith(ending) for ending in uncommon_endings):
            return False  # These are actually common endings
        
        return False
    
    def _detect_false_starts(self, words: List[str]) -> float:
        """Detect false starts and sentence restarts"""
        if len(words) < 5:
            return 0.8
        
        false_starts = 0
        
        # Look for common false start patterns
        start_phrases = [
            'i was going to', 'we were trying to', 'i thought that',
            'what i wanted to', 'the thing is', 'i mean'
        ]
        
        text_lower = ' '.join(words).lower()
        
        for phrase in start_phrases:
            false_starts += len(re.findall(rf'\b{re.escape(phrase)}\b', text_lower))
        
        # Look for sentence restart patterns
        restart_patterns = self._detect_restart_patterns(words)
        false_starts += restart_patterns
        
        false_start_ratio = false_starts / len(words)
        
        # Score false starts
        if false_start_ratio == 0.0:
            return 0.9  # No false starts
        elif false_start_ratio <= 0.03:
            return 0.8  # Few false starts
        elif false_start_ratio <= 0.06:
            return 0.7  # Some false starts
        else:
            return 0.5  # Many false starts
    
    def _detect_restart_patterns(self, words: List[str]) -> int:
        """Detect sentence restart patterns"""
        restarts = 0
        
        # Look for patterns where speaker starts over
        restart_markers = {'let me', 'what i', 'so basically', 'okay so', 'well actually'}
        
        for i, word in enumerate(words[:-1]):
            bigram = f"{word.lower()} {words[i+1].lower()}"
            if bigram in restart_markers:
                restarts += 1
        
        return restarts
    
    def _handle_pause_markers(self, text: str, words: List[str]) -> float:
        """Handle explicit pause markers and silence indicators"""
        if not words:
            return 0.8
        
        # Look for explicit pause markers that might be in transcripts
        pause_markers = {
            '[pause]', '(pause)', '[silence]', '(silence)', 
            '[inaudible]', '(inaudible)', '[crosstalk]', '(crosstalk)',
            '...', '….', '..', '—', '–'
        }
        
        pause_count = 0
        
        # Count explicit markers
        text_lower = text.lower()
        for marker in pause_markers:
            pause_count += text_lower.count(marker.lower())
        
        # Count ellipsis patterns
        ellipsis_count = len(re.findall(r'\.{2,}', text))
        pause_count += ellipsis_count
        
        # Count em-dashes that might indicate pauses
        dash_count = text.count('—') + text.count('–')
        pause_count += dash_count
        
        pause_ratio = pause_count / len(words)
        
        # Score pause markers
        if pause_ratio == 0.0:
            return 0.9  # No explicit pause markers
        elif pause_ratio <= 0.02:
            return 0.8  # Few pause markers - natural
        elif pause_ratio <= 0.05:
            return 0.7  # Some pause markers
        else:
            return 0.5  # Many pause markers - fragmented speech
    
    def _calculate_agreement_scores(self, candidates: List[Dict[str, Any]]) -> List[float]:
        """Calculate R1 - Cross-run agreement scores"""
        scores = []
        
        # Extract texts from all candidates for comparison
        candidate_texts = []
        for candidate in candidates:
            asr_data = candidate.get('asr_data', {})
            text = asr_data.get('text', '')
            candidate_texts.append(text)
        
        for i, candidate in enumerate(candidates):
            current_text = candidate_texts[i]
            other_texts = candidate_texts[:i] + candidate_texts[i+1:]
            
            # R1a: N-gram consensus
            ngram_score = self._calculate_ngram_consensus(current_text, other_texts)
            
            # R1b: Named entity consensus
            ne_score = self._calculate_named_entity_consensus(current_text, other_texts)
            
            # R1c: Topic drift penalty
            topic_score = self._calculate_topic_coherence(candidate)
            
            # Aggregate R score
            r_score = (
                0.50 * ngram_score +
                0.30 * ne_score +
                0.20 * topic_score
            )
            
            scores.append(r_score)
        
        return scores
    
    def _calculate_ngram_consensus(self, current_text: str, other_texts: List[str]) -> float:
        """Calculate enhanced multi-scale n-gram consensus with confidence weighting and fuzzy matching"""
        if not current_text or not other_texts:
            return 0.0
        
        # Filter out empty texts
        valid_other_texts = [text for text in other_texts if text.strip()]
        if not valid_other_texts:
            return 0.0
        
        # Multi-scale n-gram analysis (1-grams through 4-grams)
        ngram_weights = {1: 0.15, 2: 0.25, 3: 0.35, 4: 0.25}  # Higher weight for 3-grams
        
        total_score = 0.0
        for n in range(1, 5):
            ngram_score = self._calculate_weighted_ngram_consensus(
                current_text, valid_other_texts, n
            )
            total_score += ngram_weights[n] * ngram_score
        
        return total_score
    
    def _calculate_weighted_ngram_consensus(self, current_text: str, other_texts: List[str], n: int) -> float:
        """Calculate consensus for specific n-gram size with advanced weighting"""
        # Extract n-grams with positional information
        current_ngrams = self._extract_ngrams_with_positions(current_text, n)
        
        consensus_scores = []
        
        for other_text in other_texts:
            other_ngrams = self._extract_ngrams_with_positions(other_text, n)
            
            # Calculate multi-dimensional similarity
            exact_similarity = self._calculate_exact_ngram_similarity(current_ngrams, other_ngrams)
            fuzzy_similarity = self._calculate_fuzzy_ngram_similarity(current_ngrams, other_ngrams)
            positional_similarity = self._calculate_positional_ngram_similarity(current_ngrams, other_ngrams)
            idf_weighted_similarity = self._calculate_idf_weighted_similarity(current_ngrams, other_ngrams, other_texts, n)
            
            # Combine similarities with weights
            combined_similarity = (
                0.40 * exact_similarity +
                0.25 * fuzzy_similarity +
                0.20 * positional_similarity +
                0.15 * idf_weighted_similarity
            )
            
            consensus_scores.append(combined_similarity)
        
        return float(np.mean(consensus_scores)) if consensus_scores else 0.0
    
    def _extract_ngrams_with_positions(self, text: str, n: int) -> Dict[str, List[int]]:
        """Extract n-grams with their positions in the text"""
        words = self._clean_text_for_ngrams(text)
        ngram_positions = defaultdict(list)
        
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngram_positions[ngram].append(i)
        
        return dict(ngram_positions)
    
    def _clean_text_for_ngrams(self, text: str) -> List[str]:
        """Clean and normalize text for n-gram analysis"""
        # Convert to lowercase and remove excessive punctuation
        text = text.lower()
        # Keep basic punctuation that affects meaning
        text = re.sub(r'[^\w\s\.\?\!\,\;\:]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        # Remove very short artifacts but keep meaningful short words
        meaningful_short = {'i', 'a', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'is', 'are', 'was', 'be', 'do', 'no', 'so', 'or', 'if'}
        cleaned_words = [w for w in words if len(w) > 1 or w in meaningful_short]
        
        return cleaned_words
    
    def _calculate_exact_ngram_similarity(self, current_ngrams: Dict[str, List[int]], 
                                        other_ngrams: Dict[str, List[int]]) -> float:
        """Calculate exact n-gram overlap using Jaccard similarity"""
        current_set = set(current_ngrams.keys())
        other_set = set(other_ngrams.keys())
        
        if not current_set and not other_set:
            return 1.0
        
        intersection = len(current_set & other_set)
        union = len(current_set | other_set)
        
        return intersection / max(union, 1)
    
    def _calculate_fuzzy_ngram_similarity(self, current_ngrams: Dict[str, List[int]], 
                                        other_ngrams: Dict[str, List[int]]) -> float:
        """Calculate fuzzy n-gram similarity to handle minor variations"""
        if not current_ngrams or not other_ngrams:
            return 0.0
        
        current_list = list(current_ngrams.keys())
        other_list = list(other_ngrams.keys())
        
        fuzzy_matches = 0
        total_comparisons = 0
        
        # For each n-gram in current, find best fuzzy match in other
        for current_ngram in current_list:
            best_similarity = 0.0
            for other_ngram in other_list:
                similarity = self._calculate_string_similarity(current_ngram, other_ngram)
                best_similarity = max(best_similarity, similarity)
            
            # Count as match if similarity is high enough
            if best_similarity >= 0.8:  # 80% similarity threshold
                fuzzy_matches += 1
            total_comparisons += 1
        
        return fuzzy_matches / max(total_comparisons, 1)
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using sequence matching"""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _calculate_positional_ngram_similarity(self, current_ngrams: Dict[str, List[int]], 
                                             other_ngrams: Dict[str, List[int]]) -> float:
        """Calculate similarity considering positional alignment of n-grams"""
        if not current_ngrams or not other_ngrams:
            return 0.0
        
        position_aware_score = 0.0
        total_weight = 0.0
        
        for ngram, current_positions in current_ngrams.items():
            if ngram in other_ngrams:
                other_positions = other_ngrams[ngram]
                
                # Calculate position alignment score
                position_score = self._calculate_position_alignment(current_positions, other_positions)
                
                # Weight by frequency (more frequent n-grams get higher weight)
                frequency_weight = math.sqrt(len(current_positions) * len(other_positions))
                
                position_aware_score += position_score * frequency_weight
                total_weight += frequency_weight
        
        return position_aware_score / max(total_weight, 1)
    
    def _calculate_position_alignment(self, pos1: List[int], pos2: List[int]) -> float:
        """Calculate how well positions align between two n-gram occurrence lists"""
        if not pos1 or not pos2:
            return 0.0
        
        # Normalize positions to [0, 1] range
        max_pos1 = max(pos1)
        max_pos2 = max(pos2)
        
        norm_pos1 = [p / max(max_pos1, 1) for p in pos1]
        norm_pos2 = [p / max(max_pos2, 1) for p in pos2]
        
        # Find best alignment score
        min_distance = float('inf')
        for p1 in norm_pos1:
            for p2 in norm_pos2:
                distance = abs(p1 - p2)
                min_distance = min(min_distance, distance)
        
        # Convert distance to similarity (closer positions = higher score)
        if min_distance == float('inf'):
            return 0.0
        
        return max(0.0, 1.0 - min_distance)
    
    def _calculate_idf_weighted_similarity(self, current_ngrams: Dict[str, List[int]], 
                                         other_ngrams: Dict[str, List[int]], 
                                         all_texts: List[str], n: int) -> float:
        """Calculate IDF-weighted similarity to emphasize rare/important phrases"""
        if not current_ngrams or not other_ngrams:
            return 0.0
        
        # Calculate IDF scores for all n-grams
        idf_scores = self._calculate_ngram_idf_scores(all_texts, n)
        
        weighted_intersection = 0.0
        total_current_weight = 0.0
        total_other_weight = 0.0
        
        # Calculate weighted scores
        for ngram in current_ngrams:
            idf_weight = idf_scores.get(ngram, 1.0)
            current_weight = len(current_ngrams[ngram]) * idf_weight
            total_current_weight += current_weight
            
            if ngram in other_ngrams:
                other_weight = len(other_ngrams[ngram]) * idf_weight
                weighted_intersection += min(current_weight, other_weight)
        
        for ngram in other_ngrams:
            if ngram not in current_ngrams:
                idf_weight = idf_scores.get(ngram, 1.0)
                other_weight = len(other_ngrams[ngram]) * idf_weight
                total_other_weight += other_weight
        
        total_weight = total_current_weight + total_other_weight
        return (2 * weighted_intersection) / max(total_weight, 1)
    
    def _calculate_ngram_idf_scores(self, texts: List[str], n: int) -> Dict[str, float]:
        """Calculate IDF scores for n-grams across all texts"""
        # Count document frequency for each n-gram
        ngram_doc_count = defaultdict(int)
        total_docs = len(texts)
        
        for text in texts:
            if not text.strip():
                continue
            
            text_ngrams = set(self._get_ngrams(text, n))
            for ngram in text_ngrams:
                ngram_doc_count[ngram] += 1
        
        # Calculate IDF scores
        idf_scores = {}
        for ngram, doc_count in ngram_doc_count.items():
            idf = math.log(total_docs / max(doc_count, 1))
            idf_scores[ngram] = idf
        
        return idf_scores
    
    def _get_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text"""
        words = text.lower().split()
        ngrams = []
        
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _calculate_named_entity_consensus(self, current_text: str, other_texts: List[str]) -> float:
        """Calculate enhanced named entity consensus with advanced detection and categorization"""
        if not current_text or not other_texts:
            return 0.0
        
        # Filter out empty texts
        valid_other_texts = [text for text in other_texts if text.strip()]
        if not valid_other_texts:
            return 0.0
        
        # Extract categorized entities from current text
        current_entities = self._extract_categorized_entities(current_text)
        
        consensus_scores = []
        
        for other_text in valid_other_texts:
            other_entities = self._extract_categorized_entities(other_text)
            
            # Calculate multi-dimensional entity consensus
            exact_entity_score = self._calculate_exact_entity_consensus(current_entities, other_entities)
            fuzzy_entity_score = self._calculate_fuzzy_entity_consensus(current_entities, other_entities)
            category_consistency_score = self._calculate_category_consistency(current_entities, other_entities)
            entity_density_score = self._calculate_entity_density_consistency(current_entities, other_entities, current_text, other_text)
            
            # Weighted combination of entity consensus dimensions
            combined_score = (
                0.35 * exact_entity_score +
                0.30 * fuzzy_entity_score +
                0.20 * category_consistency_score +
                0.15 * entity_density_score
            )
            
            consensus_scores.append(combined_score)
        
        return float(np.mean(consensus_scores)) if consensus_scores else 0.0
    
    def _extract_categorized_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract and categorize named entities with confidence and context"""
        entities = {}
        
        # Extract different types of entities
        person_entities = self._extract_person_entities(text)
        place_entities = self._extract_place_entities(text)
        organization_entities = self._extract_organization_entities(text)
        date_time_entities = self._extract_datetime_entities(text)
        number_entities = self._extract_number_entities(text)
        technical_entities = self._extract_technical_entities(text)
        
        # Combine all entities with categories
        entities.update(person_entities)
        entities.update(place_entities)
        entities.update(organization_entities)
        entities.update(date_time_entities)
        entities.update(number_entities)
        entities.update(technical_entities)
        
        return entities
    
    def _extract_person_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract person name entities with enhanced patterns"""
        entities = {}
        words = text.split()
        
        # Common patterns for person names
        name_patterns = [
            # Title + Name patterns
            r'\b(?:mr|ms|mrs|dr|prof|professor|sir|madam)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # First + Last name patterns
            r'\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b',
            # Single names with name suffixes
            r'\b([A-Z][a-z]{2,})\s+(?:jr|sr|iii?|iv)\b',
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group().strip()
                normalized = self._normalize_entity(entity_text)
                
                entities[normalized] = {
                    'category': 'PERSON',
                    'confidence': self._calculate_entity_confidence(entity_text, 'PERSON'),
                    'original_form': entity_text,
                    'position': match.start()
                }
        
        # Additional heuristic: Isolated capitalized words that might be names
        capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in capitalized_words:
            if self._is_likely_person_name(word, text):
                normalized = self._normalize_entity(word)
                if normalized not in entities:
                    entities[normalized] = {
                        'category': 'PERSON',
                        'confidence': 0.6,  # Lower confidence for heuristic matches
                        'original_form': word,
                        'position': text.lower().find(word.lower())
                    }
        
        return entities
    
    def _extract_place_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract place/location entities"""
        entities = {}
        
        # Common place patterns
        place_patterns = [
            # City, State patterns
            r'\b([A-Z][a-z]+),\s*([A-Z][A-Z]|[A-Z][a-z]+)\b',
            # State/Country names
            r'\b(?:in|from|to|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            # Street addresses
            r'\b\d+\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd)\b',
        ]
        
        # Known place suffixes and indicators
        place_indicators = {
            'city', 'town', 'village', 'county', 'state', 'country', 'street', 'avenue', 
            'road', 'drive', 'boulevard', 'university', 'college', 'hospital', 'airport',
            'center', 'building', 'tower', 'plaza', 'mall', 'park'
        }
        
        for pattern in place_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group().strip()
                normalized = self._normalize_entity(entity_text)
                
                entities[normalized] = {
                    'category': 'PLACE',
                    'confidence': self._calculate_entity_confidence(entity_text, 'PLACE'),
                    'original_form': entity_text,
                    'position': match.start()
                }
        
        # Look for words near place indicators
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in place_indicators and i > 0:
                prev_word = words[i-1]
                if prev_word and prev_word[0].isupper() and len(prev_word) > 2:
                    normalized = self._normalize_entity(prev_word)
                    if normalized not in entities:
                        entities[normalized] = {
                            'category': 'PLACE',
                            'confidence': 0.7,
                            'original_form': prev_word,
                            'position': text.find(prev_word)
                        }
        
        return entities
    
    def _extract_organization_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract organization entities"""
        entities = {}
        
        # Organization patterns
        org_patterns = [
            # Company suffixes
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:inc|corp|corporation|llc|ltd|co|company)\b',
            # University patterns  
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:university|college|institute)\b',
            # Agency patterns
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:agency|department|bureau|commission)\b',
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group().strip()
                normalized = self._normalize_entity(entity_text)
                
                entities[normalized] = {
                    'category': 'ORGANIZATION',
                    'confidence': self._calculate_entity_confidence(entity_text, 'ORGANIZATION'),
                    'original_form': entity_text,
                    'position': match.start()
                }
        
        return entities
    
    def _extract_datetime_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract date and time entities"""
        entities = {}
        
        # Date/time patterns
        datetime_patterns = [
            # Date patterns
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            # Time patterns
            r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b',
            # Relative time
            r'\b(?:yesterday|today|tomorrow|last\s+week|next\s+week|this\s+morning|this\s+afternoon)\b',
        ]
        
        for pattern in datetime_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group().strip()
                normalized = self._normalize_entity(entity_text)
                
                entities[normalized] = {
                    'category': 'DATETIME',
                    'confidence': 0.9,  # High confidence for pattern matches
                    'original_form': entity_text,
                    'position': match.start()
                }
        
        return entities
    
    def _extract_number_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract number entities (quantities, percentages, etc.)"""
        entities = {}
        
        # Number patterns
        number_patterns = [
            # Percentages
            r'\b\d+(?:\.\d+)?%\b',
            # Currency
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
            # Large numbers
            r'\b\d{1,3}(?:,\d{3})+\b',
            # Decimal numbers
            r'\b\d+\.\d+\b',
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_text = match.group().strip()
                normalized = self._normalize_entity(entity_text)
                
                entities[normalized] = {
                    'category': 'NUMBER',
                    'confidence': 0.95,  # Very high confidence for number patterns
                    'original_form': entity_text,
                    'position': match.start()
                }
        
        return entities
    
    def _extract_technical_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract technical terms and domain-specific entities"""
        entities = {}
        
        # Technical patterns
        technical_patterns = [
            # Email addresses
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            # URLs
            r'\bwww\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            r'\bhttps?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b',
            # Phone numbers
            r'\b\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            # Product codes/model numbers
            r'\b[A-Z0-9]{2,}-[A-Z0-9]{2,}\b',
        ]
        
        for pattern in technical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group().strip()
                normalized = self._normalize_entity(entity_text)
                
                entities[normalized] = {
                    'category': 'TECHNICAL',
                    'confidence': 0.9,
                    'original_form': entity_text,
                    'position': match.start()
                }
        
        return entities
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity for comparison"""
        # Remove punctuation and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', entity.lower())
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _calculate_entity_confidence(self, entity: str, category: str) -> float:
        """Calculate confidence score for an entity"""
        base_confidence = 0.8
        
        # Adjust based on entity characteristics
        if len(entity) < 3:
            base_confidence -= 0.3
        elif len(entity) > 20:
            base_confidence -= 0.1
        
        # Category-specific adjustments
        if category == 'PERSON':
            # Names with titles get higher confidence
            if re.search(r'\b(?:mr|ms|mrs|dr|prof)\b', entity.lower()):
                base_confidence += 0.1
        elif category == 'PLACE':
            # Places with indicators get higher confidence
            if re.search(r'\b(?:city|street|avenue|university)\b', entity.lower()):
                base_confidence += 0.1
        
        return min(1.0, max(0.1, base_confidence))
    
    def _is_likely_person_name(self, word: str, context: str) -> bool:
        """Heuristic to determine if a capitalized word is likely a person name"""
        # Avoid common non-name capitalized words
        non_names = {
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
            'September', 'October', 'November', 'December', 'Internet', 'Google', 'Microsoft',
            'Apple', 'Facebook', 'Twitter', 'LinkedIn', 'YouTube', 'Amazon', 'Netflix'
        }
        
        if word in non_names:
            return False
        
        # Check if word appears near name indicators
        name_indicators = ['said', 'told', 'asked', 'replied', 'mentioned', 'according to']
        for indicator in name_indicators:
            if indicator in context.lower() and word in context:
                return True
        
        # Names typically have certain length characteristics
        if 3 <= len(word) <= 12:
            return True
        
        return False
    
    def _calculate_exact_entity_consensus(self, current_entities: Dict[str, Dict[str, Any]], 
                                        other_entities: Dict[str, Dict[str, Any]]) -> float:
        """Calculate exact entity overlap consensus"""
        if not current_entities and not other_entities:
            return 1.0
        
        current_names = set(current_entities.keys())
        other_names = set(other_entities.keys())
        
        intersection = len(current_names & other_names)
        union = len(current_names | other_names)
        
        return intersection / max(union, 1)
    
    def _calculate_fuzzy_entity_consensus(self, current_entities: Dict[str, Dict[str, Any]], 
                                        other_entities: Dict[str, Dict[str, Any]]) -> float:
        """Calculate fuzzy entity matching consensus"""
        if not current_entities or not other_entities:
            return 0.0
        
        current_names = list(current_entities.keys())
        other_names = list(other_entities.keys())
        
        fuzzy_matches = 0
        total_entities = len(current_names)
        
        for current_name in current_names:
            best_match_score = 0.0
            for other_name in other_names:
                # Use multiple similarity measures
                sequence_sim = SequenceMatcher(None, current_name, other_name).ratio()
                
                # Handle common variations (plurals, abbreviations)
                variation_sim = self._calculate_entity_variation_similarity(current_name, other_name)
                
                combined_sim = max(sequence_sim, variation_sim)
                best_match_score = max(best_match_score, combined_sim)
            
            # Count as fuzzy match if similarity is high enough
            if best_match_score >= 0.75:  # 75% similarity threshold
                fuzzy_matches += 1
        
        return fuzzy_matches / max(total_entities, 1)
    
    def _calculate_entity_variation_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity accounting for common name variations"""
        # Handle common variations
        variations = [
            # Remove common suffixes
            (r'\s+jr\.?$', ''),
            (r'\s+sr\.?$', ''),
            (r'\s+iii?$', ''),
            # Handle abbreviations
            (r'\b([A-Z])\w+', r'\1'),  # First letter abbreviations
        ]
        
        name1_variants = [name1]
        name2_variants = [name2]
        
        for pattern, replacement in variations:
            name1_variant = re.sub(pattern, replacement, name1, flags=re.IGNORECASE)
            name2_variant = re.sub(pattern, replacement, name2, flags=re.IGNORECASE)
            
            if name1_variant != name1:
                name1_variants.append(name1_variant)
            if name2_variant != name2:
                name2_variants.append(name2_variant)
        
        # Find best similarity among all variants
        best_similarity = 0.0
        for var1 in name1_variants:
            for var2 in name2_variants:
                similarity = SequenceMatcher(None, var1, var2).ratio()
                best_similarity = max(best_similarity, similarity)
        
        return best_similarity
    
    def _calculate_category_consistency(self, current_entities: Dict[str, Dict[str, Any]], 
                                      other_entities: Dict[str, Dict[str, Any]]) -> float:
        """Calculate consistency of entity categorization"""
        if not current_entities or not other_entities:
            return 0.0
        
        # Find entities that appear in both sets
        common_entities = set(current_entities.keys()) & set(other_entities.keys())
        
        if not common_entities:
            return 0.0
        
        category_matches = 0
        for entity in common_entities:
            current_category = current_entities[entity]['category']
            other_category = other_entities[entity]['category']
            
            if current_category == other_category:
                category_matches += 1
        
        return category_matches / len(common_entities)
    
    def _calculate_entity_density_consistency(self, current_entities: Dict[str, Dict[str, Any]], 
                                            other_entities: Dict[str, Dict[str, Any]], 
                                            current_text: str, other_text: str) -> float:
        """Calculate consistency of entity density (entities per word ratio)"""
        current_words = len(current_text.split())
        other_words = len(other_text.split())
        
        if current_words == 0 or other_words == 0:
            return 0.0
        
        current_density = len(current_entities) / current_words
        other_density = len(other_entities) / other_words
        
        # Calculate similarity of densities
        if current_density == 0 and other_density == 0:
            return 1.0
        
        max_density = max(current_density, other_density)
        min_density = min(current_density, other_density)
        
        if max_density == 0:
            return 1.0
        
        return min_density / max_density
    
    def _calculate_topic_coherence(self, candidate: Dict[str, Any]) -> float:
        """Calculate enhanced topic coherence with multi-segment analysis and conversation flow"""
        segments = candidate.get('aligned_segments', [])
        if len(segments) < 2:
            return 0.8
        
        # Extract text and metadata from segments
        segment_data = []
        for seg in segments:
            text = seg.get('text', '').strip()
            if text:
                segment_data.append({
                    'text': text,
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'speaker_id': seg.get('speaker_id', 'unknown')
                })
        
        if len(segment_data) < 2:
            return 0.5
        
        # Multi-dimensional topic coherence analysis
        adjacent_coherence = self._calculate_adjacent_segment_coherence(segment_data)
        keyword_consistency = self._calculate_keyword_consistency(segment_data)
        semantic_coherence = self._calculate_semantic_topic_coherence(segment_data)
        conversation_flow = self._calculate_conversation_flow_score(segment_data)
        topic_transitions = self._calculate_topic_transition_quality(segment_data)
        
        # Weighted combination of coherence dimensions
        final_coherence = (
            0.30 * adjacent_coherence +
            0.25 * keyword_consistency +
            0.20 * semantic_coherence +
            0.15 * conversation_flow +
            0.10 * topic_transitions
        )
        
        return float(final_coherence)
    
    def _calculate_adjacent_segment_coherence(self, segment_data: List[Dict[str, Any]]) -> float:
        """Calculate enhanced adjacent segment coherence with multiple similarity measures"""
        if len(segment_data) < 2:
            return 0.8
        
        segment_texts = [seg['text'] for seg in segment_data]
        
        try:
            # Multi-scale TF-IDF analysis using cached vectorizers
            coherence_scores = []
            
            # Fine-grained analysis (more features)
            vectorizer_fine, tfidf_fine = self._get_fitted_vectorizer(
                'coherence_fine', segment_texts,
                max_features=200, 
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=1
            )
            
            # Coarse-grained analysis (fewer features, more general topics)
            vectorizer_coarse, tfidf_coarse = self._get_fitted_vectorizer(
                'coherence_coarse', segment_texts,
                max_features=50,
                stop_words='english',
                ngram_range=(1, 1),
                min_df=1
            )
            
            # Calculate similarities at different scales
            for i in range(len(segment_texts) - 1):
                # Fine-grained similarity
                fine_sim = cosine_similarity(tfidf_fine[i:i+1], tfidf_fine[i+1:i+2])[0][0]
                
                # Coarse-grained similarity
                coarse_sim = cosine_similarity(tfidf_coarse[i:i+1], tfidf_coarse[i+1:i+2])[0][0]
                
                # Lexical overlap similarity
                lexical_sim = self._calculate_lexical_overlap(segment_texts[i], segment_texts[i+1])
                
                # Speaker consistency bonus
                speaker_bonus = self._calculate_speaker_consistency_bonus(
                    segment_data[i], segment_data[i+1]
                )
                
                # Combined similarity with speaker awareness
                combined_sim = (
                    0.40 * fine_sim +
                    0.30 * coarse_sim +
                    0.20 * lexical_sim +
                    0.10 * speaker_bonus
                )
                
                coherence_scores.append(combined_sim)
            
            return float(np.mean(coherence_scores)) if coherence_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_lexical_overlap(self, text1: str, text2: str) -> float:
        """Calculate lexical overlap between two text segments"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / max(union, 1)
    
    def _calculate_speaker_consistency_bonus(self, seg1: Dict[str, Any], seg2: Dict[str, Any]) -> float:
        """Calculate bonus for speaker consistency between segments"""
        speaker1 = seg1.get('speaker_id', 'unknown')
        speaker2 = seg2.get('speaker_id', 'unknown')
        
        # Same speaker gets higher coherence bonus
        if speaker1 == speaker2:
            return 0.8
        # Different speakers but reasonable transition gets moderate bonus
        else:
            # Check if this is a reasonable speaker turn (not too short)
            duration1 = seg1.get('end', 0) - seg1.get('start', 0)
            duration2 = seg2.get('end', 0) - seg2.get('start', 0)
            
            if duration1 >= 2.0 and duration2 >= 2.0:  # Both segments at least 2 seconds
                return 0.6
            else:
                return 0.3  # Penalty for very short segments (likely diarization errors)
    
    def _calculate_keyword_consistency(self, segment_data: List[Dict[str, Any]]) -> float:
        """Calculate consistency of important keywords across segments"""
        if len(segment_data) < 3:
            return 0.7
        
        # Extract all text for keyword analysis
        full_text = ' '.join([seg['text'] for seg in segment_data])
        
        # Identify important keywords using TF-IDF
        important_keywords = self._extract_important_keywords(full_text, segment_data)
        
        if not important_keywords:
            return 0.6
        
        # Calculate keyword distribution across segments
        keyword_distribution_score = self._analyze_keyword_distribution(important_keywords, segment_data)
        
        # Calculate keyword co-occurrence patterns
        keyword_cooccurrence_score = self._analyze_keyword_cooccurrence(important_keywords, segment_data)
        
        # Calculate keyword progression (for meeting flow)
        keyword_progression_score = self._analyze_keyword_progression(important_keywords, segment_data)
        
        # Combine keyword consistency metrics
        return (
            0.40 * keyword_distribution_score +
            0.35 * keyword_cooccurrence_score +
            0.25 * keyword_progression_score
        )
    
    def _extract_important_keywords(self, full_text: str, segment_data: List[Dict[str, Any]]) -> List[str]:
        """Extract important keywords using TF-IDF and domain patterns"""
        try:
            # Create corpus for TF-IDF
            segment_texts = [seg['text'] for seg in segment_data]
            
            vectorizer, tfidf_matrix = self._get_fitted_vectorizer(
                'keyword_extraction', segment_texts,
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,  # Must appear in at least 2 segments
                max_df=0.8  # But not in more than 80% of segments
            )
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores
            # Convert sparse matrix to dense array
            # TF-IDF vectorizer returns scipy sparse matrix which has toarray() method
            tfidf_array = np.asarray(tfidf_matrix.toarray())  # type: ignore
            mean_scores = np.mean(tfidf_array, axis=0)
            
            # Get top keywords
            top_indices = mean_scores.argsort()[-20:][::-1]  # Top 20 keywords
            important_keywords = [str(feature_names[i]) for i in top_indices if mean_scores[i] > 0.1]
            
            # Add domain-specific keywords (business, meeting terms)
            domain_keywords = self._extract_domain_keywords(full_text)
            important_keywords.extend(domain_keywords)
            
            return list(set(important_keywords))[:25]  # Limit to top 25
            
        except Exception:
            return []
    
    def _extract_domain_keywords(self, text: str) -> List[str]:
        """Extract domain-specific keywords for business/meeting contexts"""
        domain_patterns = [
            # Business terms
            r'\b(?:project|budget|deadline|timeline|milestone|deliverable|stakeholder|client|customer)\b',
            # Meeting terms
            r'\b(?:agenda|action\s+item|follow\s+up|decision|approval|review|discussion|presentation)\b',
            # Process terms
            r'\b(?:process|procedure|workflow|implementation|strategy|plan|goal|objective|target)\b',
            # Technical terms
            r'\b(?:system|platform|software|application|database|server|integration|api|framework)\b',
        ]
        
        domain_keywords = []
        for pattern in domain_patterns:
            matches = re.findall(pattern, text.lower())
            domain_keywords.extend(matches)
        
        # Clean and deduplicate
        cleaned_keywords = []
        for keyword in domain_keywords:
            # Normalize multi-word terms
            normalized = re.sub(r'\s+', ' ', keyword.strip())
            if len(normalized) > 2:
                cleaned_keywords.append(normalized)
        
        return list(set(cleaned_keywords))[:10]  # Top 10 domain keywords
    
    def _analyze_keyword_distribution(self, keywords: List[str], segment_data: List[Dict[str, Any]]) -> float:
        """Analyze how evenly important keywords are distributed across segments"""
        if not keywords:
            return 0.6
        
        keyword_segment_counts = {}
        total_segments = len(segment_data)
        
        for keyword in keywords:
            count = 0
            for seg in segment_data:
                if keyword in seg['text'].lower():
                    count += 1
            keyword_segment_counts[keyword] = count
        
        # Calculate distribution evenness
        # Good keywords should appear in multiple segments but not all
        good_distribution_count = 0
        for keyword, count in keyword_segment_counts.items():
            # Ideal: appears in 25-75% of segments
            ratio = count / total_segments
            if 0.25 <= ratio <= 0.75:
                good_distribution_count += 1
        
        return good_distribution_count / max(len(keywords), 1)
    
    def _analyze_keyword_cooccurrence(self, keywords: List[str], segment_data: List[Dict[str, Any]]) -> float:
        """Analyze how well keywords co-occur in segments (topic clustering)"""
        if len(keywords) < 2:
            return 0.7
        
        # Find segments with multiple keywords
        segments_with_multiple_keywords = 0
        total_keyword_segments = 0
        
        for seg in segment_data:
            text_lower = seg['text'].lower()
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            
            if keyword_count > 0:
                total_keyword_segments += 1
                if keyword_count >= 2:
                    segments_with_multiple_keywords += 1
        
        if total_keyword_segments == 0:
            return 0.5
        
        # Good topic coherence: segments with keywords often have multiple related keywords
        cooccurrence_ratio = segments_with_multiple_keywords / total_keyword_segments
        
        # Normalize to reasonable range
        return min(0.9, cooccurrence_ratio * 1.5)
    
    def _analyze_keyword_progression(self, keywords: List[str], segment_data: List[Dict[str, Any]]) -> float:
        """Analyze natural progression/flow of keywords through the conversation"""
        if len(keywords) < 2 or len(segment_data) < 3:
            return 0.7
        
        # Track keyword appearances over time
        keyword_timeline = {}
        for keyword in keywords:
            keyword_timeline[keyword] = []
        
        for i, seg in enumerate(segment_data):
            text_lower = seg['text'].lower()
            for keyword in keywords:
                if keyword in text_lower:
                    keyword_timeline[keyword].append(i)
        
        # Analyze progression patterns
        progression_scores = []
        
        for keyword, positions in keyword_timeline.items():
            if len(positions) >= 2:
                # Check for reasonable clustering (not too scattered)
                position_spread = max(positions) - min(positions)
                total_span = len(segment_data) - 1
                
                if total_span > 0:
                    clustering_score = 1.0 - (position_spread / total_span)
                    progression_scores.append(max(0.3, clustering_score))
        
        return float(np.mean(progression_scores)) if progression_scores else 0.6
    
    def _calculate_semantic_topic_coherence(self, segment_data: List[Dict[str, Any]]) -> float:
        """Calculate semantic coherence using advanced topic modeling"""
        if len(segment_data) < 3:
            return 0.7
        
        segment_texts = [seg['text'] for seg in segment_data]
        
        # Semantic similarity using word overlap and context
        semantic_coherence = self._calculate_semantic_similarity_matrix(segment_texts)
        
        # Topic consistency across conversation
        topic_consistency = self._calculate_segment_topic_consistency(segment_texts)
        
        # Conversation naturalness
        naturalness_score = self._calculate_conversation_naturalness(segment_data)
        
        return (
            0.40 * semantic_coherence +
            0.35 * topic_consistency +
            0.25 * naturalness_score
        )
    
    def _calculate_semantic_similarity_matrix(self, texts: List[str]) -> float:
        """Calculate semantic similarity using multiple approaches"""
        if len(texts) < 2:
            return 0.7
        
        # Character-level similarity (for handling ASR errors)
        char_similarities = []
        for i in range(len(texts) - 1):
            char_sim = SequenceMatcher(None, texts[i], texts[i+1]).ratio()
            char_similarities.append(char_sim)
        
        char_score = float(np.mean(char_similarities)) if char_similarities else 0.5
        
        # Word-level semantic overlap
        word_similarities = []
        for i in range(len(texts) - 1):
            word_sim = self._calculate_word_semantic_similarity(texts[i], texts[i+1])
            word_similarities.append(word_sim)
        
        word_score = float(np.mean(word_similarities)) if word_similarities else 0.5
        
        # Combine character and word level similarities
        return 0.3 * char_score + 0.7 * word_score
    
    def _calculate_word_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text segments"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove punctuation and short words
        words1 = {re.sub(r'[^\w]', '', w) for w in words1 if len(w) > 2}
        words2 = {re.sub(r'[^\w]', '', w) for w in words2 if len(w) > 2}
        
        # Direct overlap
        direct_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
        
        # Stemmed overlap (simple stemming)
        stemmed_words1 = {self._simple_stem(w) for w in words1}
        stemmed_words2 = {self._simple_stem(w) for w in words2}
        stemmed_overlap = len(stemmed_words1 & stemmed_words2) / max(len(stemmed_words1 | stemmed_words2), 1)
        
        return 0.6 * direct_overlap + 0.4 * stemmed_overlap
    
    def _simple_stem(self, word: str) -> str:
        """Simple stemming for basic word variants"""
        # Remove common suffixes
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'es']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def _calculate_segment_topic_consistency(self, texts: List[str]) -> float:
        """Calculate overall topic consistency across all segments"""
        if len(texts) < 3:
            return 0.7
        
        # Global keyword extraction
        all_text = ' '.join(texts)
        
        try:
            vectorizer, tfidf_matrix = self._get_fitted_vectorizer(
                'topic_consistency', texts,
                max_features=30,
                stop_words='english',
                min_df=2,
                ngram_range=(1, 2)
            )
            
            # Calculate variance in TF-IDF scores across segments
            # Lower variance = more consistent topics
            # Convert sparse matrix to dense array
            # TF-IDF vectorizer returns scipy sparse matrix which has toarray() method
            tfidf_array = np.asarray(tfidf_matrix.toarray())  # type: ignore
            feature_variances = np.var(tfidf_array, axis=0)
            mean_variance = float(np.mean(feature_variances))
            
            # Convert variance to consistency score (lower variance = higher consistency)
            consistency_score = 1.0 / (1.0 + mean_variance * 5)
            
            return float(consistency_score)
            
        except Exception:
            return 0.6
    
    def _calculate_conversation_naturalness(self, segment_data: List[Dict[str, Any]]) -> float:
        """Calculate how natural the conversation flow appears"""
        if len(segment_data) < 2:
            return 0.8
        
        # Check for natural conversation patterns
        speaker_alternation = self._calculate_speaker_alternation_naturalness(segment_data)
        segment_length_distribution = self._calculate_segment_length_naturalness(segment_data)
        response_patterns = self._calculate_response_pattern_naturalness(segment_data)
        
        return (
            0.40 * speaker_alternation +
            0.30 * segment_length_distribution +
            0.30 * response_patterns
        )
    
    def _calculate_speaker_alternation_naturalness(self, segment_data: List[Dict[str, Any]]) -> float:
        """Check for natural speaker alternation patterns"""
        if len(segment_data) < 3:
            return 0.8
        
        speakers = [seg.get('speaker_id', 'unknown') for seg in segment_data]
        
        # Count speaker changes
        changes = sum(1 for i in range(len(speakers) - 1) if speakers[i] != speakers[i+1])
        
        # Natural conversation should have some but not excessive speaker changes
        change_ratio = changes / max(len(speakers) - 1, 1)
        
        # Optimal range: 30-70% of segments involve speaker changes
        if 0.3 <= change_ratio <= 0.7:
            return 0.9
        elif 0.2 <= change_ratio <= 0.8:
            return 0.7
        else:
            return 0.5
    
    def _calculate_segment_length_naturalness(self, segment_data: List[Dict[str, Any]]) -> float:
        """Check for natural distribution of segment lengths"""
        durations = []
        word_counts = []
        
        for seg in segment_data:
            duration = seg.get('end', 0) - seg.get('start', 0)
            word_count = len(seg['text'].split())
            
            durations.append(duration)
            word_counts.append(word_count)
        
        if not durations:
            return 0.5
        
        # Check duration distribution
        mean_duration = float(np.mean(durations))
        std_duration = float(np.std(durations))
        
        # Natural speech: average 3-8 seconds, reasonable variation
        duration_score = 1.0 if 3.0 <= mean_duration <= 8.0 else max(0.4, 1.0 - abs(mean_duration - 5.5) * 0.2)
        
        # Check for reasonable variation (not all segments same length)
        variation_score = min(1.0, std_duration / max(mean_duration, 1)) if mean_duration > 0 else 0.5
        variation_score = max(0.3, variation_score)
        
        return (duration_score + variation_score) / 2
    
    def _calculate_response_pattern_naturalness(self, segment_data: List[Dict[str, Any]]) -> float:
        """Check for natural response patterns in conversation"""
        if len(segment_data) < 4:
            return 0.8
        
        # Look for question-answer patterns
        question_words = ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'do', 'does', 'did', 'can', 'could', 'would', 'should']
        
        question_answer_pairs = 0
        total_potential_pairs = 0
        
        for i in range(len(segment_data) - 1):
            current_text = segment_data[i]['text'].lower()
            next_text = segment_data[i + 1]['text'].lower()
            current_speaker = segment_data[i].get('speaker_id', 'unknown')
            next_speaker = segment_data[i + 1].get('speaker_id', 'unknown')
            
            # Check if current segment might be a question
            has_question_word = any(word in current_text for word in question_words)
            has_question_mark = '?' in segment_data[i]['text']
            
            if (has_question_word or has_question_mark) and current_speaker != next_speaker:
                total_potential_pairs += 1
                
                # Check if next segment could be an answer
                # Simple heuristics: starts with common answer patterns
                answer_patterns = ['yes', 'no', 'i think', 'well', 'actually', 'sure', 'definitely', 'probably']
                if any(next_text.startswith(pattern) for pattern in answer_patterns):
                    question_answer_pairs += 1
        
        if total_potential_pairs == 0:
            return 0.7  # No clear question patterns found
        
        return question_answer_pairs / total_potential_pairs
    
    def _calculate_conversation_flow_score(self, segment_data: List[Dict[str, Any]]) -> float:
        """Calculate overall conversation flow quality"""
        if len(segment_data) < 3:
            return 0.8
        
        # Conversation momentum (segments build on each other)
        momentum_score = self._calculate_conversation_momentum(segment_data)
        
        # Topic development (topics develop naturally over time)
        development_score = self._calculate_topic_development(segment_data)
        
        # Coherent endings and beginnings
        transition_score = self._calculate_transition_quality(segment_data)
        
        return (
            0.40 * momentum_score +
            0.35 * development_score +
            0.25 * transition_score
        )
    
    def _calculate_conversation_momentum(self, segment_data: List[Dict[str, Any]]) -> float:
        """Calculate how well the conversation maintains momentum"""
        momentum_indicators = {
            'continuation': ['and', 'also', 'furthermore', 'additionally', 'moreover'],
            'contrast': ['but', 'however', 'although', 'despite', 'nevertheless'],
            'consequence': ['so', 'therefore', 'thus', 'consequently', 'as a result'],
            'elaboration': ['specifically', 'for example', 'in particular', 'namely']
        }
        
        momentum_count = 0
        total_segments = len(segment_data)
        
        for seg in segment_data[1:]:  # Skip first segment
            text_lower = seg['text'].lower()
            
            # Check for momentum indicators
            for category, indicators in momentum_indicators.items():
                if any(indicator in text_lower for indicator in indicators):
                    momentum_count += 1
                    break
        
        return momentum_count / max(total_segments - 1, 1)
    
    def _calculate_topic_development(self, segment_data: List[Dict[str, Any]]) -> float:
        """Calculate how well topics develop through the conversation"""
        if len(segment_data) < 4:
            return 0.7
        
        # Split conversation into thirds and analyze topic evolution
        third_size = len(segment_data) // 3
        
        beginning = segment_data[:third_size]
        middle = segment_data[third_size:2*third_size]
        end = segment_data[2*third_size:]
        
        # Extract keywords from each section
        beginning_keywords = self._extract_section_keywords(beginning)
        middle_keywords = self._extract_section_keywords(middle)
        end_keywords = self._extract_section_keywords(end)
        
        # Calculate topic evolution
        beginning_middle_overlap = len(beginning_keywords & middle_keywords) / max(len(beginning_keywords | middle_keywords), 1)
        middle_end_overlap = len(middle_keywords & end_keywords) / max(len(middle_keywords | end_keywords), 1)
        
        # Good development: some continuity but also evolution
        continuity_score = (beginning_middle_overlap + middle_end_overlap) / 2
        
        # Moderate overlap is good (20-60%)
        if 0.2 <= continuity_score <= 0.6:
            return 0.9
        elif 0.1 <= continuity_score <= 0.7:
            return 0.7
        else:
            return 0.5
    
    def _extract_section_keywords(self, section_data: List[Dict[str, Any]]) -> Set[str]:
        """Extract important keywords from a section of the conversation"""
        if not section_data:
            return set()
        
        section_text = ' '.join([seg['text'] for seg in section_data])
        words = section_text.lower().split()
        
        # Remove stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        meaningful_words = [
            re.sub(r'[^\w]', '', word) for word in words 
            if len(word) > 3 and word.lower() not in stop_words
        ]
        
        # Return most frequent meaningful words
        word_counts = Counter(meaningful_words)
        top_words = [word for word, count in word_counts.most_common(10) if count >= 2]
        
        return set(top_words)
    
    def _calculate_transition_quality(self, segment_data: List[Dict[str, Any]]) -> float:
        """Calculate quality of transitions between segments"""
        if len(segment_data) < 2:
            return 0.8
        
        good_transitions = 0
        total_transitions = len(segment_data) - 1
        
        for i in range(len(segment_data) - 1):
            current_seg = segment_data[i]
            next_seg = segment_data[i + 1]
            
            # Check for natural transition indicators
            transition_quality = self._evaluate_single_transition(current_seg, next_seg)
            if transition_quality >= 0.6:
                good_transitions += 1
        
        return good_transitions / max(total_transitions, 1)
    
    def _evaluate_single_transition(self, seg1: Dict[str, Any], seg2: Dict[str, Any]) -> float:
        """Evaluate the quality of a single transition between segments"""
        text1 = seg1['text'].lower()
        text2 = seg2['text'].lower()
        speaker1 = seg1.get('speaker_id', 'unknown')
        speaker2 = seg2.get('speaker_id', 'unknown')
        
        # Lexical overlap
        overlap_score = self._calculate_lexical_overlap(text1, text2)
        
        # Speaker change appropriateness
        if speaker1 != speaker2:
            # Different speakers - check for natural handoff
            handoff_indicators = ['thank you', 'that\'s right', 'exactly', 'yes', 'i agree', 'now']
            if any(indicator in text1 for indicator in handoff_indicators):
                speaker_score = 0.8
            else:
                speaker_score = 0.6
        else:
            # Same speaker - check for continuation
            continuation_indicators = ['and', 'also', 'furthermore', 'so', 'then', 'next']
            if any(indicator in text2 for indicator in continuation_indicators):
                speaker_score = 0.8
            else:
                speaker_score = 0.7
        
        # Time gap appropriateness
        time_gap = seg2.get('start', 0) - seg1.get('end', 0)
        if 0 <= time_gap <= 3.0:  # Reasonable pause
            time_score = 0.9
        elif time_gap <= 5.0:  # Longer but acceptable pause
            time_score = 0.7
        else:  # Very long gap
            time_score = 0.4
        
        return (0.4 * overlap_score + 0.35 * speaker_score + 0.25 * time_score)
    
    def _calculate_topic_transition_quality(self, segment_data: List[Dict[str, Any]]) -> float:
        """Calculate quality of topic transitions throughout the conversation"""
        if len(segment_data) < 5:
            return 0.7
        
        # Use sliding window to detect topic shifts
        window_size = 3
        transition_scores = []
        
        for i in range(len(segment_data) - window_size + 1):
            window = segment_data[i:i + window_size]
            
            # Analyze topic coherence within window
            window_texts = [seg['text'] for seg in window]
            window_coherence = self._calculate_window_topic_coherence(window_texts)
            
            transition_scores.append(window_coherence)
        
        # Calculate overall transition quality
        return float(np.mean(transition_scores)) if transition_scores else 0.6
    
    def _calculate_window_topic_coherence(self, window_texts: List[str]) -> float:
        """Calculate topic coherence within a small window of segments"""
        if len(window_texts) < 2:
            return 0.8
        
        try:
            vectorizer, tfidf_matrix = self._get_fitted_vectorizer(
                'window_coherence', window_texts,
                max_features=20,
                stop_words='english',
                min_df=1
            )
            
            # Calculate pairwise similarities within window
            similarities = []
            for i in range(len(window_texts)):
                for j in range(i + 1, len(window_texts)):
                    sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0][0]
                    similarities.append(sim)
            
            return float(np.mean(similarities)) if similarities else 0.6
            
        except Exception:
            return 0.6
    
    def _calculate_overlap_scores(self, candidates: List[Dict[str, Any]]) -> List[float]:
        """Calculate O1 - Overlap handling quality scores"""
        scores = []
        
        for candidate in candidates:
            diar_data = candidate.get('diarization_data', {})
            
            # O1a: Double-talk correctness
            doubletalk_score = self._calculate_doubletalk_correctness(diar_data)
            
            # O1b: Back-channel precision
            backchannel_score = self._calculate_backchannel_precision(candidate)
            
            # Aggregate O score
            o_score = (
                0.65 * doubletalk_score +
                0.35 * backchannel_score
            )
            
            scores.append(o_score)
        
        return scores
    
    def _calculate_doubletalk_correctness(self, diar_data: Dict[str, Any]) -> float:
        """Calculate correctness of overlap detection"""
        # For demo purposes, return moderate score
        # In practice, this would analyze energy patterns and ASR concurrency
        overlaps = diar_data.get('overlaps', [])
        
        if not overlaps:
            return 0.7  # No overlaps - neutral score
        
        # Basic heuristic: reasonable overlap duration and frequency
        total_overlap_time = sum(o.get('duration', 0) for o in overlaps)
        total_speech_time = diar_data.get('total_speech_time', 1.0)
        
        overlap_ratio = total_overlap_time / total_speech_time
        
        # Meetings typically have 5-15% overlap time
        if 0.05 <= overlap_ratio <= 0.15:
            return 0.85
        elif overlap_ratio < 0.05:
            return 0.7
        else:
            return 0.6
    
    def _calculate_backchannel_precision(self, candidate: Dict[str, Any]) -> float:
        """Calculate precision of back-channel detection"""
        segments = candidate.get('aligned_segments', [])
        if not segments:
            return 0.5
        
        # Look for short segments with back-channel words
        backchannel_words = {'yeah', 'yes', 'mm-hmm', 'uh-huh', 'okay', 'right', 'sure'}
        
        short_segments = 0
        backchannel_segments = 0
        
        for segment in segments:
            duration = segment['end'] - segment['start']
            text = segment.get('text', '').lower().strip()
            
            # Short segments (< 2 seconds)
            if duration < 2.0:
                short_segments += 1
                
                # Check if it's a backchannel
                words = text.split()
                if len(words) <= 2 and any(word in backchannel_words for word in words):
                    backchannel_segments += 1
        
        if short_segments == 0:
            return 0.7  # No short segments
        
        # Good precision means most short segments are backchannels
        precision = backchannel_segments / short_segments
        return min(precision + 0.3, 1.0)  # Add baseline score
    
    def _normalize_scores(self, scores: List[float], dimension: Optional[str] = None) -> List[float]:
        """
        Normalize scores using calibrated absolute ranges for consistent scoring across sessions.
        
        Args:
            scores: Raw scores to normalize
            dimension: Scoring dimension ('D', 'A', 'L', 'R', 'O') for calibrated normalization
            
        Returns:
            Normalized scores with absolute meaning across different processing sessions
        """
        if not scores:
            return []
        
        # Calibrated score ranges based on empirical distributions
        # These ranges provide absolute meaning across different processing sessions
        calibration_ranges = {
            'D': {'min': 0.15, 'max': 0.95, 'median': 0.65},  # Diarization quality
            'A': {'min': 0.25, 'max': 0.98, 'median': 0.75},  # ASR confidence  
            'L': {'min': 0.30, 'max': 0.92, 'median': 0.70},  # Linguistic quality
            'R': {'min': 0.40, 'max': 0.90, 'median': 0.68},  # Cross-run agreement
            'O': {'min': 0.35, 'max': 0.88, 'median': 0.62}   # Overlap handling
        }
        
        if dimension and dimension in calibration_ranges:
            # Use calibrated absolute normalization
            cal_range = calibration_ranges[dimension]
            cal_min = cal_range['min']
            cal_max = cal_range['max']
            
            # Sigmoid-based normalization for better distribution
            normalized = []
            for score in scores:
                # Map to calibrated range first
                if cal_max > cal_min:
                    mapped_score = (score - cal_min) / (cal_max - cal_min)
                else:
                    mapped_score = 0.5
                
                # Apply sigmoid for smoother distribution around median
                sigmoid_score = 1.0 / (1.0 + math.exp(-6.0 * (mapped_score - 0.5)))
                normalized.append(sigmoid_score)
            
            # Clip to valid range
            return [max(0.0, min(1.0, score)) for score in normalized]
        
        else:
            # Fallback to relative normalization for unknown dimensions
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                return [0.5] * len(scores)  # All scores are the same
            
            # Min-max normalization with gentle compression towards center
            normalized = []
            for score in scores:
                relative_score = (score - min_score) / (max_score - min_score)
                # Apply slight compression towards 0.5 to avoid extreme values
                compressed_score = 0.5 + 0.8 * (relative_score - 0.5)
                normalized.append(compressed_score)
            
            # Clip to valid range
            return [max(0.0, min(1.0, score)) for score in normalized]
    
    def select_winner(self, scored_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the winning candidate based on final scores and tie-breakers.
        
        Args:
            scored_candidates: List of candidates with confidence scores
            
        Returns:
            Winning candidate dictionary
        """
        if not scored_candidates:
            raise ValueError("No candidates provided for winner selection")
        
        # Sort by final score (descending)
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x['confidence_scores']['final_score'],
            reverse=True
        )
        
        winner = sorted_candidates[0]
        
        # Check for ties and apply tie-breakers
        final_score = winner['confidence_scores']['final_score']
        tied_candidates = [c for c in sorted_candidates if abs(c['confidence_scores']['final_score'] - final_score) < 0.001]
        
        if len(tied_candidates) > 1:
            winner = self._apply_tie_breakers(tied_candidates)
        
        return winner
    
    def _apply_tie_breakers(self, tied_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply tie-breaking rules to select winner"""
        
        # Tie-breaker 1: Prefer higher A score
        max_a_score = max(c['confidence_scores']['A_asr_alignment'] for c in tied_candidates)
        candidates = [c for c in tied_candidates if abs(c['confidence_scores']['A_asr_alignment'] - max_a_score) < 0.001]
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Tie-breaker 2: Prefer higher R score
        max_r_score = max(c['confidence_scores']['R_agreement'] for c in candidates)
        candidates = [c for c in candidates if abs(c['confidence_scores']['R_agreement'] - max_r_score) < 0.001]
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Tie-breaker 3: Prefer fewer speaker switches per hour
        switch_rates = []
        for candidate in candidates:
            segments = candidate.get('aligned_segments', [])
            if len(segments) < 2:
                switch_rates.append(0.0)
                continue
            
            switches = 0
            for i in range(len(segments) - 1):
                if segments[i]['speaker_id'] != segments[i+1]['speaker_id']:
                    switches += 1
            
            duration_hours = (segments[-1]['end'] - segments[0]['start']) / 3600
            switch_rate = switches / max(duration_hours, 0.1)
            switch_rates.append(switch_rate)
        
        min_switch_rate = min(switch_rates)
        candidates = [candidates[i] for i, rate in enumerate(switch_rates) if abs(rate - min_switch_rate) < 0.1]
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Final tie-breaker: Return first candidate (deterministic)
        return candidates[0]
