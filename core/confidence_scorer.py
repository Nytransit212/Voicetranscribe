import numpy as np
import json
from typing import List, Dict, Any, Optional
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

class ConfidenceScorer:
    """Calculates multi-dimensional confidence scores for transcript candidates"""
    
    def __init__(self):
        self.score_weights = {
            'D': 0.28,  # Diarization consistency
            'A': 0.32,  # ASR alignment and confidence  
            'L': 0.18,  # Linguistic quality
            'R': 0.12,  # Cross-run agreement
            'O': 0.10   # Overlap handling
        }
    
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
        
        # Normalize scores to 0.00-1.00 range
        d_scores_norm = self._normalize_scores(d_scores)
        a_scores_norm = self._normalize_scores(a_scores)
        l_scores_norm = self._normalize_scores(l_scores)
        r_scores_norm = self._normalize_scores(r_scores)
        o_scores_norm = self._normalize_scores(o_scores)
        
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
        """Calculate Jaccard similarity of 3-grams with other candidates"""
        if not current_text or not other_texts:
            return 0.0
        
        # Generate 3-grams
        current_trigrams = set(self._get_ngrams(current_text, 3))
        
        similarities = []
        for other_text in other_texts:
            if not other_text:
                continue
            
            other_trigrams = set(self._get_ngrams(other_text, 3))
            
            # Jaccard similarity
            intersection = len(current_trigrams & other_trigrams)
            union = len(current_trigrams | other_trigrams)
            
            similarity = intersection / max(union, 1)
            similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _get_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text"""
        words = text.lower().split()
        ngrams = []
        
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _calculate_named_entity_consensus(self, current_text: str, other_texts: List[str]) -> float:
        """Calculate agreement on named entities"""
        # Simple heuristic: look for capitalized words
        current_entities = set(self._extract_potential_entities(current_text))
        
        agreements = []
        for other_text in other_texts:
            if not other_text:
                continue
            
            other_entities = set(self._extract_potential_entities(other_text))
            
            # Calculate F1 score
            if not current_entities and not other_entities:
                agreements.append(1.0)
            elif not current_entities or not other_entities:
                agreements.append(0.0)
            else:
                intersection = len(current_entities & other_entities)
                precision = intersection / len(current_entities)
                recall = intersection / len(other_entities)
                
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                agreements.append(f1)
        
        return float(np.mean(agreements)) if agreements else 0.0
    
    def _extract_potential_entities(self, text: str) -> List[str]:
        """Extract potential named entities (simple heuristic)"""
        words = text.split()
        entities = []
        
        for word in words:
            # Simple heuristic: capitalized words > 2 chars
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and len(clean_word) > 2 and clean_word[0].isupper():
                entities.append(clean_word.lower())
        
        return entities
    
    def _calculate_topic_coherence(self, candidate: Dict[str, Any]) -> float:
        """Calculate topic coherence across segments"""
        segments = candidate.get('aligned_segments', [])
        if len(segments) < 2:
            return 0.8
        
        # Extract text from segments
        segment_texts = [seg.get('text', '') for seg in segments]
        segment_texts = [text for text in segment_texts if text.strip()]
        
        if len(segment_texts) < 2:
            return 0.5
        
        # Calculate TF-IDF similarity between adjacent segments
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(segment_texts)
            
            # Calculate average cosine similarity between adjacent segments
            similarities = []
            for i in range(len(segment_texts) - 1):
                sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0][0]
                similarities.append(sim)
            
            return float(np.mean(similarities)) if similarities else 0.5
            
        except Exception:
            return 0.5  # Fallback score
    
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
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0.00-1.00 range using min-max scaling"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)  # All scores are the same
        
        # Min-max normalization
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        
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
