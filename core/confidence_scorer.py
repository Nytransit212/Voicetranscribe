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
        """Calculate language model plausibility score"""
        if not text:
            return 0.0
        
        # Simple heuristic based on text characteristics
        words = text.split()
        if len(words) < 5:
            return 0.5
        
        # Check for reasonable word length distribution
        word_lengths = [len(w) for w in words]
        avg_length = np.mean(word_lengths)
        
        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Heuristic scoring
        length_score = 1.0 if 3 <= avg_length <= 8 else 0.5
        sentence_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.7
        
        return (length_score + sentence_score) / 2
    
    def _calculate_punctuation_score(self, text: str) -> float:
        """Calculate punctuation and casing plausibility"""
        if not text:
            return 0.0
        
        # Check sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        proper_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence[0].isupper():
                proper_sentences += 1
        
        sentence_score = proper_sentences / max(len([s for s in sentences if s.strip()]), 1)
        
        # Check capitalization of potential names
        words = text.split()
        capitalized_words = sum(1 for w in words if w and w[0].isupper())
        
        # Reasonable capitalization ratio (10-30% for meeting transcripts)
        cap_ratio = capitalized_words / max(len(words), 1)
        cap_score = 1.0 if 0.1 <= cap_ratio <= 0.3 else 0.6
        
        return (sentence_score + cap_score) / 2
    
    def _calculate_disfluency_score(self, text: str) -> float:
        """Calculate appropriate handling of disfluencies"""
        if not text:
            return 0.0
        
        # Count common disfluencies
        fillers = ['um', 'uh', 'er', 'ah', 'you know', 'like', 'so']
        filler_count = 0
        
        text_lower = text.lower()
        for filler in fillers:
            filler_count += text_lower.count(filler)
        
        words = text.split()
        filler_ratio = filler_count / max(len(words), 1)
        
        # Moderate filler ratio is natural for speech (2-8%)
        if 0.02 <= filler_ratio <= 0.08:
            return 0.9
        elif filler_ratio < 0.02:
            return 0.7  # Maybe over-cleaned
        else:
            return 0.6  # Too many fillers kept
    
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
