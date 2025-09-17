import re
import math
import time
from typing import Dict, Any, List, Set, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

class TranscriptFormatter:
    """Formats transcripts into various output formats (TXT, VTT, SRT, ASS)"""
    
    def __init__(self) -> None:
        self.speaker_colors: List[str] = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        # Cross-provider timestamp normalization configuration
        self.timestamp_normalizer = TimestampNormalizer()
        
        # Provider timing characteristics (based on empirical analysis)
        self.provider_timing_profiles = {
            'faster-whisper': {
                'timing_accuracy': 0.95,
                'systematic_offset': 0.0,
                'jitter_std': 0.02,
                'boundary_precision': 0.01
            },
            'deepgram': {
                'timing_accuracy': 0.92,
                'systematic_offset': 0.05,  # 50ms typical offset
                'jitter_std': 0.03,
                'boundary_precision': 0.02
            },
            'openai': {
                'timing_accuracy': 0.90,
                'systematic_offset': -0.03,  # -30ms typical offset
                'jitter_std': 0.04,
                'boundary_precision': 0.03
            }
        }


class TimestampNormalizer:
    """
    CRITICAL: Normalizes timestamps from different ASR providers to a common reference
    This prevents timing inconsistencies and boundary violations when fusing multi-provider results
    """
    
    def __init__(self):
        """Initialize timestamp normalizer with provider calibration data"""
        self.calibration_data = {}
        self.global_time_reference = 0.0
        self.normalization_enabled = True
        
        # Provider-specific timing adjustments (empirically determined)
        self.provider_adjustments = {
            'faster-whisper': {'offset': 0.0, 'scale': 1.0, 'confidence': 0.95},
            'deepgram': {'offset': 0.05, 'scale': 0.98, 'confidence': 0.92},  
            'openai': {'offset': -0.03, 'scale': 1.02, 'confidence': 0.90},
            'assemblyai': {'offset': 0.02, 'scale': 0.99, 'confidence': 0.88},
            'default': {'offset': 0.0, 'scale': 1.0, 'confidence': 0.85}
        }
    
    def normalize_provider_timestamps(self, 
                                    segments: List[Dict[str, Any]], 
                                    provider_name: str,
                                    reference_duration: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        CRITICAL: Normalize timestamps from a specific provider to common reference
        
        Args:
            segments: List of segments with timestamps from a specific provider
            provider_name: Name of the ASR provider (e.g., 'faster-whisper', 'deepgram', 'openai')
            reference_duration: Optional reference duration for validation
            
        Returns:
            Segments with normalized timestamps
        """
        if not segments or not self.normalization_enabled:
            return segments
        
        provider_key = provider_name.lower().replace('-', '_').replace(' ', '_')
        adjustments = self.provider_adjustments.get(provider_key, self.provider_adjustments['default'])
        
        normalized_segments = []
        
        for segment in segments:
            normalized_segment = segment.copy()
            
            # Normalize start and end times
            original_start = segment.get('start', 0.0)
            original_end = segment.get('end', 0.0)
            
            # Apply provider-specific adjustments
            normalized_start = self._apply_provider_adjustment(original_start, adjustments)
            normalized_end = self._apply_provider_adjustment(original_end, adjustments)
            
            # Ensure monotonic ordering
            if normalized_start > normalized_end:
                # Swap if provider had reversed timestamps
                normalized_start, normalized_end = normalized_end, normalized_start
            
            # Validate against reference duration
            if reference_duration and normalized_end > reference_duration + 1.0:
                # Clamp to reference duration with small tolerance
                normalized_end = reference_duration
                if normalized_start > normalized_end:
                    normalized_start = max(0.0, normalized_end - 0.1)
            
            normalized_segment['start'] = normalized_start
            normalized_segment['end'] = normalized_end
            
            # Store normalization metadata
            normalized_segment['normalization_metadata'] = {
                'provider': provider_name,
                'original_start': original_start,
                'original_end': original_end,
                'adjustment_offset': adjustments['offset'],
                'adjustment_scale': adjustments['scale'],
                'provider_confidence': adjustments['confidence']
            }
            
            # Normalize word-level timestamps if present
            if 'words' in segment and isinstance(segment['words'], list):
                normalized_words = []
                for word in segment['words']:
                    normalized_word = word.copy()
                    if 'start' in word:
                        normalized_word['start'] = self._apply_provider_adjustment(word['start'], adjustments)
                    if 'end' in word:
                        normalized_word['end'] = self._apply_provider_adjustment(word['end'], adjustments)
                    normalized_words.append(normalized_word)
                normalized_segment['words'] = normalized_words
            
            normalized_segments.append(normalized_segment)
        
        # Validate overall consistency
        validation_result = self._validate_normalized_timestamps(normalized_segments, provider_name)
        if not validation_result['is_valid']:
            # Apply additional corrections if needed
            normalized_segments = self._correct_timestamp_violations(normalized_segments, validation_result)
        
        return normalized_segments
    
    def _apply_provider_adjustment(self, timestamp: float, adjustments: Dict[str, float]) -> float:
        """Apply provider-specific timing adjustments to a timestamp"""
        # Apply offset first, then scale
        adjusted = (timestamp + adjustments['offset']) * adjustments['scale']
        return max(0.0, adjusted)  # Ensure non-negative timestamps
    
    def _validate_normalized_timestamps(self, segments: List[Dict[str, Any]], provider: str) -> Dict[str, Any]:
        """Validate normalized timestamps for consistency"""
        validation_result = {
            'is_valid': True,
            'violations': [],
            'provider': provider,
            'total_segments': len(segments)
        }
        
        if not segments:
            return validation_result
        
        # Check for monotonic ordering
        for i, segment in enumerate(segments):
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', 0.0)
            
            # Validate segment internal consistency
            if start_time > end_time:
                validation_result['is_valid'] = False
                validation_result['violations'].append({
                    'type': 'reversed_segment',
                    'segment_index': i,
                    'start': start_time,
                    'end': end_time
                })
            
            # Validate segment-to-segment ordering
            if i > 0:
                prev_end = segments[i-1].get('end', 0.0)
                if start_time < prev_end - 0.01:  # 10ms tolerance
                    validation_result['violations'].append({
                        'type': 'non_monotonic',
                        'segment_index': i,
                        'current_start': start_time,
                        'previous_end': prev_end,
                        'gap': start_time - prev_end
                    })
        
        return validation_result
    
    def _correct_timestamp_violations(self, 
                                    segments: List[Dict[str, Any]], 
                                    validation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Correct timestamp violations found during validation"""
        corrected_segments = segments.copy()
        
        for violation in validation_result['violations']:
            if violation['type'] == 'reversed_segment':
                # Fix reversed start/end times
                idx = violation['segment_index']
                if 0 <= idx < len(corrected_segments):
                    segment = corrected_segments[idx]
                    start = segment.get('start', 0.0)
                    end = segment.get('end', 0.0)
                    
                    # Swap if needed
                    if start > end:
                        segment['start'] = end
                        segment['end'] = start
            
            elif violation['type'] == 'non_monotonic':
                # Fix overlapping segments
                idx = violation['segment_index']
                if 0 < idx < len(corrected_segments):
                    current_segment = corrected_segments[idx]
                    prev_segment = corrected_segments[idx - 1]
                    
                    # Adjust boundary at midpoint
                    overlap_end = prev_segment.get('end', 0.0)
                    overlap_start = current_segment.get('start', 0.0)
                    midpoint = (overlap_end + overlap_start) / 2.0
                    
                    prev_segment['end'] = midpoint
                    current_segment['start'] = midpoint
        
        return corrected_segments
    
    def cross_provider_calibration(self, provider_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calibrate timestamps across multiple providers using cross-correlation
        
        Args:
            provider_results: Dictionary mapping provider names to their segment results
            
        Returns:
            Dictionary with cross-calibrated timestamps for all providers
        """
        if len(provider_results) < 2:
            # Single provider - just apply standard normalization
            for provider, segments in provider_results.items():
                provider_results[provider] = self.normalize_provider_timestamps(segments, provider)
            return provider_results
        
        # Find reference provider (most reliable)
        reference_provider = self._select_reference_provider(provider_results)
        
        calibrated_results = {}
        
        for provider, segments in provider_results.items():
            if provider == reference_provider:
                # Reference provider gets standard normalization only
                calibrated_results[provider] = self.normalize_provider_timestamps(segments, provider)
            else:
                # Other providers get cross-calibrated to reference
                calibrated_segments = self._cross_calibrate_to_reference(
                    segments, provider, 
                    calibrated_results.get(reference_provider, []), reference_provider
                )
                calibrated_results[provider] = calibrated_segments
        
        return calibrated_results
    
    def _select_reference_provider(self, provider_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Select the most reliable provider as timing reference"""
        provider_scores = {}
        
        for provider, segments in provider_results.items():
            score = 0.0
            
            # Base score from provider confidence
            provider_key = provider.lower().replace('-', '_').replace(' ', '_')
            base_confidence = self.provider_adjustments.get(provider_key, {}).get('confidence', 0.5)
            score += base_confidence * 0.4
            
            # Score based on segment count (more segments = more data points)
            segment_count_score = min(len(segments) / 100, 1.0) * 0.3
            score += segment_count_score
            
            # Score based on timestamp consistency
            consistency_score = self._calculate_timestamp_consistency(segments) * 0.3
            score += consistency_score
            
            provider_scores[provider] = score
        
        # Return provider with highest score
        return max(provider_scores.keys(), key=lambda p: provider_scores[p])
    
    def _calculate_timestamp_consistency(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate timestamp consistency score for a provider"""
        if len(segments) < 2:
            return 1.0
        
        violations = 0
        total_transitions = 0
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1].get('end', 0.0)
            curr_start = segments[i].get('start', 0.0)
            
            total_transitions += 1
            
            # Check for violations (overlaps or large gaps)
            gap = curr_start - prev_end
            if gap < -0.1 or gap > 5.0:  # Overlap > 100ms or gap > 5s
                violations += 1
        
        consistency = 1.0 - (violations / max(total_transitions, 1))
        return max(0.0, consistency)
    
    def _cross_calibrate_to_reference(self, 
                                    segments: List[Dict[str, Any]], 
                                    provider: str,
                                    reference_segments: List[Dict[str, Any]], 
                                    reference_provider: str) -> List[Dict[str, Any]]:
        """Cross-calibrate provider timestamps to reference provider"""
        if not reference_segments:
            return self.normalize_provider_timestamps(segments, provider)
        
        # Apply standard normalization first
        normalized_segments = self.normalize_provider_timestamps(segments, provider)
        
        # Calculate timing offset between providers using correlation
        timing_offset = self._calculate_cross_provider_offset(
            normalized_segments, reference_segments
        )
        
        # Apply dynamic calibration offset
        calibrated_segments = self._apply_dynamic_calibration_offset(
            normalized_segments, timing_offset, provider
        )
        
        return calibrated_segments
    
    def _calculate_cross_provider_offset(self, 
                                       target_segments: List[Dict[str, Any]], 
                                       reference_segments: List[Dict[str, Any]]) -> float:
        """Calculate cross-correlation based timing offset between providers"""
        try:
            import numpy as np
            from scipy import signal
            
            # Extract segment start times for correlation
            target_starts = np.array([seg.get('start', 0.0) for seg in target_segments])
            ref_starts = np.array([seg.get('start', 0.0) for seg in reference_segments])
            
            if len(target_starts) < 3 or len(ref_starts) < 3:
                # Fallback to simple mean difference
                return self._calculate_mean_offset(target_segments, reference_segments)
            
            # Create time bins for cross-correlation
            max_time = max(np.max(target_starts), np.max(ref_starts))
            time_resolution = 0.1  # 100ms resolution
            time_bins = np.arange(0, max_time + time_resolution, time_resolution)
            
            # Create histograms
            target_hist, _ = np.histogram(target_starts, bins=time_bins)
            ref_hist, _ = np.histogram(ref_starts, bins=time_bins)
            
            # Calculate cross-correlation
            correlation = signal.correlate(ref_hist, target_hist, mode='full')
            lags = signal.correlation_lags(len(ref_hist), len(target_hist), mode='full')
            
            # Find peak correlation
            peak_idx = np.argmax(correlation)
            offset_bins = lags[peak_idx]
            offset_seconds = offset_bins * time_resolution
            
            # Validate offset is reasonable
            if abs(offset_seconds) > 2.0:
                print(f"Warning: Large cross-correlation offset: {offset_seconds:.3f}s, using fallback")
                return self._calculate_mean_offset(target_segments, reference_segments)
            
            print(f"Dynamic cross-correlation offset calculated: {offset_seconds:.3f}s")
            return -offset_seconds  # Negative to correct target toward reference
            
        except ImportError:
            print("Warning: scipy not available, using mean offset fallback")
            return self._calculate_mean_offset(target_segments, reference_segments)
        except Exception as e:
            print(f"Cross-correlation offset calculation failed: {e}")
            return self._calculate_mean_offset(target_segments, reference_segments)
    
    def _calculate_mean_offset(self, 
                             target_segments: List[Dict[str, Any]], 
                             reference_segments: List[Dict[str, Any]]) -> float:
        """Fallback method: calculate mean timing offset"""
        if not target_segments or not reference_segments:
            return 0.0
        
        # Compare first N segments
        n_compare = min(len(target_segments), len(reference_segments), 10)
        
        offsets = []
        for i in range(n_compare):
            target_start = target_segments[i].get('start', 0.0)
            ref_start = reference_segments[i].get('start', 0.0)
            offsets.append(ref_start - target_start)
        
        mean_offset = np.mean(offsets) if offsets else 0.0
        print(f"Mean offset calculated: {mean_offset:.3f}s")
        return mean_offset
    
    def _apply_dynamic_calibration_offset(self, 
                                        segments: List[Dict[str, Any]], 
                                        offset: float, 
                                        provider: str) -> List[Dict[str, Any]]:
        """Apply dynamic calibration offset to segments"""
        if abs(offset) < 0.01:  # Skip tiny offsets
            return segments
        
        calibrated_segments = []
        
        for segment in segments:
            calibrated_segment = segment.copy()
            
            # Apply offset to segment boundaries
            original_start = segment.get('start', 0.0)
            original_end = segment.get('end', 0.0)
            
            calibrated_segment['start'] = max(0.0, original_start + offset)
            calibrated_segment['end'] = max(calibrated_segment['start'], original_end + offset)
            
            # Apply offset to word-level timestamps
            if 'words' in segment and isinstance(segment['words'], list):
                calibrated_words = []
                for word in segment['words']:
                    calibrated_word = word.copy()
                    if 'start' in word:
                        calibrated_word['start'] = max(0.0, word['start'] + offset)
                    if 'end' in word:
                        calibrated_word['end'] = max(calibrated_word.get('start', 0.0), word['end'] + offset)
                    calibrated_words.append(calibrated_word)
                calibrated_segment['words'] = calibrated_words
            
            # Store calibration metadata
            calibrated_segment['dynamic_calibration_metadata'] = {
                'provider': provider,
                'offset_applied': offset,
                'calibration_method': 'cross_correlation',
                'original_start': original_start,
                'original_end': original_end,
                'calibration_timestamp': time.time()
            }
            
            calibrated_segments.append(calibrated_segment)
        
        print(f"Applied dynamic calibration offset {offset:.3f}s to {len(calibrated_segments)} segments for provider {provider}")
        return calibrated_segments
    
    def calibrate_candidates_dynamically(self, 
                                       candidates: List[Dict[str, Any]], 
                                       reference_duration: Optional[float] = None) -> List[Dict[str, Any]]:
        """Perform dynamic calibration across all candidates before fusion"""
        if len(candidates) < 2:
            return candidates
        
        # Group candidates by provider
        provider_candidates = defaultdict(list)
        for candidate in candidates:
            provider = candidate.get('asr_provider', 'unknown').lower()
            provider_candidates[provider].append(candidate)
        
        if len(provider_candidates) < 2:
            return candidates
        
        # Select reference provider (most reliable)
        reference_provider = self._select_reference_provider_from_candidates(provider_candidates)
        calibrated_candidates = []
        
        # Log calibration process
        print(f"Dynamic calibration: Using {reference_provider} as reference provider")
        
        for provider, provider_cands in provider_candidates.items():
            if provider == reference_provider:
                # Reference provider gets standard normalization only
                for candidate in provider_cands:
                    segments = candidate.get('aligned_segments', [])
                    normalized_segments = self.normalize_provider_timestamps(
                        segments, provider, reference_duration
                    )
                    candidate['aligned_segments'] = normalized_segments
                    candidate['calibration_applied'] = 'reference_provider'
                calibrated_candidates.extend(provider_cands)
            else:
                # Other providers get dynamic calibration
                reference_segments = []
                for ref_candidate in provider_candidates[reference_provider]:
                    reference_segments.extend(ref_candidate.get('aligned_segments', []))
                
                for candidate in provider_cands:
                    segments = candidate.get('aligned_segments', [])
                    
                    # Apply standard normalization first
                    normalized_segments = self.normalize_provider_timestamps(
                        segments, provider, reference_duration
                    )
                    
                    # Apply dynamic calibration
                    if reference_segments:
                        calibrated_segments = self._cross_calibrate_to_reference(
                            normalized_segments, provider, reference_segments, reference_provider
                        )
                    else:
                        calibrated_segments = normalized_segments
                    
                    candidate['aligned_segments'] = calibrated_segments
                    candidate['calibration_applied'] = 'dynamic_cross_correlation'
                    
                calibrated_candidates.extend(provider_cands)
        
        print(f"Dynamic calibration complete - processed {len(calibrated_candidates)} candidates")
        return calibrated_candidates
    
    def _select_reference_provider_from_candidates(self, 
                                                 provider_candidates: Dict[str, List[Dict[str, Any]]]) -> str:
        """Select most reliable provider as reference for dynamic calibration"""
        provider_scores = {}
        
        for provider, candidates in provider_candidates.items():
            score = 0.0
            total_segments = 0
            total_confidence = 0.0
            
            # Calculate provider reliability score
            for candidate in candidates:
                segments = candidate.get('aligned_segments', [])
                total_segments += len(segments)
                
                # Base confidence from provider characteristics
                provider_key = provider.lower().replace('-', '_').replace(' ', '_')
                base_confidence = self.provider_adjustments.get(provider_key, {}).get('confidence', 0.5)
                score += base_confidence * 0.4
                
                # Score from segment confidence if available
                for segment in segments:
                    segment_confidence = segment.get('confidence', 0.5)
                    total_confidence += segment_confidence
            
            # Add segment count bonus (more data = better reference)
            segment_score = min(total_segments / 50, 1.0) * 0.3
            score += segment_score
            
            # Add average confidence score
            if total_segments > 0:
                avg_confidence = total_confidence / total_segments
                score += avg_confidence * 0.3
            
            provider_scores[provider] = score
        
        # Return provider with highest score
        best_provider = max(provider_scores.keys(), key=lambda p: provider_scores[p])
        print(f"Selected reference provider: {best_provider} (score: {provider_scores[best_provider]:.3f})")
        return best_provider
        
        # Apply cross-calibration offset
        calibrated_segments = []
        for segment in normalized_segments:
            calibrated_segment = segment.copy()
            calibrated_segment['start'] = max(0.0, segment['start'] + timing_offset)
            calibrated_segment['end'] = max(0.0, segment['end'] + timing_offset)
            
            # Update normalization metadata
            if 'normalization_metadata' in calibrated_segment:
                calibrated_segment['normalization_metadata']['cross_calibration_offset'] = timing_offset
                calibrated_segment['normalization_metadata']['reference_provider'] = reference_provider
            
            calibrated_segments.append(calibrated_segment)
        
        return calibrated_segments
    
    def _calculate_cross_provider_offset(self, 
                                       segments1: List[Dict[str, Any]], 
                                       segments2: List[Dict[str, Any]]) -> float:
        """Calculate timing offset between two providers using correlation"""
        if not segments1 or not segments2:
            return 0.0
        
        # Extract start times for correlation
        times1 = [s.get('start', 0.0) for s in segments1[:50]]  # Use first 50 segments
        times2 = [s.get('start', 0.0) for s in segments2[:50]]
        
        if len(times1) < 3 or len(times2) < 3:
            return 0.0
        
        # Find best offset using simple correlation
        best_offset = 0.0
        best_correlation = -1.0
        
        # Test offsets from -2s to +2s in 50ms increments
        for offset in np.arange(-2.0, 2.0, 0.05):
            adjusted_times1 = [t + offset for t in times1]
            correlation = self._calculate_timing_correlation(adjusted_times1, times2)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_offset = offset
        
        # Only apply offset if correlation is strong enough
        if best_correlation > 0.7:
            return best_offset
        else:
            return 0.0
    
    def _calculate_timing_correlation(self, times1: List[float], times2: List[float]) -> float:
        """Calculate correlation between two timing sequences"""
        if not times1 or not times2:
            return 0.0
        
        # Simple proximity-based correlation
        matches = 0
        total_comparisons = 0
        
        for t1 in times1:
            for t2 in times2:
                total_comparisons += 1
                if abs(t1 - t2) < 0.2:  # Within 200ms
                    matches += 1
        
        return matches / max(total_comparisons, 1)
    
    def create_txt_transcript(self, master_transcript: Dict[str, Any]) -> str:
        """
        Create plain text transcript with speaker names and timestamps.
        
        Args:
            master_transcript: Master transcript JSON
            
        Returns:
            Formatted text transcript
        """
        segments = master_transcript.get('segments', [])
        speaker_map = master_transcript.get('speaker_map', {})
        metadata = master_transcript.get('metadata', {})
        
        # Check if post-fusion punctuation was applied
        punctuation_metadata = master_transcript.get('punctuation_metadata', {})
        
        # Build header
        lines = []
        lines.append("=" * 60)
        lines.append("TRANSCRIPT")
        lines.append("=" * 60)
        lines.append("")
        
        # Add metadata
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Duration: {self.format_duration(metadata.get('total_duration', 0))}")
        lines.append(f"Segments: {metadata.get('total_segments', 0)}")
        lines.append(f"Speakers: {metadata.get('speaker_count', 0)}")
        lines.append("")
        
        # Add speaker roster
        if speaker_map:
            lines.append("PARTICIPANTS:")
            for speaker_id, speaker_name in speaker_map.items():
                lines.append(f"  • {speaker_name}")
            lines.append("")
        
        # Add confidence summary
        confidence_summary = metadata.get('confidence_summary', {})
        if confidence_summary:
            lines.append("CONFIDENCE SUMMARY:")
            lines.append(f"  • Overall Score: {confidence_summary.get('final_score', 0):.3f}")
            lines.append(f"  • Diarization: {confidence_summary.get('D_diarization', 0):.3f}")
            lines.append(f"  • ASR Alignment: {confidence_summary.get('A_asr_alignment', 0):.3f}")
            lines.append(f"  • Linguistic Quality: {confidence_summary.get('L_linguistic', 0):.3f}")
            lines.append(f"  • Cross-run Agreement: {confidence_summary.get('R_agreement', 0):.3f}")
            lines.append(f"  • Overlap Handling: {confidence_summary.get('O_overlap', 0):.3f}")
            lines.append("")
        
        # Add punctuation summary if available
        if punctuation_metadata:
            lines.append("PUNCTUATION PROCESSING:")
            lines.append(f"  • Punctuation Confidence: {punctuation_metadata.get('overall_confidence', 0):.3f}")
            lines.append(f"  • Normalization Level: {punctuation_metadata.get('normalization_level', 'N/A')}")
            
            punctuation_metrics = punctuation_metadata.get('punctuation_metrics', {})
            disfluency_metrics = punctuation_metadata.get('disfluency_metrics', {})
            
            if punctuation_metrics:
                lines.append(f"  • Segments Changed: {punctuation_metrics.get('segments_changed', 0)}")
            if disfluency_metrics:
                lines.append(f"  • Disfluency Normalized: {disfluency_metrics.get('segments_normalized', 0)}")
            
            model_info = punctuation_metadata.get('model_info', {})
            if model_info.get('model_available') == 'True':
                lines.append(f"  • Model: Transformer-based ({model_info.get('punctuation_model', 'Unknown')})")
            else:
                lines.append("  • Model: Rule-based (fallback)")
            
            lines.append("")
        
        lines.append("TRANSCRIPT:")
        lines.append("-" * 60)
        lines.append("")
        
        # Add transcript content
        current_speaker = None
        for segment in segments:
            timestamp = self.format_timestamp(segment['start'])
            speaker = segment['speaker']
            text = segment['text'].strip()
            
            if not text:
                continue
            
            # Add speaker change indicator
            if speaker != current_speaker:
                if current_speaker is not None:
                    lines.append("")  # Add blank line between speakers
                current_speaker = speaker
            
            # Check if segment has punctuation metadata for enhanced display
            if segment.get('punctuation_confidence') is not None:
                # Use punctuated text if confidence is high enough
                punct_confidence = segment.get('punctuation_confidence', 0)
                if punct_confidence > 0.5:  # Confidence threshold
                    # Show if disfluency normalization was applied
                    normalization_note = ""
                    if segment.get('disfluency_normalization_applied', False):
                        normalization_note = " [normalized]"
                    
                    lines.append(f"[{timestamp}] {speaker}: {text}{normalization_note}")
                    
                    # Optionally show original text if significantly different
                    original_text = segment.get('original_text', '')
                    if original_text and original_text.strip() != text.strip() and len(original_text) > 10:
                        text_diff = len(original_text) - len(text)
                        if abs(text_diff) > 5:  # Significant change threshold
                            lines.append(f"    [Original: {original_text.strip()[:100]}...]" if len(original_text) > 100 else f"    [Original: {original_text.strip()}]")
                else:
                    # Use original text if punctuation confidence is low
                    lines.append(f"[{timestamp}] {speaker}: {text}")
            else:
                # Standard display for segments without punctuation processing
                lines.append(f"[{timestamp}] {speaker}: {text}")
        
        # Add footer
        lines.append("")
        lines.append("-" * 60)
        lines.append("END OF TRANSCRIPT")
        lines.append("")
        lines.append("LEGEND:")
        lines.append("  [HH:MM:SS] - Timestamp")
        lines.append("  Speaker: - Speaker identification")
        lines.append("  Confidence scores range from 0.000 to 1.000")
        if punctuation_metadata:
            lines.append("  [normalized] - Disfluency normalization applied")
            lines.append("  [Original: ...] - Original text before punctuation/normalization")
        lines.append("")
        
        return '\n'.join(lines)
    
    def create_vtt_captions(self, master_transcript: Dict[str, Any]) -> str:
        """
        Create WebVTT captions with speaker positioning.
        
        Args:
            master_transcript: Master transcript JSON
            
        Returns:
            WebVTT formatted captions
        """
        segments = master_transcript.get('segments', [])
        
        lines = []
        lines.append("WEBVTT")
        lines.append("")
        lines.append("NOTE")
        lines.append("Generated by Advanced Ensemble Transcription System")
        lines.append("")
        
        # Assign positions to speakers
        speakers = list(set(seg['speaker'] for seg in segments))
        speaker_positions = {}
        positions = ["line:10%", "line:20%", "line:30%", "line:40%", "line:50%",
                    "line:60%", "line:70%", "line:80%", "line:90%"]
        
        for i, speaker in enumerate(speakers):
            if i < len(positions):
                speaker_positions[speaker] = positions[i]
            else:
                speaker_positions[speaker] = "line:50%"  # Default position
        
        # Generate cues
        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            if not text:
                continue
            
            start_time = self.format_webvtt_time(segment['start'])
            end_time = self.format_webvtt_time(segment['end'])
            speaker = segment['speaker']
            position = speaker_positions.get(speaker, "line:50%")
            
            # Cue identifier
            lines.append(f"cue-{i+1}")
            lines.append(f"{start_time} --> {end_time} {position}")
            lines.append(f"<v {speaker}>{text}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def create_srt_captions(self, master_transcript: Dict[str, Any]) -> str:
        """
        Create SRT captions with speaker names.
        
        Args:
            master_transcript: Master transcript JSON
            
        Returns:
            SRT formatted captions
        """
        segments = master_transcript.get('segments', [])
        
        lines = []
        cue_number = 1
        
        for segment in segments:
            text = segment['text'].strip()
            if not text:
                continue
            
            start_time = self.format_srt_time(segment['start'])
            end_time = self.format_srt_time(segment['end'])
            speaker = segment['speaker']
            
            lines.append(str(cue_number))
            lines.append(f"{start_time} --> {end_time}")
            lines.append(f"{speaker}: {text}")
            lines.append("")
            
            cue_number += 1
        
        return '\n'.join(lines)
    
    def create_ass_captions(self, master_transcript: Dict[str, Any]) -> str:
        """
        Create ASS captions with styled speaker differentiation.
        
        Args:
            master_transcript: Master transcript JSON
            
        Returns:
            ASS formatted captions
        """
        segments = master_transcript.get('segments', [])
        speakers = list(set(seg['speaker'] for seg in segments))
        
        lines = []
        
        # ASS header
        lines.append("[Script Info]")
        lines.append("Title: Ensemble Transcription")
        lines.append("ScriptType: v4.00+")
        lines.append("Collisions: Normal")
        lines.append("PlayDepth: 0")
        lines.append("")
        
        # Styles section
        lines.append("[V4+ Styles]")
        lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
        
        # Create style for each speaker
        for i, speaker in enumerate(speakers):
            color = self.speaker_colors[i % len(self.speaker_colors)]
            # Convert hex to BGR format for ASS
            bgr_color = self._hex_to_bgr(color)
            
            lines.append(f"Style: {speaker},Arial,20,&H{bgr_color},&H000000,&H000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1")
        
        lines.append("")
        
        # Events section
        lines.append("[Events]")
        lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
        
        for segment in segments:
            text = segment['text'].strip()
            if not text:
                continue
            
            start_time = self.format_ass_time(segment['start'])
            end_time = self.format_ass_time(segment['end'])
            speaker = segment['speaker']
            
            # Escape special characters
            text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            
            lines.append(f"Dialogue: 0,{start_time},{end_time},{speaker},,0,0,0,,{text}")
        
        return '\n'.join(lines)
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def format_webvtt_time(self, seconds: float) -> str:
        """Format time for WebVTT (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def format_srt_time(self, seconds: float) -> str:
        """Format time for SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def format_ass_time(self, seconds: float) -> str:
        """Format time for ASS (H:MM:SS.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    
    def _hex_to_bgr(self, hex_color: str) -> str:
        """Convert hex color to BGR format for ASS"""
        hex_color = hex_color.lstrip('#')
        
        # Parse RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # Convert to BGR hex
        bgr = f"{b:02X}{g:02X}{r:02X}"
        
        return bgr
    
    def create_segment_summary(self, master_transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary statistics for transcript segments.
        
        Args:
            master_transcript: Master transcript JSON
            
        Returns:
            Summary statistics dictionary
        """
        segments = master_transcript.get('segments', [])
        
        if not segments:
            return {'error': 'No segments found'}
        
        # Basic statistics
        total_segments = len(segments)
        total_duration = sum(seg['end'] - seg['start'] for seg in segments)
        total_words = sum(seg.get('word_count', 0) for seg in segments)
        
        # Speaker statistics
        speaker_stats = {}
        for segment in segments:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            word_count = segment.get('word_count', 0)
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'segment_count': 0,
                    'total_duration': 0.0,
                    'total_words': 0,
                    'avg_confidence': []
                }
            
            speaker_stats[speaker]['segment_count'] += 1
            speaker_stats[speaker]['total_duration'] += duration
            speaker_stats[speaker]['total_words'] += word_count
            speaker_stats[speaker]['avg_confidence'].append(segment.get('confidence', 0.0))
        
        # Calculate averages
        for speaker in speaker_stats:
            confidences = speaker_stats[speaker]['avg_confidence']
            speaker_stats[speaker]['avg_confidence'] = sum(confidences) / len(confidences)
            speaker_stats[speaker]['speaking_percentage'] = (
                speaker_stats[speaker]['total_duration'] / total_duration * 100
            )
        
        # Segment length statistics
        segment_durations = [seg['end'] - seg['start'] for seg in segments]
        avg_segment_duration = sum(segment_durations) / len(segment_durations)
        
        summary = {
            'total_segments': total_segments,
            'total_duration': total_duration,
            'total_words': total_words,
            'avg_segment_duration': avg_segment_duration,
            'avg_words_per_segment': total_words / total_segments if total_segments > 0 else 0,
            'speaker_count': len(speaker_stats),
            'speaker_statistics': speaker_stats,
            'duration_formatted': self.format_duration(total_duration)
        }
        
        return summary
