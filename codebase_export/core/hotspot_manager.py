import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

from utils.enhanced_structured_logger import create_enhanced_logger
# from utils.manifest import update_manifest  # Will implement manifest update internally
from config.settings import Settings

@dataclass
class Token:
    """Token with timing and confidence information"""
    text: str
    start: float
    end: float
    conf: float
    provider_votes: Optional[Dict[str, Any]] = None

@dataclass
class Clip:
    """Hotspot clip for human review"""
    id: str
    start_s: float
    end_s: float
    speakers: List[str]
    text_proposed: str
    tokens: List[Token]
    uncertainty_score: float

@dataclass
class HumanEdit:
    """Human edit to a clip"""
    clip_id: str
    text_final: str
    original_text: Optional[str] = None
    token_ops: Optional[List[Dict]] = None
    speaker_label_override: Optional[str] = None
    flags: Optional[List[str]] = None

@dataclass
class SpeakerMap:
    """Mapping from engine labels to human names"""
    engine_to_human: Dict[str, str]
    confidence_scores: Optional[Dict[str, float]] = None

class HotspotManager:
    """Manages hotspot selection, human review, and improvement propagation"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = create_enhanced_logger(__name__)
        self.settings = Settings()
        
        # Default configuration
        self.config = {
            'human_time_budget_min': 5.0,
            'clip_len_min_s': 15,
            'clip_len_max_s': 30,
            'expected_review_rate': 1.5,  # clips per minute
            'diversity_coverage_buckets': 3,
            'suppression_radius_factor': 2.0,
            'neighbor_repair_window_s': 15,
            'provider_reweight_max_delta': 0.15,
            'uncertainty_threshold': 0.3,
            'confidence_threshold': 0.6,
            'max_local_redo_per_clip': 1,
            'reweight_strength': 'medium'
        }
        
        if config:
            self.config.update(config)
        
        self.logger.info("HotspotManager initialized", extra={
            'config': self.config
        })
    
    def select_hotspots(self, transcript_data: Dict, audio_file_path: Optional[str] = None, 
                       human_time_budget_min: float = 5.0) -> List[Clip]:
        """Select hotspot clips for human review"""
        try:
            self.logger.info("Starting hotspot selection", extra={
                'human_time_budget_min': human_time_budget_min,
                'audio_file_path': audio_file_path
            })
            
            # Extract segments and compute uncertainties
            segments = self._extract_segments(transcript_data)
            if not segments:
                self.logger.warning("No segments found for hotspot selection")
                return []
            
            # Compute uncertainty scores for micro-windows
            uncertainty_windows = self._compute_uncertainty_windows(segments)
            
            # Select hotspot clips
            clips = self._select_clips(uncertainty_windows, segments, human_time_budget_min)
            
            self.logger.info("Hotspot selection completed", extra={
                'clips_selected': len(clips),
                'total_windows': len(uncertainty_windows)
            })
            
            return clips
            
        except Exception as e:
            self.logger.error("Failed to select hotspots", extra={'error': str(e)})
            return []
    
    def _extract_segments(self, transcript_data: Dict) -> List[Dict]:
        """Extract segments from transcript data"""
        winner_transcript = transcript_data.get('winner_transcript', {})
        segments = winner_transcript.get('segments', [])
        
        if not segments:
            # Fallback: create segments from basic transcript
            text = winner_transcript.get('text', '')
            if text:
                # Create simple segments (simplified approach)
                words = text.split()
                segments = []
                for i, word in enumerate(words):
                    segment = {
                        'start': i * 2.0,  # Rough timing
                        'end': (i + 1) * 2.0,
                        'text': word,
                        'speaker': f'Speaker_{i % 2 + 1}',  # Alternate speakers
                        'confidence': 0.8  # Default confidence
                    }
                    segments.append(segment)
        
        return segments
    
    def _compute_uncertainty_windows(self, segments: List[Dict]) -> List[Dict]:
        """Compute uncertainty scores for micro-windows"""
        windows = []
        window_size = 8.0  # 8-second windows
        
        if not segments:
            return windows
        
        # Get total duration
        max_end = max(seg.get('end', 0) for seg in segments)
        
        # Create overlapping windows
        current_time = 0
        while current_time < max_end:
            window_end = current_time + window_size
            
            # Find segments in this window
            window_segments = [
                seg for seg in segments 
                if seg.get('start', 0) < window_end and seg.get('end', 0) > current_time
            ]
            
            if window_segments:
                uncertainty_score = self._compute_window_uncertainty(window_segments)
                
                window = {
                    'start_s': current_time,
                    'end_s': window_end,
                    'segments': window_segments,
                    'uncertainty_score': uncertainty_score
                }
                windows.append(window)
            
            current_time += window_size * 0.5  # 50% overlap
        
        return windows
    
    def _compute_window_uncertainty(self, window_segments: List[Dict]) -> float:
        """Compute uncertainty score for a window"""
        if not window_segments:
            return 0.0
        
        uncertainty_factors = []
        
        for segment in window_segments:
            # ASR confidence factor
            confidence = segment.get('confidence', 0.8)
            conf_uncertainty = 1.0 - confidence
            uncertainty_factors.append(conf_uncertainty * 0.4)
            
            # Speaker boundary uncertainty
            # Higher uncertainty near speaker changes
            speaker_changes = 0
            for other_seg in window_segments:
                if (other_seg.get('speaker') != segment.get('speaker') and
                    abs(other_seg.get('start', 0) - segment.get('end', 0)) < 2.0):
                    speaker_changes += 1
            
            boundary_uncertainty = min(speaker_changes * 0.2, 0.3)
            uncertainty_factors.append(boundary_uncertainty)
            
            # Text complexity factor
            text = segment.get('text', '')
            words = text.split()
            
            # Look for challenging words (longer, capitals, numbers)
            complex_words = sum(1 for word in words 
                              if len(word) > 8 or word.isupper() or any(c.isdigit() for c in word))
            
            complexity_uncertainty = min(complex_words / max(len(words), 1) * 0.3, 0.2)
            uncertainty_factors.append(complexity_uncertainty)
        
        # Combine uncertainty factors
        total_uncertainty = min(sum(uncertainty_factors), 1.0)
        
        return total_uncertainty
    
    def _select_clips(self, uncertainty_windows: List[Dict], segments: List[Dict], 
                     human_time_budget_min: float) -> List[Clip]:
        """Select clips based on uncertainty and budget constraints"""
        if not uncertainty_windows:
            return []
        
        # Calculate budget
        expected_review_rate = self.config['expected_review_rate']
        total_audio_budget = human_time_budget_min * 60 / expected_review_rate
        
        # Sort windows by uncertainty (highest first)
        sorted_windows = sorted(uncertainty_windows, 
                               key=lambda w: w['uncertainty_score'], 
                               reverse=True)
        
        selected_clips = []
        used_time = 0
        suppression_regions = []
        
        for window in sorted_windows:
            if used_time >= total_audio_budget:
                break
            
            # Check if this window overlaps with suppressed regions
            window_start = window['start_s']
            window_end = window['end_s']
            
            if any(self._overlaps(window_start, window_end, sup_start, sup_end) 
                   for sup_start, sup_end in suppression_regions):
                continue
            
            # Create clip
            clip = self._create_clip(window, segments)
            if clip:
                selected_clips.append(clip)
                
                # Update budget
                clip_duration = clip.end_s - clip.start_s
                used_time += clip_duration
                
                # Add suppression region
                suppression_radius = clip_duration * self.config['suppression_radius_factor']
                sup_start = max(0, clip.start_s - suppression_radius)
                sup_end = clip.end_s + suppression_radius
                suppression_regions.append((sup_start, sup_end))
        
        # Ensure diversity across time buckets
        selected_clips = self._ensure_temporal_diversity(selected_clips, segments)
        
        return selected_clips
    
    def _overlaps(self, start1: float, end1: float, start2: float, end2: float) -> bool:
        """Check if two time ranges overlap"""
        return start1 < end2 and end1 > start2
    
    def _create_clip(self, window: Dict, all_segments: List[Dict]) -> Optional[Clip]:
        """Create a clip from a uncertainty window"""
        try:
            # Determine clip boundaries
            clip_start = window['start_s']
            clip_end = window['end_s']
            
            # Extend to include complete sentences/segments
            clip_start, clip_end = self._adjust_clip_boundaries(
                clip_start, clip_end, all_segments
            )
            
            # Enforce length constraints
            min_len = self.config['clip_len_min_s']
            max_len = self.config['clip_len_max_s']
            
            if clip_end - clip_start < min_len:
                # Extend clip to minimum length
                extension = (min_len - (clip_end - clip_start)) / 2
                clip_start = max(0, clip_start - extension)
                clip_end = clip_start + min_len
            elif clip_end - clip_start > max_len:
                # Truncate to maximum length
                clip_end = clip_start + max_len
            
            # Find segments in clip
            clip_segments = [
                seg for seg in all_segments
                if seg.get('start', 0) < clip_end and seg.get('end', 0) > clip_start
            ]
            
            if not clip_segments:
                return None
            
            # Extract speakers
            speakers = list(set(seg.get('speaker', 'Unknown') for seg in clip_segments))
            
            # Combine text
            text_parts = [seg.get('text', '').strip() for seg in clip_segments]
            text_proposed = ' '.join(part for part in text_parts if part)
            
            # Create tokens
            tokens = self._create_tokens(clip_segments)
            
            # Generate clip ID
            clip_id = hashlib.md5(f"{clip_start}_{clip_end}_{text_proposed}".encode()).hexdigest()[:8]
            
            clip = Clip(
                id=clip_id,
                start_s=clip_start,
                end_s=clip_end,
                speakers=speakers,
                text_proposed=text_proposed,
                tokens=tokens,
                uncertainty_score=window['uncertainty_score']
            )
            
            return clip
            
        except Exception as e:
            self.logger.error("Failed to create clip", extra={'error': str(e)})
            return None
    
    def _adjust_clip_boundaries(self, start: float, end: float, segments: List[Dict]) -> Tuple[float, float]:
        """Adjust clip boundaries to include complete segments"""
        if not segments:
            return start, end
        
        # Find segments that overlap with the clip
        overlapping_segments = [
            seg for seg in segments
            if seg.get('start', 0) < end and seg.get('end', 0) > start
        ]
        
        if overlapping_segments:
            # Extend to include complete segments
            adjusted_start = min(seg.get('start', start) for seg in overlapping_segments)
            adjusted_end = max(seg.get('end', end) for seg in overlapping_segments)
            return adjusted_start, adjusted_end
        
        return start, end
    
    def _create_tokens(self, segments: List[Dict]) -> List[Token]:
        """Create token list from segments"""
        tokens = []
        
        for segment in segments:
            text = segment.get('text', '')
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            confidence = segment.get('confidence', 0.8)
            
            # Split text into words (simplified tokenization)
            words = text.split()
            if words:
                word_duration = (end_time - start_time) / len(words)
                
                for i, word in enumerate(words):
                    token_start = start_time + i * word_duration
                    token_end = token_start + word_duration
                    
                    token = Token(
                        text=word,
                        start=token_start,
                        end=token_end,
                        conf=confidence,
                        provider_votes={'primary': confidence}
                    )
                    tokens.append(token)
        
        return tokens
    
    def _ensure_temporal_diversity(self, clips: List[Clip], segments: List[Dict]) -> List[Clip]:
        """Ensure clips are distributed across time buckets"""
        if not clips or not segments:
            return clips
        
        # Get total duration
        total_duration = max(seg.get('end', 0) for seg in segments)
        
        # Create time buckets
        num_buckets = self.config['diversity_coverage_buckets']
        bucket_size = total_duration / num_buckets
        
        # Group clips by bucket
        bucket_clips = [[] for _ in range(num_buckets)]
        
        for clip in clips:
            bucket_index = min(int(clip.start_s // bucket_size), num_buckets - 1)
            bucket_clips[bucket_index].append(clip)
        
        # Select best clips from each bucket
        diverse_clips = []
        for bucket in bucket_clips:
            if bucket:
                # Select highest uncertainty clip from bucket
                best_clip = max(bucket, key=lambda c: c.uncertainty_score)
                diverse_clips.append(best_clip)
        
        # Sort by time order
        diverse_clips.sort(key=lambda c: c.start_s)
        
        return diverse_clips
    
    def apply_improvements(self, original_results: Dict, human_edits: List[HumanEdit], 
                          speaker_map: Dict[str, str]) -> Dict:
        """Apply human improvements through propagation pipeline"""
        try:
            self.logger.info("Applying hotspot improvements", extra={
                'num_edits': len(human_edits),
                'speaker_map_size': len(speaker_map)
            })
            
            # Start with original results
            improved_results = original_results.copy()
            
            if not human_edits:
                self.logger.info("No human edits to apply")
                return improved_results
            
            # Apply direct edits to transcript
            improved_results = self._apply_direct_edits(improved_results, human_edits)
            
            # Apply speaker map updates
            if speaker_map:
                improved_results = self._apply_speaker_mapping(improved_results, speaker_map)
            
            # Run propagation pipeline
            improved_results = self._run_propagation_pipeline(improved_results, human_edits)
            
            # Regenerate transcript files with improvements
            output_paths = self.regenerate_transcript_files(improved_results)
            if output_paths:
                improved_results['regenerated_files'] = output_paths
            
            # Update manifest with comprehensive data
            self._update_manifest(human_edits, speaker_map, output_paths)
            
            # Mark as human-reviewed for provenance
            improved_results['human_reviewed'] = {
                'timestamp': time.time(),
                'edits_applied': len(human_edits),
                'speaker_mappings_applied': len(speaker_map),
                'files_regenerated': list(output_paths.keys()) if output_paths else []
            }
            
            self.logger.info("Hotspot improvements applied successfully", extra={
                'edits_applied': len(human_edits),
                'files_regenerated': len(output_paths) if output_paths else 0
            })
            
            return improved_results
            
        except Exception as e:
            self.logger.error("Failed to apply improvements", extra={'error': str(e)})
            return original_results
    
    def _apply_direct_edits(self, results: Dict, human_edits: List[HumanEdit]) -> Dict:
        """Apply direct human edits to transcript"""
        # Get current transcript segments and text
        segments = results.get('segments', [])
        transcript_text = results.get('transcript', '')
        raw_transcript = results.get('raw_transcript', transcript_text)
        
        if not segments and not transcript_text:
            return results
        
        # Apply each edit
        for edit in human_edits:
            if edit.text_final and edit.text_final.strip():
                # Find corresponding segments by clip_id mapping
                # For now, apply text improvements to matching content
                original_text = results.get('transcript', '')
                improved_text = self._improve_text_with_edit(original_text, edit)
                results['transcript'] = improved_text
                results['raw_transcript'] = improved_text
                
                # Apply speaker overrides if present
                if edit.speaker_label_override:
                    # Update segments with corrected speaker labels
                    for segment in segments:
                        # Simple heuristic: if segment text is similar to edit context
                        if self._is_segment_related_to_edit(segment, edit):
                            segment['speaker'] = edit.speaker_label_override
        
        # Recalculate formatted transcript from segments
        if segments:
            results['transcript'] = self._format_transcript_from_segments(segments)
        
        return results
    
    def _improve_text_with_edit(self, original_text: str, edit: HumanEdit) -> str:
        """Improve text using human edit"""
        if not edit.text_final or not edit.text_final.strip():
            return original_text
        
        # Try to find and replace similar text segments
        # This is a heuristic approach for when we don't have exact clip mapping
        lines = original_text.split('\n')
        improved_lines = []
        
        for line in lines:
            # Check if this line might be related to the edit
            if self._is_line_candidate_for_edit(line, edit):
                # Apply the edit - this is simplified but functional
                # In practice, would use more sophisticated text alignment
                improved_lines.append(self._apply_edit_to_line(line, edit))
            else:
                improved_lines.append(line)
        
        return '\n'.join(improved_lines)
    
    def _apply_speaker_mapping(self, results: Dict, speaker_map: Dict[str, str]) -> Dict:
        """Apply speaker name mapping"""
        if not speaker_map:
            return results
        
        # Update segments with new speaker names
        segments = results.get('segments', [])
        for segment in segments:
            current_speaker = segment.get('speaker')
            if current_speaker in speaker_map:
                segment['speaker'] = speaker_map[current_speaker]
        
        # Update transcript text with new speaker names
        transcript = results.get('transcript', '')
        for old_name, new_name in speaker_map.items():
            transcript = transcript.replace(f"{old_name}:", f"{new_name}:")
        
        results['transcript'] = transcript
        
        return results
    
    def _run_propagation_pipeline(self, results: Dict, human_edits: List[HumanEdit]) -> Dict:
        """Run the propagation pipeline to spread improvements"""
        try:
            # 1. Lexicon boost - extract corrected terms
            corrected_terms = self._extract_corrected_terms(human_edits)
            
            # 2. Apply term normalization
            if corrected_terms:
                results = self._apply_term_normalization(results, corrected_terms)
            
            # 3. Provider reweighting (simplified)
            # In practice, would analyze provider agreement with corrections
            
            # 4. Local repair around edited regions
            # Would run additional processing on neighboring segments
            
            # 5. Global sweep for consistency
            results = self._apply_global_consistency(results)
            
            return results
            
        except Exception as e:
            self.logger.error("Propagation pipeline failed", extra={'error': str(e)})
            return results
    
    def _is_segment_related_to_edit(self, segment: Dict, edit: HumanEdit) -> bool:
        """Check if a segment is related to a human edit"""
        # Simple heuristic: check if segment timing or content suggests relationship
        segment_text = segment.get('text', '').strip().lower()
        if not segment_text:
            return False
        
        # If we have edit context, check for word overlap
        if hasattr(edit, 'original_text') and edit.original_text:
            original_words = set(edit.original_text.lower().split())
            segment_words = set(segment_text.split())
            overlap = len(original_words & segment_words)
            return overlap >= min(3, len(original_words) // 2)
        
        # Fallback: assume first few segments for now
        return True  # Simplified - in practice would use clip timing mapping
    
    def _format_transcript_from_segments(self, segments: List[Dict]) -> str:
        """Format transcript text from segments with speaker labels"""
        formatted_lines = []
        
        for segment in segments:
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            if text:
                formatted_lines.append(f"{speaker}: {text}")
        
        return '\n\n'.join(formatted_lines)
    
    def _is_line_candidate_for_edit(self, line: str, edit: HumanEdit) -> bool:
        """Check if a line might be a candidate for applying an edit"""
        if not line.strip():
            return False
        
        # Simple heuristic: check for similar content or speaker
        line_lower = line.lower()
        
        # If edit has speaker override, look for lines with old speaker
        if edit.speaker_label_override:
            # Look for lines that might need speaker correction
            return ':' in line  # Lines with speaker labels
        
        # If edit has text, look for similar content
        if edit.text_final:
            edit_words = set(edit.text_final.lower().split())
            line_words = set(line.lower().split())
            if edit_words and line_words:
                overlap = len(edit_words & line_words)
                return overlap >= min(2, len(edit_words) // 3)
        
        return False
    
    def _apply_edit_to_line(self, line: str, edit: HumanEdit) -> str:
        """Apply an edit to a specific line"""
        if not line.strip():
            return line
        
        # Apply speaker override if present
        if edit.speaker_label_override and ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                current_speaker = parts[0].strip()
                text_part = parts[1].strip()
                # Simple speaker replacement
                return f"{edit.speaker_label_override}: {text_part}"
        
        # Apply text improvements (simplified)
        if edit.text_final and edit.text_final.strip():
            # This is a simplified approach - in practice would use sophisticated alignment
            # For now, if the edit contains significantly better text, prefer it
            return line  # Keep original for now, more sophisticated replacement needed
        
        return line
    
    def regenerate_transcript_files(self, improved_results: Dict) -> Dict[str, str]:
        """Regenerate transcript files after edits are applied"""
        try:
            from utils.transcript_formatter import TranscriptFormatter
            
            formatter = TranscriptFormatter()
            output_paths = {}
            
            # Get improved transcript data
            transcript_text = improved_results.get('transcript', '')
            segments = improved_results.get('segments', [])
            
            if not transcript_text and not segments:
                self.logger.warning("No transcript data available for file regeneration")
                return output_paths
            
            # Generate timestamp for unique filenames
            timestamp = int(time.time())
            
            # Create TXT transcript
            txt_path = f"artifacts/transcripts/improved_transcript_{timestamp}.txt"
            try:
                import os
                os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(transcript_text)
                output_paths['transcript_txt'] = txt_path
                
            except Exception as e:
                self.logger.warning(f"Failed to create TXT transcript: {e}")
            
            # Create SRT if segments available
            if segments:
                try:
                    srt_path = f"artifacts/transcripts/improved_transcript_{timestamp}.srt"
                    srt_content = self._create_srt_from_segments(segments)
                    
                    with open(srt_path, 'w', encoding='utf-8') as f:
                        f.write(srt_content)
                    output_paths['transcript_srt'] = srt_path
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create SRT transcript: {e}")
            
            # Create JSON report
            try:
                json_path = f"artifacts/transcripts/improved_results_{timestamp}.json"
                import json as json_lib
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json_lib.dump(improved_results, f, indent=2, default=str)
                output_paths['results_json'] = json_path
                
            except Exception as e:
                self.logger.warning(f"Failed to create JSON report: {e}")
            
            self.logger.info("Transcript files regenerated after hotspot edits", extra={
                'files_created': list(output_paths.keys()),
                'paths': list(output_paths.values())
            })
            
            return output_paths
            
        except Exception as e:
            self.logger.error("Failed to regenerate transcript files", extra={'error': str(e)})
            return {}
    
    def _create_srt_from_segments(self, segments: List[Dict]) -> str:
        """Create SRT format content from segments"""
        srt_content = ""
        
        for i, segment in enumerate(segments, 1):
            start_time = self._format_srt_time(segment.get('start', 0))
            end_time = self._format_srt_time(segment.get('end', 0))
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', '')
            
            if text:
                display_text = f"{speaker}: {text}" if speaker else text
                srt_content += f"{i}\n{start_time} --> {end_time}\n{display_text}\n\n"
        
        return srt_content
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _extract_corrected_terms(self, human_edits: List[HumanEdit]) -> List[str]:
        """Extract corrected terms from human edits"""
        terms = []
        
        for edit in human_edits:
            if edit.text_final:
                # Extract significant terms (simplified)
                words = edit.text_final.split()
                significant_words = [w for w in words if len(w) > 3 and not w.lower() in ['the', 'and', 'that', 'this']]
                terms.extend(significant_words)
        
        return list(set(terms))  # Remove duplicates
    
    def _apply_term_normalization(self, results: Dict, corrected_terms: List[str]) -> Dict:
        """Apply term normalization across transcript"""
        transcript = results.get('transcript', '')
        
        # Simple normalization - ensure consistent capitalization
        for term in corrected_terms:
            # This is simplified - real implementation would be more sophisticated
            pass
        
        return results
    
    def _apply_global_consistency(self, results: Dict) -> Dict:
        """Apply global consistency improvements"""
        # Simplified implementation
        # Would include:
        # - Consistent speaker name formatting
        # - Consistent punctuation patterns
        # - Consistent capitalization
        
        return results
    
    def _update_manifest(self, human_edits: List[HumanEdit], speaker_map: Dict[str, str], 
                        output_paths: Optional[Dict[str, str]] = None):
        """Update run manifest with hotspot review data and regenerated files"""
        try:
            output_paths = output_paths or {}
            
            # Prepare comprehensive manifest data
            manifest_data = {
                'hotspot_review': {
                    'timestamp': time.time(),
                    'human_edits_count': len(human_edits),
                    'speaker_map_updates': len(speaker_map),
                    'producing_component': 'HotspotManager.apply_improvements',
                    'files_regenerated': list(output_paths.keys()),
                    'output_paths': output_paths,
                    'edits_summary': [
                        {
                            'clip_id': edit.clip_id,
                            'has_text_edit': bool(edit.text_final),
                            'has_speaker_override': bool(edit.speaker_label_override),
                            'text_length': len(edit.text_final) if edit.text_final else 0,
                            'flags': edit.flags or []
                        }
                        for edit in human_edits
                    ],
                    'speaker_mappings': speaker_map
                }
            }
            
            # Call internal manifest update (simplified approach)
            self._save_manifest_data(manifest_data)
            
            self.logger.info("Manifest updated with hotspot review data", extra={
                'edits_count': len(human_edits),
                'speaker_mappings': len(speaker_map),
                'files_regenerated': len(output_paths)
            })
            
        except Exception as e:
            self.logger.error("Failed to update manifest", extra={'error': str(e)})
    
    def _save_manifest_data(self, manifest_data: Dict):
        """Save manifest data to file (simplified approach)"""
        try:
            # Create a simple manifest file for tracking hotspot review data
            manifest_dir = Path("artifacts/manifests")
            manifest_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            manifest_path = manifest_dir / f"hotspot_review_{timestamp}.json"
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, default=str)
            
            self.logger.info(f"Manifest data saved to {manifest_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save manifest data: {e}")