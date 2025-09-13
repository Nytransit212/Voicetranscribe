"""
Selective ASR System for Ensemble Transcription
U7 Upgrade: Targeted reprocessing of low-confidence segments
"""

import os
import tempfile
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import librosa
import soundfile as sf

from core.asr_engine import ASREngine
from utils.segment_worklist import SegmentWorklistManager, SegmentFlag, get_worklist_manager
from utils.intelligent_cache import get_cache_manager, cached_operation
from utils.deterministic_processing import get_deterministic_processor, set_global_seed
from utils.enhanced_structured_logger import create_enhanced_logger

# Configure logging
selective_logger = logging.getLogger(__name__)

@dataclass
class SegmentReprocessRequest:
    """Request for segment reprocessing"""
    segment_id: str
    file_path: str
    audio_segment_path: str
    start_time: float
    end_time: float
    original_transcript: str
    current_confidence: float
    priority: int
    processing_context: Dict[str, Any]

class SelectiveASRProcessor:
    """
    Handles selective reprocessing of flagged audio segments.
    Optimized for targeted quality improvement with minimal overhead.
    """
    
    def __init__(self, target_language: Optional[str] = None,
                 enable_caching: bool = True,
                 max_concurrent_requests: int = 2):
        """
        Initialize selective ASR processor.
        
        Args:
            target_language: Target language for ASR (None for auto-detect)
            enable_caching: Whether to use caching for reprocessing
            max_concurrent_requests: Maximum concurrent ASR requests for reprocessing
        """
        self.target_language = target_language
        self.enable_caching = enable_caching
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize components
        self.asr_engine = ASREngine()
        self.worklist_manager = get_worklist_manager()
        self.cache_manager = get_cache_manager()
        self.deterministic_processor = get_deterministic_processor()
        
        # Enhanced logging
        self.logger = create_enhanced_logger("selective_asr")
        
        # Processing statistics
        self.stats = {
            'segments_reprocessed': 0,
            'segments_improved': 0,
            'total_improvement_score': 0.0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        selective_logger.info("Initialized selective ASR processor")
        selective_logger.info(f"Max concurrent requests: {max_concurrent_requests}")
    
    def process_flagged_segments(self, file_path: str, run_id: str,
                               max_segments: Optional[int] = None,
                               progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process flagged segments for a file.
        
        Args:
            file_path: Path to original file
            run_id: Processing run identifier
            max_segments: Maximum number of segments to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results with improvements
        """
        start_time = time.time()
        
        # Get prioritized segments for reprocessing
        flagged_segments = self.worklist_manager.get_prioritized_segments(
            file_path, run_id, max_segments
        )
        
        if not flagged_segments:
            selective_logger.info(f"No flagged segments found for {file_path}")
            return {
                'segments_processed': 0,
                'segments_improved': 0,
                'total_improvement': 0.0,
                'processing_time': 0.0
            }
        
        selective_logger.info(f"Processing {len(flagged_segments)} flagged segments for {file_path}")
        
        # Load original audio for segment extraction
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        except Exception as e:
            selective_logger.error(f"Failed to load audio file {file_path}: {e}")
            return {'error': f"Failed to load audio: {e}"}
        
        # Create reprocess requests
        reprocess_requests = []
        for segment in flagged_segments:
            # Extract audio segment
            segment_audio_path = self._extract_audio_segment(
                audio_data, sample_rate, segment, file_path
            )
            
            if segment_audio_path:
                request = SegmentReprocessRequest(
                    segment_id=segment.segment_id,
                    file_path=file_path,
                    audio_segment_path=segment_audio_path,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    original_transcript=segment.original_transcript,
                    current_confidence=segment.current_confidence,
                    priority=segment.processing_priority,
                    processing_context={
                        'run_id': run_id,
                        'flag_reason': segment.flag_reason,
                        'reprocess_count': segment.reprocess_count
                    }
                )
                reprocess_requests.append(request)
        
        # Process segments with intelligent scheduling
        results = self._process_segments_concurrently(
            reprocess_requests, progress_callback
        )
        
        # Update worklist with results
        improvements = []
        for result in results:
            if 'improved_transcript' in result:
                self.worklist_manager.mark_segment_processed(
                    file_path=file_path,
                    run_id=run_id,
                    segment_id=result['segment_id'],
                    new_transcript=result['improved_transcript'],
                    new_confidence=result['new_confidence']
                )
                
                improvement = result['new_confidence'] - result['original_confidence']
                improvements.append(improvement)
        
        # Cleanup temporary files
        self._cleanup_temp_files(reprocess_requests)
        
        # Calculate summary
        processing_time = time.time() - start_time
        total_improvement = sum(improvements)
        improved_count = len([imp for imp in improvements if imp > 0.05])
        
        summary = {
            'segments_processed': len(results),
            'segments_improved': improved_count,
            'total_improvement': total_improvement,
            'average_improvement': total_improvement / max(1, len(improvements)),
            'processing_time': processing_time,
            'improvements': improvements
        }
        
        # Update statistics
        self.stats['segments_reprocessed'] += len(results)
        self.stats['segments_improved'] += improved_count
        self.stats['total_improvement_score'] += total_improvement
        
        selective_logger.info(f"Selective reprocessing complete: {improved_count}/{len(results)} segments improved")
        return summary
    
    def _extract_audio_segment(self, audio_data: np.ndarray, sample_rate: int,
                             segment: SegmentFlag, original_file_path: str) -> Optional[str]:
        """
        Extract audio segment to temporary file.
        
        Args:
            audio_data: Full audio data
            sample_rate: Audio sample rate
            segment: Segment to extract
            original_file_path: Original file path for temp naming
            
        Returns:
            Path to extracted segment file or None if failed
        """
        try:
            # Calculate sample indices
            start_sample = int(segment.start_time * sample_rate)
            end_sample = int(segment.end_time * sample_rate)
            
            # Add small padding for context
            padding_samples = int(0.1 * sample_rate)  # 100ms padding
            start_sample = max(0, start_sample - padding_samples)
            end_sample = min(len(audio_data), end_sample + padding_samples)
            
            # Extract segment
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) == 0:
                selective_logger.warning(f"Empty audio segment for {segment.segment_id}")
                return None
            
            # Create temporary file
            temp_dir = tempfile.mkdtemp(prefix='selective_asr_')
            segment_filename = f"{segment.segment_id}_{int(segment.start_time)}_{int(segment.end_time)}.wav"
            segment_path = os.path.join(temp_dir, segment_filename)
            
            # Save segment
            sf.write(segment_path, segment_audio, sample_rate)
            
            selective_logger.debug(f"Extracted segment {segment.segment_id} to {segment_path}")
            return segment_path
            
        except Exception as e:
            selective_logger.error(f"Failed to extract segment {segment.segment_id}: {e}")
            return None
    
    def _process_segments_concurrently(self, requests: List[SegmentReprocessRequest],
                                     progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Process multiple segments concurrently with intelligent scheduling.
        
        Args:
            requests: List of reprocess requests
            progress_callback: Optional progress callback
            
        Returns:
            List of processing results
        """
        # Sort by priority (highest first) and duration (shortest first for quick wins)
        sorted_requests = sorted(
            requests,
            key=lambda r: (-r.priority, r.end_time - r.start_time)
        )
        
        results = []
        total_requests = len(sorted_requests)
        
        # Process with controlled concurrency
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # Submit all tasks
            future_to_request = {}
            for request in sorted_requests:
                future = executor.submit(self._process_single_segment, request)
                future_to_request[future] = request
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_request)):
                request = future_to_request[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            f"Processed segment {request.segment_id}",
                            int((i + 1) / total_requests * 100),
                            f"Segment {i + 1}/{total_requests}"
                        )
                    
                    selective_logger.debug(f"Completed processing segment {request.segment_id}")
                    
                except Exception as e:
                    selective_logger.error(f"Error processing segment {request.segment_id}: {e}")
                    results.append({
                        'segment_id': request.segment_id,
                        'error': str(e),
                        'original_confidence': request.current_confidence
                    })
        
        return results
    
    @cached_operation("selective_asr_segment")
    def _process_single_segment(self, request: SegmentReprocessRequest) -> Dict[str, Any]:
        """
        Process a single segment with caching and deterministic processing.
        
        Args:
            request: Segment reprocess request
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        # Set deterministic seed for consistent reprocessing
        set_global_seed("selective_asr", {
            'segment_id': request.segment_id,
            'file_path': request.file_path,
            'reprocess_count': request.processing_context.get('reprocess_count', 0)
        })
        
        try:
            # Create enhanced ASR configuration for reprocessing
            asr_config = self._create_enhanced_asr_config(request)
            
            # Process with ASR engine
            result = self.asr_engine._make_transcription_api_call(
                request.audio_segment_path,
                **asr_config
            )
            
            # Extract improved transcript
            improved_transcript = result.text.strip() if hasattr(result, 'text') else str(result)
            
            # Calculate confidence improvement (simplified - in real implementation would use proper confidence scoring)
            new_confidence = self._estimate_transcript_confidence(
                improved_transcript, request.original_transcript
            )
            
            processing_time = time.time() - start_time
            
            result_data = {
                'segment_id': request.segment_id,
                'original_transcript': request.original_transcript,
                'improved_transcript': improved_transcript,
                'original_confidence': request.current_confidence,
                'new_confidence': new_confidence,
                'improvement': new_confidence - request.current_confidence,
                'processing_time': processing_time,
                'asr_config': asr_config
            }
            
            selective_logger.debug(f"Processed segment {request.segment_id}: "
                                 f"confidence {request.current_confidence:.3f} → {new_confidence:.3f}")
            return result_data
            
        except Exception as e:
            selective_logger.error(f"Error in single segment processing {request.segment_id}: {e}")
            return {
                'segment_id': request.segment_id,
                'error': str(e),
                'original_confidence': request.current_confidence,
                'processing_time': time.time() - start_time
            }
    
    def _create_enhanced_asr_config(self, request: SegmentReprocessRequest) -> Dict[str, Any]:
        """
        Create enhanced ASR configuration for reprocessing.
        
        Args:
            request: Reprocess request
            
        Returns:
            ASR configuration optimized for segment reprocessing
        """
        config = {
            'model': 'whisper-1',
            'response_format': 'text',
            'temperature': 0.0,  # Deterministic
        }
        
        # Add language if specified
        if self.target_language:
            config['language'] = self.target_language
        
        # Adjust temperature based on original confidence
        if request.current_confidence < 0.3:
            # Very low confidence - try more creative transcription
            config['temperature'] = 0.2
        elif request.current_confidence < 0.5:
            # Low confidence - slight creativity
            config['temperature'] = 0.1
        
        # Add context prompts for specific improvement areas
        prompt_hints = []
        if 'overlapping' in request.processing_context.get('flag_reason', '').lower():
            prompt_hints.append("Multiple speakers may be talking simultaneously.")
        if 'unclear' in request.processing_context.get('flag_reason', '').lower():
            prompt_hints.append("Audio may contain unclear speech or background noise.")
        
        if prompt_hints:
            config['prompt'] = " ".join(prompt_hints)
        
        return config
    
    def _estimate_transcript_confidence(self, new_transcript: str, original_transcript: str) -> float:
        """
        Estimate confidence of new transcript compared to original.
        Simplified implementation - in practice would use proper confidence scoring.
        
        Args:
            new_transcript: New transcript text
            original_transcript: Original transcript text
            
        Returns:
            Estimated confidence score
        """
        # Simple heuristics for confidence estimation
        base_confidence = 0.6
        
        # Length improvements
        if len(new_transcript) > len(original_transcript) * 1.1:
            base_confidence += 0.1
        
        # Word count improvements
        new_words = len(new_transcript.split())
        original_words = len(original_transcript.split())
        
        if new_words > original_words:
            base_confidence += min(0.15, (new_words - original_words) * 0.02)
        
        # Similarity check - very different transcripts might indicate improvement
        similarity = self._calculate_text_similarity(new_transcript, original_transcript)
        if 0.3 < similarity < 0.8:  # Some change but not completely different
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _cleanup_temp_files(self, requests: List[SegmentReprocessRequest]):
        """Clean up temporary audio segment files."""
        temp_dirs = set()
        
        for request in requests:
            try:
                if os.path.exists(request.audio_segment_path):
                    temp_dir = os.path.dirname(request.audio_segment_path)
                    temp_dirs.add(temp_dir)
                    os.remove(request.audio_segment_path)
            except Exception as e:
                selective_logger.warning(f"Failed to cleanup temp file {request.audio_segment_path}: {e}")
        
        # Remove empty temp directories
        for temp_dir in temp_dirs:
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                selective_logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'segments_reprocessed': 0,
            'segments_improved': 0,
            'total_improvement_score': 0.0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }


# Global selective ASR processor instance
_selective_asr_processor: Optional[SelectiveASRProcessor] = None

def get_selective_asr_processor() -> SelectiveASRProcessor:
    """Get or create global selective ASR processor instance."""
    global _selective_asr_processor
    if _selective_asr_processor is None:
        _selective_asr_processor = SelectiveASRProcessor()
    return _selective_asr_processor