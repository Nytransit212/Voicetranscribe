import os
import tempfile
import json
import time
import uuid
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from core.audio_processor import AudioProcessor
from core.diarization_engine import DiarizationEngine
from core.asr_engine import ASREngine
from core.confidence_scorer import ConfidenceScorer
from core.consensus_module import ConsensusModule
from utils.transcript_formatter import TranscriptFormatter
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import initialize_observability, trace_stage, track_cost
from utils.profiling_manager import get_profiling_manager
from utils.observability_reporter import get_observability_reporter
from utils.dvc_versioning import DVCVersioningManager
from utils.metrics_registry import MetricsRegistryManager
from utils.intelligent_cache import get_cache_manager, cached_operation
from utils.deterministic_processing import get_deterministic_processor, ensure_deterministic_run_id
from utils.segment_worklist import get_worklist_manager, SegmentWorklistManager
from utils.selective_asr import get_selective_asr_processor, SelectiveASRProcessor
from utils.advanced_asr_scheduler import get_asr_scheduler, AdvancedASRScheduler
from core.source_separation_engine import SourceSeparationEngine, SourceSeparationResult
from core.post_fusion_punctuation_engine import PostFusionPunctuationEngine, create_punctuation_engine_from_preset
from core.dialect_handling_engine import DialectHandlingEngine
from core.dialect_config_loader import load_dialect_config

class EnsembleManager:
    """Orchestrates the entire ensemble transcription pipeline"""
    
    def __init__(self, expected_speakers: int = 10, noise_level: str = 'medium', target_language: Optional[str] = None, scoring_weights: Optional[Dict[str, float]] = None, enable_versioning: bool = True, domain: str = "general", consensus_strategy: str = "best_single_candidate", calibration_method: str = "registry_based", enable_speaker_mapping: bool = True, speaker_mapping_config: Optional[Dict[str, Any]] = None, chunked_processing_threshold: float = 900.0, enable_dialect_handling: bool = True, dialect_similarity_threshold: float = 0.7, dialect_confidence_boost: float = 0.05, supported_dialects: Optional[List[str]] = None) -> None:
        self.expected_speakers = expected_speakers
        self.noise_level = noise_level
        self.target_language = target_language  # None for auto-detect
        self.domain = domain
        self.enable_versioning = enable_versioning
        self.consensus_strategy = consensus_strategy
        self.calibration_method = calibration_method
        
        # Enhanced speaker mapping configuration with ECAPA-TDNN and backtracking
        self.enable_speaker_mapping = enable_speaker_mapping
        
        # Post-fusion punctuation configuration
        self.enable_post_fusion_punctuation = True
        self.punctuation_preset = "meeting_light"  # Default preset for meeting contexts
        self.punctuation_engine = None  # Will be initialized when needed
        self.speaker_mapping_config = speaker_mapping_config or {
            # Core parameters
            'similarity_threshold': 0.7,
            'embedding_dim': 192,  # ECAPA-TDNN standard dimension
            'min_segment_duration': 1.0,
            'cache_embeddings': True,
            'enable_metrics': True,
            # ECAPA-TDNN parameters
            'use_ecapa_tdnn': True,
            'ecapa_model_path': None,  # Use random weights for now (can be set to pre-trained path)
            # Backtracking parameters
            'enable_backtracking': True,
            'drift_threshold': 0.6,
            'stability_window': 5,
            'max_backtrack_chunks': 3,
            'consecutive_drift_threshold': 2
        }
        self.chunked_processing_threshold = chunked_processing_threshold  # Seconds (15 minutes default)
        
        # Dialect handling configuration
        self.enable_dialect_handling = enable_dialect_handling
        self.dialect_similarity_threshold = dialect_similarity_threshold
        self.dialect_confidence_boost = dialect_confidence_boost
        self.supported_dialects = supported_dialects or ['southern', 'aave', 'nyc', 'boston', 'midwest', 'west_coast']
        
        # U7 Upgrade: Initialize deterministic processing and other systems first
        # (run_id will be set later based on input for deterministic processing)
        self.run_id = None  # Will be set in process_video method
        
        # Initialize U7 systems
        self.cache_manager = get_cache_manager()
        self.deterministic_processor = get_deterministic_processor()
        self.worklist_manager = get_worklist_manager()
        self.selective_asr_processor = get_selective_asr_processor()
        self.asr_scheduler = get_asr_scheduler()
        
        # U7 configuration
        self.enable_caching = True
        self.enable_selective_reprocessing = True
        self.confidence_threshold_for_flagging = 0.65
        self.max_segments_for_selective_reprocessing = 10
        
        # Source separation configuration
        self.enable_source_separation = True
        self.overlap_probability_threshold = 0.25
        self.source_separation_providers = ['faster-whisper', 'deepgram', 'openai']
        
        # Initialize enhanced observability system
        self.obs_manager = initialize_observability(
            service_name="ensemble-transcription",
            enable_profiling=True,
            log_level="INFO"
        )
        
        # Initialize enhanced structured logging with run context
        self.structured_logger = create_enhanced_logger("ensemble_manager", run_id=self.run_id)
        
        # Initialize profiling manager and reporter
        self.profiling_manager = get_profiling_manager()
        self.observability_reporter = get_observability_reporter()
        
        # Initialize versioning system
        if self.enable_versioning:
            try:
                self.dvc_manager: Optional[DVCVersioningManager] = DVCVersioningManager()
                self.metrics_registry: Optional[MetricsRegistryManager] = MetricsRegistryManager()
                self.structured_logger.info("Versioning system initialized", context={'run_id': self.run_id})
            except Exception as e:
                self.structured_logger.warning(f"Failed to initialize versioning: {e}")
                self.enable_versioning = False
                self.dvc_manager = None
                self.metrics_registry = None
        else:
            self.dvc_manager = None
            self.metrics_registry = None
        
        # Initialize components with versioning context
        self.audio_processor = AudioProcessor()
        self.diarization_engine = DiarizationEngine(
            expected_speakers=expected_speakers, 
            noise_level=noise_level,
            enable_speaker_mapping=enable_speaker_mapping,
            speaker_mapping_config=self.speaker_mapping_config
        )
        self.asr_engine = ASREngine()
        self.confidence_scorer = ConfidenceScorer(
            scoring_weights=scoring_weights,
            use_registry_calibration=self.enable_versioning,
            domain=domain,
            speaker_count=expected_speakers,
            noise_level=noise_level,
            calibration_method=calibration_method
        )
        self.transcript_formatter = TranscriptFormatter()
        
        # Initialize source separation engine
        try:
            self.source_separation_engine = SourceSeparationEngine(
                overlap_threshold=self.overlap_probability_threshold,
                enable_caching=self.enable_caching
            )
            if self.source_separation_engine.is_available():
                self.structured_logger.info("Source separation engine initialized successfully")
            else:
                self.structured_logger.warning("Source separation engine not available - Demucs models unavailable")
                self.enable_source_separation = False
        except Exception as e:
            self.structured_logger.warning(f"Failed to initialize source separation engine: {e}")
            self.source_separation_engine = None
            self.enable_source_separation = False
        
        # Initialize post-fusion punctuation engine if enabled
        if self.enable_post_fusion_punctuation:
            try:
                self.punctuation_engine = create_punctuation_engine_from_preset(self.punctuation_preset)
                self.structured_logger.info(f"Post-fusion punctuation engine initialized with preset: {self.punctuation_preset}")
            except Exception as e:
                self.structured_logger.warning(f"Failed to initialize punctuation engine: {e}")
                self.enable_post_fusion_punctuation = False
                self.punctuation_engine = None
        
        # Initialize dialect handling engine with proper config loading
        try:
            # Load dialect configuration from YAML
            dialect_config = load_dialect_config()
            
            # Override enable setting if explicitly disabled
            if not enable_dialect_handling:
                dialect_config.enable_dialect_handling = False
            
            self.enable_dialect_handling = dialect_config.enable_dialect_handling
            
            if self.enable_dialect_handling:
                self.dialect_engine = DialectHandlingEngine(
                    similarity_threshold=dialect_config.similarity_threshold,
                    confidence_boost_factor=dialect_config.confidence_boost_factor,
                    supported_dialects=dialect_config.supported_dialects,
                    enable_g2p_fallback=dialect_config.enable_g2p_fallback,
                    config=dialect_config
                )
                
                # Update instance variables to match loaded config
                self.dialect_similarity_threshold = dialect_config.similarity_threshold
                self.dialect_confidence_boost = dialect_config.confidence_boost_factor
                self.supported_dialects = dialect_config.supported_dialects
                
                self.structured_logger.info(f"Dialect handling engine initialized with config from YAML")
                self.structured_logger.info(f"Supported dialects: {dialect_config.supported_dialects}")
                self.structured_logger.info(f"Similarity threshold: {dialect_config.similarity_threshold}")
                self.structured_logger.info(f"Confidence boost factor: {dialect_config.confidence_boost_factor}")
            else:
                self.dialect_engine = None
                self.structured_logger.info("Dialect handling disabled in configuration")
                
        except Exception as e:
            self.structured_logger.warning(f"Failed to initialize dialect handling engine: {e}")
            self.enable_dialect_handling = False
            self.dialect_engine = None
        
        # Initialize consensus module
        try:
            self.consensus_module = ConsensusModule(default_strategy=consensus_strategy)
        except Exception as e:
            # Fallback to best single candidate if consensus module fails
            self.structured_logger.warning(f"Failed to initialize consensus module: {e}")
            self.consensus_module = ConsensusModule(default_strategy="best_single_candidate")
        
        # Working directory for temporary files
        self.work_dir: Optional[str] = None
        self.temp_audio_files: List[str] = []  # Track temp audio files for cleanup
        
        # Artifact tracking for run manifest
        self.input_artifacts: Dict[str, Any] = {}
        self.intermediate_artifacts: Dict[str, Any] = {}
        self.output_artifacts: Dict[str, Any] = {}
    
    def _process_candidate_for_dialect(self, candidate: Dict[str, Any], aligned_segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Process a candidate through dialect handling engine
        
        Args:
            candidate: Original candidate dictionary
            aligned_segments: Aligned segments from the candidate
            
        Returns:
            Updated candidate with dialect adjustments or None if processing failed
        """
        if not self.dialect_engine:
            return None
        
        try:
            # Extract ASR data and create simplified ASR segments for dialect processing
            asr_data = candidate.get('asr_data', {})
            segments_for_processing = []
            
            # Convert aligned segments to ASR segments for dialect engine
            for segment in aligned_segments:
                # Create simplified ASRSegment-like object for processing
                from core.asr_providers.base import ASRSegment
                
                asr_segment = ASRSegment(
                    start=segment.get('start', 0.0),
                    end=segment.get('end', 0.0),
                    text=segment.get('text', ''),
                    confidence=segment.get('confidence', 0.0),
                    words=segment.get('words', []),
                    speaker_id=segment.get('speaker_id', 'unknown')
                )
                segments_for_processing.append(asr_segment)
            
            # Create simplified ASRResult for dialect processing
            from core.asr_providers.base import ASRResult, DecodeMode
            
            full_text = ' '.join(seg.text for seg in segments_for_processing if seg.text.strip())
            avg_confidence = sum(seg.confidence for seg in segments_for_processing) / len(segments_for_processing) if segments_for_processing else 0.0
            
            asr_result = ASRResult(
                segments=segments_for_processing,
                full_text=full_text,
                language='en',
                confidence=avg_confidence,
                calibrated_confidence=avg_confidence,
                processing_time=0.0,
                provider=asr_data.get('provider', 'unknown'),
                decode_mode=DecodeMode.DETERMINISTIC,
                model_name=asr_data.get('model_name', 'unknown'),
                metadata={'dialect_processing': True}
            )
            
            # Process through dialect handling engine
            dialect_result = self.dialect_engine.process_asr_result(asr_result)
            
            if not dialect_result or dialect_result.overall_confidence_adjustment == 0.0:
                return candidate  # No adjustment needed
            
            # Apply adjustments to the candidate
            updated_candidate = candidate.copy()
            
            # Update aligned segments with adjusted confidences
            updated_segments = []
            for i, original_segment in enumerate(aligned_segments):
                updated_segment = original_segment.copy()
                
                # Apply confidence adjustment if we have segment analysis
                if i < len(dialect_result.segment_analyses):
                    analysis = dialect_result.segment_analyses[i]
                    adjustment = analysis.confidence_adjustments.get('segment', 0.0)
                    
                    # Apply adjustment
                    original_confidence = updated_segment.get('confidence', 0.0)
                    new_confidence = min(1.0, original_confidence + adjustment)
                    updated_segment['confidence'] = new_confidence
                    
                    # Add dialect metadata
                    if adjustment > 0.0:
                        updated_segment['dialect_metadata'] = {
                            'confidence_adjustment': adjustment,
                            'patterns_matched': [p.pattern_id for p in analysis.dialect_patterns_matched],
                            'phonetic_adjustments_count': len(analysis.phonetic_adjustments)
                        }
                
                updated_segments.append(updated_segment)
            
            updated_candidate['aligned_segments'] = updated_segments
            
            # Update candidate metadata
            updated_candidate['dialect_processing_metadata'] = {
                'overall_adjustment': dialect_result.overall_confidence_adjustment,
                'patterns_detected': dialect_result.dialect_patterns_detected,
                'processing_time': dialect_result.processing_time,
                'stats': dialect_result.processing_stats
            }
            
            # Store confidence adjustments for tracking
            confidence_adjustments = []
            for analysis in dialect_result.segment_analyses:
                if analysis.confidence_adjustments.get('segment', 0.0) > 0.0:
                    confidence_adjustments.append({
                        'segment_text': analysis.original_segment.text,
                        'adjustment': analysis.confidence_adjustments.get('segment', 0.0),
                        'patterns': [p.pattern_id for p in analysis.dialect_patterns_matched]
                    })
            
            updated_candidate['confidence_adjustments'] = confidence_adjustments
            updated_candidate['dialect_patterns_detected'] = dialect_result.dialect_patterns_detected
            
            return updated_candidate
            
        except Exception as e:
            self.structured_logger.warning(f"Dialect processing error: {e}")
            return None
    
    def _apply_source_separation_patches(self, 
                                       original_segments: List[Dict[str, Any]], 
                                       separation_results: List[SourceSeparationResult]) -> List[Dict[str, Any]]:
        """
        Apply source separation patches to diarization timeline by replacing overlapped intervals
        
        This is the CRITICAL method that ensures source-separated segments actually replace
        overlapped regions in the diarization timeline instead of just being added as candidates.
        
        Args:
            original_segments: Original diarization segments
            separation_results: Results from source separation processing
            
        Returns:
            Patched segments with overlap intervals replaced by source-separated segments
        """
        if not separation_results:
            return original_segments
        
        self.structured_logger.info(f"Applying {len(separation_results)} source separation patches to timeline",
                                  context={'original_segments': len(original_segments)})
        
        # Sort original segments by start time
        working_segments = sorted(original_segments, key=lambda x: x['start'])
        
        # Apply each source separation result
        for sep_result in separation_results:
            if not sep_result.final_segments:
                continue
            
            overlap_frame = sep_result.overlap_frame
            
            # Find and remove segments that overlap with the source separation frame
            segments_to_remove = []
            segments_to_modify = []
            
            for i, segment in enumerate(working_segments):
                segment_start = segment['start']
                segment_end = segment['end']
                
                # Check for overlap with source separation frame
                if (segment_start < overlap_frame.end_time and segment_end > overlap_frame.start_time):
                    
                    # Full overlap - remove entirely
                    if (segment_start >= overlap_frame.start_time and segment_end <= overlap_frame.end_time):
                        segments_to_remove.append(i)
                        self.structured_logger.debug(f"Removing fully overlapped segment {segment_start:.3f}-{segment_end:.3f}")
                    
                    # Partial overlap - trim segment
                    else:
                        # Segment extends before overlap frame
                        if segment_start < overlap_frame.start_time < segment_end:
                            # Trim segment to end at overlap start
                            modified_segment = segment.copy()
                            modified_segment['end'] = overlap_frame.start_time
                            modified_segment['source_separation_trimmed'] = True
                            modified_segment['original_end'] = segment_end
                            segments_to_modify.append((i, modified_segment))
                            
                            self.structured_logger.debug(f"Trimming segment end {segment_start:.3f}-{segment_end:.3f} to {segment_start:.3f}-{overlap_frame.start_time:.3f}")
                        
                        # Segment extends after overlap frame
                        elif segment_start < overlap_frame.end_time < segment_end:
                            # Trim segment to start at overlap end
                            modified_segment = segment.copy()
                            modified_segment['start'] = overlap_frame.end_time
                            modified_segment['source_separation_trimmed'] = True
                            modified_segment['original_start'] = segment_start
                            segments_to_modify.append((i, modified_segment))
                            
                            self.structured_logger.debug(f"Trimming segment start {segment_start:.3f}-{segment_end:.3f} to {overlap_frame.end_time:.3f}-{segment_end:.3f}")
                        
                        # Segment spans entire overlap frame - split into two segments
                        elif segment_start < overlap_frame.start_time and segment_end > overlap_frame.end_time:
                            # Create two segments: before and after overlap
                            before_segment = segment.copy()
                            before_segment['end'] = overlap_frame.start_time
                            before_segment['source_separation_split'] = 'before'
                            before_segment['original_end'] = segment_end
                            
                            after_segment = segment.copy()
                            after_segment['start'] = overlap_frame.end_time
                            after_segment['source_separation_split'] = 'after'
                            after_segment['original_start'] = segment_start
                            
                            # Remove original and add both new segments
                            segments_to_remove.append(i)
                            working_segments.extend([before_segment, after_segment])
                            
                            self.structured_logger.debug(f"Splitting segment {segment_start:.3f}-{segment_end:.3f} around overlap {overlap_frame.start_time:.3f}-{overlap_frame.end_time:.3f}")
            
            # Apply modifications in reverse order to preserve indices
            for i, modified_segment in reversed(segments_to_modify):
                working_segments[i] = modified_segment
            
            # Remove segments in reverse order to preserve indices
            for i in sorted(segments_to_remove, reverse=True):
                removed_segment = working_segments.pop(i)
                self.structured_logger.debug(f"Removed overlapped segment: {removed_segment.get('start', 0):.3f}-{removed_segment.get('end', 0):.3f}")
            
            # Insert source-separated segments
            for final_segment in sep_result.final_segments:
                # Ensure segment is properly formatted
                patched_segment = {
                    'start': final_segment['start'],
                    'end': final_segment['end'],
                    'speaker_id': final_segment['speaker_id'],
                    'text': final_segment.get('text', ''),
                    'confidence': final_segment.get('confidence', 0.5),
                    
                    # Mark as source separated
                    'source_separated': True,
                    'separation_confidence': final_segment.get('separation_confidence', 0.0),
                    'attribution_confidence': final_segment.get('attribution_confidence', 0.0),
                    'original_overlap_prob': final_segment.get('original_overlap_prob', 0.0),
                    
                    # Timeline replacement metadata
                    'timeline_replacement': final_segment.get('timeline_replacement', {}),
                    'overlap_frame_bounds': final_segment.get('overlap_frame_bounds', {}),
                    'processing_metadata': final_segment.get('processing_metadata', {})
                }
                
                working_segments.append(patched_segment)
                self.structured_logger.debug(f"Inserted source-separated segment: {patched_segment['start']:.3f}-{patched_segment['end']:.3f} (speaker: {patched_segment['speaker_id']})")
        
        # Sort final segments by start time
        working_segments.sort(key=lambda x: x['start'])
        
        # Validate timeline consistency
        validated_segments = self._validate_patched_timeline(working_segments)
        
        self.structured_logger.info(f"Source separation patching completed",
                                  context={
                                      'original_segments': len(original_segments),
                                      'patched_segments': len(validated_segments),
                                      'separation_results_applied': len(separation_results),
                                      'source_separated_segments': len([s for s in validated_segments if s.get('source_separated', False)])
                                  })
        
        return validated_segments
    
    def _validate_patched_timeline(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and fix issues in the patched timeline
        
        Args:
            segments: Patched segments to validate
            
        Returns:
            Validated and corrected segments
        """
        if not segments:
            return segments
        
        validated = []
        
        for segment in segments:
            # Skip invalid segments
            if segment.get('end', 0) <= segment.get('start', 0):
                continue
            
            # Skip very short segments (less than 50ms)
            if segment.get('end', 0) - segment.get('start', 0) < 0.05:
                continue
            
            # Ensure required fields
            if not segment.get('speaker_id'):
                segment['speaker_id'] = 'unknown'
            
            if not isinstance(segment.get('text'), str):
                segment['text'] = str(segment.get('text', ''))
            
            if not isinstance(segment.get('confidence'), (int, float)):
                segment['confidence'] = 0.5
            
            validated.append(segment)
        
        # Check for overlaps and resolve them
        if len(validated) > 1:
            validated = self._resolve_timeline_overlaps(validated)
        
        return validated
    
    def _resolve_timeline_overlaps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve any remaining overlaps in the patched timeline
        
        Args:
            segments: Segments sorted by start time
            
        Returns:
            Segments with overlaps resolved
        """
        if len(segments) <= 1:
            return segments
        
        resolved = [segments[0]]
        overlaps_fixed = 0
        
        for current_segment in segments[1:]:
            previous_segment = resolved[-1]
            
            # Check for overlap
            if current_segment['start'] < previous_segment['end']:
                overlaps_fixed += 1
                
                # Resolve overlap by adjusting boundaries
                overlap_midpoint = (current_segment['start'] + previous_segment['end']) / 2
                
                # Adjust previous segment end
                previous_segment['end'] = overlap_midpoint
                previous_segment['overlap_resolved'] = True
                
                # Adjust current segment start
                current_segment['start'] = overlap_midpoint
                current_segment['overlap_resolved'] = True
                
                self.structured_logger.debug(f"Resolved overlap at {overlap_midpoint:.3f}s between speakers {previous_segment.get('speaker_id')} and {current_segment.get('speaker_id')}")
            
            # Only add valid segments
            if current_segment['end'] > current_segment['start']:
                resolved.append(current_segment)
        
        if overlaps_fixed > 0:
            self.structured_logger.info(f"Resolved {overlaps_fixed} timeline overlaps during source separation patching")
        
        return resolved
    
    @trace_stage("video_processing_pipeline")
    def process_video(self, video_path: str, progress_callback: Optional[Callable[[str, int, str], None]] = None) -> Dict[str, Any]:
        """
        Process video through complete ensemble pipeline.
        
        Args:
            video_path: Path to input MP4 video file OR ASR-ready WAV file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete processing results with winner transcript and metadata
        """
        start_time = time.time()
        
        # Create temporary working directory  
        self.work_dir = tempfile.mkdtemp(prefix='ensemble_transcription_')
        
        # Log pipeline start
        self.structured_logger.stage_start("pipeline", "Starting ensemble transcription pipeline", 
                                         context={'video_path': video_path, 'expected_speakers': self.expected_speakers, 'noise_level': self.noise_level})
        
        try:
            # U7 Step 0: Generate deterministic run_id based on input hash and configuration
            processing_config = {
                'expected_speakers': self.expected_speakers,
                'noise_level': self.noise_level,
                'target_language': self.target_language,
                'domain': self.domain,
                'consensus_strategy': self.consensus_strategy,
                'calibration_method': self.calibration_method,
                'confidence_threshold': self.confidence_threshold_for_flagging,
                'enable_speaker_mapping': self.enable_speaker_mapping,
                'speaker_mapping_config': self.speaker_mapping_config,
                'chunked_processing_threshold': self.chunked_processing_threshold
            }
            
            self.run_id = ensure_deterministic_run_id(video_path, processing_config)
            self.structured_logger = create_enhanced_logger("ensemble_manager", run_id=self.run_id)
            
            self.structured_logger.info("U7: Generated deterministic run_id", 
                                      context={'run_id': self.run_id, 'config_hash': str(hash(str(processing_config)))})
            
            # Check if complete result is cached
            if self.enable_caching:
                cached_result = self.cache_manager.get("complete_ensemble_processing", video_path, processing_config)
                if cached_result is not None:
                    self.structured_logger.info("U7: Complete processing result found in cache - returning cached result")
                    return cached_result
            
            # CRITICAL FIX: Detect if input is already an ASR-ready WAV file
            is_asr_wav_file = video_path.lower().endswith(('.wav', '_asr_ready.wav')) and os.path.exists(video_path)
            
            if is_asr_wav_file:
                self.structured_logger.info("🎯 CRITICAL: Detected ASR-ready WAV file input - skipping audio extraction", 
                                          context={'asr_wav_path': video_path, 'file_size': os.path.getsize(video_path)})
                # Use the provided ASR WAV file directly
                clean_audio_path = video_path
                raw_audio_path = video_path  # Same file for both in this case
                
                # Track ASR WAV as input artifact
                if self.enable_versioning and self.dvc_manager:
                    tracked_input_path, input_artifact = self.dvc_manager.track_input_file(video_path, self.run_id)
                    self.input_artifacts['input_asr_wav'] = input_artifact
                    self.structured_logger.info("ASR WAV input tracked", context={'tracked_path': tracked_input_path})
                
                # Skip to audio duration check
                if progress_callback:
                    progress_callback("A", 10, "Using provided ASR-ready WAV file...")
                    
            else:
                # Original video file processing path
                self.structured_logger.info("📹 Processing video file - extracting audio", 
                                          context={'video_path': video_path})
                
                # Step 0: Track input video (versioning)
                if self.enable_versioning and self.dvc_manager:
                    tracked_input_path, input_artifact = self.dvc_manager.track_input_file(video_path, self.run_id)
                    self.input_artifacts['input_video'] = input_artifact
                    self.structured_logger.info("Input video tracked", context={'tracked_path': tracked_input_path})
                
                # Step 1: Audio Extraction (0-10%)
                if progress_callback:
                    progress_callback("A", 5, "Extracting audio from video...")
                
                stage_start_time = time.time()
                self.structured_logger.stage_start("audio_extraction", "Extracting and cleaning audio from video")
                
                raw_audio_path, clean_audio_path = self.audio_processor.extract_audio_from_video(video_path)
                self.temp_audio_files.extend([raw_audio_path, clean_audio_path])
                
                # Track audio artifacts (versioning)
                if self.enable_versioning and self.dvc_manager:
                    audio_artifacts = self.dvc_manager.track_audio_artifacts(raw_audio_path, clean_audio_path, self.run_id)
                    self.intermediate_artifacts.update(audio_artifacts)
                
                audio_extraction_time = time.time() - stage_start_time
                self.structured_logger.stage_complete("audio_extraction", "Audio extraction completed", 
                                                    duration=audio_extraction_time,
                                                    context={'raw_audio_path': raw_audio_path, 'clean_audio_path': clean_audio_path})
            
            # VALIDATION: Log audio file details for AssemblyAI integration
            audio_size = os.path.getsize(clean_audio_path)
            audio_format = clean_audio_path.split('.')[-1].lower()
            
            self.structured_logger.info("🎯 VALIDATION: Audio file prepared for ASR processing", 
                                      context={
                                          'audio_path': clean_audio_path,
                                          'audio_size_bytes': audio_size,
                                          'audio_format': audio_format,
                                          'is_asr_ready_wav': is_asr_wav_file,
                                          'expected_format': 'WAV (mono, 16kHz, PCM)',
                                          'integration_target': 'AssemblyAI'
                                      })
            
            if is_asr_wav_file:
                self.structured_logger.info("✅ CRITICAL SUCCESS: Using ASR-ready WAV file for ensemble processing", 
                                          context={
                                              'bypassed_video_extraction': True,
                                              'audio_source': 'asr_wav_path',
                                              'file_path': clean_audio_path
                                          })
            else:
                self.structured_logger.info("📹 Using extracted audio from video file", 
                                          context={
                                              'performed_video_extraction': True,
                                              'audio_source': 'video_extraction',
                                              'raw_audio': raw_audio_path,
                                              'clean_audio': clean_audio_path
                                          })
            
            # Step 2: Audio Preprocessing (10-15%)
            if progress_callback:
                progress_callback("B", 12, "Preprocessing audio (noise reduction, normalization)...")
            
            audio_duration = self.audio_processor.get_audio_duration(clean_audio_path)
            estimated_noise = self.audio_processor.estimate_noise_level(clean_audio_path)
            
            # Handle edge cases for audio content
            if audio_duration < 5.0:
                if progress_callback:
                    progress_callback("WARN", 15, f"Very short audio ({audio_duration:.1f}s) - results may be limited")
            elif audio_duration > 5400:  # 90 minutes
                if progress_callback:
                    progress_callback("WARN", 15, f"Very long audio ({audio_duration/60:.1f}min) - processing may take extended time")
            
            # Check for silent content
            if self._is_mostly_silent(clean_audio_path):
                if progress_callback:
                    progress_callback("WARN", 15, "Audio appears mostly silent - transcription results may be minimal")
            
            # Step 3: Diarization Variants (15-35%)
            if progress_callback:
                progress_callback("C", 20, "Creating 5 diarization variants with voting fusion...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("diarization", "Creating diarization variants with voting fusion",
                                             context={'audio_duration': audio_duration, 'estimated_noise': estimated_noise})
            
            # Determine if chunked processing with speaker mapping should be used
            enable_chunked_processing = (self.enable_speaker_mapping and 
                                       audio_duration > self.chunked_processing_threshold)
            
            if enable_chunked_processing:
                self.structured_logger.info(f"🧩 Enabling chunked processing with speaker mapping for {audio_duration/60:.1f}min audio")
                if progress_callback:
                    progress_callback("C", 22, f"Processing {audio_duration/60:.1f}min audio in chunks with speaker mapping...")
            
            diarization_variants = self.diarization_engine.create_diarization_variants(
                clean_audio_path, 
                use_voting_fusion=True,
                enable_chunked_processing=enable_chunked_processing
            )
            
            # Track diarization artifacts (versioning)
            if self.enable_versioning and self.dvc_manager:
                diarization_artifacts = self.dvc_manager.track_diarization_artifacts(diarization_variants, self.run_id)
                self.intermediate_artifacts.update(diarization_artifacts)
            
            diarization_time = time.time() - stage_start_time
            self.structured_logger.stage_complete("diarization", "Diarization variants created", 
                                                duration=diarization_time,
                                                metrics={'variants_created': len(diarization_variants)})
            
            if progress_callback:
                progress_callback("C", 35, f"Diarization complete - {len(diarization_variants)} variants with fusion created")
            
            # Step 3.5: Overlap Detection and Source Separation with Timeline Integration (35-40%)
            source_separation_patches_applied = 0
            if self.enable_source_separation and self.source_separation_engine:
                if progress_callback:
                    progress_callback("C2", 36, "Detecting overlaps and performing source separation...")
                
                source_sep_start_time = time.time()
                self.structured_logger.stage_start("source_separation", "Processing overlap frames with timeline integration",
                                                 context={'overlap_threshold': self.overlap_probability_threshold})
                
                # CRITICAL: Apply source separation with timeline patching for each diarization variant
                total_overlap_frames = 0
                total_patches_applied = 0
                
                for variant_idx, diarization_variant in enumerate(diarization_variants):
                    try:
                        # Extract segments from diarization variant
                        variant_segments = diarization_variant.get('segments', [])
                        
                        if variant_segments:
                            # Run source separation on this variant
                            separation_results = self.source_separation_engine.process_audio_with_overlaps(
                                clean_audio_path,
                                variant_segments,
                                asr_providers=self.source_separation_providers
                            )
                            
                            total_overlap_frames += len(separation_results)
                            
                            # CRITICAL: Patch final_segments into diarization timeline
                            if separation_results:
                                patched_segments = self._apply_source_separation_patches(
                                    variant_segments, separation_results
                                )
                                
                                # Update the diarization variant with patched timeline
                                diarization_variant['segments'] = patched_segments
                                diarization_variant['source_separation_applied'] = True
                                diarization_variant['source_separation_metadata'] = {
                                    'overlap_frames_processed': len(separation_results),
                                    'patches_applied': len([r for r in separation_results if r.final_segments]),
                                    'processing_timestamp': time.time()
                                }
                                
                                total_patches_applied += len([r for r in separation_results if r.final_segments])
                                
                                self.structured_logger.info(
                                    f"Applied {len(separation_results)} source separation patches to variant {variant_idx}",
                                    context={
                                        'variant_id': diarization_variant.get('variant_id', variant_idx),
                                        'original_segments': len(variant_segments),
                                        'patched_segments': len(patched_segments),
                                        'overlap_frames': len(separation_results)
                                    }
                                )
                    
                    except Exception as e:
                        self.structured_logger.warning(f"Source separation failed for variant {variant_idx}: {e}")
                        # Ensure variant is marked as unmodified
                        diarization_variant['source_separation_applied'] = False
                        continue
                
                # Clean up temporary files
                if hasattr(self.source_separation_engine, 'cleanup_temp_files'):
                    try:
                        self.source_separation_engine.cleanup_temp_files([])
                    except Exception as e:
                        self.structured_logger.warning(f"Failed to cleanup source separation temp files: {e}")
                
                source_sep_time = time.time() - source_sep_start_time
                source_separation_patches_applied = total_patches_applied
                
                self.structured_logger.stage_complete("source_separation", "Source separation timeline patching completed",
                                                    duration=source_sep_time,
                                                    metrics={
                                                        'overlap_frames_processed': total_overlap_frames,
                                                        'timeline_patches_applied': total_patches_applied,
                                                        'variants_processed': len(diarization_variants)
                                                    })
                
                if progress_callback:
                    progress_callback("C2", 40, f"Source separation complete - {total_patches_applied} overlap intervals replaced in {len(diarization_variants)} diarization variants")
            
            # Step 4: ASR Ensemble (40-75%)
            if progress_callback:
                progress_callback("D", 40, "Running expanded ASR ensemble (5 passes per diarization)...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("asr_ensemble", "Running ASR ensemble across all diarization variants",
                                             context={'diarization_variants': len(diarization_variants), 'target_language': self.target_language})
            
            candidates = self.asr_engine.run_asr_ensemble(clean_audio_path, diarization_variants, target_language=self.target_language)
            
            # Log source separation integration results
            if source_separation_patches_applied > 0:
                self.structured_logger.info(
                    f"Source separation timeline integration completed - {source_separation_patches_applied} overlap intervals replaced",
                    context={
                        'total_candidates': len(candidates),
                        'patches_applied': source_separation_patches_applied,
                        'modified_variants': sum(1 for v in diarization_variants if v.get('source_separation_applied', False))
                    }
                )
            
            # Track ASR artifacts (versioning)
            if self.enable_versioning and self.dvc_manager:
                asr_artifacts = self.dvc_manager.track_asr_artifacts(candidates, self.run_id)
                self.intermediate_artifacts.update(asr_artifacts)
            
            asr_time = time.time() - stage_start_time
            self.structured_logger.stage_complete("asr_ensemble", "ASR ensemble processing completed", 
                                                duration=asr_time,
                                                metrics={'total_candidates': len(candidates), 'variants_processed': len(diarization_variants)})
            
            if progress_callback:
                progress_callback("D", 75, f"ASR ensemble complete - {len(candidates)} candidates generated (5×5 matrix)")
            
            # Step 4.5: Dialect Processing (75-77%)
            dialect_processed_candidates = candidates  # Default to original candidates
            if self.enable_dialect_handling and self.dialect_engine:
                if progress_callback:
                    progress_callback("D2", 75.5, "Processing candidates for dialect variants...")
                
                stage_start_time = time.time()
                self.structured_logger.stage_start("dialect_processing", "Processing ASR candidates for dialect handling",
                                                 context={'candidates_count': len(candidates), 'supported_dialects': len(self.supported_dialects)})
                
                # Process each candidate through dialect handling
                dialect_processed_candidates = []
                total_adjustments = 0
                total_patterns_detected = 0
                
                for i, candidate in enumerate(candidates):
                    try:
                        # Extract ASR result from candidate
                        asr_data = candidate.get('asr_data', {})
                        
                        # Create simplified ASRResult for dialect processing
                        # We'll work with the aligned segments which contain the transcription
                        aligned_segments = candidate.get('aligned_segments', [])
                        
                        if aligned_segments:
                            # Process the candidate for dialect variants
                            dialect_result = self._process_candidate_for_dialect(candidate, aligned_segments)
                            
                            if dialect_result:
                                total_adjustments += len(dialect_result['confidence_adjustments'])
                                total_patterns_detected += len(dialect_result['dialect_patterns_detected'])
                                dialect_processed_candidates.append(dialect_result)
                            else:
                                dialect_processed_candidates.append(candidate)
                        else:
                            dialect_processed_candidates.append(candidate)
                    
                    except Exception as e:
                        self.structured_logger.warning(f"Dialect processing failed for candidate {i}: {e}")
                        dialect_processed_candidates.append(candidate)  # Keep original on error
                
                dialect_processing_time = time.time() - stage_start_time
                self.structured_logger.stage_complete("dialect_processing", "Dialect processing completed",
                                                    duration=dialect_processing_time,
                                                    metrics={
                                                        'candidates_processed': len(candidates),
                                                        'total_adjustments': total_adjustments,
                                                        'patterns_detected': total_patterns_detected,
                                                        'processing_time_per_candidate': dialect_processing_time / len(candidates) if candidates else 0
                                                    })
                
                if progress_callback:
                    progress_callback("D2", 77, f"Dialect processing complete - {total_adjustments} confidence adjustments made")
            
            # Use dialect-processed candidates for further processing
            candidates = dialect_processed_candidates
            
            # Step 5: Confidence Scoring (77-85%)
            if progress_callback:
                progress_callback("E", 78, "Scoring candidates across 5 confidence dimensions...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("confidence_scoring", "Scoring all candidates across confidence dimensions")
            
            scored_candidates = self.confidence_scorer.score_all_candidates(candidates)
            
            scoring_time = time.time() - stage_start_time
            self.structured_logger.stage_complete("confidence_scoring", "Candidate scoring completed", 
                                                duration=scoring_time,
                                                metrics={'candidates_scored': len(scored_candidates)})
            
            # Step 6: Consensus Processing (85-90%)
            if progress_callback:
                progress_callback("F", 87, f"Running consensus strategy: {self.consensus_strategy}...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("consensus_processing", "Processing consensus from scored candidates",
                                             context={'consensus_strategy': self.consensus_strategy, 'candidates_count': len(scored_candidates)})
            
            # Use consensus module for winner selection
            consensus_result = self.consensus_module.process_consensus(
                candidates=scored_candidates,
                strategy=self.consensus_strategy
            )
            
            winner = consensus_result.winner_candidate
            
            selection_time = time.time() - stage_start_time
            winner_score = winner.get('confidence_scores', {}).get('final_score', 0)
            self.structured_logger.stage_complete("consensus_processing", "Consensus processing completed", 
                                                duration=selection_time,
                                                metrics={
                                                    'winner_score': winner_score, 
                                                    'candidate_id': winner.get('candidate_id'),
                                                    'consensus_strategy': consensus_result.consensus_method,
                                                    'consensus_confidence': consensus_result.consensus_confidence
                                                })
            
            # U7 Step 6.5: Worklist Creation and Selective Reprocessing (87-90%)
            if progress_callback:
                progress_callback("F2", 87, "Creating segment worklist for quality improvement...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("worklist_creation", "Creating segment worklist and selective reprocessing")
            
            # Create worklist from candidates and winner
            try:
                worklist_entry = self.worklist_manager.create_worklist_from_candidates(
                    file_path=video_path,
                    run_id=self.run_id,
                    candidates=scored_candidates,
                    winner_candidate=winner
                )
                
                self.structured_logger.info("U7: Segment worklist created", 
                                          context={'flagged_segments': worklist_entry.total_segments_flagged,
                                                 'confidence_threshold': self.confidence_threshold_for_flagging})
                
                # Selective reprocessing if enabled and flagged segments exist
                if (self.enable_selective_reprocessing and 
                    worklist_entry.total_segments_flagged > 0 and 
                    worklist_entry.total_segments_flagged <= self.max_segments_for_selective_reprocessing):
                    
                    if progress_callback:
                        progress_callback("F3", 89, f"Reprocessing {worklist_entry.total_segments_flagged} flagged segments...")
                    
                    self.structured_logger.info("U7: Starting selective reprocessing", 
                                              context={'segments_to_reprocess': worklist_entry.total_segments_flagged})
                    
                    # Perform selective reprocessing
                    reprocess_results = self.selective_asr_processor.process_flagged_segments(
                        file_path=video_path,
                        run_id=self.run_id,
                        max_segments=self.max_segments_for_selective_reprocessing,
                        progress_callback=lambda msg, pct, detail: progress_callback("F3", 89 + int(pct * 0.01), detail) if progress_callback else None
                    )
                    
                    if 'error' not in reprocess_results:
                        self.structured_logger.info("U7: Selective reprocessing completed", 
                                                   context={'segments_improved': reprocess_results.get('segments_improved', 0),
                                                          'total_improvement': reprocess_results.get('total_improvement', 0.0)})
                    else:
                        self.structured_logger.warning("U7: Selective reprocessing failed", 
                                                     context={'error': reprocess_results['error']})
                else:
                    if worklist_entry.total_segments_flagged > self.max_segments_for_selective_reprocessing:
                        self.structured_logger.info("U7: Skipping selective reprocessing - too many flagged segments", 
                                                   context={'flagged': worklist_entry.total_segments_flagged, 
                                                          'max_allowed': self.max_segments_for_selective_reprocessing})
                    elif not self.enable_selective_reprocessing:
                        self.structured_logger.info("U7: Selective reprocessing disabled")
                    else:
                        self.structured_logger.info("U7: No segments flagged for reprocessing")
                
            except Exception as e:
                self.structured_logger.warning(f"U7: Worklist creation/selective reprocessing failed: {e}")
            
            worklist_time = time.time() - stage_start_time
            self.structured_logger.stage_complete("worklist_creation", "Worklist creation and selective reprocessing completed", 
                                                duration=worklist_time)
            
            # Step 7: Output Generation (90-100%)
            if progress_callback:
                progress_callback("G", 92, "Generating final outputs...")
            
            # Step 7.5: Post-Fusion Punctuation Processing (92-95%)
            if self.enable_post_fusion_punctuation and self.punctuation_engine:
                if progress_callback:
                    progress_callback("G1", 92, "Applying punctuation and disfluency normalization...")
                
                stage_start_time = time.time()
                self.structured_logger.stage_start("post_fusion_punctuation", "Applying post-fusion punctuation and disfluency normalization")
                
                try:
                    # Extract segments from winner for punctuation processing
                    winner_segments = winner.get('segments', [])
                    
                    # Apply post-fusion punctuation
                    punctuation_result = self.punctuation_engine.process_fused_segments(winner_segments)
                    
                    # Update winner with punctuated segments
                    punctuated_segments = []
                    for punctuated_seg in punctuation_result.segments:
                        segment_dict = {
                            'start': punctuated_seg.start_time,
                            'end': punctuated_seg.end_time,
                            'speaker': punctuated_seg.speaker_id or 'Unknown',
                            'text': punctuated_seg.punctuated_text,
                            'original_text': punctuated_seg.original_text,
                            'punctuation_confidence': punctuated_seg.punctuation_confidence,
                            'disfluency_normalization_applied': punctuated_seg.disfluency_normalization is not None,
                            'processing_metadata': punctuated_seg.processing_metadata
                        }
                        punctuated_segments.append(segment_dict)
                    
                    # Update winner candidate with punctuated segments
                    winner['segments'] = punctuated_segments
                    winner['punctuation_metadata'] = {
                        'overall_confidence': punctuation_result.overall_confidence,
                        'punctuation_metrics': punctuation_result.punctuation_metrics,
                        'disfluency_metrics': punctuation_result.disfluency_metrics,
                        'processing_time': punctuation_result.processing_time,
                        'model_info': punctuation_result.model_info,
                        'normalization_level': punctuation_result.normalization_level
                    }
                    
                    punctuation_time = time.time() - stage_start_time
                    self.structured_logger.stage_complete("post_fusion_punctuation", "Post-fusion punctuation completed successfully",
                                                        duration=punctuation_time,
                                                        metrics={
                                                            'segments_processed': len(punctuated_segments),
                                                            'overall_confidence': punctuation_result.overall_confidence,
                                                            'punctuation_changes': punctuation_result.punctuation_metrics.get('segments_changed', 0),
                                                            'disfluency_normalizations': punctuation_result.disfluency_metrics.get('segments_normalized', 0)
                                                        })
                    
                    if progress_callback:
                        progress_callback("G1", 94, f"Punctuation applied - {len(punctuated_segments)} segments processed")
                        
                except Exception as e:
                    self.structured_logger.error(f"Post-fusion punctuation failed: {e}")
                    # Continue with original segments on error
                    punctuation_time = time.time() - stage_start_time
                    self.structured_logger.stage_complete("post_fusion_punctuation", "Post-fusion punctuation failed, using original segments",
                                                        duration=punctuation_time,
                                                        context={'error': str(e)})
            
            # Format outputs (now using potentially punctuated segments)
            if progress_callback:
                progress_callback("G2", 95, "Generating final transcript formats...")
            
            winner_transcript_json = self._create_master_transcript(winner)
            winner_transcript_txt = self.transcript_formatter.create_txt_transcript(winner_transcript_json)
            
            # Generate subtitles
            captions_vtt = self.transcript_formatter.create_vtt_captions(winner_transcript_json)
            captions_srt = self.transcript_formatter.create_srt_captions(winner_transcript_json)
            captions_ass = self.transcript_formatter.create_ass_captions(winner_transcript_json)
            
            # Create ensemble audit
            ensemble_audit = self._create_ensemble_audit(scored_candidates, winner)
            
            # Step 8: Finalization (100%)
            if progress_callback:
                progress_callback("H", 100, "Processing complete!")
            
            # Log successful completion
            total_time = time.time() - start_time
            self.structured_logger.stage_complete("pipeline", "Ensemble transcription pipeline completed successfully", 
                                                duration=total_time,
                                                metrics={
                                                    'total_candidates': len(candidates),
                                                    'winner_score': winner_score,
                                                    'processing_time': total_time
                                                })
            
            processing_time = time.time() - start_time
            
            # Get comprehensive cost and performance summary
            cost_summary = self.structured_logger.get_session_cost_summary()
            system_metrics = self.structured_logger.get_system_metrics()
            
            # Compile results with enhanced observability data
            results = {
                'winner_transcript': winner_transcript_json,
                'winner_transcript_txt': winner_transcript_txt,
                'captions_vtt': captions_vtt,
                'captions_srt': captions_srt,
                'captions_ass': captions_ass,
                'ensemble_audit': ensemble_audit,
                'winner_score': winner['confidence_scores']['final_score'],
                'confidence_breakdown': winner['confidence_scores'],
                'detected_speakers': self._count_unique_speakers(winner),
                'processing_time': processing_time,
                'transcript_preview': self._create_transcript_preview(winner_transcript_json),
                'session_metadata': {
                    'audio_duration': audio_duration,
                    'estimated_noise_level': estimated_noise,
                    'expected_speakers': self.expected_speakers,
                    'candidates_generated': len(scored_candidates),
                'diarization_variants': len(diarization_variants),
                'voting_fusion_applied': any(v.get('fusion_applied', False) for v in diarization_variants),
                    'processing_timestamp': time.time()
                },
                # Enhanced observability data
                'cost_summary': cost_summary,
                'system_metrics': system_metrics,
                'observability_metadata': {
                    'run_id': self.run_id,
                    'session_id': self.structured_logger.session_id,
                    'service_name': 'ensemble-transcription',
                    'pipeline_stages': [
                        'audio_extraction', 'diarization', 'asr_ensemble', 
                        'confidence_scoring', 'winner_selection', 'output_generation'
                    ],
                    'instrumentation_enabled': True,
                    'cost_tracking_enabled': True,
                    'profiling_enabled': hasattr(self, 'obs_manager') and self.obs_manager.enable_profiling,
                    'total_cost_usd': cost_summary.get('total_cost_usd', 0.0),
                    'total_api_calls': cost_summary.get('total_api_calls', 0),
                    'peak_memory_mb': system_metrics.get('memory_rss_mb', 0)
                }
            }
            
            # Step 8: Track output artifacts and create run manifest (versioning)
            if self.enable_versioning and self.dvc_manager:
                if progress_callback:
                    progress_callback("H", 95, "Creating run manifest and tracking artifacts...")
                
                # Track output artifacts
                output_artifacts = self.dvc_manager.track_output_artifacts(results, self.run_id)
                self.output_artifacts.update(output_artifacts)
                
                # Create processing configuration for manifest (includes methodology audit trail)
                processing_config = {
                    'expected_speakers': self.expected_speakers,
                    'noise_level': self.noise_level,
                    'target_language': self.target_language,
                    'domain': self.domain,
                    'audio_duration': audio_duration,
                    'estimated_noise': estimated_noise,
                    'candidates_generated': len(scored_candidates),
                    'diarization_variants': len(diarization_variants),
                    'scoring_weights': self.confidence_scorer.score_weights,
                    # Methodology audit trail for A/B testing
                    'consensus_strategy': self.consensus_strategy,
                    'calibration_method': self.calibration_method,
                    'consensus_confidence': getattr(consensus_result, 'consensus_confidence', None) if 'consensus_result' in locals() else None,
                    'consensus_method_used': getattr(consensus_result, 'consensus_method', self.consensus_strategy) if 'consensus_result' in locals() else self.consensus_strategy
                }
                
                # Get metrics registry version
                registry_version = self.metrics_registry.get_registry_version() if self.metrics_registry else "v1.0"
                
                # Create comprehensive run manifest
                run_manifest = self.dvc_manager.create_run_manifest(
                    run_id=self.run_id,
                    input_artifacts=self.input_artifacts,
                    intermediate_artifacts=self.intermediate_artifacts,
                    output_artifacts=self.output_artifacts,
                    processing_config=processing_config,
                    processing_time=processing_time,
                    metrics_registry_version=registry_version
                )
                
                # Save run manifest
                manifest_path = self.dvc_manager.save_run_manifest(run_manifest)
                
                # Add versioning metadata to results
                results['versioning_metadata'] = {
                    'run_id': self.run_id,
                    'manifest_path': manifest_path,
                    'metrics_registry_version': registry_version,
                    'input_artifacts': len(self.input_artifacts),
                    'intermediate_artifacts': len(self.intermediate_artifacts),
                    'output_artifacts': len(self.output_artifacts),
                    'dvc_enabled': True
                }
                
                self.structured_logger.info("Run manifest created", 
                                           context={'run_id': self.run_id, 'manifest_path': manifest_path})
            
            # Generate comprehensive observability reports
            profiling_summary = self.profiling_manager.get_profiling_summary()
            report_files = self.observability_reporter.generate_comprehensive_report(results, profiling_summary)
            
            # Add report files to results
            results['observability_reports'] = report_files
            
            # Log final completion with observability metrics
            final_metrics = {
                'total_cost_usd': cost_summary.get('total_cost_usd', 0.0),
                'total_api_calls': cost_summary.get('total_api_calls', 0),
                'memory_peak_mb': system_metrics.get('memory_rss_mb', 0),
                'reports_generated': len(report_files),
                'observability_enabled': True
            }
            
            self.structured_logger.info(
                f"Observability reports generated: {len(report_files)} files, ${cost_summary.get('total_cost_usd', 0.0):.4f} total cost",
                metrics=final_metrics,
                report_files=list(report_files.keys())
            )
            
            # U7: Cache the complete processing result for future use
            if self.enable_caching:
                self.cache_manager.set("complete_ensemble_processing", results, video_path, processing_config)
                self.structured_logger.info("U7: Complete processing result cached for future reuse")
            
            # CRITICAL FIX: Persist outputs to files for download
            self._persist_output_files(results)
            
            return results
            
        except Exception as e:
            raise Exception(f"Ensemble processing failed: {str(e)}")
        
        finally:
            # Clean up temporary files
            self._cleanup_temp_files()
    
    def _create_master_transcript(self, winner: Dict[str, Any]) -> Dict[str, Any]:
        """Create master transcript JSON from winning candidate"""
        segments = winner.get('aligned_segments', [])
        
        # Build speaker map (for now, use speaker IDs as names)
        speaker_ids = set(seg['speaker_id'] for seg in segments)
        speaker_map = {speaker_id: f"Speaker {speaker_id}" for speaker_id in speaker_ids}
        
        # Format segments for master transcript
        master_segments = []
        for segment in segments:
            master_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'speaker': speaker_map.get(segment['speaker_id'], segment['speaker_id']),
                'speaker_id': segment['speaker_id'],
                'text': segment['text'],
                'words': segment.get('words', []),
                'confidence': segment.get('confidence', 0.0),
                'word_count': segment.get('word_count', 0)
            }
            master_segments.append(master_segment)
        
        # Create master transcript structure
        master_transcript = {
            'metadata': {
                'version': '1.0',
                'created_at': time.time(),
                'total_duration': max(seg['end'] for seg in segments) if segments else 0.0,
                'total_segments': len(segments),
                'speaker_count': len(speaker_ids),
                'confidence_summary': winner['confidence_scores']
            },
            'speaker_map': speaker_map,
            'segments': master_segments,
            'provenance': {
                'diarization_variant': winner['diarization_variant_id'],
                'asr_variant': winner['asr_variant_id'],
                'candidate_id': winner['candidate_id']
            }
        }
        
        return master_transcript
    
    def _create_ensemble_audit(self, scored_candidates: List[Dict[str, Any]], 
                             winner: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive ensemble audit report"""
        
        # Sort candidates by final score
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x['confidence_scores']['final_score'],
            reverse=True
        )
        
        # Top candidates summary
        top_candidates = []
        for i, candidate in enumerate(sorted_candidates[:5]):
            top_candidates.append({
                'rank': i + 1,
                'candidate_id': candidate['candidate_id'],
                'final_score': candidate['confidence_scores']['final_score'],
                'confidence_breakdown': candidate['confidence_scores'],
                'variant_info': f"Diar-{candidate['diarization_variant_id']}/ASR-{candidate['asr_variant_id']}"
            })
        
        # Score distribution analysis
        final_scores = [c['confidence_scores']['final_score'] for c in scored_candidates]
        
        # Per-dimension score analysis
        dimension_stats = {}
        for dim in ['D_diarization', 'A_asr_alignment', 'L_linguistic', 'R_agreement', 'O_overlap']:
            scores = [c['confidence_scores'][dim] for c in scored_candidates]
            dimension_stats[dim] = {
                'mean': float(sum(scores) / len(scores)),
                'min': float(min(scores)),
                'max': float(max(scores)),
                'std': float((sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5)
            }
        
        audit_report = {
            'summary': {
                'total_candidates': len(scored_candidates),
                'winner_candidate_id': winner['candidate_id'],
                'winner_score': winner['confidence_scores']['final_score'],
                'score_margin': winner['confidence_scores']['final_score'] - sorted_candidates[1]['confidence_scores']['final_score'],
                'mean_score': float(sum(final_scores) / len(final_scores)),
                'score_std': float((sum((x - sum(final_scores)/len(final_scores))**2 for x in final_scores) / len(final_scores))**0.5)
            },
            'top_candidates': top_candidates,
            'dimension_statistics': dimension_stats,
            'scoring_weights': self.confidence_scorer.score_weights,
            'quality_gates': {
                'timestamp_regression_threshold': '2.0%',
                'boundary_fit_threshold': '8.0%', 
                'overlap_plausibility_threshold': '0.30'
            },
            'session_credibility': winner['confidence_scores']['final_score']
        }
        
        return audit_report
    
    def _count_unique_speakers(self, winner: Dict[str, Any]) -> int:
        """Count unique speakers in winning transcript"""
        segments = winner.get('aligned_segments', [])
        speaker_ids = set(seg['speaker_id'] for seg in segments)
        return len(speaker_ids)
    
    def _create_transcript_preview(self, master_transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create preview of first 10 transcript segments"""
        segments = master_transcript.get('segments', [])
        preview = []
        
        for segment in segments[:10]:
            preview.append({
                'timestamp': self.transcript_formatter.format_timestamp(segment['start']),
                'speaker': segment['speaker'],
                'text': segment['text'][:200] + ('...' if len(segment['text']) > 200 else ''),
                'confidence': segment['confidence']
            })
        
        return preview
    
    def _cleanup_temp_files(self):
        """Clean up temporary files and directories"""
        # Clean up audio temp files
        if hasattr(self, 'temp_audio_files'):
            self.audio_processor.cleanup_temp_files(*self.temp_audio_files)
            self.temp_audio_files.clear()
        
        # Clean up work directory
        if self.work_dir and os.path.exists(self.work_dir):
            try:
                import shutil
                shutil.rmtree(self.work_dir)
                print(f"Cleaned up work directory: {self.work_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {e}")
            finally:
                self.work_dir = None
    
    def _is_mostly_silent(self, audio_path: str) -> bool:
        """
        Check if audio file is mostly silent or very low volume.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if audio appears mostly silent
        """
        try:
            import soundfile as sf
            import numpy as np
            
            # Read first 30 seconds for analysis
            audio_data, sr = sf.read(audio_path, frames=30 * 16000)  # 30 seconds at 16kHz
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Check if RMS is below silence threshold
            silence_threshold = 0.01  # Adjust based on testing
            
            return rms < silence_threshold
        except Exception as e:
            print(f"Warning: Could not analyze audio silence: {e}")
            return False
    
    def _persist_output_files(self, results: Dict[str, Any]) -> None:
        """Persist transcript outputs to artifacts directory for download"""
        try:
            import os
            from pathlib import Path
            
            # Create artifacts/reports directory
            reports_dir = Path("artifacts/reports")
            if self.run_id is None:
                self.structured_logger.warning("No run_id available for output persistence")
                return
            run_dir = reports_dir / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Write transcript files
            files_written = []
            
            # JSON transcript
            if 'winner_transcript' in results:
                json_path = run_dir / "transcript.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(results['winner_transcript'], f, indent=2, ensure_ascii=False)
                files_written.append(str(json_path))
            
            # TXT transcript  
            if 'winner_transcript_txt' in results:
                txt_path = run_dir / "transcript.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(results['winner_transcript_txt'])
                files_written.append(str(txt_path))
            
            # VTT captions
            if 'captions_vtt' in results:
                vtt_path = run_dir / "captions.vtt"
                with open(vtt_path, 'w', encoding='utf-8') as f:
                    f.write(results['captions_vtt'])
                files_written.append(str(vtt_path))
            
            # SRT captions
            if 'captions_srt' in results:
                srt_path = run_dir / "captions.srt"
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(results['captions_srt'])
                files_written.append(str(srt_path))
            
            # ASS captions
            if 'captions_ass' in results:
                ass_path = run_dir / "captions.ass"
                with open(ass_path, 'w', encoding='utf-8') as f:
                    f.write(results['captions_ass'])
                files_written.append(str(ass_path))
            
            # Ensemble audit report
            if 'ensemble_audit' in results:
                audit_path = run_dir / "ensemble_audit.json"
                with open(audit_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(results['ensemble_audit'], f, indent=2)
                files_written.append(str(audit_path))
            
            # Add file paths to results for UI display
            results['output_files'] = {
                'directory': str(run_dir),
                'files': files_written,
                'run_id': self.run_id
            }
            
            print(f"✅ Persisted {len(files_written)} output files to {run_dir}")
            self.structured_logger.info("Output files persisted", 
                                      context={'files_written': len(files_written), 'directory': str(run_dir)})
            
        except Exception as e:
            print(f"⚠️ Warning: Could not persist output files: {e}")
            self.structured_logger.warning(f"Failed to persist output files: {e}")
