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
from utils.transcript_formatter import TranscriptFormatter
from utils.structured_logger import StructuredLogger
from utils.dvc_versioning import DVCVersioningManager
from utils.metrics_registry import MetricsRegistryManager

class EnsembleManager:
    """Orchestrates the entire ensemble transcription pipeline"""
    
    def __init__(self, expected_speakers: int = 10, noise_level: str = 'medium', target_language: Optional[str] = None, scoring_weights: Optional[Dict[str, float]] = None, enable_versioning: bool = True, domain: str = "general") -> None:
        self.expected_speakers = expected_speakers
        self.noise_level = noise_level
        self.target_language = target_language  # None for auto-detect
        self.domain = domain
        self.enable_versioning = enable_versioning
        
        # Generate unique run ID for this processing session
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Initialize structured logging
        self.structured_logger = StructuredLogger("ensemble_manager")
        
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
        self.diarization_engine = DiarizationEngine(expected_speakers, noise_level)
        self.asr_engine = ASREngine()
        self.confidence_scorer = ConfidenceScorer(
            scoring_weights=scoring_weights,
            use_registry_calibration=self.enable_versioning,
            domain=domain,
            speaker_count=expected_speakers,
            noise_level=noise_level
        )
        self.transcript_formatter = TranscriptFormatter()
        
        # Working directory for temporary files
        self.work_dir: Optional[str] = None
        self.temp_audio_files: List[str] = []  # Track temp audio files for cleanup
        
        # Artifact tracking for run manifest
        self.input_artifacts: Dict[str, Any] = {}
        self.intermediate_artifacts: Dict[str, Any] = {}
        self.output_artifacts: Dict[str, Any] = {}
    
    def process_video(self, video_path: str, progress_callback: Optional[Callable[[str, int, str], None]] = None) -> Dict[str, Any]:
        """
        Process video through complete ensemble pipeline.
        
        Args:
            video_path: Path to input MP4 video file
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
            
            diarization_variants = self.diarization_engine.create_diarization_variants(clean_audio_path, use_voting_fusion=True)
            
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
            
            # Step 4: ASR Ensemble (35-75%)
            if progress_callback:
                progress_callback("D", 40, "Running expanded ASR ensemble (5 passes per diarization)...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("asr_ensemble", "Running ASR ensemble across all diarization variants",
                                             context={'diarization_variants': len(diarization_variants), 'target_language': self.target_language})
            
            candidates = self.asr_engine.run_asr_ensemble(clean_audio_path, diarization_variants, target_language=self.target_language)
            
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
            
            # Step 5: Confidence Scoring (75-85%)
            if progress_callback:
                progress_callback("E", 78, "Scoring candidates across 5 confidence dimensions...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("confidence_scoring", "Scoring all candidates across confidence dimensions")
            
            scored_candidates = self.confidence_scorer.score_all_candidates(candidates)
            
            scoring_time = time.time() - stage_start_time
            self.structured_logger.stage_complete("confidence_scoring", "Candidate scoring completed", 
                                                duration=scoring_time,
                                                metrics={'candidates_scored': len(scored_candidates)})
            
            # Step 6: Winner Selection (85-90%)
            if progress_callback:
                progress_callback("F", 87, "Selecting winning transcript...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("winner_selection", "Selecting best transcript from scored candidates")
            
            winner = self.confidence_scorer.select_winner(scored_candidates)
            
            selection_time = time.time() - stage_start_time
            winner_score = winner.get('composite_score', 0)
            self.structured_logger.stage_complete("winner_selection", "Winner transcript selected", 
                                                duration=selection_time,
                                                metrics={'winner_score': winner_score, 'candidate_id': winner.get('candidate_id')})
            
            # Step 7: Output Generation (90-100%)
            if progress_callback:
                progress_callback("G", 92, "Generating final outputs...")
            
            # Format outputs
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
            
            # Compile results
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
                }
            }
            
            # Step 8: Track output artifacts and create run manifest (versioning)
            if self.enable_versioning and self.dvc_manager:
                if progress_callback:
                    progress_callback("H", 95, "Creating run manifest and tracking artifacts...")
                
                # Track output artifacts
                output_artifacts = self.dvc_manager.track_output_artifacts(results, self.run_id)
                self.output_artifacts.update(output_artifacts)
                
                # Create processing configuration for manifest
                processing_config = {
                    'expected_speakers': self.expected_speakers,
                    'noise_level': self.noise_level,
                    'target_language': self.target_language,
                    'domain': self.domain,
                    'audio_duration': audio_duration,
                    'estimated_noise': estimated_noise,
                    'candidates_generated': len(scored_candidates),
                    'diarization_variants': len(diarization_variants),
                    'scoring_weights': self.confidence_scorer.score_weights
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
