"""
Observability and Metrics for Overlap-Aware Processing

This module provides comprehensive metrics collection, artifacts generation,
and performance tracking for the overlap-aware diarization and source separation system.

Key Features:
- Real-time metrics collection during processing
- Overlap processing performance tracking
- Artifact generation for analysis and debugging
- WER computation specifically for overlapped regions
- Integration with existing observability infrastructure

Author: Advanced Ensemble Transcription System
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import trace_stage, track_cost
from utils.metrics_registry import MetricsRegistryManager

@dataclass
class OverlapMetrics:
    """Core overlap processing metrics"""
    # Detection metrics
    overlap_ratio_percent: float = 0.0
    overlapped_frames_detected: int = 0
    total_audio_frames: int = 0
    overlap_detection_threshold: float = 0.0
    
    # Separation metrics
    stems_count: int = 0
    separation_runtime_ms: float = 0.0
    average_stem_snr_db: float = 0.0
    average_stem_leakage_score: float = 0.0
    separation_success_rate: float = 0.0
    
    # Quality gating metrics
    stem_snr_db: List[float] = field(default_factory=list)
    stem_leakage_rate: List[float] = field(default_factory=list)
    stem_artifact_scores: List[float] = field(default_factory=list)
    quality_gates_passed: bool = False
    fallback_reason: Optional[str] = None
    fallback_applied_count: int = 0
    
    # Processing metrics
    asr_runtime_ms_by_stem: Dict[str, float] = field(default_factory=dict)
    total_asr_runtime_ms: float = 0.0
    fusion_runtime_ms: float = 0.0
    diarization_runtime_ms: float = 0.0
    
    # Quality metrics
    reconciliation_conflict_rate: float = 0.0
    reconciliation_conflicts_total: int = 0
    total_words_processed: int = 0
    dual_speaker_regions_count: int = 0
    
    # Performance impact
    overlapped_region_wer: Optional[float] = None
    baseline_wer: Optional[float] = None
    wer_improvement_percent: Optional[float] = None
    
    # System metrics
    cache_hit_rate: float = 0.0
    processing_budget_utilization: float = 0.0

@dataclass 
class OverlapArtifacts:
    """Artifacts generated during overlap processing"""
    # Input artifacts
    original_audio_path: str
    overlap_frames: List[Dict[str, Any]] = field(default_factory=list)
    
    # Separation artifacts  
    stem_paths: List[str] = field(default_factory=list)
    stem_manifest_path: Optional[str] = None
    separation_quality_report: Dict[str, Any] = field(default_factory=dict)
    
    # Diarization artifacts
    per_stem_diarization_results: Dict[str, Any] = field(default_factory=dict)
    cross_stem_overlap_regions: List[Dict[str, Any]] = field(default_factory=list)
    unified_speaker_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Fusion artifacts
    overlap_fused_transcript_json: Dict[str, Any] = field(default_factory=dict)
    overlap_spans_metadata: List[Dict[str, Any]] = field(default_factory=list)
    reconciliation_report: Dict[str, Any] = field(default_factory=dict)
    
    # Debug artifacts
    processing_timeline: List[Dict[str, Any]] = field(default_factory=list)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    performance_profile: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OverlapProcessingReport:
    """Complete overlap processing report with metrics and artifacts"""
    processing_id: str
    timestamp: float
    duration_seconds: float
    
    metrics: OverlapMetrics
    artifacts: OverlapArtifacts
    
    # Configuration context
    processing_config: Dict[str, Any] = field(default_factory=dict)
    
    # Status and errors
    status: str = "completed"  # pending, processing, completed, failed
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class WERCalculator:
    """Calculate WER specifically for overlapped regions"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("overlap_wer_calculator")
    
    def calculate_overlap_wer(self, 
                            reference_transcript: List[Dict[str, Any]],
                            hypothesis_transcript: List[Dict[str, Any]],
                            overlap_regions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate WER specifically for overlapped regions
        
        Args:
            reference_transcript: Ground truth transcript segments
            hypothesis_transcript: Generated transcript segments
            overlap_regions: List of detected overlap regions
            
        Returns:
            Dictionary with WER metrics for overlapped regions
        """
        try:
            # Extract words from overlapped regions only
            overlap_ref_words = self._extract_overlap_words(reference_transcript, overlap_regions, 'reference')
            overlap_hyp_words = self._extract_overlap_words(hypothesis_transcript, overlap_regions, 'hypothesis')
            
            if not overlap_ref_words:
                return {'overlap_wer': 0.0, 'words_in_overlap': 0, 'errors_in_overlap': 0}
            
            # Calculate WER using edit distance
            wer_metrics = self._calculate_wer_metrics(overlap_ref_words, overlap_hyp_words)
            
            self.logger.info("Overlap WER calculated",
                           context={
                               'overlap_regions': len(overlap_regions),
                               'overlap_ref_words': len(overlap_ref_words),
                               'overlap_hyp_words': len(overlap_hyp_words),
                               'overlap_wer': wer_metrics['overlap_wer']
                           })
            
            return wer_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overlap WER: {e}")
            return {'overlap_wer': 1.0, 'words_in_overlap': 0, 'errors_in_overlap': 0}
    
    def _extract_overlap_words(self, 
                             transcript: List[Dict[str, Any]], 
                             overlap_regions: List[Dict[str, Any]],
                             source: str) -> List[str]:
        """Extract words that fall within overlap regions"""
        overlap_words = []
        
        for region in overlap_regions:
            region_start = region.get('start_time', 0)
            region_end = region.get('end_time', 0)
            
            # Find transcript segments that overlap with this region
            for segment in transcript:
                seg_start = segment.get('start', 0)
                seg_end = segment.get('end', 0)
                
                # Check for temporal overlap
                if seg_start < region_end and seg_end > region_start:
                    # Extract words from this segment
                    segment_text = segment.get('text', '')
                    if segment_text.strip():
                        words = segment_text.strip().split()
                        overlap_words.extend(words)
        
        return overlap_words
    
    def _calculate_wer_metrics(self, reference_words: List[str], hypothesis_words: List[str]) -> Dict[str, float]:
        """Calculate WER metrics using edit distance"""
        
        if not reference_words:
            return {'overlap_wer': 0.0, 'words_in_overlap': 0, 'errors_in_overlap': 0}
        
        # Calculate edit distance (simplified version)
        ref_len = len(reference_words)
        hyp_len = len(hypothesis_words)
        
        # Create matrix for dynamic programming
        dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
        
        # Initialize first row and column
        for i in range(ref_len + 1):
            dp[i][0] = i  # Deletions
        for j in range(hyp_len + 1):
            dp[0][j] = j  # Insertions
        
        # Fill the matrix
        for i in range(1, ref_len + 1):
            for j in range(1, hyp_len + 1):
                if reference_words[i-1].lower() == hypothesis_words[j-1].lower():
                    dp[i][j] = dp[i-1][j-1]  # Match
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Deletion
                        dp[i][j-1],    # Insertion
                        dp[i-1][j-1]   # Substitution
                    )
        
        # Calculate metrics
        edit_distance = dp[ref_len][hyp_len]
        wer = edit_distance / ref_len if ref_len > 0 else 0.0
        
        return {
            'overlap_wer': wer,
            'words_in_overlap': ref_len,
            'errors_in_overlap': edit_distance,
            'reference_words': ref_len,
            'hypothesis_words': hyp_len
        }

class OverlapObservabilityManager:
    """Main manager for overlap processing observability"""
    
    def __init__(self,
                 enable_artifacts: bool = True,
                 enable_wer_calculation: bool = True,
                 artifacts_base_path: str = "/tmp/overlap_artifacts"):
        """
        Initialize overlap observability manager
        
        Args:
            enable_artifacts: Enable artifact generation and storage
            enable_wer_calculation: Enable WER calculation for overlapped regions
            artifacts_base_path: Base directory for storing artifacts
        """
        self.enable_artifacts = enable_artifacts
        self.enable_wer_calculation = enable_wer_calculation
        self.artifacts_base_path = Path(artifacts_base_path)
        self.artifacts_base_path.mkdir(exist_ok=True)
        
        self.logger = create_enhanced_logger("overlap_observability_manager")
        
        # Initialize components
        self.wer_calculator = WERCalculator() if enable_wer_calculation else None
        
        # Metrics registry integration
        try:
            self.metrics_registry = MetricsRegistryManager()
        except Exception:
            self.metrics_registry = None
        
        # Processing state
        self.active_reports: Dict[str, OverlapProcessingReport] = {}
    
    def start_overlap_processing_observation(self, 
                                           processing_id: str,
                                           config: Dict[str, Any]) -> OverlapProcessingReport:
        """
        Start observing an overlap processing session
        
        Args:
            processing_id: Unique identifier for this processing session
            config: Processing configuration
            
        Returns:
            Initialized processing report
        """
        report = OverlapProcessingReport(
            processing_id=processing_id,
            timestamp=time.time(),
            duration_seconds=0.0,
            metrics=OverlapMetrics(),
            artifacts=OverlapArtifacts(),
            processing_config=config,
            status="processing"
        )
        
        self.active_reports[processing_id] = report
        
        self.logger.info(f"Started overlap processing observation",
                        context={'processing_id': processing_id, 'config': config})
        
        return report
    
    @trace_stage("overlap_metrics_collection")
    def collect_separation_metrics(self,
                                 processing_id: str,
                                 stems: List[Any],  # List[SeparatedStem]
                                 separation_time: float,
                                 stem_manifest: Optional[Any] = None) -> None:  # StemManifest
        """
        Collect metrics from source separation stage
        
        Args:
            processing_id: Processing session ID
            stems: List of separated stems
            separation_time: Time taken for separation
            stem_manifest: Optional stem manifest with quality data
        """
        if processing_id not in self.active_reports:
            return
        
        report = self.active_reports[processing_id]
        
        # Update separation metrics
        report.metrics.stems_count = len(stems)
        report.metrics.separation_runtime_ms = separation_time * 1000
        
        if stem_manifest:
            report.metrics.average_stem_snr_db = stem_manifest.average_snr_db
            report.metrics.average_stem_leakage_score = stem_manifest.average_leakage_score  
            report.metrics.separation_success_rate = stem_manifest.separation_success_rate
        
        # Store artifacts
        if self.enable_artifacts:
            report.artifacts.stem_paths = [stem.stem_path for stem in stems]
            if stem_manifest:
                report.artifacts.stem_manifest_path = stem_manifest.manifest_file_path
                report.artifacts.separation_quality_report = {
                    'average_snr_db': stem_manifest.average_snr_db,
                    'average_leakage_score': stem_manifest.average_leakage_score,
                    'separation_success_rate': stem_manifest.separation_success_rate,
                    'total_stems': stem_manifest.total_stems
                }
        
        self.logger.info("Separation metrics collected",
                        context={
                            'processing_id': processing_id,
                            'stems_count': len(stems),
                            'separation_time_ms': separation_time * 1000
                        })
    
    def collect_diarization_metrics(self,
                                  processing_id: str,
                                  diarization_result: Any,  # OverlapDiarizationResult
                                  processing_time: float) -> None:
        """
        Collect metrics from overlap diarization stage
        
        Args:
            processing_id: Processing session ID
            diarization_result: Result from overlap diarization
            processing_time: Time taken for diarization
        """
        if processing_id not in self.active_reports:
            return
        
        report = self.active_reports[processing_id]
        
        # Update diarization metrics
        report.metrics.diarization_runtime_ms = processing_time * 1000
        
        # Store artifacts
        if self.enable_artifacts:
            report.artifacts.per_stem_diarization_results = {
                'total_stems': len(diarization_result.stem_results),
                'cross_stem_overlaps': len(diarization_result.cross_stem_overlaps),
                'unified_speakers': len(set(diarization_result.unified_speaker_map.values())),
                'processing_metrics': diarization_result.processing_metrics
            }
            
            report.artifacts.cross_stem_overlap_regions = [
                {
                    'start_time': overlap.start_time,
                    'end_time': overlap.end_time,
                    'duration': overlap.duration,
                    'stems_involved': overlap.stems_involved,
                    'speakers_involved': overlap.speakers_involved,
                    'overlap_confidence': overlap.overlap_confidence
                }
                for overlap in diarization_result.cross_stem_overlaps
            ]
            
            report.artifacts.unified_speaker_mapping = diarization_result.unified_speaker_map
        
        self.logger.info("Diarization metrics collected",
                        context={
                            'processing_id': processing_id,
                            'diarization_time_ms': processing_time * 1000,
                            'cross_stem_overlaps': len(diarization_result.cross_stem_overlaps)
                        })
    
    def collect_fusion_metrics(self,
                             processing_id: str,
                             fusion_result: Any,  # OverlapFusionResult
                             processing_time: float) -> None:
        """
        Collect metrics from overlap fusion stage
        
        Args:
            processing_id: Processing session ID
            fusion_result: Result from overlap fusion
            processing_time: Time taken for fusion
        """
        if processing_id not in self.active_reports:
            return
        
        report = self.active_reports[processing_id]
        
        # Update fusion metrics
        report.metrics.fusion_runtime_ms = processing_time * 1000
        report.metrics.reconciliation_conflicts_total = fusion_result.reconciliation_conflicts
        report.metrics.total_words_processed = fusion_result.total_words_processed
        report.metrics.dual_speaker_regions_count = fusion_result.dual_speaker_regions_count
        
        if fusion_result.total_words_processed > 0:
            report.metrics.reconciliation_conflict_rate = (
                fusion_result.reconciliation_conflicts / fusion_result.total_words_processed
            )
        
        # Store artifacts
        if self.enable_artifacts:
            report.artifacts.overlap_fused_transcript_json = {
                'unified_transcript': fusion_result.unified_transcript,
                'fused_segments': [asdict(seg) for seg in fusion_result.fused_segments],
                'explicit_overlap_regions': [asdict(region) for region in fusion_result.explicit_overlap_regions],
                'processing_metrics': {
                    'total_words_processed': fusion_result.total_words_processed,
                    'reconciliation_conflicts': fusion_result.reconciliation_conflicts,
                    'overlap_regions_count': fusion_result.overlap_regions_count,
                    'dual_speaker_regions_count': fusion_result.dual_speaker_regions_count,
                    'overall_confidence': fusion_result.overall_confidence,
                    'fusion_confidence': fusion_result.fusion_confidence,
                    'reconciliation_conflict_rate': fusion_result.reconciliation_conflict_rate
                }
            }
            
            report.artifacts.overlap_spans_metadata = [
                {
                    'start_time': region.start_time,
                    'end_time': region.end_time,
                    'primary_speaker': region.primary_speaker,
                    'secondary_speaker': region.secondary_speaker,
                    'overlap_confidence': region.overlap_confidence,
                    'reconciliation_confidence': region.reconciliation_confidence,
                    'conflict_resolution_method': region.conflict_resolution_method
                }
                for region in fusion_result.explicit_overlap_regions
            ]
        
        self.logger.info("Fusion metrics collected",
                        context={
                            'processing_id': processing_id,
                            'fusion_time_ms': processing_time * 1000,
                            'reconciliation_conflicts': fusion_result.reconciliation_conflicts,
                            'conflict_rate': report.metrics.reconciliation_conflict_rate
                        })
    
    def calculate_overlap_wer(self,
                            processing_id: str,
                            reference_transcript: Optional[List[Dict[str, Any]]] = None,
                            hypothesis_transcript: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, float]]:
        """
        Calculate WER for overlapped regions
        
        Args:
            processing_id: Processing session ID
            reference_transcript: Ground truth transcript (if available)
            hypothesis_transcript: Generated transcript (if available)
            
        Returns:
            WER metrics for overlapped regions or None if calculation not possible
        """
        if not self.enable_wer_calculation or not self.wer_calculator:
            return None
        
        if processing_id not in self.active_reports:
            return None
        
        report = self.active_reports[processing_id]
        
        try:
            # Use provided transcripts or try to extract from artifacts
            if not reference_transcript or not hypothesis_transcript:
                self.logger.warning("Reference or hypothesis transcript not provided for WER calculation")
                return None
            
            # Extract overlap regions from artifacts
            overlap_regions = report.artifacts.overlap_spans_metadata
            
            if not overlap_regions:
                self.logger.info("No overlap regions found for WER calculation")
                return {'overlap_wer': 0.0, 'words_in_overlap': 0, 'errors_in_overlap': 0}
            
            # Calculate WER
            wer_metrics = self.wer_calculator.calculate_overlap_wer(
                reference_transcript, hypothesis_transcript, overlap_regions
            )
            
            # Update report metrics
            report.metrics.overlapped_region_wer = wer_metrics.get('overlap_wer', 1.0)
            
            self.logger.info("Overlap WER calculated",
                           context={
                               'processing_id': processing_id,
                               'overlap_wer': wer_metrics.get('overlap_wer', 1.0),
                               'words_in_overlap': wer_metrics.get('words_in_overlap', 0)
                           })
            
            return wer_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overlap WER for {processing_id}: {e}")
            return None
    
    def finalize_processing_report(self,
                                 processing_id: str,
                                 status: str = "completed",
                                 errors: Optional[List[str]] = None,
                                 warnings: Optional[List[str]] = None) -> Optional[OverlapProcessingReport]:
        """
        Finalize overlap processing report and optionally persist artifacts
        
        Args:
            processing_id: Processing session ID
            status: Final processing status
            errors: List of errors encountered
            warnings: List of warnings encountered
            
        Returns:
            Finalized processing report or None if processing_id not found
        """
        if processing_id not in self.active_reports:
            return None
        
        report = self.active_reports[processing_id]
        
        # Update final status
        report.duration_seconds = time.time() - report.timestamp
        report.status = status
        report.errors = errors or []
        report.warnings = warnings or []
        
        # Calculate derived metrics
        self._calculate_derived_metrics(report)
        
        # Persist artifacts if enabled
        if self.enable_artifacts:
            self._persist_artifacts(report)
        
        # Submit to metrics registry if available
        if self.metrics_registry:
            try:
                self._submit_to_metrics_registry(report)
            except Exception as e:
                self.logger.warning(f"Failed to submit metrics to registry: {e}")
        
        self.logger.info("Processing report finalized",
                        context={
                            'processing_id': processing_id,
                            'status': status,
                            'duration_seconds': report.duration_seconds,
                            'errors': len(report.errors),
                            'warnings': len(report.warnings)
                        })
        
        # Remove from active reports
        finalized_report = self.active_reports.pop(processing_id)
        
        return finalized_report
    
    def _calculate_derived_metrics(self, report: OverlapProcessingReport) -> None:
        """Calculate derived metrics from collected data"""
        
        # Calculate overlap ratio if we have frame data
        if report.artifacts.overlap_frames:
            total_overlap_duration = sum(
                frame.get('duration', 0) for frame in report.artifacts.overlap_frames
            )
            if report.processing_config.get('audio_duration', 0) > 0:
                audio_duration = report.processing_config['audio_duration']
                report.metrics.overlap_ratio_percent = (total_overlap_duration / audio_duration) * 100
        
        # Calculate total ASR runtime
        report.metrics.total_asr_runtime_ms = sum(report.metrics.asr_runtime_ms_by_stem.values())
        
        # Calculate processing budget utilization
        max_processing_time = report.processing_config.get('max_overlap_processing_duration', 600.0)
        actual_processing_time = report.duration_seconds
        report.metrics.processing_budget_utilization = min(1.0, actual_processing_time / max_processing_time)
    
    def _persist_artifacts(self, report: OverlapProcessingReport) -> None:
        """Persist artifacts to disk"""
        try:
            # Create processing-specific directory
            processing_dir = self.artifacts_base_path / report.processing_id
            processing_dir.mkdir(exist_ok=True)
            
            # Save main report
            report_path = processing_dir / "overlap_processing_report.json"
            with open(report_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            # Save overlap spans separately for easy access
            spans_path = processing_dir / "overlap_spans.json"
            with open(spans_path, 'w') as f:
                json.dump(report.artifacts.overlap_spans_metadata, f, indent=2)
            
            # Save reconciliation report
            recon_path = processing_dir / "reconciliation_report.json"
            with open(recon_path, 'w') as f:
                json.dump(report.artifacts.reconciliation_report, f, indent=2)
            
            self.logger.info(f"Artifacts persisted to {processing_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist artifacts: {e}")
    
    def _submit_to_metrics_registry(self, report: OverlapProcessingReport) -> None:
        """Submit metrics to the registry system"""
        
        if not self.metrics_registry:
            return
        
        # Prepare metrics for registry
        metrics_data = {
            'overlap_processing_metrics': {
                'processing_id': report.processing_id,
                'timestamp': report.timestamp,
                'overlap_ratio_percent': report.metrics.overlap_ratio_percent,
                'stems_count': report.metrics.stems_count,
                'separation_runtime_ms': report.metrics.separation_runtime_ms,
                'total_asr_runtime_ms': report.metrics.total_asr_runtime_ms,
                'reconciliation_conflict_rate': report.metrics.reconciliation_conflict_rate,
                'overlapped_region_wer': report.metrics.overlapped_region_wer,
                'dual_speaker_regions_count': report.metrics.dual_speaker_regions_count,
                'processing_budget_utilization': report.metrics.processing_budget_utilization,
                'status': report.status
            }
        }
        
        try:
            # This would integrate with the existing metrics registry
            # The exact method would depend on the MetricsRegistryManager interface
            pass  # Placeholder for actual registry submission
        except Exception as e:
            self.logger.warning(f"Failed to submit to metrics registry: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of current processing state"""
        
        return {
            'active_reports': len(self.active_reports),
            'artifacts_enabled': self.enable_artifacts,
            'wer_calculation_enabled': self.enable_wer_calculation,
            'artifacts_base_path': str(self.artifacts_base_path),
            'active_processing_ids': list(self.active_reports.keys())
        }