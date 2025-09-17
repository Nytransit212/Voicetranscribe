"""
Atomic Stage Completion Manager for Crash-Safe Resume

This module provides crash-safe stage completion tracking for long-running
ensemble transcription jobs. Each major processing stage atomically writes
completion markers with full validation data to enable resuming from any
completed stage after a crash.

Key Features:
- Atomic stage completion markers with SHA256 validation
- Configuration and model fingerprint tracking
- Manifest integration for artifact dependencies
- Resume detection and validation on startup
- Pipeline skip logic based on completed stages
"""

import os
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.atomic_io import get_atomic_io_manager, TempDirectoryScope
from utils.manifest import create_manifest_manager
from core.run_context import get_global_run_context


class ProcessingStage(Enum):
    """Processing stages for crash-safe resume"""
    AUDIO_EXTRACTION = "audio_extraction"
    DIARIZATION = "diarization"
    OVERLAP_PROCESSING = "overlap_processing"
    ASR_PROCESSING = "asr_processing"
    CONSENSUS = "consensus"
    OUTPUT_GENERATION = "output_generation"
    

@dataclass
class StageCompletionMarker:
    """Atomic stage completion marker with full validation data"""
    stage: ProcessingStage
    run_id: str
    session_id: str
    project_id: str
    completed_at: str  # ISO 8601 timestamp
    inputs_sha256: List[str]  # SHA256 hashes of all input artifacts
    config_snapshot_id: str  # Configuration snapshot identifier
    model_fingerprint_sha256: str  # Model version fingerprint
    stage_outputs: List[Dict[str, Any]]  # Output artifacts from this stage
    stage_metadata: Dict[str, Any]  # Stage-specific processing metadata
    processing_duration_seconds: float
    next_stage: Optional[ProcessingStage] = None  # Next stage to process
    
    
class StageCompletionError(Exception):
    """Raised when stage completion operations fail"""
    def __init__(self, message: str, stage: ProcessingStage, original_error: Optional[Exception] = None):
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"Stage completion error for {stage.value}: {message}")


class StageCompletionManager:
    """
    Manages atomic stage completion markers for crash-safe resume functionality.
    
    Provides atomic operations to mark stages as complete, detect incomplete runs
    on startup, and validate resume conditions.
    """
    
    def __init__(self, base_work_dir: Optional[str] = None, enable_telemetry: bool = True):
        """
        Initialize stage completion manager.
        
        Args:
            base_work_dir: Base working directory for stage markers
            enable_telemetry: Whether to enable telemetry tracking
        """
        self.base_work_dir = Path(base_work_dir or "/tmp/ensemble_stage_completion")
        self.base_work_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_telemetry = enable_telemetry
        
        # Get managers
        self.atomic_io = get_atomic_io_manager()
        # Manifest manager will be set when run context is available
        self.manifest_manager = None
        
        # Logger
        self.logger = create_enhanced_logger("stage_completion_manager")
        
        # Completed stages cache
        self._completed_stages_cache: Dict[str, Set[ProcessingStage]] = {}
        
        self.logger.info(f"Initialized stage completion manager at {self.base_work_dir}")
    
    def set_manifest_manager(self, session_dir: str, session_id: str, project_id: str, run_id: str):
        """
        Set manifest manager once run context is available.
        
        Args:
            session_dir: Session directory path
            session_id: Session identifier
            project_id: Project identifier  
            run_id: Run identifier
        """
        try:
            self.manifest_manager = create_manifest_manager(session_dir, session_id, project_id, run_id)
            self.logger.info("Manifest manager initialized for stage completion tracking")
        except Exception as e:
            self.logger.warning(f"Failed to initialize manifest manager: {e}")
            self.manifest_manager = None
    
    def _get_run_completion_dir(self, run_id: str) -> Path:
        """Get the completion directory for a specific run"""
        return self.base_work_dir / run_id
    
    def _get_stage_marker_path(self, run_id: str, stage: ProcessingStage) -> Path:
        """Get the path for a stage completion marker file"""
        run_dir = self._get_run_completion_dir(run_id)
        return run_dir / f"stage_complete_{stage.value}.json"
    
    def _compute_config_snapshot_id(self, config: Dict[str, Any]) -> str:
        """Compute deterministic ID for configuration snapshot"""
        config_json = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:16]
    
    def _compute_model_fingerprint(self, model_versions: Dict[str, str]) -> str:
        """Compute SHA256 fingerprint for model versions"""
        fingerprint_data = json.dumps(model_versions, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(fingerprint_data.encode('utf-8')).hexdigest()
    
    def mark_stage_complete(self, 
                           stage: ProcessingStage,
                           run_id: str,
                           session_id: str,
                           project_id: str,
                           inputs_sha256: List[str],
                           config_snapshot: Dict[str, Any],
                           model_versions: Dict[str, str],
                           stage_outputs: List[Dict[str, Any]],
                           stage_metadata: Dict[str, Any],
                           processing_duration: float) -> StageCompletionMarker:
        """
        Atomically mark a processing stage as complete with full validation data.
        
        Args:
            stage: Processing stage that was completed
            run_id: Unique run identifier
            session_id: Session identifier  
            project_id: Project identifier
            inputs_sha256: SHA256 hashes of all input artifacts for this stage
            config_snapshot: Complete configuration snapshot
            model_versions: Model version information
            stage_outputs: Output artifacts produced by this stage
            stage_metadata: Stage-specific metadata
            processing_duration: Duration in seconds for this stage
            
        Returns:
            StageCompletionMarker object
            
        Raises:
            StageCompletionError: If marking stage completion fails
        """
        try:
            # Create completion directory
            run_completion_dir = self._get_run_completion_dir(run_id)
            run_completion_dir.mkdir(parents=True, exist_ok=True)
            
            # Compute fingerprints
            config_snapshot_id = self._compute_config_snapshot_id(config_snapshot)
            model_fingerprint_sha256 = self._compute_model_fingerprint(model_versions)
            
            # Determine next stage
            stage_order = list(ProcessingStage)
            current_index = stage_order.index(stage)
            next_stage = stage_order[current_index + 1] if current_index < len(stage_order) - 1 else None
            
            # Create completion marker
            marker = StageCompletionMarker(
                stage=stage,
                run_id=run_id,
                session_id=session_id,
                project_id=project_id,
                completed_at=datetime.now(timezone.utc).isoformat(),
                inputs_sha256=inputs_sha256,
                config_snapshot_id=config_snapshot_id,
                model_fingerprint_sha256=model_fingerprint_sha256,
                stage_outputs=stage_outputs,
                stage_metadata=stage_metadata,
                processing_duration_seconds=processing_duration,
                next_stage=next_stage
            )
            
            # Write marker atomically
            marker_path = self._get_stage_marker_path(run_id, stage)
            marker_data = {
                'stage': stage.value,
                'run_id': run_id,
                'session_id': session_id,
                'project_id': project_id,
                'completed_at': marker.completed_at,
                'inputs_sha256': inputs_sha256,
                'config_snapshot_id': config_snapshot_id,
                'model_fingerprint_sha256': model_fingerprint_sha256,
                'stage_outputs': stage_outputs,
                'stage_metadata': stage_metadata,
                'processing_duration_seconds': processing_duration,
                'next_stage': next_stage.value if next_stage else None,
                'marker_version': '1.0'
            }
            
            with self.atomic_io.atomic_write(marker_path, mode='w') as f:
                json.dump(marker_data, f, indent=2)
            
            # Update completed stages cache
            if run_id not in self._completed_stages_cache:
                self._completed_stages_cache[run_id] = set()
            self._completed_stages_cache[run_id].add(stage)
            
            # Track in manifest if available
            if self.manifest_manager:
                try:
                    self.manifest_manager.add_artifact(
                        artifact_type="stage_complete_marker",
                        file_path=str(marker_path),
                        producing_component=f"StageCompletionManager.mark_stage_complete",
                        input_artifacts=inputs_sha256,
                        metadata={
                            "stage": stage.value,
                            "run_id": run_id,
                            "config_snapshot_id": config_snapshot_id,
                            "model_fingerprint": model_fingerprint_sha256,
                            "processing_duration": processing_duration,
                            "next_stage": next_stage.value if next_stage else None
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to track stage marker in manifest: {e}")
            
            self.logger.info(f"Stage {stage.value} marked complete", 
                           context={
                               'run_id': run_id,
                               'processing_duration': processing_duration,
                               'outputs_count': len(stage_outputs),
                               'next_stage': next_stage.value if next_stage else 'COMPLETE',
                               'config_snapshot_id': config_snapshot_id
                           })
            
            return marker
            
        except Exception as e:
            error_msg = f"Failed to mark stage {stage.value} complete: {e}"
            self.logger.error(error_msg)
            raise StageCompletionError(error_msg, stage, e)
    
    def get_completed_stages(self, run_id: str) -> Set[ProcessingStage]:
        """
        Get set of completed stages for a run.
        
        Args:
            run_id: Run identifier to check
            
        Returns:
            Set of completed ProcessingStage enum values
        """
        # Check cache first
        if run_id in self._completed_stages_cache:
            return self._completed_stages_cache[run_id].copy()
        
        # Scan completion directory
        completed_stages = set()
        run_completion_dir = self._get_run_completion_dir(run_id)
        
        if run_completion_dir.exists():
            for stage in ProcessingStage:
                marker_path = self._get_stage_marker_path(run_id, stage)
                if marker_path.exists():
                    try:
                        # Validate marker file
                        with open(marker_path, 'r') as f:
                            marker_data = json.load(f)
                        
                        # Basic validation
                        if (marker_data.get('run_id') == run_id and 
                            marker_data.get('stage') == stage.value):
                            completed_stages.add(stage)
                    except Exception as e:
                        self.logger.warning(f"Invalid stage marker {marker_path}: {e}")
        
        # Cache results
        self._completed_stages_cache[run_id] = completed_stages
        return completed_stages.copy()
    
    def get_stage_completion_marker(self, run_id: str, stage: ProcessingStage) -> Optional[StageCompletionMarker]:
        """
        Get completion marker for a specific stage.
        
        Args:
            run_id: Run identifier
            stage: Processing stage
            
        Returns:
            StageCompletionMarker if found, None otherwise
        """
        marker_path = self._get_stage_marker_path(run_id, stage)
        
        if not marker_path.exists():
            return None
        
        try:
            with open(marker_path, 'r') as f:
                marker_data = json.load(f)
            
            # Convert to marker object
            next_stage = None
            if marker_data.get('next_stage'):
                next_stage = ProcessingStage(marker_data['next_stage'])
            
            return StageCompletionMarker(
                stage=ProcessingStage(marker_data['stage']),
                run_id=marker_data['run_id'],
                session_id=marker_data['session_id'],
                project_id=marker_data['project_id'],
                completed_at=marker_data['completed_at'],
                inputs_sha256=marker_data['inputs_sha256'],
                config_snapshot_id=marker_data['config_snapshot_id'],
                model_fingerprint_sha256=marker_data['model_fingerprint_sha256'],
                stage_outputs=marker_data['stage_outputs'],
                stage_metadata=marker_data['stage_metadata'],
                processing_duration_seconds=marker_data['processing_duration_seconds'],
                next_stage=next_stage
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load stage marker {marker_path}: {e}")
            return None
    
    def detect_incomplete_runs(self) -> List[Dict[str, Any]]:
        """
        Detect incomplete runs that can be resumed.
        
        Returns:
            List of incomplete run information dictionaries
        """
        incomplete_runs = []
        
        if not self.base_work_dir.exists():
            return incomplete_runs
        
        for run_dir in self.base_work_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            run_id = run_dir.name
            completed_stages = self.get_completed_stages(run_id)
            
            # Check if run is incomplete (not all stages completed)
            all_stages = set(ProcessingStage)
            if completed_stages and completed_stages != all_stages:
                # Get the most recent completed stage
                stage_order = list(ProcessingStage)
                last_completed_stage = None
                last_completion_time = None
                
                for stage in completed_stages:
                    marker = self.get_stage_completion_marker(run_id, stage)
                    if marker:
                        stage_index = stage_order.index(stage)
                        if (last_completed_stage is None or 
                            stage_index > stage_order.index(last_completed_stage)):
                            last_completed_stage = stage
                            last_completion_time = marker.completed_at
                
                if last_completed_stage:
                    marker = self.get_stage_completion_marker(run_id, last_completed_stage)
                    incomplete_runs.append({
                        'run_id': run_id,
                        'session_id': marker.session_id if marker else 'unknown',
                        'project_id': marker.project_id if marker else 'unknown',
                        'completed_stages': [stage.value for stage in completed_stages],
                        'last_completed_stage': last_completed_stage.value,
                        'next_stage': marker.next_stage.value if marker and marker.next_stage else 'COMPLETE',
                        'last_completion_time': last_completion_time,
                        'total_stages': len(all_stages),
                        'progress_percent': (len(completed_stages) / len(all_stages)) * 100
                    })
        
        # Sort by last completion time (most recent first)
        incomplete_runs.sort(key=lambda x: x['last_completion_time'], reverse=True)
        
        self.logger.info(f"Detected {len(incomplete_runs)} incomplete runs that can be resumed")
        return incomplete_runs
    
    def validate_resume_conditions(self, 
                                 run_id: str,
                                 current_config: Dict[str, Any],
                                 current_model_versions: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate that a run can be safely resumed with current configuration.
        
        Args:
            run_id: Run to validate
            current_config: Current processing configuration
            current_model_versions: Current model versions
            
        Returns:
            Tuple of (can_resume: bool, validation_errors: List[str])
        """
        validation_errors = []
        
        completed_stages = self.get_completed_stages(run_id)
        if not completed_stages:
            validation_errors.append("No completed stages found")
            return False, validation_errors
        
        # Validate configuration and model consistency
        current_config_id = self._compute_config_snapshot_id(current_config)
        current_model_fingerprint = self._compute_model_fingerprint(current_model_versions)
        
        for stage in completed_stages:
            marker = self.get_stage_completion_marker(run_id, stage)
            if not marker:
                validation_errors.append(f"Missing completion marker for {stage.value}")
                continue
            
            # Check configuration consistency
            if marker.config_snapshot_id != current_config_id:
                validation_errors.append(
                    f"Configuration changed since {stage.value} completion "
                    f"(was {marker.config_snapshot_id}, now {current_config_id})"
                )
            
            # Check model version consistency  
            if marker.model_fingerprint_sha256 != current_model_fingerprint:
                validation_errors.append(
                    f"Model versions changed since {stage.value} completion "
                    f"(was {marker.model_fingerprint_sha256[:8]}..., now {current_model_fingerprint[:8]}...)"
                )
        
        # Validate artifact integrity if manifest available
        if self.manifest_manager:
            try:
                for stage in completed_stages:
                    marker = self.get_stage_completion_marker(run_id, stage)
                    if marker:
                        # Verify all input artifacts still exist and have correct checksums
                        for artifact_sha256 in marker.inputs_sha256:
                            artifact = self.manifest_manager.get_artifact_by_sha256(artifact_sha256)
                            if not artifact:
                                validation_errors.append(f"Missing input artifact {artifact_sha256[:8]} for {stage.value}")
                            elif not self.manifest_manager.verify_artifact_integrity(artifact):
                                validation_errors.append(f"Corrupted input artifact {artifact_sha256[:8]} for {stage.value}")
            except Exception as e:
                validation_errors.append(f"Artifact integrity check failed: {e}")
        
        can_resume = len(validation_errors) == 0
        return can_resume, validation_errors
    
    def get_next_stage_to_process(self, run_id: str) -> Optional[ProcessingStage]:
        """
        Get the next stage that needs to be processed for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Next ProcessingStage to process, or None if run is complete
        """
        completed_stages = self.get_completed_stages(run_id)
        stage_order = list(ProcessingStage)
        
        for stage in stage_order:
            if stage not in completed_stages:
                return stage
        
        return None  # All stages completed
    
    def cleanup_run_markers(self, run_id: str):
        """
        Clean up all stage markers for a completed run.
        
        Args:
            run_id: Run identifier to clean up
        """
        try:
            run_completion_dir = self._get_run_completion_dir(run_id)
            if run_completion_dir.exists():
                import shutil
                shutil.rmtree(run_completion_dir)
                
                # Clear from cache
                self._completed_stages_cache.pop(run_id, None)
                
                self.logger.info(f"Cleaned up stage markers for run {run_id}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup run markers for {run_id}: {e}")


# Global stage completion manager instance
_stage_completion_manager = None

def get_stage_completion_manager(base_work_dir: Optional[str] = None) -> StageCompletionManager:
    """Get global stage completion manager instance"""
    global _stage_completion_manager
    if _stage_completion_manager is None:
        _stage_completion_manager = StageCompletionManager(base_work_dir)
    return _stage_completion_manager