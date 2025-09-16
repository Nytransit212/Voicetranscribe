"""
Manifest integrity system for verifiable provenance of ensemble transcription artifacts.

This module provides cryptographic verification and dependency tracking for all 
processing outputs, ensuring tamper-proof audit trails and reproducible results.
"""

import os
import json
import hashlib
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field, validator, model_validator
from utils.enhanced_structured_logger import create_enhanced_logger


class ArtifactEntry(BaseModel):
    """Single artifact entry in the manifest with full provenance tracking"""
    artifact_type: str = Field(..., description="Type of artifact (asr_wav, stem_audio, chunk_json, fused_transcript_json, srt, vtt, subtitled_mp4)")
    path: str = Field(..., description="Relative path within session directory")
    bytes: int = Field(..., ge=0, description="File size in bytes")
    sha256: str = Field(..., min_length=64, max_length=64, description="SHA256 hash of file contents")
    producing_component: str = Field(..., description="Component that created this artifact or cache_restore@{cache_key}")
    inputs: List[str] = Field(default_factory=list, description="List of input artifact SHA256s this depends on")
    created_at: str = Field(..., description="ISO 8601 timestamp when artifact was created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the artifact")

    @validator('artifact_type')
    def validate_artifact_type(cls, v):
        """Validate artifact type is supported"""
        allowed_types = {
            'input_mp4', 'asr_wav', 'stem_audio', 'chunk_json', 
            'fused_transcript_json', 'srt', 'vtt', 'subtitled_mp4',
            'diarization_json', 'asr_result_json', 'confidence_scores_json',
            'ensemble_audit_json', 'punctuation_json', 'normalization_json',
            'speaker_mapping_json', 'captions_ass', 'source_separation_manifest'
        }
        if v not in allowed_types:
            raise ValueError(f"Unsupported artifact type: {v}. Must be one of {allowed_types}")
        return v

    @validator('sha256')
    def validate_sha256_format(cls, v):
        """Validate SHA256 hash format"""
        if not all(c in '0123456789abcdef' for c in v.lower()):
            raise ValueError("SHA256 must be hexadecimal characters only")
        return v.lower()

    @validator('path')
    def validate_path_format(cls, v):
        """Validate path is relative and safe"""
        if os.path.isabs(v):
            raise ValueError("Path must be relative")
        if '..' in v or v.startswith('/'):
            raise ValueError("Path must not contain directory traversal")
        return v


class ConfigSnapshot(BaseModel):
    """Configuration snapshot for reproducibility"""
    config_id: str = Field(..., description="Unique identifier for this config")
    created_at: str = Field(..., description="When this config was captured")
    processing_config: Dict[str, Any] = Field(..., description="Complete processing configuration")
    environment_info: Dict[str, Any] = Field(..., description="Environment and dependency versions")
    config_sha256: str = Field(..., description="SHA256 of the config content for verification")


class ModelVersionFingerprint(BaseModel):
    """Model version tracking for reproducibility"""
    asr_models: Dict[str, str] = Field(..., description="ASR model names and versions")
    diarization_models: Dict[str, str] = Field(..., description="Diarization model names and versions")
    punctuation_models: Dict[str, str] = Field(..., description="Punctuation model names and versions")
    source_separation_models: Dict[str, str] = Field(..., description="Source separation model names")
    fingerprint_sha256: str = Field(..., description="SHA256 of combined model fingerprint")


class RunManifest(BaseModel):
    """Complete run manifest with cryptographic verification"""
    manifest_version: str = Field(default="1.0", description="Manifest schema version")
    run_id: str = Field(..., description="Unique identifier for this processing run")
    session_id: str = Field(..., description="Session identifier")
    project_id: str = Field(..., description="Project identifier")
    
    # Input verification
    media_sha256: str = Field(..., description="SHA256 of original input media file")
    media_path: str = Field(..., description="Original media file path")
    media_bytes: int = Field(..., ge=0, description="Original media file size")
    
    # Configuration and model tracking
    config_snapshot: ConfigSnapshot = Field(..., description="Complete configuration snapshot")
    model_fingerprint: ModelVersionFingerprint = Field(..., description="Model version tracking")
    
    # Processing metadata
    started_at: str = Field(..., description="When processing started (ISO 8601)")
    completed_at: Optional[str] = Field(None, description="When processing completed (ISO 8601)")
    processing_duration_seconds: Optional[float] = Field(None, ge=0, description="Total processing time")
    
    # Artifacts and dependencies
    artifacts: List[ArtifactEntry] = Field(default_factory=list, description="All artifacts produced")
    
    # Validation state
    last_validated_at: Optional[str] = Field(None, description="When manifest was last validated")
    validation_passed: Optional[bool] = Field(None, description="Result of last validation")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    
    # Metrics
    total_artifacts: int = Field(default=0, ge=0, description="Total number of artifacts")
    total_bytes: int = Field(default=0, ge=0, description="Total bytes of all artifacts")
    manifest_sha256: Optional[str] = Field(None, description="SHA256 of this manifest file itself")

    @model_validator(mode='after')
    def validate_timing(self):
        """Validate timing consistency"""
        if self.completed_at and self.started_at and self.processing_duration_seconds:
            start_time = datetime.fromisoformat(self.started_at.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(self.completed_at.replace('Z', '+00:00'))
            calculated_duration = (end_time - start_time).total_seconds()
            
            # Allow 10% tolerance for timing accuracy
            if abs(calculated_duration - self.processing_duration_seconds) > self.processing_duration_seconds * 0.1:
                raise ValueError("Processing duration doesn't match start/end times")
        
        return self


class ManifestManager:
    """Manages manifest creation, validation, and integrity verification"""
    
    def __init__(self, session_dir: str, session_id: str, project_id: str, run_id: str):
        """Initialize manifest manager for a specific run"""
        self.session_dir = Path(session_dir)
        self.session_id = session_id
        self.project_id = project_id
        self.run_id = run_id
        self.logger = create_enhanced_logger("manifest_manager", session_id=session_id, run_id=run_id)
        
        # Thread safety for concurrent artifact registration
        self._lock = threading.RLock()
        self._artifacts: Dict[str, ArtifactEntry] = {}
        self._manifest: Optional[RunManifest] = None
        
        # File paths
        self.manifest_path = self.session_dir / "run_manifest.json"
        
        # Initialize manifest
        self._initialize_manifest()
    
    def _initialize_manifest(self):
        """Initialize or load existing manifest"""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                self._manifest = RunManifest(**manifest_data)
                
                # Rebuild artifacts dict for fast lookups
                for artifact in self._manifest.artifacts:
                    self._artifacts[artifact.sha256] = artifact
                    
                self.logger.info(f"Loaded existing manifest with {len(self._artifacts)} artifacts")
            except Exception as e:
                self.logger.error(f"Failed to load existing manifest: {e}")
                self._create_new_manifest()
        else:
            self._create_new_manifest()
    
    def _create_new_manifest(self):
        """Create new manifest with basic structure"""
        self._manifest = RunManifest(
            run_id=self.run_id,
            session_id=self.session_id,
            project_id=self.project_id,
            media_sha256="",  # Will be set when media is processed
            media_path="",
            media_bytes=0,
            config_snapshot=ConfigSnapshot(
                config_id=f"config_{int(time.time())}",
                created_at=datetime.now(timezone.utc).isoformat(),
                processing_config={},
                environment_info={},
                config_sha256=""
            ),
            model_fingerprint=ModelVersionFingerprint(
                asr_models={},
                diarization_models={},
                punctuation_models={},
                source_separation_models={},
                fingerprint_sha256=""
            ),
            started_at=datetime.now(timezone.utc).isoformat()
        )
        self.logger.info(f"Created new manifest for run {self.run_id}")
    
    def set_input_media(self, media_path: str, config: Dict[str, Any], model_versions: Dict[str, Dict[str, str]]):
        """Set input media information and configuration snapshot"""
        try:
            # Compute media hash and size
            media_sha256, media_bytes = self._compute_file_hash_and_size(media_path)
            
            # Create config snapshot
            config_content = json.dumps(config, sort_keys=True)
            config_sha256 = hashlib.sha256(config_content.encode()).hexdigest()
            
            # Create model fingerprint
            model_content = json.dumps(model_versions, sort_keys=True)
            model_sha256 = hashlib.sha256(model_content.encode()).hexdigest()
            
            with self._lock:
                self._manifest.media_sha256 = media_sha256
                self._manifest.media_path = os.path.relpath(media_path, self.session_dir)
                self._manifest.media_bytes = media_bytes
                
                self._manifest.config_snapshot = ConfigSnapshot(
                    config_id=f"config_{int(time.time())}",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    processing_config=config,
                    environment_info=self._get_environment_info(),
                    config_sha256=config_sha256
                )
                
                self._manifest.model_fingerprint = ModelVersionFingerprint(
                    asr_models=model_versions.get('asr', {}),
                    diarization_models=model_versions.get('diarization', {}),
                    punctuation_models=model_versions.get('punctuation', {}),
                    source_separation_models=model_versions.get('source_separation', {}),
                    fingerprint_sha256=model_sha256
                )
            
            self._save_manifest()
            self.logger.info(f"Set input media: {media_path} ({media_bytes} bytes, sha256: {media_sha256[:12]}...)")
            
        except Exception as e:
            self.logger.error(f"Failed to set input media: {e}")
            raise
    
    def add_artifact(self, 
                    artifact_type: str, 
                    file_path: str, 
                    producing_component: str,
                    input_artifacts: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add artifact to manifest with automatic hash computation.
        
        Args:
            artifact_type: Type of artifact being added
            file_path: Absolute or relative path to the artifact file
            producing_component: Component that created this artifact
            input_artifacts: List of SHA256s of input artifacts this depends on
            metadata: Additional metadata for the artifact
            
        Returns:
            SHA256 hash of the added artifact
        """
        try:
            # Resolve absolute path
            if not os.path.isabs(file_path):
                abs_path = self.session_dir / file_path
            else:
                abs_path = Path(file_path)
            
            if not abs_path.exists():
                raise FileNotFoundError(f"Artifact file not found: {abs_path}")
            
            # Compute hash and size
            sha256_hash, file_bytes = self._compute_file_hash_and_size(str(abs_path))
            
            # Create relative path for storage
            try:
                relative_path = os.path.relpath(abs_path, self.session_dir)
            except ValueError:
                # Files outside session directory - store absolute path with warning
                relative_path = str(abs_path)
                self.logger.warning(f"Artifact outside session directory: {abs_path}")
            
            # Create artifact entry
            artifact = ArtifactEntry(
                artifact_type=artifact_type,
                path=relative_path,
                bytes=file_bytes,
                sha256=sha256_hash,
                producing_component=producing_component,
                inputs=input_artifacts or [],
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata=metadata or {}
            )
            
            # Add to manifest with thread safety
            with self._lock:
                self._artifacts[sha256_hash] = artifact
                self._manifest.artifacts = list(self._artifacts.values())
                self._manifest.total_artifacts = len(self._artifacts)
                self._manifest.total_bytes = sum(a.bytes for a in self._artifacts.values())
            
            self._save_manifest()
            
            self.logger.info(
                f"Added artifact: {artifact_type} ({file_bytes} bytes, sha256: {sha256_hash[:12]}...) "
                f"from {producing_component}"
            )
            
            return sha256_hash
            
        except Exception as e:
            self.logger.error(f"Failed to add artifact {artifact_type}: {e}")
            raise
    
    def add_cached_artifact(self, artifact_type: str, file_path: str, cache_key: str, 
                          input_artifacts: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add artifact that was restored from cache"""
        cache_key_short = cache_key[:12] if len(cache_key) > 12 else cache_key
        producing_component = f"cache_restore@{cache_key_short}"
        
        return self.add_artifact(
            artifact_type=artifact_type,
            file_path=file_path,
            producing_component=producing_component,
            input_artifacts=input_artifacts,
            metadata=metadata
        )
    
    def mark_completed(self):
        """Mark processing as completed and finalize manifest"""
        try:
            completion_time = datetime.now(timezone.utc).isoformat()
            start_time = datetime.fromisoformat(self._manifest.started_at.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(completion_time.replace('Z', '+00:00'))
            duration = (end_time - start_time).total_seconds()
            
            with self._lock:
                self._manifest.completed_at = completion_time
                self._manifest.processing_duration_seconds = duration
            
            self._save_manifest()
            
            # Compute manifest file hash
            manifest_sha256, _ = self._compute_file_hash_and_size(str(self.manifest_path))
            self._manifest.manifest_sha256 = manifest_sha256
            self._save_manifest()
            
            self.logger.info(
                f"Marked run as completed: {len(self._artifacts)} artifacts, "
                f"{self._manifest.total_bytes} bytes, {duration:.1f}s duration"
            )
            
            # Log completion event for telemetry
            self.logger.log_event(
                "manifest_complete",
                {
                    "artifact_count": self._manifest.total_artifacts,
                    "total_bytes": self._manifest.total_bytes,
                    "duration_seconds": duration,
                    "manifest_sha256": manifest_sha256
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to mark completion: {e}")
            raise
    
    def validate(self, recompute_hashes: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate manifest integrity by checking all artifacts.
        
        Args:
            recompute_hashes: Whether to recompute file hashes (slower but more thorough)
            
        Returns:
            Tuple of (validation_passed, list_of_errors)
        """
        errors = []
        validation_start = time.time()
        
        try:
            self.logger.info(f"Starting manifest validation (recompute_hashes={recompute_hashes})")
            
            # Check manifest exists and is parseable
            if not self.manifest_path.exists():
                errors.append("Manifest file does not exist")
                return False, errors
            
            # Validate input media if specified
            if self._manifest.media_sha256 and self._manifest.media_path:
                media_path = self.session_dir / self._manifest.media_path
                if not media_path.exists():
                    errors.append(f"Input media file missing: {media_path}")
                elif recompute_hashes:
                    computed_hash, computed_bytes = self._compute_file_hash_and_size(str(media_path))
                    if computed_hash != self._manifest.media_sha256:
                        errors.append(f"Input media hash mismatch: expected {self._manifest.media_sha256}, got {computed_hash}")
                    if computed_bytes != self._manifest.media_bytes:
                        errors.append(f"Input media size mismatch: expected {self._manifest.media_bytes}, got {computed_bytes}")
            
            # Validate all artifacts
            artifact_hashes = set()
            validated_count = 0
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                if recompute_hashes:
                    # Parallel hash computation for performance
                    future_to_artifact = {}
                    for artifact in self._manifest.artifacts:
                        future = executor.submit(self._validate_single_artifact, artifact)
                        future_to_artifact[future] = artifact
                    
                    for future in future_to_artifact:
                        artifact = future_to_artifact[future]
                        try:
                            validation_errors = future.result()
                            errors.extend(validation_errors)
                            if not validation_errors:
                                validated_count += 1
                        except Exception as e:
                            errors.append(f"Failed to validate artifact {artifact.path}: {e}")
                else:
                    # Just check file existence
                    for artifact in self._manifest.artifacts:
                        artifact_path = self.session_dir / artifact.path
                        if not artifact_path.exists():
                            errors.append(f"Artifact file missing: {artifact.path}")
                        else:
                            validated_count += 1
                
                artifact_hashes.update(a.sha256 for a in self._manifest.artifacts)
            
            # Validate dependency DAG
            dag_errors = self._validate_dependency_dag(artifact_hashes)
            errors.extend(dag_errors)
            
            # Check for duplicate SHA256s
            if len(artifact_hashes) != len(self._manifest.artifacts):
                errors.append("Duplicate SHA256 hashes found in manifest")
            
            # Validate manifest metadata consistency
            if self._manifest.total_artifacts != len(self._manifest.artifacts):
                errors.append(f"Artifact count mismatch: metadata says {self._manifest.total_artifacts}, actual {len(self._manifest.artifacts)}")
            
            computed_total_bytes = sum(a.bytes for a in self._manifest.artifacts)
            if self._manifest.total_bytes != computed_total_bytes:
                errors.append(f"Total bytes mismatch: metadata says {self._manifest.total_bytes}, computed {computed_total_bytes}")
            
            # Update validation state
            validation_passed = len(errors) == 0
            validation_time = time.time() - validation_start
            
            with self._lock:
                self._manifest.last_validated_at = datetime.now(timezone.utc).isoformat()
                self._manifest.validation_passed = validation_passed
                self._manifest.validation_errors = errors.copy()
            
            self._save_manifest()
            
            if validation_passed:
                self.logger.info(f"Manifest validation PASSED: {validated_count} artifacts verified in {validation_time:.1f}s")
            else:
                self.logger.error(f"Manifest validation FAILED: {len(errors)} errors found in {validation_time:.1f}s")
                for error in errors:
                    self.logger.error(f"  - {error}")
            
            # Log validation metrics
            self.logger.log_metrics({
                "manifest_validation_duration": validation_time,
                "manifest_validation_errors": len(errors),
                "manifest_artifacts_validated": validated_count
            })
            
            return validation_passed, errors
            
        except Exception as e:
            error_msg = f"Validation failed with exception: {e}"
            self.logger.error(error_msg)
            return False, [error_msg]
    
    def _validate_single_artifact(self, artifact: ArtifactEntry) -> List[str]:
        """Validate a single artifact (for parallel execution)"""
        errors = []
        
        artifact_path = self.session_dir / artifact.path
        
        # Check file exists
        if not artifact_path.exists():
            errors.append(f"Artifact file missing: {artifact.path}")
            return errors
        
        # Recompute hash and size
        try:
            computed_hash, computed_bytes = self._compute_file_hash_and_size(str(artifact_path))
            
            if computed_hash != artifact.sha256:
                errors.append(f"Hash mismatch for {artifact.path}: expected {artifact.sha256}, got {computed_hash}")
            
            if computed_bytes != artifact.bytes:
                errors.append(f"Size mismatch for {artifact.path}: expected {artifact.bytes}, got {computed_bytes}")
                
        except Exception as e:
            errors.append(f"Failed to validate {artifact.path}: {e}")
        
        return errors
    
    def _validate_dependency_dag(self, available_hashes: Set[str]) -> List[str]:
        """Validate that all dependencies form a valid DAG"""
        errors = []
        
        # Check all input dependencies exist
        for artifact in self._manifest.artifacts:
            for input_hash in artifact.inputs:
                if input_hash not in available_hashes:
                    errors.append(f"Missing dependency for {artifact.path}: {input_hash[:12]}...")
        
        # Check for cycles using DFS
        try:
            visited = set()
            rec_stack = set()
            
            def has_cycle(artifact_sha256: str) -> bool:
                if artifact_sha256 in rec_stack:
                    return True
                if artifact_sha256 in visited:
                    return False
                
                visited.add(artifact_sha256)
                rec_stack.add(artifact_sha256)
                
                # Find artifact by SHA256
                artifact = next((a for a in self._manifest.artifacts if a.sha256 == artifact_sha256), None)
                if artifact:
                    for input_hash in artifact.inputs:
                        if has_cycle(input_hash):
                            return True
                
                rec_stack.remove(artifact_sha256)
                return False
            
            for artifact in self._manifest.artifacts:
                if artifact.sha256 not in visited:
                    if has_cycle(artifact.sha256):
                        errors.append("Circular dependency detected in artifact DAG")
                        break
                        
        except Exception as e:
            errors.append(f"Failed to validate DAG: {e}")
        
        return errors
    
    def get_artifacts_by_type(self, artifact_type: str) -> List[ArtifactEntry]:
        """Get all artifacts of a specific type"""
        return [a for a in self._manifest.artifacts if a.artifact_type == artifact_type]
    
    def get_artifact_by_sha256(self, sha256: str) -> Optional[ArtifactEntry]:
        """Get artifact by its SHA256 hash"""
        return self._artifacts.get(sha256)
    
    def get_dependency_tree(self, artifact_sha256: str) -> Dict[str, Any]:
        """Get complete dependency tree for an artifact"""
        artifact = self.get_artifact_by_sha256(artifact_sha256)
        if not artifact:
            return {}
        
        def build_tree(sha256: str, visited: Set[str]) -> Dict[str, Any]:
            if sha256 in visited:
                return {"cycle_detected": True}
            
            visited.add(sha256)
            artifact = self.get_artifact_by_sha256(sha256)
            if not artifact:
                return {"missing": True}
            
            tree = {
                "artifact_type": artifact.artifact_type,
                "path": artifact.path,
                "bytes": artifact.bytes,
                "producing_component": artifact.producing_component,
                "created_at": artifact.created_at,
                "dependencies": {}
            }
            
            for input_hash in artifact.inputs:
                tree["dependencies"][input_hash[:12]] = build_tree(input_hash, visited.copy())
            
            return tree
        
        return build_tree(artifact_sha256, set())
    
    def export_summary(self) -> Dict[str, Any]:
        """Export manifest summary for reporting"""
        return {
            "run_id": self._manifest.run_id,
            "session_id": self._manifest.session_id,
            "project_id": self._manifest.project_id,
            "started_at": self._manifest.started_at,
            "completed_at": self._manifest.completed_at,
            "duration_seconds": self._manifest.processing_duration_seconds,
            "total_artifacts": self._manifest.total_artifacts,
            "total_bytes": self._manifest.total_bytes,
            "artifacts_by_type": {
                artifact_type: len([a for a in self._manifest.artifacts if a.artifact_type == artifact_type])
                for artifact_type in set(a.artifact_type for a in self._manifest.artifacts)
            },
            "validation_status": {
                "last_validated": self._manifest.last_validated_at,
                "passed": self._manifest.validation_passed,
                "error_count": len(self._manifest.validation_errors)
            },
            "manifest_sha256": self._manifest.manifest_sha256
        }
    
    def _compute_file_hash_and_size(self, file_path: str) -> Tuple[str, int]:
        """Compute SHA256 hash and file size efficiently"""
        hasher = hashlib.sha256()
        size = 0
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):  # 64KB chunks
                hasher.update(chunk)
                size += len(chunk)
        
        return hasher.hexdigest(), size
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Capture environment information for reproducibility"""
        import sys
        import platform
        
        try:
            import torch
            torch_version = torch.__version__
        except ImportError:
            torch_version = "not_installed"
        
        try:
            import streamlit
            streamlit_version = streamlit.__version__
        except ImportError:
            streamlit_version = "not_installed"
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch_version,
            "streamlit_version": streamlit_version,
            "environment_captured_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _save_manifest(self):
        """Save manifest to file with atomic write"""
        try:
            # Sort artifacts by creation time for deterministic output
            sorted_artifacts = sorted(self._manifest.artifacts, key=lambda a: a.created_at)
            self._manifest.artifacts = sorted_artifacts
            
            # Atomic write using temp file
            temp_path = self.manifest_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(
                    self._manifest.dict(),
                    f,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False
                )
            
            # Atomic move
            temp_path.replace(self.manifest_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save manifest: {e}")
            raise


def create_manifest_manager(session_dir: str, session_id: str, project_id: str, run_id: str) -> ManifestManager:
    """Factory function to create a manifest manager"""
    return ManifestManager(session_dir, session_id, project_id, run_id)


def verify_manifest_integrity(manifest_path: str, recompute_hashes: bool = True) -> Tuple[bool, List[str]]:
    """
    Standalone function to verify manifest integrity.
    
    Args:
        manifest_path: Path to run_manifest.json file
        recompute_hashes: Whether to recompute all file hashes
        
    Returns:
        Tuple of (validation_passed, list_of_errors)
    """
    try:
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            return False, [f"Manifest file not found: {manifest_path}"]
        
        session_dir = manifest_file.parent
        
        # Load manifest
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
        
        manifest = RunManifest(**manifest_data)
        
        # Create temporary manager for validation
        manager = ManifestManager(
            str(session_dir), 
            manifest.session_id, 
            manifest.project_id, 
            manifest.run_id
        )
        manager._manifest = manifest
        manager._artifacts = {a.sha256: a for a in manifest.artifacts}
        
        return manager.validate(recompute_hashes=recompute_hashes)
        
    except Exception as e:
        return False, [f"Failed to verify manifest: {e}"]