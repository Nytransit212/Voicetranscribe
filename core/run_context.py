"""
RunContext Implementation for Deterministic Processing

This module provides the core RunContext object that ensures identical outputs
for identical inputs and configurations by managing deterministic seeding,
configuration snapshots, and model version fingerprinting.
"""

import os
import json
import time
import uuid
import hashlib
import random
import numpy as np
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

from utils.enhanced_structured_logger import create_enhanced_logger

# Global seeding constants for deterministic processing
STAGE_SEED_OFFSETS = {
    'asr': 101,
    'diarization': 201, 
    'fusion': 301,
    'separation': 401,
    'normalization': 501,
    'punctuation': 601,
    'speaker_mapping': 701,
    'consensus': 801,
    'quality_control': 901
}

@dataclass
class RunContext:
    """
    Core context object for deterministic ensemble processing.
    
    Ensures reproducible outputs by tracking:
    - Unique run identification
    - Deterministic seeding strategy
    - Configuration snapshots  
    - Model version fingerprinting
    - Processing metadata
    """
    
    # Core identifiers
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default="")
    project_id: str = field(default="")
    
    # Timing
    start_time: float = field(default_factory=time.monotonic)
    start_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Deterministic seeding
    global_seed: int = field(default=0)
    media_sha256: str = field(default="")
    config_snapshot_id: str = field(default="")
    model_version_fingerprint: str = field(default="")
    
    # Stage seeds (computed from global_seed + offsets)
    stage_seeds: Dict[str, int] = field(default_factory=dict)
    
    # Configuration and model tracking
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    resolved_model_versions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Processing metadata
    determinism_enabled: bool = field(default=True)
    resume_mode: bool = field(default=False)
    cache_deterministic: bool = field(default=True)
    
    # Telemetry
    logger: Any = field(init=False)
    telemetry_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize logging and computed fields"""
        self.logger = create_enhanced_logger("run_context", run_id=self.run_id)
        self.telemetry_data = {
            'run_id': self.run_id,
            'start_timestamp': self.start_timestamp,
            'determinism_enabled': self.determinism_enabled
        }
    
    @classmethod
    def create_deterministic_context(cls,
                                   media_sha256: str,
                                   config_dict: Dict[str, Any],
                                   model_versions: Dict[str, Dict[str, str]],
                                   session_id: Optional[str] = None,
                                   project_id: Optional[str] = None,
                                   resume_mode: bool = False) -> 'RunContext':
        """
        Create a RunContext with deterministic seeding from inputs.
        
        Args:
            media_sha256: SHA256 hash of input media file
            config_dict: Complete configuration dictionary
            model_versions: Resolved model versions by category
            session_id: Optional session identifier
            project_id: Optional project identifier
            resume_mode: Whether this is resuming an existing run
            
        Returns:
            RunContext instance with deterministic seeding
        """
        context = cls(
            session_id=session_id or f"session_{int(time.time())}",
            project_id=project_id or "default_project",
            media_sha256=media_sha256,
            resume_mode=resume_mode
        )
        
        # Compute configuration snapshot
        context._compute_config_snapshot(config_dict)
        
        # Compute model version fingerprint
        context._compute_model_fingerprint(model_versions)
        
        # Generate deterministic global seed
        context._generate_global_seed()
        
        # Compute stage-specific seeds
        context._compute_stage_seeds()
        
        # Log determinism initialization
        context._log_determinism_init()
        
        return context
    
    def _compute_config_snapshot(self, config_dict: Dict[str, Any]):
        """Compute deterministic configuration snapshot"""
        try:
            # Create deterministic JSON representation
            config_json = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
            self.config_snapshot_id = hashlib.sha256(config_json.encode('utf-8')).hexdigest()
            self.config_snapshot = config_dict.copy()
            
            self.logger.info(f"Configuration snapshot computed: {self.config_snapshot_id[:12]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to compute config snapshot: {e}")
            self.config_snapshot_id = "unknown"
            self.config_snapshot = {}
    
    def _compute_model_fingerprint(self, model_versions: Dict[str, Dict[str, str]]):
        """Compute model version fingerprint for reproducibility"""
        try:
            # Store resolved model versions
            self.resolved_model_versions = model_versions.copy()
            
            # Create deterministic fingerprint
            fingerprint_data = {
                'asr_models': model_versions.get('asr', {}),
                'diarization_models': model_versions.get('diarization', {}),
                'punctuation_models': model_versions.get('punctuation', {}),
                'source_separation_models': model_versions.get('source_separation', {}),
                'normalization_models': model_versions.get('normalization', {})
            }
            
            fingerprint_json = json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=True)
            self.model_version_fingerprint = hashlib.sha256(fingerprint_json.encode('utf-8')).hexdigest()
            
            self.logger.info(f"Model fingerprint computed: {self.model_version_fingerprint[:12]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to compute model fingerprint: {e}")
            self.model_version_fingerprint = "unknown"
            self.resolved_model_versions = {}
    
    def _generate_global_seed(self):
        """Generate deterministic global seed from inputs"""
        try:
            # Combine media SHA256 + config snapshot for seed generation
            seed_input = f"{self.media_sha256}:{self.config_snapshot_id}:{self.model_version_fingerprint}"
            seed_bytes = hashlib.sha256(seed_input.encode('utf-8')).digest()
            
            # Convert first 4 bytes to 32-bit signed integer
            self.global_seed = int.from_bytes(seed_bytes[:4], byteorder='big', signed=True)
            
            self.logger.info(f"Global seed generated: {self.global_seed}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate global seed: {e}")
            self.global_seed = 42  # Fallback deterministic seed
    
    def _compute_stage_seeds(self):
        """Compute deterministic seeds for each processing stage"""
        try:
            self.stage_seeds = {}
            for stage, offset in STAGE_SEED_OFFSETS.items():
                stage_seed = self.global_seed + offset
                self.stage_seeds[stage] = stage_seed
                
            self.logger.info(f"Stage seeds computed for {len(self.stage_seeds)} stages")
            
        except Exception as e:
            self.logger.error(f"Failed to compute stage seeds: {e}")
            # Fallback to offset-only seeds
            self.stage_seeds = {stage: offset for stage, offset in STAGE_SEED_OFFSETS.items()}
    
    def _log_determinism_init(self):
        """Log determinism initialization telemetry"""
        try:
            init_data = {
                'run_id': self.run_id,
                'media_sha256': self.media_sha256,
                'global_seed': self.global_seed,
                'config_snapshot_id': self.config_snapshot_id,
                'model_version_fingerprint': self.model_version_fingerprint,
                'stage_count': len(self.stage_seeds),
                'determinism_enabled': self.determinism_enabled
            }
            
            self.telemetry_data.update(init_data)
            
            # Log structured event for observability
            self.logger.log_event("determinism_init", init_data)
            
            # Log summary for human readability
            self.logger.info(
                f"Deterministic context initialized - "
                f"run_id: {self.run_id}, "
                f"global_seed: {self.global_seed}, "
                f"config: {self.config_snapshot_id[:12]}..., "
                f"models: {self.model_version_fingerprint[:12]}..."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log determinism init: {e}")
    
    def get_stage_seed(self, stage: str) -> int:
        """Get deterministic seed for a specific processing stage"""
        if stage not in STAGE_SEED_OFFSETS:
            raise ValueError(f"Unknown stage: {stage}. Available stages: {list(STAGE_SEED_OFFSETS.keys())}")
        
        seed = self.stage_seeds.get(stage, STAGE_SEED_OFFSETS[stage])
        
        # Log seed usage for traceability
        self.logger.info(f"Stage '{stage}' using seed: {seed}")
        
        return seed
    
    def apply_deterministic_seed(self, stage: str):
        """Apply deterministic seeding to all random number generators for a stage"""
        try:
            seed = self.get_stage_seed(stage)
            
            # Set Python random seed
            random.seed(seed)
            
            # Set NumPy random seed
            np.random.seed(seed % (2**32))  # NumPy requires uint32
            
            # Set PyTorch seed if available
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass  # PyTorch not available
            
            self.logger.info(f"Applied deterministic seeding for stage '{stage}' with seed {seed}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply deterministic seeding for stage '{stage}': {e}")
    
    def create_deterministic_cache_key(self, component: str, inputs: Dict[str, Any]) -> str:
        """Create deterministic cache key that includes run context"""
        try:
            cache_data = {
                'component': component,
                'inputs': inputs,
                'config_snapshot_id': self.config_snapshot_id,
                'model_version_fingerprint': self.model_version_fingerprint
            }
            
            cache_json = json.dumps(cache_data, sort_keys=True, ensure_ascii=True)
            cache_key = hashlib.sha256(cache_json.encode('utf-8')).hexdigest()
            
            self.logger.debug(f"Created deterministic cache key for {component}: {cache_key[:12]}...")
            
            return cache_key
            
        except Exception as e:
            self.logger.error(f"Failed to create deterministic cache key: {e}")
            return f"fallback_{component}_{hash(str(inputs))}"
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get comprehensive telemetry summary for determinism tracking"""
        current_time = time.monotonic()
        processing_duration = current_time - self.start_time
        
        return {
            **self.telemetry_data,
            'processing_duration_seconds': processing_duration,
            'stage_seeds': self.stage_seeds.copy(),
            'resolved_model_count': sum(len(models) for models in self.resolved_model_versions.values()),
            'config_snapshot_size': len(json.dumps(self.config_snapshot)) if self.config_snapshot else 0
        }
    
    def validate_determinism_state(self) -> Dict[str, bool]:
        """Validate current determinism state for debugging"""
        validation_results = {
            'has_media_sha256': bool(self.media_sha256),
            'has_config_snapshot': bool(self.config_snapshot_id),
            'has_model_fingerprint': bool(self.model_version_fingerprint),
            'has_global_seed': self.global_seed != 0,
            'has_stage_seeds': len(self.stage_seeds) == len(STAGE_SEED_OFFSETS),
            'determinism_enabled': self.determinism_enabled
        }
        
        all_valid = all(validation_results.values())
        validation_results['all_valid'] = all_valid
        
        if not all_valid:
            self.logger.warning(f"Determinism validation failed: {validation_results}")
        else:
            self.logger.info("Determinism validation passed")
        
        return validation_results


def create_run_context(media_path: str,
                      config_dict: Dict[str, Any],
                      model_versions: Dict[str, Dict[str, str]],
                      session_id: Optional[str] = None,
                      project_id: Optional[str] = None,
                      resume_mode: bool = False) -> RunContext:
    """
    Factory function to create RunContext with automatic media hashing.
    
    Args:
        media_path: Path to input media file
        config_dict: Complete configuration dictionary
        model_versions: Resolved model versions by category
        session_id: Optional session identifier
        project_id: Optional project identifier
        resume_mode: Whether this is resuming an existing run
        
    Returns:
        RunContext instance ready for deterministic processing
    """
    # Compute media SHA256
    media_sha256 = _compute_media_sha256(media_path)
    
    # Create deterministic context
    return RunContext.create_deterministic_context(
        media_sha256=media_sha256,
        config_dict=config_dict,
        model_versions=model_versions,
        session_id=session_id,
        project_id=project_id,
        resume_mode=resume_mode
    )


def _compute_media_sha256(media_path: str) -> str:
    """Compute SHA256 hash of media file for deterministic seeding"""
    try:
        hasher = hashlib.sha256()
        with open(media_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger = create_enhanced_logger("run_context")
        logger.error(f"Failed to compute media SHA256: {e}")
        return "unknown_media"


# Global run context instance (set by EnsembleManager)
_current_run_context: Optional[RunContext] = None


def set_global_run_context(context: RunContext):
    """Set global run context for access by all components"""
    global _current_run_context
    _current_run_context = context


def get_global_run_context() -> Optional[RunContext]:
    """Get current global run context"""
    return _current_run_context


def get_stage_seed(stage: str) -> int:
    """Get deterministic seed for stage from global context"""
    context = get_global_run_context()
    if context is None:
        # Fallback to offset-based seed if no context
        return STAGE_SEED_OFFSETS.get(stage, 42)
    return context.get_stage_seed(stage)


def apply_stage_seeding(stage: str):
    """Apply deterministic seeding for stage from global context"""
    context = get_global_run_context()
    if context is not None:
        context.apply_deterministic_seed(stage)
    else:
        # Fallback seeding
        seed = STAGE_SEED_OFFSETS.get(stage, 42)
        random.seed(seed)
        np.random.seed(seed % (2**32))