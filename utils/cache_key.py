"""
Comprehensive Cache Key System for Ensemble Transcription

This module provides bulletproof cache key generation that includes ALL factors
affecting outputs to eliminate stale artifact bugs. Zero tolerance for incorrect
cache hits - prefer cache miss to wrong results.
"""

import json
import hashlib
import time
import threading
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

from utils.enhanced_structured_logger import create_enhanced_logger


class CacheScope(Enum):
    """Cache scope levels for different data isolation needs"""
    GLOBAL = "global"          # Shared across all projects/sessions
    PROJECT = "project"        # Project-specific (isolated by project_id)
    SESSION = "session"        # Session-specific (isolated by session_id)
    RUN = "run"               # Run-specific (isolated by run_id)


@dataclass
class CacheKey:
    """
    Comprehensive cache key that includes ALL factors affecting processing outputs.
    
    This schema ensures that cache hits only occur when truly safe by including
    every input parameter, model version, and configuration option that could
    change the processing result.
    """
    
    # === Core Identity Fields ===
    media_sha256: str = ""                    # SHA256 of input media file
    stage_name: str = ""                      # Processing stage (asr, diarization, etc.)
    stage_version: str = "1.0.0"             # Stage implementation version
    component_name: str = ""                  # Specific component/provider name
    
    # === Configuration Fingerprints ===
    config_snapshot_id: str = ""             # SHA256 of full config dict
    model_version_fingerprint: str = ""      # SHA256 of all model versions
    normalization_profile_id: str = ""       # Text normalization profile
    glossary_snapshot_hash: str = ""         # Custom glossary/terminology hash
    
    # === Audio Processing Parameters ===
    sample_rate: int = 0                      # Audio sample rate
    channel_layout: str = ""                  # Audio channel configuration
    chunk_profile_id: str = ""               # Audio chunking parameters
    chunk_overlap_sec: float = 0.0           # Overlap between chunks
    entropy_threshold: float = 0.0           # Audio quality/noise threshold
    
    # === Diarization Parameters ===
    diarizer_hash: str = ""                   # Diarization model/config hash
    separation_enabled: bool = False          # Source separation flag
    stems_count: int = 0                      # Number of separation stems
    speaker_count_hint: Optional[int] = None  # Expected speaker count
    
    # === Engine-Specific Parameters ===
    engine_preset_ids: Dict[str, str] = field(default_factory=dict)  # Provider-specific presets
    model_specific_params: Dict[str, Any] = field(default_factory=dict)  # Model parameters
    
    # === Language and Context ===
    language_hint: str = ""                   # Language detection hint
    dialect_config_hash: str = ""            # Dialect handling configuration
    context_window_config: str = ""          # Context handling parameters
    
    # === Processing Context ===
    deterministic_seed: Optional[int] = None  # Seed for deterministic processing
    quality_profile: str = ""                # Quality vs speed trade-off
    verification_enabled: bool = False        # Quality verification flag
    verifier_version: str = ""               # Verifier implementation version
    
    # === Cache Metadata ===
    scope: CacheScope = CacheScope.GLOBAL    # Cache isolation scope
    project_id: str = ""                     # Project identifier for scoping
    session_id: str = ""                     # Session identifier for scoping
    run_id: str = ""                         # Run identifier for scoping
    
    # === Version Tracking ===
    schema_version: str = "1.0.0"           # CacheKey schema version
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        """Validate cache key fields after initialization"""
        if not self.media_sha256:
            raise ValueError("media_sha256 is required for cache key")
        if not self.stage_name:
            raise ValueError("stage_name is required for cache key")
        if not self.component_name:
            raise ValueError("component_name is required for cache key")
    
    def to_stable_dict(self) -> Dict[str, Any]:
        """
        Convert to stable dictionary representation for hashing.
        
        Ensures deterministic serialization by:
        - Sorting all dictionary keys
        - Converting enums to string values
        - Handling optional fields consistently
        - Ensuring float precision consistency
        """
        data = asdict(self)
        
        # Convert enum to string
        data['scope'] = self.scope.value
        
        # Ensure consistent float precision
        data['chunk_overlap_sec'] = round(self.chunk_overlap_sec, 6)
        data['entropy_threshold'] = round(self.entropy_threshold, 6)
        
        # Sort nested dictionaries for consistency
        data['engine_preset_ids'] = dict(sorted(self.engine_preset_ids.items()))
        data['model_specific_params'] = self._normalize_dict(self.model_specific_params)
        
        # Handle None values consistently
        for key, value in data.items():
            if value is None:
                data[key] = ""
        
        return data
    
    def _normalize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively normalize dictionary for stable serialization"""
        if not isinstance(d, dict):
            return d
        
        normalized = {}
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                normalized[key] = self._normalize_dict(value)
            elif isinstance(value, (list, tuple)):
                # Sort lists if they contain comparable items
                try:
                    normalized[key] = sorted(value) if value else []
                except TypeError:
                    # If items aren't comparable, keep original order
                    normalized[key] = list(value)
            elif isinstance(value, float):
                # Consistent float precision
                normalized[key] = round(value, 6)
            else:
                normalized[key] = value
        
        return normalized
    
    def generate_cache_key(self) -> str:
        """
        Generate deterministic cache key from all parameters.
        
        Returns:
            SHA256 hash as hex string (64 characters)
        """
        stable_dict = self.to_stable_dict()
        stable_json = json.dumps(stable_dict, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
        
        # Generate SHA256 hash
        cache_key = hashlib.sha256(stable_json.encode('utf-8')).hexdigest()
        
        return cache_key
    
    def generate_short_key(self) -> str:
        """Generate short cache key for logging (first 12 characters)"""
        return self.generate_cache_key()[:12]
    
    def is_compatible_with(self, other: 'CacheKey') -> bool:
        """
        Check if this cache key is compatible with another for cache validation.
        
        Two cache keys are compatible if all critical fields match.
        """
        if not isinstance(other, CacheKey):
            return False
        
        # Critical fields that must match exactly
        critical_fields = [
            'media_sha256', 'stage_name', 'component_name',
            'config_snapshot_id', 'model_version_fingerprint',
            'sample_rate', 'channel_layout', 'separation_enabled'
        ]
        
        for field in critical_fields:
            if getattr(self, field) != getattr(other, field):
                return False
        
        return True
    
    def get_validation_fingerprint(self) -> str:
        """
        Get fingerprint for cache validation (subset of critical fields).
        
        This is used to quickly check if cached data is still valid
        without regenerating the full cache key.
        """
        validation_data = {
            'media_sha256': self.media_sha256,
            'config_snapshot_id': self.config_snapshot_id,
            'model_version_fingerprint': self.model_version_fingerprint,
            'stage_version': self.stage_version,
            'schema_version': self.schema_version
        }
        
        validation_json = json.dumps(validation_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(validation_json.encode('utf-8')).hexdigest()[:16]
    
    def update_from_run_context(self, run_context) -> 'CacheKey':
        """
        Update cache key fields from RunContext for deterministic processing.
        
        Args:
            run_context: RunContext instance with fingerprints and seeds
            
        Returns:
            New CacheKey instance with updated fields
        """
        updated_dict = asdict(self)
        
        # Update from RunContext
        updated_dict.update({
            'media_sha256': run_context.media_sha256,
            'config_snapshot_id': run_context.config_snapshot_id,
            'model_version_fingerprint': run_context.model_version_fingerprint,
            'deterministic_seed': run_context.get_stage_seed(self.stage_name),
            'project_id': run_context.project_id,
            'session_id': run_context.session_id,
            'run_id': run_context.run_id
        })
        
        return CacheKey(**updated_dict)
    
    @classmethod
    def create_for_stage(cls, 
                        stage_name: str,
                        component_name: str,
                        run_context,
                        stage_params: Optional[Dict[str, Any]] = None,
                        scope: CacheScope = CacheScope.PROJECT) -> 'CacheKey':
        """
        Factory method to create cache key for a processing stage.
        
        Args:
            stage_name: Name of processing stage
            component_name: Specific component/provider name
            run_context: RunContext with fingerprints and configuration
            stage_params: Stage-specific parameters
            scope: Cache isolation scope
            
        Returns:
            CacheKey instance configured for the stage
        """
        stage_params = stage_params or {}
        
        # Extract common parameters
        cache_key_params = {
            'stage_name': stage_name,
            'component_name': component_name,
            'media_sha256': run_context.media_sha256,
            'config_snapshot_id': run_context.config_snapshot_id,
            'model_version_fingerprint': run_context.model_version_fingerprint,
            'deterministic_seed': run_context.get_stage_seed(stage_name),
            'scope': scope,
            'project_id': run_context.project_id,
            'session_id': run_context.session_id,
            'run_id': run_context.run_id
        }
        
        # Add stage-specific parameters
        if 'sample_rate' in stage_params:
            cache_key_params['sample_rate'] = stage_params['sample_rate']
        if 'language_hint' in stage_params:
            cache_key_params['language_hint'] = stage_params['language_hint']
        if 'quality_profile' in stage_params:
            cache_key_params['quality_profile'] = stage_params['quality_profile']
        
        # Handle engine-specific parameters
        if 'engine_preset_id' in stage_params:
            cache_key_params['engine_preset_ids'] = {component_name: stage_params['engine_preset_id']}
        
        # Handle model-specific parameters
        model_params = stage_params.get('model_params', {})
        if model_params:
            cache_key_params['model_specific_params'] = model_params
        
        return cls(**cache_key_params)


@dataclass
class CacheMetadata:
    """
    Sidecar metadata stored alongside cached data for validation and management.
    
    This metadata is stored in .meta.json files and used for:
    - Cache validation and integrity checking
    - LRU eviction and cleanup
    - Debugging and auditing
    """
    
    # === Cache Key Information ===
    cache_key: str = ""                       # Full cache key (SHA256)
    cache_key_short: str = ""                # Short cache key for logging
    validation_fingerprint: str = ""         # Quick validation fingerprint
    
    # === Original CacheKey Fields ===
    original_cache_key_dict: Dict[str, Any] = field(default_factory=dict)
    
    # === File Metadata ===
    data_file_path: str = ""                 # Path to cached data file
    byte_size: int = 0                       # Size of cached data in bytes
    file_format: str = ""                    # Format of cached data (json, pickle, etc.)
    compression: str = ""                    # Compression algorithm if any
    
    # === Creation Information ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by_component: str = ""           # Component that created the cache
    component_version: str = ""              # Version of creating component
    producing_run_id: str = ""               # Run ID that produced this cache
    
    # === Validation State ===
    last_validated_at: str = ""             # Last successful validation timestamp
    validation_count: int = 0                # Number of successful validations
    
    # === Access Tracking ===
    last_accessed_at: str = ""              # Last cache access timestamp
    access_count: int = 0                    # Number of cache accesses
    
    # === Schema Version ===
    metadata_schema_version: str = "1.0.0"  # Metadata schema version
    
    def update_access(self):
        """Update access tracking information"""
        self.last_accessed_at = datetime.now(timezone.utc).isoformat()
        self.access_count += 1
    
    def update_validation(self):
        """Update validation tracking information"""
        self.last_validated_at = datetime.now(timezone.utc).isoformat()
        self.validation_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheMetadata':
        """Create from dictionary loaded from JSON"""
        return cls(**data)
    
    def is_stale(self, max_age_hours: int = 168) -> bool:
        """Check if cache entry is stale based on age"""
        if not self.created_at:
            return True
        
        try:
            created_time = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
            age_hours = (datetime.now(timezone.utc) - created_time).total_seconds() / 3600
            return age_hours > max_age_hours
        except (ValueError, TypeError):
            return True  # Treat invalid timestamps as stale


class CacheKeyGenerator:
    """
    Thread-safe cache key generator with logging and validation.
    
    This class provides the main interface for generating and validating
    cache keys throughout the ensemble processing pipeline.
    """
    
    def __init__(self, logger_name: str = "cache_key_generator"):
        self.logger = create_enhanced_logger(logger_name)
        self._lock = threading.Lock()
        
        # Statistics tracking
        self.stats = {
            'keys_generated': 0,
            'validation_checks': 0,
            'validation_failures': 0,
            'compatibility_checks': 0
        }
    
    def generate_for_stage(self,
                          stage_name: str,
                          component_name: str,
                          run_context,
                          stage_params: Optional[Dict[str, Any]] = None,
                          scope: CacheScope = CacheScope.PROJECT) -> Tuple[str, CacheKey]:
        """
        Generate cache key for a processing stage with full validation.
        
        Args:
            stage_name: Name of processing stage
            component_name: Specific component/provider name
            run_context: RunContext with fingerprints and configuration
            stage_params: Stage-specific parameters
            scope: Cache isolation scope
            
        Returns:
            Tuple of (cache_key_string, CacheKey_object)
        """
        with self._lock:
            try:
                # Create cache key
                cache_key = CacheKey.create_for_stage(
                    stage_name=stage_name,
                    component_name=component_name,
                    run_context=run_context,
                    stage_params=stage_params,
                    scope=scope
                )
                
                # Generate cache key string
                cache_key_string = cache_key.generate_cache_key()
                
                # Update statistics
                self.stats['keys_generated'] += 1
                
                # Log cache key generation
                self.logger.debug(
                    f"Generated cache key for {stage_name}/{component_name}",
                    cache_key_short=cache_key.generate_short_key(),
                    stage_name=stage_name,
                    component_name=component_name,
                    scope=scope.value,
                    run_id=run_context.run_id
                )
                
                return cache_key_string, cache_key
                
            except Exception as e:
                self.logger.error(
                    f"Failed to generate cache key for {stage_name}/{component_name}",
                    error=str(e),
                    stage_name=stage_name,
                    component_name=component_name
                )
                raise
    
    def validate_cache_key(self, 
                          stored_cache_key: CacheKey,
                          current_run_context) -> bool:
        """
        Validate stored cache key against current run context.
        
        Args:
            stored_cache_key: CacheKey loaded from cache metadata
            current_run_context: Current RunContext for validation
            
        Returns:
            True if cache key is valid, False if stale/incompatible
        """
        with self._lock:
            self.stats['validation_checks'] += 1
            
            try:
                # Check media SHA256
                if stored_cache_key.media_sha256 != current_run_context.media_sha256:
                    self.logger.warning(
                        "Cache validation failed: media SHA256 mismatch",
                        stored_media=stored_cache_key.media_sha256,
                        current_media=current_run_context.media_sha256,
                        cache_key_short=stored_cache_key.generate_short_key()
                    )
                    self.stats['validation_failures'] += 1
                    return False
                
                # Check config snapshot
                if stored_cache_key.config_snapshot_id != current_run_context.config_snapshot_id:
                    self.logger.warning(
                        "Cache validation failed: config snapshot mismatch",
                        stored_config=stored_cache_key.config_snapshot_id,
                        current_config=current_run_context.config_snapshot_id,
                        cache_key_short=stored_cache_key.generate_short_key()
                    )
                    self.stats['validation_failures'] += 1
                    return False
                
                # Check model version fingerprint
                if stored_cache_key.model_version_fingerprint != current_run_context.model_version_fingerprint:
                    self.logger.warning(
                        "Cache validation failed: model version fingerprint mismatch",
                        stored_models=stored_cache_key.model_version_fingerprint,
                        current_models=current_run_context.model_version_fingerprint,
                        cache_key_short=stored_cache_key.generate_short_key()
                    )
                    self.stats['validation_failures'] += 1
                    return False
                
                # Validation passed
                self.logger.debug(
                    "Cache validation passed",
                    cache_key_short=stored_cache_key.generate_short_key(),
                    stage_name=stored_cache_key.stage_name,
                    component_name=stored_cache_key.component_name
                )
                
                return True
                
            except Exception as e:
                self.logger.error(
                    "Cache validation error",
                    error=str(e),
                    cache_key_short=stored_cache_key.generate_short_key() if stored_cache_key else "unknown"
                )
                self.stats['validation_failures'] += 1
                return False
    
    def check_compatibility(self, cache_key1: CacheKey, cache_key2: CacheKey) -> bool:
        """Check if two cache keys are compatible"""
        with self._lock:
            self.stats['compatibility_checks'] += 1
            return cache_key1.is_compatible_with(cache_key2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache key generator statistics"""
        with self._lock:
            return self.stats.copy()


# Global cache key generator instance
_global_cache_key_generator: Optional[CacheKeyGenerator] = None


def get_cache_key_generator() -> CacheKeyGenerator:
    """Get or create global cache key generator instance"""
    global _global_cache_key_generator
    if _global_cache_key_generator is None:
        _global_cache_key_generator = CacheKeyGenerator()
    return _global_cache_key_generator


def generate_cache_key_for_stage(stage_name: str,
                                component_name: str,
                                run_context,
                                stage_params: Optional[Dict[str, Any]] = None,
                                scope: CacheScope = CacheScope.PROJECT) -> Tuple[str, CacheKey]:
    """
    Convenience function to generate cache key for a processing stage.
    
    This is the main entry point for cache key generation throughout the codebase.
    """
    generator = get_cache_key_generator()
    return generator.generate_for_stage(
        stage_name=stage_name,
        component_name=component_name,
        run_context=run_context,
        stage_params=stage_params,
        scope=scope
    )


def validate_cached_data(stored_cache_key: CacheKey, current_run_context) -> bool:
    """
    Convenience function to validate cached data against current context.
    
    Returns True if cache is valid, False if stale/incompatible.
    """
    generator = get_cache_key_generator()
    return generator.validate_cache_key(stored_cache_key, current_run_context)