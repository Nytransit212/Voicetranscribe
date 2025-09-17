"""
Model Version Resolver for Deterministic Processing

This module provides utilities to resolve exact model versions, validate
configurations, and compute model fingerprints for deterministic processing.
"""

import os
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import yaml

from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class ModelVersionInfo:
    """Information about a resolved model version"""
    category: str  # asr, diarization, etc.
    model_name: str
    model_version: str
    provider: str
    config_dict: Dict[str, Any]
    fingerprint: str
    resolution_time: float
    available: bool = True
    error_message: Optional[str] = None

@dataclass
class ModelFingerprint:
    """Complete model fingerprint for a processing run"""
    asr_models: Dict[str, ModelVersionInfo]
    diarization_models: Dict[str, ModelVersionInfo]
    source_separation_models: Dict[str, ModelVersionInfo]
    punctuation_models: Dict[str, ModelVersionInfo]
    normalization_models: Dict[str, ModelVersionInfo]
    
    # Computed fingerprint
    combined_fingerprint: str
    resolution_timestamp: str
    total_models: int


class ModelVersionResolver:
    """Resolves and validates exact model versions for deterministic processing"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize model version resolver
        
        Args:
            config_path: Path to model versions configuration file
        """
        self.config_path = config_path or "config/model_versions.yaml"
        self.logger = create_enhanced_logger("model_version_resolver")
        
        # Load model versions configuration
        self.model_config = self._load_model_config()
        
        # Cache for resolved models
        self._resolution_cache: Dict[str, ModelVersionInfo] = {}
        
        # Validation state
        self._validation_errors: List[str] = []
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model versions configuration from YAML"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.error(f"Model versions config not found: {self.config_path}")
                return {}
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            self.logger.info(f"Loaded model versions config: {len(config.get('model_versions', {}))} categories")
            return config.get('model_versions', {})
            
        except Exception as e:
            self.logger.error(f"Failed to load model config: {e}")
            return {}
    
    def resolve_all_models(self) -> ModelFingerprint:
        """
        Resolve all model versions and compute combined fingerprint
        
        Returns:
            ModelFingerprint with all resolved models and combined hash
        """
        self.logger.info("Starting complete model version resolution")
        resolution_start = time.time()
        
        # Resolve models by category
        asr_models = self._resolve_category_models("asr")
        diarization_models = self._resolve_category_models("diarization")
        source_separation_models = self._resolve_category_models("source_separation")
        punctuation_models = self._resolve_category_models("punctuation")
        normalization_models = self._resolve_category_models("normalization")
        
        # Count total models
        total_models = (
            len(asr_models) + len(diarization_models) + 
            len(source_separation_models) + len(punctuation_models) + 
            len(normalization_models)
        )
        
        # Compute combined fingerprint
        combined_fingerprint = self._compute_combined_fingerprint({
            'asr': asr_models,
            'diarization': diarization_models,
            'source_separation': source_separation_models,
            'punctuation': punctuation_models,
            'normalization': normalization_models
        })
        
        resolution_time = time.time() - resolution_start
        
        fingerprint = ModelFingerprint(
            asr_models=asr_models,
            diarization_models=diarization_models,
            source_separation_models=source_separation_models,
            punctuation_models=punctuation_models,
            normalization_models=normalization_models,
            combined_fingerprint=combined_fingerprint,
            resolution_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            total_models=total_models
        )
        
        self.logger.info(
            f"Model resolution complete: {total_models} models, "
            f"fingerprint: {combined_fingerprint[:12]}..., "
            f"duration: {resolution_time:.2f}s"
        )
        
        # Log telemetry
        self.logger.info("Model resolution complete", 
            event_type="model_resolution_complete",
            metrics={
                "total_models": total_models,
                "combined_fingerprint": combined_fingerprint,
                "resolution_duration": resolution_time,
                "categories": {
                    "asr": len(asr_models),
                    "diarization": len(diarization_models), 
                    "source_separation": len(source_separation_models),
                    "punctuation": len(punctuation_models),
                    "normalization": len(normalization_models)
                }
            })
        
        return fingerprint
    
    def _resolve_category_models(self, category: str) -> Dict[str, ModelVersionInfo]:
        """Resolve all models in a specific category"""
        category_config = self.model_config.get(category, {})
        resolved_models = {}
        
        for model_key, model_config in category_config.items():
            try:
                model_info = self._resolve_single_model(category, model_key, model_config)
                resolved_models[model_key] = model_info
                
            except Exception as e:
                self.logger.error(f"Failed to resolve {category}.{model_key}: {e}")
                # Create error entry
                resolved_models[model_key] = ModelVersionInfo(
                    category=category,
                    model_name=model_key,
                    model_version="unknown",
                    provider="unknown",
                    config_dict={},
                    fingerprint="error",
                    resolution_time=time.time(),
                    available=False,
                    error_message=str(e)
                )
        
        self.logger.info(f"Resolved {len(resolved_models)} models in category '{category}'")
        return resolved_models
    
    def _resolve_single_model(self, category: str, model_key: str, model_config: Dict[str, Any]) -> ModelVersionInfo:
        """Resolve a single model configuration"""
        cache_key = f"{category}.{model_key}"
        
        # Check cache first
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]
        
        resolution_start = time.time()
        
        # Extract model information
        model_name = model_config.get('model_name', model_key)
        model_version = model_config.get('model_version', 'latest')
        provider = model_config.get('provider', 'unknown')
        
        # Validate model availability
        available = self._validate_model_availability(category, model_config)
        
        # Compute model-specific fingerprint
        fingerprint = self._compute_model_fingerprint(model_config)
        
        resolution_time = time.time() - resolution_start
        
        model_info = ModelVersionInfo(
            category=category,
            model_name=model_name,
            model_version=model_version,
            provider=provider,
            config_dict=model_config.copy(),
            fingerprint=fingerprint,
            resolution_time=resolution_time,
            available=available
        )
        
        # Cache the result
        self._resolution_cache[cache_key] = model_info
        
        self.logger.debug(
            f"Resolved {category}.{model_key}: {model_name} v{model_version} "
            f"(provider: {provider}, available: {available})"
        )
        
        return model_info
    
    def _validate_model_availability(self, category: str, model_config: Dict[str, Any]) -> bool:
        """Validate that a model is available and accessible"""
        try:
            provider = model_config.get('provider', '')
            model_name = model_config.get('model_name', '')
            
            # Provider-specific validation
            if provider == 'openai':
                return self._validate_openai_model(model_config)
            elif provider == 'huggingface':
                return self._validate_huggingface_model(model_config)
            elif provider == 'assemblyai':
                return self._validate_assemblyai_model(model_config)
            elif provider == 'internal':
                return True  # Internal models are always available
            else:
                self.logger.warning(f"Unknown provider '{provider}' for model validation")
                return True  # Assume available for unknown providers
                
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
    
    def _validate_openai_model(self, model_config: Dict[str, Any]) -> bool:
        """Validate OpenAI model availability"""
        # Check if API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.warning("OPENAI_API_KEY not available - cannot validate model")
            return False
        
        # For deterministic resolution, we assume whisper-1 is always available
        model_name = model_config.get('model_name', '')
        return model_name in ['whisper-1']
    
    def _validate_huggingface_model(self, model_config: Dict[str, Any]) -> bool:
        """Validate Hugging Face model availability"""
        # For deterministic resolution, assume models in config are available
        # In production, could check model repository existence
        return True
    
    def _validate_assemblyai_model(self, model_config: Dict[str, Any]) -> bool:
        """Validate AssemblyAI model availability"""
        # Check if API key is available
        api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not api_key:
            self.logger.warning("ASSEMBLYAI_API_KEY not available - cannot validate model")
            return False
        
        return True
    
    def _compute_model_fingerprint(self, model_config: Dict[str, Any]) -> str:
        """Compute deterministic fingerprint for a single model configuration"""
        try:
            # Create deterministic representation
            fingerprint_data = {
                'model_name': model_config.get('model_name', ''),
                'model_version': model_config.get('model_version', ''),
                'provider': model_config.get('provider', ''),
                'version_lock': model_config.get('version_lock', ''),
                'config': {k: v for k, v in model_config.items() 
                          if k not in ['model_name', 'model_version', 'provider']}
            }
            
            # Create deterministic JSON and hash
            fingerprint_json = json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=True)
            fingerprint = hashlib.sha256(fingerprint_json.encode('utf-8')).hexdigest()
            
            return fingerprint
            
        except Exception as e:
            self.logger.error(f"Failed to compute model fingerprint: {e}")
            return "unknown"
    
    def _compute_combined_fingerprint(self, all_models: Dict[str, Dict[str, ModelVersionInfo]]) -> str:
        """Compute combined fingerprint for all models"""
        try:
            # Collect all model fingerprints in deterministic order
            fingerprint_data = {}
            
            for category in sorted(all_models.keys()):
                category_fingerprints = {}
                for model_key in sorted(all_models[category].keys()):
                    model_info = all_models[category][model_key]
                    category_fingerprints[model_key] = {
                        'fingerprint': model_info.fingerprint,
                        'available': model_info.available,
                        'model_name': model_info.model_name,
                        'model_version': model_info.model_version,
                        'provider': model_info.provider
                    }
                fingerprint_data[category] = category_fingerprints
            
            # Create deterministic JSON and hash
            combined_json = json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=True)
            combined_fingerprint = hashlib.sha256(combined_json.encode('utf-8')).hexdigest()
            
            return combined_fingerprint
            
        except Exception as e:
            self.logger.error(f"Failed to compute combined fingerprint: {e}")
            return "unknown"
    
    def get_model_versions_dict(self, fingerprint: ModelFingerprint) -> Dict[str, Dict[str, str]]:
        """
        Extract model versions in the format expected by RunContext
        
        Args:
            fingerprint: Resolved model fingerprint
            
        Returns:
            Dictionary of model versions by category
        """
        return {
            'asr': {
                key: f"{info.model_name}@{info.model_version}"
                for key, info in fingerprint.asr_models.items()
            },
            'diarization': {
                key: f"{info.model_name}@{info.model_version}"
                for key, info in fingerprint.diarization_models.items()
            },
            'source_separation': {
                key: f"{info.model_name}@{info.model_version}"
                for key, info in fingerprint.source_separation_models.items()
            },
            'punctuation': {
                key: f"{info.model_name}@{info.model_version}"
                for key, info in fingerprint.punctuation_models.items()
            },
            'normalization': {
                key: f"{info.model_name}@{info.model_version}"
                for key, info in fingerprint.normalization_models.items()
            }
        }
    
    def validate_fingerprint_consistency(self, expected_fingerprint: str, current_fingerprint: ModelFingerprint) -> bool:
        """
        Validate that current model fingerprint matches expected
        
        Args:
            expected_fingerprint: Expected combined fingerprint
            current_fingerprint: Currently resolved fingerprint
            
        Returns:
            True if fingerprints match
        """
        matches = expected_fingerprint == current_fingerprint.combined_fingerprint
        
        if not matches:
            self.logger.warning(
                f"Model fingerprint mismatch - "
                f"expected: {expected_fingerprint[:12]}..., "
                f"current: {current_fingerprint.combined_fingerprint[:12]}..."
            )
        
        return matches


# Global resolver instance
_model_resolver: Optional[ModelVersionResolver] = None


def get_model_resolver() -> ModelVersionResolver:
    """Get global model version resolver instance"""
    global _model_resolver
    if _model_resolver is None:
        _model_resolver = ModelVersionResolver()
    return _model_resolver


def resolve_model_versions() -> ModelFingerprint:
    """Convenience function to resolve all model versions"""
    resolver = get_model_resolver()
    return resolver.resolve_all_models()


def validate_model_consistency(expected_fingerprint: str) -> bool:
    """Validate current models match expected fingerprint"""
    resolver = get_model_resolver()
    current_fingerprint = resolver.resolve_all_models()
    return resolver.validate_fingerprint_consistency(expected_fingerprint, current_fingerprint)