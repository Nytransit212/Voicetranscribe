"""
Hydra-based configuration system for Advanced Ensemble Transcription System.
Replaces the class-based Settings with a unified Hydra configuration approach.
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

class HydraSettings:
    """Hydra-based configuration manager for the ensemble transcription system"""
    
    _instance: Optional['HydraSettings'] = None
    _config: Optional[DictConfig] = None
    
    def __new__(cls) -> 'HydraSettings':
        """Singleton pattern to ensure single configuration instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize Hydra configuration system"""
        if self._config is None:
            self._initialize_hydra()
    
    def _initialize_hydra(self) -> None:
        """Initialize Hydra with configuration directory"""
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Get absolute path to config directory
        config_dir = Path(__file__).parent.absolute()
        
        # Initialize Hydra with config directory
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Compose configuration with overrides from environment
            overrides = self._get_environment_overrides()
            self._config = compose(config_name="config", overrides=overrides)
    
    def _get_environment_overrides(self) -> list[str]:
        """Get configuration overrides from environment variables"""
        overrides = []
        
        # API keys from environment (only override if they exist)
        if openai_key := os.getenv("OPENAI_API_KEY"):
            overrides.append(f"api.openai.api_key={openai_key}")
        
        if hf_token := os.getenv("HUGGINGFACE_TOKEN"):
            overrides.append(f"api.huggingface.auth_token={hf_token}")
        
        # Runtime overrides
        if log_level := os.getenv("LOG_LEVEL"):
            overrides.append(f"logging.level={log_level}")
        
        return overrides
    
    @property
    def config(self) -> DictConfig:
        """Get the resolved configuration"""
        if self._config is None:
            self._initialize_hydra()
        assert self._config is not None, "Configuration should be initialized"
        return self._config
    
    # Convenience properties for commonly accessed config sections
    @property
    def audio(self) -> DictConfig:
        """Audio processing configuration"""
        return self.config.audio
    
    @property
    def asr(self) -> DictConfig:
        """ASR configuration"""
        return self.config.asr
    
    @property
    def diarization(self) -> DictConfig:
        """Diarization configuration"""
        return self.config.diarization
    
    @property
    def scoring(self) -> DictConfig:
        """Confidence scoring configuration"""
        return self.config.scoring
    
    @property
    def ui(self) -> DictConfig:
        """UI configuration"""
        return self.config.ui
    
    @property
    def api(self) -> DictConfig:
        """API configuration"""
        return self.config.api
    
    @property
    def quality(self) -> DictConfig:
        """Quality gates configuration"""
        return self.config.quality
    
    @property
    def processing(self) -> DictConfig:
        """Processing configuration"""
        return self.config.processing
    
    # Legacy compatibility methods for existing codebase
    def get_diarization_config(self, variant: str, expected_speakers: int) -> Dict[str, Any]:
        """
        Get diarization configuration for a specific variant.
        Maintains compatibility with existing EnsembleManager code.
        
        Args:
            variant: Variant name ('conservative', 'balanced', 'liberal', etc.)
            expected_speakers: Expected number of speakers
            
        Returns:
            Configuration dictionary
        """
        if variant not in self.diarization.variants:
            variant = "balanced"  # Default fallback
        
        base_config = OmegaConf.to_container(self.diarization.variants[variant], resolve=True)
        assert isinstance(base_config, dict), "Config should be a dictionary"
        
        # Adjust speaker count based on variant strategy
        if variant == "conservative":
            min_speakers = max(self.diarization.min_speakers, expected_speakers - 2)
            max_speakers = expected_speakers + 1
        elif variant == "balanced":
            min_speakers = max(self.diarization.min_speakers, expected_speakers - 1)
            max_speakers = expected_speakers + 2
        elif variant == "liberal":
            min_speakers = max(self.diarization.min_speakers, expected_speakers)
            max_speakers = expected_speakers + 3
        else:
            # Other variants use default ranges
            min_speakers = max(self.diarization.min_speakers, expected_speakers - 1)
            max_speakers = expected_speakers + 2
        
        base_config.update({
            'min_speakers': min_speakers,
            'max_speakers': min(max_speakers, self.diarization.max_speakers)
        })
        
        # Ensure proper typing for return
        return {str(k): v for k, v in base_config.items()}
    
    def validate_environment(self) -> Dict[str, bool]:
        """
        Validate that required environment variables are set.
        Maintains compatibility with existing validation logic.
        
        Returns:
            Dictionary of validation results
        """
        return {
            'openai_api_key': bool(os.getenv("OPENAI_API_KEY")),
            'huggingface_token': bool(os.getenv("HUGGINGFACE_TOKEN")),
        }
    
    def get_runtime_config(self) -> Dict[str, Any]:
        """
        Get runtime configuration summary.
        Maintains compatibility with existing Settings class.
        
        Returns:
            Configuration summary
        """
        return {
            'audio_sample_rate': self.audio.sample_rate,
            'max_audio_duration': self.audio.max_duration_seconds,
            'asr_model': self.asr.model,
            'confidence_weights': OmegaConf.to_container(self.scoring.weights, resolve=True),
            'quality_gates': OmegaConf.to_container(self.quality, resolve=True),
            'max_concurrent_requests': self.processing.max_concurrent_asr_requests
        }
    
    def get_scoring_weights(self, preset: str = "default") -> Dict[str, float]:
        """
        Get scoring weights for a specific preset.
        
        Args:
            preset: Weight preset name
            
        Returns:
            Dictionary of scoring weights
        """
        if preset in self.scoring.weight_presets:
            weights = OmegaConf.to_container(self.scoring.weight_presets[preset], resolve=True)
        else:
            weights = OmegaConf.to_container(self.scoring.weights, resolve=True)
        
        assert isinstance(weights, dict), "Weights should be a dictionary"
        # Ensure proper typing for return
        return {str(k): float(v) for k, v in weights.items()}
    
    def save_run_config(self, output_dir: Path, run_id: str) -> None:
        """
        Save resolved configuration with run artifacts for reproducibility.
        
        Args:
            output_dir: Directory to save configuration
            run_id: Unique run identifier
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        config_file = output_dir / f"config_{run_id}.yaml"
        
        # Save resolved configuration
        resolved_config = OmegaConf.to_container(self.config, resolve=True)
        OmegaConf.save(resolved_config, config_file)
        
        # Also save environment info
        env_file = output_dir / f"environment_{run_id}.yaml"
        env_config = {
            'python_version': sys.version,
            'environment_variables': {
                'OPENAI_API_KEY': '***' if os.getenv("OPENAI_API_KEY") else None,
                'HUGGINGFACE_TOKEN': '***' if os.getenv("HUGGINGFACE_TOKEN") else None,
                'LOG_LEVEL': os.getenv("LOG_LEVEL"),
            },
            'run_timestamp': datetime.now().isoformat(),
        }
        OmegaConf.save(env_config, env_file)

# Global instance for easy access
settings = HydraSettings()

# Legacy compatibility - expose common constants for existing code
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Initialize settings with error handling
try:
    _settings_instance = HydraSettings()
    AUDIO_SAMPLE_RATE = _settings_instance.audio.sample_rate
    MAX_AUDIO_DURATION = _settings_instance.audio.max_duration_seconds
    ASR_MODEL = getattr(_settings_instance.asr, 'model', 'whisper-1')
    CONFIDENCE_WEIGHTS = _settings_instance.get_scoring_weights()
    SPEAKER_COLORS = getattr(_settings_instance.ui, 'speaker_colors', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
except Exception as e:
    # Fallback values if config loading fails
    AUDIO_SAMPLE_RATE = 16000
    MAX_AUDIO_DURATION = 10800  # 3 hours - realistic maximum duration
    ASR_MODEL = 'whisper-1'
    CONFIDENCE_WEIGHTS = {'D': 0.28, 'A': 0.32, 'L': 0.18, 'R': 0.12, 'O': 0.10}
    SPEAKER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']