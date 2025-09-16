"""
Dialect Handling Configuration Loader

Loads and manages configuration for the dialect handling system from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from utils.enhanced_structured_logger import create_enhanced_logger


@dataclass
class DialectConfig:
    """Configuration class for dialect handling"""
    
    # Global settings
    enable_dialect_handling: bool = True
    similarity_threshold: float = 0.7
    confidence_boost_factor: float = 0.05
    enable_g2p_fallback: bool = True
    
    # Supported dialects
    supported_dialects: List[str] = field(default_factory=lambda: [
        'southern', 'aave', 'nyc', 'boston', 'midwest', 'west_coast'
    ])
    
    # Phonetic distance settings
    edit_distance_threshold: int = 3
    normalized_distance_threshold: float = 0.5
    alignment_score_threshold: float = 0.6
    
    # Confidence adjustment settings
    max_boost_per_word: float = 0.15
    max_boost_per_segment: float = 0.10
    pattern_frequency_weight: bool = True
    segment_length_penalty: bool = True
    
    # Dialect weights
    dialect_weights: Dict[str, float] = field(default_factory=lambda: {
        'southern': 1.0,
        'aave': 1.0,
        'nyc': 0.9,
        'boston': 0.95,
        'midwest': 0.8,
        'west_coast': 0.7
    })
    
    # Processing settings
    min_word_length: int = 2
    min_segment_duration: float = 0.5
    max_candidates_per_run: int = 100
    cache_phonetic_lookups: bool = True
    
    # CMUdict settings
    use_primary_pronunciation: bool = True
    include_alternatives: bool = False
    cmudict_cache_size: int = 10000
    
    # G2P settings
    enable_rule_based: bool = True
    confidence_penalty: float = 0.3
    g2p_cache_size: int = 5000
    max_phoneme_length: int = 15
    
    # Performance settings
    max_processing_time_per_candidate: float = 2.0
    parallel_processing: bool = False
    memory_cleanup_interval: int = 100
    
    # Logging settings
    log_dialect_matches: bool = True
    log_confidence_adjustments: bool = True
    log_phonetic_distances: bool = False
    log_processing_stats: bool = True
    
    # Integration settings
    pipeline_position: str = "post_asr"
    preserve_original_confidence: bool = True
    add_metadata_to_segments: bool = True
    enable_text_normalization: bool = False


class DialectConfigLoader:
    """Loads dialect handling configuration from YAML files"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = create_enhanced_logger("dialect_config_loader")
        
        # Default config path
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "config", "dialect_handling", "dialect_config.yaml")
        
        self.config_path = Path(config_path)
        self.config: Optional[DialectConfig] = None
    
    def load_config(self) -> DialectConfig:
        """
        Load dialect configuration from YAML file
        
        Returns:
            DialectConfig instance with loaded settings
        """
        try:
            if not self.config_path.exists():
                self.logger.warning(f"Dialect config file not found: {self.config_path}")
                self.logger.info("Using default dialect configuration")
                return DialectConfig()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if not yaml_data:
                self.logger.warning("Empty dialect config file, using defaults")
                return DialectConfig()
            
            # Create config from YAML data
            config = DialectConfig()
            
            # Load global settings
            if 'enable_dialect_handling' in yaml_data:
                config.enable_dialect_handling = yaml_data['enable_dialect_handling']
            if 'similarity_threshold' in yaml_data:
                config.similarity_threshold = yaml_data['similarity_threshold']
            if 'confidence_boost_factor' in yaml_data:
                config.confidence_boost_factor = yaml_data['confidence_boost_factor']
            if 'enable_g2p_fallback' in yaml_data:
                config.enable_g2p_fallback = yaml_data['enable_g2p_fallback']
            
            # Load supported dialects
            if 'supported_dialects' in yaml_data:
                config.supported_dialects = yaml_data['supported_dialects']
            
            # Load phonetic distance settings
            if 'phonetic_distance' in yaml_data:
                pd_config = yaml_data['phonetic_distance']
                if 'edit_distance_threshold' in pd_config:
                    config.edit_distance_threshold = pd_config['edit_distance_threshold']
                if 'normalized_distance_threshold' in pd_config:
                    config.normalized_distance_threshold = pd_config['normalized_distance_threshold']
                if 'alignment_score_threshold' in pd_config:
                    config.alignment_score_threshold = pd_config['alignment_score_threshold']
            
            # Load confidence adjustment settings
            if 'confidence_adjustment' in yaml_data:
                ca_config = yaml_data['confidence_adjustment']
                if 'max_boost_per_word' in ca_config:
                    config.max_boost_per_word = ca_config['max_boost_per_word']
                if 'max_boost_per_segment' in ca_config:
                    config.max_boost_per_segment = ca_config['max_boost_per_segment']
                if 'pattern_frequency_weight' in ca_config:
                    config.pattern_frequency_weight = ca_config['pattern_frequency_weight']
                if 'segment_length_penalty' in ca_config:
                    config.segment_length_penalty = ca_config['segment_length_penalty']
            
            # Load dialect weights
            if 'dialect_weights' in yaml_data:
                config.dialect_weights = yaml_data['dialect_weights']
            
            # Load processing settings
            if 'processing' in yaml_data:
                proc_config = yaml_data['processing']
                if 'min_word_length' in proc_config:
                    config.min_word_length = proc_config['min_word_length']
                if 'min_segment_duration' in proc_config:
                    config.min_segment_duration = proc_config['min_segment_duration']
                if 'max_candidates_per_run' in proc_config:
                    config.max_candidates_per_run = proc_config['max_candidates_per_run']
                if 'cache_phonetic_lookups' in proc_config:
                    config.cache_phonetic_lookups = proc_config['cache_phonetic_lookups']
            
            # Load CMUdict settings
            if 'cmudict' in yaml_data:
                cmu_config = yaml_data['cmudict']
                if 'use_primary_pronunciation' in cmu_config:
                    config.use_primary_pronunciation = cmu_config['use_primary_pronunciation']
                if 'include_alternatives' in cmu_config:
                    config.include_alternatives = cmu_config['include_alternatives']
                if 'cache_size' in cmu_config:
                    config.cmudict_cache_size = cmu_config['cache_size']
            
            # Load G2P settings
            if 'g2p' in yaml_data:
                g2p_config = yaml_data['g2p']
                if 'enable_rule_based' in g2p_config:
                    config.enable_rule_based = g2p_config['enable_rule_based']
                if 'confidence_penalty' in g2p_config:
                    config.confidence_penalty = g2p_config['confidence_penalty']
                if 'cache_size' in g2p_config:
                    config.g2p_cache_size = g2p_config['cache_size']
                if 'max_phoneme_length' in g2p_config:
                    config.max_phoneme_length = g2p_config['max_phoneme_length']
            
            # Load performance settings
            if 'performance' in yaml_data:
                perf_config = yaml_data['performance']
                if 'max_processing_time_per_candidate' in perf_config:
                    config.max_processing_time_per_candidate = perf_config['max_processing_time_per_candidate']
                if 'parallel_processing' in perf_config:
                    config.parallel_processing = perf_config['parallel_processing']
                if 'memory_cleanup_interval' in perf_config:
                    config.memory_cleanup_interval = perf_config['memory_cleanup_interval']
            
            # Load logging settings
            if 'logging' in yaml_data:
                log_config = yaml_data['logging']
                if 'log_dialect_matches' in log_config:
                    config.log_dialect_matches = log_config['log_dialect_matches']
                if 'log_confidence_adjustments' in log_config:
                    config.log_confidence_adjustments = log_config['log_confidence_adjustments']
                if 'log_phonetic_distances' in log_config:
                    config.log_phonetic_distances = log_config['log_phonetic_distances']
                if 'log_processing_stats' in log_config:
                    config.log_processing_stats = log_config['log_processing_stats']
            
            # Load integration settings
            if 'integration' in yaml_data:
                int_config = yaml_data['integration']
                if 'pipeline_position' in int_config:
                    config.pipeline_position = int_config['pipeline_position']
                if 'preserve_original_confidence' in int_config:
                    config.preserve_original_confidence = int_config['preserve_original_confidence']
                if 'add_metadata_to_segments' in int_config:
                    config.add_metadata_to_segments = int_config['add_metadata_to_segments']
                if 'enable_text_normalization' in int_config:
                    config.enable_text_normalization = int_config['enable_text_normalization']
            
            self.config = config
            self.logger.info(f"Dialect configuration loaded successfully from {self.config_path}")
            self.logger.info(f"Dialect handling enabled: {config.enable_dialect_handling}")
            self.logger.info(f"Supported dialects: {config.supported_dialects}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load dialect config from {self.config_path}: {e}")
            self.logger.info("Using default dialect configuration")
            return DialectConfig()
    
    def get_config(self) -> DialectConfig:
        """
        Get the loaded configuration, loading it if not already loaded
        
        Returns:
            DialectConfig instance
        """
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def reload_config(self) -> DialectConfig:
        """
        Force reload configuration from file
        
        Returns:
            DialectConfig instance
        """
        self.config = None
        return self.load_config()


# Global config loader instance
_config_loader: Optional[DialectConfigLoader] = None


def get_dialect_config_loader(config_path: Optional[str] = None) -> DialectConfigLoader:
    """
    Get global dialect configuration loader instance
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        DialectConfigLoader instance
    """
    global _config_loader
    if _config_loader is None or config_path is not None:
        _config_loader = DialectConfigLoader(config_path)
    return _config_loader


def load_dialect_config(config_path: Optional[str] = None) -> DialectConfig:
    """
    Load dialect configuration from YAML file
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        DialectConfig instance
    """
    loader = get_dialect_config_loader(config_path)
    return loader.load_config()