import os
import tempfile
import json
import time
import uuid
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import hydra
from omegaconf import DictConfig

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
from core.overlap_diarizer import OverlapDiarizationEngine, StemDiarizer, OverlapDiarizationResult
from core.overlap_fusion import OverlapFusionEngine, OverlapFusionResult
from utils.stem_manifest import StemManifestManager, StemManifest
from core.overlap_processing_helpers import apply_overlap_processing_patches, run_legacy_source_separation
from core.post_fusion_punctuation_engine import PostFusionPunctuationEngine, create_punctuation_engine_from_preset
from core.text_normalizer import TextNormalizer, create_text_normalizer
from core.guardrail_verifier import GuardrailVerifier, create_guardrail_verifier
from core.dialect_handling_engine import DialectHandlingEngine
from core.dialect_config_loader import load_dialect_config
from core.term_miner import create_term_mining_engine, TermMiningEngine
from core.term_store import create_project_term_store, ProjectTermStore
from core.term_bias import create_adaptive_biasing_engine, AdaptiveBiasingEngine
from core.global_speaker_linker import GlobalSpeakerLinker, ClusteringResult
from core.speaker_relabeler import SpeakerRelabeler, RelabelingResult
from utils.embedding_cache import get_embedding_cache
from utils.capability_manager import get_capability_manager, check_system_capabilities, is_feature_available, require_feature
from utils.audio_format_validator import get_audio_validator, ensure_audio_format
from utils.manifest import create_manifest_manager, ManifestManager
from core.run_context import RunContext, create_run_context, set_global_run_context, get_global_run_context
from utils.model_version_resolver import get_model_resolver
from utils.atomic_io import (
    get_atomic_io_manager, 
    atomic_write, 
    TempDirectoryScope,
    create_run_temp_directory,
    get_run_temp_subdir
)
from core.post_fusion_realigner import (
    PostFusionRealigner,
    create_post_fusion_realigner,
    convert_transcript_to_realigner_format,
    convert_realigner_result_to_transcript_format,
    RealignerConfig
)
from utils.resource_scheduler import (
    ResourceScheduler,
    get_resource_scheduler,
    initialize_resource_scheduler,
    ProcessingStage,
    QualityLevel,
    DowngradeStrategy,
    with_resource_scheduling
)
from utils.elastic_chunker import (
    ElasticChunker,
    ChunkingConfig,
    create_elastic_chunker,
    get_elastic_chunker
)
from utils.disagreement_redecode import (
    DisagreementRedecodeEngine,
    RedecodeConfig,
    create_disagreement_redecode_engine
)
from utils.metrics_alerts import (
    get_metrics_collector,
    initialize_metrics_system,
    track_processing_stage,
    record_quality_metrics,
    record_business_event,
    track_performance
)

class EnsembleManagerInitializationError(Exception):
    """Raised when EnsembleManager fails to initialize properly"""
    def __init__(self, message: str, component: str = "unknown", original_error: Exception = None):
        self.component = component
        self.original_error = original_error
        super().__init__(f"EnsembleManager initialization failed in {component}: {message}")

class EnsembleManager:
    """Orchestrates the entire ensemble transcription pipeline"""
    
    @classmethod
    def create_safe(cls, 
                    expected_speakers: int = 10, 
                    noise_level: str = 'medium', 
                    target_language: Optional[str] = None, 
                    scoring_weights: Optional[Dict[str, float]] = None, 
                    enable_versioning: bool = True, 
                    domain: str = "general", 
                    consensus_strategy: str = "best_single_candidate", 
                    calibration_method: str = "registry_based", 
                    enable_speaker_mapping: bool = True, 
                    speaker_mapping_config: Optional[Dict[str, Any]] = None, 
                    chunked_processing_threshold: float = 900.0, 
                    enable_dialect_handling: bool = True, 
                    dialect_similarity_threshold: float = 0.7, 
                    dialect_confidence_boost: float = 0.05, 
                    supported_dialects: Optional[List[str]] = None, 
                    enable_auto_glossary: bool = True, 
                    auto_glossary_config: Optional[Dict[str, Any]] = None, 
                    project_id: Optional[str] = None, 
                    enable_long_horizon_tracking: bool = True, 
                    long_horizon_config: Optional[Dict[str, Any]] = None) -> 'EnsembleManager':
        """
        Safe factory method that gracefully handles initialization failures
        
        Returns:
            EnsembleManager instance with minimal working configuration if full init fails
            
        Raises:
            EnsembleManagerInitializationError: Only for critical failures that prevent basic operation
        """
        # Store initialization status for user feedback
        initialization_warnings = []
        initialization_errors = []
        
        try:
            # First attempt: Full initialization
            manager = cls.__new__(cls)
            manager._safe_init(
                expected_speakers=expected_speakers,
                noise_level=noise_level,
                target_language=target_language,
                scoring_weights=scoring_weights,
                enable_versioning=enable_versioning,
                domain=domain,
                consensus_strategy=consensus_strategy,
                calibration_method=calibration_method,
                enable_speaker_mapping=enable_speaker_mapping,
                speaker_mapping_config=speaker_mapping_config,
                chunked_processing_threshold=chunked_processing_threshold,
                enable_dialect_handling=enable_dialect_handling,
                dialect_similarity_threshold=dialect_similarity_threshold,
                dialect_confidence_boost=dialect_confidence_boost,
                supported_dialects=supported_dialects,
                enable_auto_glossary=enable_auto_glossary,
                auto_glossary_config=auto_glossary_config,
                project_id=project_id,
                enable_long_horizon_tracking=enable_long_horizon_tracking,
                long_horizon_config=long_horizon_config
            )
            return manager
            
        except Exception as e:
            # If full initialization fails, try minimal configuration
            print(f"Warning: Full EnsembleManager initialization failed ({e}), attempting minimal configuration...")
            try:
                manager = cls.__new__(cls)
                manager._minimal_init(expected_speakers, noise_level, target_language)
                manager._initialization_warnings = [f"Reduced functionality due to init error: {str(e)}"]
                return manager
            except Exception as minimal_error:
                raise EnsembleManagerInitializationError(
                    f"Both full and minimal initialization failed. Full error: {e}. Minimal error: {minimal_error}",
                    component="factory",
                    original_error=e
                )
    
    def _minimal_init(self, expected_speakers: int, noise_level: str, target_language: Optional[str]):
        """Initialize with absolute minimum requirements for basic transcription"""
        # Only set most basic parameters
        self.expected_speakers = expected_speakers
        self.noise_level = noise_level
        self.target_language = target_language
        self.domain = "general"
        
        # Disable all advanced features for minimal mode
        self.enable_versioning = False
        self.enable_speaker_mapping = False
        self.enable_dialect_handling = False
        self.enable_auto_glossary = False
        self.enable_long_horizon_tracking = False
        self.enable_resource_scheduling = False
        self.enable_caching = False
        self.enable_source_separation = False
        self.enable_overlap_aware_processing = False
        
        # Basic logging fallback
        self.structured_logger = None
        self.obs_manager = None
        self.metrics_collector = None
        
        # Initialize only essential components
        self.run_id = None
        self.consensus_strategy = "best_single_candidate"
        self.calibration_method = "simple"
        
        print("EnsembleManager initialized in minimal mode - advanced features disabled")
    
    def _safe_init(self, **kwargs):
        """Safe initialization wrapper that handles each component separately"""
        
        # Initialize basic parameters first (these should never fail)
        self._init_basic_params(**kwargs)
        
        # Initialize logging and observability (with fallbacks)
        self._init_observability()
        
        # Initialize U7 systems (with graceful fallbacks)
        self._init_u7_systems()
        
        # Initialize other components with individual error handling
        self._init_advanced_systems()
    
    def _init_basic_params(self, **kwargs):
        """Initialize basic parameters that should never fail"""
        self.expected_speakers = kwargs.get('expected_speakers', 10)
        self.noise_level = kwargs.get('noise_level', 'medium')
        self.target_language = kwargs.get('target_language', None)
        self.domain = kwargs.get('domain', "general")
        self.enable_versioning = kwargs.get('enable_versioning', True)
        self.consensus_strategy = kwargs.get('consensus_strategy', "best_single_candidate")
        self.calibration_method = kwargs.get('calibration_method', "registry_based")
        
        # Initialize all other configuration parameters with safe defaults
        self.enable_speaker_mapping = kwargs.get('enable_speaker_mapping', True)
        self.chunked_processing_threshold = kwargs.get('chunked_processing_threshold', 900.0)
        self.enable_dialect_handling = kwargs.get('enable_dialect_handling', True)
        self.enable_auto_glossary = kwargs.get('enable_auto_glossary', True)
        self.enable_long_horizon_tracking = kwargs.get('enable_long_horizon_tracking', True)
        
        # Configuration objects with safe defaults
        self.speaker_mapping_config = kwargs.get('speaker_mapping_config', {})
        self.auto_glossary_config = kwargs.get('auto_glossary_config', {})
        self.long_horizon_config = kwargs.get('long_horizon_config', {})
        
        # Apply safe defaults for all configuration dictionaries
        self._apply_config_defaults()
    
    def _apply_config_defaults(self):
        """Apply safe defaults to all configuration dictionaries"""
        # Speaker mapping safe defaults
        speaker_defaults = {
            'similarity_threshold': 0.7,
            'embedding_dim': 192,
            'min_segment_duration': 1.0,
            'cache_embeddings': True,
            'enable_metrics': False,  # Disabled for safety
            'use_ecapa_tdnn': False,  # Disabled for safety
            'enable_backtracking': False,  # Disabled for safety
        }
        self.speaker_mapping_config = {**speaker_defaults, **self.speaker_mapping_config}
        
        # Auto-glossary safe defaults
        glossary_defaults = {
            'mining_sensitivity': 0.6,
            'min_frequency_threshold': 2,
            'max_candidates_per_session': 200,
            'enable_variant_clustering': False,  # Disabled for safety
            'decay_sessions_threshold': 10,
            'minimum_support_threshold': 2,
            'storage_base_path': 'term_bases',
            'default_bias_strength': 0.7,
            'max_bias_terms_per_session': 50,
            'min_term_confidence': 0.5,
        }
        self.auto_glossary_config = {**glossary_defaults, **self.auto_glossary_config}
        
        # Long horizon tracking safe defaults
        horizon_defaults = {
            'min_turn_duration': 0.5,
            'max_turn_duration': 60.0,
            'embedding_aggregation_method': 'weighted_average',
            'clustering_method': 'hierarchical',
            'cluster_margin': 0.15,
            'min_cluster_size': 2,
            'enable_human_friendly_names': False,  # Disabled for safety
        }
        self.long_horizon_config = {**horizon_defaults, **self.long_horizon_config}
    
    def _init_observability(self):
        """Initialize observability with safe fallbacks"""
        try:
            self.obs_manager = initialize_observability(
                service_name="ensemble-transcription",
                enable_profiling=False,  # Disabled for safety
                log_level="INFO"
            )
            self.structured_logger = create_enhanced_logger("ensemble_manager", run_id=None)
        except Exception as e:
            print(f"Warning: Observability initialization failed ({e}), using basic logging")
            self.obs_manager = None
            self.structured_logger = None
    
    def _init_u7_systems(self):
        """Initialize U7 systems with individual error handling"""
        # Initialize basic U7 tracking
        self.run_id = None
        self.manifest_manager = None
        self.enable_manifest_tracking = False  # Disabled for safety
        
        # Try to initialize each U7 system individually
        systems = [
            ('cache_manager', get_cache_manager),
            ('deterministic_processor', get_deterministic_processor),
            ('worklist_manager', get_worklist_manager),
            ('selective_asr_processor', get_selective_asr_processor),
            ('asr_scheduler', get_asr_scheduler)
        ]
        
        for system_name, system_factory in systems:
            try:
                setattr(self, system_name, system_factory())
            except Exception as e:
                print(f"Warning: {system_name} initialization failed ({e}), using None fallback")
                setattr(self, system_name, None)
    
    def _init_advanced_systems(self):
        """Initialize advanced systems with graceful degradation"""
        # Capability manager with fallback
        try:
            self.capability_manager = get_capability_manager()
            self.capability_report = check_system_capabilities()
        except Exception as e:
            print(f"Warning: Capability manager failed ({e}), using basic capabilities")
            self.capability_manager = None
            self.capability_report = {}
        
        # Resource scheduler with fallback
        try:
            if self.enable_resource_scheduling:
                self.resource_scheduler = get_resource_scheduler()
            else:
                self.resource_scheduler = None
        except Exception as e:
            print(f"Warning: Resource scheduler failed ({e}), disabled")
            self.resource_scheduler = None
            self.enable_resource_scheduling = False
        
        # Metrics system with fallback  
        try:
            metrics_config = {
                'enabled': True,
                'aggregation_window_seconds': 300,
                'enable_background_processing': False,  # Disabled for safety
                'alerting': {'enabled': False}  # Disabled for safety
            }
            self.metrics_collector = initialize_metrics_system(metrics_config, session_id=str(uuid.uuid4())[:8])
        except Exception as e:
            print(f"Warning: Metrics system failed ({e}), using None fallback")
            self.metrics_collector = None
        
        # Set safe defaults for other features
        self.enable_caching = getattr(self, 'cache_manager', None) is not None
        self.enable_selective_reprocessing = getattr(self, 'selective_asr_processor', None) is not None
        self.confidence_threshold_for_flagging = 0.65
        self.max_segments_for_selective_reprocessing = 10
        
        # Source separation and overlap processing
        self.enable_source_separation = False  # Disabled for safety
        self.enable_overlap_aware_processing = False  # Disabled for safety
        
        # Other optional engines (initialized when needed)
        self.punctuation_engine = None
        self.text_normalizer = None
        self.guardrail_verifier = None
        self.post_fusion_realigner = None
        self.elastic_chunker = None
    
    def __init__(self, expected_speakers: int = 10, noise_level: str = 'medium', target_language: Optional[str] = None, scoring_weights: Optional[Dict[str, float]] = None, enable_versioning: bool = True, domain: str = "general", consensus_strategy: str = "best_single_candidate", calibration_method: str = "registry_based", enable_speaker_mapping: bool = True, speaker_mapping_config: Optional[Dict[str, Any]] = None, chunked_processing_threshold: float = 900.0, enable_dialect_handling: bool = True, dialect_similarity_threshold: float = 0.7, dialect_confidence_boost: float = 0.05, supported_dialects: Optional[List[str]] = None, enable_auto_glossary: bool = True, auto_glossary_config: Optional[Dict[str, Any]] = None, project_id: Optional[str] = None, enable_long_horizon_tracking: bool = True, long_horizon_config: Optional[Dict[str, Any]] = None) -> None:
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
        
        # Robust text normalization configuration
        self.enable_text_normalization = True
        self.normalization_profile = "readable"  # Options: verbatim, light, readable, executive
        self.text_normalizer = None  # Will be initialized when needed
        self.guardrail_verifier = None  # Will be initialized when needed
        self.normalization_config_path = "config/normalization_profiles.yaml"
        
        # Post-fusion realigner configuration
        self.enable_post_fusion_realigner = True
        self.post_fusion_realigner = None  # Will be initialized when needed
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
        
        # Elastic chunking configuration and initialization
        self.enable_elastic_chunking = True  # Enable intelligent chunking by default
        self.elastic_chunker = None  # Will be initialized when needed
        
        # Dialect handling configuration
        self.enable_dialect_handling = enable_dialect_handling
        self.dialect_similarity_threshold = dialect_similarity_threshold
        self.dialect_confidence_boost = dialect_confidence_boost
        self.supported_dialects = supported_dialects or ['southern', 'aave', 'nyc', 'boston', 'midwest', 'west_coast']
        
        # Auto-Glossary system configuration
        self.enable_auto_glossary = enable_auto_glossary
        self.project_id = project_id or f"project_{int(time.time())}"
        self.auto_glossary_config = auto_glossary_config or {
            # Term mining configuration
            'mining_sensitivity': 0.6,
            'min_frequency_threshold': 2,
            'max_candidates_per_session': 200,
            'enable_variant_clustering': True,
            # Term store configuration
            'decay_sessions_threshold': 10,
            'minimum_support_threshold': 2,
            'storage_base_path': 'term_bases',
            # Adaptive biasing configuration
            'default_bias_strength': 0.7,
            'max_bias_terms_per_session': 50,
            'min_term_confidence': 0.5,
            'enable_asr_bias': True,
            'enable_fusion_bias': True,
            'enable_repair_constraints': True
        }
        
        # Long-horizon speaker tracking configuration
        self.enable_long_horizon_tracking = enable_long_horizon_tracking
        self.long_horizon_config = long_horizon_config or {
            # Global speaker linking configuration
            'min_turn_duration': 0.5,
            'max_turn_duration': 60.0,
            'embedding_aggregation_method': 'weighted_average',
            'clustering_method': 'hierarchical',
            'information_criterion': ['silhouette', 'calinski_harabasz', 'elbow'],
            'cluster_margin': 0.15,
            'min_cluster_size': 2,
            # Swap detection configuration
            'swap_detection_enabled': True,
            'neighborhood_size': 5,
            'discontinuity_threshold': 0.4,
            'min_block_duration': 5.0,
            'max_block_duration': 300.0,
            'contiguity_threshold': 0.7,
            'swap_correction_threshold': 0.6,
            # Human-friendly labeling
            'enable_human_friendly_names': True,
            'enable_role_assignment': True,
            # Caching and performance
            'enable_embedding_cache': True,
            'cache_memory_limit_mb': 512.0,
            'cache_ttl_seconds': 3600.0
        }
        
        # U7 Upgrade: Initialize deterministic processing and other systems first
        # (run_id will be set later based on input for deterministic processing)
        self.run_id = None  # Will be set in process_video method
        
        # Manifest integrity system (will be initialized in process_video)
        self.manifest_manager: Optional[ManifestManager] = None
        self.enable_manifest_tracking = True  # Enable/disable manifest integrity system
        
        # Initialize U7 systems
        self.cache_manager = get_cache_manager()
        self.deterministic_processor = get_deterministic_processor()
        self.worklist_manager = get_worklist_manager()
        self.selective_asr_processor = get_selective_asr_processor()
        self.asr_scheduler = get_asr_scheduler()
        
        # Initialize resource scheduler with configurable settings
        self.enable_resource_scheduling = True  # Can be configured via config
        if self.enable_resource_scheduling:
            # Initialize resource scheduler with intelligent defaults
            self.resource_scheduler = get_resource_scheduler()
            if not hasattr(self.resource_scheduler, 'session_start_time') or self.resource_scheduler.session_start_time == 0:
                # Configure scheduler based on complexity and user preferences
                downgrade_strategy = DowngradeStrategy.BALANCED  # Default to balanced approach
                global_timeout = 30.0  # 30 minutes default timeout
                
                self.resource_scheduler = initialize_resource_scheduler(
                    global_timeout_minutes=global_timeout,
                    downgrade_strategy=downgrade_strategy,
                    enable_predictive_scheduling=True,
                    enable_resource_monitoring=True
                )
            self.structured_logger.info("Resource scheduler initialized for intelligent budget management",
                                      context={'downgrade_strategy': 'balanced', 'global_timeout': 30.0})
        else:
            self.resource_scheduler = None
            self.structured_logger.info("Resource scheduling disabled")
        
        # U7 configuration
        self.enable_caching = True
        self.enable_selective_reprocessing = True
        self.confidence_threshold_for_flagging = 0.65
        self.max_segments_for_selective_reprocessing = 10
        
        # Source separation configuration
        self.enable_source_separation = True
        self.overlap_probability_threshold = 0.25
        self.source_separation_providers = ['faster-whisper', 'deepgram', 'openai']
        
        # Overlap-aware processing configuration
        self.enable_overlap_aware_processing = True
        self.overlap_frame_threshold = 0.08  # Trigger separation when >8% of frames are overlapped
        self.max_stems = 2  # Focus on 2-stem separation for focus groups
        self.stem_quality_threshold = 0.5  # Minimum stem quality for processing
        self.enable_cross_stem_reconciliation = True
        self.reconciliation_strategy = "highest_joint_score"
        
        # Overlap processing budget controls
        self.max_overlap_processing_ratio = 0.15  # Process up to 15% of total audio duration
        self.max_overlap_processing_duration = 600.0  # Maximum 10 minutes of overlap processing
        
        # Stem manifest configuration
        self.enable_stem_manifest = True
        self.stem_cache_base_path = "/tmp/stem_cache"
        
        # Initialize enhanced observability system
        self.obs_manager = initialize_observability(
            service_name="ensemble-transcription",
            enable_profiling=True,
            log_level="INFO"
        )
        
        # Initialize comprehensive metrics system
        metrics_config = {
            'enabled': True,
            'aggregation_window_seconds': 300,
            'enable_background_processing': True,
            'background_processing_interval': 60,
            'alerting': {
                'enabled': True,
                'alert_destinations': {
                    'console': {'enabled': True},
                    'file_logging': {'enabled': True, 'path': 'logs/alerts'}
                }
            }
        }
        self.metrics_collector = initialize_metrics_system(metrics_config, session_id=str(uuid.uuid4())[:8])
        
        # Determinism and reproducibility system
        self.run_context: Optional[RunContext] = None
        self.enable_determinism = True
        
        # Initialize enhanced structured logging with run context
        self.structured_logger = create_enhanced_logger("ensemble_manager", run_id=self.run_id)
        
        # Initialize capability manager and perform startup checks
        self.capability_manager = get_capability_manager()
        self.capability_report = check_system_capabilities()
        self.audio_validator = get_audio_validator()
        
        # Log capability assessment
        self._log_startup_capabilities()
        
        # Adjust configuration based on available capabilities
        self._adjust_configuration_for_capabilities()
        
    def _log_startup_capabilities(self):
        """Log startup capability assessment"""
        if not self.capability_report:
            return
            
        self.structured_logger.info("🚀 ENSEMBLE MANAGER CAPABILITY ASSESSMENT", context={
            'system_status': self.capability_report.system_status.value,
            'available_features': len(self.capability_report.available_features),
            'degraded_features': len(self.capability_report.degraded_features),
            'fallback_features': len(self.capability_report.fallback_features),
            'unavailable_features': len(self.capability_report.unavailable_features)
        })
        
        if self.capability_report.critical_missing:
            self.structured_logger.error("⚠️ CRITICAL DEPENDENCIES MISSING", context={
                'missing_dependencies': self.capability_report.critical_missing,
                'impact': 'System functionality may be severely limited'
            })
    
    def _adjust_configuration_for_capabilities(self):
        """Adjust configuration based on available capabilities"""
        
        # Disable features that require unavailable dependencies
        if not is_feature_available('advanced_speaker_diarization'):
            self.structured_logger.warning("Advanced speaker diarization unavailable, using basic clustering")
            # Keep diarization enabled but it will use fallback implementation
        
        if not is_feature_available('source_separation'):
            self.structured_logger.warning("Source separation unavailable, disabling overlap processing")
            self.enable_source_separation = False
            self.enable_overlap_aware_processing = False
            
        if not is_feature_available('text_normalization'):
            self.structured_logger.warning("Advanced text normalization unavailable, using basic rules")
            # Keep normalization enabled but it will use fallback
            
        if not is_feature_available('adaptive_biasing'):
            self.structured_logger.warning("Adaptive biasing unavailable, disabling auto-glossary")
            self.enable_auto_glossary = False
            
        if not is_feature_available('long_horizon_tracking'):
            self.structured_logger.warning("Long-horizon tracking unavailable, using per-chunk mapping only")
            self.enable_long_horizon_tracking = False
            
        # Log adjusted configuration
        self.structured_logger.info("Configuration adjusted for available capabilities", context={
            'source_separation_enabled': self.enable_source_separation,
            'overlap_processing_enabled': self.enable_overlap_aware_processing,
            'auto_glossary_enabled': self.enable_auto_glossary,
            'long_horizon_tracking_enabled': self.enable_long_horizon_tracking
        })
        
        # Initialize profiling manager and reporter
        self.profiling_manager = get_profiling_manager()
        self.observability_reporter = get_observability_reporter()
        
        # Initialize disagreement re-decode system
        self.enable_disagreement_redecode = True  # Default enabled
        self.disagreement_redecode_engine = None
        self.redecode_config = None
        
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
        
        # Initialize disagreement re-decode engine if enabled
        if self.enable_disagreement_redecode:
            try:
                # Load re-decode configuration from hydra config (if available)
                redecode_config = None
                
                # Initialize with default configuration 
                self.disagreement_redecode_engine = create_disagreement_redecode_engine(redecode_config)
                self.structured_logger.info("Disagreement re-decode engine initialized successfully")
            except Exception as e:
                self.structured_logger.warning(f"Failed to initialize disagreement re-decode engine: {e}")
                self.enable_disagreement_redecode = False
                self.disagreement_redecode_engine = None
        
        # Initialize components with versioning context
        self.audio_processor = AudioProcessor()
        self.diarization_engine = DiarizationEngine(
            expected_speakers=self.expected_speakers, 
            noise_level=self.noise_level,
            enable_speaker_mapping=self.enable_speaker_mapping,
            speaker_mapping_config=self.speaker_mapping_config
        )
        self.asr_engine = ASREngine()
        self.scoring_weights = scoring_weights or {}
        self.confidence_scorer = ConfidenceScorer(
            scoring_weights=self.scoring_weights,
            use_registry_calibration=self.enable_versioning,
            domain=self.domain,
            speaker_count=self.expected_speakers,
            noise_level=self.noise_level,
            calibration_method=self.calibration_method
        )
        self.transcript_formatter = TranscriptFormatter()
        
        # Initialize source separation engine
        try:
            self.source_separation_engine = SourceSeparationEngine(
                overlap_threshold=self.overlap_probability_threshold,
                enable_caching=self.enable_caching,
                max_stems=self.max_stems,
                max_separation_ratio=self.max_overlap_processing_ratio,
                max_separation_duration=self.max_overlap_processing_duration
            )
            if self.source_separation_engine.is_available():
                self.structured_logger.info("Source separation engine initialized successfully")
            else:
                self.structured_logger.warning("Source separation engine not available - Demucs models unavailable")
                self.enable_source_separation = False
                self.enable_overlap_aware_processing = False
        except Exception as e:
            self.structured_logger.warning(f"Failed to initialize source separation engine: {e}")
            self.source_separation_engine = None
            self.enable_source_separation = False
            self.enable_overlap_aware_processing = False
        
        # Initialize overlap-aware processing engines
        self.overlap_diarization_engine = None
        self.overlap_fusion_engine = None
        self.stem_manifest_manager = None
        
        # Initialize long-horizon speaker tracking components
        if self.enable_long_horizon_tracking:
            try:
                # Initialize embedding cache
                self.embedding_cache = get_embedding_cache(
                    max_memory_mb=self.long_horizon_config['cache_memory_limit_mb'],
                    enable_disk_cache=self.long_horizon_config['enable_embedding_cache'],
                    ttl_seconds=self.long_horizon_config['cache_ttl_seconds']
                )
                
                # Initialize global speaker linker
                self.global_speaker_linker = GlobalSpeakerLinker(
                    speaker_mapper=self.diarization_engine.speaker_mapper if hasattr(self.diarization_engine, 'speaker_mapper') else None,
                    min_turn_duration=self.long_horizon_config['min_turn_duration'],
                    max_turn_duration=self.long_horizon_config['max_turn_duration'],
                    embedding_aggregation_method=self.long_horizon_config['embedding_aggregation_method'],
                    clustering_method=self.long_horizon_config['clustering_method'],
                    information_criterion=self.long_horizon_config['information_criterion'],
                    cluster_margin=self.long_horizon_config['cluster_margin'],
                    min_cluster_size=self.long_horizon_config['min_cluster_size'],
                    enable_caching=self.long_horizon_config['enable_embedding_cache']
                )
                
                # Initialize speaker relabeler with swap detection
                from core.speaker_relabeler import SwapDetector
                swap_detector = SwapDetector(
                    neighborhood_size=self.long_horizon_config['neighborhood_size'],
                    discontinuity_threshold=self.long_horizon_config['discontinuity_threshold'],
                    min_block_duration=self.long_horizon_config['min_block_duration'],
                    max_block_duration=self.long_horizon_config['max_block_duration'],
                    contiguity_threshold=self.long_horizon_config['contiguity_threshold']
                ) if self.long_horizon_config['swap_detection_enabled'] else None
                
                self.speaker_relabeler = SpeakerRelabeler(
                    swap_detector=swap_detector,
                    swap_correction_threshold=self.long_horizon_config['swap_correction_threshold'],
                    enable_overlap_preservation=True,
                    enable_human_friendly_names=self.long_horizon_config['enable_human_friendly_names']
                )
                
                self.structured_logger.info("Long-horizon speaker tracking initialized successfully",
                                           context={
                                               'embedding_cache_enabled': self.long_horizon_config['enable_embedding_cache'],
                                               'swap_detection_enabled': self.long_horizon_config['swap_detection_enabled'],
                                               'clustering_method': self.long_horizon_config['clustering_method']
                                           })
                
            except Exception as e:
                self.structured_logger.warning(f"Failed to initialize long-horizon speaker tracking: {e}")
                self.enable_long_horizon_tracking = False
                self.global_speaker_linker = None
                self.speaker_relabeler = None
                self.embedding_cache = None
        else:
            self.global_speaker_linker = None
            self.speaker_relabeler = None
            self.embedding_cache = None
        
        if self.enable_overlap_aware_processing and self.source_separation_engine:
            try:
                # Initialize overlap diarization engine
                self.overlap_diarization_engine = OverlapDiarizationEngine(
                    enable_cross_stem_analysis=self.enable_cross_stem_reconciliation,
                    enable_performance_optimizations=True
                )
                
                # Initialize overlap fusion engine
                self.overlap_fusion_engine = OverlapFusionEngine(
                    conflict_resolution_strategy=self.reconciliation_strategy
                )
                
                # Initialize stem manifest manager
                if self.enable_stem_manifest:
                    self.stem_manifest_manager = StemManifestManager(
                        cache_base_path=self.stem_cache_base_path,
                        enable_quality_analysis=True
                    )
                
                self.structured_logger.info("Overlap-aware processing engines initialized successfully",
                                          context={
                                              'max_stems': self.max_stems,
                                              'reconciliation_strategy': self.reconciliation_strategy,
                                              'stem_manifest_enabled': self.enable_stem_manifest
                                          })
                
            except Exception as e:
                self.structured_logger.warning(f"Failed to initialize overlap processing engines: {e}")
                self.enable_overlap_aware_processing = False
                self.overlap_diarization_engine = None
                self.overlap_fusion_engine = None
                self.stem_manifest_manager = None
        
        # Initialize post-fusion punctuation engine if enabled
        if self.enable_post_fusion_punctuation:
            try:
                self.punctuation_engine = create_punctuation_engine_from_preset(self.punctuation_preset)
                self.structured_logger.info(f"Post-fusion punctuation engine initialized with preset: {self.punctuation_preset}")
            except Exception as e:
                self.structured_logger.warning(f"Failed to initialize punctuation engine: {e}")
                self.enable_post_fusion_punctuation = False
                self.punctuation_engine = None
        
        # Initialize robust text normalization engine if enabled
        if self.enable_text_normalization:
            try:
                self.text_normalizer = create_text_normalizer(self.normalization_config_path)
                self.guardrail_verifier = create_guardrail_verifier()
                self.structured_logger.info(f"Text normalization engine initialized with profile: {self.normalization_profile}")
            except Exception as e:
                self.structured_logger.warning(f"Failed to initialize text normalization engine: {e}")
                self.enable_text_normalization = False
                self.text_normalizer = None
                self.guardrail_verifier = None
        
        # Initialize dialect handling engine with proper config loading
        try:
            # Load dialect configuration from YAML
            dialect_config = load_dialect_config()
            
            # Override enable setting if explicitly disabled
            if not self.enable_dialect_handling:
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
        
        # Initialize auto-glossary system components
        self.term_mining_engine = None
        self.term_store = None
        self.adaptive_biasing_engine = None
        self.current_session_bias_list = None
        
        if self.enable_auto_glossary:
            try:
                # Initialize term mining engine
                self.term_mining_engine = create_term_mining_engine(
                    session_id="default_session",  # Will be set per session
                    mining_sensitivity=self.auto_glossary_config['mining_sensitivity'],
                    min_frequency_threshold=self.auto_glossary_config['min_frequency_threshold'],
                    max_candidates_per_session=self.auto_glossary_config['max_candidates_per_session'],
                    enable_variant_clustering=self.auto_glossary_config['enable_variant_clustering']
                )
                
                # Initialize term store
                self.term_store = create_project_term_store(
                    storage_base_path=self.auto_glossary_config['storage_base_path'],
                    decay_sessions_threshold=self.auto_glossary_config['decay_sessions_threshold'],
                    minimum_support_threshold=self.auto_glossary_config['minimum_support_threshold']
                )
                
                # Initialize adaptive biasing engine
                self.adaptive_biasing_engine = create_adaptive_biasing_engine(
                    term_store=self.term_store,
                    default_bias_strength=self.auto_glossary_config['default_bias_strength'],
                    max_bias_terms_per_session=self.auto_glossary_config['max_bias_terms_per_session'],
                    min_term_confidence=self.auto_glossary_config['min_term_confidence'],
                    enable_asr_bias=self.auto_glossary_config['enable_asr_bias'],
                    enable_fusion_bias=self.auto_glossary_config['enable_fusion_bias'],
                    enable_repair_constraints=self.auto_glossary_config['enable_repair_constraints']
                )
                
                self.structured_logger.info("Auto-glossary system initialized successfully",
                                           context={
                                               'project_id': self.project_id,
                                               'mining_sensitivity': self.auto_glossary_config['mining_sensitivity'],
                                               'bias_strength': self.auto_glossary_config['default_bias_strength'],
                                               'max_terms': self.auto_glossary_config['max_bias_terms_per_session']
                                           })
                
            except Exception as e:
                self.structured_logger.warning(f"Failed to initialize auto-glossary system: {e}")
                self.enable_auto_glossary = False
                self.term_mining_engine = None
                self.term_store = None
                self.adaptive_biasing_engine = None
        else:
            self.structured_logger.info("Auto-glossary system disabled")
        
        # Initialize consensus module
        try:
            self.consensus_module = ConsensusModule(default_strategy=self.consensus_strategy)
        except Exception as e:
            # Fallback to best single candidate if consensus module fails
            self.structured_logger.warning(f"Failed to initialize consensus module: {e}")
            self.consensus_module = ConsensusModule(default_strategy="best_single_candidate")
        
        # Working directory for temporary files
        self.work_dir: Optional[str] = None
        self.temp_audio_files: List[str] = []  # Track temp audio files for cleanup
    
    def _initialize_elastic_chunker(self) -> None:
        """
        Initialize elastic chunker with configuration from Hydra config
        """
        if self.elastic_chunker is not None:
            return  # Already initialized
            
        try:
            # Get chunking configuration from Hydra
            try:
                # Simplified hydra config access - use fallback if not available
                config = None
            except Exception:
                config = None
            chunking_config = {}
            
            if config and 'chunking' in config:
                chunking_params = config.chunking
                chunking_config = {
                    'enabled': chunking_params.get('enabled', True),
                    'min_chunk_seconds': chunking_params.get('min_chunk_seconds', 15.0),
                    'max_chunk_seconds': chunking_params.get('max_chunk_seconds', 60.0),
                    'target_chunk_seconds': chunking_params.get('target_chunk_seconds', 30.0),
                    'overlap_threshold': chunking_params.get('overlap_threshold', 0.3),
                    'vad_frame_length_ms': chunking_params.get('vad_frame_length_ms', 25.0),
                    'vad_hop_length_ms': chunking_params.get('vad_hop_length_ms', 10.0),
                    'energy_smoothing_window': chunking_params.get('energy_smoothing_window', 5),
                    'voice_activity_threshold': chunking_params.get('voice_activity_threshold', 0.015),
                    'silence_threshold_db': chunking_params.get('silence_threshold_db', -40.0),
                    'pause_min_duration': chunking_params.get('pause_min_duration', 0.3),
                    'pause_max_duration': chunking_params.get('pause_max_duration', 2.0),
                    'speaker_turn_weight': chunking_params.get('speaker_turn_weight', 1.5),
                    'energy_valley_weight': chunking_params.get('energy_valley_weight', 1.2),
                    'word_boundary_preference': chunking_params.get('word_boundary_preference', 0.8),
                    'high_overlap_threshold': chunking_params.get('high_overlap_threshold', 0.4),
                    'low_overlap_threshold': chunking_params.get('low_overlap_threshold', 0.15),
                    'complexity_analysis_window': chunking_params.get('complexity_analysis_window', 10.0),
                    'boundary_search_window': chunking_params.get('boundary_search_window', 5.0),
                    'enable_parallel_analysis': chunking_params.get('enable_parallel_analysis', True),
                    'cache_audio_analysis': chunking_params.get('cache_audio_analysis', True),
                    'max_analysis_workers': chunking_params.get('max_analysis_workers', 3),
                    'chunk_overlap_seconds': chunking_params.get('chunk_overlap_seconds', 1.0)
                }
                
                # Update instance setting from config
                self.enable_elastic_chunking = chunking_config.get('enabled', True)
                
            # Create the elastic chunker
            self.elastic_chunker = create_elastic_chunker(chunking_config)
            
            self.structured_logger.info("Elastic chunker initialized", 
                                      context={
                                          'enabled': self.enable_elastic_chunking,
                                          'config': chunking_config
                                      })
            
        except Exception as e:
            self.structured_logger.warning(f"Failed to initialize elastic chunker: {e}")
            self.elastic_chunker = None
            self.enable_elastic_chunking = False
        
        # Artifact tracking for run manifest
        self.input_artifacts: Dict[str, Any] = {}
        self.intermediate_artifacts: Dict[str, Any] = {}
        self.output_artifacts: Dict[str, Any] = {}
    
    def _prepare_candidates_for_term_mining(self, candidates: List[Dict[str, Any]], diarization_variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert candidates to format expected by term mining engine
        
        Args:
            candidates: ASR candidates from ensemble processing
            diarization_variants: Diarization variants for speaker information
            
        Returns:
            List of ASR results formatted for term mining
        """
        asr_results_for_mining = []
        
        for i, candidate in enumerate(candidates):
            try:
                # Extract segments from candidate
                segments = candidate.get('aligned_segments', [])
                if not segments:
                    segments = candidate.get('segments', [])
                
                # Extract ASR metadata
                asr_data = candidate.get('asr_data', {})
                engine_name = asr_data.get('provider', f'engine_{i}')
                
                # Convert segments to mining format
                mining_segments = []
                for segment in segments:
                    mining_segment = {
                        'start': segment.get('start', 0.0),
                        'end': segment.get('end', 0.0),
                        'text': segment.get('text', ''),
                        'confidence': segment.get('confidence', 0.0),
                        'speaker_id': segment.get('speaker', segment.get('speaker_id'))
                    }
                    mining_segments.append(mining_segment)
                
                # Create ASR result for mining
                asr_result = {
                    'engine': engine_name,
                    'segments': mining_segments,
                    'candidate_id': candidate.get('candidate_id', f'candidate_{i}'),
                    'confidence': candidate.get('confidence_scores', {}).get('final_score', 0.0)
                }
                
                asr_results_for_mining.append(asr_result)
                
            except Exception as e:
                self.structured_logger.warning(f"Failed to prepare candidate {i} for term mining: {e}")
                continue
        
        return asr_results_for_mining
    
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
    
    def _evaluate_separation_quality_gates(self, separation_results: List[SourceSeparationResult]) -> Tuple[bool, str]:
        """
        Evaluate separation quality gates to determine if separated stems should be used
        
        Args:
            separation_results: List of source separation results with QA metrics
            
        Returns:
            Tuple of (gates_passed: bool, fallback_reason: str)
        """
        try:
            if not separation_results:
                return False, "no_separation_results"
            
            # Collect QA metrics from all separation results
            all_snr_values = []
            all_leakage_values = []
            all_artifact_values = []
            total_overlap_duration = 0
            
            for sep_result in separation_results:
                total_overlap_duration += sep_result.overlap_frame.duration
                
                for stem in sep_result.separated_stems:
                    if stem.quality_metrics:
                        all_snr_values.append(stem.quality_metrics.snr_db)
                        all_leakage_values.append(stem.quality_metrics.leakage_rate)
                        all_artifact_values.append(stem.quality_metrics.artifact_score)
            
            if not all_snr_values:
                return False, "no_qa_metrics_available"
            
            # Get configuration from Hydra config
            try:
                # Simplified hydra config access - use fallback if not available
                config = None
            except Exception:
                config = None
            if not config:
                # Fallback to default thresholds
                min_stem_snr_db = 12.0
                max_stem_leakage = 0.12
                min_overlap_percent = 8.0
            else:
                min_stem_snr_db = config.get('separation', {}).get('min_stem_snr_db', 12.0)
                max_stem_leakage = config.get('separation', {}).get('max_stem_leakage', 0.12)
                min_overlap_percent = config.get('separation', {}).get('min_overlap_percent', 8.0)
            
            # Compute aggregate metrics
            avg_snr_db = sum(all_snr_values) / len(all_snr_values)
            median_leakage_rate = sorted(all_leakage_values)[len(all_leakage_values) // 2]
            avg_artifact_score = sum(all_artifact_values) / len(all_artifact_values)
            
            # Compute overlap ratio (estimated from total duration)
            # This is an approximation - in practice you'd get this from audio analysis
            overlap_ratio_percent = (total_overlap_duration / 60.0) * 100  # Rough estimate
            
            # Gate 1: Average SNR check
            if avg_snr_db < min_stem_snr_db:
                return False, f"avg_snr_too_low_{avg_snr_db:.1f}_below_{min_stem_snr_db}"
            
            # Gate 2: Median leakage rate check
            if median_leakage_rate > max_stem_leakage:
                return False, f"median_leakage_too_high_{median_leakage_rate:.3f}_above_{max_stem_leakage}"
            
            # Gate 3: Overlap ratio check
            if overlap_ratio_percent < min_overlap_percent:
                return False, f"overlap_ratio_too_low_{overlap_ratio_percent:.1f}_below_{min_overlap_percent}"
            
            # Gate 4: Artifact score check
            if avg_artifact_score > 0.5:
                return False, f"avg_artifacts_too_high_{avg_artifact_score:.3f}_above_0.5"
            
            # All gates passed
            self.structured_logger.info("Separation quality gates passed",
                                       context={
                                           'avg_snr_db': avg_snr_db,
                                           'median_leakage_rate': median_leakage_rate,
                                           'avg_artifact_score': avg_artifact_score,
                                           'overlap_ratio_percent': overlap_ratio_percent,
                                           'num_separation_results': len(separation_results),
                                           'total_stems': len(all_snr_values)
                                       })
            
            return True, "quality_gates_passed"
            
        except Exception as e:
            self.structured_logger.error(f"Failed to evaluate separation quality gates: {e}")
            return False, f"gate_evaluation_error_{str(e)}"

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
        pipeline_session_id = str(uuid.uuid4())[:8]
        
        # Record business event for file processing start
        record_business_event('file_processing_start', 'ensemble_manager', 
                             file_path=video_path, session_id=pipeline_session_id)
        
        # Create temporary working directory  
        self.work_dir = tempfile.mkdtemp(prefix='ensemble_transcription_')
        
        # Log pipeline start and start metrics tracking
        self.structured_logger.stage_start("pipeline", "Starting ensemble transcription pipeline", 
                                         context={'video_path': video_path, 'expected_speakers': self.expected_speakers, 'noise_level': self.noise_level})
        
        # Initialize resource scheduler session for intelligent budget management
        scheduler_session_started = False
        complexity_estimate = None
        if self.enable_resource_scheduling and self.resource_scheduler:
            try:
                # Get audio duration for complexity estimation (quick check)
                initial_audio_duration = 0.0
                if video_path.lower().endswith('.wav'):
                    # For WAV files, estimate duration quickly
                    import librosa
                    initial_audio_duration = librosa.get_duration(path=video_path)
                else:
                    # For video files, we'll update this after extraction
                    initial_audio_duration = 300.0  # Default estimate: 5 minutes
                
                # Start scheduler session with initial complexity estimate
                complexity_estimate = self.resource_scheduler.start_session(
                    audio_duration=initial_audio_duration,
                    audio_path=video_path,
                    expected_speakers=self.expected_speakers,
                    noise_level=self.noise_level,
                    target_language=self.target_language
                )
                scheduler_session_started = True
                
                self.structured_logger.info("🎯 Resource scheduler session started with intelligent budget management",
                                          context={
                                              'complexity_score': complexity_estimate.complexity_score,
                                              'recommended_quality': complexity_estimate.recommended_quality_level.value,
                                              'initial_duration_estimate': initial_audio_duration,
                                              'global_timeout_minutes': self.resource_scheduler.global_timeout_minutes
                                          })
                
                # Update progress callback if available
                if progress_callback:
                    progress_callback("SCHED", 2, f"Resource scheduling active (Quality: {complexity_estimate.recommended_quality_level.value})")
                    
            except Exception as e:
                self.structured_logger.warning(f"Failed to start resource scheduler session: {e}")
                scheduler_session_started = False
        
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
                'chunked_processing_threshold': self.chunked_processing_threshold,
                'enable_overlap_aware_processing': self.enable_overlap_aware_processing,
                'overlap_frame_threshold': self.overlap_frame_threshold,
                'max_stems': self.max_stems,
                'reconciliation_strategy': self.reconciliation_strategy
            }
            
            self.run_id = ensure_deterministic_run_id(video_path, processing_config)
            self.structured_logger = create_enhanced_logger("ensemble_manager", run_id=self.run_id)
            
            self.structured_logger.info("U7: Generated deterministic run_id", 
                                      context={'run_id': self.run_id, 'config_hash': str(hash(str(processing_config)))})
            
            # Initialize manifest integrity system
            if self.enable_manifest_tracking:
                try:
                    session_id = f"session_{int(time.time())}"
                    self.manifest_manager = create_manifest_manager(
                        session_dir=self.work_dir,
                        session_id=session_id,
                        project_id=self.project_id,
                        run_id=self.run_id
                    )
                    
                    # Capture model versions for reproducibility
                    model_versions = {
                        'asr': {
                            'openai_whisper': 'whisper-1',
                            'faster_whisper': getattr(self.asr_engine, 'model_version', 'unknown')
                        },
                        'diarization': {
                            'pyannote': getattr(self.diarization_engine, 'model_version', 'pyannote/speaker-diarization-3.1')
                        },
                        'punctuation': {
                            'preset': self.punctuation_preset if self.enable_post_fusion_punctuation else 'disabled'
                        },
                        'source_separation': {
                            'demucs': getattr(self.source_separation_engine, 'model_name', 'htdemucs_6s') if self.enable_source_separation else 'disabled'
                        }
                    }
                    
                    # Set input media and configuration
                    self.manifest_manager.set_input_media(video_path, processing_config, model_versions)
                    
                    self.structured_logger.info("Manifest integrity system initialized", 
                                              context={
                                                  'manifest_path': str(self.manifest_manager.manifest_path),
                                                  'session_id': session_id,
                                                  'run_id': self.run_id
                                              })
                except Exception as e:
                    self.structured_logger.warning(f"Failed to initialize manifest system: {e}")
                    self.manifest_manager = None
                    self.enable_manifest_tracking = False
            
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
                
                # Step 1: Audio Extraction (0-10%) with Resource Scheduling
                if progress_callback:
                    progress_callback("A", 5, "Extracting audio from video...")
                
                # Start resource monitoring for audio extraction stage
                stage_usage = None
                if scheduler_session_started:
                    stage_usage = self.resource_scheduler.start_stage(
                        ProcessingStage.AUDIO_EXTRACTION,
                        metadata={'input_path': video_path, 'processing_type': 'video_extraction'}
                    )
                
                # Start comprehensive metrics tracking for audio extraction
                with track_processing_stage("audio_extraction", "ensemble_manager", 
                                          audio_duration=0) as metrics_tracker:
                    stage_start_time = time.time()
                    self.structured_logger.stage_start("audio_extraction", "Extracting and cleaning audio from video")
                    
                    raw_audio_path, clean_audio_path = self.audio_processor.extract_audio_from_video(video_path)
                    
                    # End resource monitoring for audio extraction
                    if stage_usage and scheduler_session_started:
                        self.resource_scheduler.end_stage(ProcessingStage.AUDIO_EXTRACTION, success=True)
                    self.temp_audio_files.extend([raw_audio_path, clean_audio_path])
                    
                    # Track audio artifacts (versioning)
                    if self.enable_versioning and self.dvc_manager:
                        audio_artifacts = self.dvc_manager.track_audio_artifacts(raw_audio_path, clean_audio_path, self.run_id)
                        self.intermediate_artifacts.update(audio_artifacts)
                    
                    # Track audio artifacts in manifest
                    if self.manifest_manager:
                        try:
                            # Track clean audio as ASR-ready WAV
                            self.manifest_manager.add_artifact(
                                artifact_type="asr_wav",
                                file_path=clean_audio_path,
                                producing_component="AudioProcessor.extract_audio_from_video",
                                input_artifacts=[],  # Depends on input media
                                metadata={
                                    "source": "video_extraction",
                                    "processing_stage": "audio_extraction",
                                    "audio_duration_seconds": self.audio_processor.get_audio_duration(clean_audio_path)
                                }
                            )
                        except Exception as e:
                            self.structured_logger.warning(f"Failed to track audio artifact in manifest: {e}")
                    
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
            elif audio_duration > 28800:  # 8 hours - warn for extremely long content
                if progress_callback:
                    progress_callback("WARN", 15, f"Extremely long audio ({audio_duration/3600:.1f}hrs) - consider chunking for optimal processing")
            elif audio_duration > 7200:  # 2 hours - info for long content but don't truncate
                if progress_callback:
                    progress_callback("INFO", 15, f"Long-form audio ({audio_duration/3600:.1f}hrs) - processing will take extended time")
            
            # Check for silent content
            if self._is_mostly_silent(clean_audio_path):
                if progress_callback:
                    progress_callback("WARN", 15, "Audio appears mostly silent - transcription results may be minimal")
            
            # Step 3: Diarization Variants (15-35%) with Resource Scheduling
            if progress_callback:
                progress_callback("C", 20, "Creating 5 diarization variants with voting fusion...")
            
            # Start resource monitoring for diarization stage
            diarization_usage = None
            if scheduler_session_started:
                diarization_usage = self.resource_scheduler.start_stage(
                    ProcessingStage.DIARIZATION,
                    metadata={
                        'audio_duration': audio_duration,
                        'estimated_noise': estimated_noise,
                        'expected_speakers': self.expected_speakers
                    }
                )
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("diarization", "Creating diarization variants with voting fusion",
                                             context={'audio_duration': audio_duration, 'estimated_noise': estimated_noise})
            
            # Determine if chunked processing should be used (enhanced with elastic chunking)
            enable_chunked_processing = False
            elastic_chunking_result = None
            
            if self.enable_speaker_mapping and audio_duration > self.chunked_processing_threshold:
                # Initialize elastic chunker if needed
                self._initialize_elastic_chunker()
                
                if self.enable_elastic_chunking and self.elastic_chunker:
                    # Use elastic chunking analysis to determine if chunking is beneficial
                    try:
                        # Perform preliminary analysis to determine chunking strategy
                        elastic_chunking_result = self.elastic_chunker.chunk_audio(clean_audio_path)
                        
                        # Enable chunked processing if elastic chunker recommends it
                        enable_chunked_processing = (elastic_chunking_result.total_chunks > 1 and 
                                                   elastic_chunking_result.quality_metrics.get('overall_quality_score', 0) > 0.3)
                        
                        if enable_chunked_processing:
                            self.structured_logger.info(f"🧩 Elastic chunking analysis recommends chunked processing", 
                                                      context={
                                                          'total_chunks': elastic_chunking_result.total_chunks,
                                                          'average_chunk_size': elastic_chunking_result.average_chunk_size,
                                                          'quality_score': elastic_chunking_result.quality_metrics.get('overall_quality_score', 0),
                                                          'boundary_types': elastic_chunking_result.quality_metrics.get('boundary_type_distribution', {})
                                                      })
                        else:
                            self.structured_logger.info("🔄 Elastic chunking analysis suggests keeping audio as single unit",
                                                      context={
                                                          'reason': 'low_benefit_analysis',
                                                          'quality_score': elastic_chunking_result.quality_metrics.get('overall_quality_score', 0)
                                                      })
                    except Exception as e:
                        self.structured_logger.warning(f"Elastic chunking analysis failed, falling back to threshold-based chunking: {e}")
                        # Fallback to original threshold-based logic
                        enable_chunked_processing = True
                else:
                    # Fallback to original threshold-based logic if elastic chunker disabled
                    enable_chunked_processing = True
            
            # Check resource scheduler for potential downgrades
            original_variants_count = 5  # Default
            use_voting_fusion = True
            downgrade_applied = False
            
            if scheduler_session_started and diarization_usage:
                # Check if we should apply preemptive downgrades
                if diarization_usage.quality_level == QualityLevel.FAST:
                    original_variants_count = 3  # Reduce variants
                    downgrade_applied = True
                    if progress_callback:
                        progress_callback("C", 20, "⚡ Fast mode: Creating 3 diarization variants...")
                elif diarization_usage.quality_level == QualityLevel.MINIMAL:
                    original_variants_count = 1  # Single variant only
                    use_voting_fusion = False
                    downgrade_applied = True
                    if progress_callback:
                        progress_callback("C", 20, "⚡ Minimal mode: Creating single diarization variant...")
            
            if enable_chunked_processing:
                self.structured_logger.info(f"🧩 Enabling chunked processing with speaker mapping for {audio_duration/60:.1f}min audio")
                if progress_callback:
                    progress_callback("C", 22, f"Processing {audio_duration/60:.1f}min audio in chunks with speaker mapping...")
            
            if downgrade_applied:
                self.structured_logger.info(f"Resource scheduler downgrade applied to diarization",
                                          context={
                                              'original_variants': 5,
                                              'reduced_variants': original_variants_count,
                                              'voting_fusion_enabled': use_voting_fusion,
                                              'quality_level': diarization_usage.quality_level.value
                                          })
            
            # Prepare chunk boundaries for diarization engine if elastic chunking was used
            chunk_boundaries = None
            if enable_chunked_processing and elastic_chunking_result:
                # Convert elastic chunk boundaries to the format expected by diarization engine
                chunk_boundaries = [boundary.timestamp for boundary in elastic_chunking_result.boundaries]
                
                self.structured_logger.info("🧩 Passing elastic chunk boundaries to diarization engine", 
                                          context={
                                              'chunk_count': len(chunk_boundaries),
                                              'boundaries': chunk_boundaries[:5],  # Log first 5 boundaries
                                              'boundary_types': [b.boundary_type for b in elastic_chunking_result.boundaries[:5]]
                                          })
            
            diarization_variants = self.diarization_engine.create_diarization_variants(
                clean_audio_path, 
                use_voting_fusion=use_voting_fusion,
                enable_chunked_processing=enable_chunked_processing,
                max_variants=original_variants_count,  # Limit variants if downgrade applied
                chunk_boundaries=chunk_boundaries  # Pass elastic chunk boundaries if available
            )
            
            # Track diarization artifacts (versioning)
            if self.enable_versioning and self.dvc_manager:
                diarization_artifacts = self.dvc_manager.track_diarization_artifacts(diarization_variants, self.run_id)
                self.intermediate_artifacts.update(diarization_artifacts)
            
            # Track diarization results in manifest
            if self.manifest_manager:
                try:
                    # Save diarization results to JSON file and track in manifest
                    diarization_results_path = os.path.join(self.work_dir, "diarization_results.json")
                    with open(diarization_results_path, 'w') as f:
                        json.dump({
                            'variants': diarization_variants,
                            'metadata': {
                                'audio_duration': audio_duration,
                                'estimated_noise': estimated_noise,
                                'chunked_processing_enabled': enable_chunked_processing,
                                'variants_count': len(diarization_variants)
                            }
                        }, f, indent=2)
                    
                    # Get clean audio artifact SHA256 as input dependency
                    clean_audio_artifacts = self.manifest_manager.get_artifacts_by_type("asr_wav")
                    input_artifacts = [a.sha256 for a in clean_audio_artifacts] if clean_audio_artifacts else []
                    
                    self.manifest_manager.add_artifact(
                        artifact_type="diarization_json",
                        file_path=diarization_results_path,
                        producing_component="DiarizationEngine.create_diarization_variants",
                        input_artifacts=input_artifacts,
                        metadata={
                            "processing_stage": "diarization",
                            "variants_count": len(diarization_variants),
                            "chunked_processing": enable_chunked_processing,
                            "voting_fusion_enabled": True
                        }
                    )
                except Exception as e:
                    self.structured_logger.warning(f"Failed to track diarization results in manifest: {e}")
            
            diarization_time = time.time() - stage_start_time
            
            # End resource monitoring for diarization stage
            if diarization_usage and scheduler_session_started:
                usage_result = self.resource_scheduler.end_stage(ProcessingStage.DIARIZATION, success=True)
                
                # Check if budget was exceeded and apply reactive downgrades if needed
                if usage_result.budget_exceeded:
                    self.structured_logger.warning("Diarization exceeded RTF budget - future stages may be downgraded",
                                                  context={
                                                      'actual_rtf': usage_result.actual_rtf,
                                                      'target_rtf': self.resource_scheduler.rtf_budgets[ProcessingStage.DIARIZATION].target_rtf
                                                  })
            
            self.structured_logger.stage_complete("diarization", "Diarization variants created", 
                                                duration=diarization_time,
                                                metrics={'variants_created': len(diarization_variants)})
            
            if progress_callback:
                progress_callback("C", 35, f"Diarization complete - {len(diarization_variants)} variants with fusion created")
            
            # Step 3.5: Overlap-Aware Processing with Source Separation, Diarization, and Fusion (35-40%)
            overlap_processing_results = []
            source_separation_patches_applied = 0
            
            if self.enable_overlap_aware_processing and self.source_separation_engine and self.overlap_diarization_engine:
                if progress_callback:
                    progress_callback("C2", 36, "Running overlap-aware processing: separation, diarization, and fusion...")
                
                overlap_processing_start_time = time.time()
                self.structured_logger.stage_start("overlap_aware_processing", "Processing overlaps with full separation-diarization-fusion pipeline",
                                                 context={
                                                     'overlap_threshold': self.overlap_probability_threshold,
                                                     'max_stems': self.max_stems,
                                                     'reconciliation_strategy': self.reconciliation_strategy
                                                 })
                
                # Process each diarization variant through the overlap-aware pipeline
                total_overlap_frames = 0
                total_fusion_results = 0
                
                for variant_idx, diarization_variant in enumerate(diarization_variants):
                    try:
                        # Extract segments from diarization variant
                        variant_segments = diarization_variant.get('segments', [])
                        
                        if variant_segments:
                            # Step 1: Source Separation - Detect and separate overlap frames
                            separation_results = self.source_separation_engine.process_audio_with_overlaps(
                                clean_audio_path,
                                variant_segments,
                                asr_providers=self.source_separation_providers
                            )
                            
                            # Step 1.5: Quality Gating - Evaluate stem quality and decide routing
                            if separation_results:
                                gates_passed, fallback_reason = self._evaluate_separation_quality_gates(separation_results)
                                
                                if not gates_passed:
                                    self.structured_logger.warning(f"Separation quality gates failed: {fallback_reason}",
                                                                 context={
                                                                     'variant_idx': variant_idx,
                                                                     'num_separation_results': len(separation_results),
                                                                     'fallback_reason': fallback_reason
                                                                 })
                                    # Route to single-channel fallback path - clear separation results
                                    separation_results = []
                            
                            total_overlap_frames += len(separation_results)
                            
                            if separation_results:
                                # Step 2: Overlap-Aware Processing for each separation result
                                variant_overlap_results = []
                                
                                for separation_result in separation_results:
                                    try:
                                        # Step 2a: Overlap Diarization - Process separated stems
                                        diarization_result = self.overlap_diarization_engine.process_source_separation_result(
                                            separation_result
                                        )
                                        
                                        # Step 2b: Create stem manifest if enabled
                                        stem_manifest = None
                                        if self.stem_manifest_manager and separation_result.separated_stems:
                                            try:
                                                stem_manifest = self.stem_manifest_manager.create_stem_manifest(
                                                    overlap_frame_id=f"overlap_{variant_idx}_{separation_result.overlap_frame.start_time:.3f}",
                                                    separated_stems=separation_result.separated_stems,
                                                    asr_engine_ids=self.source_separation_providers[:2],  # Limit to 2 engines for performance
                                                    source_separation_config={
                                                        'model_name': getattr(self.source_separation_engine, 'model_name', 'htdemucs'),
                                                        'overlap_threshold': self.overlap_probability_threshold,
                                                        'max_stems': self.max_stems
                                                    },
                                                    start_time=separation_result.overlap_frame.start_time,
                                                    end_time=separation_result.overlap_frame.end_time
                                                )
                                            except Exception as e:
                                                self.structured_logger.warning(f"Failed to create stem manifest: {e}")
                                        
                                        # Step 2c: Overlap Fusion - Fuse multi-stem transcriptions
                                        if self.overlap_fusion_engine and separation_result.stem_transcriptions:
                                            fusion_result = self.overlap_fusion_engine.fuse_overlap_transcriptions(
                                                diarization_result,
                                                separation_result.stem_transcriptions
                                            )
                                            
                                            variant_overlap_results.append({
                                                'separation_result': separation_result,
                                                'diarization_result': diarization_result,
                                                'fusion_result': fusion_result,
                                                'stem_manifest': stem_manifest
                                            })
                                            
                                            total_fusion_results += 1
                                            
                                            self.structured_logger.info(
                                                f"Completed overlap processing for frame {separation_result.overlap_frame.start_time:.3f}-{separation_result.overlap_frame.end_time:.3f}",
                                                context={
                                                    'fusion_confidence': fusion_result.fusion_confidence,
                                                    'overlap_regions': fusion_result.overlap_regions_count,
                                                    'reconciliation_conflicts': fusion_result.reconciliation_conflicts
                                                }
                                            )
                                        else:
                                            # Fallback to original source separation behavior
                                            variant_overlap_results.append({
                                                'separation_result': separation_result,
                                                'diarization_result': diarization_result,
                                                'fusion_result': None,
                                                'stem_manifest': stem_manifest,
                                                'fallback_applied': True
                                            })
                                    
                                    except Exception as e:
                                        self.structured_logger.warning(
                                            f"Overlap processing failed for separation result {separation_result.overlap_frame.start_time:.3f}: {e}"
                                        )
                                        continue
                                
                                # Apply overlap processing results to diarization variant
                                if variant_overlap_results:
                                    patched_segments = apply_overlap_processing_patches(
                                        variant_segments, variant_overlap_results
                                    )
                                    
                                    # Update the diarization variant
                                    diarization_variant['segments'] = patched_segments
                                    diarization_variant['overlap_processing_applied'] = True
                                    diarization_variant['overlap_processing_metadata'] = {
                                        'overlap_frames_processed': len(separation_results),
                                        'fusion_results_count': len([r for r in variant_overlap_results if r.get('fusion_result')]),
                                        'fallback_results_count': len([r for r in variant_overlap_results if r.get('fallback_applied')]),
                                        'processing_timestamp': time.time(),
                                        'stem_manifests_created': len([r for r in variant_overlap_results if r.get('stem_manifest')])
                                    }
                                    
                                    overlap_processing_results.extend(variant_overlap_results)
                                    
                                    self.structured_logger.info(
                                        f"Applied overlap processing to variant {variant_idx}",
                                        context={
                                            'variant_id': diarization_variant.get('variant_id', variant_idx),
                                            'original_segments': len(variant_segments),
                                            'patched_segments': len(patched_segments),
                                            'overlap_results': len(variant_overlap_results)
                                        }
                                    )
                    
                    except Exception as e:
                        self.structured_logger.warning(f"Overlap processing failed for variant {variant_idx}: {e}")
                        # Ensure variant is marked as unmodified
                        diarization_variant['overlap_processing_applied'] = False
                        continue
                
                # Clean up temporary files
                if hasattr(self.source_separation_engine, 'cleanup_temp_files'):
                    try:
                        all_separation_results = [r['separation_result'] for r in overlap_processing_results]
                        self.source_separation_engine.cleanup_temp_files(all_separation_results)
                    except Exception as e:
                        self.structured_logger.warning(f"Failed to cleanup overlap processing temp files: {e}")
                
                overlap_processing_time = time.time() - overlap_processing_start_time
                source_separation_patches_applied = total_fusion_results
                
                self.structured_logger.stage_complete("overlap_aware_processing", "Overlap-aware processing completed",
                                                    duration=overlap_processing_time,
                                                    metrics={
                                                        'overlap_frames_processed': total_overlap_frames,
                                                        'fusion_results_generated': total_fusion_results,
                                                        'variants_processed': len(diarization_variants),
                                                        'stem_manifests_created': len([r for r in overlap_processing_results if r.get('stem_manifest')])
                                                    })
                
                if progress_callback:
                    progress_callback("C2", 40, f"Overlap processing complete - {total_fusion_results} overlap regions processed with full pipeline")
                
                # Step 3.5: Long-Horizon Speaker Tracking (40-45%)
                if self.enable_long_horizon_tracking and overlap_processing_results:
                    if progress_callback:
                        progress_callback("C3", 41, "Running long-horizon speaker tracking and relabeling...")
                    
                    stage_start_time = time.time()
                    self.structured_logger.stage_start("long_horizon_speaker_tracking", 
                                                      "Processing long-horizon speaker tracking",
                                                      context={
                                                          'diarization_variants': len(diarization_variants),
                                                          'overlap_results': len(overlap_processing_results),
                                                          'session_id': self.run_id
                                                      })
                    
                    try:
                        # Process each diarization variant that had overlap processing applied
                        long_horizon_results = []
                        
                        for variant_idx, diarization_variant in enumerate(diarization_variants):
                            if not diarization_variant.get('overlap_processing_applied', False):
                                continue
                            
                            # Find overlap processing results for this variant
                            variant_overlap_results = [r for r in overlap_processing_results 
                                                     if r.get('diarization_variant_id') == variant_idx]
                            
                            if not variant_overlap_results:
                                continue
                            
                            # Extract fusion results for processing
                            fusion_results = [r.get('fusion_result') for r in variant_overlap_results 
                                            if r.get('fusion_result')]
                            
                            if not fusion_results:
                                continue
                            
                            self.structured_logger.info(f"Processing long-horizon tracking for variant {variant_idx}",
                                                       context={
                                                           'fusion_results': len(fusion_results),
                                                           'variant_id': diarization_variant.get('variant_id', variant_idx)
                                                       })
                            
                            # Process the first fusion result for this variant (most representative)
                            fusion_result = fusion_results[0]
                            
                            try:
                                # Step 1: Global speaker clustering
                                clustering_result = self.global_speaker_linker.process_fusion_result(
                                    fusion_result=fusion_result,
                                    session_id=self.run_id,
                                    audio_path=clean_audio_path,
                                    user_hint_k=self.expected_speakers if self.expected_speakers > 0 else None
                                )
                                
                                # Step 2: Speaker relabeling and swap detection
                                if clustering_result.clusters:
                                    # Extract turn embeddings for swap detection
                                    turn_embeddings = self.global_speaker_linker.extract_turn_embeddings(
                                        fusion_result, self.run_id, clean_audio_path
                                    )
                                    
                                    relabeling_result = self.speaker_relabeler.relabel_speakers(
                                        fusion_result=fusion_result,
                                        clustering_result=clustering_result,
                                        turn_embeddings=turn_embeddings,
                                        session_id=self.run_id
                                    )
                                    
                                    # Store results
                                    long_horizon_result = {
                                        'diarization_variant_id': variant_idx,
                                        'clustering_result': clustering_result,
                                        'relabeling_result': relabeling_result,
                                        'speaker_mapping': relabeling_result.global_speaker_mapping,
                                        'speaker_names': relabeling_result.speaker_display_names,
                                        'swap_corrections': len(relabeling_result.swaps_corrected),
                                        'processing_metadata': {
                                            'clusters_found': len(clustering_result.clusters),
                                            'optimal_k': clustering_result.optimal_k,
                                            'clustering_confidence': clustering_result.clustering_confidence,
                                            'speaker_consistency': relabeling_result.speaker_consistency_score,
                                            'der_improvement': relabeling_result.der_improvement_estimate
                                        }
                                    }
                                    
                                    long_horizon_results.append(long_horizon_result)
                                    
                                    self.structured_logger.info("Long-horizon processing completed for variant",
                                                               context={
                                                                   'global_speakers': len(clustering_result.clusters),
                                                                   'swaps_corrected': len(relabeling_result.swaps_corrected),
                                                                   'der_improvement': relabeling_result.der_improvement_estimate,
                                                                   'validation_passed': relabeling_result.validation_passed
                                                               })
                                    
                                    # Apply speaker mapping to the diarization variant
                                    if relabeling_result.validation_passed and relabeling_result.global_speaker_mapping:
                                        updated_segments = []
                                        for seg in diarization_variant.get('segments', []):
                                            updated_seg = seg.copy()
                                            original_speaker = seg.get('speaker_id', '')
                                            
                                            if original_speaker in relabeling_result.global_speaker_mapping:
                                                updated_seg['speaker_id'] = relabeling_result.global_speaker_mapping[original_speaker]
                                                updated_seg['original_speaker_id'] = original_speaker
                                                updated_seg['long_horizon_processed'] = True
                                                updated_seg['global_speaker_metadata'] = {
                                                    'display_name': relabeling_result.speaker_display_names.get(
                                                        relabeling_result.global_speaker_mapping[original_speaker], 
                                                        f"Speaker {relabeling_result.global_speaker_mapping[original_speaker]}"
                                                    ),
                                                    'speaker_role': relabeling_result.speaker_roles.get(
                                                        relabeling_result.global_speaker_mapping[original_speaker], 
                                                        'Participant'
                                                    )
                                                }
                                            
                                            updated_segments.append(updated_seg)
                                        
                                        # Update the variant with global speaker IDs
                                        diarization_variant['segments'] = updated_segments
                                        diarization_variant['long_horizon_processed'] = True
                                        diarization_variant['long_horizon_metadata'] = long_horizon_result['processing_metadata']
                                        diarization_variant['speaker_mapping'] = relabeling_result.global_speaker_mapping
                                        diarization_variant['speaker_display_names'] = relabeling_result.speaker_display_names
                            
                            except Exception as e:
                                self.structured_logger.warning(f"Long-horizon processing failed for variant {variant_idx}: {e}")
                                continue
                        
                        long_horizon_processing_time = time.time() - stage_start_time
                        
                        self.structured_logger.stage_complete("long_horizon_speaker_tracking", 
                                                            "Long-horizon speaker tracking completed",
                                                            duration=long_horizon_processing_time,
                                                            metrics={
                                                                'variants_processed': len([v for v in diarization_variants if v.get('long_horizon_processed', False)]),
                                                                'total_swaps_corrected': sum(r.get('swap_corrections', 0) for r in long_horizon_results),
                                                                'average_der_improvement': sum(r.get('processing_metadata', {}).get('der_improvement', 0) for r in long_horizon_results) / max(len(long_horizon_results), 1),
                                                                'average_speaker_consistency': sum(r.get('processing_metadata', {}).get('speaker_consistency', 0) for r in long_horizon_results) / max(len(long_horizon_results), 1)
                                                            })
                        
                        if progress_callback:
                            total_swaps = sum(r.get('swap_corrections', 0) for r in long_horizon_results)
                            progress_callback("C3", 45, f"Long-horizon tracking complete - {len(long_horizon_results)} variants processed, {total_swaps} swaps corrected")
                    
                    except Exception as e:
                        self.structured_logger.warning(f"Long-horizon speaker tracking failed: {e}")
                        if progress_callback:
                            progress_callback("C3", 45, "Long-horizon tracking skipped due to error")
            
            elif self.enable_source_separation and self.source_separation_engine:
                # Fallback to legacy source separation behavior
                if progress_callback:
                    progress_callback("C2", 36, "Running legacy source separation with timeline patching...")
                
                source_separation_patches_applied = run_legacy_source_separation(
                    self.source_separation_engine,
                    clean_audio_path, 
                    diarization_variants,
                    self.source_separation_providers,
                    self.overlap_probability_threshold,
                    self.structured_logger,
                    progress_callback
                )
                
                if progress_callback:
                    progress_callback("C2", 40, f"Legacy source separation complete - {source_separation_patches_applied} overlap intervals replaced")
            
            # Step 4: ASR Ensemble (40-75%) with Resource Scheduling
            if progress_callback:
                progress_callback("D", 40, "Running expanded ASR ensemble (5 passes per diarization)...")
            
            # Start resource monitoring for ASR ensemble stage
            asr_usage = None
            if scheduler_session_started:
                asr_usage = self.resource_scheduler.start_stage(
                    ProcessingStage.ASR_ENSEMBLE,
                    metadata={
                        'diarization_variants': len(diarization_variants),
                        'target_language': self.target_language,
                        'audio_duration': audio_duration
                    }
                )
                
                # Apply resource-aware downgrade decisions
                if asr_usage.quality_level == QualityLevel.FAST:
                    if progress_callback:
                        progress_callback("D", 40, "⚡ Fast mode: Running reduced ASR ensemble (3 passes per diarization)...")
                elif asr_usage.quality_level == QualityLevel.MINIMAL:
                    if progress_callback:
                        progress_callback("D", 40, "⚡ Minimal mode: Running single ASR engine...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("asr_ensemble", "Running ASR ensemble across all diarization variants",
                                             context={'diarization_variants': len(diarization_variants), 'target_language': self.target_language})
            
            # Run ASR ensemble with potential resource-aware adjustments
            candidates = self.asr_engine.run_asr_ensemble(
                clean_audio_path, 
                diarization_variants, 
                target_language=self.target_language,
                quality_level=asr_usage.quality_level.value if asr_usage else None
            )
            
            # Log overlap processing integration results
            if source_separation_patches_applied > 0:
                if self.enable_overlap_aware_processing:
                    self.structured_logger.info(
                        f"Overlap-aware processing completed - {source_separation_patches_applied} overlap regions processed",
                        context={
                            'total_candidates': len(candidates),
                            'overlap_regions_processed': source_separation_patches_applied,
                            'overlap_processing_results': len(overlap_processing_results),
                            'modified_variants': sum(1 for v in diarization_variants if v.get('overlap_processing_applied', False))
                        }
                    )
                else:
                    self.structured_logger.info(
                        f"Legacy source separation completed - {source_separation_patches_applied} overlap intervals replaced",
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
            
            # End resource monitoring for ASR ensemble stage
            if asr_usage and scheduler_session_started:
                asr_result = self.resource_scheduler.end_stage(ProcessingStage.ASR_ENSEMBLE, success=True)
                
                # Check for budget exceedance and potential future downgrades
                if asr_result.budget_exceeded:
                    self.structured_logger.warning("ASR ensemble exceeded RTF budget - downstream stages may be downgraded",
                                                  context={
                                                      'actual_rtf': asr_result.actual_rtf,
                                                      'target_rtf': self.resource_scheduler.rtf_budgets[ProcessingStage.ASR_ENSEMBLE].target_rtf,
                                                      'candidates_generated': len(candidates)
                                                  })
            
            self.structured_logger.stage_complete("asr_ensemble", "ASR ensemble processing completed", 
                                                duration=asr_time,
                                                metrics={'total_candidates': len(candidates), 'variants_processed': len(diarization_variants)})
            
            # Step 4.5: Disagreement-Triggered Re-decode (75-80%)
            redecode_report = None
            if self.enable_disagreement_redecode and self.disagreement_redecode_engine:
                if progress_callback:
                    progress_callback("D2", 75, "Analyzing disagreement and performing selective re-decode...")
                
                stage_start_time = time.time()
                self.structured_logger.stage_start("disagreement_redecode", "Analyzing candidates for uncertainty and performing selective re-decode",
                                                 context={
                                                     'total_candidates': len(candidates),
                                                     'redecode_enabled': True,
                                                     'audio_duration': audio_duration
                                                 })
                
                try:
                    # Perform disagreement analysis and selective re-decode
                    improved_candidates, redecode_report = self.disagreement_redecode_engine.analyze_and_redecode(
                        candidates=candidates,
                        audio_path=clean_audio_path
                    )
                    
                    # Replace original candidates with improved ones
                    original_candidate_count = len(candidates)
                    candidates = improved_candidates
                    
                    redecode_time = time.time() - stage_start_time
                    
                    # Log re-decode results
                    uncertain_spans_detected = redecode_report.get('uncertain_spans', {}).get('total_count', 0)
                    successful_redecodes = redecode_report.get('redecode_attempts', {}).get('improved_attempts', 0)
                    total_improvement = redecode_report.get('quality_improvements', {}).get('total_improvement_score', 0.0)
                    
                    self.structured_logger.stage_complete("disagreement_redecode", 
                                                        "Disagreement-triggered re-decode completed",
                                                        duration=redecode_time,
                                                        metrics={
                                                            'original_candidates': original_candidate_count,
                                                            'improved_candidates': len(candidates),
                                                            'uncertain_spans_detected': uncertain_spans_detected,
                                                            'successful_redecodes': successful_redecodes,
                                                            'total_improvement_score': total_improvement,
                                                            'processing_time': redecode_time
                                                        })
                    
                    if progress_callback:
                        progress_callback("D2", 80, f"Re-decode complete - {uncertain_spans_detected} uncertain spans, {successful_redecodes} improvements")
                    
                except Exception as e:
                    redecode_time = time.time() - stage_start_time
                    self.structured_logger.error(f"Disagreement re-decode failed: {e}")
                    # Continue with original candidates if re-decode fails
                    redecode_report = {
                        'redecode_enabled': True,
                        'error': str(e),
                        'processing_time': redecode_time
                    }
                    if progress_callback:
                        progress_callback("D2", 80, "Re-decode skipped due to error - using original candidates")
            else:
                if progress_callback:
                    progress_callback("D2", 80, "Re-decode disabled - proceeding with original candidates")
                redecode_report = {'redecode_enabled': False}
            
            if progress_callback:
                progress_callback("D", 75, f"ASR ensemble complete - {len(candidates)} candidates generated (5×5 matrix)")
            
            # Step 4.25: Auto-Glossary Term Mining (75-76%)
            session_term_candidates = []
            if self.enable_auto_glossary and self.term_mining_engine and candidates:
                if progress_callback:
                    progress_callback("D1", 75.5, "Mining domain-specific terms from first-pass hypotheses...")
                
                stage_start_time = time.time()
                self.structured_logger.stage_start("term_mining", "Mining terms from first-pass ASR hypotheses",
                                                 context={'total_candidates': len(candidates), 'project_id': self.project_id})
                
                try:
                    # Convert candidates to format expected by term miner
                    asr_results_for_mining = self._prepare_candidates_for_term_mining(candidates, diarization_variants)
                    
                    # Update term mining engine with current session ID
                    session_id = f"{self.run_id}_{int(time.time())}"
                    self.term_mining_engine.session_id = session_id
                    
                    # Mine terms from first-pass hypotheses
                    mining_results = self.term_mining_engine.mine_terms_from_hypotheses(
                        asr_results=asr_results_for_mining,
                        diarization_results=diarization_variants[0] if diarization_variants else None
                    )
                    
                    session_term_candidates = mining_results.candidates
                    
                    # Export session candidates for debugging/analysis using atomic I/O
                    if session_term_candidates:
                        try:
                            # Use atomic I/O for session candidates export
                            atomic_io = get_atomic_io_manager()
                            temp_subdir = get_run_temp_subdir(self.run_id, TempDirectoryScope.ARTIFACTS)
                            session_candidates_path = temp_subdir / f"{session_id}_term_candidates.json"
                            
                            # Export candidates atomically
                            with atomic_write(session_candidates_path) as f:
                                import json
                                json.dump(mining_results, f, indent=2, ensure_ascii=False)
                            
                            self.structured_logger.debug(f"Exported session candidates atomically: {session_candidates_path}")
                            
                        except Exception as e:
                            # Fallback to legacy method if atomic I/O fails
                            self.structured_logger.warning(f"Atomic export failed, using fallback: {e}")
                            session_candidates_path = f"/tmp/{session_id}_term_candidates.json"
                            self.term_mining_engine.export_session_candidates(mining_results, session_candidates_path)
                    
                    mining_time = time.time() - stage_start_time
                    self.structured_logger.stage_complete("term_mining", "Term mining completed",
                                                        duration=mining_time,
                                                        metrics={
                                                            'candidates_found': mining_results.total_candidates,
                                                            'high_confidence': mining_results.high_confidence_candidates,
                                                            'technical_terms': mining_results.technical_term_count,
                                                            'proper_nouns': mining_results.proper_noun_count
                                                        })
                    
                    # Merge session candidates into project term base
                    if self.term_store and session_term_candidates:
                        candidate_dicts = [{
                            'token': c.token,
                            'weight': c.weight,
                            'supporting_engines': list(c.supporting_engines),
                            'local_context': c.local_context,
                            'scores': {
                                'frequency': c.frequency_score,
                                'case_pattern': c.case_pattern_score,
                                'technical_pattern': c.technical_pattern_score,
                                'multi_speaker': c.multi_speaker_score,
                                'unit_proximity': c.unit_proximity_score,
                                'final_mining': c.final_mining_score
                            },
                            'variants': list(c.variants)
                        } for c in session_term_candidates]
                        
                        merge_result = self.term_store.merge_session_candidates(
                            project_id=self.project_id,
                            session_candidates=candidate_dicts,
                            session_id=session_id
                        )
                        
                        self.structured_logger.info("Session terms merged into project term base",
                                                   context={
                                                       'terms_added': merge_result.terms_added,
                                                       'terms_updated': merge_result.terms_updated,
                                                       'terms_decayed': merge_result.terms_decayed,
                                                       'merge_time': merge_result.total_processing_time
                                                   })
                
                except Exception as e:
                    self.structured_logger.warning(f"Term mining failed: {e}", context={'error_type': type(e).__name__})
                    session_term_candidates = []
            
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
            
            # Step 6: Consensus Processing with Auto-Glossary Biasing (85-90%)
            if progress_callback:
                progress_callback("F", 87, f"Running consensus strategy: {self.consensus_strategy}...")
            
            stage_start_time = time.time()
            self.structured_logger.stage_start("consensus_processing", "Processing consensus from scored candidates",
                                             context={'consensus_strategy': self.consensus_strategy, 'candidates_count': len(scored_candidates)})
            
            # Step 6.1: Generate session bias list for fusion biasing
            if self.enable_auto_glossary and self.adaptive_biasing_engine:
                try:
                    session_id = f"{self.run_id}_{int(time.time())}"
                    self.current_session_bias_list = self.adaptive_biasing_engine.generate_session_bias_list(
                        project_id=self.project_id,
                        session_id=session_id,
                        context_type="meeting"
                    )
                    
                    self.structured_logger.info("Generated session bias list for fusion",
                                               context={
                                                   'bias_terms': self.current_session_bias_list.total_bias_terms,
                                                   'bias_strength': self.current_session_bias_list.bias_strength,
                                                   'term_types': self.current_session_bias_list.term_type_distribution
                                               })
                except Exception as e:
                    self.structured_logger.warning(f"Failed to generate session bias list: {e}")
                    self.current_session_bias_list = None
            
            # Use consensus module for winner selection with adaptive biasing
            consensus_result = self.consensus_module.process_consensus(
                candidates=scored_candidates,
                strategy=self.consensus_strategy,
                session_bias_list=self.current_session_bias_list
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
                progress_callback("G", 90, "Generating final outputs...")
            
            # Step 7.4: Post-Fusion Boundary Realignment (90-92%)
            if self.enable_post_fusion_realigner:
                if progress_callback:
                    progress_callback("G0", 90, "Applying post-fusion boundary realignment...")
                
                stage_start_time = time.time()
                self.structured_logger.stage_start("post_fusion_realignment", "Applying boundary realignment to reduce micro boundary drift")
                
                try:
                    # Initialize realigner if not already done
                    if self.post_fusion_realigner is None:
                        self.post_fusion_realigner = create_post_fusion_realigner()
                    
                    # Convert winner transcript to realigner format
                    winner_words = convert_transcript_to_realigner_format(winner)
                    
                    # Get VAD energy curve from overlap detector if available
                    energy_frames = []
                    if hasattr(self, 'unified_overlap_detector') and self.unified_overlap_detector.last_detection_result:
                        energy_frames = [
                            {
                                'timestamp': frame.timestamp,
                                'energy_level': frame.energy_level,
                                'is_voiced': frame.is_voiced,
                                'is_boundary_candidate': frame.is_boundary_candidate,
                                'metadata': frame.metadata
                            }
                            for frame in self.unified_overlap_detector.last_detection_result.vad_energy_curve
                        ]
                    
                    # Apply boundary realignment
                    realignment_result = self.post_fusion_realigner.realign_boundaries(
                        words=winner_words,
                        energy_frames=energy_frames,
                        audio_duration=audio_duration
                    )
                    
                    # Update winner with realigned boundaries if successful
                    if realignment_result.realignment_applied:
                        winner = convert_realigner_result_to_transcript_format(realignment_result, winner)
                        
                        self.structured_logger.info(
                            f"Post-fusion realignment applied successfully",
                            context={
                                'boundary_shifts_applied': len(realignment_result.boundary_shifts),
                                'mean_shift_ms': realignment_result.mean_shift_ms,
                                'max_shift_ms': realignment_result.max_shift_ms,
                                'energy_alignment_score': realignment_result.energy_alignment_score,
                                'processing_time': realignment_result.processing_time
                            }
                        )
                    else:
                        self.structured_logger.info(
                            f"Post-fusion realignment skipped",
                            context={
                                'fallback_reason': realignment_result.fallback_reason,
                                'processing_time': realignment_result.processing_time
                            }
                        )
                    
                    # Add realignment metrics to processing metadata
                    processing_metadata['post_fusion_realignment'] = {
                        'realignment_applied': realignment_result.realignment_applied,
                        'boundary_shifts_count': len(realignment_result.boundary_shifts),
                        'mean_shift_ms': realignment_result.mean_shift_ms,
                        'max_shift_ms': realignment_result.max_shift_ms,
                        'energy_alignment_score': realignment_result.energy_alignment_score,
                        'processing_time': realignment_result.processing_time,
                        'fallback_reason': realignment_result.fallback_reason
                    }
                    
                    realignment_time = time.time() - stage_start_time
                    self.structured_logger.stage_complete("post_fusion_realignment", "Post-fusion realignment completed",
                                                        duration=realignment_time,
                                                        metrics={
                                                            'realignment_applied': realignment_result.realignment_applied,
                                                            'boundary_shifts': len(realignment_result.boundary_shifts),
                                                            'mean_shift_ms': realignment_result.mean_shift_ms,
                                                            'energy_alignment_score': realignment_result.energy_alignment_score
                                                        })
                    
                except Exception as e:
                    realignment_time = time.time() - stage_start_time
                    self.structured_logger.error(f"Post-fusion realignment failed: {e}")
                    self.structured_logger.stage_complete("post_fusion_realignment", "Post-fusion realignment failed",
                                                        duration=realignment_time, error=str(e))
                    # Continue with original winner on error
            
            # Step 7.5: Text Normalization Processing (92-95%)
            if self.enable_text_normalization and self.text_normalizer:
                if progress_callback:
                    progress_callback("G1", 92, f"Applying robust text normalization (profile: {self.normalization_profile})...")
                
                stage_start_time = time.time()
                self.structured_logger.stage_start("text_normalization", f"Applying robust text normalization with profile: {self.normalization_profile}")
                
                try:
                    # Extract segments from winner for normalization processing
                    winner_segments = winner.get('segments', [])
                    
                    # Apply robust text normalization
                    normalization_results = self.text_normalizer.normalize_segments(
                        winner_segments, profile=self.normalization_profile
                    )
                    
                    # Apply guardrail verification and collect metrics
                    normalized_segments = []
                    total_violations = 0
                    profile_downgrades = 0
                    normalization_metrics = {
                        'segments_processed': len(normalization_results),
                        'tokens_changed': 0,
                        'fillers_removed': 0,
                        'acronyms_protected': 0,
                        'sentences_adjusted': 0,
                        'guardrail_violations': 0,
                        'profile_downgrades': 0,
                        'avg_readability_improvement': 0.0
                    }
                    
                    readability_improvements = []
                    
                    for result in normalization_results:
                        # Validate with guardrails
                        if self.guardrail_verifier:
                            guardrail_result = self.guardrail_verifier.verify_normalization(
                                original_text=result.original_text,
                                normalized_text=result.normalized_text,
                                original_tokens=[{'word': t.text, 'start': t.start_time, 'end': t.end_time} for t in result.original_tokens],
                                normalized_tokens=[{'word': t.text, 'start': t.start_time, 'end': t.end_time} for t in result.normalized_tokens],
                                normalization_changes=result.changes,
                                protected_tokens=[t.text for t in result.original_tokens if t.is_protected],
                                current_profile=result.profile_used
                            )
                            
                            # Log guardrail violations
                            if guardrail_result.violations:
                                total_violations += len(guardrail_result.violations)
                                for violation in guardrail_result.violations:
                                    self.structured_logger.warning(f"Guardrail violation: {violation.rule_name} - {violation.details}")
                        
                        # Create normalized segment
                        segment_dict = {
                            'start': result.original_tokens[0].start_time if result.original_tokens else 0.0,
                            'end': result.original_tokens[-1].end_time if result.original_tokens else 0.0,
                            'speaker': winner_segments[len(normalized_segments)].get('speaker', 'Unknown') if len(normalized_segments) < len(winner_segments) else 'Unknown',
                            'text': result.normalized_text,
                            'original_text': result.original_text,
                            'normalization_confidence': 1.0 - (len(result.guardrail_violations) * 0.1),  # Simple confidence calculation
                            'normalization_applied': len(result.changes) > 0,
                            'profile_used': result.profile_used,
                            'profile_downgrades': result.profile_downgrades,
                            'readability_score': result.readability_score_after,
                            'readability_improvement': result.readability_score_after - result.readability_score_before,
                            'guardrail_violations': result.guardrail_violations,
                            'processing_metadata': {
                                'changes_applied': len(result.changes),
                                'tokens_changed': len([c for c in result.changes if c.change_type in ['capitalization', 'formatting']]),
                                'fillers_removed': len([c for c in result.changes if c.change_type == 'disfluency' and not c.normalized_text]),
                                'processing_time_ms': result.processing_time_ms
                            }
                        }
                        normalized_segments.append(segment_dict)
                        
                        # Update metrics
                        normalization_metrics['tokens_changed'] += segment_dict['processing_metadata']['tokens_changed']
                        normalization_metrics['fillers_removed'] += segment_dict['processing_metadata']['fillers_removed']
                        normalization_metrics['acronyms_protected'] += len([t for t in result.original_tokens if t.is_protected])
                        normalization_metrics['sentences_adjusted'] += len([c for c in result.changes if c.change_type == 'punctuation'])
                        normalization_metrics['guardrail_violations'] += len(result.guardrail_violations)
                        normalization_metrics['profile_downgrades'] += len(result.profile_downgrades)
                        
                        if result.readability_score_after > result.readability_score_before:
                            readability_improvements.append(result.readability_score_after - result.readability_score_before)
                    
                    # Calculate average readability improvement
                    if readability_improvements:
                        normalization_metrics['avg_readability_improvement'] = sum(readability_improvements) / len(readability_improvements)
                    
                    # Update winner candidate with normalized segments
                    winner['segments'] = normalized_segments
                    winner['normalization_metadata'] = {
                        'profile_used': self.normalization_profile,
                        'processing_time': time.time() - stage_start_time,
                        'metrics': normalization_metrics,
                        'guardrail_summary': {
                            'total_violations': total_violations,
                            'segments_with_violations': len([s for s in normalized_segments if s['guardrail_violations']]),
                            'profile_downgrades': profile_downgrades
                        }
                    }
                    
                    normalization_time = time.time() - stage_start_time
                    self.structured_logger.stage_complete("text_normalization", "Text normalization completed successfully",
                                                        duration=normalization_time,
                                                        metrics=normalization_metrics)
                    
                    if progress_callback:
                        progress_callback("G1", 94, f"Text normalization applied - {len(normalized_segments)} segments processed with {normalization_metrics['guardrail_violations']} violations")
                        
                except Exception as e:
                    self.structured_logger.error(f"Text normalization failed: {e}")
                    # Continue with original segments on error
                    normalization_time = time.time() - stage_start_time
                    self.structured_logger.stage_complete("text_normalization", "Text normalization failed, using original segments",
                                                        duration=normalization_time,
                                                        context={'error': str(e)})
            
            # Fallback to legacy post-fusion punctuation if text normalization is disabled
            elif self.enable_post_fusion_punctuation and self.punctuation_engine:
                if progress_callback:
                    progress_callback("G1", 92, "Applying legacy punctuation and disfluency normalization...")
                
                stage_start_time = time.time()
                self.structured_logger.stage_start("post_fusion_punctuation", "Applying legacy post-fusion punctuation and disfluency normalization")
                
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
                    self.structured_logger.stage_complete("post_fusion_punctuation", "Legacy post-fusion punctuation completed successfully",
                                                        duration=punctuation_time,
                                                        metrics={
                                                            'segments_processed': len(punctuated_segments),
                                                            'overall_confidence': punctuation_result.overall_confidence,
                                                            'punctuation_changes': punctuation_result.punctuation_metrics.get('segments_changed', 0),
                                                            'disfluency_normalizations': punctuation_result.disfluency_metrics.get('segments_normalized', 0)
                                                        })
                    
                    if progress_callback:
                        progress_callback("G1", 94, f"Legacy punctuation applied - {len(punctuated_segments)} segments processed")
                        
                except Exception as e:
                    self.structured_logger.error(f"Legacy post-fusion punctuation failed: {e}")
                    # Continue with original segments on error
                    punctuation_time = time.time() - stage_start_time
                    self.structured_logger.stage_complete("post_fusion_punctuation", "Legacy post-fusion punctuation failed, using original segments",
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
            output_paths = self._persist_output_files(results)
            
            # Track final outputs in manifest
            if self.manifest_manager:
                self._track_final_outputs_in_manifest(output_paths, results)
                
                # Mark processing as completed
                self.manifest_manager.mark_completed()
                
                # Validate manifest integrity
                validation_passed, validation_errors = self.manifest_manager.validate(recompute_hashes=True)
                if validation_passed:
                    self.structured_logger.info("✅ Manifest validation PASSED - All artifacts verified", 
                                              context={
                                                  'total_artifacts': self.manifest_manager._manifest.total_artifacts,
                                                  'total_bytes': self.manifest_manager._manifest.total_bytes,
                                                  'manifest_sha256': self.manifest_manager._manifest.manifest_sha256
                                              })
                else:
                    self.structured_logger.error("❌ Manifest validation FAILED", 
                                                context={
                                                    'validation_errors': validation_errors,
                                                    'error_count': len(validation_errors)
                                                })
            
            # Resource Scheduler Session Completion and Performance Reporting
            if scheduler_session_started and self.resource_scheduler:
                try:
                    # Complete resource scheduler session with comprehensive reporting
                    session_summary = self.resource_scheduler.stop_session()
                    
                    # Log comprehensive performance summary
                    self.structured_logger.info("🎯 Resource Scheduler Performance Summary",
                                              context={
                                                  'session_rtf': session_summary['session_rtf'],
                                                  'performance_grade': session_summary['performance_grade'], 
                                                  'downgrades_applied': session_summary['downgrades_applied'],
                                                  'budget_violations': session_summary['budget_violations'],
                                                  'recommendations': session_summary['recommendations'],
                                                  'complexity_score': session_summary.get('complexity_estimate', {}).get('complexity_score', 0)
                                              })
                    
                    # Add resource scheduler metrics to results for user visibility
                    results['resource_scheduler_summary'] = {
                        'session_rtf': session_summary['session_rtf'],
                        'performance_grade': session_summary['performance_grade'],
                        'quality_preserved': session_summary['downgrades_applied'] < 3,
                        'budget_compliant': session_summary['budget_violations'] == 0,
                        'stage_metrics': session_summary.get('stage_metrics', {}),
                        'optimization_recommendations': session_summary.get('recommendations', []),
                        'complexity_assessment': session_summary.get('complexity_estimate', {})
                    }
                    
                    # Update progress with final scheduler status
                    if progress_callback:
                        grade_emoji = {"A": "🥇", "B": "🥈", "C": "🥉", "D": "📊", "F": "⚠️"}.get(session_summary['performance_grade'], "📊")
                        progress_callback("FINAL", 100, f"{grade_emoji} Processing complete (Performance: {session_summary['performance_grade']}, RTF: {session_summary['session_rtf']:.2f}x)")
                    
                except Exception as e:
                    self.structured_logger.warning(f"Failed to complete resource scheduler session: {e}")
            
            return results
            
        except Exception as e:
            # End resource scheduler session on error
            if scheduler_session_started and self.resource_scheduler:
                try:
                    error_summary = self.resource_scheduler.stop_session()
                    self.structured_logger.error("Resource scheduler session ended due to processing error",
                                               context={'partial_results': error_summary})
                except:
                    pass  # Avoid masking the original error
            
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
    
    def _persist_output_files(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Persist transcript outputs to artifacts directory using atomic I/O operations"""
        try:
            import os
            from pathlib import Path
            
            # Create artifacts/reports directory
            reports_dir = Path("artifacts/reports")
            if self.run_id is None:
                self.structured_logger.warning("No run_id available for output persistence")
                return {}
            run_dir = reports_dir / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Write transcript files using atomic operations
            files_written = {}
            
            try:
                # Get cache key from run context for collision prevention
                run_context = get_global_run_context()
                cache_key = run_context.config_snapshot_id if run_context else None
                
                # JSON transcript - atomic write
                if 'winner_transcript' in results:
                    json_path = run_dir / "transcript.json"
                    with atomic_write(json_path, cache_key=cache_key) as f:
                        import json
                        json.dump(results['winner_transcript'], f, indent=2, ensure_ascii=False)
                    files_written['fused_transcript_json'] = str(json_path)
                    self.structured_logger.debug(f"Atomically wrote JSON transcript: {json_path}")
                
                # TXT transcript - atomic write  
                if 'winner_transcript_txt' in results:
                    txt_path = run_dir / "transcript.txt"
                    with atomic_write(txt_path, cache_key=cache_key) as f:
                        f.write(results['winner_transcript_txt'])
                    files_written['transcript_txt'] = str(txt_path)
                    self.structured_logger.debug(f"Atomically wrote TXT transcript: {txt_path}")
                
                # VTT captions - atomic write
                if 'captions_vtt' in results:
                    vtt_path = run_dir / "captions.vtt"
                    with atomic_write(vtt_path, cache_key=cache_key) as f:
                        f.write(results['captions_vtt'])
                    files_written['vtt'] = str(vtt_path)
                    self.structured_logger.debug(f"Atomically wrote VTT captions: {vtt_path}")
                
                # SRT captions - atomic write
                if 'captions_srt' in results:
                    srt_path = run_dir / "captions.srt"
                    with atomic_write(srt_path, cache_key=cache_key) as f:
                        f.write(results['captions_srt'])
                    files_written['srt'] = str(srt_path)
                    self.structured_logger.debug(f"Atomically wrote SRT captions: {srt_path}")
                
                # ASS captions - atomic write
                if 'captions_ass' in results:
                    ass_path = run_dir / "captions.ass"
                    with atomic_write(ass_path, cache_key=cache_key) as f:
                        f.write(results['captions_ass'])
                    files_written['captions_ass'] = str(ass_path)
                    self.structured_logger.debug(f"Atomically wrote ASS captions: {ass_path}")
                
                # Ensemble audit - atomic write
                if 'ensemble_audit' in results:
                    audit_path = run_dir / "ensemble_audit.json"
                    with atomic_write(audit_path, cache_key=cache_key) as f:
                        import json
                        json.dump(results['ensemble_audit'], f, indent=2, ensure_ascii=False)
                    files_written['ensemble_audit_json'] = str(audit_path)
                    self.structured_logger.debug(f"Atomically wrote ensemble audit: {audit_path}")
                
                self.structured_logger.info(f"Successfully persisted {len(files_written)} output files atomically")
                
            except Exception as atomic_error:
                # Fallback to legacy method if atomic I/O fails
                self.structured_logger.warning(f"Atomic file persistence failed, using fallback: {atomic_error}")
                
                # Legacy fallback operations
                if 'winner_transcript' in results and 'fused_transcript_json' not in files_written:
                    json_path = run_dir / "transcript.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(results['winner_transcript'], f, indent=2, ensure_ascii=False)
                    files_written['fused_transcript_json'] = str(json_path)
                
                if 'winner_transcript_txt' in results and 'transcript_txt' not in files_written:
                    txt_path = run_dir / "transcript.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(results['winner_transcript_txt'])
                    files_written['transcript_txt'] = str(txt_path)
                
                # Add other fallback writes as needed...
                self.structured_logger.warning("Used legacy file writing as fallback")
            
            return files_written
        
        except Exception as e:
            self.structured_logger.error(f"Failed to persist output files: {e}")
            return {}
    
    def _track_final_outputs_in_manifest(self, output_paths: Dict[str, str], results: Dict[str, Any]):
        """Track final output files in manifest system"""
        if not self.manifest_manager:
            return
        
        try:
            # Get diarization artifacts as inputs for final outputs
            diarization_artifacts = self.manifest_manager.get_artifacts_by_type("diarization_json")
            input_artifacts = [a.sha256 for a in diarization_artifacts] if diarization_artifacts else []
            
            # Track all final output files
            for artifact_type, file_path in output_paths.items():
                try:
                    producing_component = {
                        'fused_transcript_json': 'ConsensusModule.process_consensus',
                        'transcript_txt': 'TranscriptFormatter.create_txt_transcript', 
                        'vtt': 'TranscriptFormatter.create_vtt_captions',
                        'srt': 'TranscriptFormatter.create_srt_captions',
                        'captions_ass': 'TranscriptFormatter.create_ass_captions',
                        'ensemble_audit_json': 'EnsembleManager._create_ensemble_audit'
                    }.get(artifact_type, 'EnsembleManager._persist_output_files')
                    
                    metadata = {
                        "processing_stage": "final_output",
                        "output_type": artifact_type
                    }
                    
                    # Add specific metadata for different artifact types
                    if artifact_type == 'fused_transcript_json' and 'winner_transcript' in results:
                        winner = results['winner_transcript']
                        metadata["segment_count"] = str(len(winner.get('segments', [])))
                        metadata["speaker_count"] = str(len(winner.get('speaker_map', {})))
                        metadata["total_duration"] = str(float(winner.get('metadata', {}).get('total_duration', 0.0)))
                    elif artifact_type == 'ensemble_audit_json' and 'ensemble_audit' in results:
                        audit = results['ensemble_audit']
                        metadata.update({
                            "total_candidates": audit.get('summary', {}).get('total_candidates', 0),
                            "winner_score": audit.get('summary', {}).get('winner_score', 0.0)
                        })
                    
                    self.manifest_manager.add_artifact(
                        artifact_type=artifact_type,
                        file_path=file_path,
                        producing_component=producing_component,
                        input_artifacts=input_artifacts,
                        metadata=metadata
                    )
                    
                except Exception as e:
                    self.structured_logger.warning(f"Failed to track {artifact_type} in manifest: {e}")
                    
        except Exception as e:
            self.structured_logger.error(f"Failed to track final outputs in manifest: {e}")
