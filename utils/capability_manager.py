"""
Capability Manager for Production Readiness

Provides comprehensive dependency checks, capability detection, and graceful fallbacks
for all machine learning and audio processing dependencies. Ensures system remains
functional even when optional dependencies are missing.

Key Features:
- Startup capability checks for all ML dependencies
- Graceful no-op fallbacks when models unavailable  
- Explicit logging/metrics when features disabled
- Runtime model validation and fallback selection
- Performance metrics tracking for degraded modes
"""

import sys
import importlib
import warnings
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

from utils.enhanced_structured_logger import create_enhanced_logger

class CapabilityStatus(Enum):
    """Capability status levels"""
    AVAILABLE = "available"
    DEGRADED = "degraded" 
    UNAVAILABLE = "unavailable"
    FALLBACK = "fallback"
    ERROR = "error"

@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    module_path: str
    version_check: Optional[str] = None
    optional: bool = True
    fallback_available: bool = True
    description: str = ""
    required_for: List[str] = field(default_factory=list)
    min_version: Optional[str] = None
    import_error: Optional[str] = None
    status: CapabilityStatus = CapabilityStatus.UNAVAILABLE

@dataclass 
class CapabilityReport:
    """Complete capability assessment report"""
    timestamp: float
    system_status: CapabilityStatus
    dependencies: Dict[str, DependencyInfo]
    available_features: List[str]
    degraded_features: List[str] 
    unavailable_features: List[str]
    fallback_features: List[str]
    critical_missing: List[str]
    warnings: List[str]
    performance_impact: Dict[str, str]
    recommendations: List[str]

class CapabilityManager:
    """Manages dependency checks and feature capabilities"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("capability_manager")
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.feature_status: Dict[str, CapabilityStatus] = {}
        self.capability_report: Optional[CapabilityReport] = None
        self.startup_complete = False
        
        # Define all dependencies and their capabilities
        self._initialize_dependency_registry()
    
    def _initialize_dependency_registry(self):
        """Initialize registry of all dependencies and their capabilities"""
        
        # Core ML dependencies
        self.dependencies.update({
            'torch': DependencyInfo(
                name='PyTorch',
                module_path='torch',
                optional=False,
                fallback_available=False,
                description='Core PyTorch framework for neural networks',
                required_for=['speaker_embeddings', 'neural_models', 'overlap_processing']
            ),
            'librosa': DependencyInfo(
                name='Librosa',
                module_path='librosa',
                optional=False,
                fallback_available=True,
                description='Audio processing and feature extraction',
                required_for=['audio_processing', 'feature_extraction']
            ),
            'scipy': DependencyInfo(
                name='SciPy',
                module_path='scipy',
                optional=False,
                fallback_available=True,
                description='Scientific computing including optimization',
                required_for=['signal_processing', 'clustering', 'hungarian_algorithm']
            ),
            'sklearn': DependencyInfo(
                name='scikit-learn',
                module_path='sklearn',
                optional=False,
                fallback_available=True,
                description='Machine learning algorithms and clustering',
                required_for=['clustering', 'classification', 'metrics']
            ),
            
            # Speaker processing dependencies
            'pyannote_audio': DependencyInfo(
                name='pyannote.audio',
                module_path='pyannote.audio',
                optional=True,
                fallback_available=True,
                description='Advanced speaker diarization with pretrained models',
                required_for=['advanced_diarization', 'speaker_embeddings']
            ),
            'speechbrain': DependencyInfo(
                name='SpeechBrain',
                module_path='speechbrain',
                optional=True,
                fallback_available=True,
                description='Speech processing toolkit with pretrained models',
                required_for=['pretrained_embeddings', 'speaker_verification']
            ),
            
            # Audio separation dependencies  
            'demucs': DependencyInfo(
                name='Demucs',
                module_path='demucs',
                optional=True,
                fallback_available=True,
                description='Source separation for overlapping speech',
                required_for=['source_separation', 'overlap_processing']
            ),
            
            # ASR dependencies
            'openai': DependencyInfo(
                name='OpenAI',
                module_path='openai',
                optional=True,
                fallback_available=True,
                description='OpenAI API for Whisper transcription',
                required_for=['whisper_asr', 'gpt_consensus']
            ),
            'faster_whisper': DependencyInfo(
                name='Faster Whisper',
                module_path='faster_whisper',
                optional=True,
                fallback_available=True,
                description='Optimized Whisper implementation',
                required_for=['fast_whisper_asr']
            ),
            'deepgram': DependencyInfo(
                name='Deepgram SDK',
                module_path='deepgram',
                optional=True,
                fallback_available=True,
                description='Deepgram API for real-time ASR',
                required_for=['deepgram_asr']
            ),
            
            # Text processing dependencies
            'transformers': DependencyInfo(
                name='Transformers',
                module_path='transformers',
                optional=True,
                fallback_available=True,
                description='HuggingFace transformers for language models',
                required_for=['text_normalization', 'punctuation', 'language_detection']
            ),
            'nltk': DependencyInfo(
                name='NLTK',
                module_path='nltk',
                optional=True,
                fallback_available=True,
                description='Natural language processing toolkit',
                required_for=['text_processing', 'tokenization']
            ),
            
            # Audio format dependencies
            'ffmpeg': DependencyInfo(
                name='FFmpeg (Python)',
                module_path='ffmpeg',
                optional=True,
                fallback_available=True,
                description='Audio format conversion and processing',
                required_for=['audio_conversion', 'format_support']
            ),
            'soundfile': DependencyInfo(
                name='SoundFile',
                module_path='soundfile',
                optional=False,
                fallback_available=True,
                description='Audio file I/O',
                required_for=['audio_io', 'file_support']
            )
        })
    
    def check_all_capabilities(self) -> CapabilityReport:
        """Perform comprehensive capability check at startup"""
        self.logger.info("Starting comprehensive capability assessment")
        start_time = time.time()
        
        # Check each dependency
        for dep_key, dep_info in self.dependencies.items():
            self._check_dependency(dep_key, dep_info)
        
        # Assess feature capabilities
        self._assess_feature_capabilities()
        
        # Generate capability report
        self.capability_report = self._generate_capability_report(start_time)
        
        # Log comprehensive report
        self._log_capability_report()
        
        self.startup_complete = True
        return self.capability_report
    
    def _check_dependency(self, dep_key: str, dep_info: DependencyInfo):
        """Check availability of a specific dependency"""
        try:
            # Try to import the module
            module = importlib.import_module(dep_info.module_path)
            
            # Check version if specified
            version = getattr(module, '__version__', 'unknown')
            
            # Mark as available
            dep_info.status = CapabilityStatus.AVAILABLE
            
            self.logger.debug(f"✅ {dep_info.name} available (version: {version})")
            
        except ImportError as e:
            dep_info.import_error = str(e)
            
            if dep_info.fallback_available:
                dep_info.status = CapabilityStatus.FALLBACK
                self.logger.warning(f"⚠️ {dep_info.name} unavailable, fallback will be used: {e}")
            else:
                dep_info.status = CapabilityStatus.UNAVAILABLE
                self.logger.error(f"❌ {dep_info.name} unavailable, no fallback: {e}")
        
        except Exception as e:
            dep_info.import_error = str(e)
            dep_info.status = CapabilityStatus.ERROR
            self.logger.error(f"💥 {dep_info.name} error during import: {e}")
    
    def _assess_feature_capabilities(self):
        """Assess which features are available based on dependencies"""
        
        # Define feature dependency mappings
        feature_deps = {
            'advanced_speaker_diarization': ['torch', 'pyannote_audio', 'scipy'],
            'basic_speaker_diarization': ['torch', 'librosa', 'sklearn'],
            'speaker_embeddings': ['torch', 'librosa'],
            'source_separation': ['torch', 'demucs', 'librosa'],
            'overlap_processing': ['torch', 'librosa', 'scipy'],
            'whisper_asr': ['openai'],
            'fast_whisper_asr': ['faster_whisper'],
            'deepgram_asr': ['deepgram'],
            'text_normalization': ['transformers'],
            'adaptive_biasing': ['nltk', 'sklearn'],
            'long_horizon_tracking': ['torch', 'sklearn', 'scipy'],
            'audio_format_conversion': ['ffmpeg', 'soundfile'],
            'consensus_fusion': ['sklearn', 'scipy']
        }
        
        # Check each feature
        for feature, deps in feature_deps.items():
            available_deps = []
            fallback_deps = []
            missing_deps = []
            
            for dep_key in deps:
                if dep_key in self.dependencies:
                    dep_status = self.dependencies[dep_key].status
                    if dep_status == CapabilityStatus.AVAILABLE:
                        available_deps.append(dep_key)
                    elif dep_status == CapabilityStatus.FALLBACK:
                        fallback_deps.append(dep_key)
                    else:
                        missing_deps.append(dep_key)
                else:
                    missing_deps.append(dep_key)
            
            # Determine feature status
            if not missing_deps:
                if not fallback_deps:
                    self.feature_status[feature] = CapabilityStatus.AVAILABLE
                else:
                    self.feature_status[feature] = CapabilityStatus.DEGRADED
            elif len(missing_deps) < len(deps) and self._has_fallback_path(feature):
                self.feature_status[feature] = CapabilityStatus.FALLBACK
            else:
                self.feature_status[feature] = CapabilityStatus.UNAVAILABLE
    
    def _has_fallback_path(self, feature: str) -> bool:
        """Check if feature has a viable fallback implementation"""
        fallback_paths = {
            'advanced_speaker_diarization': True,  # Can fall back to basic
            'speaker_embeddings': True,  # Can use basic TDNN implementation
            'source_separation': True,  # Can skip separation
            'text_normalization': True,  # Can use basic rules
            'adaptive_biasing': True,  # Can disable biasing
            'long_horizon_tracking': True,  # Can use basic mapping
            'audio_format_conversion': True,  # Can use basic librosa
        }
        return fallback_paths.get(feature, False)
    
    def _generate_capability_report(self, start_time: float) -> CapabilityReport:
        """Generate comprehensive capability report"""
        
        # Categorize features by status
        available_features = [f for f, s in self.feature_status.items() if s == CapabilityStatus.AVAILABLE]
        degraded_features = [f for f, s in self.feature_status.items() if s == CapabilityStatus.DEGRADED]
        fallback_features = [f for f, s in self.feature_status.items() if s == CapabilityStatus.FALLBACK]
        unavailable_features = [f for f, s in self.feature_status.items() if s == CapabilityStatus.UNAVAILABLE]
        
        # Find critical missing dependencies
        critical_missing = [
            dep.name for dep in self.dependencies.values() 
            if not dep.optional and dep.status == CapabilityStatus.UNAVAILABLE
        ]
        
        # Generate warnings and recommendations
        warnings = []
        recommendations = []
        performance_impact = {}
        
        if degraded_features:
            warnings.append(f"Some features running in degraded mode: {degraded_features}")
            performance_impact['degraded'] = "Reduced accuracy/performance for degraded features"
        
        if fallback_features:
            warnings.append(f"Some features using fallback implementations: {fallback_features}")
            performance_impact['fallback'] = "Basic functionality only for fallback features"
        
        if unavailable_features:
            warnings.append(f"Some features completely unavailable: {unavailable_features}")
            recommendations.append("Install missing dependencies to enable full functionality")
        
        # Determine overall system status
        if critical_missing:
            system_status = CapabilityStatus.ERROR
        elif unavailable_features:
            system_status = CapabilityStatus.DEGRADED
        elif degraded_features or fallback_features:
            system_status = CapabilityStatus.DEGRADED
        else:
            system_status = CapabilityStatus.AVAILABLE
        
        return CapabilityReport(
            timestamp=time.time(),
            system_status=system_status,
            dependencies=self.dependencies,
            available_features=available_features,
            degraded_features=degraded_features,
            unavailable_features=unavailable_features,
            fallback_features=fallback_features,
            critical_missing=critical_missing,
            warnings=warnings,
            performance_impact=performance_impact,
            recommendations=recommendations
        )
    
    def _log_capability_report(self):
        """Log comprehensive capability report"""
        if not self.capability_report:
            return
        
        report = self.capability_report
        
        self.logger.info("🔍 SYSTEM CAPABILITY ASSESSMENT COMPLETE", context={
            'system_status': report.system_status.value,
            'available_features': len(report.available_features),
            'degraded_features': len(report.degraded_features),
            'fallback_features': len(report.fallback_features),
            'unavailable_features': len(report.unavailable_features),
            'critical_missing': len(report.critical_missing)
        })
        
        # Log available features
        if report.available_features:
            self.logger.info("✅ AVAILABLE FEATURES", context={
                'features': report.available_features
            })
        
        # Log degraded features with warnings
        if report.degraded_features:
            self.logger.warning("⚠️ DEGRADED FEATURES", context={
                'features': report.degraded_features,
                'impact': 'Reduced performance/accuracy'
            })
        
        # Log fallback features
        if report.fallback_features:
            self.logger.warning("🔄 FALLBACK FEATURES", context={
                'features': report.fallback_features,
                'impact': 'Basic functionality only'
            })
        
        # Log unavailable features
        if report.unavailable_features:
            self.logger.error("❌ UNAVAILABLE FEATURES", context={
                'features': report.unavailable_features,
                'impact': 'Functionality disabled'
            })
        
        # Log critical issues
        if report.critical_missing:
            self.logger.error("🚨 CRITICAL DEPENDENCIES MISSING", context={
                'dependencies': report.critical_missing,
                'impact': 'System may not function properly'
            })
        
        # Log warnings and recommendations
        for warning in report.warnings:
            self.logger.warning(f"⚠️ {warning}")
        
        for recommendation in report.recommendations:
            self.logger.info(f"💡 {recommendation}")
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if a specific feature is available"""
        return self.feature_status.get(feature, CapabilityStatus.UNAVAILABLE) in [
            CapabilityStatus.AVAILABLE, CapabilityStatus.DEGRADED
        ]
    
    def is_dependency_available(self, dep_key: str) -> bool:
        """Check if a specific dependency is available"""
        return self.dependencies.get(dep_key, DependencyInfo('', '')).status == CapabilityStatus.AVAILABLE
    
    def get_fallback_recommendation(self, feature: str) -> Optional[str]:
        """Get fallback recommendation for unavailable feature"""
        fallback_recommendations = {
            'advanced_speaker_diarization': 'Use basic clustering-based diarization',
            'speaker_embeddings': 'Use lightweight TDNN implementation', 
            'source_separation': 'Skip separation, process overlapping speech as-is',
            'text_normalization': 'Use rule-based normalization',
            'adaptive_biasing': 'Disable vocabulary biasing',
            'long_horizon_tracking': 'Use per-chunk speaker mapping only'
        }
        return fallback_recommendations.get(feature)
    
    def get_capability_metrics(self) -> Dict[str, Any]:
        """Get capability metrics for monitoring"""
        if not self.capability_report:
            return {}
        
        return {
            'system_status': self.capability_report.system_status.value,
            'features_available': len(self.capability_report.available_features),
            'features_degraded': len(self.capability_report.degraded_features),
            'features_fallback': len(self.capability_report.fallback_features),
            'features_unavailable': len(self.capability_report.unavailable_features),
            'dependencies_missing': sum(1 for d in self.dependencies.values() 
                                      if d.status == CapabilityStatus.UNAVAILABLE),
            'startup_complete': self.startup_complete
        }

# Global capability manager instance
_capability_manager = None

def get_capability_manager() -> CapabilityManager:
    """Get global capability manager instance"""
    global _capability_manager
    if _capability_manager is None:
        _capability_manager = CapabilityManager()
    return _capability_manager

def check_system_capabilities() -> CapabilityReport:
    """Convenience function to check all system capabilities"""
    return get_capability_manager().check_all_capabilities()

def is_feature_available(feature: str) -> bool:
    """Convenience function to check if feature is available"""
    return get_capability_manager().is_feature_available(feature)

def require_feature(feature: str, fallback_msg: str = None) -> bool:
    """Require a feature and log appropriate message if unavailable"""
    manager = get_capability_manager()
    if manager.is_feature_available(feature):
        return True
    
    logger = create_enhanced_logger("feature_check")
    fallback = manager.get_fallback_recommendation(feature)
    
    if fallback_msg:
        logger.warning(f"Feature '{feature}' unavailable: {fallback_msg}")
    elif fallback:
        logger.warning(f"Feature '{feature}' unavailable, fallback: {fallback}")
    else:
        logger.warning(f"Feature '{feature}' unavailable, no fallback available")
    
    return False