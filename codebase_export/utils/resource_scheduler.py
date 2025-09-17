"""
Resource-Aware Scheduler with RTF Budgets and Auto-Downgrade Rules

Provides intelligent resource management for the ensemble transcription pipeline,
ensuring predictable processing times while maximizing quality through dynamic
stage downgrading and RTF budget enforcement.

Key Features:
- RTF (Real-Time Factor) budget management per stage
- Auto-downgrade rules when budgets exceeded
- Real-time resource monitoring (CPU, memory, processing time)
- Predictive audio complexity estimation
- Adaptive scheduling based on content characteristics  
- Integration with existing observability system
- Quality vs speed trade-offs with user controls

RTF Budget Definitions:
- RTF = processing_time / audio_duration
- Separation: 8.0x (heavy) → 2.0x (light) → skip
- Diarization: 4.0x (multi) → 1.5x (single) → 0.5x (basic)
- ASR: 6.0x (ensemble) → 3.0x (reduced) → 1.0x (single)
- Normalization: 0.2x (all profiles)
"""

import os
import time
import psutil
import threading
import json
from typing import Dict, Any, List, Optional, Tuple, Callable, Union, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import hashlib

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import get_observability_manager, trace_stage, track_cost
from utils.intelligent_cache import get_cache_manager
from utils.capability_manager import get_capability_manager, CapabilityStatus


class ProcessingStage(Enum):
    """Processing stages with defined order and dependencies"""
    AUDIO_EXTRACTION = "audio_extraction"
    AUDIO_PREPROCESSING = "audio_preprocessing" 
    DIARIZATION = "diarization"
    OVERLAP_PROCESSING = "overlap_processing"
    SOURCE_SEPARATION = "source_separation"
    SPEAKER_TRACKING = "speaker_tracking"
    ASR_ENSEMBLE = "asr_ensemble"
    TERM_MINING = "term_mining"
    DIALECT_PROCESSING = "dialect_processing"
    CONFIDENCE_SCORING = "confidence_scoring"
    CONSENSUS = "consensus"
    WORKLIST_REPROCESSING = "worklist_reprocessing"
    BOUNDARY_REALIGNMENT = "boundary_realignment"
    TEXT_NORMALIZATION = "text_normalization"
    OUTPUT_GENERATION = "output_generation"


class QualityLevel(Enum):
    """Quality levels for auto-downgrade system"""
    MAXIMUM = "maximum"      # Highest quality, longest processing
    HIGH = "high"           # High quality, moderate processing
    BALANCED = "balanced"   # Balance quality vs speed
    FAST = "fast"          # Fast processing, good quality
    MINIMAL = "minimal"     # Minimal processing, basic quality


class DowngradeStrategy(Enum):
    """Downgrade aggressiveness strategies"""
    CONSERVATIVE = "conservative"  # Prefer quality, minimal downgrades
    BALANCED = "balanced"         # Balance quality vs time constraints
    AGGRESSIVE = "aggressive"     # Prefer speed, aggressive downgrades


@dataclass
class RTFBudget:
    """RTF budget configuration for a processing stage"""
    stage: ProcessingStage
    maximum_rtf: float         # Maximum RTF before mandatory downgrade
    target_rtf: float          # Target RTF for optimal quality
    minimum_rtf: float         # Minimum RTF for basic functionality
    timeout_minutes: float     # Hard timeout in minutes
    quality_levels: Dict[QualityLevel, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize quality level RTF mappings if not provided"""
        if not self.quality_levels:
            self.quality_levels = {
                QualityLevel.MAXIMUM: self.maximum_rtf,
                QualityLevel.HIGH: self.target_rtf,
                QualityLevel.BALANCED: (self.target_rtf + self.minimum_rtf) / 2,
                QualityLevel.FAST: self.minimum_rtf,
                QualityLevel.MINIMAL: self.minimum_rtf * 0.5
            }


@dataclass 
class StageResourceUsage:
    """Real-time resource usage tracking for a stage"""
    stage: ProcessingStage
    start_time: float = 0.0
    end_time: float = 0.0
    processing_duration: float = 0.0
    audio_duration: float = 0.0
    actual_rtf: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    quality_level: QualityLevel = QualityLevel.HIGH
    budget_exceeded: bool = False
    downgrade_applied: bool = False
    downgrade_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_rtf(self):
        """Calculate RTF from processing time and audio duration"""
        if self.audio_duration > 0 and self.processing_duration > 0:
            self.actual_rtf = self.processing_duration / self.audio_duration
        return self.actual_rtf


@dataclass
class ResourceMonitoringSnapshot:
    """System resource snapshot at a point in time"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    load_average: Tuple[float, float, float]  # 1min, 5min, 15min
    active_threads: int
    
    @classmethod
    def capture_current(cls):
        """Capture current system resource snapshot"""
        try:
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            memory = psutil.virtual_memory()
            
            return cls(
                timestamp=time.time(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=memory.percent,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=psutil.disk_usage('/').percent,
                load_average=load_avg,
                active_threads=threading.active_count()
            )
        except Exception as e:
            logging.warning(f"Failed to capture resource snapshot: {e}")
            return cls(
                timestamp=time.time(),
                cpu_percent=0, memory_percent=0, memory_available_mb=0,
                disk_usage_percent=0, load_average=(0, 0, 0), active_threads=0
            )


@dataclass
class DowngradeRule:
    """Auto-downgrade rule for a specific stage"""
    stage: ProcessingStage
    trigger_rtf_threshold: float
    trigger_memory_mb_threshold: Optional[float] = None
    trigger_cpu_percent_threshold: Optional[float] = None
    downgrade_action: str = ""
    fallback_quality_level: QualityLevel = QualityLevel.FAST
    skip_stage: bool = False
    condition: Optional[str] = None  # Additional conditions (e.g., "audio_duration > 300")


@dataclass
class AudioComplexityEstimate:
    """Audio complexity estimation for budget allocation"""
    audio_duration: float
    estimated_speakers: int
    noise_level: str  # low, medium, high
    overlap_probability: float
    dialect_complexity: float
    complexity_score: float = 0.0
    recommended_quality_level: QualityLevel = QualityLevel.BALANCED
    
    def __post_init__(self):
        """Calculate overall complexity score"""
        # Base complexity from duration (longer = more complex)
        duration_factor = min(self.audio_duration / 1800, 2.0)  # Cap at 30min = 2.0x
        
        # Speaker complexity (more speakers = more complex)
        speaker_factor = 1.0 + (self.estimated_speakers - 1) * 0.2
        
        # Noise level complexity
        noise_factors = {"low": 1.0, "medium": 1.3, "high": 1.8}
        noise_factor = noise_factors.get(self.noise_level, 1.3)
        
        # Overlap and dialect complexity
        overlap_factor = 1.0 + self.overlap_probability * 0.5
        dialect_factor = 1.0 + self.dialect_complexity * 0.3
        
        self.complexity_score = (duration_factor * speaker_factor * 
                               noise_factor * overlap_factor * dialect_factor)
        
        # Recommend quality level based on complexity
        if self.complexity_score < 1.5:
            self.recommended_quality_level = QualityLevel.HIGH
        elif self.complexity_score < 2.0:
            self.recommended_quality_level = QualityLevel.BALANCED  
        elif self.complexity_score < 3.0:
            self.recommended_quality_level = QualityLevel.FAST
        else:
            self.recommended_quality_level = QualityLevel.MINIMAL


class ResourceScheduler:
    """
    Resource-aware scheduler with RTF budgets and auto-downgrade rules
    
    Manages processing time budgets, monitors system resources, and applies
    intelligent downgrade strategies to ensure predictable processing times
    while maximizing transcription quality.
    """
    
    def __init__(self, 
                 global_timeout_minutes: float = 30.0,
                 downgrade_strategy: DowngradeStrategy = DowngradeStrategy.BALANCED,
                 enable_predictive_scheduling: bool = True,
                 enable_resource_monitoring: bool = True,
                 memory_limit_mb: Optional[float] = None,
                 cpu_limit_percent: Optional[float] = None):
        """
        Initialize resource scheduler
        
        Args:
            global_timeout_minutes: Maximum total processing time
            downgrade_strategy: Strategy for applying downgrades
            enable_predictive_scheduling: Enable audio complexity prediction
            enable_resource_monitoring: Enable system resource monitoring
            memory_limit_mb: Memory limit for triggering downgrades
            cpu_limit_percent: CPU limit for triggering downgrades
        """
        self.global_timeout_minutes = global_timeout_minutes
        self.downgrade_strategy = downgrade_strategy
        self.enable_predictive_scheduling = enable_predictive_scheduling
        self.enable_resource_monitoring = enable_resource_monitoring
        self.memory_limit_mb = memory_limit_mb or 8192  # 8GB default
        self.cpu_limit_percent = cpu_limit_percent or 90  # 90% CPU default
        
        # Initialize core systems
        self.logger = create_enhanced_logger("resource_scheduler")
        self.obs_manager = get_observability_manager()
        self.cache_manager = get_cache_manager()
        self.capability_manager = get_capability_manager()
        
        # Initialize RTF budgets for each stage
        self._initialize_rtf_budgets()
        
        # Initialize downgrade rules
        self._initialize_downgrade_rules()
        
        # Runtime state
        self.session_start_time: float = 0.0
        self.audio_duration: float = 0.0
        self.complexity_estimate: Optional[AudioComplexityEstimate] = None
        self.stage_usage: Dict[ProcessingStage, StageResourceUsage] = {}
        self.resource_snapshots: List[ResourceMonitoringSnapshot] = []
        self.downgrades_applied: List[Dict[str, Any]] = []
        
        # Thread-safe resource monitoring
        self._monitoring_lock = threading.Lock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        self.logger.info("Resource scheduler initialized", 
                        context={
                            'global_timeout': global_timeout_minutes,
                            'strategy': downgrade_strategy.value,
                            'predictive_enabled': enable_predictive_scheduling,
                            'monitoring_enabled': enable_resource_monitoring
                        })
    
    def _initialize_rtf_budgets(self):
        """Initialize RTF budgets for all processing stages"""
        self.rtf_budgets = {
            ProcessingStage.AUDIO_EXTRACTION: RTFBudget(
                ProcessingStage.AUDIO_EXTRACTION, 
                maximum_rtf=0.5, target_rtf=0.2, minimum_rtf=0.1, timeout_minutes=2.0
            ),
            ProcessingStage.AUDIO_PREPROCESSING: RTFBudget(
                ProcessingStage.AUDIO_PREPROCESSING,
                maximum_rtf=1.0, target_rtf=0.3, minimum_rtf=0.1, timeout_minutes=2.0
            ),
            ProcessingStage.DIARIZATION: RTFBudget(
                ProcessingStage.DIARIZATION,
                maximum_rtf=4.0, target_rtf=2.0, minimum_rtf=1.5, timeout_minutes=8.0
            ),
            ProcessingStage.SOURCE_SEPARATION: RTFBudget(
                ProcessingStage.SOURCE_SEPARATION,
                maximum_rtf=8.0, target_rtf=4.0, minimum_rtf=2.0, timeout_minutes=10.0
            ),
            ProcessingStage.OVERLAP_PROCESSING: RTFBudget(
                ProcessingStage.OVERLAP_PROCESSING,
                maximum_rtf=6.0, target_rtf=3.0, minimum_rtf=1.0, timeout_minutes=8.0
            ),
            ProcessingStage.SPEAKER_TRACKING: RTFBudget(
                ProcessingStage.SPEAKER_TRACKING,
                maximum_rtf=2.0, target_rtf=1.0, minimum_rtf=0.5, timeout_minutes=3.0
            ),
            ProcessingStage.ASR_ENSEMBLE: RTFBudget(
                ProcessingStage.ASR_ENSEMBLE,
                maximum_rtf=6.0, target_rtf=4.0, minimum_rtf=1.0, timeout_minutes=12.0
            ),
            ProcessingStage.TERM_MINING: RTFBudget(
                ProcessingStage.TERM_MINING,
                maximum_rtf=0.8, target_rtf=0.4, minimum_rtf=0.1, timeout_minutes=2.0
            ),
            ProcessingStage.DIALECT_PROCESSING: RTFBudget(
                ProcessingStage.DIALECT_PROCESSING,
                maximum_rtf=1.0, target_rtf=0.5, minimum_rtf=0.2, timeout_minutes=2.0
            ),
            ProcessingStage.CONFIDENCE_SCORING: RTFBudget(
                ProcessingStage.CONFIDENCE_SCORING,
                maximum_rtf=0.5, target_rtf=0.3, minimum_rtf=0.1, timeout_minutes=2.0
            ),
            ProcessingStage.CONSENSUS: RTFBudget(
                ProcessingStage.CONSENSUS,
                maximum_rtf=0.3, target_rtf=0.1, minimum_rtf=0.05, timeout_minutes=1.0
            ),
            ProcessingStage.WORKLIST_REPROCESSING: RTFBudget(
                ProcessingStage.WORKLIST_REPROCESSING,
                maximum_rtf=2.0, target_rtf=1.0, minimum_rtf=0.3, timeout_minutes=3.0
            ),
            ProcessingStage.BOUNDARY_REALIGNMENT: RTFBudget(
                ProcessingStage.BOUNDARY_REALIGNMENT,
                maximum_rtf=0.4, target_rtf=0.2, minimum_rtf=0.1, timeout_minutes=1.0
            ),
            ProcessingStage.TEXT_NORMALIZATION: RTFBudget(
                ProcessingStage.TEXT_NORMALIZATION,
                maximum_rtf=0.2, target_rtf=0.1, minimum_rtf=0.05, timeout_minutes=1.0
            ),
            ProcessingStage.OUTPUT_GENERATION: RTFBudget(
                ProcessingStage.OUTPUT_GENERATION,
                maximum_rtf=0.1, target_rtf=0.05, minimum_rtf=0.02, timeout_minutes=1.0
            )
        }
    
    def _initialize_downgrade_rules(self):
        """Initialize auto-downgrade rules for each stage"""
        self.downgrade_rules = {
            # Source Separation: Heavy → Light → Skip
            ProcessingStage.SOURCE_SEPARATION: [
                DowngradeRule(
                    ProcessingStage.SOURCE_SEPARATION,
                    trigger_rtf_threshold=6.0,
                    downgrade_action="switch_to_light_model",
                    fallback_quality_level=QualityLevel.FAST
                ),
                DowngradeRule(
                    ProcessingStage.SOURCE_SEPARATION,
                    trigger_rtf_threshold=4.0,
                    downgrade_action="skip_separation",
                    fallback_quality_level=QualityLevel.MINIMAL,
                    skip_stage=True,
                    condition="complexity_score > 2.0"
                )
            ],
            
            # Diarization: Multi-variant → Single variant → Basic clustering
            ProcessingStage.DIARIZATION: [
                DowngradeRule(
                    ProcessingStage.DIARIZATION,
                    trigger_rtf_threshold=3.0,
                    downgrade_action="reduce_variants",
                    fallback_quality_level=QualityLevel.FAST
                ),
                DowngradeRule(
                    ProcessingStage.DIARIZATION,
                    trigger_rtf_threshold=2.0,
                    downgrade_action="basic_clustering_only",
                    fallback_quality_level=QualityLevel.MINIMAL
                )
            ],
            
            # ASR: Ensemble → Reduced variants → Single engine
            ProcessingStage.ASR_ENSEMBLE: [
                DowngradeRule(
                    ProcessingStage.ASR_ENSEMBLE,
                    trigger_rtf_threshold=4.0,
                    downgrade_action="reduce_asr_variants",
                    fallback_quality_level=QualityLevel.BALANCED
                ),
                DowngradeRule(
                    ProcessingStage.ASR_ENSEMBLE,
                    trigger_rtf_threshold=2.0,
                    downgrade_action="single_asr_engine",
                    fallback_quality_level=QualityLevel.FAST
                )
            ],
            
            # Text Normalization: Executive → Readable → Light → Verbatim
            ProcessingStage.TEXT_NORMALIZATION: [
                DowngradeRule(
                    ProcessingStage.TEXT_NORMALIZATION,
                    trigger_rtf_threshold=0.15,
                    downgrade_action="lighter_normalization",
                    fallback_quality_level=QualityLevel.FAST
                ),
                DowngradeRule(
                    ProcessingStage.TEXT_NORMALIZATION,
                    trigger_rtf_threshold=0.1,
                    downgrade_action="minimal_normalization",
                    fallback_quality_level=QualityLevel.MINIMAL
                )
            ]
        }
    
    def start_session(self, audio_duration: float, 
                     audio_path: Optional[str] = None,
                     expected_speakers: int = 2,
                     noise_level: str = "medium",
                     target_language: Optional[str] = None) -> AudioComplexityEstimate:
        """
        Start a new processing session with audio complexity estimation
        
        Args:
            audio_duration: Duration of audio in seconds
            audio_path: Path to audio file for analysis
            expected_speakers: Estimated number of speakers
            noise_level: Noise level assessment
            target_language: Target language for processing
            
        Returns:
            Audio complexity estimate with recommended quality level
        """
        self.session_start_time = time.time()
        self.audio_duration = audio_duration
        self.stage_usage.clear()
        self.resource_snapshots.clear()
        self.downgrades_applied.clear()
        
        # Start resource monitoring if enabled
        if self.enable_resource_monitoring:
            self._start_resource_monitoring()
        
        # Generate audio complexity estimate
        if self.enable_predictive_scheduling:
            self.complexity_estimate = self._estimate_audio_complexity(
                audio_duration, expected_speakers, noise_level, audio_path
            )
        else:
            self.complexity_estimate = AudioComplexityEstimate(
                audio_duration=audio_duration,
                estimated_speakers=expected_speakers,
                noise_level=noise_level,
                overlap_probability=0.3,  # Default estimate
                dialect_complexity=0.2    # Default estimate
            )
        
        self.logger.info("Processing session started",
                        context={
                            'audio_duration': audio_duration,
                            'complexity_score': self.complexity_estimate.complexity_score,
                            'recommended_quality': self.complexity_estimate.recommended_quality_level.value,
                            'global_timeout_minutes': self.global_timeout_minutes
                        })
        
        return self.complexity_estimate
    
    def _estimate_audio_complexity(self, audio_duration: float, 
                                 expected_speakers: int, 
                                 noise_level: str,
                                 audio_path: Optional[str] = None) -> AudioComplexityEstimate:
        """
        Estimate audio complexity for predictive scheduling
        
        Args:
            audio_duration: Duration of audio in seconds
            expected_speakers: Number of expected speakers
            noise_level: Assessed noise level
            audio_path: Path to audio file for detailed analysis
            
        Returns:
            Audio complexity estimate
        """
        # Basic complexity estimation
        overlap_probability = min(0.1 + (expected_speakers - 1) * 0.15, 0.8)
        dialect_complexity = 0.2  # Default estimate
        
        # Enhanced analysis if audio file is available
        if audio_path and os.path.exists(audio_path):
            try:
                # Check cache for complexity estimate
                cache_key = f"complexity_{hashlib.md5(audio_path.encode()).hexdigest()}"
                cached_complexity = self.cache_manager.get("audio_complexity", cache_key)
                if cached_complexity:
                    return AudioComplexityEstimate(**cached_complexity)
                
                # Perform lightweight audio analysis
                enhanced_estimate = self._analyze_audio_complexity(audio_path)
                if enhanced_estimate:
                    overlap_probability = enhanced_estimate.get('overlap_probability', overlap_probability)
                    dialect_complexity = enhanced_estimate.get('dialect_complexity', dialect_complexity)
                
                # Cache the result
                complexity_data = {
                    'audio_duration': audio_duration,
                    'estimated_speakers': expected_speakers,
                    'noise_level': noise_level,
                    'overlap_probability': overlap_probability,
                    'dialect_complexity': dialect_complexity
                }
                self.cache_manager.set("audio_complexity", cache_key, complexity_data, ttl=3600)
                
            except Exception as e:
                self.logger.warning(f"Enhanced complexity analysis failed: {e}")
        
        return AudioComplexityEstimate(
            audio_duration=audio_duration,
            estimated_speakers=expected_speakers, 
            noise_level=noise_level,
            overlap_probability=overlap_probability,
            dialect_complexity=dialect_complexity
        )
    
    def _analyze_audio_complexity(self, audio_path: str) -> Optional[Dict[str, float]]:
        """
        Perform lightweight audio analysis for complexity estimation
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with complexity metrics or None if analysis fails
        """
        try:
            import librosa
            import numpy as np
            
            # Load audio for analysis (first 30 seconds to keep it fast)
            y, sr = librosa.load(audio_path, sr=16000, duration=30.0)
            
            # Analyze energy variance (proxy for overlap probability)
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            energy = np.array([
                np.sum(y[i:i+frame_length]**2)
                for i in range(0, len(y)-frame_length, hop_length)
            ])
            
            # High energy variance suggests overlapping speech
            energy_variance = np.var(energy) / np.mean(energy) if np.mean(energy) > 0 else 0
            overlap_probability = min(energy_variance * 0.1, 0.8)
            
            # Analyze spectral features for dialect complexity estimation
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_variance = np.mean(np.var(mfccs, axis=1))
            dialect_complexity = min(spectral_variance * 0.01, 0.6)
            
            return {
                'overlap_probability': float(overlap_probability),
                'dialect_complexity': float(dialect_complexity)
            }
            
        except Exception as e:
            self.logger.warning(f"Audio complexity analysis failed: {e}")
            return None
    
    def start_stage(self, stage: ProcessingStage, 
                   metadata: Optional[Dict[str, Any]] = None) -> StageResourceUsage:
        """
        Start monitoring a processing stage
        
        Args:
            stage: Processing stage being started
            metadata: Additional metadata for the stage
            
        Returns:
            Stage resource usage tracker
        """
        current_time = time.time()
        
        # Check if we're approaching global timeout
        elapsed_minutes = (current_time - self.session_start_time) / 60.0
        remaining_minutes = self.global_timeout_minutes - elapsed_minutes
        
        if remaining_minutes <= 0:
            raise TimeoutError(f"Global timeout exceeded ({self.global_timeout_minutes} minutes)")
        
        # Initialize stage usage tracking
        usage = StageResourceUsage(
            stage=stage,
            start_time=current_time,
            audio_duration=self.audio_duration,
            metadata=metadata or {}
        )
        
        # Determine quality level based on complexity and remaining time
        if self.complexity_estimate:
            usage.quality_level = self._determine_quality_level(stage, remaining_minutes)
        
        # Apply preemptive downgrades if needed
        usage = self._apply_preemptive_downgrades(usage, remaining_minutes)
        
        self.stage_usage[stage] = usage
        
        self.logger.info(f"Started stage: {stage.value}",
                        context={
                            'quality_level': usage.quality_level.value,
                            'remaining_minutes': remaining_minutes,
                            'budget_rtf': self.rtf_budgets[stage].target_rtf,
                            'preemptive_downgrade': usage.downgrade_applied
                        })
        
        return usage
    
    def _determine_quality_level(self, stage: ProcessingStage, 
                               remaining_minutes: float) -> QualityLevel:
        """
        Determine appropriate quality level for stage based on constraints
        
        Args:
            stage: Processing stage
            remaining_minutes: Time remaining in session
            
        Returns:
            Recommended quality level
        """
        if not self.complexity_estimate:
            return QualityLevel.BALANCED
        
        # Base quality from complexity estimate
        base_quality = self.complexity_estimate.recommended_quality_level
        
        # Adjust based on remaining time and downgrade strategy
        if remaining_minutes < 5.0:  # Less than 5 minutes remaining
            if self.downgrade_strategy == DowngradeStrategy.AGGRESSIVE:
                return QualityLevel.MINIMAL
            elif self.downgrade_strategy == DowngradeStrategy.BALANCED:
                return QualityLevel.FAST
            else:  # CONSERVATIVE
                return max(base_quality, QualityLevel.FAST)
        
        elif remaining_minutes < 10.0:  # Less than 10 minutes remaining
            if self.downgrade_strategy == DowngradeStrategy.AGGRESSIVE:
                return QualityLevel.FAST
            else:
                return base_quality
        
        else:
            return base_quality
    
    def _apply_preemptive_downgrades(self, usage: StageResourceUsage,
                                   remaining_minutes: float) -> StageResourceUsage:
        """
        Apply preemptive downgrades based on predicted resource constraints
        
        Args:
            usage: Stage usage tracker
            remaining_minutes: Time remaining in session
            
        Returns:
            Modified usage tracker with any downgrades applied
        """
        # Check for preemptive downgrades based on system resources
        if self.enable_resource_monitoring:
            current_resources = ResourceMonitoringSnapshot.capture_current()
            
            # Memory pressure check
            if (current_resources.memory_percent > 85 or 
                current_resources.memory_available_mb < self.memory_limit_mb * 0.2):
                
                usage.downgrade_applied = True
                usage.downgrade_reason = "memory_pressure"
                usage.quality_level = QualityLevel.FAST
                
                self.logger.warning("Applied preemptive downgrade due to memory pressure",
                                  context={
                                      'memory_percent': current_resources.memory_percent,
                                      'available_mb': current_resources.memory_available_mb
                                  })
            
            # CPU pressure check  
            elif current_resources.cpu_percent > self.cpu_limit_percent:
                usage.downgrade_applied = True
                usage.downgrade_reason = "cpu_pressure"  
                usage.quality_level = QualityLevel.FAST
                
                self.logger.warning("Applied preemptive downgrade due to CPU pressure",
                                  context={'cpu_percent': current_resources.cpu_percent})
        
        # Time pressure check
        stage_timeout = self.rtf_budgets[usage.stage].timeout_minutes
        if remaining_minutes < stage_timeout * 1.5:
            usage.downgrade_applied = True
            usage.downgrade_reason = "time_pressure"
            usage.quality_level = QualityLevel.FAST
            
            self.logger.warning("Applied preemptive downgrade due to time pressure",
                              context={
                                  'remaining_minutes': remaining_minutes,
                                  'stage_timeout': stage_timeout
                              })
        
        return usage
    
    def end_stage(self, stage: ProcessingStage, 
                 success: bool = True,
                 error_message: Optional[str] = None) -> StageResourceUsage:
        """
        End monitoring of a processing stage and calculate final metrics
        
        Args:
            stage: Processing stage being completed
            success: Whether stage completed successfully
            error_message: Error message if stage failed
            
        Returns:
            Final stage resource usage with calculated metrics
        """
        if stage not in self.stage_usage:
            raise ValueError(f"Stage {stage.value} was not started")
        
        usage = self.stage_usage[stage]
        usage.end_time = time.time()
        usage.processing_duration = usage.end_time - usage.start_time
        usage.calculate_rtf()
        
        # Capture final resource snapshot
        if self.enable_resource_monitoring:
            final_snapshot = ResourceMonitoringSnapshot.capture_current()
            usage.cpu_percent = final_snapshot.cpu_percent
            usage.memory_mb = final_snapshot.memory_available_mb
        
        # Check if budget was exceeded
        budget = self.rtf_budgets[stage]
        target_rtf = budget.quality_levels.get(usage.quality_level, budget.target_rtf)
        
        if usage.actual_rtf > target_rtf * 1.2:  # 20% tolerance
            usage.budget_exceeded = True
            
            self.logger.warning(f"Stage {stage.value} exceeded RTF budget",
                              context={
                                  'actual_rtf': usage.actual_rtf,
                                  'target_rtf': target_rtf,
                                  'processing_duration': usage.processing_duration,
                                  'audio_duration': usage.audio_duration
                              })
        
        # Log stage completion
        self.logger.info(f"Completed stage: {stage.value}",
                        context={
                            'success': success,
                            'rtf': usage.actual_rtf,
                            'duration_seconds': usage.processing_duration,
                            'quality_level': usage.quality_level.value,
                            'budget_exceeded': usage.budget_exceeded,
                            'downgrade_applied': usage.downgrade_applied
                        })
        
        if error_message:
            usage.metadata['error_message'] = error_message
            
        return usage
    
    def check_budget_and_downgrade(self, stage: ProcessingStage,
                                 current_progress: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Check if stage is approaching budget limits and recommend downgrades
        
        Args:
            stage: Current processing stage
            current_progress: Progress through current stage (0.0 to 1.0)
            
        Returns:
            Downgrade recommendation or None if no action needed
        """
        if stage not in self.stage_usage:
            return None
        
        usage = self.stage_usage[stage]
        current_time = time.time()
        elapsed_time = current_time - usage.start_time
        
        # Estimate total time based on current progress
        if current_progress > 0.1:  # Only estimate after 10% progress
            estimated_total_time = elapsed_time / current_progress
            estimated_rtf = estimated_total_time / self.audio_duration
            
            # Check applicable downgrade rules
            applicable_rules = self.downgrade_rules.get(stage, [])
            for rule in applicable_rules:
                if estimated_rtf > rule.trigger_rtf_threshold:
                    # Check additional conditions
                    if rule.condition and not self._evaluate_condition(rule.condition):
                        continue
                    
                    downgrade_recommendation = {
                        'stage': stage,
                        'rule': rule,
                        'estimated_rtf': estimated_rtf,
                        'trigger_threshold': rule.trigger_rtf_threshold,
                        'recommended_action': rule.downgrade_action,
                        'new_quality_level': rule.fallback_quality_level,
                        'skip_stage': rule.skip_stage,
                        'reason': f"Estimated RTF {estimated_rtf:.2f} exceeds threshold {rule.trigger_rtf_threshold:.2f}"
                    }
                    
                    self.logger.warning(f"Budget exceeded for {stage.value}, recommending downgrade",
                                      context=downgrade_recommendation)
                    
                    return downgrade_recommendation
        
        return None
    
    def _evaluate_condition(self, condition: str) -> bool:
        """
        Evaluate additional downgrade condition
        
        Args:
            condition: Condition string to evaluate
            
        Returns:
            True if condition is met
        """
        try:
            # Simple condition evaluation - extend as needed
            if "complexity_score >" in condition:
                threshold = float(condition.split(">")[1].strip())
                if self.complexity_estimate:
                    return self.complexity_estimate.complexity_score > threshold
            
            elif "audio_duration >" in condition:
                threshold = float(condition.split(">")[1].strip())
                return self.audio_duration > threshold
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def apply_downgrade(self, downgrade_recommendation: Dict[str, Any]) -> bool:
        """
        Apply a recommended downgrade to the current stage
        
        Args:
            downgrade_recommendation: Downgrade recommendation from check_budget_and_downgrade
            
        Returns:
            True if downgrade was applied successfully
        """
        try:
            stage = downgrade_recommendation['stage']
            rule = downgrade_recommendation['rule']
            
            if stage in self.stage_usage:
                usage = self.stage_usage[stage]
                usage.downgrade_applied = True
                usage.downgrade_reason = downgrade_recommendation['reason']
                usage.quality_level = rule.fallback_quality_level
                
                # Record downgrade for telemetry
                self.downgrades_applied.append({
                    'timestamp': time.time(),
                    'stage': stage.value,
                    'action': rule.downgrade_action,
                    'reason': downgrade_recommendation['reason'],
                    'old_quality_level': usage.quality_level.value,
                    'new_quality_level': rule.fallback_quality_level.value
                })
                
                self.logger.info(f"Applied downgrade to {stage.value}",
                               context={
                                   'action': rule.downgrade_action,
                                   'new_quality_level': rule.fallback_quality_level.value,
                                   'skip_stage': rule.skip_stage
                               })
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to apply downgrade: {e}")
            return False
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring thread"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return  # Already running
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._resource_monitoring_loop,
            name="ResourceMonitoring"
        )
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
    
    def _resource_monitoring_loop(self):
        """Background resource monitoring loop"""
        while not self._stop_monitoring.wait(1.0):  # Check every second
            try:
                snapshot = ResourceMonitoringSnapshot.capture_current()
                
                with self._monitoring_lock:
                    self.resource_snapshots.append(snapshot)
                    
                    # Keep only last 300 snapshots (5 minutes at 1Hz)
                    if len(self.resource_snapshots) > 300:
                        self.resource_snapshots = self.resource_snapshots[-300:]
                
            except Exception as e:
                self.logger.warning(f"Resource monitoring error: {e}")
    
    def stop_session(self) -> Dict[str, Any]:
        """
        Stop the current processing session and generate final report
        
        Returns:
            Session summary with performance metrics and recommendations
        """
        session_end_time = time.time()
        total_session_time = session_end_time - self.session_start_time
        
        # Stop resource monitoring
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
        
        # Calculate session metrics
        total_processing_time = sum(
            usage.processing_duration for usage in self.stage_usage.values()
            if usage.processing_duration > 0
        )
        
        session_rtf = total_processing_time / self.audio_duration if self.audio_duration > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        session_summary = {
            'session_duration_minutes': total_session_time / 60.0,
            'total_processing_time': total_processing_time,
            'audio_duration': self.audio_duration,
            'session_rtf': session_rtf,
            'complexity_estimate': asdict(self.complexity_estimate) if self.complexity_estimate else None,
            'stages_processed': len(self.stage_usage),
            'downgrades_applied': len(self.downgrades_applied),
            'budget_violations': sum(1 for u in self.stage_usage.values() if u.budget_exceeded),
            'stage_metrics': {
                stage.value: {
                    'rtf': usage.actual_rtf,
                    'duration_seconds': usage.processing_duration,
                    'quality_level': usage.quality_level.value,
                    'budget_exceeded': usage.budget_exceeded,
                    'downgrade_applied': usage.downgrade_applied
                }
                for stage, usage in self.stage_usage.items()
            },
            'downgrades_applied': self.downgrades_applied,
            'recommendations': recommendations,
            'performance_grade': self._calculate_performance_grade()
        }
        
        self.logger.info("Processing session completed",
                        context=session_summary)
        
        return session_summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on session performance"""
        recommendations = []
        
        # Check for frequent budget violations
        violations = [u for u in self.stage_usage.values() if u.budget_exceeded]
        if len(violations) > len(self.stage_usage) * 0.3:  # More than 30% violations
            recommendations.append(
                "Consider using more aggressive downgrade strategy for better time predictability"
            )
        
        # Check for excessive downgrades
        if len(self.downgrades_applied) > 3:
            recommendations.append(
                "Multiple downgrades applied - consider increasing global timeout or using faster hardware"
            )
        
        # Check for specific stage issues
        high_rtf_stages = [
            (stage.value, usage.actual_rtf) 
            for stage, usage in self.stage_usage.items()
            if usage.actual_rtf > self.rtf_budgets[stage].maximum_rtf
        ]
        
        if high_rtf_stages:
            stage_names = ", ".join([name for name, _ in high_rtf_stages])
            recommendations.append(
                f"Stages with high RTF detected ({stage_names}) - consider optimizing or adding more budget"
            )
        
        # Memory usage recommendations
        if self.resource_snapshots:
            avg_memory = sum(s.memory_percent for s in self.resource_snapshots) / len(self.resource_snapshots)
            if avg_memory > 80:
                recommendations.append(
                    "High memory usage detected - consider reducing concurrent processing or adding RAM"
                )
        
        return recommendations
    
    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade for the session"""
        total_score = 0
        max_score = 0
        
        # Time efficiency score (30%)
        session_time_minutes = (time.time() - self.session_start_time) / 60.0
        time_efficiency = min(self.global_timeout_minutes / session_time_minutes, 1.0)
        total_score += time_efficiency * 30
        max_score += 30
        
        # Budget compliance score (25%)  
        violations = sum(1 for u in self.stage_usage.values() if u.budget_exceeded)
        budget_compliance = max(0, 1.0 - violations / len(self.stage_usage)) if self.stage_usage else 0
        total_score += budget_compliance * 25
        max_score += 25
        
        # Quality preservation score (25%)
        downgrades = len(self.downgrades_applied)
        max_possible_downgrades = len(self.stage_usage) * 2  # Assume max 2 downgrades per stage
        quality_score = max(0, 1.0 - downgrades / max_possible_downgrades) if max_possible_downgrades > 0 else 1.0
        total_score += quality_score * 25
        max_score += 25
        
        # System efficiency score (20%)
        if self.resource_snapshots:
            avg_cpu = sum(s.cpu_percent for s in self.resource_snapshots) / len(self.resource_snapshots)
            avg_memory = sum(s.memory_percent for s in self.resource_snapshots) / len(self.resource_snapshots)
            system_efficiency = 1.0 - max(avg_cpu, avg_memory) / 100.0
            total_score += max(0, system_efficiency) * 20
        max_score += 20
        
        # Calculate final percentage
        percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        
        # Convert to letter grade
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B" 
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status and metrics
        
        Returns:
            Dictionary with current status information
        """
        current_time = time.time()
        elapsed_minutes = (current_time - self.session_start_time) / 60.0
        remaining_minutes = max(0, self.global_timeout_minutes - elapsed_minutes)
        
        # Current resource snapshot
        current_resources = None
        if self.enable_resource_monitoring:
            current_resources = ResourceMonitoringSnapshot.capture_current()
        
        # Active stage information
        active_stages = {
            stage.value: {
                'elapsed_seconds': current_time - usage.start_time,
                'estimated_rtf': (current_time - usage.start_time) / self.audio_duration if self.audio_duration > 0 else 0,
                'quality_level': usage.quality_level.value,
                'downgrade_applied': usage.downgrade_applied
            }
            for stage, usage in self.stage_usage.items()
            if usage.end_time == 0.0  # Still active
        }
        
        return {
            'session_active': self.session_start_time > 0,
            'elapsed_minutes': elapsed_minutes,
            'remaining_minutes': remaining_minutes,
            'audio_duration': self.audio_duration,
            'complexity_score': self.complexity_estimate.complexity_score if self.complexity_estimate else 0,
            'active_stages': active_stages,
            'downgrades_applied_count': len(self.downgrades_applied),
            'budget_violations_count': sum(1 for u in self.stage_usage.values() if u.budget_exceeded),
            'current_resources': asdict(current_resources) if current_resources else None
        }


# Singleton instance for global access
_resource_scheduler_instance: Optional[ResourceScheduler] = None


def get_resource_scheduler() -> ResourceScheduler:
    """
    Get global resource scheduler instance
    
    Returns:
        Global ResourceScheduler instance
    """
    global _resource_scheduler_instance
    if _resource_scheduler_instance is None:
        _resource_scheduler_instance = ResourceScheduler()
    return _resource_scheduler_instance


def initialize_resource_scheduler(global_timeout_minutes: float = 30.0,
                                downgrade_strategy: DowngradeStrategy = DowngradeStrategy.BALANCED,
                                **kwargs) -> ResourceScheduler:
    """
    Initialize global resource scheduler with custom configuration
    
    Args:
        global_timeout_minutes: Maximum total processing time
        downgrade_strategy: Strategy for applying downgrades
        **kwargs: Additional configuration options
        
    Returns:
        Initialized ResourceScheduler instance
    """
    global _resource_scheduler_instance
    _resource_scheduler_instance = ResourceScheduler(
        global_timeout_minutes=global_timeout_minutes,
        downgrade_strategy=downgrade_strategy,
        **kwargs
    )
    return _resource_scheduler_instance


@trace_stage("resource_scheduled_stage")
def with_resource_scheduling(stage: ProcessingStage, 
                           audio_duration: Optional[float] = None):
    """
    Decorator for automatic resource scheduling of processing functions
    
    Args:
        stage: Processing stage being decorated
        audio_duration: Optional audio duration for RTF calculation
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            scheduler = get_resource_scheduler()
            
            # Use provided audio_duration or try to extract from scheduler
            duration = audio_duration or scheduler.audio_duration
            
            # Start stage monitoring
            usage = scheduler.start_stage(stage, metadata={'function': func.__name__})
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # End stage monitoring
                scheduler.end_stage(stage, success=True)
                
                return result
                
            except Exception as e:
                # End stage monitoring with error
                scheduler.end_stage(stage, success=False, error_message=str(e))
                raise
        
        return wrapper
    return decorator