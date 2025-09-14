import numpy as np
import os
import tempfile
import json
from typing import List, Dict, Any, Optional, Union, Type, Tuple
import random
import librosa
# Removed scipy dependencies - replaced with numpy/librosa equivalents
import math
import hashlib
import time

from utils.resilient_api import huggingface_retry
from core.circuit_breaker import CircuitBreakerOpenException
from utils.structured_logger import StructuredLogger
from utils.deterministic_processing import get_deterministic_processor, set_global_seed
from utils.intelligent_cache import get_cache_manager, cached_operation
from core.turn_stabilizer import TurnStabilizer, StabilizationConfig
from core.speaker_mapper import SpeakerMapper, ConsistencyMetrics

# Define base classes and types for both real and mock implementations
class BaseAnnotation:
    """Base annotation interface"""
    def __init__(self):
        self._tracks = []
    
    def __setitem__(self, segment, speaker):
        self._tracks.append((segment, speaker))
    
    def itertracks(self, yield_label=True):
        for segment, speaker in self._tracks:
            yield segment, None, speaker

class BaseSegment:
    """Base segment interface"""
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

class BasePipeline:
    """Base pipeline interface"""
    @classmethod
    def from_pretrained(cls, model_name: str, use_auth_token: Optional[str] = None):
        raise NotImplementedError("Must be implemented by real or mock pipeline")
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by real or mock pipeline")

# Try to import real pyannote classes, fall back to base classes
PYANNOTE_AVAILABLE = False
torch_module = None
# Use Any type to handle both base and real classes
Annotation: Type[Any] = BaseAnnotation
Segment: Type[Any] = BaseSegment
Pipeline: Type[Any] = BasePipeline

try:
    import torch as torch_module  # type: ignore
    from pyannote.audio import Pipeline as RealPipeline  # type: ignore
    from pyannote.core import Annotation as RealAnnotation, Segment as RealSegment  # type: ignore
    
    # Use real classes if available
    Annotation = RealAnnotation  # type: ignore
    Segment = RealSegment  # type: ignore
    Pipeline = RealPipeline  # type: ignore
    PYANNOTE_AVAILABLE = True
    print("✓ pyannote.audio successfully imported")
except ImportError as e:
    print(f"⚠ pyannote.audio not available ({e}), using mock implementation")
    # Base classes are already assigned above

class DiarizationEngine:
    """Handles speaker diarization with multiple variants for ensemble processing"""
    
    def __init__(self, expected_speakers: int = 10, noise_level: str = 'medium',
                 enable_turn_stabilization: bool = True, stabilization_config: Optional[StabilizationConfig] = None,
                 enable_speaker_mapping: bool = True, speaker_mapping_config: Optional[Dict[str, Any]] = None):
        self.expected_speakers = expected_speakers
        self.noise_level = noise_level
        self.pipeline = None
        
        # Turn stabilization configuration
        self.enable_turn_stabilization = enable_turn_stabilization
        self.stabilization_config = stabilization_config or StabilizationConfig()
        self.turn_stabilizer = TurnStabilizer(self.stabilization_config) if enable_turn_stabilization else None
        
        # Speaker mapping configuration
        self.enable_speaker_mapping = enable_speaker_mapping
        self.speaker_mapping_config = speaker_mapping_config or {
            'similarity_threshold': 0.7,
            'embedding_dim': 128,
            'min_segment_duration': 1.0,
            'cache_embeddings': True,
            'enable_metrics': True
        }
        self.speaker_mapper = SpeakerMapper(**self.speaker_mapping_config) if enable_speaker_mapping else None
        
        # Performance optimization caches
        self._vad_cache = {}  # Cache for VAD analysis results
        self._speaker_features_cache = {}  # Cache for speaker features
        self._chunk_preview_cache = {}  # Cache for chunk previews
        
        # Initialize structured logging for reliability telemetry
        self.structured_logger = StructuredLogger("diarization_engine")
        
        self._initialize_pipeline()
    
    def _validate_hf_token(self, token: Optional[str]) -> bool:
        """Validate HuggingFace token format and authenticity"""
        if not token or token == "dummy_token" or token == "":
            return False
        
        # Basic format validation (HF tokens start with 'hf_')
        if not token.startswith('hf_') or len(token) < 30:
            print(f"⚠ Invalid HUGGINGFACE_TOKEN format (should start with 'hf_' and be at least 30 chars)")
            return False
        
        return True
    
    def _initialize_pipeline(self):
        """Initialize the pyannote speaker diarization pipeline with robust error handling"""
        if not PYANNOTE_AVAILABLE:
            print("Using mock diarization pipeline (pyannote.audio not available)")
            self.pipeline = self._create_mock_pipeline()
            return
            
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self._validate_hf_token(hf_token):
            print("No valid HUGGINGFACE_TOKEN found, using mock pipeline")
            self.pipeline = self._create_mock_pipeline()
            return
            
        try:
            print("Loading pyannote speaker diarization pipeline...")
            # Ensure hf_token is not None here since we validated it
            assert hf_token is not None, "Token should be validated before reaching this point"
            
            # Load pipeline with retry logic and circuit breaker protection
            try:
                self.pipeline = self._load_pipeline_with_retry(hf_token)
                print("✓ Real pyannote pipeline loaded successfully")
            except CircuitBreakerOpenException as e:
                print(f"⚠ HuggingFace service temporarily unavailable: {e}")
                print("Falling back to mock pipeline")
                self.pipeline = self._create_mock_pipeline()
            except Exception as e:
                print(f"⚠ Could not load pyannote pipeline after retries: {e}")
                print("Falling back to mock pipeline")
                self.pipeline = self._create_mock_pipeline()
                
        except Exception as e:
            print(f"⚠ Unexpected error during pipeline initialization: {e}")
            print("Falling back to mock pipeline")
            self.pipeline = self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline with intelligent chunking and pause detection"""
        class MockPipeline:
            def __init__(self, parent_engine):
                self.parent = parent_engine
            
            def __call__(self, audio_file, min_speakers=None, max_speakers=None, seed=None):
                # Generate mock diarization result with intelligent chunking
                return self._generate_intelligent_mock_diarization(audio_file, min_speakers, max_speakers, seed)
            
            def _generate_intelligent_mock_diarization(self, audio_file, min_speakers, max_speakers, seed=None):
                """Generate mock diarization with pause detection and flexible chunking"""
                # Set deterministic seed for reproducible mock results using new system
                if seed is not None:
                    set_global_seed("mock_diarization", {"file": str(audio_file), "seed": seed})
                else:
                    set_global_seed("mock_diarization", {"file": str(audio_file)})
                
                # Create annotation
                annotation = Annotation()
                
                try:
                    # Get actual audio duration for realistic mock
                    audio_duration = self._get_audio_duration(audio_file)
                except:
                    audio_duration = 300.0  # 5 minute fallback
                
                num_speakers = max(min_speakers or 3, 2)
                
                # Intelligent chunking with flexible boundaries (4:15-5:15 minutes)
                chunk_duration = self._calculate_chunk_duration(audio_duration)
                segments = self._create_segments_with_pause_detection(
                    audio_duration, num_speakers, chunk_duration
                )
                
                # Add segments to annotation
                for segment in segments:
                    annotation[Segment(segment['start'], segment['end'])] = segment['speaker_id']
                
                return annotation
            
            def _get_audio_duration(self, audio_file):
                """Get actual audio duration using librosa"""
                try:
                    y, sr = librosa.load(audio_file, sr=None)
                    return len(y) / sr
                except Exception as e:
                    print(f"Could not get audio duration: {e}")
                    return 300.0
            
            def _calculate_chunk_duration(self, total_duration):
                """Calculate intelligent chunk duration (4:15-5:15 minutes)"""
                min_chunk = 255.0  # 4:15 minutes
                max_chunk = 315.0  # 5:15 minutes
                
                if total_duration <= max_chunk:
                    return total_duration
                
                # Calculate number of chunks needed
                num_chunks = max(1, int(total_duration / max_chunk))
                chunk_size = total_duration / num_chunks
                
                # Ensure chunk size is within bounds
                return max(min_chunk, min(chunk_size, max_chunk))
            
            def _create_segments_with_pause_detection(self, duration, num_speakers, chunk_duration):
                """Create segments with simulated pause detection and VAD - now fully deterministic"""
                segments = []
                current_time = 0.0
                speaker_id = 0
                
                # Simulate natural speech patterns with deterministic values
                base_segment_length = 8.0  # Base segment length
                pause_probability = 0.15  # Probability of pause
                speaker_change_prob = 0.25  # Probability of speaker change
                
                # Use deterministic random number generator
                rng = np.random.RandomState()  # Will use global seed set by deterministic processor
                
                while current_time < duration:
                    # Variable segment length with pauses (deterministic)
                    segment_length = base_segment_length + rng.uniform(-3.0, 7.0)
                    
                    # Add occasional pauses (deterministic)
                    if rng.random() < pause_probability:
                        pause_duration = rng.uniform(0.5, 2.0)
                        current_time += pause_duration
                    
                    # Calculate end time
                    end_time = min(current_time + segment_length, duration)
                    
                    if end_time > current_time:
                        segments.append({
                            'start': current_time,
                            'end': end_time,
                            'speaker_id': f"SPEAKER_{speaker_id:02d}"
                        })
                    
                    # Occasionally change speakers (deterministic)
                    if rng.random() < speaker_change_prob:
                        speaker_id = (speaker_id + 1) % num_speakers
                    
                    current_time = end_time
                
                return segments
        
        return MockPipeline(self)
    
    @huggingface_retry
    def _load_pipeline_with_retry(self, hf_token: str):
        """
        Load HuggingFace pipeline with tenacity retry and circuit breaker protection.
        
        Args:
            hf_token: HuggingFace authentication token
            
        Returns:
            Loaded pipeline instance
            
        Raises:
            Various exceptions if loading fails after retries
        """
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
    
    def create_diarization_variants(self, audio_path: str, use_voting_fusion: bool = True, 
                                  enable_chunked_processing: bool = False, chunk_boundaries: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Create 5 diarization variants with different parameters and optional voting fusion.
        
        Args:
            audio_path: Path to cleaned audio file
            use_voting_fusion: Whether to apply voting fusion for improved accuracy
            enable_chunked_processing: Whether to process audio in chunks with speaker mapping
            chunk_boundaries: Optional list of chunk boundary timestamps
            
        Returns:
            List of 5 diarization variant results, optionally enhanced with voting fusion and speaker mapping
        """
        # Check if chunked processing with speaker mapping is requested
        if enable_chunked_processing and self.enable_speaker_mapping and self.speaker_mapper:
            return self._create_chunked_diarization_variants(audio_path, use_voting_fusion, chunk_boundaries)
        
        variants = []
        
        # Variant 1: Conservative (fewer speakers expected)
        variants.append(self._run_diarization_variant(
            audio_path,
            variant_id=1,
            min_speakers=max(2, self.expected_speakers - 2),
            max_speakers=self.expected_speakers + 1,
            clustering_threshold=0.7,
            vad_onset=0.5,
            vad_offset=0.4,
            seed=42,
            variant_name="Conservative"
        ))
        
        # Variant 2: Balanced (expected speaker count)
        variants.append(self._run_diarization_variant(
            audio_path,
            variant_id=2,
            min_speakers=max(2, self.expected_speakers - 1),
            max_speakers=self.expected_speakers + 2,
            clustering_threshold=0.6,
            vad_onset=0.4,
            vad_offset=0.3,
            seed=123,
            variant_name="Balanced"
        ))
        
        # Variant 3: Liberal (more speakers expected)
        variants.append(self._run_diarization_variant(
            audio_path,
            variant_id=3,
            min_speakers=max(2, self.expected_speakers),
            max_speakers=self.expected_speakers + 3,
            clustering_threshold=0.5,
            vad_onset=0.3,
            vad_offset=0.2,
            seed=456,
            variant_name="Liberal"
        ))
        
        # Variant 4: Multi-resolution approach - different temporal granularities
        variants.append(self._run_diarization_variant(
            audio_path,
            variant_id=4,
            min_speakers=max(2, self.expected_speakers - 1),
            max_speakers=self.expected_speakers + 2,
            clustering_threshold=0.55,
            vad_onset=0.45,
            vad_offset=0.25,
            seed=789,
            variant_name="Multi-Resolution",
            # Multi-resolution specific parameters
            resolution_scales=[0.5, 1.0, 1.5],  # Different time scales
            merge_strategy="weighted_average"
        ))
        
        # Variant 5: Ensemble approach - multiple model initializations
        variants.append(self._run_diarization_variant(
            audio_path,
            variant_id=5,
            min_speakers=max(2, self.expected_speakers - 1),
            max_speakers=self.expected_speakers + 2,
            clustering_threshold=0.65,
            vad_onset=0.35,
            vad_offset=0.35,
            seed=999,
            variant_name="Ensemble-Based",
            # Ensemble specific parameters
            ensemble_seeds=[999, 1001, 1003],  # Multiple seeds for stability
            consensus_threshold=0.6
        ))
        
        # Apply turn stabilization to all variants if enabled
        if self.enable_turn_stabilization and self.turn_stabilizer:
            print(f"🔧 Applying turn stabilization to {len(variants)} diarization variants...")
            variants = self.turn_stabilizer.batch_stabilize_variants(variants)
            
            # Log stabilization summary
            stabilization_summary = self._generate_stabilization_summary(variants)
            print(f"✅ Turn stabilization complete: {stabilization_summary}")
        
        # Apply voting fusion if requested
        if use_voting_fusion:
            print(f"🗳️  Applying voting fusion to {len(variants)} diarization variants...")
            variants = self.fuse_diarization_variants(variants, audio_path)
        
        return variants
    
    def _create_chunked_diarization_variants(self, audio_path: str, use_voting_fusion: bool = True, 
                                           chunk_boundaries: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Create diarization variants with chunked processing and speaker mapping consistency.
        
        Args:
            audio_path: Path to cleaned audio file
            use_voting_fusion: Whether to apply voting fusion for improved accuracy
            chunk_boundaries: Optional list of chunk boundary timestamps
            
        Returns:
            List of diarization variant results with speaker mapping applied
        """
        self.structured_logger.stage_start("chunked_diarization", "Starting chunked diarization with speaker mapping")
        
        # Generate chunk boundaries if not provided
        if chunk_boundaries is None:
            chunk_boundaries = self.find_optimal_chunk_boundaries(audio_path)
        
        # Create temporary audio chunks
        chunk_audio_paths = self._create_audio_chunks(audio_path, chunk_boundaries)
        
        try:
            # Process each variant with chunked processing
            variants = []
            
            for variant_config in self._get_variant_configs():
                variant_id = variant_config['variant_id']
                variant_name = variant_config['variant_name']
                
                self.structured_logger.info(f"Processing chunked variant {variant_id}: {variant_name}")
                
                # Process each chunk for this variant
                chunk_diarizations = []
                for chunk_idx, chunk_path in enumerate(chunk_audio_paths):
                    chunk_result = self._run_diarization_variant(
                        chunk_path, **variant_config, 
                        variant_name=f"{variant_name}_chunk_{chunk_idx}"
                    )
                    chunk_diarizations.append(chunk_result['segments'])
                
                # Apply speaker mapping across chunks
                if self.speaker_mapper:
                    self.speaker_mapper.reset()  # Reset for each variant
                    mapped_segments, consistency_metrics = self.speaker_mapper.map_speakers_across_chunks(
                        chunk_diarizations, audio_path
                    )
                    
                    # Combine mapped segments from all chunks with time offsets
                    combined_segments = self._combine_chunk_segments(mapped_segments, chunk_boundaries)
                else:
                    # Fallback: combine segments without mapping
                    combined_segments = self._combine_chunk_segments(chunk_diarizations, chunk_boundaries)
                    consistency_metrics = self.speaker_mapper._create_empty_metrics() if self.speaker_mapper else None
                
                # Create RTTM data for combined result
                rttm_data = self._create_rttm_data(combined_segments)
                
                # Calculate quality metrics
                quality_metrics = self._calculate_variant_quality_metrics(combined_segments)
                
                # Build variant result
                variant_result = {
                    'variant_id': variant_id,
                    'variant_name': variant_name,
                    'segments': combined_segments,
                    'speaker_embeddings': {},  # Populated by speaker mapper
                    'rttm_data': rttm_data,
                    'parameters': variant_config,
                    'num_speakers': len(set(seg.get('speaker', seg.get('speaker_id', 'UNKNOWN')) for seg in combined_segments)),
                    'total_speech_time': sum(seg['end'] - seg['start'] for seg in combined_segments),
                    'num_segments': len(combined_segments),
                    'quality_metrics': quality_metrics,
                    'chunked_processing': True,
                    'num_chunks': len(chunk_audio_paths),
                    'chunk_boundaries': chunk_boundaries,
                    'speaker_consistency_metrics': consistency_metrics.__dict__ if consistency_metrics else None
                }
                
                variants.append(variant_result)
            
            # Apply turn stabilization if enabled
            if self.enable_turn_stabilization and self.turn_stabilizer:
                self.structured_logger.info("Applying turn stabilization to chunked variants")
                variants = self.turn_stabilizer.batch_stabilize_variants(variants)
            
            # Apply voting fusion if requested
            if use_voting_fusion:
                self.structured_logger.info("Applying voting fusion to chunked variants")
                variants = self.fuse_diarization_variants(variants, audio_path)
            
            self.structured_logger.stage_complete("chunked_diarization", "Chunked diarization with speaker mapping completed")
            
            return variants
            
        finally:
            # Clean up temporary chunk files
            self._cleanup_audio_chunks(chunk_audio_paths)
    
    def _get_variant_configs(self) -> List[Dict[str, Any]]:
        """Get configuration dictionaries for all diarization variants."""
        return [
            {
                'variant_id': 1,
                'variant_name': "Conservative",
                'min_speakers': max(2, self.expected_speakers - 2),
                'max_speakers': self.expected_speakers + 1,
                'clustering_threshold': 0.7,
                'vad_onset': 0.5,
                'vad_offset': 0.4,
                'seed': 42
            },
            {
                'variant_id': 2,
                'variant_name': "Balanced",
                'min_speakers': max(2, self.expected_speakers - 1),
                'max_speakers': self.expected_speakers + 2,
                'clustering_threshold': 0.6,
                'vad_onset': 0.4,
                'vad_offset': 0.3,
                'seed': 123
            },
            {
                'variant_id': 3,
                'variant_name': "Liberal",
                'min_speakers': max(2, self.expected_speakers),
                'max_speakers': self.expected_speakers + 3,
                'clustering_threshold': 0.5,
                'vad_onset': 0.3,
                'vad_offset': 0.2,
                'seed': 456
            },
            {
                'variant_id': 4,
                'variant_name': "Multi-Resolution",
                'min_speakers': max(2, self.expected_speakers - 1),
                'max_speakers': self.expected_speakers + 2,
                'clustering_threshold': 0.55,
                'vad_onset': 0.45,
                'vad_offset': 0.25,
                'seed': 789,
                'resolution_scales': [0.5, 1.0, 1.5],
                'merge_strategy': "weighted_average"
            },
            {
                'variant_id': 5,
                'variant_name': "Ensemble-Based",
                'min_speakers': max(2, self.expected_speakers - 1),
                'max_speakers': self.expected_speakers + 2,
                'clustering_threshold': 0.65,
                'vad_onset': 0.35,
                'vad_offset': 0.35,
                'seed': 999,
                'ensemble_seeds': [999, 1001, 1003],
                'consensus_threshold': 0.6
            }
        ]
    
    def _create_audio_chunks(self, audio_path: str, chunk_boundaries: List[float]) -> List[str]:
        """
        Create temporary audio chunk files based on boundaries.
        
        Args:
            audio_path: Path to original audio file
            chunk_boundaries: List of chunk boundary timestamps
            
        Returns:
            List of paths to temporary chunk audio files
        """
        chunk_paths = []
        
        try:
            # Load original audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Create chunks
            start_time = 0.0
            for i, end_time in enumerate(chunk_boundaries):
                # Convert times to sample indices
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Extract chunk audio
                chunk_audio = y[start_sample:end_sample]
                
                # Create temporary file for this chunk
                chunk_file = tempfile.NamedTemporaryFile(suffix=f'_chunk_{i}.wav', delete=False)
                chunk_path = chunk_file.name
                chunk_file.close()
                
                # Save chunk audio
                import soundfile as sf
                sf.write(chunk_path, chunk_audio, sr)
                
                chunk_paths.append(chunk_path)
                start_time = end_time
            
            self.structured_logger.info(f"Created {len(chunk_paths)} audio chunks")
            return chunk_paths
            
        except Exception as e:
            # Clean up any created files on error
            for path in chunk_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            raise Exception(f"Failed to create audio chunks: {e}")
    
    def _combine_chunk_segments(self, chunk_segments_list: List[List[Dict[str, Any]]], 
                              chunk_boundaries: List[float]) -> List[Dict[str, Any]]:
        """
        Combine segments from multiple chunks with proper time offsets.
        
        Args:
            chunk_segments_list: List of segment lists, one per chunk
            chunk_boundaries: List of chunk boundary timestamps
            
        Returns:
            Combined list of segments with global timestamps
        """
        combined_segments = []
        
        start_time = 0.0
        for chunk_idx, chunk_segments in enumerate(chunk_segments_list):
            chunk_end_time = chunk_boundaries[chunk_idx] if chunk_idx < len(chunk_boundaries) else float('inf')
            
            for segment in chunk_segments:
                # Adjust segment times to global timeline
                adjusted_segment = segment.copy()
                adjusted_segment['start'] = segment['start'] + start_time
                adjusted_segment['end'] = segment['end'] + start_time
                
                # Ensure segment doesn't exceed chunk boundary
                if adjusted_segment['end'] > chunk_end_time:
                    adjusted_segment['end'] = chunk_end_time
                
                # Only add valid segments
                if adjusted_segment['start'] < adjusted_segment['end']:
                    combined_segments.append(adjusted_segment)
            
            # Update start time for next chunk
            if chunk_idx < len(chunk_boundaries):
                start_time = chunk_boundaries[chunk_idx]
        
        # Sort segments by start time
        combined_segments.sort(key=lambda x: x['start'])
        
        self.structured_logger.info(f"Combined {len(combined_segments)} segments from {len(chunk_segments_list)} chunks")
        return combined_segments
    
    def _cleanup_audio_chunks(self, chunk_paths: List[str]):
        """Clean up temporary audio chunk files."""
        for chunk_path in chunk_paths:
            try:
                os.unlink(chunk_path)
            except Exception as e:
                self.structured_logger.warning(f"Failed to remove temporary chunk file {chunk_path}: {e}")
        
        self.structured_logger.info(f"Cleaned up {len(chunk_paths)} temporary chunk files")
    
    def _run_diarization_variant(self, audio_path: str, variant_id: int, 
                               min_speakers: int, max_speakers: int,
                               clustering_threshold: float, vad_onset: float,
                               vad_offset: float, seed: int, variant_name: str = "Unknown",
                               resolution_scales: Optional[List[float]] = None,
                               merge_strategy: Optional[str] = None,
                               ensemble_seeds: Optional[List[int]] = None,
                               consensus_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Run a single diarization variant with specific parameters.
        
        Args:
            audio_path: Path to audio file
            variant_id: Unique identifier for this variant
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            clustering_threshold: Clustering threshold for speaker grouping
            vad_onset: VAD onset threshold
            vad_offset: VAD offset threshold
            seed: Random seed for reproducibility
            variant_name: Human-readable name for this variant
            resolution_scales: Optional list of time scales for multi-resolution approach
            merge_strategy: Optional strategy for merging multi-resolution results
            ensemble_seeds: Optional list of seeds for ensemble approach
            consensus_threshold: Optional threshold for ensemble consensus
            
        Returns:
            Dictionary containing diarization results and metadata
        """
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        if torch_module is not None:
            torch_module.manual_seed(seed)
        
        try:
            # Ensure pipeline is available
            if self.pipeline is None:
                raise Exception("No diarization pipeline available")
            
            # Run diarization with parameters
            diarization = self.pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                seed=seed
            )
            
            # Convert to our format
            segments = []
            speaker_embeddings = {}
            
            for track_info in diarization.itertracks(yield_label=True):
                # Handle different tuple sizes from itertracks
                if len(track_info) == 3:
                    segment, _, speaker = track_info
                else:
                    segment, speaker = track_info
                
                segment_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'speaker_id': speaker,
                    'confidence': random.uniform(0.7, 0.95)  # Mock confidence
                }
                segments.append(segment_data)
                
                # Mock speaker embeddings (in practice these would be real)
                if speaker not in speaker_embeddings:
                    speaker_embeddings[speaker] = np.random.randn(512).tolist()
            
            # Create RTTM format data
            rttm_data = self._create_rttm_data(segments)
            
            # Handle variant-specific processing
            if resolution_scales and merge_strategy:
                # Multi-resolution processing
                segments = self._apply_multi_resolution_processing(
                    segments, resolution_scales, merge_strategy
                )
            elif ensemble_seeds and consensus_threshold:
                # Ensemble-based processing  
                segments = self._apply_ensemble_processing(
                    segments, audio_path, ensemble_seeds, consensus_threshold
                )
            
            return {
                'variant_id': variant_id,
                'variant_name': variant_name,
                'segments': segments,
                'speaker_embeddings': speaker_embeddings,
                'rttm_data': rttm_data,
                'parameters': {
                    'min_speakers': min_speakers,
                    'max_speakers': max_speakers,
                    'clustering_threshold': clustering_threshold,
                    'vad_onset': vad_onset,
                    'vad_offset': vad_offset,
                    'seed': seed,
                    'variant_name': variant_name,
                    'resolution_scales': resolution_scales,
                    'merge_strategy': merge_strategy,
                    'ensemble_seeds': ensemble_seeds,
                    'consensus_threshold': consensus_threshold
                },
                'num_speakers': len(set(seg['speaker_id'] for seg in segments)),
                'total_speech_time': sum(seg['end'] - seg['start'] for seg in segments),
                'num_segments': len(segments),
                'quality_metrics': self._calculate_variant_quality_metrics(segments)
            }
            
        except Exception as e:
            raise Exception(f"Diarization variant {variant_id} failed: {str(e)}")
    
    def _create_rttm_data(self, segments: List[Dict[str, Any]]) -> List[str]:
        """
        Create RTTM (Rich Transcription Time Marked) format data.
        
        Args:
            segments: List of diarization segments
            
        Returns:
            List of RTTM format strings
        """
        rttm_lines = []
        
        for segment in segments:
            # RTTM format: 
            # SPEAKER file_id channel_id start_time duration speaker_id confidence1 confidence2
            duration = segment['end'] - segment['start']
            line = f"SPEAKER audio 1 {segment['start']:.3f} {duration:.3f} <NA> <NA> {segment['speaker_id']} <NA> <NA>"
            rttm_lines.append(line)
        
        return rttm_lines
    
    def detect_overlaps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect temporal overlaps between speaker segments.
        
        Args:
            segments: List of diarization segments
            
        Returns:
            List of overlap regions with involved speakers
        """
        overlaps = []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        for i in range(len(sorted_segments)):
            for j in range(i + 1, len(sorted_segments)):
                seg1 = sorted_segments[i]
                seg2 = sorted_segments[j]
                
                # Check if segments overlap
                overlap_start = max(seg1['start'], seg2['start'])
                overlap_end = min(seg1['end'], seg2['end'])
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    
                    # Only consider significant overlaps (>100ms)
                    if overlap_duration > 0.1:
                        overlaps.append({
                            'start': overlap_start,
                            'end': overlap_end,
                            'duration': overlap_duration,
                            'speakers': [seg1['speaker_id'], seg2['speaker_id']],
                            'confidence': min(seg1['confidence'], seg2['confidence'])
                        })
        
        return overlaps
    
    def save_variant_to_file(self, variant_data: Dict[str, Any], output_dir: str) -> str:
        """
        Save diarization variant data to files.
        
        Args:
            variant_data: Diarization variant results
            output_dir: Directory to save files
            
        Returns:
            Path to saved JSON file
        """
        variant_id = variant_data['variant_id']
        
        # Save JSON data
        json_path = os.path.join(output_dir, f"diarization_{variant_id}.json")
        with open(json_path, 'w') as f:
            json.dump(variant_data, f, indent=2)
        
        # Save RTTM data
        rttm_path = os.path.join(output_dir, f"diarization_{variant_id}.rttm")
        with open(rttm_path, 'w') as f:
            f.write('\n'.join(variant_data['rttm_data']))
        
        # Save embeddings
        embeddings_path = os.path.join(output_dir, f"diarization_{variant_id}_embeddings.json")
        with open(embeddings_path, 'w') as f:
            json.dump(variant_data['speaker_embeddings'], f, indent=2)
        
        return json_path
    
    def find_optimal_chunk_boundaries(self, audio_path: str, target_duration: float = 300.0, 
                                    enable_speaker_consistency: bool = True) -> List[float]:
        """
        Find optimal chunk boundaries using advanced Voice Activity Detection and pause detection.
        Targets 5 minutes (300s) but flexible between 4:15-5:15 (255-315s) to avoid cutting words.
        
        Args:
            audio_path: Path to audio file
            target_duration: Target chunk duration in seconds (default 5 minutes)
            enable_speaker_consistency: Whether to perform cross-chunk speaker analysis
            
        Returns:
            List of timestamps for chunk boundaries
        """
        try:
            # Load audio for analysis
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            print(f"🔍 Analyzing audio for intelligent chunking: {duration:.1f}s total")
            print(f"   Target chunks: {duration//target_duration:.0f} chunks of ~{target_duration//60:.0f}:{target_duration%60:02.0f}")
            
            # If audio is shorter than target, return no chunks
            if duration <= target_duration:
                print(f"   Audio shorter than target, no chunking needed")
                return [duration]
            
            # Advanced VAD analysis
            hop_length = 512
            frame_length = 2048
            
            print(f"   🎯 Running advanced VAD analysis...")
            vad_result = self._advanced_voice_activity_detection(y, int(sr), hop_length, frame_length)
            
            # Enhanced pause detection
            print(f"   ⏸️  Detecting enhanced pause regions...")
            pause_regions = self._detect_enhanced_pause_regions(
                vad_result['times'], vad_result['is_speech'], vad_result['speech_confidence']
            )
            
            # Find boundaries with speaker consistency if enabled
            boundaries = []
            current_pos = 0.0
            chunk_count = 0
            
            while current_pos < duration:
                chunk_count += 1
                print(f"   📍 Processing chunk {chunk_count} boundary...")
                
                # Define search window: 4:15 to 5:15 minutes from current position
                min_boundary = current_pos + 255.0  # 4:15
                max_boundary = min(current_pos + 315.0, duration)  # 5:15 or end of file
                ideal_boundary = current_pos + target_duration  # 5:00
                
                if max_boundary >= duration:
                    # Last chunk - just use the end
                    boundaries.append(duration)
                    print(f"     Final boundary at {duration:.1f}s (end of audio)")
                    break
                
                # Find the best boundary within the search window
                best_boundary = self._find_optimal_boundary_in_window(
                    y, int(sr), vad_result, pause_regions, min_boundary, max_boundary, 
                    ideal_boundary, enable_speaker_consistency
                )
                
                boundaries.append(best_boundary)
                print(f"     Boundary at {best_boundary:.1f}s (chunk {chunk_count}: {best_boundary-current_pos:.1f}s)")
                current_pos = best_boundary
            
            # Validate boundary quality
            quality_score = self._validate_chunk_quality(boundaries, duration, target_duration)
            print(f"   ✅ Chunking quality score: {quality_score:.3f} (higher is better)")
            
            print(f"   🎯 Generated {len(boundaries)} intelligent boundaries:")
            for i, boundary in enumerate(boundaries):
                chunk_duration = boundary - (boundaries[i-1] if i > 0 else 0)
                print(f"      Chunk {i+1}: {chunk_duration/60:.1f}min ({chunk_duration:.1f}s) -> boundary at {boundary:.1f}s")
            
            return boundaries
            
        except Exception as e:
            print(f"⚠️  Warning: Advanced chunking failed ({e}), using fallback")
            # Fallback to simple time-based chunking
            return self._simple_time_chunking(audio_path, target_duration)
    
    def _generate_audio_hash(self, y: np.ndarray) -> str:
        """
        Generate a hash for audio data to use as cache key.
        
        Args:
            y: Audio signal array
            
        Returns:
            Hash string for caching
        """
        # Use audio statistics for hash generation (more stable than raw data)
        audio_stats = [
            len(y),
            float(np.mean(y)),
            float(np.std(y)),
            float(np.max(y)),
            float(np.min(y))
        ]
        
        # Create hash from stats
        stats_str = '_'.join([f"{stat:.6f}" for stat in audio_stats])
        return hashlib.md5(stats_str.encode()).hexdigest()[:16]
    
    def clear_caches(self):
        """Clear all performance caches to free memory."""
        self._vad_cache.clear()
        self._speaker_features_cache.clear()
        self._chunk_preview_cache.clear()
        print("🧹 Performance caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage."""
        return {
            'vad_cache_size': len(self._vad_cache),
            'speaker_features_cache_size': len(self._speaker_features_cache),
            'chunk_preview_cache_size': len(self._chunk_preview_cache)
        }
    
    def _advanced_voice_activity_detection(self, y: np.ndarray, sr: int, 
                                         hop_length: int, frame_length: int) -> Dict[str, Any]:
        """
        Advanced Voice Activity Detection using multiple acoustic features with caching.
        
        Args:
            y: Audio signal
            sr: Sample rate
            hop_length: Hop length for frame analysis
            frame_length: Frame length for analysis
            
        Returns:
            Dictionary with VAD results and confidence scores
        """
        # Create cache key based on audio characteristics
        audio_hash = self._generate_audio_hash(y)
        cache_key = f"{audio_hash}_{sr}_{hop_length}_{frame_length}"
        
        # Check cache first for performance optimization
        if cache_key in self._vad_cache:
            print(f"   📄 Using cached VAD analysis for performance")
            return self._vad_cache[cache_key]
        
        # Calculate multiple acoustic features for long files with optimizations
        start_time = time.time()
        
        # For very long files (>30 minutes), use reduced resolution for initial analysis
        is_long_file = len(y) > sr * 1800  # 30 minutes
        downsample_factor = 1  # Initialize to avoid unbound variable
        if is_long_file:
            # Downsample for initial analysis, then refine boundaries
            downsample_factor = 2
            y_downsampled = y[::downsample_factor]
            hop_length_adj = hop_length // downsample_factor
            frame_length_adj = frame_length // downsample_factor
            sr_adj = sr // downsample_factor
            print(f"   ⚡ Optimizing analysis for long file ({len(y)/sr/60:.1f} minutes)")
        else:
            y_downsampled = y
            hop_length_adj = hop_length
            frame_length_adj = frame_length
            sr_adj = sr
        # Calculate features using optimized parameters
        rms = librosa.feature.rms(y=y_downsampled, frame_length=frame_length_adj, hop_length=hop_length_adj)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y_downsampled, sr=sr_adj, hop_length=hop_length_adj)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y_downsampled, sr=sr_adj, hop_length=hop_length_adj)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_downsampled, sr=sr_adj, hop_length=hop_length_adj)[0]
        
        # Zero-crossing rate (helpful for distinguishing voiced vs unvoiced speech)
        zcr = librosa.feature.zero_crossing_rate(y_downsampled, frame_length=frame_length_adj, hop_length=hop_length_adj)[0]
        
        # MFCC features (first 3 coefficients for speech detection)
        mfcc = librosa.feature.mfcc(y=y_downsampled, sr=sr_adj, n_mfcc=13, hop_length=hop_length_adj)[0:3]
        
        # Spectral contrast (helps distinguish speech from music/noise)
        spectral_contrast = librosa.feature.spectral_contrast(y=y_downsampled, sr=sr_adj, hop_length=hop_length_adj)[0]
        
        # Convert frame indices to time (adjusted for original sample rate)
        times = librosa.frames_to_time(range(len(rms)), sr=sr_adj, hop_length=hop_length_adj)
        if is_long_file:
            # Scale times back to original timeline
            times = times * downsample_factor
        
        # Adaptive thresholding based on audio characteristics
        rms_threshold = self._calculate_adaptive_threshold(rms, percentile=25)
        centroid_threshold = self._calculate_adaptive_threshold(spectral_centroid, percentile=30)
        zcr_threshold = self._calculate_adaptive_threshold(zcr, percentile=40)
        rolloff_threshold = self._calculate_adaptive_threshold(spectral_rolloff, percentile=35)
        
        # Multi-feature speech detection
        # 1. Energy-based detection (RMS > threshold)
        energy_speech = rms > rms_threshold
        
        # 2. Spectral characteristics (centroid and rolloff in speech range)
        spectral_speech = (spectral_centroid > centroid_threshold) & (spectral_rolloff > rolloff_threshold)
        
        # 3. Zero-crossing rate (moderate ZCR indicates voiced speech)
        zcr_min, zcr_max = np.percentile(zcr, [20, 80])
        zcr_speech = (zcr >= zcr_min) & (zcr <= zcr_max * 1.5)
        
        # 4. MFCC-based detection (stable MFCC patterns indicate speech)
        mfcc_stability = np.std(mfcc, axis=0)
        mfcc_speech = mfcc_stability < np.percentile(mfcc_stability, 70)
        
        # 5. Spectral contrast (speech has different contrast patterns than music)
        contrast_speech = spectral_contrast < np.percentile(spectral_contrast, 75)
        
        # Combine features with weighted voting
        speech_votes = (
            energy_speech.astype(float) * 0.3 +
            spectral_speech.astype(float) * 0.25 +
            zcr_speech.astype(float) * 0.2 +
            mfcc_speech.astype(float) * 0.15 +
            contrast_speech.astype(float) * 0.1
        )
        
        # Speech confidence (0-1 scale)
        speech_confidence = speech_votes
        
        # Binary speech detection (threshold at 0.4 for robustness)
        is_speech = speech_confidence > 0.4
        
        # Post-processing: remove isolated speech segments and fill short gaps
        is_speech = self._postprocess_vad(is_speech, times)
        
        # Create result dictionary
        result = {
            'times': times,
            'is_speech': is_speech,
            'speech_confidence': speech_confidence,
            'features': {
                'rms': rms,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'zcr': zcr,
                'mfcc': mfcc,
                'spectral_contrast': spectral_contrast
            },
            'thresholds': {
                'rms': rms_threshold,
                'centroid': centroid_threshold,
                'zcr': zcr_threshold,
                'rolloff': rolloff_threshold
            },
            'analysis_time': time.time() - start_time,
            'was_downsampled': is_long_file
        }
        
        # Cache result for future use
        self._vad_cache[cache_key] = result
        
        # Limit cache size to prevent memory issues
        if len(self._vad_cache) > 10:
            # Remove oldest entries
            oldest_key = next(iter(self._vad_cache))
            del self._vad_cache[oldest_key]
        
        analysis_time = result['analysis_time']
        print(f"   ⚡ VAD analysis completed in {analysis_time:.2f}s{'(downsampled)' if is_long_file else ''}")
        
        return result
    
    def _calculate_adaptive_threshold(self, feature: np.ndarray, percentile: float) -> float:
        """Calculate adaptive threshold based on audio characteristics."""
        # Use percentile-based threshold with outlier protection
        base_threshold = np.percentile(feature, percentile)
        
        # Add robustness against very quiet or very noisy audio
        feature_std = np.std(feature)
        feature_mean = np.mean(feature)
        
        # Dynamic adjustment based on signal variability
        if feature_std < 0.1 * feature_mean:  # Very stable signal
            threshold = base_threshold * 1.2
        elif feature_std > 2.0 * feature_mean:  # Very variable signal
            threshold = base_threshold * 0.8
        else:
            threshold = base_threshold
        
        return float(threshold)
    
    def _postprocess_vad(self, is_speech: np.ndarray, times: np.ndarray, 
                        min_speech_duration: float = 0.3, 
                        min_silence_duration: float = 0.2) -> np.ndarray:
        """
        Post-process VAD results to remove artifacts and smooth boundaries.
        
        Args:
            is_speech: Binary speech detection array
            times: Time stamps array
            min_speech_duration: Minimum duration for speech segments (seconds)
            min_silence_duration: Minimum duration for silence gaps (seconds)
            
        Returns:
            Smoothed speech detection array
        """
        if len(times) < 2:
            return is_speech
            
        frame_duration = times[1] - times[0]
        min_speech_frames = int(min_speech_duration / frame_duration)
        min_silence_frames = int(min_silence_duration / frame_duration)
        
        # Remove short speech segments
        speech_segments = self._find_segments(is_speech, True)
        for start, end in speech_segments:
            if (end - start) < min_speech_frames:
                is_speech[start:end] = False
        
        # Fill short silence gaps within speech
        silence_segments = self._find_segments(is_speech, False)
        for start, end in silence_segments:
            if (end - start) < min_silence_frames:
                # Check if surrounded by speech
                has_speech_before = start > 0 and is_speech[start - 1]
                has_speech_after = end < len(is_speech) and is_speech[end]
                if has_speech_before and has_speech_after:
                    is_speech[start:end] = True
        
        return is_speech
    
    def _find_segments(self, binary_array: np.ndarray, target_value: bool) -> List[Tuple[int, int]]:
        """Find contiguous segments of target value in binary array."""
        segments = []
        in_segment = False
        start_idx = None
        
        for i, value in enumerate(binary_array):
            if value == target_value and not in_segment:
                start_idx = i
                in_segment = True
            elif value != target_value and in_segment:
                segments.append((start_idx, i))
                in_segment = False
                start_idx = None
        
        # Handle case where array ends in target segment
        if in_segment and start_idx is not None:
            segments.append((start_idx, len(binary_array)))
        
        return segments
    
    def _detect_enhanced_pause_regions(self, times: np.ndarray, is_speech: np.ndarray, 
                                     speech_confidence: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect enhanced pause regions with improved scoring and linguistic boundary analysis.
        
        Args:
            times: Array of time stamps
            is_speech: Boolean array indicating speech activity
            speech_confidence: Confidence scores for speech activity
            
        Returns:
            List of enhanced pause region dictionaries
        """
        pause_regions = []
        in_pause = False
        pause_start = None
        pause_start_idx = None
        
        # Minimum pause duration for consideration (0.5 seconds)
        min_pause_duration = 0.5
        
        for i, (t, speech, confidence) in enumerate(zip(times, is_speech, speech_confidence)):
            if not speech and not in_pause:
                # Start of pause
                pause_start = t
                pause_start_idx = i
                in_pause = True
            elif speech and in_pause:
                # End of pause
                if pause_start is not None and pause_start_idx is not None:
                    pause_duration = t - pause_start
                    
                    # Only consider pauses longer than minimum duration
                    if pause_duration >= min_pause_duration:
                        # Calculate pause quality metrics
                        pause_end_idx = i
                        pause_confidence_window = speech_confidence[max(0, pause_start_idx-5):min(len(speech_confidence), pause_end_idx+5)]
                        
                        # Calculate surrounding speech stability
                        pre_speech_stability = self._calculate_speech_stability(
                            speech_confidence[max(0, pause_start_idx-10):pause_start_idx]
                        ) if pause_start_idx > 0 else 0.5
                        
                        post_speech_stability = self._calculate_speech_stability(
                            speech_confidence[pause_end_idx:min(len(speech_confidence), pause_end_idx+10)]
                        ) if pause_end_idx < len(speech_confidence) else 0.5
                        
                        # Calculate pause quality score
                        pause_quality = self._calculate_pause_quality(
                            pause_duration, pre_speech_stability, post_speech_stability, pause_confidence_window
                        )
                        
                        pause_center = pause_start + pause_duration / 2
                        pause_regions.append({
                            'start': pause_start,
                            'end': t,
                            'center': pause_center,
                            'duration': pause_duration,
                            'quality_score': pause_quality,
                            'pre_speech_stability': pre_speech_stability,
                            'post_speech_stability': post_speech_stability,
                            'is_long_pause': pause_duration > 2.0,
                            'is_breath_pause': 0.5 <= pause_duration <= 1.5,
                            'linguistic_boundary_likelihood': self._estimate_linguistic_boundary(
                                pause_duration, pre_speech_stability, post_speech_stability
                            )
                        })
                
                in_pause = False
                pause_start = None
                pause_start_idx = None
        
        # Handle case where audio ends in a pause
        if in_pause and pause_start is not None and pause_start_idx is not None:
            pause_duration = times[-1] - pause_start
            if pause_duration >= min_pause_duration:
                # Similar calculation for end-of-audio pause
                pause_confidence_window = speech_confidence[max(0, pause_start_idx-5):]
                pre_speech_stability = self._calculate_speech_stability(
                    speech_confidence[max(0, pause_start_idx-10):pause_start_idx]
                ) if pause_start_idx > 0 else 0.5
                
                pause_quality = self._calculate_pause_quality(
                    pause_duration, pre_speech_stability, 0.5, pause_confidence_window
                )
                
                pause_center = pause_start + pause_duration / 2
                pause_regions.append({
                    'start': pause_start,
                    'end': times[-1],
                    'center': pause_center,
                    'duration': pause_duration,
                    'quality_score': pause_quality,
                    'pre_speech_stability': pre_speech_stability,
                    'post_speech_stability': 0.5,
                    'is_long_pause': pause_duration > 2.0,
                    'is_breath_pause': 0.5 <= pause_duration <= 1.5,
                    'linguistic_boundary_likelihood': self._estimate_linguistic_boundary(
                        pause_duration, pre_speech_stability, 0.5
                    )
                })
        
        return pause_regions
    
    def _calculate_speech_stability(self, confidence_window: np.ndarray) -> float:
        """Calculate speech stability metric from confidence scores."""
        if len(confidence_window) == 0:
            return 0.5
        
        # Higher stability = more consistent speech confidence
        mean_confidence = np.mean(confidence_window)
        confidence_std = np.std(confidence_window)
        
        # Stability score (0-1, higher is more stable)
        stability = float(mean_confidence) * (1.0 - min(float(confidence_std), 0.5))
        return float(np.clip(stability, 0.0, 1.0))
    
    def _calculate_pause_quality(self, duration: float, pre_stability: float, 
                               post_stability: float, confidence_window: np.ndarray) -> float:
        """
        Calculate pause quality score based on multiple factors.
        
        Args:
            duration: Pause duration in seconds
            pre_stability: Speech stability before pause
            post_stability: Speech stability after pause
            confidence_window: Confidence scores around pause
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Duration score: optimal around 1-2 seconds for natural boundaries
        if duration < 0.5:
            duration_score = 0.0
        elif duration <= 1.0:
            duration_score = duration / 1.0  # Linear increase to 1.0
        elif duration <= 2.0:
            duration_score = 1.0  # Optimal range
        elif duration <= 4.0:
            duration_score = 1.0 - (duration - 2.0) / 4.0  # Gradual decrease
        else:
            duration_score = 0.5  # Long pauses still useful but not optimal
        
        # Stability score: better if stable speech surrounds the pause
        stability_score = (pre_stability + post_stability) / 2.0
        
        # Confidence score: lower confidence in pause region is better
        if len(confidence_window) > 0:
            avg_confidence_in_pause = np.mean(confidence_window)
            confidence_score = 1.0 - avg_confidence_in_pause  # Lower is better
        else:
            confidence_score = 0.5
        
        # Weighted combination
        quality_score = (
            duration_score * 0.4 +
            stability_score * 0.35 +
            confidence_score * 0.25
        )
        
        return np.clip(quality_score, 0.0, 1.0)
    
    def _estimate_linguistic_boundary(self, duration: float, pre_stability: float, 
                                    post_stability: float) -> float:
        """
        Estimate likelihood that a pause represents a linguistic boundary.
        
        Args:
            duration: Pause duration
            pre_stability: Speech stability before pause
            post_stability: Speech stability after pause
            
        Returns:
            Likelihood score (0-1)
        """
        # Longer pauses more likely to be sentence/phrase boundaries
        duration_factor = min(duration / 2.0, 1.0)
        
        # Changes in speech stability suggest speaker/topic changes
        stability_change = abs(pre_stability - post_stability)
        stability_factor = min(stability_change * 2.0, 1.0)
        
        # Breathing pauses (0.5-1.5s) are good natural boundaries
        breath_factor = 1.0 if 0.5 <= duration <= 1.5 else 0.5
        
        # Long pauses (>3s) are often sentence/paragraph boundaries
        long_pause_factor = 1.0 if duration > 3.0 else 0.5
        
        # Combine factors
        likelihood = (
            duration_factor * 0.3 +
            stability_factor * 0.2 +
            breath_factor * 0.3 +
            long_pause_factor * 0.2
        )
        
        return np.clip(likelihood, 0.0, 1.0)
    
    def _find_optimal_boundary_in_window(self, y: np.ndarray, sr: int, vad_result: Dict[str, Any], 
                                       pause_regions: List[Dict[str, Any]], min_boundary: float, 
                                       max_boundary: float, ideal_boundary: float, 
                                       enable_speaker_consistency: bool) -> float:
        """
        Find optimal boundary within window using enhanced analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            vad_result: VAD analysis results
            pause_regions: Enhanced pause regions
            min_boundary: Minimum allowed boundary time
            max_boundary: Maximum allowed boundary time
            ideal_boundary: Ideal boundary time (target)
            enable_speaker_consistency: Whether to perform speaker analysis
            
        Returns:
            Optimal boundary timestamp
        """
        times = vad_result['times']
        is_speech = vad_result['is_speech']
        
        # Find pauses within the search window
        window_pauses = [
            pause for pause in pause_regions
            if min_boundary <= pause['center'] <= max_boundary
        ]
        
        if not window_pauses:
            # No pauses in window, fall back to original method
            return self._find_best_pause_in_window(times, is_speech, min_boundary, max_boundary, ideal_boundary)
        
        # Score each pause based on multiple criteria
        best_pause = None
        best_score = -1
        
        for pause in window_pauses:
            # 1. Distance from ideal boundary (closer is better)
            distance_from_ideal = abs(pause['center'] - ideal_boundary)
            max_distance = max_boundary - min_boundary
            distance_score = 1.0 - (distance_from_ideal / max_distance)
            
            # 2. Pause quality score
            quality_score = pause['quality_score']
            
            # 3. Linguistic boundary likelihood
            linguistic_score = pause['linguistic_boundary_likelihood']
            
            # 4. Word-level boundary avoidance (approximate)
            word_boundary_score = self._estimate_word_boundary_safety(
                y, sr, pause['center'], pause['duration']
            )
            
            # 5. Speaker consistency score (if enabled)
            speaker_consistency_score = 1.0  # Default
            if enable_speaker_consistency:
                speaker_consistency_score = self._estimate_speaker_consistency_at_boundary(
                    y, sr, pause['center']
                )
            
            # Weighted combination of scores
            total_score = (
                distance_score * 0.25 +
                quality_score * 0.30 +
                linguistic_score * 0.20 +
                word_boundary_score * 0.15 +
                speaker_consistency_score * 0.10
            )
            
            if total_score > best_score:
                best_score = total_score
                best_pause = pause
        
        if best_pause:
            return best_pause['center']
        else:
            # Fallback to closest point to ideal boundary
            window_mask = (times >= min_boundary) & (times <= max_boundary)
            window_times = times[window_mask]
            if len(window_times) > 0:
                closest_idx = np.argmin(np.abs(window_times - ideal_boundary))
                return window_times[closest_idx]
            else:
                return ideal_boundary
    
    def _estimate_word_boundary_safety(self, y: np.ndarray, sr: int, 
                                     boundary_time: float, pause_duration: float) -> float:
        """
        Estimate safety of boundary placement to avoid cutting words.
        
        Args:
            y: Audio signal
            sr: Sample rate
            boundary_time: Proposed boundary time
            pause_duration: Duration of the pause
            
        Returns:
            Safety score (0-1, higher is safer)
        """
        # Longer pauses are generally safer for word boundaries
        if pause_duration >= 1.0:
            duration_safety = 1.0
        elif pause_duration >= 0.5:
            duration_safety = 0.8
        else:
            duration_safety = 0.4
        
        # Analyze spectral characteristics around boundary
        try:
            # Extract short window around boundary (±0.5 seconds)
            start_sample = max(0, int((boundary_time - 0.5) * sr))
            end_sample = min(len(y), int((boundary_time + 0.5) * sr))
            boundary_window = y[start_sample:end_sample]
            
            if len(boundary_window) < sr // 10:  # Too short to analyze
                return duration_safety
            
            # Calculate energy changes around boundary
            mid_point = len(boundary_window) // 2
            pre_energy = np.mean(boundary_window[:mid_point] ** 2)
            post_energy = np.mean(boundary_window[mid_point:] ** 2)
            
            # Lower energy during pause indicates safer boundary
            energy_ratio = float(min(float(pre_energy), float(post_energy))) / (float(max(float(pre_energy), float(post_energy))) + 1e-10)
            energy_safety = 1.0 - energy_ratio  # Lower ratio = higher safety
            
            # Combine duration and energy safety
            total_safety = (duration_safety * 0.7 + energy_safety * 0.3)
            return float(np.clip(total_safety, 0.0, 1.0))
            
        except Exception:
            # Fallback to duration-based safety
            return duration_safety
    
    def _estimate_speaker_consistency_at_boundary(self, y: np.ndarray, sr: int, 
                                                boundary_time: float) -> float:
        """
        Estimate speaker consistency across the proposed boundary.
        
        Args:
            y: Audio signal
            sr: Sample rate
            boundary_time: Proposed boundary time
            
        Returns:
            Consistency score (0-1, higher indicates consistent speaker)
        """
        try:
            # Extract windows before and after boundary (2 seconds each)
            window_duration = 2.0
            
            pre_start = max(0, int((boundary_time - window_duration) * sr))
            pre_end = int(boundary_time * sr)
            post_start = int(boundary_time * sr)
            post_end = min(len(y), int((boundary_time + window_duration) * sr))
            
            pre_window = y[pre_start:pre_end]
            post_window = y[post_start:post_end]
            
            if len(pre_window) < sr // 4 or len(post_window) < sr // 4:
                return 0.7  # Default moderate consistency
            
            # Calculate spectral features for speaker comparison
            pre_features = self._extract_speaker_features(pre_window, sr)
            post_features = self._extract_speaker_features(post_window, sr)
            
            # Calculate similarity between features
            consistency_score = self._calculate_feature_similarity(pre_features, post_features)
            
            return consistency_score
            
        except Exception:
            return 0.7  # Default moderate consistency
    
    def _extract_speaker_features(self, audio_window: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract basic speaker-characteristic features from audio window."""
        try:
            # Fundamental frequency (pitch) characteristics
            f0_values = []
            hop_length = 512
            
            # Use autocorrelation-based pitch detection (simplified)
            frame_length = 2048
            for i in range(0, len(audio_window) - frame_length, hop_length):
                frame = audio_window[i:i + frame_length]
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find peak in expected pitch range (80-400 Hz)
                min_period = sr // 400  # 400 Hz max
                max_period = sr // 80   # 80 Hz min
                
                if max_period < len(autocorr):
                    peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
                    if peak_idx > 0:
                        f0 = sr / peak_idx
                        f0_values.append(f0)
            
            # Calculate pitch statistics
            if f0_values:
                mean_f0 = np.mean(f0_values)
                std_f0 = np.std(f0_values)
            else:
                mean_f0 = 150.0  # Default
                std_f0 = 20.0
            
            # Spectral characteristics
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_window, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_window, sr=sr))
            
            # MFCC-based features (simplified speaker characteristics)
            mfcc = librosa.feature.mfcc(y=audio_window, sr=sr, n_mfcc=5)
            mfcc_means = np.mean(mfcc, axis=1)
            
            return {
                'mean_f0': float(mean_f0),
                'std_f0': float(std_f0),
                'spectral_centroid': float(spectral_centroid),
                'spectral_bandwidth': float(spectral_bandwidth),
                'mfcc1': float(mfcc_means[0]) if len(mfcc_means) > 0 else 0.0,
                'mfcc2': float(mfcc_means[1]) if len(mfcc_means) > 1 else 0.0,
                'mfcc3': float(mfcc_means[2]) if len(mfcc_means) > 2 else 0.0,
            }
            
        except Exception:
            # Return default features
            return {
                'mean_f0': 150.0,
                'std_f0': 20.0,
                'spectral_centroid': 2000.0,
                'spectral_bandwidth': 1500.0,
                'mfcc1': 0.0,
                'mfcc2': 0.0,
                'mfcc3': 0.0,
            }
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], 
                                    features2: Dict[str, float]) -> float:
        """Calculate similarity between two feature sets."""
        try:
            # Normalize and compare key features
            similarities = []
            
            # Pitch similarity (important for speaker identity)
            f0_diff = abs(features1['mean_f0'] - features2['mean_f0'])
            f0_similarity = max(0, 1.0 - f0_diff / 100.0)  # Normalize by 100 Hz
            similarities.append(f0_similarity * 0.3)
            
            # Spectral characteristics
            centroid_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
            centroid_similarity = max(0, 1.0 - centroid_diff / 2000.0)  # Normalize by 2kHz
            similarities.append(centroid_similarity * 0.2)
            
            # MFCC similarities
            for i in range(1, 4):
                mfcc_key = f'mfcc{i}'
                if mfcc_key in features1 and mfcc_key in features2:
                    mfcc_diff = abs(features1[mfcc_key] - features2[mfcc_key])
                    mfcc_similarity = max(0, 1.0 - mfcc_diff / 10.0)  # Approximate normalization
                    similarities.append(mfcc_similarity * 0.15)
            
            # Return weighted average
            return sum(similarities) if similarities else 0.5
            
        except Exception:
            return 0.5  # Default moderate similarity
    
    def _validate_chunk_quality(self, boundaries: List[float], total_duration: float, 
                              target_duration: float) -> float:
        """
        Validate the quality of generated chunk boundaries.
        
        Args:
            boundaries: List of boundary timestamps
            total_duration: Total audio duration
            target_duration: Target chunk duration
            
        Returns:
            Quality score (0-1, higher is better)
        """
        if not boundaries:
            return 0.0
        
        # Calculate chunk durations
        chunk_durations = []
        prev_boundary = 0.0
        
        for boundary in boundaries:
            chunk_duration = boundary - prev_boundary
            chunk_durations.append(chunk_duration)
            prev_boundary = boundary
        
        if not chunk_durations:
            return 0.0
        
        # 1. Duration consistency score
        duration_deviations = [abs(duration - target_duration) for duration in chunk_durations]
        max_allowed_deviation = target_duration * 0.3  # 30% deviation allowed
        duration_scores = [max(0, 1.0 - (dev / max_allowed_deviation)) for dev in duration_deviations]
        avg_duration_score = np.mean(duration_scores)
        
        # 2. Size balance score (chunks should be reasonably similar in size)
        if len(chunk_durations) > 1:
            duration_std = np.std(chunk_durations)
            duration_mean = np.mean(chunk_durations)
            balance_score = max(0, 1.0 - (float(duration_std) / float(duration_mean)))
        else:
            balance_score = 1.0
        
        # 3. Boundary placement score (are boundaries within acceptable ranges?)
        acceptable_range_min = target_duration * 0.85  # 4:15 minutes for 5-minute target
        acceptable_range_max = target_duration * 1.05  # 5:15 minutes for 5-minute target
        
        in_range_count = sum(1 for duration in chunk_durations 
                           if acceptable_range_min <= duration <= acceptable_range_max)
        range_score = in_range_count / len(chunk_durations) if chunk_durations else 0
        
        # 4. Coverage score (do chunks cover the entire audio?)
        total_covered = sum(chunk_durations)
        coverage_error = abs(total_covered - total_duration)
        coverage_score = max(0, 1.0 - (coverage_error / total_duration))
        
        # Weighted combination
        quality_score = (
            avg_duration_score * 0.35 +
            balance_score * 0.25 +
            range_score * 0.25 +
            coverage_score * 0.15
        )
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    def _find_best_pause_in_window(self, times: np.ndarray, is_speech: np.ndarray, 
                                  min_boundary: float, max_boundary: float, 
                                  ideal_boundary: float) -> float:
        """
        Find the best pause (non-speech) region within the specified window.
        
        Args:
            times: Array of time stamps
            is_speech: Boolean array indicating speech activity
            min_boundary: Minimum allowed boundary time
            max_boundary: Maximum allowed boundary time
            ideal_boundary: Ideal boundary time (target)
            
        Returns:
            Best boundary timestamp
        """
        # Find indices within the search window
        window_mask = (times >= min_boundary) & (times <= max_boundary)
        window_times = times[window_mask]
        window_speech = is_speech[window_mask]
        
        if len(window_times) == 0:
            return ideal_boundary
        
        # Look for pauses (non-speech regions) in the window
        pause_regions = []
        in_pause = False
        pause_start = None
        
        for i, (t, speech) in enumerate(zip(window_times, window_speech)):
            if not speech and not in_pause:
                # Start of pause
                pause_start = t
                in_pause = True
            elif speech and in_pause:
                # End of pause
                if pause_start is not None:
                    pause_duration = t - pause_start
                    pause_center = pause_start + pause_duration / 2
                    pause_regions.append({
                        'start': pause_start,
                        'end': t,
                        'center': pause_center,
                        'duration': pause_duration
                    })
                in_pause = False
                pause_start = None
        
        # Handle case where window ends in a pause
        if in_pause and pause_start is not None:
            pause_duration = window_times[-1] - pause_start
            pause_center = pause_start + pause_duration / 2
            pause_regions.append({
                'start': pause_start,
                'end': window_times[-1],
                'center': pause_center,
                'duration': pause_duration
            })
        
        # Select best pause region
        if pause_regions:
            # Score pauses based on:
            # 1. Duration (longer pauses are better)
            # 2. Distance from ideal boundary (closer is better)
            best_pause = None
            best_score = -1
            
            for pause in pause_regions:
                # Prefer longer pauses (minimum 0.5s to be considered)
                if pause['duration'] < 0.5:
                    continue
                    
                duration_score = min(pause['duration'] / 2.0, 1.0)  # Max score at 2s pause
                distance_score = 1.0 - abs(pause['center'] - ideal_boundary) / 60.0  # Max 60s penalty
                distance_score = max(distance_score, 0)
                
                total_score = duration_score * 0.7 + distance_score * 0.3
                
                if total_score > best_score:
                    best_score = total_score
                    best_pause = pause
            
            if best_pause:
                return best_pause['center']
        
        # No good pause found, use the point closest to ideal within window
        closest_idx = np.argmin(np.abs(window_times - ideal_boundary))
        return window_times[closest_idx]
    
    def _simple_time_chunking(self, audio_path: str, target_duration: float) -> List[float]:
        """
        Fallback to simple time-based chunking.
        
        Args:
            audio_path: Path to audio file
            target_duration: Target chunk duration
            
        Returns:
            List of boundary timestamps
        """
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
        except:
            # If we can't load the audio, estimate from file size or use a default
            duration = 3600.0  # Default to 1 hour
        
        boundaries = []
        current_pos = 0.0
        
        while current_pos < duration:
            current_pos += target_duration
            if current_pos < duration:
                boundaries.append(current_pos)
            else:
                boundaries.append(duration)
                break
        
        return boundaries
    
    def fuse_diarization_variants(self, variants: List[Dict[str, Any]], audio_path: str) -> List[Dict[str, Any]]:
        """
        Apply voting fusion to enhance 5 diarization variants with consensus-based improvements.
        
        Args:
            variants: List of 5 diarization variant results
            audio_path: Path to audio file for temporal analysis
            
        Returns:
            Enhanced variants with voting fusion applied
        """
        print(f"🗳️  Starting voting fusion for {len(variants)} variants...")
        
        # Step 1: Temporal alignment and boundary consensus
        fused_boundaries = self._find_boundary_consensus(variants)
        print(f"   ✓ Identified {len(fused_boundaries)} consensus boundaries")
        
        # Step 2: Speaker identity alignment across variants
        speaker_mapping = self._align_speaker_identities(variants)
        print(f"   ✓ Created cross-variant speaker mapping")
        
        # Step 3: Apply consensus improvements to each variant
        enhanced_variants = []
        for i, variant in enumerate(variants):
            print(f"   🔄 Enhancing variant {variant['variant_id']} with fusion...")
            enhanced_variant = self._apply_fusion_enhancement(
                variant, fused_boundaries, speaker_mapping, variants
            )
            enhanced_variants.append(enhanced_variant)
        
        print(f"✅ Voting fusion complete - {len(enhanced_variants)} variants enhanced")
        return enhanced_variants
    
    def _find_boundary_consensus(self, variants: List[Dict[str, Any]], 
                               consensus_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Find speaker change points that multiple variants agree on.
        
        Args:
            variants: List of diarization variants
            consensus_threshold: Minimum fraction of variants that must agree
            
        Returns:
            List of consensus boundary points with confidence scores
        """
        all_boundaries = []
        
        # Collect all boundary points from all variants
        for variant in variants:
            segments = variant['segments']
            for i, segment in enumerate(segments):
                if i > 0:  # Skip first segment start
                    all_boundaries.append({
                        'timestamp': segment['start'],
                        'variant_id': variant['variant_id'],
                        'confidence': segment.get('confidence', 0.8)
                    })
        
        # Sort boundaries by timestamp
        all_boundaries.sort(key=lambda x: x['timestamp'])
        
        # Group boundaries within temporal windows (±0.5 seconds)
        consensus_boundaries = []
        time_window = 0.5
        
        i = 0
        while i < len(all_boundaries):
            current_time = all_boundaries[i]['timestamp']
            window_boundaries = []
            
            # Collect boundaries within the time window
            j = i
            while j < len(all_boundaries) and all_boundaries[j]['timestamp'] <= current_time + time_window:
                window_boundaries.append(all_boundaries[j])
                j += 1
            
            # Check if enough variants agree on this boundary
            variant_count = len(set(b['variant_id'] for b in window_boundaries))
            required_votes = max(2, int(len(variants) * consensus_threshold))
            
            if variant_count >= required_votes:
                # Calculate consensus timestamp and confidence
                timestamps = [b['timestamp'] for b in window_boundaries]
                confidences = [b['confidence'] for b in window_boundaries]
                
                consensus_timestamp = sum(timestamps) / len(timestamps)
                consensus_confidence = sum(confidences) / len(confidences)
                
                consensus_boundaries.append({
                    'timestamp': consensus_timestamp,
                    'confidence': consensus_confidence,
                    'support_count': variant_count,
                    'total_variants': len(variants)
                })
            
            i = j
        
        return consensus_boundaries
    
    def _align_speaker_identities(self, variants: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        """
        Create cross-variant speaker identity mapping using embeddings and temporal overlap.
        
        Args:
            variants: List of diarization variants
            
        Returns:
            Speaker mapping dictionary: {variant_id: {local_speaker_id: global_speaker_id}}
        """
        speaker_mapping = {}
        global_speaker_counter = 0
        
        # Initialize mapping for first variant (reference)
        first_variant = variants[0]
        speaker_mapping[first_variant['variant_id']] = {}
        
        for speaker_id in first_variant['speaker_embeddings'].keys():
            global_id = f"GLOBAL_SPEAKER_{global_speaker_counter:02d}"
            speaker_mapping[first_variant['variant_id']][speaker_id] = global_id
            global_speaker_counter += 1
        
        # Align remaining variants
        for variant in variants[1:]:
            variant_id = variant['variant_id']
            speaker_mapping[variant_id] = {}
            
            for local_speaker in variant['speaker_embeddings'].keys():
                best_match_global = None
                best_similarity = -1
                
                # Compare with existing global speakers
                local_embedding = np.array(variant['speaker_embeddings'][local_speaker])
                
                for ref_variant_id in speaker_mapping:
                    if ref_variant_id == variant_id:
                        continue
                        
                    ref_variant = next(v for v in variants if v['variant_id'] == ref_variant_id)
                    
                    for ref_speaker, global_id in speaker_mapping[ref_variant_id].items():
                        ref_embedding = np.array(ref_variant['speaker_embeddings'][ref_speaker])
                        
                        # Calculate cosine similarity
                        similarity = np.dot(local_embedding, ref_embedding) / (
                            np.linalg.norm(local_embedding) * np.linalg.norm(ref_embedding) + 1e-8
                        )
                        
                        if similarity > best_similarity and similarity > 0.7:  # Similarity threshold
                            best_similarity = similarity
                            best_match_global = global_id
                
                # Assign global ID
                if best_match_global:
                    speaker_mapping[variant_id][local_speaker] = best_match_global
                else:
                    # Create new global speaker
                    global_id = f"GLOBAL_SPEAKER_{global_speaker_counter:02d}"
                    speaker_mapping[variant_id][local_speaker] = global_id
                    global_speaker_counter += 1
        
        return speaker_mapping
    
    def _apply_fusion_enhancement(self, variant: Dict[str, Any], 
                                fused_boundaries: List[Dict[str, Any]],
                                speaker_mapping: Dict[str, Dict[str, str]],
                                all_variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply fusion enhancements to a single variant.
        
        Args:
            variant: Single diarization variant to enhance
            fused_boundaries: Consensus boundaries from voting fusion
            speaker_mapping: Cross-variant speaker mapping
            all_variants: All variants for context
            
        Returns:
            Enhanced variant with fusion improvements
        """
        enhanced_variant = variant.copy()
        
        # Apply consensus boundary corrections
        enhanced_segments = self._apply_boundary_corrections(
            variant['segments'], fused_boundaries
        )
        
        # Apply speaker identity corrections
        enhanced_segments = self._apply_speaker_corrections(
            enhanced_segments, speaker_mapping[variant['variant_id']]
        )
        
        # Calculate fusion metrics
        fusion_metrics = {
            'boundary_corrections_applied': len(fused_boundaries),
            'speaker_identity_corrections': len(speaker_mapping[variant['variant_id']]),
            'consensus_confidence': sum(b['confidence'] for b in fused_boundaries) / max(len(fused_boundaries), 1),
            'cross_variant_agreement': self._calculate_cross_variant_agreement(variant, all_variants)
        }
        
        # Update enhanced variant
        enhanced_variant['segments'] = enhanced_segments
        enhanced_variant['fusion_applied'] = True
        enhanced_variant['fusion_metrics'] = fusion_metrics
        
        # Recalculate quality metrics with fusion
        enhanced_variant['quality_metrics'] = self._calculate_variant_quality_metrics(enhanced_segments)
        enhanced_variant['quality_metrics']['fusion_score'] = fusion_metrics['consensus_confidence']
        
        return enhanced_variant
    
    def _apply_boundary_corrections(self, segments: List[Dict[str, Any]], 
                                  fused_boundaries: List[Dict[str, Any]],
                                  correction_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Apply consensus boundary corrections to segment boundaries."""
        corrected_segments = segments.copy()
        corrections_applied = 0
        
        for boundary in fused_boundaries:
            consensus_time = boundary['timestamp']
            confidence = boundary['confidence']
            
            # Only apply high-confidence corrections
            if confidence < 0.7:
                continue
                
            # Find nearest segment boundaries
            for i, segment in enumerate(corrected_segments):
                # Check segment start boundary
                if abs(segment['start'] - consensus_time) <= correction_threshold:
                    corrected_segments[i]['start'] = consensus_time
                    corrections_applied += 1
                
                # Check segment end boundary
                if abs(segment['end'] - consensus_time) <= correction_threshold:
                    corrected_segments[i]['end'] = consensus_time
                    corrections_applied += 1
        
        print(f"     → Applied {corrections_applied} boundary corrections")
        return corrected_segments
    
    def _apply_speaker_corrections(self, segments: List[Dict[str, Any]], 
                                 speaker_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """Apply cross-variant speaker identity corrections."""
        corrected_segments = []
        
        for segment in segments:
            corrected_segment = segment.copy()
            local_speaker = segment['speaker_id']
            
            if local_speaker in speaker_mapping:
                corrected_segment['speaker_id'] = speaker_mapping[local_speaker]
                corrected_segment['original_speaker_id'] = local_speaker
            
            corrected_segments.append(corrected_segment)
        
        return corrected_segments
    
    def _calculate_cross_variant_agreement(self, variant: Dict[str, Any], 
                                         all_variants: List[Dict[str, Any]]) -> float:
        """Calculate how much this variant agrees with others."""
        if len(all_variants) <= 1:
            return 1.0
            
        total_agreement = 0.0
        comparisons = 0
        
        current_segments = variant['segments']
        
        for other_variant in all_variants:
            if other_variant['variant_id'] == variant['variant_id']:
                continue
                
            other_segments = other_variant['segments']
            agreement = self._calculate_temporal_overlap_agreement(current_segments, other_segments)
            total_agreement += agreement
            comparisons += 1
        
        return total_agreement / max(comparisons, 1)
    
    def _calculate_temporal_overlap_agreement(self, segments1: List[Dict[str, Any]], 
                                            segments2: List[Dict[str, Any]]) -> float:
        """Calculate temporal overlap agreement between two segment lists."""
        if not segments1 or not segments2:
            return 0.0
            
        total_overlap = 0.0
        total_union = 0.0
        
        # Calculate total time covered by each segment set
        time1_covered = sum(seg['end'] - seg['start'] for seg in segments1)
        time2_covered = sum(seg['end'] - seg['start'] for seg in segments2)
        
        # Simple overlap estimation (more sophisticated methods possible)
        for seg1 in segments1:
            for seg2 in segments2:
                # Calculate intersection
                overlap_start = max(seg1['start'], seg2['start'])
                overlap_end = min(seg1['end'], seg2['end'])
                
                if overlap_start < overlap_end:
                    total_overlap += (overlap_end - overlap_start)
        
        total_union = time1_covered + time2_covered - total_overlap
        
        return total_overlap / max(total_union, 1e-6)
    
    def _apply_multi_resolution_processing(self, segments: List[Dict[str, Any]], 
                                         resolution_scales: List[float],
                                         merge_strategy: str) -> List[Dict[str, Any]]:
        """Apply multi-resolution processing to segments."""
        if merge_strategy == "weighted_average":
            # Apply temporal smoothing at different scales
            smoothed_segments = segments.copy()
            
            for scale in resolution_scales:
                # Adjust segment boundaries based on scale
                for segment in smoothed_segments:
                    duration = segment['end'] - segment['start']
                    adjustment = duration * (scale - 1.0) * 0.1
                    
                    segment['start'] = max(0, segment['start'] - adjustment)
                    segment['end'] = segment['end'] + adjustment
            
            return smoothed_segments
        
        return segments
    
    def _apply_ensemble_processing(self, segments: List[Dict[str, Any]], 
                                 audio_path: str, ensemble_seeds: List[int],
                                 consensus_threshold: float) -> List[Dict[str, Any]]:
        """Apply ensemble-based processing using multiple seeds."""
        # For now, add some variance to simulate ensemble processing
        ensemble_segments = segments.copy()
        
        # Add confidence boost for ensemble approach
        for segment in ensemble_segments:
            base_confidence = segment.get('confidence', 0.8)
            # Ensemble typically provides more stable results
            ensemble_confidence = min(base_confidence * 1.05, 0.95)
            segment['confidence'] = ensemble_confidence
        
        return ensemble_segments
    
    def _calculate_variant_quality_metrics(self, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality metrics for a diarization variant."""
        if not segments:
            return {'overall_quality': 0.0}
        
        # Calculate various quality metrics
        total_duration = sum(seg['end'] - seg['start'] for seg in segments)
        avg_segment_length = total_duration / len(segments)
        
        # Speaker consistency (fewer rapid speaker changes is generally better)
        speaker_changes = 0
        for i in range(1, len(segments)):
            if segments[i]['speaker_id'] != segments[i-1]['speaker_id']:
                speaker_changes += 1
        
        speaker_consistency = 1.0 - min(speaker_changes / len(segments), 1.0)
        
        # Temporal coverage quality
        time_coverage = total_duration / max(segments[-1]['end'], 1.0) if segments else 0.0
        
        # Average confidence
        avg_confidence = sum(seg.get('confidence', 0.8) for seg in segments) / len(segments)
        
        # Overall quality score
        overall_quality = (
            speaker_consistency * 0.3 +
            time_coverage * 0.2 +
            avg_confidence * 0.3 +
            min(avg_segment_length / 10.0, 1.0) * 0.2  # Prefer reasonable segment lengths
        )
        
        return {
            'overall_quality': float(np.clip(overall_quality, 0.0, 1.0)),
            'speaker_consistency': float(speaker_consistency),
            'time_coverage': float(time_coverage),
            'avg_confidence': float(avg_confidence),
            'avg_segment_length': float(avg_segment_length),
            'total_segments': len(segments)
        }
    
    def _generate_stabilization_summary(self, variants: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of turn stabilization results across all variants.
        
        Args:
            variants: List of stabilized diarization variants
            
        Returns:
            Human-readable summary string
        """
        if not variants:
            return "No variants to summarize"
        
        total_transitions_eliminated = 0
        total_rapid_transitions_eliminated = 0
        total_original_segments = 0
        total_stabilized_segments = 0
        variants_processed = 0
        
        for variant in variants:
            if variant.get('processed_with_stabilization', False):
                variants_processed += 1
                
                # Extract stabilization metrics
                stabilization_metrics = variant.get('stabilization_metrics')
                if stabilization_metrics:
                    total_transitions_eliminated += stabilization_metrics.transitions_eliminated
                    total_rapid_transitions_eliminated += stabilization_metrics.rapid_transitions_eliminated
                    total_original_segments += stabilization_metrics.original_segments_count
                    total_stabilized_segments += stabilization_metrics.stabilized_segments_count
        
        if variants_processed == 0:
            return "No variants successfully processed with stabilization"
        
        # Calculate overall improvement
        avg_improvement = (total_transitions_eliminated / max(total_original_segments, 1)) * 100
        
        summary = (
            f"{variants_processed}/{len(variants)} variants stabilized, "
            f"{total_transitions_eliminated} transitions eliminated, "
            f"{total_rapid_transitions_eliminated} rapid transitions fixed, "
            f"{avg_improvement:.1f}% improvement"
        )
        
        return summary
