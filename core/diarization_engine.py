import numpy as np
import os
import tempfile
import json
from typing import List, Dict, Any, Optional, Union, Type
import random
import librosa

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
Annotation: Type[BaseAnnotation] = BaseAnnotation
Segment: Type[BaseSegment] = BaseSegment
Pipeline: Type[BasePipeline] = BasePipeline

try:
    import torch as torch_module
    from pyannote.audio import Pipeline as RealPipeline
    from pyannote.core import Annotation as RealAnnotation, Segment as RealSegment
    
    # Use real classes if available
    Annotation = RealAnnotation
    Segment = RealSegment
    Pipeline = RealPipeline
    PYANNOTE_AVAILABLE = True
    print("✓ pyannote.audio successfully imported")
except ImportError as e:
    print(f"⚠ pyannote.audio not available ({e}), using mock implementation")
    # Base classes are already assigned above

class DiarizationEngine:
    """Handles speaker diarization with multiple variants for ensemble processing"""
    
    def __init__(self, expected_speakers: int = 10, noise_level: str = 'medium'):
        self.expected_speakers = expected_speakers
        self.noise_level = noise_level
        self.pipeline = None
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
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            print("✓ Real pyannote pipeline loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load pyannote pipeline: {e}")
            print("Falling back to mock pipeline")
            self.pipeline = self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline with intelligent chunking and pause detection"""
        class MockPipeline:
            def __init__(self, parent_engine):
                self.parent = parent_engine
            
            def __call__(self, audio_file, min_speakers=None, max_speakers=None):
                # Generate mock diarization result with intelligent chunking
                return self._generate_intelligent_mock_diarization(audio_file, min_speakers, max_speakers)
            
            def _generate_intelligent_mock_diarization(self, audio_file, min_speakers, max_speakers):
                """Generate mock diarization with pause detection and flexible chunking"""
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
                """Create segments with simulated pause detection and VAD"""
                segments = []
                current_time = 0.0
                speaker_id = 0
                
                # Simulate natural speech patterns
                base_segment_length = 8.0  # Base segment length
                pause_probability = 0.15  # Probability of pause
                speaker_change_prob = 0.25  # Probability of speaker change
                
                while current_time < duration:
                    # Variable segment length with pauses
                    segment_length = base_segment_length + random.uniform(-3.0, 7.0)
                    
                    # Add occasional pauses
                    if random.random() < pause_probability:
                        pause_duration = random.uniform(0.5, 2.0)
                        current_time += pause_duration
                    
                    # Calculate end time
                    end_time = min(current_time + segment_length, duration)
                    
                    if end_time > current_time:
                        segments.append({
                            'start': current_time,
                            'end': end_time,
                            'speaker_id': f"SPEAKER_{speaker_id:02d}"
                        })
                    
                    # Occasionally change speakers
                    if random.random() < speaker_change_prob:
                        speaker_id = (speaker_id + 1) % num_speakers
                    
                    current_time = end_time
                
                return segments
        
        return MockPipeline(self)
    
    def create_diarization_variants(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Create 3 diarization variants with different parameters.
        
        Args:
            audio_path: Path to cleaned audio file
            
        Returns:
            List of 3 diarization variant results
        """
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
            seed=42
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
            seed=123
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
            seed=456
        ))
        
        return variants
    
    def _run_diarization_variant(self, audio_path: str, variant_id: int, 
                               min_speakers: int, max_speakers: int,
                               clustering_threshold: float, vad_onset: float,
                               vad_offset: float, seed: int) -> Dict[str, Any]:
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
                max_speakers=max_speakers
            )
            
            # Convert to our format
            segments = []
            speaker_embeddings = {}
            
            for segment, _, speaker in diarization.itertracks(yield_label=True):
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
            
            return {
                'variant_id': variant_id,
                'segments': segments,
                'speaker_embeddings': speaker_embeddings,
                'rttm_data': rttm_data,
                'parameters': {
                    'min_speakers': min_speakers,
                    'max_speakers': max_speakers,
                    'clustering_threshold': clustering_threshold,
                    'vad_onset': vad_onset,
                    'vad_offset': vad_offset,
                    'seed': seed
                },
                'num_speakers': len(set(seg['speaker_id'] for seg in segments)),
                'total_speech_time': sum(seg['end'] - seg['start'] for seg in segments),
                'num_segments': len(segments)
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
    
    def find_optimal_chunk_boundaries(self, audio_path: str, target_duration: float = 300.0) -> List[float]:
        """
        Find optimal chunk boundaries using Voice Activity Detection and pause detection.
        Targets 5 minutes (300s) but flexible between 4:15-5:15 (255-315s) to avoid cutting words.
        
        Args:
            audio_path: Path to audio file
            target_duration: Target chunk duration in seconds (default 5 minutes)
            
        Returns:
            List of timestamps for chunk boundaries
        """
        try:
            # Load audio for analysis
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            print(f"Analyzing audio for intelligent chunking: {duration:.1f}s total")
            
            # If audio is shorter than target, return no chunks
            if duration <= target_duration:
                return [duration]
            
            # Detect voice activity using RMS energy and spectral centroid
            hop_length = 512
            frame_length = 2048
            
            # Calculate RMS energy (voice activity indicator)
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Calculate spectral centroid (helps identify speech vs silence/noise)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            
            # Convert frame indices to time
            times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
            
            # Determine voice activity threshold (adaptive based on audio characteristics)
            rms_threshold = np.percentile(rms, 20)  # 20th percentile as silence threshold
            centroid_threshold = np.percentile(spectral_centroid, 30)
            
            # Identify potential pause regions (low RMS and low spectral centroid)
            is_speech = (rms > rms_threshold) & (spectral_centroid > centroid_threshold)
            
            # Find boundaries
            boundaries = []
            current_pos = 0.0
            
            while current_pos < duration:
                # Define search window: 4:15 to 5:15 minutes from current position
                min_boundary = current_pos + 255.0  # 4:15
                max_boundary = min(current_pos + 315.0, duration)  # 5:15 or end of file
                ideal_boundary = current_pos + target_duration  # 5:00
                
                if max_boundary >= duration:
                    # Last chunk - just use the end
                    boundaries.append(duration)
                    break
                
                # Find the best pause within the search window
                best_boundary = self._find_best_pause_in_window(
                    times, is_speech, min_boundary, max_boundary, ideal_boundary
                )
                
                boundaries.append(best_boundary)
                current_pos = best_boundary
            
            print(f"Found {len(boundaries)} intelligent chunk boundaries at: {[f'{b:.1f}s' for b in boundaries]}")
            return boundaries
            
        except Exception as e:
            print(f"Warning: Could not perform intelligent chunking: {e}")
            # Fallback to simple time-based chunking
            return self._simple_time_chunking(audio_path, target_duration)
    
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
