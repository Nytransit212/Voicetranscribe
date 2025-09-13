import numpy as np
try:
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    # Mock classes for demonstration when pyannote is not available
    torch = None
    PYANNOTE_AVAILABLE = False
    
    class Annotation:
        def __init__(self):
            self._tracks = []
        
        def __setitem__(self, segment, speaker):
            self._tracks.append((segment, speaker))
        
        def itertracks(self, yield_label=True):
            for segment, speaker in self._tracks:
                yield segment, None, speaker
    
    class Segment:
        def __init__(self, start, end):
            self.start = start
            self.end = end
import tempfile
import os
import json
from typing import List, Dict, Any, Optional
import random

class DiarizationEngine:
    """Handles speaker diarization with multiple variants for ensemble processing"""
    
    def __init__(self, expected_speakers: int = 10, noise_level: str = 'medium'):
        self.expected_speakers = expected_speakers
        self.noise_level = noise_level
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the pyannote speaker diarization pipeline"""
        try:
            # Use pyannote speaker diarization pipeline
            # Note: This requires a Hugging Face token in practice
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN", "dummy_token")
            )
        except Exception as e:
            print(f"Warning: Could not load pyannote pipeline: {e}")
            # Fallback to mock pipeline for demonstration
            self.pipeline = self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for demonstration purposes"""
        class MockPipeline:
            def __call__(self, audio_file, min_speakers=None, max_speakers=None):
                # Generate mock diarization result
                return self._generate_mock_diarization(audio_file, min_speakers, max_speakers)
            
            def _generate_mock_diarization(self, audio_file, min_speakers, max_speakers):
                # Create a simple mock annotation
                annotation = Annotation()
                
                # Estimate duration (rough approximation for demo)
                duration = 300.0  # 5 minutes for demo
                
                num_speakers = min_speakers or 3
                segment_length = 15.0  # 15 second segments
                
                current_time = 0.0
                speaker_id = 0
                
                while current_time < duration:
                    end_time = min(current_time + segment_length, duration)
                    
                    # Add some randomness to speaker changes
                    if random.random() < 0.3:  # 30% chance to switch speaker
                        speaker_id = (speaker_id + 1) % num_speakers
                    
                    annotation[Segment(current_time, end_time)] = f"SPEAKER_{speaker_id:02d}"
                    current_time = end_time
                
                return annotation
        
        return MockPipeline()
    
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
        if torch is not None:
            torch.manual_seed(seed)
        
        try:
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
