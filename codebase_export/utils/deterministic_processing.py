"""
Deterministic Processing System for Ensemble Transcription
U7 Upgrade: Ensures reproducible results across runs with identical inputs
"""

import os
import random
import hashlib
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import logging

# Configure logging
det_logger = logging.getLogger(__name__)

class DeterministicProcessor:
    """
    Manages deterministic processing across the entire ensemble system.
    Ensures reproducible results for identical inputs.
    """
    
    def __init__(self, base_seed: Optional[int] = None):
        """
        Initialize deterministic processor.
        
        Args:
            base_seed: Base seed for deterministic processing (defaults to 42)
        """
        self.base_seed = base_seed or 42
        self.current_seed = self.base_seed
        self.seed_history: List[Dict[str, Any]] = []
        self.deterministic_mode = True
        
        det_logger.info(f"Initialized deterministic processor with base seed: {self.base_seed}")
    
    def generate_deterministic_run_id(self, video_path: str, 
                                    processing_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate deterministic run ID based on input hash and configuration.
        Ensures identical inputs produce identical run IDs.
        
        Args:
            video_path: Path to input video file
            processing_config: Processing configuration parameters
            
        Returns:
            Deterministic run ID string
        """
        # Create hash components
        hash_components = []
        
        # Add video file hash
        video_hash = self._hash_video_file(video_path)
        hash_components.append(video_hash)
        
        # Add configuration hash if provided
        if processing_config:
            config_hash = self._hash_dict(processing_config)
            hash_components.append(config_hash)
        
        # Add system version/identifier for cache invalidation across upgrades
        system_version = "u7_deterministic_v1.0"
        hash_components.append(system_version)
        
        # Create final hash
        combined = "_".join(hash_components)
        run_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        # Format as readable run ID
        run_id = f"det_{run_hash}"
        
        det_logger.info(f"Generated deterministic run ID: {run_id}")
        return run_id
    
    def _hash_video_file(self, video_path: str) -> str:
        """
        Generate stable hash for video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Hash string for file
        """
        try:
            # Get file metadata
            stat = os.stat(video_path)
            file_components = [
                str(stat.st_size),
                str(int(stat.st_mtime)),
                Path(video_path).name
            ]
            
            # Add content sample for verification
            with open(video_path, 'rb') as f:
                # Read strategic chunks for hash stability
                chunk_size = min(8192, stat.st_size // 10)  # Adaptive chunk size
                
                # Read beginning, middle, and end
                chunks = []
                chunks.append(f.read(chunk_size))
                
                if stat.st_size > chunk_size * 2:
                    f.seek(stat.st_size // 2)
                    chunks.append(f.read(chunk_size))
                    
                    f.seek(-chunk_size, os.SEEK_END)
                    chunks.append(f.read(chunk_size))
                
                content_hash = hashlib.md5(b''.join(chunks)).hexdigest()[:16]
                file_components.append(content_hash)
            
            combined = "_".join(file_components)
            return hashlib.sha256(combined.encode()).hexdigest()[:32]
            
        except Exception as e:
            det_logger.warning(f"Failed to hash video file {video_path}: {e}")
            # Fallback to basic file path hash
            return hashlib.md5(str(video_path).encode()).hexdigest()[:16]
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """
        Generate deterministic hash for dictionary data.
        
        Args:
            data: Dictionary to hash
            
        Returns:
            Hash string
        """
        try:
            # Sort keys for determinism
            sorted_data = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(sorted_data.encode()).hexdigest()[:16]
        except Exception as e:
            det_logger.warning(f"Failed to hash dictionary: {e}")
            return hashlib.md5(str(data).encode()).hexdigest()[:16]
    
    def set_deterministic_seed(self, component: str, additional_entropy: Any = None) -> int:
        """
        Set deterministic seed for a specific component.
        
        Args:
            component: Name of the component (e.g., 'diarization', 'asr', 'confidence')
            additional_entropy: Additional data to include in seed generation
            
        Returns:
            The generated seed value
        """
        if not self.deterministic_mode:
            return random.randint(0, 2**31 - 1)
        
        # Generate component-specific seed
        seed_components = [str(self.base_seed), component]
        
        if additional_entropy is not None:
            if isinstance(additional_entropy, (dict, list)):
                entropy_str = json.dumps(additional_entropy, sort_keys=True, default=str)
            else:
                entropy_str = str(additional_entropy)
            seed_components.append(entropy_str)
        
        # Create deterministic seed
        combined = "_".join(seed_components)
        seed_hash = hashlib.md5(combined.encode()).hexdigest()
        seed = int(seed_hash[:8], 16) % (2**31 - 1)  # Ensure valid seed range
        
        # Apply seed to all random number generators
        random.seed(seed)
        np.random.seed(seed)
        
        # Try to set torch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        # Try to set tensorflow seed if available
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        # Record seed usage
        self.seed_history.append({
            'component': component,
            'seed': seed,
            'entropy': additional_entropy,
            'timestamp': np.datetime64('now').item()
        })
        
        det_logger.debug(f"Set deterministic seed for {component}: {seed}")
        return seed
    
    def get_deterministic_variant_seed(self, base_component: str, variant_id: int) -> int:
        """
        Get deterministic seed for a specific variant of a component.
        
        Args:
            base_component: Base component name
            variant_id: Variant identifier
            
        Returns:
            Deterministic seed for the variant
        """
        return self.set_deterministic_seed(f"{base_component}_variant_{variant_id}", variant_id)
    
    def ensure_deterministic_ordering(self, items: List[Any], 
                                    sort_key: Optional[str] = None) -> List[Any]:
        """
        Ensure deterministic ordering of items for consistent parallel processing.
        
        Args:
            items: List of items to order
            sort_key: Key to use for sorting (if items are dicts)
            
        Returns:
            Deterministically ordered list
        """
        if not items:
            return items
        
        try:
            if sort_key and all(isinstance(item, dict) and sort_key in item for item in items):
                # Sort by specified key
                return sorted(items, key=lambda x: str(x[sort_key]))
            elif all(isinstance(item, dict) for item in items):
                # Sort by dictionary hash for determinism
                return sorted(items, key=lambda x: self._hash_dict(x))
            else:
                # Sort by string representation
                return sorted(items, key=lambda x: str(x))
        except Exception as e:
            det_logger.warning(f"Failed to sort items deterministically: {e}")
            return items
    
    def create_deterministic_mock_data(self, component: str, 
                                     data_type: str, 
                                     size: int,
                                     **kwargs) -> Any:
        """
        Create deterministic mock data for testing and fallback scenarios.
        
        Args:
            component: Component generating the mock data
            data_type: Type of data to generate
            size: Size/length of data to generate
            **kwargs: Additional parameters for data generation
            
        Returns:
            Deterministic mock data
        """
        # Set deterministic seed for mock data
        mock_seed = self.set_deterministic_seed(f"mock_{component}_{data_type}", kwargs)
        
        if data_type == "diarization_segments":
            return self._create_mock_diarization_segments(size, **kwargs)
        elif data_type == "asr_transcript":
            return self._create_mock_asr_transcript(size, **kwargs)
        elif data_type == "confidence_scores":
            return self._create_mock_confidence_scores(size, **kwargs)
        else:
            det_logger.warning(f"Unknown mock data type: {data_type}")
            return None
    
    def _create_mock_diarization_segments(self, duration: float, 
                                        num_speakers: int = 3,
                                        **kwargs) -> List[Dict[str, Any]]:
        """Create deterministic mock diarization segments."""
        segments = []
        current_time = 0.0
        
        # Generate deterministic segment durations
        np.random.seed(self.current_seed)
        segment_durations = np.random.exponential(2.0, size=int(duration * 2))  # Expect ~2 segments per second
        segment_durations = np.clip(segment_durations, 0.5, 10.0)  # Reasonable segment lengths
        
        for i, segment_duration in enumerate(segment_durations):
            if current_time >= duration:
                break
            
            # Deterministic speaker assignment
            speaker_id = f"SPEAKER_{i % num_speakers:02d}"
            
            segment = {
                'start': current_time,
                'end': min(current_time + segment_duration, duration),
                'speaker_id': speaker_id,
                'confidence': 0.7 + 0.2 * np.random.random()  # Deterministic confidence
            }
            
            segments.append(segment)
            current_time += segment_duration + np.random.exponential(0.1)  # Small gaps
        
        return segments
    
    def _create_mock_asr_transcript(self, num_words: int, **kwargs) -> str:
        """Create deterministic mock ASR transcript."""
        # Deterministic word pool
        word_pool = [
            "hello", "world", "this", "is", "a", "test", "transcript", "with", "multiple",
            "speakers", "discussing", "various", "topics", "during", "the", "meeting",
            "presentation", "project", "team", "work", "collaboration", "discussion",
            "analysis", "report", "data", "results", "conclusions", "recommendations"
        ]
        
        np.random.seed(self.current_seed)
        selected_words = np.random.choice(word_pool, size=num_words, replace=True)
        
        return " ".join(selected_words)
    
    def _create_mock_confidence_scores(self, num_candidates: int, **kwargs) -> List[float]:
        """Create deterministic mock confidence scores."""
        np.random.seed(self.current_seed)
        # Generate scores with realistic distribution
        scores = np.random.beta(2, 1, size=num_candidates)  # Skewed toward higher scores
        return scores.tolist()
    
    def get_processing_fingerprint(self, video_path: str, 
                                 config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get unique fingerprint for processing configuration.
        
        Args:
            video_path: Path to input video
            config: Processing configuration
            
        Returns:
            Fingerprint string for this processing setup
        """
        components = [
            self._hash_video_file(video_path),
            str(self.base_seed)
        ]
        
        if config:
            components.append(self._hash_dict(config))
        
        # Add seed history for complete reproducibility tracking
        seed_summary = self._hash_dict({
            'seed_count': len(self.seed_history),
            'base_seed': self.base_seed,
            'deterministic_mode': self.deterministic_mode
        })
        components.append(seed_summary)
        
        combined = "_".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:24]
    
    def reset_seeds(self, new_base_seed: Optional[int] = None):
        """
        Reset all seeds and start fresh.
        
        Args:
            new_base_seed: New base seed to use
        """
        if new_base_seed is not None:
            self.base_seed = new_base_seed
        
        self.current_seed = self.base_seed
        self.seed_history.clear()
        
        # Reset all RNG states
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        
        det_logger.info(f"Reset seeds to base: {self.base_seed}")
    
    def get_seed_report(self) -> Dict[str, Any]:
        """Get report of all seed usage during processing."""
        return {
            'base_seed': self.base_seed,
            'deterministic_mode': self.deterministic_mode,
            'total_seeds_used': len(self.seed_history),
            'components_seeded': list(set(h['component'] for h in self.seed_history)),
            'seed_history': self.seed_history.copy() if len(self.seed_history) < 100 else self.seed_history[-100:]  # Last 100 for space
        }


# Global deterministic processor instance
_deterministic_processor: Optional[DeterministicProcessor] = None

def get_deterministic_processor() -> DeterministicProcessor:
    """Get or create global deterministic processor instance."""
    global _deterministic_processor
    if _deterministic_processor is None:
        _deterministic_processor = DeterministicProcessor()
    return _deterministic_processor

def set_global_seed(component: str, additional_entropy: Any = None) -> int:
    """Convenience function to set deterministic seed globally."""
    return get_deterministic_processor().set_deterministic_seed(component, additional_entropy)

def ensure_deterministic_run_id(video_path: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to generate deterministic run ID."""
    return get_deterministic_processor().generate_deterministic_run_id(video_path, config)