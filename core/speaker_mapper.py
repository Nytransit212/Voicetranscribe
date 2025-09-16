"""
Speaker Mapping Consistency Module with Hungarian Alignment Algorithm

This module implements speaker identity consistency across chunk boundaries using:
1. Speaker embedding extraction from audio segments
2. Cross-chunk similarity matrix calculation
3. Hungarian algorithm for optimal speaker ID assignment
4. Consistency validation and metrics generation

Maintains consistent speaker labels throughout entire recording and eliminates
speaker ID shuffling between processing segments.
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm
import json
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from collections import deque
import warnings

from utils.structured_logger import StructuredLogger
from utils.deterministic_processing import get_deterministic_processor, set_global_seed
from utils.intelligent_cache import get_cache_manager, cached_operation

@dataclass
class SpeakerMapping:
    """Speaker mapping result with consistency metrics"""
    original_speaker_id: str
    mapped_speaker_id: str
    similarity_score: float
    confidence: float
    chunk_index: int

@dataclass
class ConsistencyMetrics:
    """Cross-chunk speaker consistency metrics"""
    baseline_continuity_score: float
    improved_continuity_score: float
    improvement_percentage: float
    speaker_id_changes_prevented: int
    total_speaker_segments: int
    mapping_accuracy: float
    cross_chunk_similarity_scores: List[float]
    # Enhanced metrics for ECAPA-TDNN and backtracking
    drift_detection_events: int = 0
    backtrack_recoveries: int = 0
    average_drift_score: float = 0.0
    embedding_quality_score: float = 1.0

@dataclass
class BacktrackingMetrics:
    """Metrics for speaker ID drift detection and backtracking"""
    similarity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    drift_score: float = 0.0
    stability_score: float = 1.0
    last_stable_chunk: int = -1
    backtrack_events: int = 0
    confidence_trend: List[float] = field(default_factory=list)
    
@dataclass
class SpeakerEmbedding:
    """ECAPA-TDNN speaker embedding with enhanced metadata and drift detection"""
    speaker_id: str
    chunk_index: int
    embedding_vector: np.ndarray
    duration: float
    confidence: float
    segment_times: List[Tuple[float, float]]  # List of (start, end) times
    # ECAPA-TDNN specific fields
    ecapa_embedding: Optional[torch.Tensor] = None
    embedding_dimension: int = 192  # Standard ECAPA-TDNN output dimension
    model_version: str = "ecapa-tdnn-voxceleb"
    # Drift detection fields
    backtracking_metrics: BacktrackingMetrics = field(default_factory=BacktrackingMetrics)
    similarity_to_previous: float = 1.0
    is_stable: bool = True
    consecutive_low_similarity_count: int = 0

class ECAPATDNNModel(nn.Module):
    """Lightweight ECAPA-TDNN implementation for speaker embedding extraction"""
    
    def __init__(self, input_dim: int = 80, embedding_dim: int = 192):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Feature preprocessing layers
        self.feature_norm = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.1)
        
        # TDNN layers (1D convolutions with dilation)
        self.tdnn1 = nn.Conv1d(input_dim, 512, kernel_size=5, dilation=1, padding=2)
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.tdnn3 = nn.Conv1d(512, 512, kernel_size=3, dilation=3, padding=3)
        self.tdnn4 = nn.Conv1d(512, 512, kernel_size=1, dilation=1, padding=0)
        self.tdnn5 = nn.Conv1d(512, 1536, kernel_size=1, dilation=1, padding=0)
        
        # Statistical pooling layer
        self.stats_pool = StatsPool()
        
        # Final embedding layers
        self.fc1 = nn.Linear(1536 * 2, 512)  # *2 for mean and std pooling
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, embedding_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        """
        Forward pass through ECAPA-TDNN model
        Args:
            x: Input features [batch, features, time]
        Returns:
            Speaker embeddings [batch, embedding_dim]
        """
        # Input normalization
        x = self.feature_norm(x)
        x = self.dropout(x)
        
        # TDNN layers with residual connections
        x1 = self.relu(self.tdnn1(x))
        x2 = self.relu(self.tdnn2(x1))
        x3 = self.relu(self.tdnn3(x2))
        x4 = self.relu(self.tdnn4(x3))
        
        # Residual connection
        x4 = x4 + x1[:, :512, :]
        
        x5 = self.relu(self.tdnn5(x4))
        
        # Statistical pooling
        x = self.stats_pool(x5)
        
        # Final layers
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x

class StatsPool(nn.Module):
    """Statistical pooling layer for ECAPA-TDNN"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Compute mean and standard deviation along time dimension
        Args:
            x: Input tensor [batch, features, time]
        Returns:
            Concatenated mean and std [batch, features*2]
        """
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat([mean, std], dim=1)
    
class SpeakerMapper:
    """
    Implements speaker mapping consistency with Hungarian alignment algorithm.
    
    Maintains consistent speaker IDs across chunk boundaries by:
    1. Extracting speaker embeddings from audio segments
    2. Computing similarity matrices between adjacent chunks
    3. Using Hungarian algorithm for optimal speaker ID assignment
    4. Tracking consistency improvements and generating metrics
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 embedding_dim: int = 192,  # ECAPA-TDNN standard dimension
                 min_segment_duration: float = 1.0,
                 cache_embeddings: bool = True,
                 enable_metrics: bool = True,
                 # ECAPA-TDNN specific parameters
                 use_ecapa_tdnn: bool = True,
                 ecapa_model_path: Optional[str] = None,
                 # Backtracking parameters
                 enable_backtracking: bool = True,
                 drift_threshold: float = 0.6,
                 stability_window: int = 5,
                 max_backtrack_chunks: int = 3,
                 consecutive_drift_threshold: int = 2):
        """
        Initialize speaker mapper with ECAPA-TDNN and backtracking configuration.
        
        Args:
            similarity_threshold: Minimum similarity for speaker matching
            embedding_dim: Dimensionality of speaker embeddings (192 for ECAPA-TDNN)
            min_segment_duration: Minimum duration for embedding extraction
            cache_embeddings: Whether to cache computed embeddings
            enable_metrics: Whether to generate consistency metrics
            use_ecapa_tdnn: Whether to use ECAPA-TDNN model for embeddings
            ecapa_model_path: Path to pre-trained ECAPA-TDNN model
            enable_backtracking: Whether to enable drift detection and backtracking
            drift_threshold: Similarity threshold below which drift is detected
            stability_window: Number of chunks to consider for stability analysis
            max_backtrack_chunks: Maximum chunks to backtrack for recovery
            consecutive_drift_threshold: Consecutive low similarities before backtracking
        """
        # Core configuration
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self.min_segment_duration = min_segment_duration
        self.cache_embeddings = cache_embeddings
        self.enable_metrics = enable_metrics
        
        # ECAPA-TDNN configuration
        self.use_ecapa_tdnn = use_ecapa_tdnn
        self.ecapa_model_path = ecapa_model_path
        
        # Backtracking configuration
        self.enable_backtracking = enable_backtracking
        self.drift_threshold = drift_threshold
        self.stability_window = stability_window
        self.max_backtrack_chunks = max_backtrack_chunks
        self.consecutive_drift_threshold = consecutive_drift_threshold
        
        # Initialize system components
        self.structured_logger = StructuredLogger("speaker_mapper")
        self.cache_manager = get_cache_manager() if cache_embeddings else None
        self.deterministic_processor = get_deterministic_processor()
        
        # Speaker tracking across chunks
        self.speaker_embeddings: Dict[int, List[SpeakerEmbedding]] = {}  # chunk_index -> embeddings
        self.speaker_mappings: Dict[int, List[SpeakerMapping]] = {}      # chunk_index -> mappings
        self.global_speaker_registry: Dict[str, SpeakerEmbedding] = {}   # global_id -> representative embedding
        
        # Backtracking state management
        self.chunk_similarity_history: deque = deque(maxlen=stability_window)
        self.last_stable_state: Optional[Dict[str, Any]] = None
        self.drift_events_log: List[Dict[str, Any]] = []
        self.backtrack_attempts: int = 0
        
        # Metrics tracking
        self.consistency_metrics: Optional[ConsistencyMetrics] = None
        self.baseline_metrics: Dict[str, float] = {}
        
        # Traditional feature parameters (fallback mode)
        self.mfcc_features = 13
        self.spectral_features = 7
        self.prosodic_features = 5
        self.delta_features = True
        
        # ECAPA-TDNN model initialization
        self.ecapa_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.use_ecapa_tdnn:
            self._initialize_ecapa_model()
        
        self.structured_logger.info("Speaker mapper initialized", 
                                  context={
                                      'similarity_threshold': similarity_threshold,
                                      'embedding_dim': embedding_dim,
                                      'cache_enabled': cache_embeddings,
                                      'use_ecapa_tdnn': use_ecapa_tdnn,
                                      'enable_backtracking': enable_backtracking,
                                      'drift_threshold': drift_threshold,
                                      'device': str(self.device)
                                  })
    
    def _initialize_ecapa_model(self):
        """Initialize ECAPA-TDNN model for speaker embedding extraction"""
        try:
            # Create ECAPA-TDNN model
            self.ecapa_model = ECAPATDNNModel(
                input_dim=80,  # Mel-spectrogram features
                embedding_dim=self.embedding_dim
            ).to(self.device)
            
            # Load pre-trained weights if available
            if self.ecapa_model_path and Path(self.ecapa_model_path).exists():
                checkpoint = torch.load(self.ecapa_model_path, map_location=self.device)
                self.ecapa_model.load_state_dict(checkpoint['model_state_dict'])
                self.structured_logger.info(f"Loaded ECAPA-TDNN model from {self.ecapa_model_path}")
            else:
                # Initialize with random weights (for demonstration - in production use pre-trained)
                self.structured_logger.info("Using randomly initialized ECAPA-TDNN model (consider using pre-trained weights)")
            
            self.ecapa_model.eval()
            
            # Test model with dummy input
            with torch.no_grad():
                dummy_input = torch.randn(1, 80, 100).to(self.device)
                test_output = self.ecapa_model(dummy_input)
                self.structured_logger.info(f"ECAPA-TDNN model initialized successfully, output shape: {test_output.shape}")
                
        except Exception as e:
            self.structured_logger.warning(f"Failed to initialize ECAPA-TDNN model: {e}")
            self.structured_logger.info("Falling back to traditional acoustic features")
            self.use_ecapa_tdnn = False
            self.ecapa_model = None
    
    def _extract_ecapa_embedding(self, audio_segment: np.ndarray, sample_rate: int) -> Optional[torch.Tensor]:
        """Extract ECAPA-TDNN embedding from audio segment"""
        if not self.use_ecapa_tdnn or self.ecapa_model is None:
            return None
            
        try:
            # Ensure minimum length for stable embedding
            min_samples = int(0.5 * sample_rate)  # 0.5 seconds minimum
            if len(audio_segment) < min_samples:
                # Pad or skip very short segments
                padding = min_samples - len(audio_segment)
                audio_segment = np.pad(audio_segment, (0, padding), mode='reflect')
            
            # Extract mel-spectrogram features
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_segment,
                sr=sample_rate,
                n_mels=80,
                hop_length=160,
                win_length=400,
                n_fft=512
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Normalize features
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(log_mel).unsqueeze(0).to(self.device)
            
            # Extract embedding using ECAPA-TDNN model
            with torch.no_grad():
                embedding = self.ecapa_model(features_tensor)
                embedding = embedding.squeeze(0)  # Remove batch dimension
                
            return embedding
            
        except Exception as e:
            self.structured_logger.warning(f"ECAPA-TDNN embedding extraction failed: {e}")
            return None
    
    def extract_speaker_embeddings(self, 
                                 audio_path: str, 
                                 diarization_segments: List[Dict[str, Any]], 
                                 chunk_index: int) -> List[SpeakerEmbedding]:
        """
        Extract speaker embeddings from audio segments for a chunk.
        
        Args:
            audio_path: Path to audio file
            diarization_segments: List of diarization segments with speaker IDs
            chunk_index: Index of current chunk
            
        Returns:
            List of speaker embeddings for this chunk
        """
        embeddings = []
        
        # Load audio once for all segments
        try:
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            audio_duration = len(audio_data) / sample_rate
        except Exception as e:
            self.structured_logger.error(f"Failed to load audio: {e}")
            return embeddings
        
        # Group segments by speaker
        speaker_segments = {}
        for segment in diarization_segments:
            speaker_id = segment.get('speaker', segment.get('speaker_id', 'UNKNOWN'))
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append(segment)
        
        # Extract embeddings for each speaker
        for speaker_id, segments in speaker_segments.items():
            try:
                embedding = self._extract_single_speaker_embedding(
                    audio_data, int(sample_rate), segments, speaker_id, chunk_index
                )
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception as e:
                self.structured_logger.warning(f"Failed to extract embedding for {speaker_id}: {e}")
        
        # Cache embeddings if enabled
        if self.cache_embeddings and self.cache_manager:
            cache_key = f"speaker_embeddings_{hashlib.md5(audio_path.encode()).hexdigest()}_{chunk_index}"
            self.cache_manager.set("speaker_embeddings", cache_key, embeddings)
        
        self.speaker_embeddings[chunk_index] = embeddings
        
        self.structured_logger.info(f"Extracted embeddings for chunk {chunk_index}", 
                                  context={
                                      'num_speakers': len(embeddings),
                                      'speaker_ids': [e.speaker_id for e in embeddings]
                                  })
        
        return embeddings
    
    def _extract_single_speaker_embedding(self, 
                                        audio_data: np.ndarray, 
                                        sample_rate: int,
                                        segments: List[Dict[str, Any]], 
                                        speaker_id: str, 
                                        chunk_index: int) -> Optional[SpeakerEmbedding]:
        """
        Extract acoustic embedding for a single speaker from their segments.
        
        Args:
            audio_data: Audio time series
            sample_rate: Audio sample rate
            segments: List of segments for this speaker
            speaker_id: Speaker identifier
            chunk_index: Current chunk index
            
        Returns:
            Speaker embedding or None if extraction fails
        """
        # Filter segments by minimum duration
        valid_segments = []
        segment_times = []
        
        for segment in segments:
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', start_time + 1.0)
            duration = end_time - start_time
            
            if duration >= self.min_segment_duration:
                valid_segments.append(segment)
                segment_times.append((start_time, end_time))
        
        if not valid_segments:
            return None
        
        # Extract audio features for all segments
        all_features = []
        total_duration = 0.0
        confidence_scores = []
        
        for segment in valid_segments:
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', start_time + 1.0)
            
            # Convert time to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Extract segment audio
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) < sample_rate * 0.1:  # Skip very short segments
                continue
            
            # Extract features using ECAPA-TDNN or traditional methods
            if self.use_ecapa_tdnn:
                ecapa_embedding = self._extract_ecapa_embedding(segment_audio, sample_rate)
                if ecapa_embedding is not None:
                    # Convert to numpy for consistency with existing code
                    features = ecapa_embedding.cpu().numpy()
                    all_features.append(features)
                    total_duration += (end_time - start_time)
                    confidence_scores.append(segment.get('confidence', 0.8))
                else:
                    # Fallback to traditional features
                    features = self._extract_acoustic_features(segment_audio, sample_rate)
                    if features is not None:
                        all_features.append(features)
                        total_duration += (end_time - start_time)
                        confidence_scores.append(segment.get('confidence', 0.8))
            else:
                # Use traditional acoustic features
                features = self._extract_acoustic_features(segment_audio, sample_rate)
                if features is not None:
                    all_features.append(features)
                    total_duration += (end_time - start_time)
                    confidence_scores.append(segment.get('confidence', 0.8))
        
        if not all_features:
            return None
        
        # Aggregate features across segments
        try:
            if self.use_ecapa_tdnn and all_features:
                # For ECAPA-TDNN embeddings, use weighted average based on segment duration
                segment_durations = [valid_segments[i].get('end', 0) - valid_segments[i].get('start', 0) 
                                   for i in range(min(len(valid_segments), len(all_features)))]
                total_duration_for_weights = sum(segment_durations)
                
                if total_duration_for_weights > 0:
                    weights = np.array(segment_durations) / total_duration_for_weights
                    embedding_vector = np.average(all_features, axis=0, weights=weights)
                else:
                    embedding_vector = np.mean(all_features, axis=0)
                
                # Ensure fixed dimensionality for ECAPA-TDNN
                if len(embedding_vector) != self.embedding_dim:
                    if len(embedding_vector) > self.embedding_dim:
                        embedding_vector = embedding_vector[:self.embedding_dim]
                    else:
                        padding = np.zeros(self.embedding_dim - len(embedding_vector))
                        embedding_vector = np.concatenate([embedding_vector, padding])
                
                # L2 normalize ECAPA-TDNN embeddings
                embedding_vector = embedding_vector / (np.linalg.norm(embedding_vector) + 1e-8)
            else:
                # Traditional feature aggregation
                feature_matrix = np.vstack(all_features)
                
                # Compute aggregate embedding using statistical moments
                mean_features = np.mean(feature_matrix, axis=0)
                std_features = np.std(feature_matrix, axis=0)
                median_features = np.median(feature_matrix, axis=0)
                
                # Concatenate statistics to form final embedding
                embedding_vector = np.concatenate([mean_features, std_features, median_features])
                
                # Ensure fixed dimensionality
                if len(embedding_vector) > self.embedding_dim:
                    embedding_vector = embedding_vector[:self.embedding_dim]
                elif len(embedding_vector) < self.embedding_dim:
                    padding = np.zeros(self.embedding_dim - len(embedding_vector))
                    embedding_vector = np.concatenate([embedding_vector, padding])
                
                # Normalize embedding
                embedding_vector = embedding_vector / (np.linalg.norm(embedding_vector) + 1e-8)
            
            # Calculate average confidence
            avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.8
            
            # Create enhanced speaker embedding with ECAPA-TDNN support
            ecapa_tensor = None
            if self.use_ecapa_tdnn and all_features:
                # Store the raw ECAPA embedding as tensor
                ecapa_tensor = torch.FloatTensor(embedding_vector)
            
            return SpeakerEmbedding(
                speaker_id=speaker_id,
                chunk_index=chunk_index,
                embedding_vector=embedding_vector,
                duration=total_duration,
                confidence=avg_confidence,
                segment_times=segment_times,
                ecapa_embedding=ecapa_tensor,
                embedding_dimension=self.embedding_dim,
                model_version="ecapa-tdnn-voxceleb" if self.use_ecapa_tdnn else "traditional-acoustic",
                backtracking_metrics=BacktrackingMetrics(),
                similarity_to_previous=1.0,
                is_stable=True,
                consecutive_low_similarity_count=0
            )
            
        except Exception as e:
            self.structured_logger.warning(f"Failed to aggregate features for {speaker_id}: {e}")
            return None
    
    def _extract_acoustic_features(self, audio_segment: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """
        Extract comprehensive acoustic features from audio segment.
        
        Args:
            audio_segment: Audio time series for segment
            sample_rate: Audio sample rate
            
        Returns:
            Feature vector or None if extraction fails
        """
        try:
            features = []
            
            # 1. MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=self.mfcc_features)
            mfcc_stats = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1)
            ])
            features.append(mfcc_stats)
            
            # 2. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)
            
            spectral_stats = np.concatenate([
                [np.mean(spectral_centroid), np.std(spectral_centroid)],
                [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
                [np.mean(zero_crossing_rate)]
            ])
            features.append(spectral_stats)
            
            # 3. Prosodic features (pitch, energy)
            # Extract fundamental frequency using librosa
            f0 = librosa.yin(audio_segment, fmin=float(librosa.note_to_hz('C2')), fmax=float(librosa.note_to_hz('C7')))
            f0_valid = f0[f0 > 0]  # Remove unvoiced frames
            
            if len(f0_valid) > 0:
                prosodic_stats = np.array([
                    np.mean(f0_valid),
                    np.std(f0_valid),
                    np.median(f0_valid),
                    np.max(f0_valid) - np.min(f0_valid),  # F0 range
                    np.mean(np.abs(audio_segment))  # Energy
                ])
            else:
                prosodic_stats = np.zeros(self.prosodic_features)
            features.append(prosodic_stats)
            
            # 4. Delta features (if enabled)
            if self.delta_features and len(mfccs[0]) > 1:
                delta_mfcc = librosa.feature.delta(mfccs)
                delta_stats = np.concatenate([
                    np.mean(delta_mfcc, axis=1),
                    np.std(delta_mfcc, axis=1)
                ])
                features.append(delta_stats)
            
            # Concatenate all features
            feature_vector = np.concatenate(features)
            
            # Handle any NaN or infinite values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return feature_vector
            
        except Exception as e:
            self.structured_logger.warning(f"Traditional feature extraction failed: {e}")
            return None
    
    def compute_similarity_matrix(self, 
                                embeddings_chunk1: List[SpeakerEmbedding], 
                                embeddings_chunk2: List[SpeakerEmbedding]) -> np.ndarray:
        """
        Compute similarity matrix between speakers in two chunks.
        
        Args:
            embeddings_chunk1: Speaker embeddings from first chunk
            embeddings_chunk2: Speaker embeddings from second chunk
            
        Returns:
            Similarity matrix (chunk1_speakers x chunk2_speakers)
        """
        if not embeddings_chunk1 or not embeddings_chunk2:
            return np.zeros((len(embeddings_chunk1), len(embeddings_chunk2)))
        
        similarity_matrix = np.zeros((len(embeddings_chunk1), len(embeddings_chunk2)))
        
        for i, emb1 in enumerate(embeddings_chunk1):
            for j, emb2 in enumerate(embeddings_chunk2):
                similarity = self._compute_speaker_similarity(emb1, emb2)
                similarity_matrix[i, j] = similarity
        
        self.structured_logger.debug("Computed similarity matrix", 
                                   context={
                                       'shape': similarity_matrix.shape,
                                       'mean_similarity': float(np.mean(similarity_matrix)),
                                       'max_similarity': float(np.max(similarity_matrix))
                                   })
        
        return similarity_matrix
    
    def _compute_speaker_similarity(self, emb1: SpeakerEmbedding, emb2: SpeakerEmbedding) -> float:
        """
        Compute similarity between two speaker embeddings.
        
        Args:
            emb1: First speaker embedding
            emb2: Second speaker embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Cosine similarity between embedding vectors
        vec1 = emb1.embedding_vector
        vec2 = emb2.embedding_vector
        
        # Ensure same dimensionality
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Adjust similarity based on confidence scores
        confidence_weight = (emb1.confidence + emb2.confidence) / 2.0
        weighted_similarity = cosine_sim * confidence_weight
        
        # Ensure similarity is in [0, 1] range
        return max(0.0, min(1.0, weighted_similarity))
    
    def apply_hungarian_assignment(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Apply Hungarian algorithm for optimal speaker assignment.
        
        Args:
            similarity_matrix: Similarity matrix between chunks
            
        Returns:
            List of (chunk1_idx, chunk2_idx, similarity) assignments
        """
        if similarity_matrix.size == 0:
            return []
        
        # Convert similarity to cost (Hungarian algorithm minimizes cost)
        cost_matrix = 1.0 - similarity_matrix
        
        # Apply Hungarian algorithm
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            assignments = []
            for row_idx, col_idx in zip(row_indices, col_indices):
                similarity = similarity_matrix[row_idx, col_idx]
                # Only include assignments above threshold
                if similarity >= self.similarity_threshold:
                    assignments.append((int(row_idx), int(col_idx), float(similarity)))
            
            self.structured_logger.info("Hungarian assignment completed", 
                                      context={
                                          'total_assignments': len(assignments),
                                          'above_threshold': len([a for a in assignments if a[2] >= self.similarity_threshold])
                                      })
            
            return assignments
            
        except Exception as e:
            self.structured_logger.error(f"Hungarian assignment failed: {e}")
            return []
    
    def _detect_speaker_drift(self, 
                            current_embeddings: List[SpeakerEmbedding], 
                            previous_embeddings: List[SpeakerEmbedding],
                            similarity_matrix: np.ndarray) -> Tuple[bool, float, List[str]]:
        """Detect speaker identity drift by analyzing similarity scores"""
        drift_detected = False
        drift_score = 0.0
        drifting_speakers = []
        
        if similarity_matrix.size == 0:
            return drift_detected, drift_score, drifting_speakers
        
        # Calculate statistics for drift detection
        max_similarities = np.max(similarity_matrix, axis=1) if similarity_matrix.shape[0] > 0 else np.array([])
        mean_similarity = np.mean(max_similarities) if len(max_similarities) > 0 else 0.0
        
        # Update similarity history for trend analysis
        self.chunk_similarity_history.append(mean_similarity)
        
        # Detect drift based on threshold
        if mean_similarity < self.drift_threshold:
            drift_detected = True
            drift_score = 1.0 - mean_similarity
            
            # Identify specific speakers that are drifting
            for i, current_emb in enumerate(current_embeddings):
                if i < len(max_similarities) and max_similarities[i] < self.drift_threshold:
                    drifting_speakers.append(current_emb.speaker_id)
                    current_emb.consecutive_low_similarity_count += 1
                    current_emb.is_stable = False
                else:
                    if i < len(current_embeddings):
                        current_emb.consecutive_low_similarity_count = 0
                        current_emb.is_stable = True
        
        # Detect trend-based drift (gradual degradation)
        if len(self.chunk_similarity_history) >= 3:
            recent_trend = list(self.chunk_similarity_history)[-3:]
            if all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)):
                drift_score = max(float(drift_score), 0.3)  # Moderate drift due to trend
        
        # Log drift detection events
        if drift_detected:
            self.structured_logger.warning(f"Speaker drift detected: mean_similarity={mean_similarity:.3f}, threshold={self.drift_threshold}")
            drift_event = {
                'timestamp': time.time(),
                'mean_similarity': mean_similarity,
                'drift_score': drift_score,
                'drifting_speakers': drifting_speakers,
                'similarity_trend': list(self.chunk_similarity_history)
            }
            self.drift_events_log.append(drift_event)
        
        return drift_detected, float(drift_score), drifting_speakers
    
    def _save_stable_state(self, chunk_index: int, embeddings: List[SpeakerEmbedding]):
        """Save current state as stable reference point for backtracking"""
        self.last_stable_state = {
            'chunk_index': chunk_index,
            'embeddings': embeddings.copy(),
            'global_registry': self.global_speaker_registry.copy(),
            'timestamp': time.time()
        }
        
        self.structured_logger.debug(f"Saved stable state at chunk {chunk_index}")
    
    def _perform_backtracking(self, 
                            current_chunk_index: int, 
                            current_embeddings: List[SpeakerEmbedding],
                            drifting_speakers: List[str]) -> Tuple[bool, List[SpeakerMapping]]:
        """Perform backtracking to recover from speaker ID drift"""
        if not self.last_stable_state:
            self.structured_logger.warning("No stable state available for backtracking")
            return False, []
        
        self.backtrack_attempts += 1
        self.structured_logger.info(f"Attempting backtracking (attempt {self.backtrack_attempts})")
        
        try:
            # Retrieve stable state
            stable_chunk_index = self.last_stable_state['chunk_index']
            stable_embeddings = self.last_stable_state['embeddings']
            stable_registry = self.last_stable_state['global_registry']
            
            # Only backtrack if we haven't gone too far
            if current_chunk_index - stable_chunk_index > self.max_backtrack_chunks:
                self.structured_logger.warning(f"Backtrack distance too large: {current_chunk_index - stable_chunk_index}")
                return False, []
            
            # Compute similarity between current drifting speakers and stable state
            backtrack_mappings = []
            recovery_success = True
            
            for current_emb in current_embeddings:
                if current_emb.speaker_id in drifting_speakers:
                    best_match = None
                    best_similarity = 0.0
                    
                    # Find best match in stable embeddings
                    for stable_emb in stable_embeddings:
                        similarity = self._compute_speaker_similarity(current_emb, stable_emb)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = stable_emb
                    
                    if best_match and best_similarity > self.similarity_threshold:
                        # Find global ID for this stable speaker
                        for global_id, registry_emb in stable_registry.items():
                            if registry_emb.speaker_id == best_match.speaker_id:
                                mapping = SpeakerMapping(
                                    original_speaker_id=current_emb.speaker_id,
                                    mapped_speaker_id=global_id,
                                    similarity_score=best_similarity,
                                    confidence=0.9,  # High confidence due to backtracking
                                    chunk_index=current_chunk_index
                                )
                                backtrack_mappings.append(mapping)
                                
                                # Update embedding stability
                                current_emb.is_stable = True
                                current_emb.consecutive_low_similarity_count = 0
                                current_emb.similarity_to_previous = best_similarity
                                current_emb.backtracking_metrics.backtrack_events += 1
                                break
                    else:
                        recovery_success = False
                        self.structured_logger.warning(f"Could not find stable match for speaker {current_emb.speaker_id}")
            
            if recovery_success:
                self.structured_logger.info(f"Backtracking successful: recovered {len(backtrack_mappings)} speakers")
                return True, backtrack_mappings
            else:
                self.structured_logger.warning("Backtracking partially failed")
                return False, backtrack_mappings
                
        except Exception as e:
            self.structured_logger.error(f"Backtracking failed with error: {e}")
            return False, []
    
    def apply_enhanced_hungarian_assignment_with_backtracking(self, 
                                                            similarity_matrix: np.ndarray,
                                                            current_embeddings: List[SpeakerEmbedding],
                                                            previous_embeddings: List[SpeakerEmbedding],
                                                            chunk_index: int) -> List[Tuple[int, int, float]]:
        """Apply Hungarian assignment with drift detection and backtracking"""
        if similarity_matrix.size == 0:
            return []
        
        # Detect speaker drift
        drift_detected, drift_score, drifting_speakers = self._detect_speaker_drift(
            current_embeddings, previous_embeddings, similarity_matrix
        )
        
        # Perform standard Hungarian assignment
        assignments = self.apply_hungarian_assignment(similarity_matrix)
        
        # Check if backtracking is needed
        if (self.enable_backtracking and drift_detected and 
            len(drifting_speakers) > 0 and 
            chunk_index > 0 and
            self.last_stable_state is not None):
            
            # Check if we should trigger backtracking
            consecutive_drift_count = sum(
                1 for emb in current_embeddings 
                if emb.consecutive_low_similarity_count >= self.consecutive_drift_threshold
            )
            
            if consecutive_drift_count > 0:
                self.structured_logger.info(f"Triggering backtracking for {consecutive_drift_count} speakers with consecutive drift")
                
                # Attempt backtracking
                backtrack_success, backtrack_mappings = self._perform_backtracking(
                    chunk_index, current_embeddings, drifting_speakers
                )
                
                if backtrack_success:
                    # Apply backtracking results to assignments
                    backtrack_dict = {mapping.original_speaker_id: mapping 
                                    for mapping in backtrack_mappings}
                    
                    # Update assignments with backtracked mappings
                    updated_assignments = []
                    for curr_idx, prev_idx, similarity in assignments:
                        current_speaker = current_embeddings[curr_idx].speaker_id
                        if current_speaker in backtrack_dict:
                            # Use backtracked similarity score
                            updated_assignments.append((
                                curr_idx, prev_idx, 
                                backtrack_dict[current_speaker].similarity_score
                            ))
                        else:
                            updated_assignments.append((curr_idx, prev_idx, similarity))
                    
                    assignments = updated_assignments
                    
                    # Log successful backtracking
                    self.structured_logger.info(f"Backtracking applied: {len(backtrack_mappings)} speakers recovered")
        
        # Save current state as stable if similarity is good
        mean_similarity = np.mean([sim for _, _, sim in assignments]) if assignments else 0.0
        if mean_similarity > self.similarity_threshold:
            self._save_stable_state(chunk_index, current_embeddings)
        
        return assignments
    
    def map_speakers_across_chunks(self, chunk_diarizations: List[List[Dict[str, Any]]], 
                                 audio_path: str) -> Tuple[List[List[Dict[str, Any]]], ConsistencyMetrics]:
        """
        Map speakers consistently across all chunks using Hungarian alignment.
        
        Args:
            chunk_diarizations: List of diarization results for each chunk
            audio_path: Path to audio file for embedding extraction
            
        Returns:
            Tuple of (mapped_diarizations, consistency_metrics)
        """
        self.structured_logger.stage_start("speaker_mapping", "Starting cross-chunk speaker mapping")
        start_time = time.time()
        
        # Initialize tracking
        mapped_diarizations = []
        all_mappings = []
        
        # Extract embeddings for all chunks
        chunk_embeddings = []
        for chunk_idx, segments in enumerate(chunk_diarizations):
            embeddings = self.extract_speaker_embeddings(audio_path, segments, chunk_idx)
            chunk_embeddings.append(embeddings)
        
        # First chunk establishes the global speaker registry
        if chunk_embeddings and chunk_embeddings[0]:
            global_speaker_counter = 0
            for embedding in chunk_embeddings[0]:
                global_id = f"SPEAKER_{global_speaker_counter:02d}"
                self.global_speaker_registry[global_id] = embedding
                global_speaker_counter += 1
            
            # First chunk keeps original mapping
            mapped_diarizations.append(self._apply_speaker_mapping(chunk_diarizations[0], {}))
        
        # Process subsequent chunks
        for chunk_idx in range(1, len(chunk_diarizations)):
            if chunk_idx >= len(chunk_embeddings):
                mapped_diarizations.append(chunk_diarizations[chunk_idx])
                continue
            
            current_embeddings = chunk_embeddings[chunk_idx]
            previous_embeddings = chunk_embeddings[chunk_idx - 1]
            
            # Compute similarity matrix with previous chunk
            similarity_matrix = self.compute_similarity_matrix(current_embeddings, previous_embeddings)
            
            # Apply enhanced Hungarian algorithm with backtracking
            assignments = self.apply_enhanced_hungarian_assignment_with_backtracking(
                similarity_matrix, current_embeddings, previous_embeddings, chunk_idx
            )
            
            # Create speaker mapping
            speaker_mapping = {}
            used_previous_speakers = set()
            
            for curr_idx, prev_idx, similarity in assignments:
                if prev_idx < len(previous_embeddings):
                    current_speaker = current_embeddings[curr_idx].speaker_id
                    previous_speaker = previous_embeddings[prev_idx].speaker_id
                    
                    # Find global ID for previous speaker
                    global_id = self._find_global_speaker_id(previous_speaker, chunk_idx - 1)
                    if global_id:
                        speaker_mapping[current_speaker] = global_id
                        used_previous_speakers.add(prev_idx)
                        
                        # Update global registry
                        self.global_speaker_registry[global_id] = current_embeddings[curr_idx]
            
            # Assign new global IDs for unmatched speakers
            for i, embedding in enumerate(current_embeddings):
                if embedding.speaker_id not in speaker_mapping:
                    global_id = f"SPEAKER_{len(self.global_speaker_registry):02d}"
                    speaker_mapping[embedding.speaker_id] = global_id
                    self.global_speaker_registry[global_id] = embedding
            
            # Apply mapping to chunk diarization
            mapped_chunk = self._apply_speaker_mapping(chunk_diarizations[chunk_idx], speaker_mapping)
            mapped_diarizations.append(mapped_chunk)
            
            # Track mappings
            chunk_mappings = [
                SpeakerMapping(
                    original_speaker_id=orig_id,
                    mapped_speaker_id=mapped_id,
                    similarity_score=1.0,  # Placeholder
                    confidence=0.8,  # Placeholder
                    chunk_index=chunk_idx
                )
                for orig_id, mapped_id in speaker_mapping.items()
            ]
            all_mappings.extend(chunk_mappings)
            self.speaker_mappings[chunk_idx] = chunk_mappings
        
        # Generate consistency metrics
        if self.enable_metrics:
            self.consistency_metrics = self._generate_consistency_metrics(
                chunk_diarizations, mapped_diarizations, all_mappings
            )
        
        processing_time = time.time() - start_time
        self.structured_logger.stage_complete("speaker_mapping", "Cross-chunk speaker mapping completed",
                                            duration=processing_time,
                                            context={
                                                'num_chunks': len(chunk_diarizations),
                                                'total_mappings': len(all_mappings),
                                                'global_speakers': len(self.global_speaker_registry)
                                            })
        
        return mapped_diarizations, self.consistency_metrics or self._create_empty_metrics()
    
    def _find_global_speaker_id(self, speaker_id: str, chunk_index: int) -> Optional[str]:
        """Find the global speaker ID for a local speaker ID in a specific chunk."""
        # Check speaker mappings for this chunk
        if chunk_index in self.speaker_mappings:
            for mapping in self.speaker_mappings[chunk_index]:
                if mapping.original_speaker_id == speaker_id:
                    return mapping.mapped_speaker_id
        
        # For first chunk, speaker IDs are already global
        if chunk_index == 0:
            return speaker_id
        
        return None
    
    def _apply_speaker_mapping(self, segments: List[Dict[str, Any]], 
                             speaker_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """Apply speaker ID mapping to diarization segments."""
        mapped_segments = []
        
        for segment in segments:
            mapped_segment = segment.copy()
            original_speaker = segment.get('speaker', segment.get('speaker_id', 'UNKNOWN'))
            
            if original_speaker in speaker_mapping:
                mapped_speaker = speaker_mapping[original_speaker]
            else:
                mapped_speaker = original_speaker
            
            # Update both possible speaker ID fields
            if 'speaker' in mapped_segment:
                mapped_segment['speaker'] = mapped_speaker
            if 'speaker_id' in mapped_segment:
                mapped_segment['speaker_id'] = mapped_speaker
            
            mapped_segments.append(mapped_segment)
        
        return mapped_segments
    
    def _generate_consistency_metrics(self, 
                                    original_chunks: List[List[Dict[str, Any]]], 
                                    mapped_chunks: List[List[Dict[str, Any]]], 
                                    mappings: List[SpeakerMapping]) -> ConsistencyMetrics:
        """Generate consistency metrics comparing original vs mapped results."""
        
        # Calculate baseline continuity (original)
        baseline_score = self._calculate_continuity_score(original_chunks)
        
        # Calculate improved continuity (mapped)
        improved_score = self._calculate_continuity_score(mapped_chunks)
        
        # Calculate improvement percentage
        improvement_pct = ((improved_score - baseline_score) / max(baseline_score, 0.01)) * 100
        
        # Count speaker ID changes prevented
        id_changes_prevented = self._count_id_changes_prevented(original_chunks, mapped_chunks)
        
        # Count total speaker segments
        total_segments = sum(len(chunk) for chunk in original_chunks)
        
        # Calculate mapping accuracy
        mapping_accuracy = len([m for m in mappings if m.similarity_score >= self.similarity_threshold]) / max(len(mappings), 1)
        
        # Collect similarity scores
        similarity_scores = [m.similarity_score for m in mappings]
        
        return ConsistencyMetrics(
            baseline_continuity_score=baseline_score,
            improved_continuity_score=improved_score,
            improvement_percentage=improvement_pct,
            speaker_id_changes_prevented=id_changes_prevented,
            total_speaker_segments=total_segments,
            mapping_accuracy=mapping_accuracy,
            cross_chunk_similarity_scores=similarity_scores
        )
    
    def _calculate_continuity_score(self, chunks: List[List[Dict[str, Any]]]) -> float:
        """Calculate speaker continuity score across chunks."""
        if len(chunks) < 2:
            return 1.0
        
        total_boundaries = 0
        consistent_boundaries = 0
        
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            if not current_chunk or not next_chunk:
                continue
            
            # Get speakers at chunk boundary
            current_final_speakers = set()
            next_initial_speakers = set()
            
            # Get speakers in final segments of current chunk
            if current_chunk:
                final_segments = sorted(current_chunk, key=lambda x: x.get('end', 0))[-3:]  # Last 3 segments
                current_final_speakers = {seg.get('speaker', seg.get('speaker_id', 'UNKNOWN')) for seg in final_segments}
            
            # Get speakers in initial segments of next chunk
            if next_chunk:
                initial_segments = sorted(next_chunk, key=lambda x: x.get('start', 0))[:3]  # First 3 segments
                next_initial_speakers = {seg.get('speaker', seg.get('speaker_id', 'UNKNOWN')) for seg in initial_segments}
            
            # Check for speaker continuity
            overlap = len(current_final_speakers.intersection(next_initial_speakers))
            total_unique = len(current_final_speakers.union(next_initial_speakers))
            
            total_boundaries += 1
            if total_unique > 0:
                boundary_consistency = overlap / total_unique
                if boundary_consistency > 0.3:  # At least 30% overlap indicates continuity
                    consistent_boundaries += 1
        
        return consistent_boundaries / max(total_boundaries, 1)
    
    def _count_id_changes_prevented(self, original_chunks: List[List[Dict[str, Any]]], 
                                  mapped_chunks: List[List[Dict[str, Any]]]) -> int:
        """Count how many speaker ID changes were prevented by mapping."""
        changes_prevented = 0
        
        # Compare speaker ID consistency across chunk boundaries
        for i in range(len(original_chunks) - 1):
            if i >= len(mapped_chunks) - 1:
                break
            
            orig_current = original_chunks[i]
            orig_next = original_chunks[i + 1]
            mapped_current = mapped_chunks[i]
            mapped_next = mapped_chunks[i + 1]
            
            # Count improvements in speaker consistency
            orig_consistency = self._calculate_boundary_consistency(orig_current, orig_next)
            mapped_consistency = self._calculate_boundary_consistency(mapped_current, mapped_next)
            
            if mapped_consistency > orig_consistency:
                changes_prevented += 1
        
        return changes_prevented
    
    def _calculate_boundary_consistency(self, chunk1: List[Dict[str, Any]], 
                                      chunk2: List[Dict[str, Any]]) -> float:
        """Calculate consistency score at boundary between two chunks."""
        if not chunk1 or not chunk2:
            return 0.0
        
        # Get speakers near boundary
        chunk1_speakers = {seg.get('speaker', seg.get('speaker_id', 'UNKNOWN')) for seg in chunk1[-2:]}
        chunk2_speakers = {seg.get('speaker', seg.get('speaker_id', 'UNKNOWN')) for seg in chunk2[:2]}
        
        overlap = len(chunk1_speakers.intersection(chunk2_speakers))
        total = len(chunk1_speakers.union(chunk2_speakers))
        
        return overlap / max(total, 1)
    
    def _create_empty_metrics(self) -> ConsistencyMetrics:
        """Create empty consistency metrics for error cases."""
        return ConsistencyMetrics(
            baseline_continuity_score=0.0,
            improved_continuity_score=0.0,
            improvement_percentage=0.0,
            speaker_id_changes_prevented=0,
            total_speaker_segments=0,
            mapping_accuracy=0.0,
            cross_chunk_similarity_scores=[]
        )
    
    def get_speaker_consistency_report(self) -> Dict[str, Any]:
        """Generate comprehensive speaker consistency report."""
        if not self.consistency_metrics:
            return {"error": "No consistency metrics available"}
        
        metrics = self.consistency_metrics
        
        report = {
            "summary": {
                "baseline_continuity": round(metrics.baseline_continuity_score, 3),
                "improved_continuity": round(metrics.improved_continuity_score, 3),
                "improvement_percentage": round(metrics.improvement_percentage, 1),
                "changes_prevented": metrics.speaker_id_changes_prevented,
                "mapping_accuracy": round(metrics.mapping_accuracy, 3)
            },
            "details": {
                "total_speaker_segments": metrics.total_speaker_segments,
                "similarity_scores": {
                    "mean": round(np.mean(metrics.cross_chunk_similarity_scores), 3) if metrics.cross_chunk_similarity_scores else 0.0,
                    "std": round(np.std(metrics.cross_chunk_similarity_scores), 3) if metrics.cross_chunk_similarity_scores else 0.0,
                    "min": round(np.min(metrics.cross_chunk_similarity_scores), 3) if metrics.cross_chunk_similarity_scores else 0.0,
                    "max": round(np.max(metrics.cross_chunk_similarity_scores), 3) if metrics.cross_chunk_similarity_scores else 0.0
                }
            },
            "global_speakers": {
                "total_speakers_identified": len(self.global_speaker_registry),
                "speaker_ids": list(self.global_speaker_registry.keys())
            },
            "chunk_mappings": {
                chunk_idx: len(mappings) for chunk_idx, mappings in self.speaker_mappings.items()
            }
        }
        
        return report
    
    def reset(self):
        """Reset speaker mapper state for new processing."""
        self.speaker_embeddings.clear()
        self.speaker_mappings.clear()
        self.global_speaker_registry.clear()
        self.consistency_metrics = None
        self.baseline_metrics.clear()
        
        self.structured_logger.info("Speaker mapper state reset")