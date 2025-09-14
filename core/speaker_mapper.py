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
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm
import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
import tempfile

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

@dataclass
class SpeakerEmbedding:
    """Speaker acoustic embedding with metadata"""
    speaker_id: str
    chunk_index: int
    embedding_vector: np.ndarray
    duration: float
    confidence: float
    segment_times: List[Tuple[float, float]]  # List of (start, end) times
    
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
                 embedding_dim: int = 128,
                 min_segment_duration: float = 1.0,
                 cache_embeddings: bool = True,
                 enable_metrics: bool = True):
        """
        Initialize speaker mapper with configuration parameters.
        
        Args:
            similarity_threshold: Minimum similarity for speaker matching
            embedding_dim: Dimensionality of speaker embeddings
            min_segment_duration: Minimum duration for embedding extraction
            cache_embeddings: Whether to cache computed embeddings
            enable_metrics: Whether to generate consistency metrics
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self.min_segment_duration = min_segment_duration
        self.cache_embeddings = cache_embeddings
        self.enable_metrics = enable_metrics
        
        # Initialize system components
        self.structured_logger = StructuredLogger("speaker_mapper")
        self.cache_manager = get_cache_manager() if cache_embeddings else None
        self.deterministic_processor = get_deterministic_processor()
        
        # Speaker tracking across chunks
        self.speaker_embeddings: Dict[int, List[SpeakerEmbedding]] = {}  # chunk_index -> embeddings
        self.speaker_mappings: Dict[int, List[SpeakerMapping]] = {}      # chunk_index -> mappings
        self.global_speaker_registry: Dict[str, SpeakerEmbedding] = {}   # global_id -> representative embedding
        
        # Metrics tracking
        self.consistency_metrics: Optional[ConsistencyMetrics] = None
        self.baseline_metrics: Dict[str, float] = {}
        
        # Algorithm parameters
        self.mfcc_features = 13
        self.spectral_features = 7
        self.prosodic_features = 5
        self.delta_features = True
        
        self.structured_logger.info("Speaker mapper initialized", 
                                  context={
                                      'similarity_threshold': similarity_threshold,
                                      'embedding_dim': embedding_dim,
                                      'cache_enabled': cache_embeddings
                                  })
    
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
                    audio_data, sample_rate, segments, speaker_id, chunk_index
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
            
            # Extract acoustic features
            features = self._extract_acoustic_features(segment_audio, sample_rate)
            if features is not None:
                all_features.append(features)
                total_duration += (end_time - start_time)
                confidence_scores.append(segment.get('confidence', 0.8))
        
        if not all_features:
            return None
        
        # Aggregate features across segments
        try:
            # Stack features and compute statistics
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
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.8
            
            return SpeakerEmbedding(
                speaker_id=speaker_id,
                chunk_index=chunk_index,
                embedding_vector=embedding_vector,
                duration=total_duration,
                confidence=avg_confidence,
                segment_times=segment_times
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
            f0 = librosa.yin(audio_segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
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
            self.structured_logger.warning(f"Feature extraction failed: {e}")
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
            
            # Apply Hungarian algorithm
            assignments = self.apply_hungarian_assignment(similarity_matrix)
            
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