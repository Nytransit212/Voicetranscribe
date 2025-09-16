"""
Global Speaker Linking Engine for Long-Horizon Speaker Tracking

This module establishes consistent global speaker identities across multi-hour sessions
by clustering speaker embeddings from all processed segments. It uses information 
criterion-based clustering with configurable parameters to avoid over-merging while
maintaining speaker consistency across temporal boundaries.

Key Features:
- Per-turn speaker embedding extraction from stable segments
- Agglomerative clustering with automatic K selection using information criteria
- Global speaker ID assignment with consistency scoring  
- Turn-level embedding aggregation for robust representation
- Integration with existing ECAPA-TDNN embedding pipeline
- Cluster quality validation and degenerate cluster detection

Process:
1. Extract embeddings from stable turns after overlap fusion
2. Aggregate per-speaker embeddings across all chunks/stems
3. Apply hierarchical clustering with information criterion stopping
4. Validate cluster quality and assign global speaker IDs
5. Compute speaker consistency metrics across session

Author: Advanced Ensemble Transcription System
"""

import numpy as np
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import warnings

from core.overlap_fusion import OverlapFusionResult, FusedTranscriptSegment, WordAlignment
from core.speaker_mapper import SpeakerMapper, SpeakerEmbedding, ECAPATDNNModel
from utils.embedding_cache import get_embedding_cache, CacheEntry
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import trace_stage, track_cost
from utils.intelligent_cache import cached_operation

@dataclass 
class TurnEmbedding:
    """Speaker embedding for a stable turn with metadata"""
    session_id: str
    chunk_id: str
    stem_id: str
    local_speaker_id: str
    
    # Turn information
    turn_start: float
    turn_end: float
    turn_duration: float
    word_count: int
    
    # Embedding data
    embedding: np.ndarray
    embedding_confidence: float = 0.0
    
    # Quality metrics
    stability_score: float = 0.0  # How stable/consistent the turn is
    acoustic_quality: float = 0.0  # Signal quality metrics
    
    # Aggregation information
    segments_aggregated: int = 1
    source_segments: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.embedding) == 0:
            raise ValueError("Embedding cannot be empty")

@dataclass
class SpeakerCluster:
    """Global speaker cluster with aggregated embeddings and metadata"""
    global_speaker_id: str
    
    # Cluster embeddings and statistics  
    centroid_embedding: np.ndarray
    member_embeddings: List[TurnEmbedding] = field(default_factory=list)
    
    # Temporal information
    total_duration: float = 0.0
    first_appearance: float = float('inf')
    last_appearance: float = 0.0
    turns_count: int = 0
    
    # Quality metrics
    intra_cluster_variance: float = 0.0
    average_confidence: float = 0.0
    temporal_consistency: float = 0.0  # How consistent across time
    
    # Cluster validation metrics
    silhouette_score: float = 0.0
    cluster_cohesion: float = 0.0
    cluster_separation: float = 0.0
    
    # Metadata
    source_chunks: Set[str] = field(default_factory=set)
    source_stems: Set[str] = field(default_factory=set)
    local_speaker_ids: Set[str] = field(default_factory=set)
    
    def add_turn_embedding(self, turn_embedding: TurnEmbedding):
        """Add turn embedding to cluster and update statistics"""
        self.member_embeddings.append(turn_embedding)
        self.turns_count = len(self.member_embeddings)
        
        # Update temporal information
        self.total_duration += turn_embedding.turn_duration
        self.first_appearance = min(self.first_appearance, turn_embedding.turn_start)
        self.last_appearance = max(self.last_appearance, turn_embedding.turn_end)
        
        # Update source tracking
        self.source_chunks.add(turn_embedding.chunk_id)
        self.source_stems.add(turn_embedding.stem_id)
        self.local_speaker_ids.add(turn_embedding.local_speaker_id)
        
        # Recompute centroid
        self._recompute_centroid()
    
    def _recompute_centroid(self):
        """Recompute centroid embedding from all member embeddings"""
        if not self.member_embeddings:
            return
        
        # Weighted average by confidence and duration
        embeddings = []
        weights = []
        
        for turn in self.member_embeddings:
            embeddings.append(turn.embedding)
            # Weight by confidence and duration, but cap to avoid single long segment dominance
            weight = turn.embedding_confidence * min(turn.turn_duration, 30.0)  # Cap at 30s
            weights.append(max(weight, 0.1))  # Minimum weight to avoid zero weights
        
        embeddings = np.array(embeddings)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        self.centroid_embedding = np.average(embeddings, axis=0, weights=weights)
        
        # Update quality metrics
        self._compute_cluster_quality()
    
    def _compute_cluster_quality(self):
        """Compute cluster quality metrics"""
        if len(self.member_embeddings) < 2:
            self.intra_cluster_variance = 0.0
            self.average_confidence = self.member_embeddings[0].embedding_confidence if self.member_embeddings else 0.0
            return
        
        # Compute intra-cluster variance
        distances = []
        confidences = []
        
        for turn in self.member_embeddings:
            distance = cosine(self.centroid_embedding, turn.embedding)
            distances.append(distance)
            confidences.append(turn.embedding_confidence)
        
        self.intra_cluster_variance = np.var(distances)
        self.average_confidence = np.mean(confidences)
        
        # Compute temporal consistency (how spread out over time)
        if len(self.member_embeddings) > 1:
            time_spans = [turn.turn_start for turn in self.member_embeddings]
            time_variance = np.var(time_spans)
            session_duration = self.last_appearance - self.first_appearance
            self.temporal_consistency = 1.0 / (1.0 + time_variance / max(session_duration, 1.0))
        else:
            self.temporal_consistency = 1.0

@dataclass
class ClusteringResult:
    """Result of global speaker clustering"""
    session_id: str
    clusters: List[SpeakerCluster]
    
    # Input data
    total_turns_processed: int
    unique_local_speakers: int
    
    # Clustering parameters used
    clustering_method: str
    optimal_k: int
    information_criterion: str
    
    # Quality metrics
    overall_silhouette_score: float = 0.0
    calinski_harabasz_score: float = 0.0
    clustering_confidence: float = 0.0
    
    # Validation results
    cluster_quality_passed: bool = True
    quality_issues: List[str] = field(default_factory=list)
    
    # Timing
    processing_time: float = 0.0
    
    # Mapping from local to global speaker IDs
    local_to_global_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class InformationCriterionSelector:
    """Information criterion-based cluster number selection"""
    
    def __init__(self,
                 methods: List[str] = ['silhouette', 'calinski_harabasz', 'elbow'],
                 min_clusters: int = 2,
                 max_clusters: int = 15,
                 elbow_threshold: float = 0.1):
        """
        Initialize information criterion selector
        
        Args:
            methods: List of methods to use for K selection
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            elbow_threshold: Threshold for elbow method
        """
        self.methods = methods
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.elbow_threshold = elbow_threshold
        
        self.logger = create_enhanced_logger("information_criterion_selector")
    
    def select_optimal_k(self, embeddings: np.ndarray, distance_matrix: Optional[np.ndarray] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Select optimal number of clusters using multiple information criteria
        
        Args:
            embeddings: Speaker embeddings matrix
            distance_matrix: Precomputed distance matrix (optional)
            
        Returns:
            Tuple of optimal K and selection metrics
        """
        n_samples = embeddings.shape[0]
        
        # Adjust cluster range based on data size
        max_k = min(self.max_clusters, n_samples // 2, 15)
        min_k = min(self.min_clusters, max_k - 1)
        
        if max_k <= min_k:
            return min_k, {'method': 'forced', 'reason': 'insufficient_data'}
        
        k_range = range(min_k, max_k + 1)
        
        # Compute scores for each K
        scores = {
            'silhouette': [],
            'calinski_harabasz': [], 
            'elbow': [],
            'k_values': list(k_range)
        }
        
        clustering_results = {}
        
        for k in k_range:
            try:
                # Perform clustering
                clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
                cluster_labels = clustering.fit_predict(embeddings)
                
                clustering_results[k] = cluster_labels
                
                # Silhouette score
                if 'silhouette' in self.methods and len(set(cluster_labels)) > 1:
                    try:
                        silhouette = silhouette_score(embeddings, cluster_labels)
                        scores['silhouette'].append(silhouette)
                    except:
                        scores['silhouette'].append(-1.0)
                else:
                    scores['silhouette'].append(-1.0)
                
                # Calinski-Harabasz score
                if 'calinski_harabasz' in self.methods and len(set(cluster_labels)) > 1:
                    try:
                        ch_score = calinski_harabasz_score(embeddings, cluster_labels)
                        scores['calinski_harabasz'].append(ch_score)
                    except:
                        scores['calinski_harabasz'].append(0.0)
                else:
                    scores['calinski_harabasz'].append(0.0)
                
                # Within-cluster sum of squares for elbow method
                if 'elbow' in self.methods:
                    wcss = self._compute_wcss(embeddings, cluster_labels)
                    scores['elbow'].append(wcss)
                
            except Exception as e:
                self.logger.warning(f"Failed to compute scores for k={k}: {e}")
                scores['silhouette'].append(-1.0)
                scores['calinski_harabasz'].append(0.0)
                scores['elbow'].append(float('inf'))
        
        # Select optimal K based on methods
        optimal_k_candidates = {}
        
        # Silhouette method (maximize)
        if 'silhouette' in self.methods and scores['silhouette']:
            silhouette_scores = np.array(scores['silhouette'])
            if np.max(silhouette_scores) > 0:
                optimal_k_candidates['silhouette'] = k_range[np.argmax(silhouette_scores)]
        
        # Calinski-Harabasz method (maximize)  
        if 'calinski_harabasz' in self.methods and scores['calinski_harabasz']:
            ch_scores = np.array(scores['calinski_harabasz'])
            if np.max(ch_scores) > 0:
                optimal_k_candidates['calinski_harabasz'] = k_range[np.argmax(ch_scores)]
        
        # Elbow method
        if 'elbow' in self.methods and scores['elbow']:
            elbow_k = self._find_elbow_point(scores['elbow'], k_range)
            if elbow_k is not None:
                optimal_k_candidates['elbow'] = elbow_k
        
        # Consensus selection
        if optimal_k_candidates:
            # Use majority vote or fall back to silhouette
            k_votes = Counter(optimal_k_candidates.values())
            optimal_k = k_votes.most_common(1)[0][0]
            
            selection_info = {
                'method': 'consensus',
                'candidates': optimal_k_candidates,
                'votes': dict(k_votes),
                'scores': scores
            }
        else:
            # Fallback to middle value
            optimal_k = (min_k + max_k) // 2
            selection_info = {
                'method': 'fallback',
                'reason': 'no_valid_scores',
                'scores': scores
            }
        
        self.logger.info(f"Selected optimal K={optimal_k}",
                        context={
                            'method': selection_info['method'],
                            'candidates': optimal_k_candidates,
                            'n_samples': n_samples,
                            'k_range': f"{min_k}-{max_k}"
                        })
        
        return optimal_k, selection_info
    
    def _compute_wcss(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Compute within-cluster sum of squares"""
        wcss = 0.0
        
        for cluster_id in set(cluster_labels):
            cluster_points = embeddings[cluster_labels == cluster_id]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                wcss += np.sum((cluster_points - centroid) ** 2)
        
        return wcss
    
    def _find_elbow_point(self, wcss_values: List[float], k_values: List[int]) -> Optional[int]:
        """Find elbow point in WCSS curve"""
        if len(wcss_values) < 3:
            return None
        
        wcss_array = np.array(wcss_values)
        k_array = np.array(k_values)
        
        # Compute second derivative to find elbow
        first_derivative = np.diff(wcss_array)
        second_derivative = np.diff(first_derivative)
        
        # Find point where second derivative is maximum (most curved)
        if len(second_derivative) > 0:
            elbow_idx = np.argmax(second_derivative) + 1  # +1 because of diff offset
            if elbow_idx < len(k_values):
                return k_values[elbow_idx]
        
        return None

class GlobalSpeakerLinker:
    """
    Global speaker linking engine for long-horizon speaker tracking
    
    Establishes consistent speaker identities across entire sessions by:
    1. Extracting embeddings from stable turns after overlap fusion
    2. Clustering embeddings using information criteria for optimal K
    3. Assigning global speaker IDs with quality validation
    4. Computing consistency metrics across temporal boundaries
    """
    
    def __init__(self,
                 speaker_mapper: Optional[SpeakerMapper] = None,
                 min_turn_duration: float = 0.5,  # Minimum turn duration to consider
                 max_turn_duration: float = 60.0,  # Maximum turn duration for single embedding
                 embedding_aggregation_method: str = "weighted_average",  # or "median", "max_confidence"
                 clustering_method: str = "hierarchical",  # or "kmeans", "spectral"
                 information_criterion: List[str] = None,
                 cluster_margin: float = 0.15,  # Margin to avoid over-merging
                 min_cluster_size: int = 2,  # Minimum turns per cluster
                 enable_caching: bool = True):
        """
        Initialize global speaker linker
        
        Args:
            speaker_mapper: Speaker mapper for embedding extraction
            min_turn_duration: Minimum turn duration to process
            max_turn_duration: Maximum turn duration for single embedding  
            embedding_aggregation_method: Method to aggregate turn embeddings
            clustering_method: Clustering algorithm to use
            information_criterion: List of criteria for K selection
            cluster_margin: Similarity margin to avoid over-merging
            min_cluster_size: Minimum number of turns per cluster
            enable_caching: Enable embedding caching
        """
        self.speaker_mapper = speaker_mapper or SpeakerMapper()
        self.min_turn_duration = min_turn_duration
        self.max_turn_duration = max_turn_duration
        self.embedding_aggregation_method = embedding_aggregation_method
        self.clustering_method = clustering_method
        self.cluster_margin = cluster_margin
        self.min_cluster_size = min_cluster_size
        self.enable_caching = enable_caching
        
        # Information criterion setup
        self.information_criterion = information_criterion or ['silhouette', 'calinski_harabasz', 'elbow']
        self.ic_selector = InformationCriterionSelector(methods=self.information_criterion)
        
        # Caching
        if self.enable_caching:
            self.embedding_cache = get_embedding_cache()
        else:
            self.embedding_cache = None
        
        # Logging
        self.logger = create_enhanced_logger("global_speaker_linker")
        
        self.logger.info("Global speaker linker initialized",
                        context={
                            'min_turn_duration': min_turn_duration,
                            'clustering_method': clustering_method,
                            'information_criteria': self.information_criterion,
                            'caching_enabled': enable_caching
                        })
    
    @trace_stage("extract_turn_embeddings")
    def extract_turn_embeddings(self, 
                               fusion_result: OverlapFusionResult,
                               session_id: str,
                               audio_path: str) -> List[TurnEmbedding]:
        """
        Extract speaker embeddings from stable turns in fusion result
        
        Args:
            fusion_result: Result from overlap fusion processing
            session_id: Session identifier
            audio_path: Path to audio file for embedding extraction
            
        Returns:
            List of turn embeddings with metadata
        """
        self.logger.info(f"Extracting turn embeddings from {len(fusion_result.fused_segments)} segments")
        
        turn_embeddings = []
        
        for i, segment in enumerate(fusion_result.fused_segments):
            # Skip short segments
            if segment.duration < self.min_turn_duration:
                continue
            
            # Skip segments with overlaps for stable embedding extraction
            if segment.has_overlap:
                continue
            
            # Check cache first
            cache_key = f"{session_id}:segment_{i}:main:{segment.speaker_id}"
            cached_embedding = None
            
            if self.embedding_cache:
                cached_entry = self.embedding_cache.get(session_id, f"segment_{i}", "main", segment.speaker_id)
                if cached_entry:
                    cached_embedding = cached_entry.embedding
            
            if cached_embedding is not None:
                # Use cached embedding
                turn_embedding = TurnEmbedding(
                    session_id=session_id,
                    chunk_id=f"segment_{i}",
                    stem_id="main",
                    local_speaker_id=segment.speaker_id,
                    turn_start=segment.start_time,
                    turn_end=segment.end_time,
                    turn_duration=segment.duration,
                    word_count=len(segment.words),
                    embedding=cached_embedding,
                    embedding_confidence=segment.confidence,
                    stability_score=self._compute_stability_score(segment),
                    acoustic_quality=segment.confidence
                )
                
                turn_embeddings.append(turn_embedding)
                self.logger.debug(f"Used cached embedding for segment {i}")
                
            else:
                # Extract new embedding
                try:
                    # Handle long segments by splitting if necessary
                    if segment.duration > self.max_turn_duration:
                        sub_embeddings = self._extract_multi_segment_embedding(
                            segment, session_id, f"segment_{i}", audio_path
                        )
                        turn_embeddings.extend(sub_embeddings)
                    else:
                        embedding = self._extract_single_turn_embedding(
                            segment, audio_path
                        )
                        
                        if embedding is not None:
                            turn_embedding = TurnEmbedding(
                                session_id=session_id,
                                chunk_id=f"segment_{i}",
                                stem_id="main",
                                local_speaker_id=segment.speaker_id,
                                turn_start=segment.start_time,
                                turn_end=segment.end_time,
                                turn_duration=segment.duration,
                                word_count=len(segment.words),
                                embedding=embedding,
                                embedding_confidence=segment.confidence,
                                stability_score=self._compute_stability_score(segment),
                                acoustic_quality=segment.confidence
                            )
                            
                            turn_embeddings.append(turn_embedding)
                            
                            # Cache the embedding
                            if self.embedding_cache:
                                self.embedding_cache.put(
                                    session_id, f"segment_{i}", "main", segment.speaker_id,
                                    embedding, segment.duration, segment.confidence
                                )
                
                except Exception as e:
                    self.logger.warning(f"Failed to extract embedding for segment {i}: {e}")
                    continue
        
        self.logger.info(f"Extracted {len(turn_embeddings)} turn embeddings from {len(fusion_result.fused_segments)} segments")
        return turn_embeddings
    
    def _extract_single_turn_embedding(self, segment: FusedTranscriptSegment, audio_path: str) -> Optional[np.ndarray]:
        """Extract embedding from a single turn segment"""
        try:
            # Use speaker mapper's ECAPA-TDNN model for embedding extraction
            if hasattr(self.speaker_mapper, 'ecapa_model') and self.speaker_mapper.ecapa_model:
                # Extract audio features for this segment
                import librosa
                
                # Load audio segment
                y, sr = librosa.load(audio_path, sr=16000, 
                                   offset=segment.start_time, 
                                   duration=segment.duration)
                
                if len(y) < 0.1 * sr:  # Skip very short segments
                    return None
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
                
                # Convert to torch tensor and get embedding
                import torch
                mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0)  # Add batch dimension
                
                with torch.no_grad():
                    embedding = self.speaker_mapper.ecapa_model(mfcc_tensor)
                    embedding = embedding.squeeze().numpy()
                
                return embedding
            
            else:
                # Fallback to simple spectral features
                return self._extract_simple_embedding(segment, audio_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract ECAPA embedding: {e}")
            return self._extract_simple_embedding(segment, audio_path)
    
    def _extract_simple_embedding(self, segment: FusedTranscriptSegment, audio_path: str) -> Optional[np.ndarray]:
        """Simple embedding extraction fallback"""
        try:
            import librosa
            
            # Load audio segment
            y, sr = librosa.load(audio_path, sr=16000, 
                               offset=segment.start_time, 
                               duration=segment.duration)
            
            if len(y) < 0.1 * sr:
                return None
            
            # Extract spectral features as simple embedding
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Aggregate features
            features = [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                [np.mean(spectral_centroid), np.std(spectral_centroid)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
                [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)]
            ]
            
            # Flatten and concatenate
            embedding = np.concatenate([f.flatten() if hasattr(f, 'flatten') else f for f in features])
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to extract simple embedding: {e}")
            return None
    
    def _extract_multi_segment_embedding(self, 
                                       segment: FusedTranscriptSegment, 
                                       session_id: str, 
                                       chunk_id: str, 
                                       audio_path: str) -> List[TurnEmbedding]:
        """Extract embeddings from long segments by splitting them"""
        sub_embeddings = []
        
        # Split into smaller chunks
        chunk_duration = self.max_turn_duration
        chunks = int(np.ceil(segment.duration / chunk_duration))
        
        for i in range(chunks):
            start_offset = i * chunk_duration
            end_offset = min((i + 1) * chunk_duration, segment.duration)
            
            if end_offset - start_offset < self.min_turn_duration:
                continue
            
            # Create subsegment
            sub_start = segment.start_time + start_offset
            sub_end = segment.start_time + end_offset
            sub_duration = sub_end - sub_start
            
            # Estimate word count for this subsegment
            words_in_subsegment = [w for w in segment.words 
                                 if sub_start <= w.start_time < sub_end]
            
            # Create temporary segment for embedding extraction
            temp_segment = FusedTranscriptSegment(
                start_time=sub_start,
                end_time=sub_end,
                duration=sub_duration,
                text=" ".join(w.word for w in words_in_subsegment),
                speaker_id=segment.speaker_id,
                confidence=segment.confidence,
                words=words_in_subsegment
            )
            
            # Extract embedding
            embedding = self._extract_single_turn_embedding(temp_segment, audio_path)
            
            if embedding is not None:
                sub_embedding = TurnEmbedding(
                    session_id=session_id,
                    chunk_id=f"{chunk_id}_sub_{i}",
                    stem_id="main",
                    local_speaker_id=segment.speaker_id,
                    turn_start=sub_start,
                    turn_end=sub_end,
                    turn_duration=sub_duration,
                    word_count=len(words_in_subsegment),
                    embedding=embedding,
                    embedding_confidence=segment.confidence,
                    stability_score=self._compute_stability_score(temp_segment),
                    acoustic_quality=segment.confidence,
                    source_segments=[{
                        'original_start': segment.start_time,
                        'original_end': segment.end_time,
                        'sub_index': i
                    }]
                )
                
                sub_embeddings.append(sub_embedding)
        
        return sub_embeddings
    
    def _compute_stability_score(self, segment: FusedTranscriptSegment) -> float:
        """Compute stability score for a segment based on confidence and consistency"""
        base_score = segment.confidence
        
        # Penalize very short segments
        duration_factor = min(segment.duration / self.min_turn_duration, 1.0)
        
        # Boost for longer, consistent segments
        consistency_boost = 1.0
        if segment.words:
            word_confidences = [getattr(w, 'acoustic_confidence', 0.8) for w in segment.words]
            if word_confidences:
                consistency_boost = 1.0 - np.std(word_confidences)
        
        # Penalize overlap regions
        overlap_penalty = 1.0
        if segment.has_overlap:
            overlap_penalty = 0.8
        
        stability = base_score * duration_factor * consistency_boost * overlap_penalty
        return np.clip(stability, 0.0, 1.0)
    
    @trace_stage("cluster_speaker_embeddings") 
    def cluster_speaker_embeddings(self, 
                                 turn_embeddings: List[TurnEmbedding],
                                 session_id: str,
                                 user_hint_k: Optional[int] = None) -> ClusteringResult:
        """
        Cluster speaker embeddings to establish global speaker identities
        
        Args:
            turn_embeddings: List of turn embeddings to cluster
            session_id: Session identifier
            user_hint_k: User-provided hint for number of speakers
            
        Returns:
            Clustering result with global speaker assignments
        """
        start_time = time.time()
        
        if len(turn_embeddings) < 2:
            self.logger.warning("Insufficient turn embeddings for clustering")
            return self._create_single_cluster_result(turn_embeddings, session_id, time.time() - start_time)
        
        self.logger.info(f"Clustering {len(turn_embeddings)} turn embeddings")
        
        # Prepare embedding matrix
        embeddings_matrix = np.vstack([turn.embedding for turn in turn_embeddings])
        
        # Determine optimal number of clusters
        if user_hint_k is not None:
            optimal_k = max(1, min(user_hint_k, len(turn_embeddings) // 2))
            selection_info = {'method': 'user_hint', 'k': optimal_k}
        else:
            optimal_k, selection_info = self.ic_selector.select_optimal_k(embeddings_matrix)
        
        self.logger.info(f"Using K={optimal_k} clusters", context=selection_info)
        
        # Perform clustering
        try:
            if self.clustering_method == "hierarchical":
                clustering = AgglomerativeClustering(
                    n_clusters=optimal_k,
                    linkage='ward',
                    distance_threshold=None
                )
            else:
                # Fallback to hierarchical
                clustering = AgglomerativeClustering(n_clusters=optimal_k)
            
            cluster_labels = clustering.fit_predict(embeddings_matrix)
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return self._create_fallback_clustering_result(turn_embeddings, session_id, time.time() - start_time)
        
        # Build cluster objects
        clusters = self._build_speaker_clusters(turn_embeddings, cluster_labels, optimal_k)
        
        # Compute clustering quality metrics
        quality_metrics = self._compute_clustering_quality(embeddings_matrix, cluster_labels, clusters)
        
        # Validate clusters
        validation_result = self._validate_clusters(clusters, quality_metrics)
        
        # Create local-to-global mapping
        local_to_global = self._create_speaker_mapping(clusters)
        
        processing_time = time.time() - start_time
        
        result = ClusteringResult(
            session_id=session_id,
            clusters=clusters,
            total_turns_processed=len(turn_embeddings),
            unique_local_speakers=len(set(turn.local_speaker_id for turn in turn_embeddings)),
            clustering_method=self.clustering_method,
            optimal_k=optimal_k,
            information_criterion=str(selection_info),
            overall_silhouette_score=quality_metrics.get('silhouette_score', 0.0),
            calinski_harabasz_score=quality_metrics.get('calinski_harabasz_score', 0.0),
            clustering_confidence=quality_metrics.get('clustering_confidence', 0.0),
            cluster_quality_passed=validation_result['passed'],
            quality_issues=validation_result['issues'],
            processing_time=processing_time,
            local_to_global_mapping=local_to_global,
            metadata={
                'selection_info': selection_info,
                'quality_metrics': quality_metrics,
                'validation': validation_result
            }
        )
        
        self.logger.info(f"Clustering completed in {processing_time:.2f}s",
                        context={
                            'clusters': len(clusters),
                            'quality_passed': validation_result['passed'],
                            'silhouette_score': quality_metrics.get('silhouette_score', 0.0)
                        })
        
        return result
    
    def _build_speaker_clusters(self, 
                              turn_embeddings: List[TurnEmbedding], 
                              cluster_labels: np.ndarray, 
                              num_clusters: int) -> List[SpeakerCluster]:
        """Build speaker cluster objects from clustering results"""
        clusters = []
        
        for cluster_id in range(num_clusters):
            global_id = f"SPEAKER_{cluster_id + 1:02d}"
            
            # Get embeddings for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_turn_embeddings = [turn_embeddings[i] for i in range(len(turn_embeddings)) if cluster_mask[i]]
            
            if not cluster_turn_embeddings:
                continue
            
            # Create initial cluster
            cluster = SpeakerCluster(
                global_speaker_id=global_id,
                centroid_embedding=np.zeros_like(cluster_turn_embeddings[0].embedding)
            )
            
            # Add all turn embeddings
            for turn in cluster_turn_embeddings:
                cluster.add_turn_embedding(turn)
            
            clusters.append(cluster)
        
        return clusters
    
    def _compute_clustering_quality(self, 
                                  embeddings_matrix: np.ndarray, 
                                  cluster_labels: np.ndarray,
                                  clusters: List[SpeakerCluster]) -> Dict[str, Any]:
        """Compute comprehensive clustering quality metrics"""
        quality_metrics = {}
        
        try:
            # Silhouette score
            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(embeddings_matrix, cluster_labels)
                quality_metrics['silhouette_score'] = silhouette
            else:
                quality_metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz score
            if len(set(cluster_labels)) > 1:
                ch_score = calinski_harabasz_score(embeddings_matrix, cluster_labels)
                quality_metrics['calinski_harabasz_score'] = ch_score
            else:
                quality_metrics['calinski_harabasz_score'] = 0.0
            
            # Cluster-specific metrics
            cluster_qualities = []
            for cluster in clusters:
                cluster_qualities.append({
                    'intra_cluster_variance': cluster.intra_cluster_variance,
                    'temporal_consistency': cluster.temporal_consistency,
                    'average_confidence': cluster.average_confidence,
                    'turns_count': cluster.turns_count
                })
            
            quality_metrics['cluster_qualities'] = cluster_qualities
            
            # Overall clustering confidence
            avg_silhouette = quality_metrics['silhouette_score']
            avg_temporal_consistency = np.mean([c.temporal_consistency for c in clusters])
            avg_cluster_confidence = np.mean([c.average_confidence for c in clusters])
            
            clustering_confidence = (avg_silhouette + avg_temporal_consistency + avg_cluster_confidence) / 3.0
            quality_metrics['clustering_confidence'] = np.clip(clustering_confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Failed to compute some quality metrics: {e}")
            quality_metrics.setdefault('silhouette_score', 0.0)
            quality_metrics.setdefault('calinski_harabasz_score', 0.0)
            quality_metrics.setdefault('clustering_confidence', 0.5)
        
        return quality_metrics
    
    def _validate_clusters(self, clusters: List[SpeakerCluster], quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cluster quality and detect issues"""
        issues = []
        passed = True
        
        # Check for degenerate clusters (too few turns)
        small_clusters = [c for c in clusters if c.turns_count < self.min_cluster_size]
        if small_clusters:
            issues.append(f"Found {len(small_clusters)} clusters with fewer than {self.min_cluster_size} turns")
            if len(small_clusters) == len(clusters):
                passed = False
        
        # Check silhouette score
        silhouette_score = quality_metrics.get('silhouette_score', 0.0)
        if silhouette_score < 0.2:
            issues.append(f"Low silhouette score: {silhouette_score:.3f}")
            if silhouette_score < 0.0:
                passed = False
        
        # Check cluster separation
        if len(clusters) > 1:
            inter_cluster_distances = []
            for i, cluster_a in enumerate(clusters):
                for j, cluster_b in enumerate(clusters[i+1:], i+1):
                    distance = cosine(cluster_a.centroid_embedding, cluster_b.centroid_embedding)
                    inter_cluster_distances.append(distance)
            
            if inter_cluster_distances:
                min_separation = min(inter_cluster_distances)
                if min_separation < self.cluster_margin:
                    issues.append(f"Clusters too close: minimum separation {min_separation:.3f}")
        
        # Check temporal consistency
        avg_temporal_consistency = np.mean([c.temporal_consistency for c in clusters])
        if avg_temporal_consistency < 0.3:
            issues.append(f"Low temporal consistency: {avg_temporal_consistency:.3f}")
        
        return {
            'passed': passed,
            'issues': issues,
            'separation_scores': inter_cluster_distances if 'inter_cluster_distances' in locals() else [],
            'temporal_consistency': avg_temporal_consistency if 'avg_temporal_consistency' in locals() else 0.0
        }
    
    def _create_speaker_mapping(self, clusters: List[SpeakerCluster]) -> Dict[str, str]:
        """Create mapping from local speaker IDs to global speaker IDs"""
        local_to_global = {}
        
        for cluster in clusters:
            # Map all local speaker IDs in this cluster to the global ID
            for local_id in cluster.local_speaker_ids:
                local_to_global[local_id] = cluster.global_speaker_id
        
        return local_to_global
    
    def _create_single_cluster_result(self, turn_embeddings: List[TurnEmbedding], session_id: str, processing_time: float) -> ClusteringResult:
        """Create clustering result for single cluster case"""
        if not turn_embeddings:
            clusters = []
            local_to_global = {}
        else:
            # Create single cluster
            cluster = SpeakerCluster(
                global_speaker_id="SPEAKER_01",
                centroid_embedding=np.zeros_like(turn_embeddings[0].embedding)
            )
            
            for turn in turn_embeddings:
                cluster.add_turn_embedding(turn)
            
            clusters = [cluster]
            local_to_global = self._create_speaker_mapping(clusters)
        
        return ClusteringResult(
            session_id=session_id,
            clusters=clusters,
            total_turns_processed=len(turn_embeddings),
            unique_local_speakers=len(set(turn.local_speaker_id for turn in turn_embeddings)) if turn_embeddings else 0,
            clustering_method=self.clustering_method,
            optimal_k=1,
            information_criterion="single_cluster",
            processing_time=processing_time,
            local_to_global_mapping=local_to_global,
            metadata={'reason': 'insufficient_data'}
        )
    
    def _create_fallback_clustering_result(self, turn_embeddings: List[TurnEmbedding], session_id: str, processing_time: float) -> ClusteringResult:
        """Create fallback clustering result when clustering fails"""
        # Create one cluster per unique local speaker ID
        local_speaker_ids = list(set(turn.local_speaker_id for turn in turn_embeddings))
        clusters = []
        
        for i, local_id in enumerate(local_speaker_ids):
            global_id = f"SPEAKER_{i + 1:02d}"
            cluster = SpeakerCluster(
                global_speaker_id=global_id,
                centroid_embedding=np.zeros_like(turn_embeddings[0].embedding)
            )
            
            # Add turns for this local speaker
            for turn in turn_embeddings:
                if turn.local_speaker_id == local_id:
                    cluster.add_turn_embedding(turn)
            
            clusters.append(cluster)
        
        local_to_global = self._create_speaker_mapping(clusters)
        
        return ClusteringResult(
            session_id=session_id,
            clusters=clusters,
            total_turns_processed=len(turn_embeddings),
            unique_local_speakers=len(local_speaker_ids),
            clustering_method="fallback",
            optimal_k=len(clusters),
            information_criterion="fallback",
            cluster_quality_passed=False,
            quality_issues=["Clustering algorithm failed"],
            processing_time=processing_time,
            local_to_global_mapping=local_to_global,
            metadata={'reason': 'clustering_failed'}
        )
    
    def process_fusion_result(self, 
                            fusion_result: OverlapFusionResult, 
                            session_id: str,
                            audio_path: str,
                            user_hint_k: Optional[int] = None) -> ClusteringResult:
        """
        Complete pipeline: extract embeddings and perform clustering
        
        Args:
            fusion_result: Overlap fusion result to process
            session_id: Session identifier
            audio_path: Path to audio file
            user_hint_k: User hint for number of speakers
            
        Returns:
            Clustering result with global speaker assignments
        """
        # Extract turn embeddings
        turn_embeddings = self.extract_turn_embeddings(fusion_result, session_id, audio_path)
        
        if not turn_embeddings:
            self.logger.warning("No turn embeddings extracted")
            return self._create_single_cluster_result([], session_id, 0.0)
        
        # Perform clustering
        clustering_result = self.cluster_speaker_embeddings(turn_embeddings, session_id, user_hint_k)
        
        return clustering_result