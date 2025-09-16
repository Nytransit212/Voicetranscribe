"""
Thread-Safe Speaker Embedding Cache System

This module provides thread-safe caching for speaker embeddings to support 
long-horizon speaker tracking across multi-hour sessions. It includes:

1. In-memory LRU caching with automatic eviction
2. Persistent disk-based caching for cross-session optimization  
3. Thread-safe operations with proper locking
4. Memory management and cache statistics
5. Hierarchical caching by session, chunk, and stem

Key Features:
- Thread-safe operations using threading locks
- Configurable memory limits and TTL
- Disk persistence with compression and integrity checks
- Cache statistics and performance monitoring
- Automatic cleanup of stale cache entries

Author: Advanced Ensemble Transcription System
"""

import os
import json
import pickle
import gzip
import hashlib
import threading
import time
import tempfile
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import numpy as np

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.intelligent_cache import cached_operation

@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    embedding: np.ndarray
    timestamp: float
    access_count: int
    chunk_id: str
    stem_id: str
    speaker_id: str
    session_id: str
    
    # Metadata
    duration: float = 0.0
    confidence: float = 0.0
    embedding_dim: int = 0
    embedding_method: str = "ecapa_tdnn"
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.embedding_dim == 0 and self.embedding is not None:
            self.embedding_dim = len(self.embedding)

@dataclass
class CacheStats:
    """Cache statistics and performance metrics"""
    total_entries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_memory_mb: float = 0.0
    entries_evicted: int = 0
    disk_saves: int = 0
    disk_loads: int = 0
    
    # Session-specific stats
    sessions_active: int = 0
    average_embeddings_per_session: float = 0.0
    
    # Performance metrics
    average_lookup_time_ms: float = 0.0
    average_save_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0
    
    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB"""
        return self.total_memory_mb

class ThreadSafeEmbeddingCache:
    """
    Thread-safe LRU cache for speaker embeddings with disk persistence
    
    Provides hierarchical caching by session -> chunk -> stem -> speaker
    with automatic memory management and cross-session persistence.
    """
    
    def __init__(self,
                 max_memory_mb: float = 512.0,
                 max_entries: int = 10000,
                 ttl_seconds: float = 3600.0,  # 1 hour default TTL
                 enable_disk_cache: bool = True,
                 disk_cache_path: Optional[str] = None,
                 compression_level: int = 6,
                 enable_integrity_checks: bool = True):
        """
        Initialize thread-safe embedding cache
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_entries: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries
            enable_disk_cache: Enable persistent disk caching
            disk_cache_path: Path for disk cache storage
            compression_level: gzip compression level (0-9)
            enable_integrity_checks: Enable data integrity validation
        """
        self.max_memory_mb = max_memory_mb
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.enable_disk_cache = enable_disk_cache
        self.compression_level = compression_level
        self.enable_integrity_checks = enable_integrity_checks
        
        # Thread synchronization
        self._lock = threading.RLock()
        self._stats_lock = threading.Lock()
        
        # In-memory cache storage (LRU using OrderedDict)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Session-based organization for efficient cleanup
        self._session_entries: Dict[str, set] = defaultdict(set)
        
        # Cache statistics
        self._stats = CacheStats()
        
        # Disk cache setup
        if self.enable_disk_cache:
            if disk_cache_path:
                self.disk_cache_path = Path(disk_cache_path)
            else:
                self.disk_cache_path = Path(tempfile.gettempdir()) / "speaker_embedding_cache"
            
            self.disk_cache_path.mkdir(parents=True, exist_ok=True)
            self._disk_index_path = self.disk_cache_path / "cache_index.json"
            self._load_disk_index()
        else:
            self.disk_cache_path = None
            self._disk_index = {}
        
        # Performance tracking
        self._lookup_times: List[float] = []
        self._save_times: List[float] = []
        
        # Initialize structured logging
        self.logger = create_enhanced_logger("embedding_cache")
        
        self.logger.info("Embedding cache initialized",
                        context={
                            'max_memory_mb': max_memory_mb,
                            'max_entries': max_entries,
                            'ttl_seconds': ttl_seconds,
                            'disk_cache_enabled': enable_disk_cache,
                            'disk_cache_path': str(self.disk_cache_path) if self.disk_cache_path else None
                        })
    
    def _generate_cache_key(self, session_id: str, chunk_id: str, stem_id: str, speaker_id: str) -> str:
        """Generate hierarchical cache key"""
        return f"{session_id}:{chunk_id}:{stem_id}:{speaker_id}"
    
    def _generate_disk_key(self, cache_key: str) -> str:
        """Generate disk cache filename from cache key"""
        # Use hash to ensure valid filename and avoid path length issues
        hash_object = hashlib.sha256(cache_key.encode())
        return hash_object.hexdigest()[:32]
    
    def _calculate_entry_size_mb(self, entry: CacheEntry) -> float:
        """Calculate approximate memory size of cache entry in MB"""
        base_size = 200  # Base overhead in bytes
        embedding_size = entry.embedding.nbytes if entry.embedding is not None else 0
        metadata_size = len(str(entry.processing_metadata)) * 4  # Rough estimate
        
        total_bytes = base_size + embedding_size + metadata_size
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _update_memory_usage(self):
        """Update total memory usage statistics"""
        with self._stats_lock:
            total_mb = sum(self._calculate_entry_size_mb(entry) for entry in self._cache.values())
            self._stats.total_memory_mb = total_mb
            self._stats.total_entries = len(self._cache)
    
    def _evict_lru_entries(self):
        """Evict least recently used entries to stay within memory limits"""
        with self._lock:
            while (self._stats.total_memory_mb > self.max_memory_mb or 
                   len(self._cache) > self.max_entries) and self._cache:
                
                # Remove oldest entry (LRU)
                cache_key, entry = self._cache.popitem(last=False)
                
                # Update session tracking
                self._session_entries[entry.session_id].discard(cache_key)
                if not self._session_entries[entry.session_id]:
                    del self._session_entries[entry.session_id]
                
                # Save to disk if enabled before evicting
                if self.enable_disk_cache:
                    self._save_to_disk(cache_key, entry)
                
                with self._stats_lock:
                    self._stats.entries_evicted += 1
                
                self.logger.debug(f"Evicted cache entry: {cache_key}")
            
            self._update_memory_usage()
    
    def _cleanup_expired_entries(self):
        """Remove expired entries based on TTL"""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for cache_key, entry in self._cache.items():
                if current_time - entry.timestamp > self.ttl_seconds:
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                entry = self._cache.pop(cache_key)
                
                # Update session tracking
                self._session_entries[entry.session_id].discard(cache_key)
                if not self._session_entries[entry.session_id]:
                    del self._session_entries[entry.session_id]
                
                self.logger.debug(f"Expired cache entry: {cache_key}")
            
            if expired_keys:
                self._update_memory_usage()
                self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _load_disk_index(self):
        """Load disk cache index"""
        self._disk_index = {}
        
        if not self.enable_disk_cache or not self._disk_index_path.exists():
            return
        
        try:
            with open(self._disk_index_path, 'r') as f:
                self._disk_index = json.load(f)
            
            self.logger.info(f"Loaded disk cache index with {len(self._disk_index)} entries")
            
        except Exception as e:
            self.logger.warning(f"Failed to load disk cache index: {e}")
            self._disk_index = {}
    
    def _save_disk_index(self):
        """Save disk cache index"""
        if not self.enable_disk_cache:
            return
        
        try:
            with open(self._disk_index_path, 'w') as f:
                json.dump(self._disk_index, f, indent=2)
            
        except Exception as e:
            self.logger.warning(f"Failed to save disk cache index: {e}")
    
    def _save_to_disk(self, cache_key: str, entry: CacheEntry):
        """Save cache entry to disk with compression"""
        if not self.enable_disk_cache:
            return
        
        disk_key = self._generate_disk_key(cache_key)
        disk_file = self.disk_cache_path / f"{disk_key}.pkl.gz"
        
        try:
            start_time = time.time()
            
            # Prepare entry data for serialization
            entry_data = {
                'embedding': entry.embedding,
                'timestamp': entry.timestamp,
                'access_count': entry.access_count,
                'chunk_id': entry.chunk_id,
                'stem_id': entry.stem_id,
                'speaker_id': entry.speaker_id,
                'session_id': entry.session_id,
                'duration': entry.duration,
                'confidence': entry.confidence,
                'embedding_dim': entry.embedding_dim,
                'embedding_method': entry.embedding_method,
                'processing_metadata': entry.processing_metadata
            }
            
            # Add integrity check if enabled
            if self.enable_integrity_checks:
                entry_data['_checksum'] = self._calculate_checksum(entry.embedding)
            
            # Compress and save
            with gzip.open(disk_file, 'wb', compresslevel=self.compression_level) as f:
                pickle.dump(entry_data, f)
            
            # Update disk index
            self._disk_index[cache_key] = {
                'disk_key': disk_key,
                'timestamp': entry.timestamp,
                'session_id': entry.session_id,
                'file_path': str(disk_file)
            }
            
            save_time = (time.time() - start_time) * 1000  # Convert to ms
            self._save_times.append(save_time)
            if len(self._save_times) > 1000:
                self._save_times = self._save_times[-500:]  # Keep recent times
            
            with self._stats_lock:
                self._stats.disk_saves += 1
                self._stats.average_save_time_ms = sum(self._save_times) / len(self._save_times)
            
        except Exception as e:
            self.logger.error(f"Failed to save cache entry to disk: {e}")
    
    def _load_from_disk(self, cache_key: str) -> Optional[CacheEntry]:
        """Load cache entry from disk"""
        if not self.enable_disk_cache or cache_key not in self._disk_index:
            return None
        
        disk_info = self._disk_index[cache_key]
        disk_file = Path(disk_info['file_path'])
        
        if not disk_file.exists():
            # Clean up stale index entry
            del self._disk_index[cache_key]
            return None
        
        try:
            start_time = time.time()
            
            with gzip.open(disk_file, 'rb') as f:
                entry_data = pickle.load(f)
            
            # Integrity check if enabled
            if (self.enable_integrity_checks and 
                '_checksum' in entry_data and 
                entry_data['_checksum'] != self._calculate_checksum(entry_data['embedding'])):
                self.logger.warning(f"Integrity check failed for cache entry: {cache_key}")
                return None
            
            # Reconstruct cache entry
            entry = CacheEntry(
                embedding=entry_data['embedding'],
                timestamp=entry_data['timestamp'],
                access_count=entry_data['access_count'],
                chunk_id=entry_data['chunk_id'],
                stem_id=entry_data['stem_id'],
                speaker_id=entry_data['speaker_id'],
                session_id=entry_data['session_id'],
                duration=entry_data.get('duration', 0.0),
                confidence=entry_data.get('confidence', 0.0),
                embedding_dim=entry_data.get('embedding_dim', 0),
                embedding_method=entry_data.get('embedding_method', 'ecapa_tdnn'),
                processing_metadata=entry_data.get('processing_metadata', {})
            )
            
            load_time = (time.time() - start_time) * 1000  # Convert to ms
            self._lookup_times.append(load_time)
            if len(self._lookup_times) > 1000:
                self._lookup_times = self._lookup_times[-500:]  # Keep recent times
            
            with self._stats_lock:
                self._stats.disk_loads += 1
                if self._lookup_times:
                    self._stats.average_lookup_time_ms = sum(self._lookup_times) / len(self._lookup_times)
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Failed to load cache entry from disk: {e}")
            return None
    
    def _calculate_checksum(self, embedding: np.ndarray) -> str:
        """Calculate checksum for integrity verification"""
        if embedding is None:
            return ""
        
        return hashlib.md5(embedding.tobytes()).hexdigest()
    
    def put(self, session_id: str, chunk_id: str, stem_id: str, speaker_id: str,
            embedding: np.ndarray, duration: float = 0.0, confidence: float = 0.0,
            processing_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store speaker embedding in cache
        
        Args:
            session_id: Session identifier
            chunk_id: Chunk identifier
            stem_id: Stem identifier
            speaker_id: Speaker identifier
            embedding: Speaker embedding vector
            duration: Duration of audio segment
            confidence: Confidence score for embedding
            processing_metadata: Additional processing metadata
            
        Returns:
            True if successfully stored, False otherwise
        """
        if embedding is None or len(embedding) == 0:
            return False
        
        cache_key = self._generate_cache_key(session_id, chunk_id, stem_id, speaker_id)
        
        try:
            with self._lock:
                # Create cache entry
                entry = CacheEntry(
                    embedding=embedding.copy(),  # Create copy to avoid external modifications
                    timestamp=time.time(),
                    access_count=0,
                    chunk_id=chunk_id,
                    stem_id=stem_id,
                    speaker_id=speaker_id,
                    session_id=session_id,
                    duration=duration,
                    confidence=confidence,
                    embedding_dim=len(embedding),
                    processing_metadata=processing_metadata or {}
                )
                
                # Store in cache (moves to end for LRU)
                self._cache[cache_key] = entry
                self._cache.move_to_end(cache_key)
                
                # Update session tracking
                self._session_entries[session_id].add(cache_key)
                
                # Update memory usage and evict if necessary
                self._update_memory_usage()
                self._evict_lru_entries()
                
                self.logger.debug(f"Stored cache entry: {cache_key}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store cache entry: {e}")
            return False
    
    def get(self, session_id: str, chunk_id: str, stem_id: str, speaker_id: str) -> Optional[CacheEntry]:
        """
        Retrieve speaker embedding from cache
        
        Args:
            session_id: Session identifier
            chunk_id: Chunk identifier
            stem_id: Stem identifier
            speaker_id: Speaker identifier
            
        Returns:
            CacheEntry if found, None otherwise
        """
        cache_key = self._generate_cache_key(session_id, chunk_id, stem_id, speaker_id)
        
        start_time = time.time()
        entry = None
        
        try:
            with self._lock:
                # Check in-memory cache first
                if cache_key in self._cache:
                    entry = self._cache[cache_key]
                    
                    # Check if expired
                    if time.time() - entry.timestamp > self.ttl_seconds:
                        del self._cache[cache_key]
                        self._session_entries[entry.session_id].discard(cache_key)
                        entry = None
                    else:
                        # Move to end (most recently used) and update access count
                        self._cache.move_to_end(cache_key)
                        entry.access_count += 1
                        with self._stats_lock:
                            self._stats.cache_hits += 1
                
                # If not in memory, try disk cache
                if entry is None and self.enable_disk_cache:
                    disk_entry = self._load_from_disk(cache_key)
                    if disk_entry is not None:
                        # Load back into memory cache
                        self._cache[cache_key] = disk_entry
                        self._cache.move_to_end(cache_key)
                        self._session_entries[disk_entry.session_id].add(cache_key)
                        
                        disk_entry.access_count += 1
                        entry = disk_entry
                        
                        with self._stats_lock:
                            self._stats.cache_hits += 1
                        
                        # Update memory and evict if needed
                        self._update_memory_usage()
                        self._evict_lru_entries()
                
                # Record miss if no entry found
                if entry is None:
                    with self._stats_lock:
                        self._stats.cache_misses += 1
        
        except Exception as e:
            self.logger.error(f"Failed to retrieve cache entry: {e}")
            with self._stats_lock:
                self._stats.cache_misses += 1
        
        # Record lookup time
        lookup_time = (time.time() - start_time) * 1000
        self._lookup_times.append(lookup_time)
        if len(self._lookup_times) > 1000:
            self._lookup_times = self._lookup_times[-500:]
        
        with self._stats_lock:
            if self._lookup_times:
                self._stats.average_lookup_time_ms = sum(self._lookup_times) / len(self._lookup_times)
        
        return entry
    
    def get_session_embeddings(self, session_id: str) -> Dict[str, CacheEntry]:
        """
        Get all embeddings for a specific session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary mapping cache keys to cache entries
        """
        session_entries = {}
        
        with self._lock:
            if session_id in self._session_entries:
                for cache_key in self._session_entries[session_id]:
                    if cache_key in self._cache:
                        entry = self._cache[cache_key]
                        # Check if expired
                        if time.time() - entry.timestamp <= self.ttl_seconds:
                            session_entries[cache_key] = entry
        
        return session_entries
    
    def clear_session(self, session_id: str) -> int:
        """
        Clear all cache entries for a specific session
        
        Args:
            session_id: Session identifier to clear
            
        Returns:
            Number of entries cleared
        """
        cleared_count = 0
        
        with self._lock:
            if session_id in self._session_entries:
                keys_to_remove = list(self._session_entries[session_id])
                
                for cache_key in keys_to_remove:
                    if cache_key in self._cache:
                        del self._cache[cache_key]
                        cleared_count += 1
                
                del self._session_entries[session_id]
                
                # Also clear from disk index
                if self.enable_disk_cache:
                    disk_keys_to_remove = [k for k, v in self._disk_index.items() 
                                         if v.get('session_id') == session_id]
                    for disk_key in disk_keys_to_remove:
                        disk_info = self._disk_index.pop(disk_key)
                        # Optionally remove disk file
                        try:
                            disk_file = Path(disk_info['file_path'])
                            if disk_file.exists():
                                disk_file.unlink()
                        except Exception as e:
                            self.logger.debug(f"Failed to remove disk file: {e}")
                
                self._update_memory_usage()
                self.logger.info(f"Cleared {cleared_count} cache entries for session: {session_id}")
        
        return cleared_count
    
    def clear_all(self):
        """Clear all cache entries and reset statistics"""
        with self._lock:
            self._cache.clear()
            self._session_entries.clear()
            
            # Clear disk cache if enabled
            if self.enable_disk_cache:
                try:
                    for disk_info in self._disk_index.values():
                        disk_file = Path(disk_info['file_path'])
                        if disk_file.exists():
                            disk_file.unlink()
                except Exception as e:
                    self.logger.debug(f"Error clearing disk cache: {e}")
                
                self._disk_index.clear()
                self._save_disk_index()
            
            # Reset statistics
            with self._stats_lock:
                self._stats = CacheStats()
            
            self._lookup_times.clear()
            self._save_times.clear()
            
            self.logger.info("Cleared all cache entries")
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        with self._stats_lock:
            # Update session statistics
            self._stats.sessions_active = len(self._session_entries)
            if self._session_entries:
                total_entries = sum(len(entries) for entries in self._session_entries.values())
                self._stats.average_embeddings_per_session = total_entries / len(self._session_entries)
            
            return CacheStats(
                total_entries=self._stats.total_entries,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                total_memory_mb=self._stats.total_memory_mb,
                entries_evicted=self._stats.entries_evicted,
                disk_saves=self._stats.disk_saves,
                disk_loads=self._stats.disk_loads,
                sessions_active=self._stats.sessions_active,
                average_embeddings_per_session=self._stats.average_embeddings_per_session,
                average_lookup_time_ms=self._stats.average_lookup_time_ms,
                average_save_time_ms=self._stats.average_save_time_ms
            )
    
    def maintenance(self):
        """Perform cache maintenance tasks"""
        try:
            # Clean up expired entries
            self._cleanup_expired_entries()
            
            # Save disk index if dirty
            if self.enable_disk_cache:
                self._save_disk_index()
            
            # Log statistics periodically
            stats = self.get_stats()
            self.logger.info("Cache maintenance completed",
                           context={
                               'total_entries': stats.total_entries,
                               'memory_mb': stats.memory_usage_mb,
                               'hit_rate': stats.hit_rate,
                               'sessions_active': stats.sessions_active
                           })
            
        except Exception as e:
            self.logger.error(f"Cache maintenance failed: {e}")
    
    def __del__(self):
        """Cleanup when cache is destroyed"""
        try:
            if hasattr(self, 'enable_disk_cache') and self.enable_disk_cache:
                self._save_disk_index()
        except:
            pass

# Global cache instance
_global_embedding_cache: Optional[ThreadSafeEmbeddingCache] = None

def get_embedding_cache(
    max_memory_mb: float = 512.0,
    max_entries: int = 10000,
    enable_disk_cache: bool = True,
    **kwargs
) -> ThreadSafeEmbeddingCache:
    """Get global embedding cache instance (singleton pattern)"""
    global _global_embedding_cache
    
    if _global_embedding_cache is None:
        _global_embedding_cache = ThreadSafeEmbeddingCache(
            max_memory_mb=max_memory_mb,
            max_entries=max_entries,
            enable_disk_cache=enable_disk_cache,
            **kwargs
        )
    
    return _global_embedding_cache

def clear_global_cache():
    """Clear and reset global cache instance"""
    global _global_embedding_cache
    
    if _global_embedding_cache is not None:
        _global_embedding_cache.clear_all()
        _global_embedding_cache = None