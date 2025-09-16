"""
Auto-Glossary Term Store Management

Manages persistent project_term_base.json with:
- Project-scoped term storage and retrieval
- Exponential decay for unused terms
- Minimum support thresholds for quality control
- Hard denylist support for sensitive terms
- Canonical form management with variant tracking
- Confidence score aggregation and statistics

Lifecycle:
1. Load existing project term base or create new
2. Merge session candidates with gating thresholds
3. Update term statistics and apply decay
4. Persist updated term base to storage
"""

import os
import json
import time
import math
import hashlib
from typing import Dict, Any, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
import threading
from contextlib import contextmanager
import fcntl

from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class TermEntry:
    """Individual term entry in the project term base"""
    canonical_form: str
    variants: Set[str]
    total_count: int
    session_count: int
    confidence_mean: float
    confidence_variance: float
    first_seen_timestamp: float
    last_seen_timestamp: float
    example_spans: List[Dict[str, Any]]  # Up to 5 example contexts
    supporting_engines: Set[str]
    decay_factor: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure variants are properly initialized"""
        if isinstance(self.variants, list):
            self.variants = set(self.variants)
        if isinstance(self.supporting_engines, list):
            self.supporting_engines = set(self.supporting_engines)

@dataclass  
class ProjectTermBase:
    """Complete project term base with metadata"""
    project_id: str
    terms: Dict[str, TermEntry]
    creation_timestamp: float
    last_updated_timestamp: float
    session_count: int
    decay_sessions_threshold: int
    minimum_support_threshold: int
    denylist: Set[str]
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure denylist is properly initialized"""
        if isinstance(self.denylist, list):
            self.denylist = set(self.denylist)

@dataclass
class MergeResult:
    """Result from merging session candidates into term base"""
    terms_added: int
    terms_updated: int
    terms_decayed: int
    terms_denied: int
    total_processing_time: float
    merge_metadata: Dict[str, Any]

class ProjectTermStore:
    """Manages persistent project term bases with thread-safe operations"""
    
    def __init__(self, 
                 storage_base_path: str = "term_bases",
                 decay_sessions_threshold: int = 10,
                 minimum_support_threshold: int = 2,
                 max_variants_per_term: int = 5,
                 max_example_spans: int = 5,
                 confidence_smoothing_factor: float = 0.1,
                 enable_file_locking: bool = True):
        """
        Initialize project term store
        
        Args:
            storage_base_path: Base directory for term base files
            decay_sessions_threshold: Sessions before decay starts
            minimum_support_threshold: Minimum support needed to add term
            max_variants_per_term: Maximum variants to track per term
            max_example_spans: Maximum example spans to store per term
            confidence_smoothing_factor: Smoothing for confidence updates
            enable_file_locking: Whether to use file locking for thread safety
        """
        self.storage_base_path = Path(storage_base_path)
        self.decay_sessions_threshold = decay_sessions_threshold
        self.minimum_support_threshold = minimum_support_threshold
        self.max_variants_per_term = max_variants_per_term
        self.max_example_spans = max_example_spans
        self.confidence_smoothing_factor = confidence_smoothing_factor
        self.enable_file_locking = enable_file_locking
        
        # Thread-safe cache for loaded term bases
        self._term_base_cache = {}
        self._cache_lock = threading.Lock()
        
        # Initialize logger
        self.logger = create_enhanced_logger("project_term_store")
        
        # Create storage directory if it doesn't exist
        self.storage_base_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Project term store initialized", 
                        context={
                            'storage_path': str(self.storage_base_path),
                            'decay_threshold': decay_sessions_threshold,
                            'min_support': minimum_support_threshold,
                            'max_variants': max_variants_per_term,
                            'file_locking': enable_file_locking
                        })
    
    def get_project_term_base_path(self, project_id: str) -> Path:
        """Get the file path for a project's term base"""
        # Sanitize project ID for filename
        safe_project_id = self._sanitize_project_id(project_id)
        return self.storage_base_path / f"{safe_project_id}_term_base.json"
    
    def _sanitize_project_id(self, project_id: str) -> str:
        """Sanitize project ID for safe filesystem usage"""
        # Remove or replace unsafe characters
        safe_id = "".join(c for c in project_id if c.isalnum() or c in ('-', '_'))
        # Limit length and add hash if too long
        if len(safe_id) > 50:
            hash_suffix = hashlib.md5(project_id.encode()).hexdigest()[:8]
            safe_id = safe_id[:42] + "_" + hash_suffix
        return safe_id or "unknown_project"
    
    @contextmanager
    def _file_lock(self, file_path: Path):
        """Context manager for file locking if enabled"""
        if not self.enable_file_locking:
            yield
            return
        
        lock_file_path = file_path.with_suffix(file_path.suffix + '.lock')
        try:
            lock_file = open(lock_file_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                lock_file_path.unlink(missing_ok=True)
            except:
                pass
    
    def load_project_term_base(self, project_id: str) -> ProjectTermBase:
        """Load or create project term base"""
        # Check cache first
        with self._cache_lock:
            if project_id in self._term_base_cache:
                cached_base, cache_time = self._term_base_cache[project_id]
                # Cache valid for 5 minutes
                if time.time() - cache_time < 300:
                    return cached_base
        
        term_base_path = self.get_project_term_base_path(project_id)
        
        with self._file_lock(term_base_path):
            try:
                if term_base_path.exists():
                    # Load existing term base
                    with open(term_base_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Reconstruct term base from JSON
                    term_base = self._deserialize_project_term_base(data)
                    
                    self.logger.info("Loaded existing project term base", 
                                   context={
                                       'project_id': project_id,
                                       'terms_count': len(term_base.terms),
                                       'sessions': term_base.session_count,
                                       'last_updated': term_base.last_updated_timestamp
                                   })
                else:
                    # Create new term base
                    term_base = ProjectTermBase(
                        project_id=project_id,
                        terms={},
                        creation_timestamp=time.time(),
                        last_updated_timestamp=time.time(),
                        session_count=0,
                        decay_sessions_threshold=self.decay_sessions_threshold,
                        minimum_support_threshold=self.minimum_support_threshold,
                        denylist=set(),
                        statistics={}
                    )
                    
                    self.logger.info("Created new project term base", 
                                   context={'project_id': project_id})
                
                # Update cache
                with self._cache_lock:
                    self._term_base_cache[project_id] = (term_base, time.time())
                
                return term_base
                
            except Exception as e:
                self.logger.error("Failed to load project term base", 
                                context={'project_id': project_id, 'error': str(e)})
                # Return empty term base on error
                return ProjectTermBase(
                    project_id=project_id,
                    terms={},
                    creation_timestamp=time.time(),
                    last_updated_timestamp=time.time(),
                    session_count=0,
                    decay_sessions_threshold=self.decay_sessions_threshold,
                    minimum_support_threshold=self.minimum_support_threshold,
                    denylist=set()
                )
    
    def save_project_term_base(self, term_base: ProjectTermBase) -> bool:
        """Save project term base to persistent storage"""
        term_base_path = self.get_project_term_base_path(term_base.project_id)
        
        with self._file_lock(term_base_path):
            try:
                # Update metadata
                term_base.last_updated_timestamp = time.time()
                
                # Serialize term base to JSON-compatible format
                serializable_data = self._serialize_project_term_base(term_base)
                
                # Write atomically using temporary file
                temp_path = term_base_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                temp_path.rename(term_base_path)
                
                # Update cache
                with self._cache_lock:
                    self._term_base_cache[term_base.project_id] = (term_base, time.time())
                
                self.logger.info("Saved project term base", 
                               context={
                                   'project_id': term_base.project_id,
                                   'terms_count': len(term_base.terms),
                                   'file_path': str(term_base_path)
                               })
                return True
                
            except Exception as e:
                self.logger.error("Failed to save project term base", 
                                context={
                                    'project_id': term_base.project_id,
                                    'error': str(e),
                                    'file_path': str(term_base_path)
                                })
                return False
    
    def merge_session_candidates(self, 
                                project_id: str,
                                session_candidates: List[Dict[str, Any]],
                                session_id: str = None) -> MergeResult:
        """
        Merge session candidates into project term base with quality gating
        
        Args:
            project_id: Project identifier
            session_candidates: List of session term candidates
            session_id: Optional session identifier for tracking
            
        Returns:
            MergeResult with statistics about the merge operation
        """
        start_time = time.time()
        session_id = session_id or f"session_{int(time.time())}"
        
        self.logger.info("Starting session candidate merge", 
                        context={
                            'project_id': project_id,
                            'session_id': session_id,
                            'candidates': len(session_candidates)
                        })
        
        # Load existing term base
        term_base = self.load_project_term_base(project_id)
        
        # Initialize merge statistics
        terms_added = 0
        terms_updated = 0
        terms_decayed = 0
        terms_denied = 0
        
        # Process each candidate
        for candidate in session_candidates:
            try:
                result = self._process_session_candidate(term_base, candidate)
                if result == 'added':
                    terms_added += 1
                elif result == 'updated':
                    terms_updated += 1
                elif result == 'denied':
                    terms_denied += 1
            except Exception as e:
                self.logger.warning("Error processing candidate", 
                                  context={
                                      'candidate': candidate.get('token', 'unknown'),
                                      'error': str(e)
                                  })
                continue
        
        # Apply decay to terms not seen in this session
        session_tokens = {c.get('token', '').lower() for c in session_candidates}
        terms_decayed = self._apply_decay(term_base, session_tokens)
        
        # Update session statistics
        term_base.session_count += 1
        term_base.statistics.update(self._calculate_term_base_statistics(term_base))
        
        # Save updated term base
        save_success = self.save_project_term_base(term_base)
        
        processing_time = time.time() - start_time
        
        # Create merge result
        merge_result = MergeResult(
            terms_added=terms_added,
            terms_updated=terms_updated,
            terms_decayed=terms_decayed,
            terms_denied=terms_denied,
            total_processing_time=processing_time,
            merge_metadata={
                'session_id': session_id,
                'project_id': project_id,
                'save_success': save_success,
                'total_terms_after_merge': len(term_base.terms),
                'session_count': term_base.session_count
            }
        )
        
        self.logger.info("Session candidate merge completed", 
                        context={
                            'project_id': project_id,
                            'session_id': session_id,
                            'added': terms_added,
                            'updated': terms_updated,
                            'decayed': terms_decayed,
                            'denied': terms_denied,
                            'processing_time': processing_time
                        })
        
        return merge_result
    
    def _process_session_candidate(self, term_base: ProjectTermBase, candidate: Dict[str, Any]) -> str:
        """Process individual session candidate"""
        token = candidate.get('token', '').strip()
        if not token:
            return 'denied'
        
        canonical_token = token.lower()
        
        # Check denylist
        if canonical_token in term_base.denylist or token in term_base.denylist:
            return 'denied'
        
        # Extract candidate information
        weight = candidate.get('weight', 0.0)
        supporting_engines = set(candidate.get('supporting_engines', []))
        local_context = candidate.get('local_context', [])
        scores = candidate.get('scores', {})
        final_mining_score = scores.get('final_mining', weight)
        variants = set(candidate.get('variants', []))
        
        # Check minimum support threshold
        if final_mining_score < (self.minimum_support_threshold / 10.0):  # Convert to 0-1 scale
            return 'denied'
        
        current_time = time.time()
        
        if canonical_token in term_base.terms:
            # Update existing term
            term_entry = term_base.terms[canonical_token]
            
            # Update counts and timestamps
            term_entry.total_count += 1
            term_entry.session_count += 1
            term_entry.last_seen_timestamp = current_time
            
            # Update confidence using exponential smoothing
            old_mean = term_entry.confidence_mean
            term_entry.confidence_mean = (
                (1 - self.confidence_smoothing_factor) * old_mean +
                self.confidence_smoothing_factor * final_mining_score
            )
            
            # Update confidence variance (simplified)
            diff = final_mining_score - old_mean
            term_entry.confidence_variance = (
                (1 - self.confidence_smoothing_factor) * term_entry.confidence_variance +
                self.confidence_smoothing_factor * diff * diff
            )
            
            # Add variants (up to max limit)
            new_variants = variants - term_entry.variants
            if new_variants and len(term_entry.variants) < self.max_variants_per_term:
                variants_to_add = list(new_variants)[:self.max_variants_per_term - len(term_entry.variants)]
                term_entry.variants.update(variants_to_add)
            
            # Update supporting engines
            term_entry.supporting_engines.update(supporting_engines)
            
            # Add example span if we have context and room for more examples
            if local_context and len(term_entry.example_spans) < self.max_example_spans:
                example_span = {
                    'context': local_context,
                    'timestamp': current_time,
                    'confidence': final_mining_score
                }
                term_entry.example_spans.append(example_span)
            
            # Reset decay factor
            term_entry.decay_factor = 1.0
            
            return 'updated'
        else:
            # Add new term
            example_spans = []
            if local_context:
                example_spans = [{
                    'context': local_context,
                    'timestamp': current_time,
                    'confidence': final_mining_score
                }]
            
            term_entry = TermEntry(
                canonical_form=token,  # Use original case as canonical
                variants=variants,
                total_count=1,
                session_count=1,
                confidence_mean=final_mining_score,
                confidence_variance=0.0,
                first_seen_timestamp=current_time,
                last_seen_timestamp=current_time,
                example_spans=example_spans,
                supporting_engines=supporting_engines,
                decay_factor=1.0,
                metadata={
                    'scores': scores,
                    'initial_weight': weight
                }
            )
            
            term_base.terms[canonical_token] = term_entry
            return 'added'
    
    def _apply_decay(self, term_base: ProjectTermBase, seen_tokens: Set[str]) -> int:
        """Apply exponential decay to terms not seen in current session"""
        decayed_count = 0
        decay_rate = 0.95  # 5% decay per session not seen
        
        terms_to_remove = []
        
        for canonical_token, term_entry in term_base.terms.items():
            # Check if term was seen in current session (case insensitive)
            if canonical_token not in seen_tokens and term_entry.canonical_form.lower() not in seen_tokens:
                # Apply decay
                term_entry.decay_factor *= decay_rate
                decayed_count += 1
                
                # Mark for removal if decay factor is too low
                if term_entry.decay_factor < 0.1:  # Remove if decayed to 10% of original
                    terms_to_remove.append(canonical_token)
        
        # Remove heavily decayed terms
        for canonical_token in terms_to_remove:
            del term_base.terms[canonical_token]
            self.logger.debug("Removed decayed term", 
                            context={'term': canonical_token, 'project': term_base.project_id})
        
        if decayed_count > 0:
            self.logger.debug("Applied decay to terms", 
                            context={
                                'decayed_count': decayed_count,
                                'removed_count': len(terms_to_remove),
                                'project': term_base.project_id
                            })
        
        return decayed_count
    
    def _calculate_term_base_statistics(self, term_base: ProjectTermBase) -> Dict[str, Any]:
        """Calculate statistics for the term base"""
        if not term_base.terms:
            return {
                'total_terms': 0,
                'average_confidence': 0.0,
                'high_confidence_terms': 0,
                'terms_with_variants': 0,
                'average_session_count': 0.0
            }
        
        total_terms = len(term_base.terms)
        confidences = [term.confidence_mean for term in term_base.terms.values()]
        session_counts = [term.session_count for term in term_base.terms.values()]
        
        return {
            'total_terms': total_terms,
            'average_confidence': sum(confidences) / len(confidences),
            'high_confidence_terms': sum(1 for c in confidences if c >= 0.8),
            'terms_with_variants': sum(1 for term in term_base.terms.values() if term.variants),
            'average_session_count': sum(session_counts) / len(session_counts),
            'max_session_count': max(session_counts),
            'terms_by_engine': self._calculate_engine_distribution(term_base)
        }
    
    def _calculate_engine_distribution(self, term_base: ProjectTermBase) -> Dict[str, int]:
        """Calculate distribution of terms by supporting engines"""
        engine_counts = defaultdict(int)
        for term in term_base.terms.values():
            for engine in term.supporting_engines:
                engine_counts[engine] += 1
        return dict(engine_counts)
    
    def update_denylist(self, project_id: str, denylist_terms: List[str], append: bool = True) -> bool:
        """Update project denylist"""
        try:
            term_base = self.load_project_term_base(project_id)
            
            if append:
                term_base.denylist.update(term.lower() for term in denylist_terms)
            else:
                term_base.denylist = set(term.lower() for term in denylist_terms)
            
            # Remove any existing terms that are now denylisted
            terms_to_remove = []
            for canonical_token in term_base.terms:
                if canonical_token in term_base.denylist:
                    terms_to_remove.append(canonical_token)
            
            for canonical_token in terms_to_remove:
                del term_base.terms[canonical_token]
            
            success = self.save_project_term_base(term_base)
            
            self.logger.info("Updated project denylist", 
                           context={
                               'project_id': project_id,
                               'denylist_size': len(term_base.denylist),
                               'terms_removed': len(terms_to_remove),
                               'append_mode': append
                           })
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to update denylist", 
                            context={'project_id': project_id, 'error': str(e)})
            return False
    
    def get_top_terms(self, 
                     project_id: str, 
                     limit: int = 50, 
                     min_confidence: float = 0.5,
                     min_session_count: int = 1) -> List[Dict[str, Any]]:
        """Get top terms from project term base for biasing"""
        try:
            term_base = self.load_project_term_base(project_id)
            
            # Filter and rank terms
            candidate_terms = []
            for canonical_token, term_entry in term_base.terms.items():
                # Apply filters
                if (term_entry.confidence_mean >= min_confidence and 
                    term_entry.session_count >= min_session_count and
                    term_entry.decay_factor >= 0.3):  # Only active terms
                    
                    # Calculate ranking score (confidence + decay + session frequency)
                    ranking_score = (
                        term_entry.confidence_mean * 0.4 +
                        term_entry.decay_factor * 0.3 +
                        min(1.0, term_entry.session_count / 10.0) * 0.3
                    )
                    
                    candidate_terms.append({
                        'token': term_entry.canonical_form,
                        'canonical_token': canonical_token,
                        'weight': ranking_score,
                        'confidence': term_entry.confidence_mean,
                        'session_count': term_entry.session_count,
                        'decay_factor': term_entry.decay_factor,
                        'variants': list(term_entry.variants),
                        'supporting_engines': list(term_entry.supporting_engines)
                    })
            
            # Sort by ranking score and limit
            candidate_terms.sort(key=lambda x: x['weight'], reverse=True)
            top_terms = candidate_terms[:limit]
            
            self.logger.debug("Retrieved top terms for biasing", 
                            context={
                                'project_id': project_id,
                                'requested_limit': limit,
                                'returned_count': len(top_terms),
                                'min_confidence': min_confidence,
                                'min_session_count': min_session_count
                            })
            
            return top_terms
            
        except Exception as e:
            self.logger.error("Failed to get top terms", 
                            context={'project_id': project_id, 'error': str(e)})
            return []
    
    def get_project_statistics(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a project term base"""
        try:
            term_base = self.load_project_term_base(project_id)
            
            stats = {
                'project_id': project_id,
                'creation_timestamp': term_base.creation_timestamp,
                'last_updated_timestamp': term_base.last_updated_timestamp,
                'session_count': term_base.session_count,
                'denylist_size': len(term_base.denylist),
                'statistics': term_base.statistics
            }
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get project statistics", 
                            context={'project_id': project_id, 'error': str(e)})
            return {}
    
    def _serialize_project_term_base(self, term_base: ProjectTermBase) -> Dict[str, Any]:
        """Convert ProjectTermBase to JSON-serializable format"""
        serializable_terms = {}
        
        for canonical_token, term_entry in term_base.terms.items():
            serializable_terms[canonical_token] = {
                'canonical_form': term_entry.canonical_form,
                'variants': list(term_entry.variants),
                'total_count': term_entry.total_count,
                'session_count': term_entry.session_count,
                'confidence_mean': term_entry.confidence_mean,
                'confidence_variance': term_entry.confidence_variance,
                'first_seen_timestamp': term_entry.first_seen_timestamp,
                'last_seen_timestamp': term_entry.last_seen_timestamp,
                'example_spans': term_entry.example_spans,
                'supporting_engines': list(term_entry.supporting_engines),
                'decay_factor': term_entry.decay_factor,
                'metadata': term_entry.metadata
            }
        
        return {
            'project_id': term_base.project_id,
            'terms': serializable_terms,
            'creation_timestamp': term_base.creation_timestamp,
            'last_updated_timestamp': term_base.last_updated_timestamp,
            'session_count': term_base.session_count,
            'decay_sessions_threshold': term_base.decay_sessions_threshold,
            'minimum_support_threshold': term_base.minimum_support_threshold,
            'denylist': list(term_base.denylist),
            'statistics': term_base.statistics,
            'metadata': term_base.metadata
        }
    
    def _deserialize_project_term_base(self, data: Dict[str, Any]) -> ProjectTermBase:
        """Convert JSON data back to ProjectTermBase"""
        terms = {}
        
        for canonical_token, term_data in data.get('terms', {}).items():
            terms[canonical_token] = TermEntry(
                canonical_form=term_data['canonical_form'],
                variants=set(term_data.get('variants', [])),
                total_count=term_data['total_count'],
                session_count=term_data['session_count'],
                confidence_mean=term_data['confidence_mean'],
                confidence_variance=term_data.get('confidence_variance', 0.0),
                first_seen_timestamp=term_data['first_seen_timestamp'],
                last_seen_timestamp=term_data['last_seen_timestamp'],
                example_spans=term_data.get('example_spans', []),
                supporting_engines=set(term_data.get('supporting_engines', [])),
                decay_factor=term_data.get('decay_factor', 1.0),
                metadata=term_data.get('metadata', {})
            )
        
        return ProjectTermBase(
            project_id=data['project_id'],
            terms=terms,
            creation_timestamp=data['creation_timestamp'],
            last_updated_timestamp=data.get('last_updated_timestamp', time.time()),
            session_count=data.get('session_count', 0),
            decay_sessions_threshold=data.get('decay_sessions_threshold', self.decay_sessions_threshold),
            minimum_support_threshold=data.get('minimum_support_threshold', self.minimum_support_threshold),
            denylist=set(data.get('denylist', [])),
            statistics=data.get('statistics', {}),
            metadata=data.get('metadata', {})
        )

def create_project_term_store(**config) -> ProjectTermStore:
    """Factory function to create project term store with configuration"""
    return ProjectTermStore(
        storage_base_path=config.get('storage_base_path', 'term_bases'),
        decay_sessions_threshold=config.get('decay_sessions_threshold', 10),
        minimum_support_threshold=config.get('minimum_support_threshold', 2),
        max_variants_per_term=config.get('max_variants_per_term', 5),
        max_example_spans=config.get('max_example_spans', 5),
        confidence_smoothing_factor=config.get('confidence_smoothing_factor', 0.1),
        enable_file_locking=config.get('enable_file_locking', True)
    )

# Example usage and testing
if __name__ == "__main__":
    # Test the project term store
    term_store = create_project_term_store(storage_base_path='/tmp/test_term_bases')
    
    # Mock session candidates
    mock_session_candidates = [
        {
            'token': 'ModelX-150',
            'weight': 0.85,
            'supporting_engines': ['whisper', 'deepgram'],
            'local_context': ['our', 'ModelX-150', 'sales', 'increased'],
            'scores': {'final_mining': 0.85},
            'variants': {'Model X-150', 'ModelX150'}
        },
        {
            'token': 'Q3-2024',
            'weight': 0.92,
            'supporting_engines': ['whisper', 'openai'],
            'local_context': ['our', 'Q3-2024', 'revenue', 'was'],
            'scores': {'final_mining': 0.92},
            'variants': {'Q3 2024', 'Q3-24'}
        }
    ]
    
    # Test merge operation
    result = term_store.merge_session_candidates(
        project_id='test_project',
        session_candidates=mock_session_candidates,
        session_id='test_session_1'
    )
    
    print(f"Merge result: {result.terms_added} added, {result.terms_updated} updated")
    
    # Test getting top terms
    top_terms = term_store.get_top_terms('test_project', limit=10)
    print(f"Top terms: {[t['token'] for t in top_terms]}")
    
    # Test statistics
    stats = term_store.get_project_statistics('test_project')
    print(f"Project stats: {stats.get('statistics', {})}")