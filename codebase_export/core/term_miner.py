"""
Auto-Glossary Term Mining Engine

Extracts domain-specific terminology from first-pass ASR hypotheses using:
- TF-IDF scoring against background corpus
- Mixed case pattern recognition for proper nouns and acronyms
- Multi-speaker agreement scoring
- Technical term patterns (digits, hyphens, units)
- Visual similarity clustering for variant detection
- Stoplist and profanity filtering

Output: session_term_candidates.json with scored candidates
"""

import os
import re
import json
import math
import time
from typing import Dict, Any, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import numpy as np
from difflib import SequenceMatcher
import unicodedata

from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class TermCandidate:
    """Individual term candidate with mining metadata"""
    token: str
    weight: float
    first_seen_time: float
    supporting_engines: Set[str]
    local_context: List[str]  # Surrounding words
    frequency_score: float
    case_pattern_score: float
    technical_pattern_score: float
    multi_speaker_score: float
    unit_proximity_score: float
    final_mining_score: float
    source_segments: List[Dict[str, Any]]  # Source segment info
    variants: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionTermResults:
    """Complete session term mining results"""
    candidates: List[TermCandidate]
    total_candidates: int
    high_confidence_candidates: int
    technical_term_count: int
    proper_noun_count: int
    multi_speaker_agreements: int
    processing_time: float
    mining_metadata: Dict[str, Any]

class BackgroundCorpus:
    """Manages background frequency statistics for TF-IDF scoring"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("background_corpus")
        
        # English background word frequencies (top 10k words)
        # In production, load from external corpus data
        self.background_frequencies = self._load_background_frequencies()
        self.total_background_tokens = sum(self.background_frequencies.values())
        
        # English stopwords
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'you', 'your', 'i', 'we',
            'they', 'them', 'their', 'this', 'these', 'those', 'there', 'here',
            'where', 'when', 'what', 'who', 'why', 'how', 'but', 'or', 'so',
            'if', 'can', 'could', 'should', 'would', 'will', 'shall', 'may',
            'might', 'must', 'do', 'does', 'did', 'have', 'had', 'been',
            'being', 'get', 'got', 'go', 'went', 'going', 'come', 'came',
            'coming', 'see', 'saw', 'seen', 'look', 'looked', 'looking',
            'say', 'said', 'saying', 'tell', 'told', 'telling', 'think',
            'thought', 'thinking', 'know', 'knew', 'known', 'knowing',
            'want', 'wanted', 'wanting', 'like', 'liked', 'liking', 'use',
            'used', 'using', 'work', 'worked', 'working', 'make', 'made',
            'making', 'take', 'took', 'taken', 'taking', 'give', 'gave',
            'given', 'giving', 'put', 'putting', 'let', 'letting', 'try',
            'tried', 'trying', 'keep', 'kept', 'keeping', 'turn', 'turned',
            'turning', 'start', 'started', 'starting', 'end', 'ended', 'ending',
            'find', 'found', 'finding', 'leave', 'left', 'leaving', 'feel',
            'felt', 'feeling', 'seem', 'seemed', 'seeming', 'become', 'became',
            'becoming', 'call', 'called', 'calling', 'ask', 'asked', 'asking',
            'need', 'needed', 'needing', 'help', 'helped', 'helping', 'move',
            'moved', 'moving', 'show', 'showed', 'showing', 'play', 'played',
            'playing', 'run', 'ran', 'running', 'live', 'lived', 'living'
        }
        
        # Basic profanity list (minimal for professional contexts)
        self.profanity_list = {
            'damn', 'hell', 'shit', 'fuck', 'fucking', 'fucked', 'bitch',
            'bastard', 'asshole', 'ass', 'crap', 'piss', 'goddamn'
        }
        
        self.logger.info("Background corpus initialized", 
                        context={'background_tokens': self.total_background_tokens,
                                'stopwords': len(self.stopwords),
                                'profanity_terms': len(self.profanity_list)})
    
    def _load_background_frequencies(self) -> Dict[str, int]:
        """Load background word frequencies (simplified for demo)"""
        # In production, load from external corpus files
        # For now, create basic frequency distribution
        background = {
            'the': 100000, 'be': 80000, 'to': 75000, 'of': 70000, 'and': 65000,
            'a': 60000, 'in': 55000, 'that': 50000, 'have': 45000, 'i': 40000,
            'it': 38000, 'for': 35000, 'not': 33000, 'on': 30000, 'with': 28000,
            'he': 26000, 'as': 24000, 'you': 22000, 'do': 20000, 'at': 18000,
            'this': 16000, 'but': 14000, 'his': 12000, 'by': 11000, 'from': 10000,
            'they': 9500, 'we': 9000, 'say': 8500, 'her': 8000, 'she': 7500,
            'or': 7000, 'an': 6500, 'will': 6000, 'my': 5500, 'one': 5000,
            'all': 4800, 'would': 4600, 'there': 4400, 'their': 4200, 'what': 4000,
            'so': 3800, 'up': 3600, 'out': 3400, 'if': 3200, 'about': 3000,
            'who': 2800, 'get': 2600, 'which': 2400, 'go': 2200, 'me': 2000,
            'when': 1900, 'make': 1800, 'can': 1700, 'like': 1600, 'time': 1500,
            'no': 1400, 'just': 1300, 'him': 1200, 'know': 1100, 'take': 1000,
            'people': 950, 'into': 900, 'year': 850, 'your': 800, 'good': 750,
            'some': 700, 'could': 650, 'them': 600, 'see': 550, 'other': 500,
            'than': 480, 'then': 460, 'now': 440, 'look': 420, 'only': 400,
            'come': 380, 'its': 360, 'over': 340, 'think': 320, 'also': 300,
            'back': 290, 'after': 280, 'use': 270, 'two': 260, 'how': 250,
            'our': 240, 'work': 230, 'first': 220, 'well': 210, 'way': 200,
            'even': 195, 'new': 190, 'want': 185, 'because': 180, 'any': 175,
            'these': 170, 'give': 165, 'day': 160, 'most': 155, 'us': 150
        }
        return background
    
    def get_background_probability(self, token: str) -> float:
        """Get background probability for a token"""
        token_lower = token.lower()
        frequency = self.background_frequencies.get(token_lower, 1)  # Smoothing
        return frequency / self.total_background_tokens
    
    def is_stopword(self, token: str) -> bool:
        """Check if token is a stopword"""
        return token.lower() in self.stopwords
    
    def is_profanity(self, token: str) -> bool:
        """Check if token is profanity"""
        return token.lower() in self.profanity_list

class PatternAnalyzer:
    """Analyzes tokens for various linguistic and technical patterns"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("pattern_analyzer")
        
        # Unit markers and technical indicators
        self.unit_markers = {
            # Measurements
            'mg', 'ml', 'kg', 'lb', 'oz', 'ft', 'in', 'cm', 'mm', 'm', 'km',
            'mph', 'kph', 'rpm', 'hz', 'khz', 'mhz', 'ghz', 'mb', 'gb', 'tb',
            'kb', 'bits', 'bytes', 'px', 'dpi', 'psi', 'bar', 'atm',
            # Business/Tech
            'sku', 'upc', 'api', 'sdk', 'url', 'uri', 'css', 'html', 'xml',
            'json', 'sql', 'tcp', 'udp', 'http', 'https', 'ftp', 'ssh',
            'version', 'v', 'model', 'rev', 'build', 'id', 'uuid', 'guid',
            # Financial
            'usd', 'eur', 'gbp', 'cad', 'aud', 'jpy', 'cny', 'inr', 'btc',
            'eth', 'ltc', 'xrp', 'ada', 'dot', 'roi', 'kpi', 'p&l', 'ebitda',
            # Time periods
            'q1', 'q2', 'q3', 'q4', 'h1', 'h2', 'fy', 'cy', 'ytd', 'mtd',
            'qoq', 'yoy', 'mom', 'wow'
        }
        
        self.technical_patterns = {
            'version_number': re.compile(r'v?\d+\.[\d\.]+(?:-[a-zA-Z0-9]+)?', re.IGNORECASE),
            'model_number': re.compile(r'[a-zA-Z]{1,4}-?\d{2,6}[a-zA-Z]?', re.IGNORECASE),
            'sku_pattern': re.compile(r'[a-zA-Z0-9]{6,12}', re.IGNORECASE),
            'date_quarter': re.compile(r'[q]\d{1}-?\d{2,4}', re.IGNORECASE),
            'percentage': re.compile(r'\d+\.?\d*%'),
            'measurement': re.compile(r'\d+\.?\d*\s*[a-zA-Z]{1,4}'),
            'currency': re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+\s*(?:usd|eur|gbp)', re.IGNORECASE),
            'phone_number': re.compile(r'\d{3}-?\d{3}-?\d{4}'),
            'email_like': re.compile(r'[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'url_like': re.compile(r'https?://[^\s]+|www\.[^\s]+\.[a-zA-Z]{2,}'),
            'ip_address': re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'),
            'mac_address': re.compile(r'[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}')
        }
        
        self.logger.info("Pattern analyzer initialized", 
                        context={'unit_markers': len(self.unit_markers),
                                'technical_patterns': len(self.technical_patterns)})
    
    def analyze_case_pattern(self, token: str) -> float:
        """Score token based on capitalization patterns"""
        if len(token) <= 1:
            return 0.0
        
        # All uppercase (acronyms) - high value
        if token.isupper() and len(token) >= 2:
            return 1.0
        
        # Mixed case (proper nouns, brands) - high value
        if token[0].isupper() and any(c.islower() for c in token[1:]):
            return 0.9
        
        # camelCase or PascalCase - moderate value
        if any(c.isupper() for c in token[1:]) and any(c.islower() for c in token):
            return 0.8
        
        # All lowercase with digits/hyphens - moderate value
        if token.islower() and (any(c.isdigit() for c in token) or '-' in token):
            return 0.6
        
        # All lowercase - low value (common words)
        if token.islower():
            return 0.1
        
        return 0.3
    
    def analyze_technical_patterns(self, token: str) -> Tuple[float, List[str]]:
        """Score token for technical patterns"""
        score = 0.0
        matched_patterns = []
        
        for pattern_name, pattern in self.technical_patterns.items():
            if pattern.fullmatch(token):
                matched_patterns.append(pattern_name)
                # Different patterns get different weights
                if pattern_name in ['version_number', 'model_number', 'sku_pattern']:
                    score = max(score, 1.0)
                elif pattern_name in ['date_quarter', 'currency', 'measurement']:
                    score = max(score, 0.9)
                elif pattern_name in ['percentage', 'phone_number', 'email_like']:
                    score = max(score, 0.8)
                else:
                    score = max(score, 0.7)
        
        # Check for digits and hyphens (general technical indicators)
        if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            score = max(score, 0.6)
            matched_patterns.append('alphanumeric_mix')
        
        if '-' in token and len(token) > 3:
            score = max(score, 0.5)
            matched_patterns.append('hyphenated')
        
        return score, matched_patterns
    
    def calculate_unit_proximity_score(self, token: str, context_words: List[str]) -> float:
        """Score based on proximity to unit markers"""
        score = 0.0
        
        # Check if token itself is a unit marker
        if token.lower() in self.unit_markers:
            return 1.0
        
        # Check context for unit markers (within 2 words)
        context_lower = [w.lower() for w in context_words]
        for unit in self.unit_markers:
            if unit in context_lower:
                # Higher score for closer proximity
                try:
                    unit_index = context_lower.index(unit)
                    token_context_pos = len(context_words) // 2  # Assume token is in middle
                    distance = abs(unit_index - token_context_pos)
                    if distance <= 1:
                        score = max(score, 0.8)
                    elif distance <= 2:
                        score = max(score, 0.6)
                    else:
                        score = max(score, 0.4)
                except ValueError:
                    continue
        
        return score

class VariantClusterer:
    """Clusters visually similar term variants"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.logger = create_enhanced_logger("variant_clusterer")
    
    def calculate_visual_similarity(self, term1: str, term2: str) -> float:
        """Calculate visual similarity between two terms"""
        # Normalize both terms
        norm1 = self._normalize_term(term1)
        norm2 = self._normalize_term(term2)
        
        # Use sequence matcher for basic similarity
        basic_similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Additional similarity checks
        edit_distance_similarity = self._edit_distance_similarity(norm1, norm2)
        token_similarity = self._token_overlap_similarity(term1, term2)
        
        # Weighted combination
        final_similarity = (
            basic_similarity * 0.4 +
            edit_distance_similarity * 0.3 +
            token_similarity * 0.3
        )
        
        return final_similarity
    
    def _normalize_term(self, term: str) -> str:
        """Normalize term for comparison"""
        # Remove non-alphanumeric except spaces and hyphens
        normalized = re.sub(r'[^\w\s-]', '', term.lower())
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        # Remove accents
        normalized = unicodedata.normalize('NFKD', normalized)
        normalized = ''.join(c for c in normalized if not unicodedata.combining(c))
        return normalized
    
    def _edit_distance_similarity(self, term1: str, term2: str) -> float:
        """Calculate similarity based on edit distance"""
        if not term1 or not term2:
            return 0.0
        
        max_len = max(len(term1), len(term2))
        if max_len == 0:
            return 1.0
        
        # Simple edit distance implementation
        distances = [[0] * (len(term2) + 1) for _ in range(len(term1) + 1)]
        
        for i in range(len(term1) + 1):
            distances[i][0] = i
        for j in range(len(term2) + 1):
            distances[0][j] = j
        
        for i in range(1, len(term1) + 1):
            for j in range(1, len(term2) + 1):
                cost = 0 if term1[i-1] == term2[j-1] else 1
                distances[i][j] = min(
                    distances[i-1][j] + 1,      # deletion
                    distances[i][j-1] + 1,      # insertion
                    distances[i-1][j-1] + cost  # substitution
                )
        
        edit_distance = distances[len(term1)][len(term2)]
        return 1.0 - (edit_distance / max_len)
    
    def _token_overlap_similarity(self, term1: str, term2: str) -> float:
        """Calculate similarity based on token overlap"""
        tokens1 = set(re.split(r'[\s\-_]+', term1.lower()))
        tokens2 = set(re.split(r'[\s\-_]+', term2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def cluster_variants(self, terms: List[str]) -> Dict[str, List[str]]:
        """Cluster terms into variant groups"""
        if not terms:
            return {}
        
        clusters = {}
        processed = set()
        
        for i, term1 in enumerate(terms):
            if term1 in processed:
                continue
            
            # Start new cluster with this term as representative
            cluster_key = term1
            cluster_members = [term1]
            processed.add(term1)
            
            # Find similar terms
            for j, term2 in enumerate(terms[i+1:], i+1):
                if term2 in processed:
                    continue
                
                similarity = self.calculate_visual_similarity(term1, term2)
                if similarity >= self.similarity_threshold:
                    cluster_members.append(term2)
                    processed.add(term2)
            
            # Only create cluster if it has multiple members or is a high-value single term
            if len(cluster_members) > 1:
                clusters[cluster_key] = cluster_members
                self.logger.debug(f"Created variant cluster", 
                                context={'representative': cluster_key, 
                                        'variants': cluster_members,
                                        'size': len(cluster_members)})
        
        return clusters

class TermMiningEngine:
    """Main term mining engine"""
    
    def __init__(self, 
                 session_id: str = None,
                 mining_sensitivity: float = 0.5,
                 min_frequency_threshold: int = 2,
                 max_candidates_per_session: int = 200,
                 enable_variant_clustering: bool = True):
        """
        Initialize term mining engine
        
        Args:
            session_id: Unique session identifier
            mining_sensitivity: Sensitivity threshold (0.0-1.0, higher = more selective)
            min_frequency_threshold: Minimum frequency to consider a term
            max_candidates_per_session: Maximum candidates to return per session
            enable_variant_clustering: Whether to cluster similar variants
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.mining_sensitivity = mining_sensitivity
        self.min_frequency_threshold = min_frequency_threshold
        self.max_candidates_per_session = max_candidates_per_session
        self.enable_variant_clustering = enable_variant_clustering
        
        # Initialize components
        self.background_corpus = BackgroundCorpus()
        self.pattern_analyzer = PatternAnalyzer()
        self.variant_clusterer = VariantClusterer() if enable_variant_clustering else None
        
        self.logger = create_enhanced_logger("term_mining_engine")
        
        self.logger.info("Term mining engine initialized", 
                        context={'session_id': self.session_id,
                                'sensitivity': mining_sensitivity,
                                'min_frequency': min_frequency_threshold,
                                'max_candidates': max_candidates_per_session,
                                'variant_clustering': enable_variant_clustering})
    
    def mine_terms_from_hypotheses(self, 
                                  asr_results: List[Dict[str, Any]], 
                                  diarization_results: Dict[str, Any] = None) -> SessionTermResults:
        """
        Mine terms from first-pass ASR hypotheses
        
        Args:
            asr_results: List of ASR results from different engines/variants
            diarization_results: Optional diarization data for multi-speaker analysis
            
        Returns:
            SessionTermResults with scored candidates
        """
        start_time = time.time()
        self.logger.info("Starting term mining from ASR hypotheses", 
                        context={'asr_results_count': len(asr_results),
                                'has_diarization': diarization_results is not None})
        
        # Step 1: Extract and tokenize all text from hypotheses
        all_tokens = self._extract_tokens_from_results(asr_results)
        self.logger.debug("Token extraction complete", 
                         context={'total_tokens': len(all_tokens)})
        
        # Step 2: Calculate frequency statistics
        token_stats = self._calculate_token_statistics(all_tokens, asr_results, diarization_results)
        self.logger.debug("Token statistics calculated", 
                         context={'unique_tokens': len(token_stats)})
        
        # Step 3: Score each candidate token
        candidates = []
        for token, stats in token_stats.items():
            if self._should_skip_token(token, stats):
                continue
            
            candidate = self._score_term_candidate(token, stats, all_tokens)
            if candidate and candidate.final_mining_score >= self.mining_sensitivity:
                candidates.append(candidate)
        
        self.logger.debug("Initial candidate scoring complete", 
                         context={'initial_candidates': len(candidates)})
        
        # Step 4: Apply variant clustering if enabled
        if self.enable_variant_clustering and self.variant_clusterer:
            candidates = self._apply_variant_clustering(candidates)
            self.logger.debug("Variant clustering applied", 
                             context={'final_candidates': len(candidates)})
        
        # Step 5: Rank and limit candidates
        candidates.sort(key=lambda c: c.final_mining_score, reverse=True)
        candidates = candidates[:self.max_candidates_per_session]
        
        # Step 6: Generate results summary
        processing_time = time.time() - start_time
        results = self._generate_session_results(candidates, processing_time)
        
        self.logger.info("Term mining completed", 
                        context={'candidates_found': len(candidates),
                                'high_confidence': results.high_confidence_candidates,
                                'technical_terms': results.technical_term_count,
                                'proper_nouns': results.proper_noun_count,
                                'processing_time': processing_time})
        
        return results
    
    def _extract_tokens_from_results(self, asr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and annotate tokens from ASR results"""
        all_tokens = []
        
        for result_idx, asr_result in enumerate(asr_results):
            engine_name = asr_result.get('engine', f'engine_{result_idx}')
            segments = asr_result.get('segments', [])
            
            for segment_idx, segment in enumerate(segments):
                segment_text = segment.get('text', '')
                segment_start = segment.get('start', 0.0)
                segment_end = segment.get('end', 0.0)
                speaker_id = segment.get('speaker_id')
                
                # Tokenize segment text
                tokens = self._tokenize_text(segment_text)
                
                for token_idx, token in enumerate(tokens):
                    # Estimate token timing within segment
                    token_start = segment_start + (token_idx / len(tokens)) * (segment_end - segment_start)
                    
                    # Get surrounding context
                    context_start = max(0, token_idx - 3)
                    context_end = min(len(tokens), token_idx + 4)
                    context = tokens[context_start:context_end]
                    
                    all_tokens.append({
                        'token': token,
                        'start_time': token_start,
                        'engine': engine_name,
                        'speaker_id': speaker_id,
                        'segment_idx': segment_idx,
                        'context': context,
                        'source_segment': segment
                    })
        
        return all_tokens
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text preserving meaningful patterns"""
        # Clean text but preserve structure
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on whitespace but preserve hyphenated words and special patterns
        tokens = []
        words = text.split()
        
        for word in words:
            # Remove leading/trailing punctuation but preserve internal patterns
            cleaned = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', word)
            if cleaned and len(cleaned) > 0:
                tokens.append(cleaned)
        
        return tokens
    
    def _calculate_token_statistics(self, 
                                   all_tokens: List[Dict[str, Any]], 
                                   asr_results: List[Dict[str, Any]],
                                   diarization_results: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive statistics for each token"""
        token_stats = defaultdict(lambda: {
            'frequency': 0,
            'engines': set(),
            'speakers': set(),
            'first_seen': float('inf'),
            'contexts': [],
            'source_segments': []
        })
        
        # Calculate basic statistics
        for token_info in all_tokens:
            token = token_info['token']
            stats = token_stats[token]
            
            stats['frequency'] += 1
            stats['engines'].add(token_info['engine'])
            if token_info['speaker_id']:
                stats['speakers'].add(token_info['speaker_id'])
            stats['first_seen'] = min(stats['first_seen'], token_info['start_time'])
            stats['contexts'].append(token_info['context'])
            stats['source_segments'].append(token_info['source_segment'])
        
        # Convert sets to counts for JSON serialization later
        for token, stats in token_stats.items():
            stats['engine_count'] = len(stats['engines'])
            stats['speaker_count'] = len(stats['speakers'])
            stats['engines'] = list(stats['engines'])  # Convert set to list
            stats['speakers'] = list(stats['speakers'])  # Convert set to list
        
        return dict(token_stats)
    
    def _should_skip_token(self, token: str, stats: Dict[str, Any]) -> bool:
        """Determine if token should be skipped"""
        # Skip if too short or too long
        if len(token) < 2 or len(token) > 50:
            return True
        
        # Skip if frequency is too low
        if stats['frequency'] < self.min_frequency_threshold:
            return True
        
        # Skip stopwords
        if self.background_corpus.is_stopword(token):
            return True
        
        # Skip profanity
        if self.background_corpus.is_profanity(token):
            return True
        
        # Skip if purely numeric (unless it's a year)
        if token.isdigit() and not (1900 <= int(token) <= 2100):
            return True
        
        return False
    
    def _score_term_candidate(self, token: str, stats: Dict[str, Any], all_tokens: List[Dict[str, Any]]) -> Optional[TermCandidate]:
        """Score a term candidate using multiple criteria"""
        try:
            # Calculate TF-IDF score
            tf = stats['frequency']
            total_tokens = len(all_tokens)
            tf_normalized = tf / total_tokens if total_tokens > 0 else 0
            
            background_prob = self.background_corpus.get_background_probability(token)
            tf_idf_score = tf_normalized / background_prob if background_prob > 0 else tf_normalized
            
            # Normalize TF-IDF to 0-1 scale using log
            frequency_score = min(1.0, math.log(tf_idf_score + 1) / math.log(100))
            
            # Calculate case pattern score
            case_pattern_score = self.pattern_analyzer.analyze_case_pattern(token)
            
            # Calculate technical pattern score
            technical_pattern_score, matched_patterns = self.pattern_analyzer.analyze_technical_patterns(token)
            
            # Calculate multi-speaker agreement score
            multi_speaker_score = min(1.0, stats['speaker_count'] / 3.0) if stats['speaker_count'] > 0 else 0.0
            
            # Calculate unit proximity score (using first context as representative)
            first_context = stats['contexts'][0] if stats['contexts'] else []
            unit_proximity_score = self.pattern_analyzer.calculate_unit_proximity_score(token, first_context)
            
            # Weight combination for final score
            final_score = (
                frequency_score * 0.25 +
                case_pattern_score * 0.20 +
                technical_pattern_score * 0.20 +
                multi_speaker_score * 0.15 +
                unit_proximity_score * 0.20
            )
            
            # Create candidate
            candidate = TermCandidate(
                token=token,
                weight=final_score,
                first_seen_time=stats['first_seen'],
                supporting_engines=set(stats['engines']),
                local_context=first_context,
                frequency_score=frequency_score,
                case_pattern_score=case_pattern_score,
                technical_pattern_score=technical_pattern_score,
                multi_speaker_score=multi_speaker_score,
                unit_proximity_score=unit_proximity_score,
                final_mining_score=final_score,
                source_segments=stats['source_segments'][:5],  # Limit to 5 examples
                metadata={
                    'frequency': stats['frequency'],
                    'engine_count': stats['engine_count'],
                    'speaker_count': stats['speaker_count'],
                    'matched_technical_patterns': matched_patterns if 'matched_patterns' in locals() else []
                }
            )
            
            return candidate
            
        except Exception as e:
            self.logger.warning(f"Error scoring term candidate '{token}'", 
                              context={'error': str(e)})
            return None
    
    def _apply_variant_clustering(self, candidates: List[TermCandidate]) -> List[TermCandidate]:
        """Apply variant clustering to merge similar terms"""
        if not self.variant_clusterer:
            return candidates
        
        # Extract tokens for clustering
        tokens = [c.token for c in candidates]
        clusters = self.variant_clusterer.cluster_variants(tokens)
        
        if not clusters:
            return candidates
        
        # Create map from token to candidate
        token_to_candidate = {c.token: c for c in candidates}
        
        # Merge clustered candidates
        merged_candidates = []
        processed_tokens = set()
        
        for representative, variants in clusters.items():
            if representative in processed_tokens:
                continue
            
            # Get the representative candidate (highest scoring variant)
            variant_candidates = [token_to_candidate[token] for token in variants if token in token_to_candidate]
            if not variant_candidates:
                continue
            
            best_candidate = max(variant_candidates, key=lambda c: c.final_mining_score)
            
            # Merge information from all variants
            all_engines = set()
            all_contexts = []
            all_segments = []
            total_frequency = 0
            
            for variant_candidate in variant_candidates:
                all_engines.update(variant_candidate.supporting_engines)
                all_contexts.extend([variant_candidate.local_context])
                all_segments.extend(variant_candidate.source_segments)
                total_frequency += variant_candidate.metadata.get('frequency', 1)
                processed_tokens.add(variant_candidate.token)
            
            # Update best candidate with merged information
            best_candidate.supporting_engines = all_engines
            best_candidate.variants = set(variants) - {best_candidate.token}
            best_candidate.source_segments = all_segments[:10]  # Limit examples
            best_candidate.metadata['total_variant_frequency'] = total_frequency
            best_candidate.metadata['variant_count'] = len(variants)
            
            # Boost score for having variants
            variant_boost = min(0.2, len(variants) * 0.05)
            best_candidate.final_mining_score = min(1.0, best_candidate.final_mining_score + variant_boost)
            
            merged_candidates.append(best_candidate)
        
        # Add non-clustered candidates
        for candidate in candidates:
            if candidate.token not in processed_tokens:
                merged_candidates.append(candidate)
        
        return merged_candidates
    
    def _generate_session_results(self, candidates: List[TermCandidate], processing_time: float) -> SessionTermResults:
        """Generate comprehensive session results"""
        high_confidence_count = sum(1 for c in candidates if c.final_mining_score >= 0.8)
        technical_term_count = sum(1 for c in candidates if c.technical_pattern_score >= 0.5)
        proper_noun_count = sum(1 for c in candidates if c.case_pattern_score >= 0.8)
        multi_speaker_agreements = sum(1 for c in candidates if c.multi_speaker_score >= 0.5)
        
        mining_metadata = {
            'session_id': self.session_id,
            'mining_sensitivity': self.mining_sensitivity,
            'variant_clustering_enabled': self.enable_variant_clustering,
            'background_corpus_size': self.background_corpus.total_background_tokens,
            'average_candidate_score': np.mean([c.final_mining_score for c in candidates]) if candidates else 0.0,
            'score_distribution': {
                'high': high_confidence_count,
                'medium': sum(1 for c in candidates if 0.5 <= c.final_mining_score < 0.8),
                'low': sum(1 for c in candidates if c.final_mining_score < 0.5)
            }
        }
        
        return SessionTermResults(
            candidates=candidates,
            total_candidates=len(candidates),
            high_confidence_candidates=high_confidence_count,
            technical_term_count=technical_term_count,
            proper_noun_count=proper_noun_count,
            multi_speaker_agreements=multi_speaker_agreements,
            processing_time=processing_time,
            mining_metadata=mining_metadata
        )
    
    def export_session_candidates(self, results: SessionTermResults, output_path: str) -> bool:
        """Export session candidates to JSON file"""
        try:
            # Convert candidates to serializable format
            candidates_data = []
            for candidate in results.candidates:
                candidate_dict = {
                    'token': candidate.token,
                    'weight': candidate.weight,
                    'first_seen_time': candidate.first_seen_time,
                    'supporting_engines': list(candidate.supporting_engines),
                    'local_context': candidate.local_context,
                    'scores': {
                        'frequency': candidate.frequency_score,
                        'case_pattern': candidate.case_pattern_score,
                        'technical_pattern': candidate.technical_pattern_score,
                        'multi_speaker': candidate.multi_speaker_score,
                        'unit_proximity': candidate.unit_proximity_score,
                        'final_mining': candidate.final_mining_score
                    },
                    'variants': list(candidate.variants),
                    'metadata': candidate.metadata,
                    'source_segments_count': len(candidate.source_segments)
                }
                candidates_data.append(candidate_dict)
            
            # Create export data
            export_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'candidates': candidates_data,
                'session_summary': {
                    'total_candidates': results.total_candidates,
                    'high_confidence_candidates': results.high_confidence_candidates,
                    'technical_term_count': results.technical_term_count,
                    'proper_noun_count': results.proper_noun_count,
                    'multi_speaker_agreements': results.multi_speaker_agreements,
                    'processing_time': results.processing_time
                },
                'mining_metadata': results.mining_metadata
            }
            
            # Write to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Session candidates exported successfully", 
                           context={'output_path': output_path, 'candidates': len(candidates_data)})
            return True
            
        except Exception as e:
            self.logger.error("Failed to export session candidates", 
                            context={'output_path': output_path, 'error': str(e)})
            return False

def create_term_mining_engine(session_id: str = None, **config) -> TermMiningEngine:
    """Factory function to create term mining engine with configuration"""
    return TermMiningEngine(
        session_id=session_id,
        mining_sensitivity=config.get('mining_sensitivity', 0.5),
        min_frequency_threshold=config.get('min_frequency_threshold', 2),
        max_candidates_per_session=config.get('max_candidates_per_session', 200),
        enable_variant_clustering=config.get('enable_variant_clustering', True)
    )

# Example usage and testing
if __name__ == "__main__":
    # Test the term mining engine
    mining_engine = create_term_mining_engine(session_id="test_session")
    
    # Mock ASR results for testing
    mock_asr_results = [
        {
            'engine': 'whisper',
            'segments': [
                {
                    'start': 0.0, 'end': 5.0, 'text': 'Our Q3-2024 revenue was $2.5M with Model X-150 sales up 15%',
                    'speaker_id': 'speaker_1'
                },
                {
                    'start': 5.0, 'end': 10.0, 'text': 'The SKU-12345 performed well in the Northeast region',
                    'speaker_id': 'speaker_2'
                }
            ]
        },
        {
            'engine': 'deepgram',
            'segments': [
                {
                    'start': 0.0, 'end': 5.0, 'text': 'Our Q3 2024 revenue reached $2.5 million with ModelX150 sales increasing 15%',
                    'speaker_id': 'speaker_1'
                }
            ]
        }
    ]
    
    # Mine terms
    results = mining_engine.mine_terms_from_hypotheses(mock_asr_results)
    
    # Export results
    mining_engine.export_session_candidates(results, '/tmp/session_term_candidates.json')
    
    print(f"Mined {results.total_candidates} term candidates in {results.processing_time:.2f}s")
    for candidate in results.candidates[:5]:
        print(f"  {candidate.token}: {candidate.final_mining_score:.3f} (variants: {len(candidate.variants)})")