"""
Robust Text Normalization Engine

Provides comprehensive text normalization with timestamp-anchored punctuation,
acronym protection, semantic disfluency removal, and multiple style presets.
Includes comprehensive guardrail systems to prevent token invention.
"""

import re
import time
import yaml
import math
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter
import difflib

from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class WordToken:
    """Represents a single word token with timing and metadata"""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    is_protected: bool = False  # Protected from normalization
    protection_reason: str = ""  # Why it was protected
    original_index: int = 0  # Original position in segment

@dataclass 
class NormalizationChange:
    """Tracks a single normalization change"""
    change_type: str  # "punctuation", "capitalization", "disfluency", "formatting"
    original_text: str
    normalized_text: str
    position: int  # Character position
    confidence: float
    timestamp: Optional[float] = None
    reason: str = ""

@dataclass
class SegmentNormalizationResult:
    """Result of normalizing a single segment"""
    original_text: str
    normalized_text: str
    original_tokens: List[WordToken]
    normalized_tokens: List[WordToken]
    changes: List[NormalizationChange]
    readability_score_before: float
    readability_score_after: float
    guardrail_violations: List[str]
    processing_time_ms: float
    profile_used: str
    profile_downgrades: List[str] = field(default_factory=list)

@dataclass
class NormalizationMetrics:
    """Comprehensive normalization metrics"""
    tokens_changed_count: int = 0
    tokens_added_count: int = 0
    tokens_removed_count: int = 0
    sentences_adjusted_count: int = 0
    fillers_removed_count: int = 0
    acronyms_protected_count: int = 0
    guardrail_violations_count: int = 0
    profile_downgrades_count: int = 0
    avg_readability_improvement: float = 0.0
    total_processing_time_ms: float = 0.0

class AcronymProtector:
    """Handles protection of acronyms and technical terms"""
    
    def __init__(self, patterns: Dict[str, List[str]]):
        self.patterns = patterns
        self.logger = create_enhanced_logger("acronym_protector")
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for category, pattern_list in patterns.items():
            self.compiled_patterns[category] = [re.compile(p) for p in pattern_list]
    
    def identify_protected_tokens(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Identify tokens that should be protected from normalization
        
        Returns:
            List of (start_pos, end_pos, token_text, protection_reason) tuples
        """
        protected = []
        
        for category, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    protected.append((
                        match.start(),
                        match.end(),
                        match.group(),
                        f"protected_{category}"
                    ))
        
        # Sort by start position and merge overlapping protections
        protected.sort(key=lambda x: x[0])
        merged = []
        
        for start, end, token, reason in protected:
            if merged and start <= merged[-1][1]:
                # Overlapping - extend the previous protection
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]), 
                             text[merged[-1][0]:max(end, merged[-1][1])], 
                             f"{merged[-1][3]}+{reason}")
            else:
                merged.append((start, end, token, reason))
        
        return merged

class SemanticDisfluencyAnalyzer:
    """Analyzes disfluencies with semantic awareness"""
    
    def __init__(self, profile_config: Dict[str, Any]):
        self.config = profile_config
        self.logger = create_enhanced_logger("semantic_disfluency")
        
        # Load semantic filler words from global config
        self.semantic_fillers = set(profile_config.get('semantic_fillers', []))
        
        # Context patterns for semantic analysis
        self.meaningful_contexts = {
            'like': [
                r'\bi\s+like\s+\w+',           # "I like..."
                r'\blike\s+(this|that|it)',    # "like this"
                r'\blook\s+like',              # "look like"
                r'\bfeel\s+like'               # "feel like"
            ],
            'right': [
                r'\bturn\s+right',             # "turn right"
                r'\bright\s+(here|there|now)', # "right here"
                r'\ball\s+right',              # "all right"
                r'\byou\'?re\s+right'          # "you're right"
            ],
            'okay': [
                r'\bthat\'?s\s+okay',          # "that's okay"
                r'\bokay\s+with',              # "okay with"
                r'\bokay\s+to',                # "okay to"
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_contexts = {}
        for filler, patterns in self.meaningful_contexts.items():
            self.compiled_contexts[filler] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def analyze_filler_context(self, text: str, filler_word: str, position: int, 
                              context_window: int = 3) -> Tuple[bool, str]:
        """
        Analyze if a filler word is semantically meaningful in context
        
        Returns:
            (is_meaningful, reason) tuple
        """
        if filler_word.lower() not in self.semantic_fillers:
            return False, "not_semantic_filler"
        
        # Extract context window around the filler
        words = text.split()
        filler_idx = None
        
        # Find the filler word position
        for i, word in enumerate(words):
            if re.sub(r'[^\w]', '', word.lower()) == filler_word.lower():
                if abs(i - position) <= 1:  # Allow some position tolerance
                    filler_idx = i
                    break
        
        if filler_idx is None:
            return False, "filler_not_found"
        
        # Extract context
        start_idx = max(0, filler_idx - context_window)
        end_idx = min(len(words), filler_idx + context_window + 1)
        context = ' '.join(words[start_idx:end_idx])
        
        # Check for meaningful patterns
        if filler_word.lower() in self.compiled_contexts:
            for pattern in self.compiled_contexts[filler_word.lower()]:
                if pattern.search(context):
                    return True, f"meaningful_pattern_{pattern.pattern}"
        
        # Additional semantic checks
        if filler_word.lower() == "actually":
            # "Actually" at sentence start is often meaningful
            if filler_idx == 0 or words[filler_idx - 1].endswith('.'):
                return True, "sentence_start_actually"
        
        if filler_word.lower() == "basically":
            # "Basically" followed by explanation is often meaningful
            if filler_idx < len(words) - 1:
                next_word = words[filler_idx + 1].lower()
                if next_word in ['what', 'how', 'the', 'we', 'this']:
                    return True, "explanatory_basically"
        
        return False, "filler_without_semantic_value"

class ReadabilityScorer:
    """Calculates readability scores for text"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factors = config.get('factors', {})
        
    def calculate_score(self, text: str, word_tokens: List[WordToken] = None) -> float:
        """Calculate comprehensive readability score (0.0 to 1.0)"""
        if not text.strip():
            return 0.0
        
        scores = {}
        
        # Sentence length factor
        sentences = self._split_sentences(text)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Optimal range: 15-20 words per sentence
            scores['sentence_length'] = max(0, 1.0 - abs(avg_sentence_length - 17.5) / 17.5)
        else:
            scores['sentence_length'] = 0.5
        
        # Punctuation density factor
        words = text.split()
        punctuation_count = len(re.findall(r'[.!?,:;]', text))
        if words:
            punctuation_density = punctuation_count / len(words)
            # Optimal range: 0.05 to 0.15 punctuation per word
            scores['punctuation_density'] = max(0, 1.0 - abs(punctuation_density - 0.1) / 0.1)
        else:
            scores['punctuation_density'] = 0.5
        
        # Disfluency ratio factor
        filler_words = len(re.findall(r'\b(um|uh|er|ah|like|you know|I mean)\b', text, re.IGNORECASE))
        if words:
            disfluency_ratio = filler_words / len(words)
            scores['disfluency_ratio'] = max(0, 1.0 - disfluency_ratio * 5)  # Penalize disfluencies
        else:
            scores['disfluency_ratio'] = 1.0
        
        # Capitalization consistency factor
        sentences_with_caps = len([s for s in sentences if s and s[0].isupper()])
        if sentences:
            cap_consistency = sentences_with_caps / len(sentences)
            scores['capitalization_consistency'] = cap_consistency
        else:
            scores['capitalization_consistency'] = 0.5
        
        # Technical term preservation (if tokens provided)
        if word_tokens:
            protected_count = len([t for t in word_tokens if t.is_protected])
            if protected_count > 0:
                scores['technical_term_preservation'] = 1.0
            else:
                scores['technical_term_preservation'] = 0.8
        else:
            scores['technical_term_preservation'] = 0.8
        
        # Flow continuity factor (word repetition penalty)
        word_counter = Counter(word.lower() for word in words if len(word) > 3)
        repeated_words = sum(max(0, count - 1) for count in word_counter.values())
        if words:
            repetition_ratio = repeated_words / len(words)
            scores['flow_continuity'] = max(0, 1.0 - repetition_ratio * 2)
        else:
            scores['flow_continuity'] = 1.0
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for factor, weight in self.factors.items():
            if factor in scores:
                total_score += scores[factor] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - could be enhanced with more sophisticated NLP
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

class TextNormalizer:
    """Main text normalization engine with guardrails"""
    
    def __init__(self, config_path: str = "config/normalization_profiles.yaml"):
        """Initialize the text normalizer"""
        self.logger = create_enhanced_logger("text_normalizer")
        
        # Load configuration
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.acronym_protector = AcronymProtector(
            self.config['global_settings']['protected_patterns']
        )
        self.readability_scorer = ReadabilityScorer(
            self.config.get('readability_metrics', {})
        )
        
        # Initialize metrics
        self.metrics = NormalizationMetrics()
        
        self.logger.info("Text normalizer initialized", 
                        context={'profiles': list(self.config['profiles'].keys())})
    
    def _load_config(self) -> Dict[str, Any]:
        """Load normalization configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            # Return minimal fallback config
            return {
                'global_settings': {'protected_patterns': {'acronyms': [], 'technical_terms': [], 'numbers': []}},
                'profiles': {'light': {'description': 'fallback', 'enabled': True}},
                'readability_metrics': {'factors': {}},
                'guardrails': {'token_preservation': {'enabled': True}}
            }
    
    def normalize_segments(self, segments: List[Dict[str, Any]], 
                          profile: str = "readable") -> List[SegmentNormalizationResult]:
        """
        Normalize a list of transcript segments
        
        Args:
            segments: List of segments with 'text', 'start', 'end', 'speaker' keys
            profile: Normalization profile to use
            
        Returns:
            List of SegmentNormalizationResult objects
        """
        start_time = time.time()
        
        if profile not in self.config['profiles']:
            self.logger.warning(f"Unknown profile '{profile}', using 'light'")
            profile = "light"
        
        if not self.config['profiles'][profile].get('enabled', True):
            self.logger.warning(f"Profile '{profile}' disabled, using 'light'")
            profile = "light"
        
        profile_config = self.config['profiles'][profile]
        self.logger.info(f"Normalizing {len(segments)} segments with profile '{profile}'")
        
        results = []
        for i, segment in enumerate(segments):
            try:
                segment_result = self._normalize_segment(segment, profile, profile_config)
                results.append(segment_result)
                
                # Update global metrics
                self._update_metrics(segment_result)
                
            except Exception as e:
                self.logger.error(f"Failed to normalize segment {i}: {e}")
                # Create fallback result
                fallback_result = SegmentNormalizationResult(
                    original_text=segment.get('text', ''),
                    normalized_text=segment.get('text', ''),
                    original_tokens=[],
                    normalized_tokens=[],
                    changes=[],
                    readability_score_before=0.5,
                    readability_score_after=0.5,
                    guardrail_violations=[f"normalization_failed: {e}"],
                    processing_time_ms=0.0,
                    profile_used=profile,
                    profile_downgrades=["failed_to_process"]
                )
                results.append(fallback_result)
        
        total_time = (time.time() - start_time) * 1000
        self.metrics.total_processing_time_ms += total_time
        
        self.logger.info(f"Completed normalization", context={
            'segments_processed': len(results),
            'total_time_ms': total_time,
            'avg_time_per_segment_ms': total_time / len(results) if results else 0
        })
        
        return results
    
    def _normalize_segment(self, segment: Dict[str, Any], profile: str, 
                          profile_config: Dict[str, Any]) -> SegmentNormalizationResult:
        """Normalize a single segment"""
        start_time = time.time()
        
        original_text = segment.get('text', '').strip()
        if not original_text:
            return self._create_empty_result(original_text, profile)
        
        # Tokenize with timing information
        original_tokens = self._tokenize_with_timing(segment)
        
        # Calculate initial readability score
        readability_before = self.readability_scorer.calculate_score(original_text, original_tokens)
        
        # Identify protected tokens
        protected_regions = self.acronym_protector.identify_protected_tokens(original_text)
        self._mark_protected_tokens(original_tokens, protected_regions)
        
        # Apply normalization steps
        working_text = original_text
        working_tokens = original_tokens.copy()
        changes = []
        guardrail_violations = []
        profile_downgrades = []
        
        # Initialize semantic analyzer for this profile
        semantic_analyzer = SemanticDisfluencyAnalyzer(
            self.config['global_settings']
        )
        
        # Step 1: Disfluency removal (if enabled)
        if profile_config.get('disfluency_removal', {}).get('enabled', False):
            working_text, disfluency_changes, violations = self._apply_disfluency_removal(
                working_text, working_tokens, profile_config, semantic_analyzer
            )
            changes.extend(disfluency_changes)
            guardrail_violations.extend(violations)
        
        # Step 2: Punctuation normalization
        if any(profile_config.get('punctuation', {}).values()):
            working_text, punct_changes, violations = self._apply_punctuation_normalization(
                working_text, working_tokens, profile_config
            )
            changes.extend(punct_changes)
            guardrail_violations.extend(violations)
        
        # Step 3: Capitalization normalization
        if any(profile_config.get('capitalization', {}).values()):
            working_text, cap_changes, violations = self._apply_capitalization_normalization(
                working_text, working_tokens, profile_config
            )
            changes.extend(cap_changes)
            guardrail_violations.extend(violations)
        
        # Step 4: Number and style formatting
        if any(profile_config.get('number_formatting', {}).values()) or \
           any(profile_config.get('style', {}).values()):
            working_text, format_changes, violations = self._apply_formatting_normalization(
                working_text, working_tokens, profile_config
            )
            changes.extend(format_changes)
            guardrail_violations.extend(violations)
        
        # Final guardrail check
        final_violations = self._validate_final_result(
            original_text, working_text, original_tokens, profile_config
        )
        guardrail_violations.extend(final_violations)
        
        # If guardrails violated, downgrade profile
        if guardrail_violations and profile != "verbatim":
            downgraded_profile = self._get_downgrade_profile(profile)
            if downgraded_profile != profile:
                self.logger.warning(f"Guardrail violations detected, downgrading from '{profile}' to '{downgraded_profile}'")
                profile_downgrades.append(f"{profile}->{downgraded_profile}")
                # Recursively apply with downgraded profile
                return self._normalize_segment(segment, downgraded_profile, 
                                              self.config['profiles'][downgraded_profile])
        
        # Calculate final readability score
        normalized_tokens = self._retokenize_normalized_text(working_text, original_tokens)
        readability_after = self.readability_scorer.calculate_score(working_text, normalized_tokens)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SegmentNormalizationResult(
            original_text=original_text,
            normalized_text=working_text,
            original_tokens=original_tokens,
            normalized_tokens=normalized_tokens,
            changes=changes,
            readability_score_before=readability_before,
            readability_score_after=readability_after,
            guardrail_violations=guardrail_violations,
            processing_time_ms=processing_time,
            profile_used=profile,
            profile_downgrades=profile_downgrades
        )
    
    def _tokenize_with_timing(self, segment: Dict[str, Any]) -> List[WordToken]:
        """Tokenize segment text with timing information"""
        text = segment.get('text', '')
        start_time = segment.get('start', 0.0)
        end_time = segment.get('end', 0.0)
        
        # Extract word-level timing if available
        words_data = segment.get('words', [])
        
        if words_data and len(words_data) > 0:
            # Use actual word timing data
            tokens = []
            for i, word_data in enumerate(words_data):
                token = WordToken(
                    text=word_data.get('word', ''),
                    start_time=word_data.get('start', start_time),
                    end_time=word_data.get('end', end_time),
                    confidence=word_data.get('confidence', 1.0),
                    original_index=i
                )
                tokens.append(token)
            return tokens
        else:
            # Estimate timing for each word
            words = text.split()
            if not words:
                return []
            
            duration = end_time - start_time
            word_duration = duration / len(words) if len(words) > 0 else 0.0
            
            tokens = []
            for i, word in enumerate(words):
                word_start = start_time + (i * word_duration)
                word_end = word_start + word_duration
                
                token = WordToken(
                    text=word,
                    start_time=word_start,
                    end_time=word_end,
                    confidence=1.0,
                    original_index=i
                )
                tokens.append(token)
            
            return tokens
    
    def _mark_protected_tokens(self, tokens: List[WordToken], 
                              protected_regions: List[Tuple[int, int, str, str]]):
        """Mark tokens that fall within protected regions"""
        text = ' '.join(t.text for t in tokens)
        
        for start_pos, end_pos, protected_text, reason in protected_regions:
            # Find tokens that overlap with this protected region
            current_pos = 0
            for token in tokens:
                token_start = current_pos
                token_end = current_pos + len(token.text)
                
                # Check for overlap
                if (token_start < end_pos and token_end > start_pos):
                    token.is_protected = True
                    token.protection_reason = reason
                    self.metrics.acronyms_protected_count += 1
                
                current_pos = token_end + 1  # +1 for space
    
    def _apply_disfluency_removal(self, text: str, tokens: List[WordToken], 
                                 profile_config: Dict[str, Any],
                                 semantic_analyzer: SemanticDisfluencyAnalyzer) -> Tuple[str, List[NormalizationChange], List[str]]:
        """Apply disfluency removal with semantic awareness"""
        changes = []
        violations = []
        
        disfluency_config = profile_config.get('disfluency_removal', {})
        if not disfluency_config.get('enabled', False):
            return text, changes, violations
        
        # Define filler patterns by strictness level
        strictness = disfluency_config.get('filler_strictness', 'moderate')
        
        filler_patterns = {
            'conservative': [
                r'\b(um|uh|er|ah){2,}\b',  # Only repeated fillers
                r'\b(um|uh|er|ah)\s+(um|uh|er|ah)\b'  # Adjacent fillers
            ],
            'moderate': [
                r'\b(um|uh|er|ah){2,}\b',
                r'\b(um|uh|er|ah)\s+(um|uh|er|ah)\b',
                r'\bum\b(?!\s+(hm|hmm))',  # "um" but not "um hm"
                r'\buh\b(?!\s+(oh|huh))'   # "uh" but not "uh oh"
            ],
            'aggressive': [
                r'\b(um|uh|er|ah)+\b',
                r'\b(you\s+know|I\s+mean)\b',
                r'\b(like)\b(?=\s+(um|uh|you know|I mean))'  # "like" before fillers
            ]
        }
        
        patterns_to_use = filler_patterns.get(strictness, filler_patterns['moderate'])
        
        working_text = text
        removed_count = 0
        
        for pattern in patterns_to_use:
            for match in re.finditer(pattern, working_text, re.IGNORECASE):
                filler_text = match.group()
                
                # Check semantic context if enabled
                if disfluency_config.get('semantic_awareness', True):
                    is_meaningful, reason = semantic_analyzer.analyze_filler_context(
                        working_text, filler_text, match.start()
                    )
                    
                    if is_meaningful:
                        self.logger.debug(f"Preserving meaningful filler '{filler_text}': {reason}")
                        continue
                
                # Remove the filler
                working_text = working_text[:match.start()] + working_text[match.end():]
                
                changes.append(NormalizationChange(
                    change_type="disfluency",
                    original_text=filler_text,
                    normalized_text="",
                    position=match.start(),
                    confidence=0.8,
                    reason=f"filler_removal_{strictness}"
                ))
                
                removed_count += 1
                self.metrics.fillers_removed_count += 1
        
        # Handle repetitions if enabled
        if disfluency_config.get('remove_repetitions', False):
            # Preserve emphasis repetitions
            preserve_emphasis = disfluency_config.get('preserve_emphasis', True)
            
            repetition_pattern = r'\b(\w+)\s+\1\b'
            for match in re.finditer(repetition_pattern, working_text, re.IGNORECASE):
                repeated_word = match.group(1)
                
                # Skip if emphasis should be preserved
                if preserve_emphasis and repeated_word.lower() in ['very', 'really', 'so', 'really']:
                    continue
                
                # Remove one instance of repetition
                working_text = working_text[:match.start()] + repeated_word + working_text[match.end():]
                
                changes.append(NormalizationChange(
                    change_type="disfluency",
                    original_text=match.group(),
                    normalized_text=repeated_word,
                    position=match.start(),
                    confidence=0.7,
                    reason="repetition_removal"
                ))
        
        return working_text, changes, violations
    
    def _apply_punctuation_normalization(self, text: str, tokens: List[WordToken], 
                                        profile_config: Dict[str, Any]) -> Tuple[str, List[NormalizationChange], List[str]]:
        """Apply punctuation normalization with timestamp anchoring"""
        changes = []
        violations = []
        
        punct_config = profile_config.get('punctuation', {})
        if not any(punct_config.values()):
            return text, changes, violations
        
        working_text = text
        
        # Sentence segmentation and period addition
        if punct_config.get('add_periods', False):
            # Find sentence boundaries based on timing gaps and content
            sentences = self._identify_sentence_boundaries(working_text, tokens)
            
            for sentence_end in sentences:
                if sentence_end < len(working_text) and not working_text[sentence_end] in '.!?':
                    # Check timestamp anchoring
                    if self._is_timestamp_anchored(sentence_end, tokens):
                        working_text = working_text[:sentence_end] + '.' + working_text[sentence_end:]
                        
                        changes.append(NormalizationChange(
                            change_type="punctuation",
                            original_text="",
                            normalized_text=".",
                            position=sentence_end,
                            confidence=0.8,
                            reason="sentence_end_period"
                        ))
                        
                        self.metrics.sentences_adjusted_count += 1
        
        # Question mark addition
        if punct_config.get('add_question_marks', False):
            question_patterns = [
                r'\b(what|where|when|why|how|who|which)\b.*?\?',
                r'\b(is|are|was|were|do|does|did|can|could|will|would|should)\s+\w+.*?\?',
                r'\b(right|correct|okay)\?$'
            ]
            
            for pattern in question_patterns:
                for match in re.finditer(pattern, working_text, re.IGNORECASE):
                    if not match.group().endswith('?'):
                        # Add question mark if timestamp-anchored
                        pos = match.end()
                        if self._is_timestamp_anchored(pos, tokens):
                            working_text = working_text[:pos] + '?' + working_text[pos:]
                            
                            changes.append(NormalizationChange(
                                change_type="punctuation",
                                original_text="",
                                normalized_text="?",
                                position=pos,
                                confidence=0.7,
                                reason="question_detection"
                            ))
        
        # Comma addition (conservative)
        if punct_config.get('add_commas', False):
            # Only add commas for clear pause indicators
            comma_patterns = [
                r'\band\s+(?=\w)',  # "and" followed by word
                r'\bbut\s+(?=\w)',  # "but" followed by word
                r'\bor\s+(?=\w)',   # "or" followed by word
            ]
            
            for pattern in comma_patterns:
                for match in re.finditer(pattern, working_text, re.IGNORECASE):
                    # Check if there's already punctuation
                    prev_char = working_text[match.start()-1] if match.start() > 0 else ''
                    if prev_char not in ',.!?;:':
                        pos = match.start()
                        if self._is_timestamp_anchored(pos, tokens):
                            working_text = working_text[:pos] + ', ' + working_text[pos:]
                            
                            changes.append(NormalizationChange(
                                change_type="punctuation",
                                original_text="",
                                normalized_text=", ",
                                position=pos,
                                confidence=0.6,
                                reason="conjunction_comma"
                            ))
        
        return working_text, changes, violations
    
    def _apply_capitalization_normalization(self, text: str, tokens: List[WordToken], 
                                           profile_config: Dict[str, Any]) -> Tuple[str, List[NormalizationChange], List[str]]:
        """Apply capitalization normalization"""
        changes = []
        violations = []
        
        cap_config = profile_config.get('capitalization', {})
        if not any(cap_config.values()):
            return text, changes, violations
        
        working_text = text
        
        # Sentence start capitalization
        if cap_config.get('sentence_starts', False):
            sentences = re.split(r'[.!?]+', working_text)
            
            result_parts = []
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    words = sentence.strip().split()
                    if words and words[0] and not words[0][0].isupper():
                        # Check if first word is protected
                        is_protected = any(token.is_protected and token.text.lower() == words[0].lower() 
                                         for token in tokens)
                        
                        if not is_protected:
                            words[0] = words[0][0].upper() + words[0][1:]
                            
                            changes.append(NormalizationChange(
                                change_type="capitalization",
                                original_text=sentence.strip().split()[0],
                                normalized_text=words[0],
                                position=len(' '.join(result_parts)),
                                confidence=0.9,
                                reason="sentence_start"
                            ))
                    
                    result_parts.append(' '.join(words))
                else:
                    result_parts.append(sentence)
            
            working_text = '. '.join(part for part in result_parts if part.strip())
        
        # Proper noun capitalization (basic patterns)
        if cap_config.get('proper_nouns', False):
            # Days, months, etc.
            proper_patterns = [
                (r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 'day_names'),
                (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', 'month_names'),
                (r'\b(mr|mrs|ms|dr|prof|president|ceo|cto|cfo)\b(?=\s+[A-Z])', 'titles'),
            ]
            
            for pattern, reason in proper_patterns:
                for match in re.finditer(pattern, working_text, re.IGNORECASE):
                    original = match.group()
                    capitalized = original.capitalize()
                    
                    # Check if token is protected
                    is_protected = any(token.is_protected and token.text.lower() == original.lower() 
                                     for token in tokens)
                    
                    if not is_protected and original != capitalized:
                        working_text = working_text[:match.start()] + capitalized + working_text[match.end():]
                        
                        changes.append(NormalizationChange(
                            change_type="capitalization",
                            original_text=original,
                            normalized_text=capitalized,
                            position=match.start(),
                            confidence=0.8,
                            reason=reason
                        ))
        
        return working_text, changes, violations
    
    def _apply_formatting_normalization(self, text: str, tokens: List[WordToken], 
                                       profile_config: Dict[str, Any]) -> Tuple[str, List[NormalizationChange], List[str]]:
        """Apply number and style formatting"""
        changes = []
        violations = []
        
        working_text = text
        
        # Number formatting
        number_config = profile_config.get('number_formatting', {})
        if number_config.get('spell_out_small_numbers', False):
            number_words = {
                '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
                '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
            }
            
            for digit, word in number_words.items():
                pattern = r'\b' + digit + r'\b'
                for match in re.finditer(pattern, working_text):
                    # Check if number is protected
                    is_protected = any(token.is_protected for token in tokens 
                                     if token.text == digit)
                    
                    if not is_protected:
                        working_text = working_text[:match.start()] + word + working_text[match.end():]
                        
                        changes.append(NormalizationChange(
                            change_type="formatting",
                            original_text=digit,
                            normalized_text=word,
                            position=match.start(),
                            confidence=0.9,
                            reason="number_spelling"
                        ))
        
        # Style formatting
        style_config = profile_config.get('style', {})
        
        # Remove excessive spaces
        if style_config.get('remove_excessive_spaces', False):
            # Replace multiple spaces with single space
            original_spaces = len(re.findall(r'\s{2,}', working_text))
            working_text = re.sub(r'\s{2,}', ' ', working_text)
            
            if original_spaces > 0:
                changes.append(NormalizationChange(
                    change_type="formatting",
                    original_text=f"{original_spaces} excessive spaces",
                    normalized_text="normalized spacing",
                    position=0,
                    confidence=1.0,
                    reason="space_normalization"
                ))
        
        # Normalize quotations
        if style_config.get('normalize_quotations', False):
            # Replace various quote marks with standard ones
            quote_replacements = [
                (r'[""]', '"'),  # Smart quotes to straight quotes
                (r"['']", "'"),  # Smart apostrophes to straight apostrophes
            ]
            
            for pattern, replacement in quote_replacements:
                original_count = len(re.findall(pattern, working_text))
                working_text = re.sub(pattern, replacement, working_text)
                
                if original_count > 0:
                    changes.append(NormalizationChange(
                        change_type="formatting",
                        original_text=f"{original_count} smart quotes",
                        normalized_text="standard quotes",
                        position=0,
                        confidence=1.0,
                        reason="quote_normalization"
                    ))
        
        return working_text, changes, violations
    
    def _identify_sentence_boundaries(self, text: str, tokens: List[WordToken]) -> List[int]:
        """Identify sentence boundaries using timing and content cues"""
        boundaries = []
        
        # Look for timing gaps that might indicate sentence boundaries
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i + 1]
            
            # Calculate gap between tokens
            gap = next_token.start_time - current_token.end_time
            
            # If gap is significant (>0.5 seconds), it might be a sentence boundary
            if gap > 0.5:
                # Find position in text
                position = self._find_token_position(text, current_token, i)
                if position >= 0:
                    boundaries.append(position + len(current_token.text))
        
        # Also look for content-based boundaries
        content_boundaries = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        pos = 0
        for sentence in sentences[:-1]:  # Exclude last sentence
            pos += len(sentence)
            content_boundaries.append(pos)
            pos += 1  # Account for space
        
        # Combine and deduplicate
        all_boundaries = sorted(set(boundaries + content_boundaries))
        return all_boundaries
    
    def _is_timestamp_anchored(self, position: int, tokens: List[WordToken]) -> bool:
        """Check if a text position is properly anchored to word timing"""
        # Find the token at or near this position
        current_pos = 0
        for token in tokens:
            token_start = current_pos
            token_end = current_pos + len(token.text)
            
            if token_start <= position <= token_end + 1:  # Allow small tolerance
                return True
            
            current_pos = token_end + 1  # +1 for space
        
        return False
    
    def _find_token_position(self, text: str, token: WordToken, token_index: int) -> int:
        """Find character position of a token in text"""
        words = text.split()
        if token_index < len(words):
            # Calculate character position
            pos = sum(len(words[i]) + 1 for i in range(token_index))  # +1 for spaces
            return pos - 1 if pos > 0 else 0
        return -1
    
    def _validate_final_result(self, original_text: str, normalized_text: str, 
                              original_tokens: List[WordToken], 
                              profile_config: Dict[str, Any]) -> List[str]:
        """Validate the final normalization result against guardrails"""
        violations = []
        
        # Token preservation check
        original_words = set(re.findall(r'\w+', original_text.lower()))
        normalized_words = set(re.findall(r'\w+', normalized_text.lower()))
        
        # Check for new tokens (potential hallucinations)
        new_tokens = normalized_words - original_words
        if new_tokens:
            violations.append(f"token_invention: {list(new_tokens)}")
        
        # Check change ratio
        guardrail_config = profile_config.get('guardrails', {})
        max_change_ratio = guardrail_config.get('max_segment_change_ratio', 0.15)
        
        original_length = len(original_text.split())
        normalized_length = len(normalized_text.split())
        
        if original_length > 0:
            change_ratio = abs(normalized_length - original_length) / original_length
            if change_ratio > max_change_ratio:
                violations.append(f"excessive_change_ratio: {change_ratio:.3f} > {max_change_ratio}")
        
        # Check for protected token violations
        for token in original_tokens:
            if token.is_protected:
                if token.text.lower() not in normalized_text.lower():
                    violations.append(f"protected_token_removed: {token.text} ({token.protection_reason})")
        
        return violations
    
    def _get_downgrade_profile(self, current_profile: str) -> str:
        """Get a less aggressive profile for downgrading"""
        downgrade_chain = ["executive", "readable", "light", "verbatim"]
        
        try:
            current_index = downgrade_chain.index(current_profile)
            if current_index < len(downgrade_chain) - 1:
                return downgrade_chain[current_index + 1]
        except ValueError:
            pass
        
        return "verbatim"  # Safest fallback
    
    def _retokenize_normalized_text(self, normalized_text: str, 
                                   original_tokens: List[WordToken]) -> List[WordToken]:
        """Create new token list for normalized text, preserving timing"""
        normalized_words = normalized_text.split()
        
        # Try to map normalized words to original timing
        new_tokens = []
        original_idx = 0
        
        for norm_word in normalized_words:
            if original_idx < len(original_tokens):
                # Use timing from corresponding original token
                orig_token = original_tokens[original_idx]
                new_token = WordToken(
                    text=norm_word,
                    start_time=orig_token.start_time,
                    end_time=orig_token.end_time,
                    confidence=orig_token.confidence,
                    is_protected=orig_token.is_protected,
                    protection_reason=orig_token.protection_reason,
                    original_index=orig_token.original_index
                )
                new_tokens.append(new_token)
                original_idx += 1
            else:
                # Estimate timing for additional words
                if new_tokens:
                    last_token = new_tokens[-1]
                    new_token = WordToken(
                        text=norm_word,
                        start_time=last_token.end_time,
                        end_time=last_token.end_time + 0.5,  # Default duration
                        confidence=0.8,
                        original_index=len(new_tokens)
                    )
                    new_tokens.append(new_token)
        
        return new_tokens
    
    def _create_empty_result(self, text: str, profile: str) -> SegmentNormalizationResult:
        """Create empty result for edge cases"""
        return SegmentNormalizationResult(
            original_text=text,
            normalized_text=text,
            original_tokens=[],
            normalized_tokens=[],
            changes=[],
            readability_score_before=0.5,
            readability_score_after=0.5,
            guardrail_violations=[],
            processing_time_ms=0.0,
            profile_used=profile
        )
    
    def _update_metrics(self, result: SegmentNormalizationResult):
        """Update global metrics with segment result"""
        self.metrics.tokens_changed_count += len([c for c in result.changes if c.change_type in ['capitalization', 'formatting']])
        self.metrics.tokens_removed_count += len([c for c in result.changes if c.original_text and not c.normalized_text])
        self.metrics.tokens_added_count += len([c for c in result.changes if not c.original_text and c.normalized_text])
        self.metrics.sentences_adjusted_count += len([c for c in result.changes if c.change_type == 'punctuation'])
        self.metrics.guardrail_violations_count += len(result.guardrail_violations)
        self.metrics.profile_downgrades_count += len(result.profile_downgrades)
        
        # Update readability improvement
        improvement = result.readability_score_after - result.readability_score_before
        if hasattr(self.metrics, '_readability_improvements'):
            self.metrics._readability_improvements.append(improvement)
        else:
            self.metrics._readability_improvements = [improvement]
        
        if self.metrics._readability_improvements:
            self.metrics.avg_readability_improvement = sum(self.metrics._readability_improvements) / len(self.metrics._readability_improvements)
    
    def get_metrics(self) -> NormalizationMetrics:
        """Get current normalization metrics"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset metrics counters"""
        self.metrics = NormalizationMetrics()
        self.logger.info("Normalization metrics reset")

def create_text_normalizer(config_path: str = "config/normalization_profiles.yaml") -> TextNormalizer:
    """Factory function to create a text normalizer"""
    return TextNormalizer(config_path)