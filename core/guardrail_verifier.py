"""
Guardrail Verification System

Provides comprehensive validation to prevent token invention, timing violations,
and semantic drift during text normalization. Implements automatic downgrading
and detailed change tracking with comprehensive diff analysis.
"""

import re
import time
import difflib
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math

from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class TokenAnalysis:
    """Analysis of token changes between original and normalized text"""
    original_tokens: Set[str]
    normalized_tokens: Set[str]
    added_tokens: Set[str]
    removed_tokens: Set[str]
    preserved_tokens: Set[str]
    protected_tokens: Set[str]
    token_change_ratio: float

@dataclass
class TimingViolation:
    """Represents a timing preservation violation"""
    violation_type: str  # "word_drift", "segment_drift", "token_order"
    original_position: int
    normalized_position: int
    drift_amount: float  # In seconds
    threshold_exceeded: float
    severity: str  # "minor", "moderate", "severe"

@dataclass
class GuardrailViolation:
    """Represents a guardrail rule violation"""
    rule_name: str
    violation_type: str  # "token_invention", "timing_drift", "protected_change", "semantic_drift"
    severity: str  # "minor", "moderate", "severe", "critical"
    details: str
    position: Optional[int] = None
    affected_tokens: List[str] = field(default_factory=list)
    recommended_action: str = ""
    automatic_fix_available: bool = False

@dataclass
class ChangeAudit:
    """Comprehensive audit of normalization changes"""
    original_text: str
    normalized_text: str
    character_changes: int
    word_changes: int
    sentence_changes: int
    punctuation_changes: int
    capitalization_changes: int
    disfluency_changes: int
    formatting_changes: int
    protected_items_count: int
    protected_items_preserved: int
    change_categories: Dict[str, int] = field(default_factory=dict)

@dataclass
class GuardrailResult:
    """Result of guardrail verification"""
    passed: bool
    violations: List[GuardrailViolation]
    token_analysis: TokenAnalysis
    timing_violations: List[TimingViolation]
    change_audit: ChangeAudit
    recommended_profile: Optional[str] = None
    confidence_score: float = 1.0  # 0.0 to 1.0
    processing_notes: List[str] = field(default_factory=list)

class TokenInventionDetector:
    """Detects potential token invention/hallucination"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("token_invention_detector")
        
        # Allowed new tokens (punctuation and formatting)
        self.allowed_new_tokens = {
            '.', ',', '!', '?', ':', ';', '"', "'", '-', '(', ')', '[', ']',
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
        }
        
        # Common contractions and their expansions
        self.allowed_expansions = {
            "can't": ["cannot", "can", "not"],
            "won't": ["will", "not"],
            "isn't": ["is", "not"],
            "aren't": ["are", "not"],
            "wasn't": ["was", "not"],
            "weren't": ["were", "not"],
            "don't": ["do", "not"],
            "doesn't": ["does", "not"],
            "didn't": ["did", "not"],
            "haven't": ["have", "not"],
            "hasn't": ["has", "not"],
            "hadn't": ["had", "not"],
            "shouldn't": ["should", "not"],
            "wouldn't": ["would", "not"],
            "couldn't": ["could", "not"],
            "mustn't": ["must", "not"],
        }
        
        # Common abbreviation expansions
        self.allowed_abbreviation_expansions = {
            "it's": ["it", "is"],
            "that's": ["that", "is"],
            "he's": ["he", "is"],
            "she's": ["she", "is"],
            "we're": ["we", "are"],
            "they're": ["they", "are"],
            "I'm": ["I", "am"],
            "you're": ["you", "are"],
        }
    
    def analyze_token_changes(self, original_text: str, normalized_text: str, 
                             protected_tokens: List[str] = None) -> TokenAnalysis:
        """
        Analyze token changes between original and normalized text
        
        Args:
            original_text: Original text
            normalized_text: Normalized text  
            protected_tokens: List of tokens that should be preserved
            
        Returns:
            TokenAnalysis object with detailed analysis
        """
        # Extract tokens (alphanumeric sequences)
        original_tokens = set(re.findall(r'\w+', original_text.lower()))
        normalized_tokens = set(re.findall(r'\w+', normalized_text.lower()))
        
        # Calculate changes
        added_tokens = normalized_tokens - original_tokens
        removed_tokens = original_tokens - normalized_tokens
        preserved_tokens = original_tokens & normalized_tokens
        
        # Filter allowed additions
        legitimate_additions = set()
        problematic_additions = set()
        
        for token in added_tokens:
            if self._is_legitimate_addition(token, original_text, normalized_text):
                legitimate_additions.add(token)
            else:
                problematic_additions.add(token)
        
        # Calculate change ratio
        if original_tokens:
            change_ratio = len(problematic_additions) / len(original_tokens)
        else:
            change_ratio = 1.0 if problematic_additions else 0.0
        
        # Identify protected tokens
        protected_set = set((protected_tokens or []))
        
        return TokenAnalysis(
            original_tokens=original_tokens,
            normalized_tokens=normalized_tokens,
            added_tokens=problematic_additions,
            removed_tokens=removed_tokens,
            preserved_tokens=preserved_tokens,
            protected_tokens=protected_set,
            token_change_ratio=change_ratio
        )
    
    def _is_legitimate_addition(self, token: str, original_text: str, normalized_text: str) -> bool:
        """Check if a new token is a legitimate addition"""
        
        # Check if it's in allowed new tokens
        if token.lower() in self.allowed_new_tokens:
            return True
        
        # Check if it's from contraction expansion
        for contraction, expansion in self.allowed_expansions.items():
            if contraction.lower() in original_text.lower() and token.lower() in [e.lower() for e in expansion]:
                return True
        
        # Check if it's from abbreviation expansion
        for abbrev, expansion in self.allowed_abbreviation_expansions.items():
            if abbrev.lower() in original_text.lower() and token.lower() in [e.lower() for e in expansion]:
                return True
        
        # Check if it's a number spelling (1 -> one)
        number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        if token.lower() in number_words:
            # Check if corresponding digit exists in original
            digit_map = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
            if digit_map.get(token.lower()) in original_text:
                return True
        
        # Check if it's part of a word split (e.g., "cannot" -> "can not")
        if len(token) <= 3:  # Short words might be from splits
            combined_variants = []
            words = normalized_text.split()
            for i, word in enumerate(words):
                if word.lower() == token.lower():
                    # Check surrounding words
                    if i > 0:
                        combined_variants.append(words[i-1] + word)
                    if i < len(words) - 1:
                        combined_variants.append(word + words[i+1])
            
            for variant in combined_variants:
                if variant.lower() in original_text.lower():
                    return True
        
        return False

class TimingValidator:
    """Validates timing preservation during normalization"""
    
    def __init__(self, max_word_drift: float = 0.1, max_segment_drift: float = 0.2):
        self.max_word_drift = max_word_drift  # seconds
        self.max_segment_drift = max_segment_drift  # seconds
        self.logger = create_enhanced_logger("timing_validator")
    
    def validate_timing_preservation(self, original_tokens: List[Dict[str, Any]], 
                                   normalized_tokens: List[Dict[str, Any]]) -> List[TimingViolation]:
        """
        Validate that timing is preserved during normalization
        
        Args:
            original_tokens: Original tokens with timing information
            normalized_tokens: Normalized tokens with timing information
            
        Returns:
            List of timing violations found
        """
        violations = []
        
        # Align tokens for comparison
        aligned_pairs = self._align_tokens(original_tokens, normalized_tokens)
        
        for orig_token, norm_token in aligned_pairs:
            if orig_token and norm_token:
                # Check word-level drift
                start_drift = abs(orig_token.get('start', 0) - norm_token.get('start', 0))
                end_drift = abs(orig_token.get('end', 0) - norm_token.get('end', 0))
                
                if start_drift > self.max_word_drift:
                    violations.append(TimingViolation(
                        violation_type="word_drift",
                        original_position=orig_token.get('start', 0),
                        normalized_position=norm_token.get('start', 0),
                        drift_amount=start_drift,
                        threshold_exceeded=start_drift - self.max_word_drift,
                        severity=self._calculate_drift_severity(start_drift, self.max_word_drift)
                    ))
                
                if end_drift > self.max_word_drift:
                    violations.append(TimingViolation(
                        violation_type="word_drift",
                        original_position=orig_token.get('end', 0),
                        normalized_position=norm_token.get('end', 0),
                        drift_amount=end_drift,
                        threshold_exceeded=end_drift - self.max_word_drift,
                        severity=self._calculate_drift_severity(end_drift, self.max_word_drift)
                    ))
        
        return violations
    
    def _align_tokens(self, original_tokens: List[Dict[str, Any]], 
                     normalized_tokens: List[Dict[str, Any]]) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
        """Align original and normalized tokens for comparison"""
        aligned = []
        
        # Simple alignment based on position
        max_len = max(len(original_tokens), len(normalized_tokens))
        
        for i in range(max_len):
            orig = original_tokens[i] if i < len(original_tokens) else None
            norm = normalized_tokens[i] if i < len(normalized_tokens) else None
            aligned.append((orig, norm))
        
        return aligned
    
    def _calculate_drift_severity(self, drift: float, threshold: float) -> str:
        """Calculate severity of timing drift"""
        ratio = drift / threshold
        
        if ratio <= 1.5:
            return "minor"
        elif ratio <= 3.0:
            return "moderate"
        else:
            return "severe"

class SemanticDriftDetector:
    """Detects potential semantic drift during normalization"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("semantic_drift_detector")
        
        # Critical semantic markers that should not be removed
        self.critical_negations = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere'}
        self.critical_quantifiers = {'all', 'every', 'each', 'some', 'any', 'many', 'few', 'several'}
        self.critical_connectors = {'but', 'however', 'although', 'unless', 'because', 'since'}
        
    def detect_semantic_drift(self, original_text: str, normalized_text: str) -> List[GuardrailViolation]:
        """Detect potential semantic changes"""
        violations = []
        
        # Check for removed critical words
        original_words = set(re.findall(r'\w+', original_text.lower()))
        normalized_words = set(re.findall(r'\w+', normalized_text.lower()))
        
        removed_words = original_words - normalized_words
        
        # Check critical negations
        removed_negations = removed_words & self.critical_negations
        if removed_negations:
            violations.append(GuardrailViolation(
                rule_name="semantic_preservation",
                violation_type="semantic_drift",
                severity="critical",
                details=f"Critical negation words removed: {list(removed_negations)}",
                affected_tokens=list(removed_negations),
                recommended_action="restore_removed_negations",
                automatic_fix_available=True
            ))
        
        # Check critical quantifiers
        removed_quantifiers = removed_words & self.critical_quantifiers
        if removed_quantifiers:
            violations.append(GuardrailViolation(
                rule_name="semantic_preservation",
                violation_type="semantic_drift",
                severity="moderate",
                details=f"Critical quantifier words removed: {list(removed_quantifiers)}",
                affected_tokens=list(removed_quantifiers),
                recommended_action="restore_removed_quantifiers",
                automatic_fix_available=True
            ))
        
        # Check critical connectors
        removed_connectors = removed_words & self.critical_connectors
        if removed_connectors:
            violations.append(GuardrailViolation(
                rule_name="semantic_preservation",
                violation_type="semantic_drift",
                severity="moderate",
                details=f"Critical connector words removed: {list(removed_connectors)}",
                affected_tokens=list(removed_connectors),
                recommended_action="restore_removed_connectors",
                automatic_fix_available=True
            ))
        
        return violations

class ChangeAuditor:
    """Audits and categorizes normalization changes"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("change_auditor")
    
    def audit_changes(self, original_text: str, normalized_text: str,
                     normalization_changes: List[Any] = None,
                     protected_tokens: List[str] = None) -> ChangeAudit:
        """
        Perform comprehensive audit of normalization changes
        
        Args:
            original_text: Original text
            normalized_text: Normalized text
            normalization_changes: List of normalization changes applied
            protected_tokens: List of tokens that should be preserved
            
        Returns:
            ChangeAudit object with detailed analysis
        """
        # Calculate basic change metrics
        char_changes = self._calculate_character_changes(original_text, normalized_text)
        word_changes = self._calculate_word_changes(original_text, normalized_text)
        
        # Categorize changes
        change_categories = defaultdict(int)
        
        if normalization_changes:
            for change in normalization_changes:
                change_type = getattr(change, 'change_type', 'unknown')
                change_categories[change_type] += 1
        
        # Count specific change types
        punct_changes = self._count_punctuation_changes(original_text, normalized_text)
        cap_changes = self._count_capitalization_changes(original_text, normalized_text)
        sentence_changes = self._count_sentence_changes(original_text, normalized_text)
        
        # Analyze protected items
        protected_count = len(protected_tokens) if protected_tokens else 0
        protected_preserved = self._count_preserved_protected_items(
            original_text, normalized_text, protected_tokens or []
        )
        
        return ChangeAudit(
            original_text=original_text,
            normalized_text=normalized_text,
            character_changes=char_changes,
            word_changes=word_changes,
            sentence_changes=sentence_changes,
            punctuation_changes=punct_changes,
            capitalization_changes=cap_changes,
            disfluency_changes=change_categories.get('disfluency', 0),
            formatting_changes=change_categories.get('formatting', 0),
            protected_items_count=protected_count,
            protected_items_preserved=protected_preserved,
            change_categories=dict(change_categories)
        )
    
    def _calculate_character_changes(self, original: str, normalized: str) -> int:
        """Calculate number of character-level changes"""
        differ = difflib.SequenceMatcher(None, original, normalized)
        changes = 0
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag != 'equal':
                changes += max(i2 - i1, j2 - j1)
        
        return changes
    
    def _calculate_word_changes(self, original: str, normalized: str) -> int:
        """Calculate number of word-level changes"""
        orig_words = original.split()
        norm_words = normalized.split()
        
        differ = difflib.SequenceMatcher(None, orig_words, norm_words)
        changes = 0
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag != 'equal':
                changes += max(i2 - i1, j2 - j1)
        
        return changes
    
    def _count_punctuation_changes(self, original: str, normalized: str) -> int:
        """Count punctuation changes"""
        orig_punct = re.findall(r'[.!?,:;()"\'-]', original)
        norm_punct = re.findall(r'[.!?,:;()"\'-]', normalized)
        
        return abs(len(norm_punct) - len(orig_punct))
    
    def _count_capitalization_changes(self, original: str, normalized: str) -> int:
        """Count capitalization changes"""
        orig_caps = [c for c in original if c.isupper()]
        norm_caps = [c for c in normalized if c.isupper()]
        
        return abs(len(norm_caps) - len(orig_caps))
    
    def _count_sentence_changes(self, original: str, normalized: str) -> int:
        """Count sentence boundary changes"""
        orig_sentences = len(re.split(r'[.!?]+', original))
        norm_sentences = len(re.split(r'[.!?]+', normalized))
        
        return abs(norm_sentences - orig_sentences)
    
    def _count_preserved_protected_items(self, original: str, normalized: str, 
                                        protected: List[str]) -> int:
        """Count how many protected items were preserved"""
        preserved = 0
        
        for item in protected:
            if item.lower() in original.lower() and item.lower() in normalized.lower():
                preserved += 1
        
        return preserved

class GuardrailVerifier:
    """Main guardrail verification system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize guardrail verifier"""
        self.config = config or {}
        self.logger = create_enhanced_logger("guardrail_verifier")
        
        # Initialize components
        self.token_detector = TokenInventionDetector()
        self.timing_validator = TimingValidator(
            max_word_drift=self.config.get('max_word_drift_seconds', 0.1),
            max_segment_drift=self.config.get('max_segment_drift_seconds', 0.2)
        )
        self.semantic_detector = SemanticDriftDetector()
        self.change_auditor = ChangeAuditor()
        
        # Violation thresholds
        self.token_invention_threshold = self.config.get('token_invention_threshold', 0.0)  # Zero tolerance
        self.max_change_ratio = self.config.get('max_change_ratio', 0.15)
        self.protected_error_threshold = self.config.get('protected_error_threshold', 0.002)  # 0.2%
        
        self.logger.info("Guardrail verifier initialized", context={
            'token_invention_threshold': self.token_invention_threshold,
            'max_change_ratio': self.max_change_ratio,
            'protected_error_threshold': self.protected_error_threshold
        })
    
    def verify_normalization(self, original_text: str, normalized_text: str,
                           original_tokens: List[Dict[str, Any]] = None,
                           normalized_tokens: List[Dict[str, Any]] = None,
                           normalization_changes: List[Any] = None,
                           protected_tokens: List[str] = None,
                           current_profile: str = "readable") -> GuardrailResult:
        """
        Perform comprehensive guardrail verification
        
        Args:
            original_text: Original text
            normalized_text: Normalized text
            original_tokens: Original tokens with timing
            normalized_tokens: Normalized tokens with timing
            normalization_changes: Applied normalization changes
            protected_tokens: Tokens that should be preserved
            current_profile: Current normalization profile
            
        Returns:
            GuardrailResult with verification results
        """
        start_time = time.time()
        violations = []
        processing_notes = []
        
        # 1. Token invention detection
        token_analysis = self.token_detector.analyze_token_changes(
            original_text, normalized_text, protected_tokens
        )
        
        if token_analysis.added_tokens:
            violations.append(GuardrailViolation(
                rule_name="token_preservation",
                violation_type="token_invention",
                severity="critical",
                details=f"New tokens detected: {list(token_analysis.added_tokens)}",
                affected_tokens=list(token_analysis.added_tokens),
                recommended_action="downgrade_profile",
                automatic_fix_available=True
            ))
        
        # 2. Token change ratio check
        if token_analysis.token_change_ratio > self.max_change_ratio:
            violations.append(GuardrailViolation(
                rule_name="change_ratio_limit",
                violation_type="excessive_changes",
                severity="moderate",
                details=f"Change ratio {token_analysis.token_change_ratio:.3f} exceeds limit {self.max_change_ratio}",
                recommended_action="downgrade_profile",
                automatic_fix_available=True
            ))
        
        # 3. Timing validation (if timing data available)
        timing_violations = []
        if original_tokens and normalized_tokens:
            timing_violations = self.timing_validator.validate_timing_preservation(
                original_tokens, normalized_tokens
            )
            
            for timing_violation in timing_violations:
                if timing_violation.severity in ['moderate', 'severe']:
                    violations.append(GuardrailViolation(
                        rule_name="timing_preservation",
                        violation_type="timing_drift",
                        severity=timing_violation.severity,
                        details=f"Timing drift of {timing_violation.drift_amount:.3f}s exceeds threshold",
                        position=timing_violation.original_position,
                        recommended_action="restore_original_timing",
                        automatic_fix_available=False
                    ))
        
        # 4. Semantic drift detection
        semantic_violations = self.semantic_detector.detect_semantic_drift(
            original_text, normalized_text
        )
        violations.extend(semantic_violations)
        
        # 5. Protected token preservation
        if protected_tokens:
            protected_violations = self._check_protected_token_preservation(
                original_text, normalized_text, protected_tokens
            )
            violations.extend(protected_violations)
        
        # 6. Change audit
        change_audit = self.change_auditor.audit_changes(
            original_text, normalized_text, normalization_changes, protected_tokens
        )
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(violations, token_analysis, timing_violations)
        
        # Determine if verification passed
        passed = len([v for v in violations if v.severity in ['critical', 'severe']]) == 0
        
        # Recommend profile downgrade if needed
        recommended_profile = None
        if not passed:
            recommended_profile = self._recommend_profile_downgrade(current_profile, violations)
            processing_notes.append(f"Recommending profile downgrade from '{current_profile}' to '{recommended_profile}'")
        
        processing_time = (time.time() - start_time) * 1000
        processing_notes.append(f"Guardrail verification completed in {processing_time:.2f}ms")
        
        self.logger.info(f"Guardrail verification completed", context={
            'passed': passed,
            'violations_count': len(violations),
            'critical_violations': len([v for v in violations if v.severity == 'critical']),
            'recommended_profile': recommended_profile,
            'confidence_score': confidence_score,
            'processing_time_ms': processing_time
        })
        
        return GuardrailResult(
            passed=passed,
            violations=violations,
            token_analysis=token_analysis,
            timing_violations=timing_violations,
            change_audit=change_audit,
            recommended_profile=recommended_profile,
            confidence_score=confidence_score,
            processing_notes=processing_notes
        )
    
    def _check_protected_token_preservation(self, original_text: str, normalized_text: str,
                                          protected_tokens: List[str]) -> List[GuardrailViolation]:
        """Check that protected tokens are preserved"""
        violations = []
        
        original_lower = original_text.lower()
        normalized_lower = normalized_text.lower()
        
        for token in protected_tokens:
            token_lower = token.lower()
            
            # Check if token was in original but missing from normalized
            if token_lower in original_lower and token_lower not in normalized_lower:
                violations.append(GuardrailViolation(
                    rule_name="protected_token_preservation",
                    violation_type="protected_change",
                    severity="critical",
                    details=f"Protected token '{token}' was removed during normalization",
                    affected_tokens=[token],
                    recommended_action="restore_protected_token",
                    automatic_fix_available=True
                ))
        
        return violations
    
    def _calculate_confidence_score(self, violations: List[GuardrailViolation],
                                   token_analysis: TokenAnalysis,
                                   timing_violations: List[TimingViolation]) -> float:
        """Calculate overall confidence score for normalization quality"""
        base_score = 1.0
        
        # Penalize based on violation severity
        severity_penalties = {'minor': 0.05, 'moderate': 0.15, 'severe': 0.3, 'critical': 0.5}
        
        for violation in violations:
            penalty = severity_penalties.get(violation.severity, 0.1)
            base_score -= penalty
        
        # Penalize timing violations
        for timing_violation in timing_violations:
            penalty = severity_penalties.get(timing_violation.severity, 0.1)
            base_score -= penalty * 0.5  # Timing violations are less critical
        
        # Penalize token changes
        if token_analysis.token_change_ratio > 0:
            base_score -= token_analysis.token_change_ratio * 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def _recommend_profile_downgrade(self, current_profile: str, 
                                   violations: List[GuardrailViolation]) -> str:
        """Recommend a less aggressive profile based on violations"""
        profile_hierarchy = ["executive", "readable", "light", "verbatim"]
        
        # Count critical and severe violations
        critical_count = len([v for v in violations if v.severity == 'critical'])
        severe_count = len([v for v in violations if v.severity == 'severe'])
        
        try:
            current_index = profile_hierarchy.index(current_profile)
        except ValueError:
            current_index = 1  # Default to 'readable' position
        
        # Determine downgrade level
        if critical_count > 0:
            # Critical violations require maximum downgrade
            return "verbatim"
        elif severe_count > 2:
            # Multiple severe violations require significant downgrade
            target_index = min(len(profile_hierarchy) - 1, current_index + 2)
            return profile_hierarchy[target_index]
        elif severe_count > 0:
            # Single severe violation requires moderate downgrade
            target_index = min(len(profile_hierarchy) - 1, current_index + 1)
            return profile_hierarchy[target_index]
        else:
            # Minor violations might not require downgrade
            return current_profile
    
    def generate_diff_summary(self, change_audit: ChangeAudit, 
                             token_analysis: TokenAnalysis) -> Dict[str, Any]:
        """Generate comprehensive diff summary"""
        summary = {
            'sentence_starts_adjusted': change_audit.sentence_changes,
            'fillers_removed': change_audit.disfluency_changes,
            'acronyms_protected': token_analysis.protected_tokens.__len__(),
            'total_character_changes': change_audit.character_changes,
            'total_word_changes': change_audit.word_changes,
            'punctuation_changes': change_audit.punctuation_changes,
            'capitalization_changes': change_audit.capitalization_changes,
            'formatting_changes': change_audit.formatting_changes,
            'protected_items_preserved_ratio': (
                change_audit.protected_items_preserved / change_audit.protected_items_count
                if change_audit.protected_items_count > 0 else 1.0
            ),
            'token_change_ratio': token_analysis.token_change_ratio,
            'tokens_added': len(token_analysis.added_tokens),
            'tokens_removed': len(token_analysis.removed_tokens),
            'change_categories': change_audit.change_categories
        }
        
        return summary

def create_guardrail_verifier(config: Dict[str, Any] = None) -> GuardrailVerifier:
    """Factory function to create a guardrail verifier"""
    return GuardrailVerifier(config)