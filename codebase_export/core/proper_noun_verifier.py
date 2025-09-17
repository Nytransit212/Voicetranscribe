"""
Constrained Proper Noun and Number Verifier

Enforces strict constraints on entity variants to prevent hallucination:
- Only allows variants observed from engines or promoted by auto-glossary
- Preserves exact formatting for numbers and alphanumerics
- Applies sophisticated scoring with engine votes, confidence, and glossary bonuses
- Blocks unseen variants with zero tolerance for hallucination

Integrates with RepairEngine and auto-glossary system for comprehensive
entity accuracy control across the transcription pipeline.
"""

import re
import json
import time
import hashlib
from typing import Dict, Any, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from pathlib import Path

from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class SpanCandidate:
    """Individual span candidate from an engine"""
    text: str
    engine_name: str
    confidence: float
    acoustic_score: float
    start_time: float
    end_time: float
    chunk_id: Optional[str] = None
    stem_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GlossaryEntry:
    """Auto-glossary entry for verification"""
    canonical_form: str
    variants: Set[str]
    weight: float
    confidence_score: float
    session_count: int
    term_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerificationResult:
    """Result of proper noun/number verification"""
    verified_text: str
    is_verified: bool
    verification_source: str  # 'observed', 'glossary', 'current_kept'
    score: float
    margin: float
    blocked_variants: List[str]
    processing_time: float
    verification_metadata: Dict[str, Any]

@dataclass
class VerificationTelemetry:
    """Telemetry data for verification operation"""
    invocation_count: int = 0
    block_count: int = 0
    change_count: int = 0
    blocked_unseen_count: int = 0
    blocked_insufficient_margin_count: int = 0
    top_blocked_variants: Counter = field(default_factory=Counter)
    processing_times: List[float] = field(default_factory=list)
    daily_reports: List[Dict[str, Any]] = field(default_factory=list)

class ProperNounVerifier:
    """Constrained verifier for proper nouns and numbers with zero hallucination tolerance"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the proper noun verifier
        
        Args:
            config: Optional configuration override
        """
        # Load configuration with defaults
        default_config = {
            'enabled': True,
            'min_margin': 0.1,
            'max_glossary_entries': 50,
            'glossary_weight_bonus': 0.2,
            'block_unseen_variants': True,
            'enable_number_formatting': True,
            'enable_telemetry': True
        }
        self.config = config or default_config
        
        # Configuration parameters
        self.enabled = self.config.get('enabled', True)
        self.min_margin = self.config.get('min_margin', 0.1)
        self.max_glossary_entries = self.config.get('max_glossary_entries', 50)
        self.glossary_weight_bonus = self.config.get('glossary_weight_bonus', 0.2)
        self.block_unseen_variants = self.config.get('block_unseen_variants', True)
        self.enable_number_formatting = self.config.get('enable_number_formatting', True)
        self.enable_telemetry = self.config.get('enable_telemetry', True)
        
        # Initialize enhanced logging
        self.logger = create_enhanced_logger("proper_noun_verifier")
        
        # Telemetry tracking
        self.telemetry = VerificationTelemetry()
        
        # Precompiled regex patterns for number/alphanumeric detection
        self.number_pattern = re.compile(r'\b\d+([\/\-\.]\d+)*\b')
        self.alphanumeric_pattern = re.compile(r'\b[A-Z0-9]+[\-\/\._]*[A-Z0-9]*\b')
        self.proper_noun_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        
        self.logger.info("Proper noun verifier initialized", 
                        context={
                            'enabled': self.enabled,
                            'min_margin': self.min_margin,
                            'max_glossary_entries': self.max_glossary_entries,
                            'glossary_bonus': self.glossary_weight_bonus,
                            'block_unseen': self.block_unseen_variants
                        })
    
    def verify(self, 
               span_candidates: List[SpanCandidate],
               current_text: str,
               glossary_entries: List[GlossaryEntry],
               time_window: Tuple[float, float],
               rules: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """
        Verify and select proper noun/number variant with strict constraints
        
        Args:
            span_candidates: List of candidates from different engines
            current_text: Current transcript text for this span
            glossary_entries: Relevant auto-glossary entries
            time_window: Time window (start, end) for this span
            rules: Optional additional verification rules
            
        Returns:
            VerificationResult with final choice and metadata
        """
        start_time = time.time()
        
        if not self.enabled:
            return VerificationResult(
                verified_text=current_text,
                is_verified=False,
                verification_source='disabled',
                score=0.0,
                margin=0.0,
                blocked_variants=[],
                processing_time=0.0,
                verification_metadata={'reason': 'verifier_disabled'}
            )
        
        self.telemetry.invocation_count += 1
        
        try:
            # Create allowed set: observed variants ∪ glossary variants
            allowed_set = self._build_allowed_set(span_candidates, glossary_entries)
            
            # Filter candidates to only allowed variants
            valid_candidates = self._filter_to_allowed_variants(span_candidates, allowed_set)
            
            # Calculate scores for all valid candidates
            scored_candidates = self._score_candidates(
                valid_candidates, glossary_entries, current_text, rules
            )
            
            # Select best candidate based on scoring and margin
            verification_result = self._select_best_candidate(
                scored_candidates, current_text, time_window
            )
            
            # Apply strict formatting rules for numbers/alphanumerics
            if self.enable_number_formatting:
                verification_result = self._apply_formatting_rules(
                    verification_result, span_candidates
                )
            
            # Update telemetry
            self._update_telemetry(verification_result, span_candidates, allowed_set)
            
            processing_time = time.time() - start_time
            verification_result.processing_time = processing_time
            self.telemetry.processing_times.append(processing_time)
            
            self.logger.debug("Verification completed", 
                            context={
                                'original_text': current_text,
                                'verified_text': verification_result.verified_text,
                                'is_verified': verification_result.is_verified,
                                'source': verification_result.verification_source,
                                'score': verification_result.score,
                                'margin': verification_result.margin,
                                'processing_time': processing_time,
                                'candidates_count': len(span_candidates),
                                'allowed_set_size': len(allowed_set),
                                'blocked_count': len(verification_result.blocked_variants)
                            })
            
            return verification_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error("Verification failed", 
                            context={
                                'error': str(e),
                                'current_text': current_text,
                                'candidates_count': len(span_candidates),
                                'time_window': time_window,
                                'processing_time': processing_time
                            })
            
            # Return current text unchanged on error
            return VerificationResult(
                verified_text=current_text,
                is_verified=False,
                verification_source='error_fallback',
                score=0.0,
                margin=0.0,
                blocked_variants=[],
                processing_time=processing_time,
                verification_metadata={'error': str(e)}
            )
    
    def _build_allowed_set(self, 
                          span_candidates: List[SpanCandidate],
                          glossary_entries: List[GlossaryEntry]) -> Set[str]:
        """
        Build allowed set from observed variants and glossary entries
        
        Returns:
            Set of allowed text variants
        """
        allowed_set = set()
        
        # Add all observed variants from engines
        for candidate in span_candidates:
            allowed_set.add(candidate.text.strip())
        
        # Add top K glossary entries
        sorted_glossary = sorted(glossary_entries, 
                               key=lambda e: e.weight, 
                               reverse=True)[:self.max_glossary_entries]
        
        for entry in sorted_glossary:
            allowed_set.add(entry.canonical_form)
            allowed_set.update(entry.variants)
        
        # Remove empty strings
        allowed_set.discard('')
        
        self.logger.debug("Built allowed set", 
                        context={
                            'observed_variants': len([c.text for c in span_candidates]),
                            'glossary_entries': len(sorted_glossary),
                            'total_allowed': len(allowed_set),
                            'allowed_preview': list(allowed_set)[:10]
                        })
        
        return allowed_set
    
    def _filter_to_allowed_variants(self, 
                                   span_candidates: List[SpanCandidate],
                                   allowed_set: Set[str]) -> List[SpanCandidate]:
        """
        Filter candidates to only those in the allowed set
        
        Returns:
            List of valid candidates
        """
        if not self.block_unseen_variants:
            return span_candidates
        
        valid_candidates = []
        blocked_variants = []
        
        for candidate in span_candidates:
            if candidate.text.strip() in allowed_set:
                valid_candidates.append(candidate)
            else:
                blocked_variants.append(candidate.text)
                self.telemetry.blocked_unseen_count += 1
                self.telemetry.top_blocked_variants[candidate.text] += 1
        
        if blocked_variants:
            self.logger.warning("Blocked unseen variants", 
                              context={
                                  'blocked_variants': blocked_variants,
                                  'blocked_count': len(blocked_variants),
                                  'valid_count': len(valid_candidates)
                              })
        
        return valid_candidates
    
    def _score_candidates(self, 
                         candidates: List[SpanCandidate],
                         glossary_entries: List[GlossaryEntry],
                         current_text: str,
                         rules: Optional[Dict[str, Any]]) -> List[Tuple[SpanCandidate, float]]:
        """
        Score candidates using engine votes, confidence, and glossary bonuses
        
        Returns:
            List of (candidate, score) tuples
        """
        if not candidates:
            return []
        
        # Group candidates by text for vote counting
        text_groups = defaultdict(list)
        for candidate in candidates:
            text_groups[candidate.text].append(candidate)
        
        # Build glossary lookup for bonuses
        glossary_lookup = {}
        for entry in glossary_entries:
            glossary_lookup[entry.canonical_form] = entry.weight
            for variant in entry.variants:
                glossary_lookup[variant] = entry.weight
        
        scored_candidates = []
        
        for text, group in text_groups.items():
            # Base score: engine vote count and mean confidence
            vote_count = len(group)
            mean_confidence = np.mean([c.confidence for c in group])
            mean_acoustic = np.mean([c.acoustic_score for c in group])
            
            # Start with vote count and confidence
            score = vote_count + mean_confidence
            
            # Add glossary weight bonus
            if text in glossary_lookup:
                glossary_bonus = self.glossary_weight_bonus * glossary_lookup[text]
                score += glossary_bonus
                self.logger.debug("Applied glossary bonus", 
                                context={
                                    'text': text,
                                    'glossary_weight': glossary_lookup[text],
                                    'bonus': glossary_bonus
                                })
            
            # Subtract penalties for low acoustic support
            if mean_acoustic < 0.5:  # Configurable threshold
                acoustic_penalty = (0.5 - mean_acoustic) * 0.3  # Configurable penalty
                score -= acoustic_penalty
            
            # Apply custom rules if provided
            if rules:
                rule_adjustment = self._apply_custom_rules(text, rules)
                score += rule_adjustment
            
            # Add the best candidate from this text group
            best_candidate = max(group, key=lambda c: c.confidence)
            scored_candidates.append((best_candidate, score))
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.debug("Scored candidates", 
                        context={
                            'total_candidates': len(scored_candidates),
                            'top_scores': [(c.text, score) for c, score in scored_candidates[:5]]
                        })
        
        return scored_candidates
    
    def _select_best_candidate(self, 
                              scored_candidates: List[Tuple[SpanCandidate, float]],
                              current_text: str,
                              time_window: Tuple[float, float]) -> VerificationResult:
        """
        Select best candidate based on scoring and margin requirements
        
        Returns:
            VerificationResult with final choice
        """
        if not scored_candidates:
            return VerificationResult(
                verified_text=current_text,
                is_verified=False,
                verification_source='no_valid_candidates',
                score=0.0,
                margin=0.0,
                blocked_variants=[],
                processing_time=0.0,
                verification_metadata={'reason': 'no_valid_candidates'}
            )
        
        # Get top candidate and its score
        best_candidate, best_score = scored_candidates[0]
        
        # Calculate current text score for margin comparison
        current_score = self._calculate_current_text_score(current_text, scored_candidates)
        
        # Calculate margin
        margin = best_score - current_score
        
        # Check if margin meets threshold
        if margin >= self.min_margin:
            # Accept the best candidate
            self.telemetry.change_count += 1
            
            # Determine verification source
            verification_source = 'observed'
            if any(best_candidate.text in entry.canonical_form or 
                   best_candidate.text in entry.variants 
                   for entry in self._get_current_glossary_entries()):
                verification_source = 'glossary'
            
            return VerificationResult(
                verified_text=best_candidate.text,
                is_verified=True,
                verification_source=verification_source,
                score=best_score,
                margin=margin,
                blocked_variants=[],
                processing_time=0.0,
                verification_metadata={
                    'engine': best_candidate.engine_name,
                    'confidence': best_candidate.confidence,
                    'acoustic_score': best_candidate.acoustic_score,
                    'vote_count': len([c for c, _ in scored_candidates if c.text == best_candidate.text])
                }
            )
        else:
            # Keep current text - insufficient margin
            self.telemetry.blocked_insufficient_margin_count += 1
            
            return VerificationResult(
                verified_text=current_text,
                is_verified=False,
                verification_source='current_kept',
                score=current_score,
                margin=margin,
                blocked_variants=[c.text for c, _ in scored_candidates[:3] if c.text != current_text],
                processing_time=0.0,
                verification_metadata={
                    'reason': 'insufficient_margin',
                    'required_margin': self.min_margin,
                    'actual_margin': margin,
                    'best_candidate': best_candidate.text,
                    'best_score': best_score
                }
            )
    
    def _calculate_current_text_score(self, 
                                    current_text: str,
                                    scored_candidates: List[Tuple[SpanCandidate, float]]) -> float:
        """Calculate score for current text based on scoring system"""
        # Find if current text is among candidates
        for candidate, score in scored_candidates:
            if candidate.text == current_text:
                return score
        
        # If current text not in candidates, assign baseline score
        return 0.5  # Baseline score for existing text
    
    def _apply_formatting_rules(self, 
                               result: VerificationResult,
                               span_candidates: List[SpanCandidate]) -> VerificationResult:
        """
        Apply strict formatting rules for numbers and alphanumerics
        
        Returns:
            Updated verification result with formatting preserved
        """
        text = result.verified_text
        
        # Check if this is a number or alphanumeric
        is_number = bool(self.number_pattern.match(text))
        is_alphanumeric = bool(self.alphanumeric_pattern.match(text))
        
        if is_number or is_alphanumeric:
            # For numbers/alphanumerics, require exact character match
            # and preserve formatting like slashes, hyphens
            
            # Find exact matches in candidates
            exact_matches = [c for c in span_candidates if c.text == text]
            
            if not exact_matches and self.block_unseen_variants:
                # Block if no exact match found
                self.logger.warning("Blocked number/alphanumeric without exact match", 
                                  context={
                                      'text': text,
                                      'type': 'number' if is_number else 'alphanumeric',
                                      'candidates': [c.text for c in span_candidates]
                                  })
                
                # Find a safe fallback from candidates
                fallback_text = self._find_safe_fallback(span_candidates, text)
                
                result.verified_text = fallback_text
                result.verification_source = 'formatting_fallback'
                result.verification_metadata['formatting_block'] = True
                result.verification_metadata['original_blocked'] = text
                
                self.telemetry.block_count += 1
        
        return result
    
    def _find_safe_fallback(self, 
                           span_candidates: List[SpanCandidate],
                           blocked_text: str) -> str:
        """Find a safe fallback text from candidates"""
        if not span_candidates:
            return blocked_text  # No choice but to keep original
        
        # Prefer candidates with highest confidence
        best_candidate = max(span_candidates, key=lambda c: c.confidence)
        return best_candidate.text
    
    def _apply_custom_rules(self, text: str, rules: Dict[str, Any]) -> float:
        """Apply custom verification rules for scoring adjustment"""
        adjustment = 0.0
        
        # Example rule implementations
        if 'prefer_uppercase' in rules and text.isupper():
            adjustment += rules['prefer_uppercase']
        
        if 'prefer_titlecase' in rules and text.istitle():
            adjustment += rules['prefer_titlecase']
        
        if 'penalize_lowercase' in rules and text.islower():
            adjustment -= rules['penalize_lowercase']
        
        return adjustment
    
    def _get_current_glossary_entries(self) -> List[GlossaryEntry]:
        """Get current glossary entries (placeholder for integration)"""
        # This will be implemented when integrating with term_bias.py
        return []
    
    def _update_telemetry(self, 
                         result: VerificationResult,
                         span_candidates: List[SpanCandidate],
                         allowed_set: Set[str]):
        """Update telemetry counters and metrics"""
        if result.blocked_variants:
            self.telemetry.block_count += 1
            for variant in result.blocked_variants:
                self.telemetry.top_blocked_variants[variant] += 1
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get current telemetry summary for reporting"""
        if not self.telemetry.processing_times:
            avg_processing_time = 0.0
        else:
            avg_processing_time = np.mean(self.telemetry.processing_times)
        
        return {
            'verifier_invocations': self.telemetry.invocation_count,
            'verifier_block_rate': (self.telemetry.block_count / max(1, self.telemetry.invocation_count)),
            'verifier_change_rate': (self.telemetry.change_count / max(1, self.telemetry.invocation_count)),
            'blocked_unseen_variants_count': self.telemetry.blocked_unseen_count,
            'blocked_insufficient_margin_count': self.telemetry.blocked_insufficient_margin_count,
            'avg_processing_time': avg_processing_time,
            'top_blocked_variants': dict(self.telemetry.top_blocked_variants.most_common(20))
        }
    
    def generate_daily_report(self, project_id: str) -> Dict[str, Any]:
        """Generate daily report of verification activity"""
        summary = self.get_telemetry_summary()
        
        report = {
            'project_id': project_id,
            'date': datetime.now().isoformat(),
            'verification_summary': summary,
            'top_blocked_variants': summary['top_blocked_variants'],
            'recommendations': self._generate_recommendations(summary)
        }
        
        self.telemetry.daily_reports.append(report)
        
        self.logger.info("Generated daily verification report", 
                        context={
                            'project_id': project_id,
                            'invocations': summary['verifier_invocations'],
                            'block_rate': summary['verifier_block_rate'],
                            'top_blocked_count': len(summary['top_blocked_variants'])
                        })
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on telemetry data"""
        recommendations = []
        
        if summary['verifier_block_rate'] > 0.3:
            recommendations.append(
                "High block rate detected. Consider reviewing auto-glossary coverage."
            )
        
        if summary['blocked_unseen_variants_count'] > 50:
            recommendations.append(
                "Many unseen variants blocked. Review top blocked variants for glossary additions."
            )
        
        if summary['avg_processing_time'] > 0.1:
            recommendations.append(
                "High processing time. Consider optimizing candidate filtering."
            )
        
        return recommendations

def create_proper_noun_verifier(config: Optional[Dict[str, Any]] = None) -> ProperNounVerifier:
    """Factory function to create a proper noun verifier instance"""
    return ProperNounVerifier(config=config)

# Utility functions for integration
def is_proper_noun_span(text: str) -> bool:
    """Check if text span contains proper nouns"""
    return bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

def is_number_span(text: str) -> bool:
    """Check if text span contains numbers or alphanumerics"""
    return bool(re.search(r'\b\d+([\/\-\.]\d+)*\b|\b[A-Z0-9]+[\-\/\._]*[A-Z0-9]*\b', text))

def requires_verification(text: str) -> bool:
    """Check if text span requires verification"""
    return is_proper_noun_span(text) or is_number_span(text)