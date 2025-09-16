"""
Auto-Glossary Adaptive Biasing Engine

Applies project-specific term biases to:
- ASR re-ask prompts with vocabulary hints
- Fusion engine prior adjustments for term matching
- Repair engine constraints to prevent term hallucinations
- Decoder-specific hint systems for supported providers

Integrates with existing ensemble pipeline to provide session-scoped
bias lists that improve entity accuracy without forcing insertions.
"""

import re
import json
import time
import math
from typing import Dict, Any, List, Set, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import numpy as np
from difflib import SequenceMatcher

from core.term_store import ProjectTermStore, create_project_term_store
from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class BiasedTerm:
    """Individual biased term with application metadata"""
    canonical_form: str
    variants: Set[str]
    bias_weight: float
    confidence_score: float
    application_contexts: Set[str]  # 'asr_prompt', 'fusion_prior', 'repair_constraint'
    decay_adjusted_weight: float
    session_count: int
    supporting_engines: Set[str]
    example_contexts: List[str]
    term_type: str  # 'proper_noun', 'technical', 'acronym', 'general'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionBiasList:
    """Complete session bias list with application instructions"""
    session_id: str
    project_id: str
    biased_terms: List[BiasedTerm]
    total_bias_terms: int
    generation_timestamp: float
    bias_strength: float
    application_config: Dict[str, Any]
    term_type_distribution: Dict[str, int]
    bias_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BiasApplicationResult:
    """Result from applying bias to a specific operation"""
    operation_type: str  # 'asr_prompt', 'fusion_prior', 'repair_constraint'
    terms_applied: int
    bias_strength_used: float
    application_success: bool
    processing_time: float
    application_metadata: Dict[str, Any]

class ASRPromptBiasApplicator:
    """Applies bias to ASR provider prompts and vocabulary hints"""
    
    def __init__(self, max_prompt_terms: int = 20):
        self.max_prompt_terms = max_prompt_terms
        self.logger = create_enhanced_logger("asr_prompt_bias")
        
        # Provider-specific prompt templates
        self.prompt_templates = {
            'openai': {
                'vocabulary_hint': "Pay attention to these domain-specific terms: {terms}. Use them when acoustically plausible but do not force insertions.",
                'context_prompt': "This audio contains technical discussion with terms like: {top_terms}. Transcribe naturally with attention to these vocabulary items."
            },
            'deepgram': {
                'vocabulary_keywords': True,  # Uses keywords parameter
                'context_prompt': "Technical audio with domain terms: {top_terms}"
            },
            'assemblyai': {
                'vocabulary_boost': True,  # Uses word_boost parameter
                'context_prompt': "Domain-specific audio containing: {top_terms}"
            },
            'faster-whisper': {
                'vocabulary_hint': "Domain vocabulary includes: {terms}. Prefer these terms when acoustically supported.",
                'context_prompt': "Audio with specialized terminology: {top_terms}"
            }
        }
    
    def generate_biased_prompt(self, 
                              provider: str, 
                              biased_terms: List[BiasedTerm], 
                              base_prompt: str = None,
                              context_type: str = "general") -> Dict[str, Any]:
        """
        Generate biased prompt for ASR provider
        
        Args:
            provider: ASR provider name
            biased_terms: List of terms to bias toward
            base_prompt: Optional base prompt to enhance
            context_type: Type of audio context
            
        Returns:
            Dictionary with prompt and provider-specific parameters
        """
        if not biased_terms:
            return {'prompt': base_prompt or '', 'vocabulary_parameters': {}}
        
        # Sort terms by bias weight and limit
        sorted_terms = sorted(biased_terms, key=lambda t: t.bias_weight, reverse=True)
        top_terms = sorted_terms[:self.max_prompt_terms]
        
        # Prepare term lists
        all_forms = []
        for term in top_terms:
            all_forms.append(term.canonical_form)
            all_forms.extend(list(term.variants))
        
        # Remove duplicates while preserving order
        unique_forms = list(dict.fromkeys(all_forms))[:self.max_prompt_terms]
        
        # Get provider-specific template
        templates = self.prompt_templates.get(provider, self.prompt_templates['openai'])
        
        # Generate enhanced prompt
        enhanced_prompt = base_prompt or ""
        vocabulary_parameters = {}
        
        if provider == 'openai':
            # Use prompt enhancement
            if 'vocabulary_hint' in templates:
                vocab_hint = templates['vocabulary_hint'].format(
                    terms=', '.join(unique_forms[:10])  # Limit for readability
                )
                enhanced_prompt = f"{enhanced_prompt}\n{vocab_hint}".strip()
        
        elif provider == 'deepgram':
            # Use keywords parameter
            if templates.get('vocabulary_keywords'):
                vocabulary_parameters['keywords'] = unique_forms[:15]
                vocabulary_parameters['keywords_intensifier'] = 0.3  # Moderate boost
        
        elif provider == 'assemblyai':
            # Use word_boost parameter
            if templates.get('vocabulary_boost'):
                word_boost = {form: min(3000, int(1000 + term.bias_weight * 2000)) 
                             for term, form in zip(top_terms, unique_forms[:10])}
                vocabulary_parameters['word_boost'] = word_boost
        
        elif provider in ['faster-whisper', 'whisper']:
            # Use vocabulary context in prompt
            if 'vocabulary_hint' in templates:
                vocab_hint = templates['vocabulary_hint'].format(
                    terms=', '.join(unique_forms[:8])
                )
                enhanced_prompt = f"{enhanced_prompt}\n{vocab_hint}".strip()
        
        # Add context-aware prompt if no base prompt
        if not base_prompt and 'context_prompt' in templates:
            context_prompt = templates['context_prompt'].format(
                top_terms=', '.join([t.canonical_form for t in top_terms[:5]])
            )
            enhanced_prompt = context_prompt
        
        result = {
            'prompt': enhanced_prompt,
            'vocabulary_parameters': vocabulary_parameters,
            'biased_terms_count': len(top_terms),
            'total_forms': len(unique_forms)
        }
        
        self.logger.debug("Generated biased prompt", 
                         context={
                             'provider': provider,
                             'biased_terms': len(top_terms),
                             'vocabulary_params': len(vocabulary_parameters),
                             'prompt_length': len(enhanced_prompt)
                         })
        
        return result

class FusionPriorBiasApplicator:
    """Applies bias to fusion engine token posteriors"""
    
    def __init__(self, 
                 max_prior_boost: float = 0.15,
                 exact_match_bonus: float = 0.10,
                 variant_match_bonus: float = 0.05):
        """
        Initialize fusion prior bias applicator
        
        Args:
            max_prior_boost: Maximum prior boost for any term
            exact_match_bonus: Bonus for exact canonical form matches
            variant_match_bonus: Bonus for variant matches
        """
        self.max_prior_boost = max_prior_boost
        self.exact_match_bonus = exact_match_bonus
        self.variant_match_bonus = variant_match_bonus
        self.logger = create_enhanced_logger("fusion_prior_bias")
        
        # Precompile term lookup structures for efficiency
        self._term_lookup = {}
        self._variant_lookup = {}
        self._last_update_time = 0
    
    def update_bias_terms(self, biased_terms: List[BiasedTerm]):
        """Update internal lookup structures for biased terms"""
        self._term_lookup = {}
        self._variant_lookup = {}
        
        for term in biased_terms:
            # Canonical form lookup
            canonical_lower = term.canonical_form.lower()
            self._term_lookup[canonical_lower] = term
            
            # Variant lookup
            for variant in term.variants:
                variant_lower = variant.lower()
                if variant_lower not in self._variant_lookup:
                    self._variant_lookup[variant_lower] = []
                self._variant_lookup[variant_lower].append(term)
        
        self._last_update_time = time.time()
        
        self.logger.debug("Updated bias term lookups", 
                         context={
                             'canonical_terms': len(self._term_lookup),
                             'variant_mappings': len(self._variant_lookup),
                             'total_terms': len(biased_terms)
                         })
    
    def calculate_token_bias(self, 
                           token: str, 
                           baseline_posterior: float,
                           engine_name: str = None) -> Tuple[float, Optional[BiasedTerm]]:
        """
        Calculate bias adjustment for a token
        
        Args:
            token: Token to check for bias
            baseline_posterior: Original token posterior
            engine_name: Optional engine name for engine-specific bias
            
        Returns:
            (adjusted_posterior, matched_term) tuple
        """
        if not token or not self._term_lookup and not self._variant_lookup:
            return baseline_posterior, None
        
        token_lower = token.lower()
        matched_term = None
        bias_adjustment = 0.0
        
        # Check for exact canonical match first
        if token_lower in self._term_lookup:
            matched_term = self._term_lookup[token_lower]
            # Calculate bias based on term weight and decay
            base_bias = matched_term.bias_weight * matched_term.decay_adjusted_weight
            bias_adjustment = min(self.max_prior_boost, base_bias * self.exact_match_bonus)
            
            self.logger.debug("Exact match bias applied", 
                            context={
                                'token': token,
                                'matched_term': matched_term.canonical_form,
                                'bias_adjustment': bias_adjustment
                            })
        
        # Check for variant match
        elif token_lower in self._variant_lookup:
            variant_terms = self._variant_lookup[token_lower]
            # Use highest-weight term if multiple matches
            matched_term = max(variant_terms, key=lambda t: t.bias_weight)
            base_bias = matched_term.bias_weight * matched_term.decay_adjusted_weight
            bias_adjustment = min(self.max_prior_boost, base_bias * self.variant_match_bonus)
            
            self.logger.debug("Variant match bias applied", 
                            context={
                                'token': token,
                                'matched_term': matched_term.canonical_form,
                                'variant_of': matched_term.canonical_form,
                                'bias_adjustment': bias_adjustment
                            })
        
        # Engine-specific bias modulation
        if matched_term and engine_name and engine_name in matched_term.supporting_engines:
            # Boost bias if this engine has seen the term before
            bias_adjustment *= 1.2
        
        # Apply bias adjustment
        adjusted_posterior = min(1.0, baseline_posterior + bias_adjustment)
        
        return adjusted_posterior, matched_term
    
    def apply_fusion_bias(self, 
                         confusion_network_tokens: List[Dict[str, Any]], 
                         bias_strength: float = 1.0) -> List[Dict[str, Any]]:
        """
        Apply bias to confusion network token posteriors
        
        Args:
            confusion_network_tokens: List of token dictionaries with posteriors
            bias_strength: Global bias strength multiplier (0.0-2.0)
            
        Returns:
            List of tokens with adjusted posteriors
        """
        if not confusion_network_tokens or bias_strength <= 0:
            return confusion_network_tokens
        
        adjusted_tokens = []
        bias_applications = 0
        
        for token_info in confusion_network_tokens:
            token = token_info.get('token', '')
            original_posterior = token_info.get('posterior', 0.0)
            engine_name = token_info.get('source_engine')
            
            # Calculate bias adjustment
            adjusted_posterior, matched_term = self.calculate_token_bias(
                token, original_posterior, engine_name
            )
            
            # Apply global bias strength
            if matched_term:
                bias_amount = adjusted_posterior - original_posterior
                adjusted_posterior = original_posterior + (bias_amount * bias_strength)
                bias_applications += 1
            
            # Create adjusted token info
            adjusted_token_info = token_info.copy()
            adjusted_token_info['posterior'] = adjusted_posterior
            
            if matched_term:
                adjusted_token_info['bias_metadata'] = {
                    'matched_term': matched_term.canonical_form,
                    'bias_applied': adjusted_posterior - original_posterior,
                    'bias_source': 'project_glossary'
                }
            
            adjusted_tokens.append(adjusted_token_info)
        
        self.logger.debug("Applied fusion bias", 
                         context={
                             'total_tokens': len(confusion_network_tokens),
                             'bias_applications': bias_applications,
                             'bias_strength': bias_strength
                         })
        
        return adjusted_tokens

class RepairConstraintBiasApplicator:
    """Applies bias constraints to repair engine operations"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("repair_constraint_bias")
        
        # Term lookup for constraint checking
        self._term_forms = set()
        self._variant_map = {}
    
    def update_constraint_terms(self, biased_terms: List[BiasedTerm]):
        """Update constraint terms for repair operations"""
        self._term_forms.clear()
        self._variant_map.clear()
        
        for term in biased_terms:
            # Add canonical form
            canonical_lower = term.canonical_form.lower()
            self._term_forms.add(canonical_lower)
            
            # Map variants to canonical
            for variant in term.variants:
                variant_lower = variant.lower()
                self._term_forms.add(variant_lower)
                self._variant_map[variant_lower] = term.canonical_form
        
        self.logger.debug("Updated repair constraints", 
                         context={
                             'total_forms': len(self._term_forms),
                             'variant_mappings': len(self._variant_map)
                         })
    
    def validate_proper_noun_edit(self, 
                                 original_token: str, 
                                 proposed_edit: str,
                                 edit_confidence: float) -> Tuple[bool, str, Optional[str]]:
        """
        Validate proposed edits to proper nouns against term base
        
        Args:
            original_token: Original token being edited
            proposed_edit: Proposed replacement token
            edit_confidence: Confidence in the edit
            
        Returns:
            (is_valid, final_token, constraint_reason) tuple
        """
        original_lower = original_token.lower()
        proposed_lower = proposed_edit.lower()
        
        # If original is in term base, only allow edits to other term base entries
        if original_lower in self._term_forms:
            if proposed_lower in self._term_forms:
                # Edit between known terms - allowed
                canonical_form = self._variant_map.get(proposed_lower, proposed_edit)
                return True, canonical_form, "edit_between_known_terms"
            else:
                # Edit to unknown term - blocked to prevent hallucination
                return False, original_token, "prevent_hallucination_from_known_term"
        
        # If proposed edit is a known term, prefer it
        if proposed_lower in self._term_forms:
            canonical_form = self._variant_map.get(proposed_lower, proposed_edit)
            return True, canonical_form, "prefer_known_term"
        
        # Neither term is known - allow edit if confidence is high enough
        confidence_threshold = 0.8  # High bar for unknown-to-unknown edits
        if edit_confidence >= confidence_threshold:
            return True, proposed_edit, "high_confidence_unknown_edit"
        else:
            return False, original_token, "low_confidence_unknown_edit"
    
    def constrain_repair_candidates(self, 
                                   repair_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply constraints to repair candidates to prevent hallucinations
        
        Args:
            repair_candidates: List of repair candidate dictionaries
            
        Returns:
            Filtered and adjusted repair candidates
        """
        if not repair_candidates:
            return repair_candidates
        
        constrained_candidates = []
        constraint_applications = 0
        
        for candidate in repair_candidates:
            original_segment = candidate.get('original_segment', {})
            repaired_segment = candidate.get('repaired_segment', {})
            
            # Apply constraints to segment text
            original_text = original_segment.get('text', '')
            repaired_text = repaired_segment.get('text', '')
            
            if original_text and repaired_text:
                constrained_text, was_constrained = self._constrain_segment_text(
                    original_text, repaired_text
                )
                
                if was_constrained:
                    constraint_applications += 1
                    # Update repaired segment
                    constrained_candidate = candidate.copy()
                    constrained_segment = repaired_segment.copy()
                    constrained_segment['text'] = constrained_text
                    constrained_candidate['repaired_segment'] = constrained_segment
                    
                    # Add constraint metadata
                    constrained_candidate['constraint_metadata'] = {
                        'constraints_applied': True,
                        'original_repair': repaired_text,
                        'constrained_repair': constrained_text
                    }
                    
                    constrained_candidates.append(constrained_candidate)
                else:
                    constrained_candidates.append(candidate)
            else:
                constrained_candidates.append(candidate)
        
        self.logger.debug("Applied repair constraints", 
                         context={
                             'total_candidates': len(repair_candidates),
                             'constraint_applications': constraint_applications
                         })
        
        return constrained_candidates
    
    def _constrain_segment_text(self, original_text: str, repaired_text: str) -> Tuple[str, bool]:
        """Apply constraints to individual segment text repair"""
        # For now, implement basic word-level constraint checking
        # In production, this could use more sophisticated NLP
        
        original_words = original_text.split()
        repaired_words = repaired_text.split()
        
        if len(original_words) != len(repaired_words):
            # Different word counts - harder to constrain, allow for now
            return repaired_text, False
        
        constrained_words = []
        was_constrained = False
        
        for orig_word, repair_word in zip(original_words, repaired_words):
            # Clean words for comparison
            orig_clean = re.sub(r'[^\w]', '', orig_word.lower())
            repair_clean = re.sub(r'[^\w]', '', repair_word.lower())
            
            if orig_clean != repair_clean:
                # Word changed - check constraint
                is_valid, final_word, reason = self.validate_proper_noun_edit(
                    orig_clean, repair_clean, 0.7  # Assume moderate confidence
                )
                
                if not is_valid:
                    # Use original word
                    constrained_words.append(orig_word)
                    was_constrained = True
                else:
                    # Use validated word (preserve punctuation from repair)
                    if final_word != repair_clean:
                        # Canonical form substitution
                        final_with_punct = re.sub(r'\w+', final_word, repair_word, count=1)
                        constrained_words.append(final_with_punct)
                        was_constrained = True
                    else:
                        constrained_words.append(repair_word)
            else:
                constrained_words.append(repair_word)
        
        constrained_text = ' '.join(constrained_words)
        return constrained_text, was_constrained

class AdaptiveBiasingEngine:
    """Main adaptive biasing engine that coordinates all bias applications"""
    
    def __init__(self,
                 term_store: ProjectTermStore = None,
                 default_bias_strength: float = 0.7,
                 max_bias_terms_per_session: int = 50,
                 min_term_confidence: float = 0.5,
                 enable_asr_bias: bool = True,
                 enable_fusion_bias: bool = True,
                 enable_repair_constraints: bool = True):
        """
        Initialize adaptive biasing engine
        
        Args:
            term_store: Project term store instance
            default_bias_strength: Default bias strength (0.0-2.0)
            max_bias_terms_per_session: Maximum terms to bias per session
            min_term_confidence: Minimum confidence for bias application
            enable_asr_bias: Whether to apply ASR prompt biasing
            enable_fusion_bias: Whether to apply fusion prior biasing
            enable_repair_constraints: Whether to apply repair constraints
        """
        self.term_store = term_store or create_project_term_store()
        self.default_bias_strength = default_bias_strength
        self.max_bias_terms_per_session = max_bias_terms_per_session
        self.min_term_confidence = min_term_confidence
        self.enable_asr_bias = enable_asr_bias
        self.enable_fusion_bias = enable_fusion_bias
        self.enable_repair_constraints = enable_repair_constraints
        
        # Initialize bias applicators
        self.asr_bias_applicator = ASRPromptBiasApplicator() if enable_asr_bias else None
        self.fusion_bias_applicator = FusionPriorBiasApplicator() if enable_fusion_bias else None
        self.repair_constraint_applicator = RepairConstraintBiasApplicator() if enable_repair_constraints else None
        
        # Session cache
        self._session_bias_cache = {}
        
        self.logger = create_enhanced_logger("adaptive_biasing_engine")
        
        self.logger.info("Adaptive biasing engine initialized", 
                        context={
                            'bias_strength': default_bias_strength,
                            'max_terms': max_bias_terms_per_session,
                            'min_confidence': min_term_confidence,
                            'asr_bias': enable_asr_bias,
                            'fusion_bias': enable_fusion_bias,
                            'repair_constraints': enable_repair_constraints
                        })
    
    def generate_session_bias_list(self, 
                                  project_id: str, 
                                  session_id: str,
                                  bias_strength: float = None,
                                  context_type: str = "general") -> SessionBiasList:
        """
        Generate bias list for a session
        
        Args:
            project_id: Project identifier
            session_id: Session identifier
            bias_strength: Override bias strength for this session
            context_type: Context type for bias selection
            
        Returns:
            SessionBiasList with biased terms and metadata
        """
        start_time = time.time()
        bias_strength = bias_strength or self.default_bias_strength
        
        # Check cache first
        cache_key = f"{project_id}_{session_id}_{bias_strength}_{context_type}"
        if cache_key in self._session_bias_cache:
            cached_list, cache_time = self._session_bias_cache[cache_key]
            if time.time() - cache_time < 300:  # 5 minute cache
                return cached_list
        
        self.logger.info("Generating session bias list", 
                        context={
                            'project_id': project_id,
                            'session_id': session_id,
                            'bias_strength': bias_strength,
                            'context_type': context_type
                        })
        
        # Get top terms from term store
        top_terms = self.term_store.get_top_terms(
            project_id=project_id,
            limit=self.max_bias_terms_per_session,
            min_confidence=self.min_term_confidence,
            min_session_count=1
        )
        
        # Convert to BiasedTerm objects
        biased_terms = []
        term_type_distribution = defaultdict(int)
        
        for term_data in top_terms:
            # Determine term type
            term_type = self._classify_term_type(term_data['token'])
            term_type_distribution[term_type] += 1
            
            # Create BiasedTerm
            biased_term = BiasedTerm(
                canonical_form=term_data['token'],
                variants=set(term_data.get('variants', [])),
                bias_weight=term_data['weight'],
                confidence_score=term_data['confidence'],
                application_contexts={'asr_prompt', 'fusion_prior', 'repair_constraint'},
                decay_adjusted_weight=term_data.get('decay_factor', 1.0),
                session_count=term_data.get('session_count', 1),
                supporting_engines=set(term_data.get('supporting_engines', [])),
                example_contexts=[],  # Could be populated from term store
                term_type=term_type
            )
            
            biased_terms.append(biased_term)
        
        # Create session bias list
        bias_list = SessionBiasList(
            session_id=session_id,
            project_id=project_id,
            biased_terms=biased_terms,
            total_bias_terms=len(biased_terms),
            generation_timestamp=time.time(),
            bias_strength=bias_strength,
            application_config={
                'enable_asr_bias': self.enable_asr_bias,
                'enable_fusion_bias': self.enable_fusion_bias,
                'enable_repair_constraints': self.enable_repair_constraints,
                'context_type': context_type
            },
            term_type_distribution=dict(term_type_distribution),
            bias_metadata={
                'generation_time': time.time() - start_time,
                'source_terms_available': len(top_terms),
                'min_confidence_threshold': self.min_term_confidence
            }
        )
        
        # Update applicator caches
        if self.fusion_bias_applicator:
            self.fusion_bias_applicator.update_bias_terms(biased_terms)
        if self.repair_constraint_applicator:
            self.repair_constraint_applicator.update_constraint_terms(biased_terms)
        
        # Cache the result
        self._session_bias_cache[cache_key] = (bias_list, time.time())
        
        self.logger.info("Session bias list generated", 
                        context={
                            'project_id': project_id,
                            'session_id': session_id,
                            'bias_terms': len(biased_terms),
                            'term_types': dict(term_type_distribution),
                            'generation_time': time.time() - start_time
                        })
        
        return bias_list
    
    def apply_asr_bias(self, 
                      session_bias_list: SessionBiasList,
                      provider: str,
                      base_prompt: str = None) -> BiasApplicationResult:
        """Apply bias to ASR provider prompt"""
        if not self.enable_asr_bias or not self.asr_bias_applicator:
            return BiasApplicationResult(
                operation_type='asr_prompt',
                terms_applied=0,
                bias_strength_used=0.0,
                application_success=False,
                processing_time=0.0,
                application_metadata={'reason': 'asr_bias_disabled'}
            )
        
        start_time = time.time()
        
        # Filter terms for ASR application
        asr_terms = [term for term in session_bias_list.biased_terms 
                    if 'asr_prompt' in term.application_contexts]
        
        try:
            # Generate biased prompt
            prompt_result = self.asr_bias_applicator.generate_biased_prompt(
                provider=provider,
                biased_terms=asr_terms,
                base_prompt=base_prompt
            )
            
            processing_time = time.time() - start_time
            
            return BiasApplicationResult(
                operation_type='asr_prompt',
                terms_applied=prompt_result['biased_terms_count'],
                bias_strength_used=session_bias_list.bias_strength,
                application_success=True,
                processing_time=processing_time,
                application_metadata={
                    'provider': provider,
                    'prompt_length': len(prompt_result['prompt']),
                    'vocabulary_parameters': prompt_result['vocabulary_parameters'],
                    'total_forms': prompt_result['total_forms']
                }
            )
            
        except Exception as e:
            self.logger.error("ASR bias application failed", 
                            context={'provider': provider, 'error': str(e)})
            return BiasApplicationResult(
                operation_type='asr_prompt',
                terms_applied=0,
                bias_strength_used=0.0,
                application_success=False,
                processing_time=time.time() - start_time,
                application_metadata={'error': str(e)}
            )
    
    def apply_fusion_bias(self, 
                         session_bias_list: SessionBiasList,
                         confusion_network_tokens: List[Dict[str, Any]]) -> BiasApplicationResult:
        """Apply bias to fusion engine confusion networks"""
        if not self.enable_fusion_bias or not self.fusion_bias_applicator:
            return BiasApplicationResult(
                operation_type='fusion_prior',
                terms_applied=0,
                bias_strength_used=0.0,
                application_success=False,
                processing_time=0.0,
                application_metadata={'reason': 'fusion_bias_disabled'}
            )
        
        start_time = time.time()
        
        try:
            # Apply bias to confusion network
            adjusted_tokens = self.fusion_bias_applicator.apply_fusion_bias(
                confusion_network_tokens=confusion_network_tokens,
                bias_strength=session_bias_list.bias_strength
            )
            
            # Count bias applications
            bias_applications = sum(1 for token in adjusted_tokens 
                                  if token.get('bias_metadata'))
            
            processing_time = time.time() - start_time
            
            return BiasApplicationResult(
                operation_type='fusion_prior',
                terms_applied=bias_applications,
                bias_strength_used=session_bias_list.bias_strength,
                application_success=True,
                processing_time=processing_time,
                application_metadata={
                    'total_tokens': len(confusion_network_tokens),
                    'adjusted_tokens': len(adjusted_tokens),
                    'bias_applications': bias_applications
                }
            )
            
        except Exception as e:
            self.logger.error("Fusion bias application failed", 
                            context={'error': str(e)})
            return BiasApplicationResult(
                operation_type='fusion_prior',
                terms_applied=0,
                bias_strength_used=0.0,
                application_success=False,
                processing_time=time.time() - start_time,
                application_metadata={'error': str(e)}
            )
    
    def apply_repair_constraints(self, 
                               session_bias_list: SessionBiasList,
                               repair_candidates: List[Dict[str, Any]]) -> BiasApplicationResult:
        """Apply constraints to repair engine candidates"""
        if not self.enable_repair_constraints or not self.repair_constraint_applicator:
            return BiasApplicationResult(
                operation_type='repair_constraint',
                terms_applied=0,
                bias_strength_used=0.0,
                application_success=False,
                processing_time=0.0,
                application_metadata={'reason': 'repair_constraints_disabled'}
            )
        
        start_time = time.time()
        
        try:
            # Apply constraints
            constrained_candidates = self.repair_constraint_applicator.constrain_repair_candidates(
                repair_candidates
            )
            
            # Count constraint applications
            constraint_applications = sum(1 for candidate in constrained_candidates
                                        if candidate.get('constraint_metadata', {}).get('constraints_applied'))
            
            processing_time = time.time() - start_time
            
            return BiasApplicationResult(
                operation_type='repair_constraint',
                terms_applied=constraint_applications,
                bias_strength_used=session_bias_list.bias_strength,
                application_success=True,
                processing_time=processing_time,
                application_metadata={
                    'total_candidates': len(repair_candidates),
                    'constrained_candidates': len(constrained_candidates),
                    'constraint_applications': constraint_applications
                }
            )
            
        except Exception as e:
            self.logger.error("Repair constraint application failed", 
                            context={'error': str(e)})
            return BiasApplicationResult(
                operation_type='repair_constraint',
                terms_applied=0,
                bias_strength_used=0.0,
                application_success=False,
                processing_time=time.time() - start_time,
                application_metadata={'error': str(e)}
            )
    
    def _classify_term_type(self, token: str) -> str:
        """Classify term type for bias application strategy"""
        if not token:
            return 'general'
        
        # Acronym detection
        if len(token) >= 2 and token.isupper():
            return 'acronym'
        
        # Proper noun detection
        if len(token) > 1 and token[0].isupper() and any(c.islower() for c in token[1:]):
            return 'proper_noun'
        
        # Technical term detection
        if (any(c.isdigit() for c in token) or 
            '-' in token or 
            any(c.isupper() for c in token[1:]) and any(c.islower() for c in token)):
            return 'technical'
        
        return 'general'

def create_adaptive_biasing_engine(term_store: ProjectTermStore = None, **config) -> AdaptiveBiasingEngine:
    """Factory function to create adaptive biasing engine"""
    return AdaptiveBiasingEngine(
        term_store=term_store,
        default_bias_strength=config.get('default_bias_strength', 0.7),
        max_bias_terms_per_session=config.get('max_bias_terms_per_session', 50),
        min_term_confidence=config.get('min_term_confidence', 0.5),
        enable_asr_bias=config.get('enable_asr_bias', True),
        enable_fusion_bias=config.get('enable_fusion_bias', True),
        enable_repair_constraints=config.get('enable_repair_constraints', True)
    )

# Example usage and testing
if __name__ == "__main__":
    # Test the adaptive biasing engine
    from core.term_store import create_project_term_store
    
    # Create components
    term_store = create_project_term_store(storage_base_path='/tmp/test_term_bases')
    biasing_engine = create_adaptive_biasing_engine(term_store)
    
    # Generate session bias list
    bias_list = biasing_engine.generate_session_bias_list(
        project_id='test_project',
        session_id='test_session'
    )
    
    print(f"Generated bias list with {bias_list.total_bias_terms} terms")
    
    # Test ASR bias application
    asr_result = biasing_engine.apply_asr_bias(
        bias_list, 
        provider='openai',
        base_prompt="Transcribe this meeting audio."
    )
    
    print(f"ASR bias applied: {asr_result.terms_applied} terms, success: {asr_result.application_success}")
    
    # Test fusion bias application
    mock_tokens = [
        {'token': 'ModelX-150', 'posterior': 0.6, 'source_engine': 'whisper'},
        {'token': 'revenue', 'posterior': 0.8, 'source_engine': 'deepgram'}
    ]
    
    fusion_result = biasing_engine.apply_fusion_bias(bias_list, mock_tokens)
    print(f"Fusion bias applied: {fusion_result.terms_applied} terms, success: {fusion_result.application_success}")