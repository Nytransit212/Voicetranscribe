"""
Post-Fusion Punctuation Engine for US Meeting Transcripts

Implements transformer-based punctuation restoration and light disfluency 
normalization specifically tuned for US meeting contexts. Applied after 
fusion consensus but before final output generation.
"""

import re
import time
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

# Suppress transformers warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

try:
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification, 
        pipeline, logging as transformers_logging
    )
    # Set transformers logging to error level to reduce noise
    transformers_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForTokenClassification = None
    pipeline = None

from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class DisfluencyPattern:
    """Pattern for detecting and normalizing disfluencies"""
    pattern_type: str  # "filler", "repeat", "false_start", "pause_marker"
    regex_pattern: str
    replacement: str
    confidence_threshold: float = 0.8
    context_dependent: bool = False
    preserve_timing: bool = True

@dataclass
class PunctuationCandidate:
    """Candidate punctuation for a position"""
    position: int  # Character position in text
    punctuation: str  # ".", "?", "!", ",", ":", ";", etc.
    confidence: float
    context_score: float
    meeting_relevance_score: float
    source: str  # "transformer", "rule_based", "context"

@dataclass
class NormalizedSegment:
    """Segment after disfluency normalization"""
    original_text: str
    normalized_text: str
    removed_tokens: List[Tuple[int, str]]  # (position, token) pairs
    confidence: float
    normalization_applied: List[str]  # Types of normalization applied

@dataclass
class PunctuatedSegment:
    """Segment after punctuation restoration"""
    start_time: float
    end_time: float
    speaker_id: Optional[str]
    original_text: str
    punctuated_text: str
    punctuation_confidence: float
    word_level_punctuation: List[Dict[str, Any]]
    disfluency_normalization: Optional[NormalizedSegment] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PunctuationResult:
    """Complete punctuation restoration result"""
    segments: List[PunctuatedSegment]
    overall_confidence: float
    punctuation_metrics: Dict[str, Any]
    disfluency_metrics: Dict[str, Any]
    processing_time: float
    model_info: Dict[str, str]
    normalization_level: str

class MeetingVocabularyHandler:
    """Handles meeting-specific vocabulary and context detection"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("meeting_vocabulary")
        
        # Meeting-specific vocabulary patterns
        self.formal_indicators = {
            # Meeting structure terms
            'agenda', 'minutes', 'action items', 'follow up', 'next steps',
            'meeting', 'discussion', 'presentation', 'proposal', 'decision',
            'motion', 'second', 'vote', 'approved', 'rejected', 'tabled',
            # Business terms
            'quarterly', 'revenue', 'budget', 'forecast', 'strategy', 'metrics',
            'kpi', 'roi', 'deliverable', 'milestone', 'deadline', 'timeline',
            # Professional titles
            'ceo', 'cfo', 'cto', 'vp', 'director', 'manager', 'lead',
            'coordinator', 'analyst', 'specialist', 'consultant', 'advisor'
        }
        
        self.informal_indicators = {
            # Casual expressions
            'yeah', 'yep', 'nope', 'okay', 'ok', 'sure', 'alright',
            'gonna', 'wanna', 'kinda', 'sorta', 'dunno',
            # Discourse markers
            'anyway', 'basically', 'actually', 'obviously', 'definitely',
            'totally', 'absolutely', 'exactly', 'right', 'like'
        }
        
        # Technical meeting terms requiring proper capitalization
        self.technical_terms = {
            'api', 'ui', 'ux', 'qa', 'ci', 'cd', 'mvp', 'poc', 'sla', 'kpi',
            'ai', 'ml', 'nlp', 'aws', 'gcp', 'azure', 'docker', 'kubernetes',
            'javascript', 'python', 'react', 'angular', 'nodejs', 'sql'
        }
        
        # Common meeting participant name patterns
        self.name_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper names
            r'\b[A-Z][a-z]+\s+from\s+[A-Z][a-z]+\b',  # "John from Marketing"
        ]
        
    def detect_formality_level(self, text: str) -> Tuple[str, float]:
        """
        Detect formality level of text (formal, informal, mixed)
        
        Args:
            text: Input text to analyze
            
        Returns:
            (formality_level, confidence) tuple
        """
        words = text.lower().split()
        word_count = len(words)
        
        if word_count == 0:
            return "neutral", 1.0
        
        formal_count = sum(1 for word in words if word in self.formal_indicators)
        informal_count = sum(1 for word in words if word in self.informal_indicators)
        
        formal_ratio = formal_count / word_count
        informal_ratio = informal_count / word_count
        
        # Determine formality level
        if formal_ratio > informal_ratio * 2 and formal_ratio > 0.05:
            return "formal", min(formal_ratio * 10, 1.0)
        elif informal_ratio > formal_ratio * 2 and informal_ratio > 0.05:
            return "informal", min(informal_ratio * 10, 1.0)
        elif abs(formal_ratio - informal_ratio) < 0.02:
            return "mixed", 0.8
        else:
            return "neutral", 0.6
    
    def identify_technical_terms(self, text: str) -> List[Tuple[int, str]]:
        """
        Identify technical terms that need proper capitalization
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of (position, proper_form) tuples
        """
        corrections = []
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            # Check technical terms
            if word_lower in self.technical_terms:
                if word_lower in ['api', 'ui', 'ux', 'qa', 'ci', 'cd', 'ai', 'ml', 'nlp']:
                    corrections.append((i, word_lower.upper()))
                elif word_lower in ['javascript', 'python', 'nodejs']:
                    corrections.append((i, word_lower.capitalize()))
                elif word_lower == 'sql':
                    corrections.append((i, 'SQL'))
        
        return corrections
    
    def extract_meeting_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract meeting-specific entities (names, companies, products)
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'names': [],
            'companies': [],
            'products': [],
            'roles': []
        }
        
        # Extract proper names
        for pattern in self.name_patterns:
            matches = re.findall(pattern, text)
            entities['names'].extend(matches)
        
        # Extract company/organization mentions
        company_pattern = r'\b(?:at|from|with)\s+([A-Z][a-zA-Z\s&]+?)(?:\s|,|\.|\b)'
        companies = re.findall(company_pattern, text)
        entities['companies'].extend(companies)
        
        return entities

class DisfluencyNormalizer:
    """Handles disfluency detection and normalization"""
    
    def __init__(self, normalization_level: str = "light"):
        """
        Initialize disfluency normalizer
        
        Args:
            normalization_level: "light", "moderate", or "aggressive"
        """
        self.normalization_level = normalization_level
        self.logger = create_enhanced_logger("disfluency_normalizer")
        
        # Define disfluency patterns by normalization level
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, List[DisfluencyPattern]]:
        """Initialize disfluency patterns for each normalization level"""
        patterns = {
            "light": [
                # Excessive fillers (3+ repetitions)
                DisfluencyPattern(
                    pattern_type="filler",
                    regex_pattern=r'\b(um|uh|er|ah){3,}\b',
                    replacement='',
                    confidence_threshold=0.9
                ),
                # Clear false starts with restart
                DisfluencyPattern(
                    pattern_type="false_start",
                    regex_pattern=r'\b(\w+)\s+\w+\s+\1\b',
                    replacement=r'\1',
                    confidence_threshold=0.8
                ),
                # Multiple consecutive pause markers
                DisfluencyPattern(
                    pattern_type="pause_marker",
                    regex_pattern=r'(\[pause\]|\[silence\]){2,}',
                    replacement='[pause]',
                    confidence_threshold=0.95
                )
            ],
            
            "moderate": [
                # All light patterns plus:
                # Moderate filler reduction (2+ repetitions)
                DisfluencyPattern(
                    pattern_type="filler",
                    regex_pattern=r'\b(um|uh|er|ah){2,}\b',
                    replacement='',
                    confidence_threshold=0.85
                ),
                # Single excessive fillers in formal contexts
                DisfluencyPattern(
                    pattern_type="filler",
                    regex_pattern=r'\b(um|uh|er|ah)\b',
                    replacement='',
                    confidence_threshold=0.7,
                    context_dependent=True
                ),
                # Simple word repetitions
                DisfluencyPattern(
                    pattern_type="repeat",
                    regex_pattern=r'\b(\w+)\s+\1\b',
                    replacement=r'\1',
                    confidence_threshold=0.8
                )
            ],
            
            "aggressive": [
                # All moderate patterns plus:
                # All single fillers
                DisfluencyPattern(
                    pattern_type="filler",
                    regex_pattern=r'\b(um|uh|er|ah|like|you know)\b',
                    replacement='',
                    confidence_threshold=0.6
                ),
                # Partial word repetitions
                DisfluencyPattern(
                    pattern_type="repeat",
                    regex_pattern=r'\b(\w{1,3})-\s*\1\w+\b',
                    replacement=r'\1',
                    confidence_threshold=0.7
                ),
                # Discourse markers in excess
                DisfluencyPattern(
                    pattern_type="discourse_marker",
                    regex_pattern=r'\b(basically|actually|obviously){2,}\b',
                    replacement=r'\1',
                    confidence_threshold=0.8
                )
            ]
        }
        
        return patterns
    
    def normalize_segment(self, text: str, formality_level: str = "neutral") -> NormalizedSegment:
        """
        Apply disfluency normalization to a text segment
        
        Args:
            text: Input text to normalize
            formality_level: Context formality level
            
        Returns:
            NormalizedSegment with normalization results
        """
        original_text = text
        normalized_text = text
        removed_tokens = []
        normalization_applied = []
        
        # Get patterns for current normalization level
        level_patterns = self.patterns.get(self.normalization_level, self.patterns["light"])
        
        # Apply each pattern
        for pattern in level_patterns:
            # Check context dependency
            if pattern.context_dependent and formality_level == "informal":
                continue  # Skip formal-context patterns in informal speech
            
            # Apply pattern
            before_length = len(normalized_text)
            matches = list(re.finditer(pattern.regex_pattern, normalized_text, re.IGNORECASE))
            
            if matches:
                # Track removed tokens
                for match in reversed(matches):  # Reverse to maintain positions
                    start_pos = match.start()
                    removed_text = match.group()
                    removed_tokens.append((start_pos, removed_text))
                
                # Apply replacement
                normalized_text = re.sub(
                    pattern.regex_pattern, 
                    pattern.replacement, 
                    normalized_text, 
                    flags=re.IGNORECASE
                )
                
                after_length = len(normalized_text)
                if after_length != before_length:
                    normalization_applied.append(pattern.pattern_type)
        
        # Clean up extra whitespace
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        # Calculate confidence based on amount of normalization
        confidence = max(0.3, 1.0 - (len(removed_tokens) * 0.1))
        
        return NormalizedSegment(
            original_text=original_text,
            normalized_text=normalized_text,
            removed_tokens=removed_tokens,
            confidence=confidence,
            normalization_applied=list(set(normalization_applied))
        )

class TransformerPunctuationModel:
    """Wrapper for transformer-based punctuation restoration models"""
    
    def __init__(self, model_name: str = "oliverguhr/fullstop-punctuation-multilang-large"):
        """
        Initialize punctuation model
        
        Args:
            model_name: Hugging Face model name for punctuation restoration
        """
        self.model_name = model_name
        self.logger = create_enhanced_logger("punctuation_model")
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers library not available - using mock punctuation")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            return
        
        try:
            # Initialize model and tokenizer
            self.logger.info(f"Loading punctuation model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # Create pipeline
            self.pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            
            self.logger.info(f"Successfully loaded punctuation model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load punctuation model: {e}")
            self.logger.info("Falling back to rule-based punctuation")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def is_available(self) -> bool:
        """Check if transformer model is available"""
        return self.pipeline is not None
    
    def predict_punctuation(self, text: str) -> List[PunctuationCandidate]:
        """
        Predict punctuation for input text
        
        Args:
            text: Input text without punctuation
            
        Returns:
            List of punctuation candidates
        """
        if not self.is_available():
            return self._rule_based_punctuation(text)
        
        try:
            # Get token predictions
            results = self.pipeline(text)
            candidates = []
            
            # Convert to PunctuationCandidate objects
            for result in results:
                if result['entity_group'] in ['PERIOD', 'COMMA', 'QUESTION', 'EXCLAMATION']:
                    punctuation_map = {
                        'PERIOD': '.',
                        'COMMA': ',',
                        'QUESTION': '?',
                        'EXCLAMATION': '!'
                    }
                    
                    punctuation = punctuation_map.get(result['entity_group'], '.')
                    position = result.get('end', 0)  # Position in text
                    confidence = result.get('score', 0.5)
                    
                    candidates.append(PunctuationCandidate(
                        position=position,
                        punctuation=punctuation,
                        confidence=confidence,
                        context_score=0.8,  # Default context score
                        meeting_relevance_score=0.8,  # Default meeting relevance
                        source="transformer"
                    ))
            
            return candidates
            
        except Exception as e:
            self.logger.warning(f"Transformer prediction failed: {e}, falling back to rules")
            return self._rule_based_punctuation(text)
    
    def _rule_based_punctuation(self, text: str) -> List[PunctuationCandidate]:
        """Fallback rule-based punctuation"""
        candidates = []
        
        # Simple sentence boundary detection
        sentences = re.split(r'(?<=[.!?])\s+', text)
        position = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Add period at end of declarative sentences
            if i < len(sentences) - 1 or not text.endswith(('.', '!', '?')):
                candidates.append(PunctuationCandidate(
                    position=position + sentence_length,
                    punctuation='.',
                    confidence=0.7,
                    context_score=0.6,
                    meeting_relevance_score=0.6,
                    source="rule_based"
                ))
            
            # Add commas for basic pauses (simple heuristic)
            comma_positions = [m.start() for m in re.finditer(r'\s+(and|but|or|however|therefore)\s+', sentence)]
            for comma_pos in comma_positions:
                candidates.append(PunctuationCandidate(
                    position=position + comma_pos,
                    punctuation=',',
                    confidence=0.6,
                    context_score=0.5,
                    meeting_relevance_score=0.5,
                    source="rule_based"
                ))
            
            position += sentence_length + 1  # +1 for space
        
        return candidates

class PostFusionPunctuationEngine:
    """
    Main engine for post-fusion punctuation restoration and disfluency normalization
    """
    
    def __init__(self, 
                 punctuation_model: str = "oliverguhr/fullstop-punctuation-multilang-large",
                 disfluency_level: str = "light",
                 enable_meeting_vocabulary: bool = True,
                 enable_capitalization: bool = True,
                 preserve_speaker_boundaries: bool = True):
        """
        Initialize Post-Fusion Punctuation Engine
        
        Args:
            punctuation_model: Transformer model for punctuation restoration
            disfluency_level: Level of disfluency normalization ("light", "moderate", "aggressive")
            enable_meeting_vocabulary: Enable meeting-specific vocabulary handling
            enable_capitalization: Enable intelligent capitalization
            preserve_speaker_boundaries: Preserve speaker timing and boundaries
        """
        self.logger = create_enhanced_logger("post_fusion_punctuation")
        
        # Configuration
        self.disfluency_level = disfluency_level
        self.enable_meeting_vocabulary = enable_meeting_vocabulary
        self.enable_capitalization = enable_capitalization
        self.preserve_speaker_boundaries = preserve_speaker_boundaries
        
        # Initialize components
        self.logger.info("Initializing Post-Fusion Punctuation Engine")
        
        # Punctuation model
        self.punctuation_model = TransformerPunctuationModel(punctuation_model)
        
        # Disfluency normalizer
        self.disfluency_normalizer = DisfluencyNormalizer(disfluency_level)
        
        # Meeting vocabulary handler
        if enable_meeting_vocabulary:
            self.vocabulary_handler = MeetingVocabularyHandler()
        else:
            self.vocabulary_handler = None
        
        # Metrics tracking
        self.processing_metrics = {
            'segments_processed': 0,
            'punctuation_changes': 0,
            'disfluency_normalizations': 0,
            'capitalization_fixes': 0,
            'processing_time': 0.0
        }
        
        self.logger.info(f"Punctuation engine initialized - Model available: {self.punctuation_model.is_available()}")
    
    def process_fused_segments(self, segments: List[Dict[str, Any]]) -> PunctuationResult:
        """
        Process fused segments to add punctuation and normalize disfluencies
        
        Args:
            segments: List of fused segments from consensus module
            
        Returns:
            PunctuationResult with punctuated and normalized segments
        """
        start_time = time.time()
        
        self.logger.info(f"Processing {len(segments)} fused segments for punctuation and normalization")
        
        punctuated_segments = []
        overall_confidences = []
        punctuation_metrics = defaultdict(int)
        disfluency_metrics = defaultdict(int)
        
        for segment in segments:
            try:
                # Extract segment information
                start_time_seg = segment.get('start', 0.0)
                end_time_seg = segment.get('end', 0.0) 
                speaker_id = segment.get('speaker', 'Unknown')
                text = segment.get('text', '').strip()
                
                if not text:
                    continue
                
                # Step 1: Detect formality level if vocabulary handler available
                formality_level = "neutral"
                if self.vocabulary_handler:
                    formality_level, _ = self.vocabulary_handler.detect_formality_level(text)
                
                # Step 2: Apply disfluency normalization
                normalized_segment = self.disfluency_normalizer.normalize_segment(text, formality_level)
                working_text = normalized_segment.normalized_text
                
                # Update metrics
                if normalized_segment.normalization_applied:
                    disfluency_metrics['segments_normalized'] += 1
                    for norm_type in normalized_segment.normalization_applied:
                        disfluency_metrics[f'normalization_{norm_type}'] += 1
                
                # Step 3: Apply punctuation restoration
                punctuation_candidates = self.punctuation_model.predict_punctuation(working_text)
                punctuated_text = self._apply_punctuation(working_text, punctuation_candidates)
                
                # Step 4: Apply capitalization if enabled
                if self.enable_capitalization:
                    punctuated_text = self._apply_capitalization(punctuated_text, formality_level)
                
                # Step 5: Handle technical terms if vocabulary handler available
                if self.vocabulary_handler:
                    technical_corrections = self.vocabulary_handler.identify_technical_terms(punctuated_text)
                    punctuated_text = self._apply_technical_corrections(punctuated_text, technical_corrections)
                
                # Calculate segment confidence
                punctuation_confidence = self._calculate_punctuation_confidence(punctuation_candidates)
                overall_confidences.append(punctuation_confidence)
                
                # Create punctuated segment
                punctuated_segment = PunctuatedSegment(
                    start_time=start_time_seg,
                    end_time=end_time_seg,
                    speaker_id=speaker_id,
                    original_text=text,
                    punctuated_text=punctuated_text,
                    punctuation_confidence=punctuation_confidence,
                    word_level_punctuation=self._generate_word_level_punctuation(punctuated_text, punctuation_candidates),
                    disfluency_normalization=normalized_segment,
                    processing_metadata={
                        'formality_level': formality_level,
                        'punctuation_source': 'transformer' if self.punctuation_model.is_available() else 'rule_based',
                        'normalization_level': self.disfluency_level
                    }
                )
                
                punctuated_segments.append(punctuated_segment)
                
                # Update punctuation metrics
                punctuation_metrics['segments_punctuated'] += 1
                if text != punctuated_text:
                    punctuation_metrics['segments_changed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing segment: {e}")
                # Add original segment on error
                punctuated_segments.append(PunctuatedSegment(
                    start_time=segment.get('start', 0.0),
                    end_time=segment.get('end', 0.0),
                    speaker_id=segment.get('speaker', 'Unknown'),
                    original_text=segment.get('text', ''),
                    punctuated_text=segment.get('text', ''),
                    punctuation_confidence=0.0,
                    word_level_punctuation=[],
                    processing_metadata={'error': str(e)}
                ))
        
        processing_time = time.time() - start_time
        overall_confidence = float(np.mean(overall_confidences)) if overall_confidences else 0.0
        
        # Create result
        result = PunctuationResult(
            segments=punctuated_segments,
            overall_confidence=overall_confidence,
            punctuation_metrics=dict(punctuation_metrics),
            disfluency_metrics=dict(disfluency_metrics),
            processing_time=processing_time,
            model_info={
                'punctuation_model': self.punctuation_model.model_name,
                'model_available': str(self.punctuation_model.is_available()),
                'normalization_level': self.disfluency_level
            },
            normalization_level=self.disfluency_level
        )
        
        self.logger.info(f"Completed punctuation processing - {len(punctuated_segments)} segments, {processing_time:.2f}s, confidence: {overall_confidence:.3f}")
        
        return result
    
    def _apply_punctuation(self, text: str, candidates: List[PunctuationCandidate]) -> str:
        """Apply punctuation candidates to text"""
        if not candidates:
            return text + '.'  # Add basic sentence ending
        
        # Sort candidates by position
        candidates.sort(key=lambda x: x.position)
        
        # Apply punctuation from end to beginning to preserve positions
        result_text = text
        for candidate in reversed(candidates):
            if candidate.confidence > 0.5:  # Confidence threshold
                # Insert punctuation at position
                pos = min(candidate.position, len(result_text))
                result_text = result_text[:pos] + candidate.punctuation + result_text[pos:]
        
        return result_text
    
    def _apply_capitalization(self, text: str, formality_level: str) -> str:
        """Apply intelligent capitalization"""
        # Basic sentence-level capitalization
        sentences = re.split(r'([.!?]+)', text)
        capitalized_sentences = []
        
        for sentence in sentences:
            if sentence.strip() and not re.match(r'[.!?]+', sentence):
                # Capitalize first word of sentence
                words = sentence.split()
                if words:
                    words[0] = words[0].capitalize()
                    sentence = ' '.join(words)
            
            capitalized_sentences.append(sentence)
        
        return ''.join(capitalized_sentences)
    
    def _apply_technical_corrections(self, text: str, corrections: List[Tuple[int, str]]) -> str:
        """Apply technical term corrections"""
        words = text.split()
        
        for position, correct_form in corrections:
            if 0 <= position < len(words):
                words[position] = correct_form
        
        return ' '.join(words)
    
    def _calculate_punctuation_confidence(self, candidates: List[PunctuationCandidate]) -> float:
        """Calculate overall punctuation confidence for segment"""
        if not candidates:
            return 0.5  # Default confidence for no punctuation
        
        confidences = [c.confidence for c in candidates]
        return float(np.mean(confidences))
    
    def _generate_word_level_punctuation(self, text: str, candidates: List[PunctuationCandidate]) -> List[Dict[str, Any]]:
        """Generate word-level punctuation information"""
        words = text.split()
        word_punctuation = []
        
        for i, word in enumerate(words):
            # Check if word has punctuation
            punctuation_chars = re.findall(r'[.!?,:;]', word)
            
            word_info = {
                'word': word,
                'position': i,
                'has_punctuation': bool(punctuation_chars),
                'punctuation': punctuation_chars,
                'confidence': 1.0 if punctuation_chars else 0.8
            }
            
            word_punctuation.append(word_info)
        
        return word_punctuation
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status and configuration"""
        return {
            'punctuation_model_available': self.punctuation_model.is_available(),
            'punctuation_model_name': self.punctuation_model.model_name,
            'disfluency_level': self.disfluency_level,
            'meeting_vocabulary_enabled': self.enable_meeting_vocabulary,
            'capitalization_enabled': self.enable_capitalization,
            'speaker_boundaries_preserved': self.preserve_speaker_boundaries,
            'processing_metrics': self.processing_metrics.copy()
        }
    
    def update_configuration(self, config: Dict[str, Any]) -> None:
        """Update engine configuration"""
        if 'disfluency_level' in config:
            self.disfluency_level = config['disfluency_level']
            self.disfluency_normalizer = DisfluencyNormalizer(self.disfluency_level)
        
        if 'enable_meeting_vocabulary' in config:
            self.enable_meeting_vocabulary = config['enable_meeting_vocabulary']
            if self.enable_meeting_vocabulary and not self.vocabulary_handler:
                self.vocabulary_handler = MeetingVocabularyHandler()
        
        if 'enable_capitalization' in config:
            self.enable_capitalization = config['enable_capitalization']
        
        if 'preserve_speaker_boundaries' in config:
            self.preserve_speaker_boundaries = config['preserve_speaker_boundaries']
        
        self.logger.info(f"Configuration updated: {config}")

# Factory function for easy instantiation
def create_punctuation_engine(
    punctuation_model: str = "oliverguhr/fullstop-punctuation-multilang-large",
    disfluency_level: str = "light",
    enable_meeting_vocabulary: bool = True,
    **kwargs
) -> PostFusionPunctuationEngine:
    """
    Create and configure a punctuation engine
    
    Args:
        punctuation_model: Transformer model name
        disfluency_level: Normalization level
        enable_meeting_vocabulary: Enable meeting vocabulary handling
        **kwargs: Additional configuration options
        
    Returns:
        Configured PostFusionPunctuationEngine
    """
    return PostFusionPunctuationEngine(
        punctuation_model=punctuation_model,
        disfluency_level=disfluency_level,
        enable_meeting_vocabulary=enable_meeting_vocabulary,
        **kwargs
    )

# Configuration presets
PUNCTUATION_PRESETS = {
    "meeting_light": {
        "punctuation_model": "oliverguhr/fullstop-punctuation-multilang-large",
        "disfluency_level": "light",
        "enable_meeting_vocabulary": True,
        "enable_capitalization": True,
        "preserve_speaker_boundaries": True
    },
    
    "meeting_moderate": {
        "punctuation_model": "oliverguhr/fullstop-punctuation-multilang-large", 
        "disfluency_level": "moderate",
        "enable_meeting_vocabulary": True,
        "enable_capitalization": True,
        "preserve_speaker_boundaries": True
    },
    
    "meeting_aggressive": {
        "punctuation_model": "oliverguhr/fullstop-punctuation-multilang-large",
        "disfluency_level": "aggressive", 
        "enable_meeting_vocabulary": True,
        "enable_capitalization": True,
        "preserve_speaker_boundaries": True
    },
    
    "general_light": {
        "punctuation_model": "oliverguhr/fullstop-punctuation-multilang-large",
        "disfluency_level": "light",
        "enable_meeting_vocabulary": False,
        "enable_capitalization": True,
        "preserve_speaker_boundaries": True
    }
}

def create_punctuation_engine_from_preset(preset_name: str) -> PostFusionPunctuationEngine:
    """Create punctuation engine from preset configuration"""
    if preset_name not in PUNCTUATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PUNCTUATION_PRESETS.keys())}")
    
    config = PUNCTUATION_PRESETS[preset_name]
    return create_punctuation_engine(**config)