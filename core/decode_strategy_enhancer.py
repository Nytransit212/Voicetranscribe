import os
import re
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
from openai import OpenAI
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import get_observability_manager, trace_stage, track_cost
from utils.reliability_config import get_concurrency_config, get_timeout_config
from utils.resilient_api import openai_retry
from core.circuit_breaker import CircuitBreakerOpenException


@dataclass
class DecodeStrategy:
    """Configuration for a specific decode strategy"""
    strategy_id: str
    temperature: float
    language: Optional[str]
    prompt: str
    response_format: str
    vocabulary_priming: List[str]
    decode_parameters: Dict[str, Any]
    correlation_target: str  # 'diverse', 'conservative', 'exploratory'


@dataclass
class DomainLexicon:
    """Domain-specific vocabulary extracted from previous segments"""
    domain_terms: Set[str]
    technical_terms: Set[str]
    proper_nouns: Set[str]
    recurring_phrases: List[str]
    confidence_terms: Dict[str, float]
    extraction_metadata: Dict[str, Any]


@dataclass
class DecodeResult:
    """Result from a decode strategy execution"""
    strategy_id: str
    transcript_text: str
    words: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]
    language: str
    confidence_scores: Dict[str, float]
    decode_parameters: Dict[str, Any]
    lexicon_usage: Dict[str, Any]
    diversity_metrics: Dict[str, float]


class DecodeStrategyEnhancer:
    """Enhanced decode strategy with domain lexicon extraction and context-aware prompting"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "whisper-1"
        
        # Load reliability configuration
        self.concurrency_config = get_concurrency_config()
        self.timeout_config = get_timeout_config()
        
        # Initialize enhanced observability
        self.obs_manager = get_observability_manager()
        self.structured_logger = create_enhanced_logger("decode_strategy_enhancer")
        
        # Domain lexicon tracking
        self.session_lexicon: Optional[DomainLexicon] = None
        self.previous_segments: List[Dict[str, Any]] = []
        
        # Decode strategy configurations
        self.base_strategies = self._initialize_base_strategies()
        
        # Configure logging for detailed tracking
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_base_strategies(self) -> List[DecodeStrategy]:
        """Initialize base decode strategies with different correlation targets"""
        
        strategies = [
            # Conservative strategy: Low temperature, standard prompting
            DecodeStrategy(
                strategy_id="conservative_standard",
                temperature=0.0,
                language=None,  # Auto-detect
                prompt="",
                response_format="verbose_json",
                vocabulary_priming=[],
                decode_parameters={
                    "temperature": 0.0,
                    "focus": "accuracy",
                    "style": "conservative"
                },
                correlation_target="conservative"
            ),
            
            # Exploratory strategy: Higher temperature, creative prompting
            DecodeStrategy(
                strategy_id="exploratory_creative",
                temperature=0.3,
                language=None,
                prompt="This is a dynamic conversation with multiple speakers discussing various topics.",
                response_format="verbose_json",
                vocabulary_priming=[],
                decode_parameters={
                    "temperature": 0.3,
                    "focus": "creativity",
                    "style": "exploratory"
                },
                correlation_target="exploratory"
            ),
            
            # Diverse strategy: Medium temperature, context-aware
            DecodeStrategy(
                strategy_id="diverse_contextual",
                temperature=0.15,
                language=None,
                prompt="This recording contains domain-specific terminology and technical discussions.",
                response_format="verbose_json",
                vocabulary_priming=[],
                decode_parameters={
                    "temperature": 0.15,
                    "focus": "diversity",
                    "style": "contextual"
                },
                correlation_target="diverse"
            ),
            
            # Domain-focused strategy: Low-medium temperature, vocabulary priming
            DecodeStrategy(
                strategy_id="domain_focused",
                temperature=0.1,
                language=None,
                prompt="",  # Will be enhanced with domain context
                response_format="verbose_json",
                vocabulary_priming=[],  # Will be populated with domain terms
                decode_parameters={
                    "temperature": 0.1,
                    "focus": "domain_accuracy",
                    "style": "domain_focused"
                },
                correlation_target="diverse"
            ),
            
            # Language-specific strategy: Auto-adapt based on detected language
            DecodeStrategy(
                strategy_id="language_adaptive",
                temperature=0.05,
                language=None,  # Will be set based on previous detections
                prompt="",  # Will be enhanced with language-specific context
                response_format="verbose_json",
                vocabulary_priming=[],
                decode_parameters={
                    "temperature": 0.05,
                    "focus": "language_accuracy",
                    "style": "adaptive"
                },
                correlation_target="conservative"
            )
        ]
        
        return strategies
    
    @trace_stage("enhanced_decode_processing")
    def process_audio_with_enhanced_decode(self, audio_path: str, diarization_data: Dict[str, Any], 
                                         asr_variant_id: int, target_language: Optional[str] = None) -> List[DecodeResult]:
        """
        Process audio with enhanced decode strategies including domain lexicon and context awareness.
        
        Args:
            audio_path: Path to audio file
            diarization_data: Diarization results for context
            asr_variant_id: ID of the ASR variant being processed
            target_language: Optional target language override
            
        Returns:
            List of decode results from different strategies
        """
        self.structured_logger.stage_start("enhanced_decode", 
                                         f"Starting enhanced decode processing for ASR variant {asr_variant_id}",
                                         context={'asr_variant_id': asr_variant_id, 'target_language': target_language})
        
        print(f"🎯 Enhanced decode processing for ASR variant {asr_variant_id}...")
        
        # Step 1: Update domain lexicon from previous segments
        start_time = time.time()
        self._update_domain_lexicon()
        
        # Step 2: Enhance strategies with current context
        enhanced_strategies = self._enhance_strategies_with_context(diarization_data, target_language)
        
        # Step 3: Execute decode strategies in parallel
        decode_results = self._execute_decode_strategies(audio_path, enhanced_strategies)
        
        # Step 4: Calculate diversity metrics
        self._calculate_decode_diversity_metrics(decode_results)
        
        # Step 5: Update session context with results
        self._update_session_context_from_results(decode_results)
        
        total_time = time.time() - start_time
        self.structured_logger.stage_complete("enhanced_decode", 
                                            f"Enhanced decode completed: {len(decode_results)} strategies executed",
                                            duration=total_time,
                                            metrics={
                                                'strategies_executed': len(decode_results),
                                                'lexicon_terms': len(self.session_lexicon.domain_terms) if self.session_lexicon else 0,
                                                'diversity_score': self._calculate_overall_diversity(decode_results)
                                            })
        
        print(f"✓ Enhanced decode complete: {len(decode_results)} strategies executed")
        return decode_results
    
    def _update_domain_lexicon(self) -> None:
        """Update domain lexicon from previous segments using NLP techniques"""
        
        if not self.previous_segments:
            # Initialize empty lexicon
            self.session_lexicon = DomainLexicon(
                domain_terms=set(),
                technical_terms=set(),
                proper_nouns=set(),
                recurring_phrases=[],
                confidence_terms={},
                extraction_metadata={
                    'segments_analyzed': 0,
                    'extraction_time': time.time(),
                    'extraction_method': 'rule_based_nlp'
                }
            )
            return
        
        self.structured_logger.info("Updating domain lexicon from previous segments", 
                                   segments_count=len(self.previous_segments))
        
        # Combine all text from previous segments
        all_text = " ".join([seg.get('text', '') for seg in self.previous_segments])
        
        # Extract domain-specific terms using multiple techniques
        domain_terms = self._extract_domain_terms(all_text)
        technical_terms = self._extract_technical_terms(all_text)
        proper_nouns = self._extract_proper_nouns(all_text)
        recurring_phrases = self._extract_recurring_phrases(all_text)
        confidence_terms = self._calculate_term_confidence(all_text)
        
        # Update session lexicon
        self.session_lexicon = DomainLexicon(
            domain_terms=domain_terms,
            technical_terms=technical_terms,
            proper_nouns=proper_nouns,
            recurring_phrases=recurring_phrases,
            confidence_terms=confidence_terms,
            extraction_metadata={
                'segments_analyzed': len(self.previous_segments),
                'extraction_time': time.time(),
                'total_text_length': len(all_text),
                'extraction_method': 'enhanced_nlp_pipeline'
            }
        )
        
        self.structured_logger.info("Domain lexicon updated", 
                                   domain_terms=len(domain_terms),
                                   technical_terms=len(technical_terms),
                                   proper_nouns=len(proper_nouns),
                                   recurring_phrases=len(recurring_phrases))
    
    def _extract_domain_terms(self, text: str) -> Set[str]:
        """Extract domain-specific terms using frequency and pattern analysis"""
        
        # Clean and tokenize text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = Counter(words)
        
        # Identify domain terms by frequency and characteristics
        domain_terms = set()
        
        # High-frequency terms (but not common English words)
        common_words = {'the', 'and', 'or', 'but', 'for', 'with', 'this', 'that', 'from', 'they', 'have', 'been', 'will', 'can', 'could', 'would', 'should', 'may', 'might', 'must', 'shall', 'was', 'were', 'are', 'been', 'being', 'had', 'has', 'having', 'did', 'does', 'done', 'doing', 'get', 'got', 'getting', 'put', 'putting', 'take', 'taking', 'took', 'taken', 'make', 'making', 'made', 'come', 'coming', 'came', 'go', 'going', 'went', 'gone', 'see', 'seeing', 'saw', 'seen', 'know', 'knowing', 'knew', 'known', 'think', 'thinking', 'thought', 'say', 'saying', 'said', 'tell', 'telling', 'told', 'ask', 'asking', 'asked', 'use', 'using', 'used', 'work', 'working', 'worked', 'find', 'finding', 'found', 'give', 'giving', 'gave', 'given', 'feel', 'feeling', 'felt', 'seem', 'seeming', 'seemed', 'look', 'looking', 'looked', 'want', 'wanting', 'wanted', 'need', 'needing', 'needed', 'try', 'trying', 'tried', 'help', 'helping', 'helped', 'keep', 'keeping', 'kept', 'start', 'starting', 'started', 'stop', 'stopping', 'stopped', 'turn', 'turning', 'turned', 'move', 'moving', 'moved', 'play', 'playing', 'played', 'run', 'running', 'ran', 'walk', 'walking', 'walked', 'sit', 'sitting', 'sat', 'stand', 'standing', 'stood', 'hear', 'hearing', 'heard', 'speak', 'speaking', 'spoke', 'spoken', 'read', 'reading', 'write', 'writing', 'wrote', 'written', 'learn', 'learning', 'learned', 'teach', 'teaching', 'taught', 'show', 'showing', 'showed', 'shown', 'mean', 'meaning', 'meant', 'understand', 'understanding', 'understood', 'remember', 'remembering', 'remembered', 'forget', 'forgetting', 'forgot', 'forgotten', 'believe', 'believing', 'believed', 'hope', 'hoping', 'hoped', 'expect', 'expecting', 'expected', 'become', 'becoming', 'became', 'happen', 'happening', 'happened', 'change', 'changing', 'changed', 'end', 'ending', 'ended', 'begin', 'beginning', 'began', 'begun', 'follow', 'following', 'followed', 'stay', 'staying', 'stayed', 'leave', 'leaving', 'left', 'meet', 'meeting', 'met', 'bring', 'bringing', 'brought', 'build', 'building', 'built', 'break', 'breaking', 'broke', 'broken', 'buy', 'buying', 'bought', 'sell', 'selling', 'sold', 'pay', 'paying', 'paid', 'spend', 'spending', 'spent', 'cost', 'costing', 'cut', 'cutting', 'open', 'opening', 'opened', 'close', 'closing', 'closed', 'win', 'winning', 'won', 'lose', 'losing', 'lost', 'send', 'sending', 'sent', 'receive', 'receiving', 'received', 'call', 'calling', 'called', 'reach', 'reaching', 'reached', 'pass', 'passing', 'passed', 'carry', 'carrying', 'carried', 'hold', 'holding', 'held', 'catch', 'catching', 'caught', 'throw', 'throwing', 'threw', 'thrown', 'hit', 'hitting', 'fall', 'falling', 'fell', 'fallen', 'rise', 'rising', 'rose', 'risen', 'set', 'setting', 'lay', 'laying', 'laid', 'lie', 'lying', 'lied'}
        
        for word, freq in word_freq.items():
            if freq > 2 and word not in common_words and len(word) > 4:
                domain_terms.add(word)
        
        # Identify technical patterns (camelCase, snake_case, hyphenated)
        technical_patterns = re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z]+\b|\b[a-z]+-[a-z]+\b', text)
        domain_terms.update(technical_patterns)
        
        # Limit to most relevant terms
        return set(list(domain_terms)[:50])
    
    def _extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms using pattern recognition"""
        
        technical_terms = set()
        
        # Technical acronyms (2-6 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        technical_terms.update(acronyms)
        
        # Technical compound words
        compounds = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
        technical_terms.update(compounds)
        
        # Technical terms with numbers
        tech_numbers = re.findall(r'\b[a-zA-Z]+\d+[a-zA-Z]*\b|\b\d+[a-zA-Z]+\b', text)
        technical_terms.update(tech_numbers)
        
        # Common technical suffixes
        tech_suffixes = re.findall(r'\b\w*(?:tion|sion|ment|ness|ity|ing|ful|less|able|ible)\b', text.lower())
        technical_terms.update([term for term in tech_suffixes if len(term) > 6])
        
        return technical_terms
    
    def _extract_proper_nouns(self, text: str) -> Set[str]:
        """Extract proper nouns using capitalization patterns"""
        
        # Simple proper noun detection (capitalized words not at sentence start)
        sentences = re.split(r'[.!?]+', text)
        proper_nouns = set()
        
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) > 1:  # Skip first word of sentence
                for word in words[1:]:
                    # Check if word is capitalized and alphabetic
                    if word and word[0].isupper() and word.isalpha() and len(word) > 2:
                        proper_nouns.add(word)
        
        # Filter out common capitalized words
        common_caps = {'I', 'The', 'This', 'That', 'And', 'Or', 'But', 'So', 'Yet', 'For'}
        proper_nouns = {noun for noun in proper_nouns if noun not in common_caps}
        
        return proper_nouns
    
    def _extract_recurring_phrases(self, text: str) -> List[str]:
        """Extract recurring phrases and collocations"""
        
        # Extract 2-4 word phrases
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        phrases = []
        
        # 2-word phrases
        two_grams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        phrase_freq = Counter(two_grams)
        phrases.extend([phrase for phrase, freq in phrase_freq.items() if freq > 2])
        
        # 3-word phrases
        three_grams = []
        if len(words) >= 3:
            three_grams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
            phrase_freq = Counter(three_grams)
            phrases.extend([phrase for phrase, freq in phrase_freq.items() if freq > 1])
        
        # Return most frequent phrases
        all_grams = two_grams + three_grams
        return sorted(set(phrases), key=lambda x: Counter(all_grams)[x], reverse=True)[:20]
    
    def _calculate_term_confidence(self, text: str) -> Dict[str, float]:
        """Calculate confidence scores for extracted terms"""
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        word_freq = Counter(words)
        total_words = len(words)
        
        confidence_terms = {}
        
        for word, freq in word_freq.items():
            if freq > 1 and len(word) > 3:
                # Confidence based on frequency and word characteristics
                freq_score = min(freq / total_words * 100, 1.0)  # Frequency-based score
                length_score = min(len(word) / 10.0, 1.0)  # Length-based score (longer = more specific)
                
                # Combined confidence score
                confidence = (freq_score * 0.7) + (length_score * 0.3)
                confidence_terms[word] = confidence
        
        return confidence_terms
    
    def _enhance_strategies_with_context(self, diarization_data: Dict[str, Any], 
                                       target_language: Optional[str]) -> List[DecodeStrategy]:
        """Enhance base strategies with current context and domain lexicon"""
        
        enhanced_strategies = []
        
        for strategy in self.base_strategies:
            # Create enhanced copy
            enhanced_strategy = DecodeStrategy(
                strategy_id=strategy.strategy_id,
                temperature=strategy.temperature,
                language=target_language or strategy.language,
                prompt=strategy.prompt,
                response_format=strategy.response_format,
                vocabulary_priming=strategy.vocabulary_priming.copy(),
                decode_parameters=strategy.decode_parameters.copy(),
                correlation_target=strategy.correlation_target
            )
            
            # Enhance with domain context
            if self.session_lexicon and strategy.strategy_id == "domain_focused":
                # Add domain vocabulary to priming
                confidence_terms = self.session_lexicon.confidence_terms or {}
                top_domain_terms = sorted(
                    self.session_lexicon.domain_terms, 
                    key=lambda x: confidence_terms.get(x, 0.0), 
                    reverse=True
                )[:10]
                enhanced_strategy.vocabulary_priming.extend(top_domain_terms)
                
                # Enhance prompt with domain context
                if top_domain_terms:
                    domain_context = ", ".join(top_domain_terms[:5])
                    enhanced_strategy.prompt = f"This recording discusses topics related to: {domain_context}. Listen for technical terminology and domain-specific language."
            
            # Enhance language-adaptive strategy
            if strategy.strategy_id == "language_adaptive" and self.previous_segments:
                # Detect most common language from previous segments
                languages = [seg.get('language', 'en') for seg in self.previous_segments if seg.get('language')]
                if languages:
                    most_common_lang = Counter(languages).most_common(1)[0][0]
                    enhanced_strategy.language = most_common_lang
                    enhanced_strategy.prompt = f"This is a recording in {most_common_lang}. Focus on accurate transcription for this language."
            
            # Add speaker context from diarization
            if diarization_data.get('segments'):
                speaker_count = len(set(seg.get('speaker_id', 'unknown') for seg in diarization_data['segments']))
                if strategy.strategy_id == "diverse_contextual":
                    enhanced_strategy.prompt += f" This recording has approximately {speaker_count} speakers with varied speaking styles."
            
            # Add correlation diversification parameters
            enhanced_strategy.decode_parameters.update({
                'lexicon_terms_used': len(enhanced_strategy.vocabulary_priming),
                'prompt_enhancement': bool(enhanced_strategy.prompt != strategy.prompt),
                'context_segments': len(self.previous_segments),
                'target_correlation': strategy.correlation_target
            })
            
            enhanced_strategies.append(enhanced_strategy)
        
        return enhanced_strategies
    
    def _execute_decode_strategies(self, audio_path: str, strategies: List[DecodeStrategy]) -> List[DecodeResult]:
        """Execute decode strategies in parallel with rate limiting"""
        
        print(f"  Executing {len(strategies)} enhanced decode strategies...")
        
        # Use bounded thread pool for controlled concurrency
        from utils.bounded_executor import get_asr_executor
        
        decode_results = []
        executor = get_asr_executor()
        
        try:
            # Submit all decode tasks
            future_to_strategy = {}
            for strategy in strategies:
                future = executor.submit_with_backpressure(
                    self._execute_single_decode_strategy,
                    audio_path,
                    strategy,
                    max_wait=3.0
                )
                if future:
                    future_to_strategy[future] = strategy
                else:
                    print(f"  ⚠ Decode strategy {strategy.strategy_id} rejected due to queue backpressure")
            
            # Collect results as they complete
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result(timeout=self.timeout_config.asr_variant)
                    if result:
                        decode_results.append(result)
                        print(f"  ✓ Decode strategy {strategy.strategy_id} completed")
                        
                except Exception as e:
                    print(f"  ⚠ Decode strategy {strategy.strategy_id} failed: {e}")
                    continue
        
        except Exception as e:
            self.structured_logger.error(f"Error in decode strategy execution: {e}")
            raise
        
        return decode_results
    
    @openai_retry
    @trace_stage("single_decode_strategy")
    def _execute_single_decode_strategy(self, audio_path: str, strategy: DecodeStrategy) -> DecodeResult:
        """Execute a single decode strategy with OpenAI Whisper"""
        
        try:
            # Prepare transcription parameters
            transcription_params = {
                'model': self.model,
                'response_format': strategy.response_format,
                'temperature': strategy.temperature
            }
            
            # Add optional parameters
            if strategy.language:
                transcription_params['language'] = strategy.language
            
            # Build enhanced prompt
            prompt_parts = []
            if strategy.prompt:
                prompt_parts.append(strategy.prompt)
            
            # Add vocabulary priming to prompt
            if strategy.vocabulary_priming:
                vocab_context = " ".join(strategy.vocabulary_priming[:8])  # Limit to avoid token limits
                prompt_parts.append(f"Key terms: {vocab_context}")
            
            if prompt_parts:
                transcription_params['prompt'] = " ".join(prompt_parts)
            
            # Execute transcription
            start_time = time.time()
            
            try:
                with open(audio_path, 'rb') as audio_file:
                    response = self.client.audio.transcriptions.create(
                        file=audio_file,
                        timeout=self.timeout_config.api_request,
                        **transcription_params
                    )
                
            except CircuitBreakerOpenException as e:
                self.structured_logger.error(
                    f"Circuit breaker open during decode strategy {strategy.strategy_id}",
                    context={'strategy_id': strategy.strategy_id, 'error': str(e)}
                )
                raise Exception(f"OpenAI service temporarily unavailable: {str(e)}")
            
            api_duration = time.time() - start_time
            
            # Extract word-level timestamps
            words = []
            if hasattr(response, 'words') and response.words:
                words = [
                    {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'confidence': getattr(word, 'confidence', 0.9)
                    }
                    for word in response.words
                ]
            else:
                # Fallback: create mock word timestamps
                words = self._create_mock_word_timestamps(response)
            
            # Extract segments
            segments = []
            if hasattr(response, 'segments') and response.segments:
                segments = [
                    {
                        'id': seg.id,
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text,
                        'confidence': getattr(seg, 'avg_logprob', 0.0)
                    }
                    for seg in response.segments
                ]
            
            # Calculate lexicon usage metrics
            lexicon_usage = self._calculate_lexicon_usage(response.text, strategy.vocabulary_priming)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_decode_confidence_scores(words, segments, api_duration)
            
            return DecodeResult(
                strategy_id=strategy.strategy_id,
                transcript_text=response.text,
                words=words,
                segments=segments,
                language=getattr(response, 'language', 'en'),
                confidence_scores=confidence_scores,
                decode_parameters=strategy.decode_parameters,
                lexicon_usage=lexicon_usage,
                diversity_metrics={}  # Will be calculated later
            )
            
        except Exception as e:
            raise Exception(f"Decode strategy {strategy.strategy_id} failed: {str(e)}")
    
    def _create_mock_word_timestamps(self, response) -> List[Dict[str, Any]]:
        """Create mock word timestamps when not available from API"""
        
        words = []
        
        if hasattr(response, 'segments') and response.segments:
            for segment in response.segments:
                segment_words = segment.text.split()
                segment_duration = segment.end - segment.start
                word_duration = segment_duration / max(len(segment_words), 1)
                
                for i, word in enumerate(segment_words):
                    word_start = segment.start + (i * word_duration)
                    word_end = word_start + word_duration
                    
                    words.append({
                        'word': word,
                        'start': word_start,
                        'end': word_end,
                        'confidence': max(0.1, getattr(segment, 'avg_logprob', -1.0) + 1.0)
                    })
        
        return words
    
    def _calculate_lexicon_usage(self, transcript_text: str, vocabulary_priming: List[str]) -> Dict[str, Any]:
        """Calculate how well the vocabulary priming was utilized"""
        
        if not vocabulary_priming:
            return {
                'priming_terms_used': 0,
                'priming_coverage': 0.0,
                'novel_terms_found': 0,
                'lexicon_effectiveness': 0.0
            }
        
        transcript_lower = transcript_text.lower()
        
        # Count vocabulary terms that appeared in transcript
        used_terms = [term for term in vocabulary_priming if term.lower() in transcript_lower]
        
        # Find novel terms in transcript
        transcript_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', transcript_lower))
        priming_words = set(term.lower() for term in vocabulary_priming)
        novel_terms = transcript_words - priming_words
        
        # Calculate effectiveness metrics
        priming_coverage = len(used_terms) / len(vocabulary_priming) if vocabulary_priming else 0.0
        lexicon_effectiveness = (len(used_terms) * 2 + len(novel_terms)) / float(max(len(transcript_words), 1))
        
        return {
            'priming_terms_used': len(used_terms),
            'priming_coverage': priming_coverage,
            'novel_terms_found': len(novel_terms),
            'lexicon_effectiveness': lexicon_effectiveness,
            'used_terms': used_terms,
            'novel_terms': list(novel_terms)[:10]  # Sample of novel terms
        }
    
    def _calculate_decode_confidence_scores(self, words: List[Dict[str, Any]], 
                                          segments: List[Dict[str, Any]], 
                                          api_duration: float) -> Dict[str, float]:
        """Calculate confidence scores for decode result"""
        
        if not words:
            return {
                'word_confidence_mean': 0.0,
                'segment_confidence_mean': 0.0,
                'temporal_consistency': 0.0,
                'decode_efficiency': 0.0
            }
        
        # Word-level confidence
        word_confidences = [w['confidence'] for w in words if 'confidence' in w]
        word_confidence_mean = np.mean(word_confidences) if word_confidences else 0.0
        
        # Segment-level confidence
        segment_confidences = [s['confidence'] for s in segments if 'confidence' in s]
        segment_confidence_mean = np.mean(segment_confidences) if segment_confidences else 0.0
        
        # Temporal consistency (check for timestamp ordering)
        temporal_consistency = 1.0
        if len(words) > 1:
            timestamp_errors = sum(1 for i in range(1, len(words)) 
                                 if words[i]['start'] < words[i-1]['end'])
            temporal_consistency = 1.0 - (timestamp_errors / len(words))
        
        # Decode efficiency (faster = potentially less thorough)
        target_duration = len(words) * 0.1  # Rough estimate
        decode_efficiency = float(min(target_duration / max(api_duration, 0.1), 1.0))
        
        return {
            'word_confidence_mean': float(word_confidence_mean),
            'segment_confidence_mean': float(segment_confidence_mean),
            'temporal_consistency': float(temporal_consistency),
            'decode_efficiency': float(decode_efficiency),
            'word_count': len(words),
            'segment_count': len(segments)
        }
    
    def _calculate_decode_diversity_metrics(self, decode_results: List[DecodeResult]) -> None:
        """Calculate diversity metrics across decode results"""
        
        if len(decode_results) < 2:
            return
        
        # Calculate pairwise text similarities
        texts = [result.transcript_text for result in decode_results]
        
        for i, result in enumerate(decode_results):
            diversity_metrics = {
                'text_length_variance': self._calculate_length_variance(texts, i),
                'lexical_diversity': self._calculate_lexical_diversity(texts[i], texts),
                'structural_diversity': self._calculate_structural_diversity(result, decode_results),
                'confidence_variance': self._calculate_confidence_variance(decode_results, i)
            }
            
            result.diversity_metrics = diversity_metrics
    
    def _calculate_length_variance(self, texts: List[str], index: int) -> float:
        """Calculate variance in text length compared to other results"""
        
        lengths = [len(text.split()) for text in texts]
        current_length = lengths[index]
        other_lengths = lengths[:index] + lengths[index+1:]
        
        if not other_lengths:
            return 0.0
        
        mean_other_length = float(np.mean(other_lengths))
        variance = abs(current_length - mean_other_length) / max(mean_other_length, 1.0)
        
        return min(variance, 1.0)
    
    def _calculate_lexical_diversity(self, current_text: str, all_texts: List[str]) -> float:
        """Calculate lexical diversity compared to other results"""
        
        current_words = set(re.findall(r'\b[a-zA-Z]+\b', current_text.lower()))
        
        # Calculate unique words compared to others
        other_texts = [text for text in all_texts if text != current_text]
        if not other_texts:
            return 0.0
        
        other_words = set()
        for text in other_texts:
            other_words.update(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
        
        unique_words = current_words - other_words
        diversity = len(unique_words) / max(len(current_words), 1.0)
        
        return diversity
    
    def _calculate_structural_diversity(self, current_result: DecodeResult, 
                                      all_results: List[DecodeResult]) -> float:
        """Calculate structural diversity in transcription approach"""
        
        other_results = [r for r in all_results if r.strategy_id != current_result.strategy_id]
        if not other_results:
            return 0.0
        
        # Compare number of segments, average segment length, etc.
        current_segments = len(current_result.segments)
        other_segments = [len(r.segments) for r in other_results]
        
        if not other_segments:
            return 0.0
        
        mean_other_segments = float(np.mean(other_segments))
        variance = abs(current_segments - mean_other_segments) / max(mean_other_segments, 1.0)
        
        return min(variance, 1.0)
    
    def _calculate_confidence_variance(self, all_results: List[DecodeResult], index: int) -> float:
        """Calculate variance in confidence patterns"""
        
        current_confidence = all_results[index].confidence_scores.get('word_confidence_mean', 0.0)
        other_confidences = [
            result.confidence_scores.get('word_confidence_mean', 0.0) 
            for i, result in enumerate(all_results) if i != index
        ]
        
        if not other_confidences:
            return 0.0
        
        mean_other_confidence = float(np.mean(other_confidences))
        variance = abs(current_confidence - mean_other_confidence)
        
        return variance
    
    def _calculate_overall_diversity(self, decode_results: List[DecodeResult]) -> float:
        """Calculate overall diversity score across all decode results"""
        
        if len(decode_results) < 2:
            return 0.0
        
        diversity_scores = []
        
        for result in decode_results:
            if result.diversity_metrics:
                # Combine diversity metrics
                result_diversity = np.mean([
                    result.diversity_metrics.get('lexical_diversity', 0.0),
                    result.diversity_metrics.get('structural_diversity', 0.0),
                    result.diversity_metrics.get('confidence_variance', 0.0)
                ])
                diversity_scores.append(result_diversity)
        
        return float(np.mean(diversity_scores)) if diversity_scores else 0.0
    
    def _update_session_context_from_results(self, decode_results: List[DecodeResult]) -> None:
        """Update session context with results for future processing"""
        
        # Add results to previous segments for lexicon building
        for result in decode_results:
            segment_data = {
                'text': result.transcript_text,
                'language': result.language,
                'confidence': result.confidence_scores.get('word_confidence_mean', 0.0),
                'decode_strategy': result.strategy_id,
                'timestamp': time.time()
            }
            self.previous_segments.append(segment_data)
        
        # Limit session history to prevent memory bloat
        if len(self.previous_segments) > 20:
            self.previous_segments = self.previous_segments[-20:]
        
        self.structured_logger.info("Updated session context with new decode results", 
                                   decode_results_count=len(decode_results))
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session state for debugging/monitoring"""
        
        return {
            'session_lexicon': {
                'domain_terms': len(self.session_lexicon.domain_terms) if self.session_lexicon else 0,
                'technical_terms': len(self.session_lexicon.technical_terms) if self.session_lexicon else 0,
                'proper_nouns': len(self.session_lexicon.proper_nouns) if self.session_lexicon else 0,
                'recurring_phrases': len(self.session_lexicon.recurring_phrases) if self.session_lexicon else 0
            },
            'session_context': {
                'previous_segments': len(self.previous_segments),
                'unique_languages': len(set(seg.get('language', 'unknown') for seg in self.previous_segments)),
                'average_confidence': float(np.mean([seg.get('confidence', 0.0) for seg in self.previous_segments])) if self.previous_segments else 0.0
            },
            'decode_strategies': {
                'total_strategies': len(self.base_strategies),
                'strategy_types': [s.correlation_target for s in self.base_strategies]
            }
        }