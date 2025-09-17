import os
import json
import tempfile
import random
from typing import List, Dict, Any, Optional
from openai import OpenAI
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.deterministic_parallel import DeterministicThreadPoolExecutor, ensure_deterministic_futures_processing
import queue
import time
import logging
from openai import OpenAIError, RateLimitError, APITimeoutError, APIConnectionError

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import get_observability_manager, trace_stage, track_cost
from utils.reliability_config import get_concurrency_config, get_timeout_config
from utils.resilient_api import openai_retry
from core.circuit_breaker import CircuitBreakerOpenException
from core.decode_strategy_enhancer import DecodeStrategyEnhancer, DecodeResult
from core.run_context import get_global_run_context, apply_stage_seeding

class ASREngine:
    """Handles Automatic Speech Recognition with ensemble variants"""
    
    def __init__(self, enable_enhanced_decode: bool = True):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "whisper-1"  # OpenAI Whisper model for audio transcription
        
        # Load reliability configuration
        self.concurrency_config = get_concurrency_config()
        self.timeout_config = get_timeout_config()
        
        # Initialize enhanced observability
        self.obs_manager = get_observability_manager()
        self.structured_logger = create_enhanced_logger("asr_engine")
        
        # Initialize enhanced decode strategy enhancer
        self.enable_enhanced_decode = enable_enhanced_decode
        if self.enable_enhanced_decode:
            try:
                self.decode_enhancer = DecodeStrategyEnhancer()
                print("✅ Enhanced decode strategy system initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize enhanced decode system: {e}")
                self.enable_enhanced_decode = False
                self.decode_enhancer = None
        else:
            self.decode_enhancer = None
        
        # Configure basic logging for detailed retry tracking
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @trace_stage("asr_ensemble_processing")
    def run_asr_ensemble(self, audio_path: str, diarization_variants: List[Dict[str, Any]], target_language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run 5 ASR variants for each of the 3 diarization results (15 total candidates).
        
        Args:
            audio_path: Path to cleaned audio file
            diarization_variants: List of 3 diarization variant results
            
        Returns:
            List of 15 candidate transcripts with ASR and diarization data
        """
        # Log ASR ensemble start
        self.structured_logger.stage_start("asr_ensemble", 
                                          f"Starting ASR ensemble processing for {len(diarization_variants)} diarization variants",
                                          context={'diarization_variants': len(diarization_variants), 'target_language': target_language})
        
        print(f"🎤 Starting ASR ensemble processing for {len(diarization_variants)} diarization variants...")
        
        # Apply deterministic seeding for ASR stage
        apply_stage_seeding('asr')
        
        candidates = []
        
        # For each diarization variant, run 5 ASR passes
        for i, diar_variant in enumerate(diarization_variants, 1):
            variant_id = f"diar_v{diar_variant.get('variant_id', i)}"
            
            self.structured_logger.variant_start(variant_id, "asr_processing", 
                                               f"Processing diarization variant {i}/{len(diarization_variants)}")
            
            print(f"Processing diarization variant {i}/{len(diarization_variants)}...")
            try:
                start_time = time.time()
                asr_variants = self._create_asr_variants(audio_path, diar_variant, target_language)
                candidates.extend(asr_variants)
                
                variant_time = time.time() - start_time
                self.structured_logger.variant_complete(variant_id, "asr_processing", 
                                                      f"Completed diarization variant {i}: {len(asr_variants)} ASR candidates generated",
                                                      metrics={'asr_variants_generated': len(asr_variants), 'processing_time': variant_time})
                
                print(f"✓ Completed diarization variant {i}: {len(asr_variants)} ASR candidates generated")
            except Exception as e:
                self.structured_logger.error(f"Error processing diarization variant {i}: {e}", 
                                           variant_id=variant_id, stage="asr_processing")
                print(f"⚠ Error processing diarization variant {i}: {e}")
                continue
        
        # Log ASR ensemble completion
        self.structured_logger.stage_complete("asr_ensemble", 
                                            f"ASR ensemble complete: {len(candidates)} total candidates generated",
                                            metrics={'total_candidates': len(candidates)})
        
        print(f"🎤 ASR ensemble complete: {len(candidates)} total candidates generated")
        return candidates
    
    @openai_retry
    @trace_stage("openai_transcription_api")
    def _make_transcription_api_call(self, audio_file_path: str, **transcription_params) -> Any:
        """
        Make OpenAI transcription API call with tenacity retry and circuit breaker protection.
        Enhanced with cost tracking and OpenTelemetry tracing.
        
        Args:
            audio_file_path: Path to audio file
            **transcription_params: Parameters for transcription API
            
        Returns:
            API response if successful
            
        Raises:
            Various OpenAI exceptions if call fails after retries
        """
        import librosa
        
        # Calculate audio duration for cost tracking
        try:
            audio_duration = librosa.get_duration(path=audio_file_path)
        except Exception:
            # Fallback estimation
            import os
            file_size = os.path.getsize(audio_file_path)
            audio_duration = file_size / 32000  # Rough estimate for 16kHz mono
        
        start_time = time.time()
        
        with open(audio_file_path, 'rb') as audio_file:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                timeout=self.timeout_config.api_request,
                **transcription_params
            )
        
        api_duration = time.time() - start_time
        
        # Track cost and usage
        usage_data = {
            "audio_duration_seconds": audio_duration,
            "model": transcription_params.get('model', self.model),
            "language": transcription_params.get('language', 'auto'),
            "temperature": transcription_params.get('temperature', 0.0)
        }
        
        cost = self.structured_logger.track_api_cost(
            service="openai_whisper",
            model=self.model,
            usage_data=usage_data,
            duration=api_duration,
            stage="asr_transcription"
        )
        
        # Log detailed API call info
        self.structured_logger.info(
            f"OpenAI Whisper API call completed",
            api_duration=api_duration,
            audio_duration=audio_duration,
            cost_usd=cost,
            model=self.model,
            response_format=transcription_params.get('response_format', 'json'),
            language=transcription_params.get('language', 'auto'),
            temperature=transcription_params.get('temperature', 0.0)
        )
        
        return response
    
    def _create_asr_variants(self, audio_path: str, diarization_data: Dict[str, Any], target_language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create 5 ASR variants for a single diarization result using enhanced decode strategies or traditional processing.
        
        Args:
            audio_path: Path to audio file
            diarization_data: Single diarization variant data
            target_language: Optional target language for transcription
            
        Returns:
            List of 5 ASR variant results
        """
        # Use enhanced decode strategy system if available
        if self.enable_enhanced_decode and self.decode_enhancer:
            return self._create_enhanced_asr_variants(audio_path, diarization_data, target_language)
        else:
            return self._create_traditional_asr_variants(audio_path, diarization_data, target_language)
    
    def _create_enhanced_asr_variants(self, audio_path: str, diarization_data: Dict[str, Any], target_language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create ASR variants using enhanced decode strategy system with domain lexicon and context awareness.
        
        Args:
            audio_path: Path to audio file
            diarization_data: Single diarization variant data
            target_language: Optional target language for transcription
            
        Returns:
            List of ASR variant results with enhanced decode information
        """
        print(f"  Running enhanced decode strategies with domain lexicon...")
        start_time = time.time()
        
        try:
            # Use decode strategy enhancer to process audio with enhanced strategies
            if not self.decode_enhancer:
                raise Exception("Decode enhancer not initialized")
            
            decode_results = self.decode_enhancer.process_audio_with_enhanced_decode(
                audio_path=audio_path,
                diarization_data=diarization_data,
                asr_variant_id=diarization_data.get('variant_id', 1),
                target_language=target_language
            )
            
            # Convert decode results to expected ASR variant format
            variants = []
            for i, decode_result in enumerate(decode_results, 1):
                variant = self._convert_decode_result_to_asr_variant(
                    decode_result, diarization_data, i
                )
                variants.append(variant)
            
            elapsed = time.time() - start_time
            print(f"  ✓ Enhanced decode strategies completed in {elapsed:.1f}s ({len(variants)} strategies executed)")
            
            # Log enhanced decode metrics
            if self.decode_enhancer:
                session_summary = self.decode_enhancer.get_session_summary()
                self.structured_logger.info("Enhanced decode session summary", context=session_summary)
            
            return variants
            
        except Exception as e:
            self.structured_logger.error(f"Enhanced decode failed, falling back to traditional: {e}")
            print(f"  ⚠ Enhanced decode failed, falling back to traditional ASR: {e}")
            return self._create_traditional_asr_variants(audio_path, diarization_data, target_language)
    
    def _create_traditional_asr_variants(self, audio_path: str, diarization_data: Dict[str, Any], target_language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create ASR variants using traditional approach (fallback method).
        
        Args:
            audio_path: Path to audio file
            diarization_data: Single diarization variant data
            target_language: Optional target language for transcription
            
        Returns:
            List of 5 traditional ASR variant results
        """
        # Define 5 different ASR parameter sets with language support
        # Implement proper auto-detection ensemble when target_language is None
        if target_language is not None:
            # User specified language - use it for most variants with None for comparison
            lang_variants = [target_language, target_language, target_language, None, target_language]
        else:
            # Auto-detection mode - use diverse language assumptions for better ensemble
            lang_variants = [None, 'en', None, None, 'es']  # None for auto-detect, common languages as fallbacks
        
        asr_configs = [
            {
                'variant_id': 1,
                'temperature': 0.0,
                'language': lang_variants[0],
                'prompt': "The following is a recording of a meeting with multiple speakers.",
                'response_format': 'verbose_json'
            },
            {
                'variant_id': 2, 
                'temperature': 0.1,
                'language': lang_variants[1],
                'prompt': "This is a multi-speaker conversation with clear speaker changes.",
                'response_format': 'verbose_json'
            },
            {
                'variant_id': 3,
                'temperature': 0.2,
                'language': lang_variants[2], 
                'prompt': "Transcribe this meeting recording with multiple participants.",
                'response_format': 'verbose_json'
            },
            {
                'variant_id': 4,
                'temperature': 0.0,
                'language': lang_variants[3],  # Always None for auto-detect comparison
                'prompt': "",
                'response_format': 'verbose_json'
            },
            {
                'variant_id': 5,
                'temperature': 0.3,
                'language': lang_variants[4],
                'prompt': "This recording contains multiple speakers in a meeting discussion.",
                'response_format': 'verbose_json'
            }
        ]
        
        # Run ASR variants in parallel with rate limiting
        print(f"  Running 5 traditional ASR variants in parallel...")
        start_time = time.time()
        
        # Use bounded thread pool executor with configurable concurrency and queue limits
        from utils.bounded_executor import get_asr_executor
        
        variants = []
        executor = get_asr_executor()
        
        try:
            # Submit all tasks with backpressure handling
            future_to_config = {}
            for config in asr_configs:
                future = executor.submit_with_backpressure(
                    self._run_asr_variant_with_metadata, 
                    audio_path, 
                    diarization_data, 
                    config,
                    max_wait=5.0  # Wait up to 5 seconds for queue space
                )
                if future:
                    future_to_config[future] = config
                else:
                    print(f"  ⚠ ASR variant {config['variant_id']} rejected due to queue backpressure")
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result(timeout=self.timeout_config.asr_variant)  # Configurable timeout per variant
                    if result:
                        variants.append(result)
                        variant_id = config['variant_id']
                        print(f"  ✓ ASR variant {variant_id} completed")
                    
                except Exception as e:
                    variant_id = config['variant_id']
                    print(f"  ⚠ ASR variant {variant_id} failed: {e}")
                    continue
                    
        except Exception as e:
            self.structured_logger.error(f"Error in ASR variant processing: {e}")
            raise
        
        elapsed = time.time() - start_time
        print(f"  ✓ All ASR variants completed in {elapsed:.1f}s ({len(variants)}/{len(asr_configs)} successful)")
        
        return variants
    
    def _convert_decode_result_to_asr_variant(self, decode_result: 'DecodeResult', diarization_data: Dict[str, Any], variant_id: int) -> Dict[str, Any]:
        """
        Convert a DecodeResult to the expected ASR variant format for compatibility with the rest of the system.
        
        Args:
            decode_result: Enhanced decode result
            diarization_data: Diarization data for this variant
            variant_id: Sequential variant ID
            
        Returns:
            Dictionary in expected ASR variant format
        """
        # Create ASR result data in expected format
        asr_result = {
            'variant_id': variant_id,
            'text': decode_result.transcript_text,
            'words': decode_result.words,
            'segments': decode_result.segments,
            'language': decode_result.language,
            'duration': sum(w.get('end', 0) - w.get('start', 0) for w in decode_result.words) if decode_result.words else 0.0,
            'confidence_scores': decode_result.confidence_scores,
            # Enhanced decode specific data
            'decode_strategy': decode_result.strategy_id,
            'decode_parameters': decode_result.decode_parameters,
            'lexicon_usage': decode_result.lexicon_usage,
            'diversity_metrics': decode_result.diversity_metrics
        }
        
        # Combine diarization and ASR data in expected format
        candidate = {
            'candidate_id': f"diar_{diarization_data['variant_id']}_asr_{variant_id}_enhanced_{decode_result.strategy_id}",
            'diarization_variant_id': diarization_data['variant_id'],
            'asr_variant_id': variant_id,
            'diarization_data': diarization_data,
            'asr_data': asr_result,
            'aligned_segments': self._align_asr_to_diarization(asr_result, diarization_data),
            'parameters': {
                'diarization': diarization_data['parameters'],
                'asr': {
                    'variant_id': variant_id,
                    'temperature': decode_result.decode_parameters.get('temperature', 0.0),
                    'language': decode_result.language,
                    'response_format': 'verbose_json',
                    'decode_strategy': decode_result.strategy_id,
                    'enhanced_decode': True
                }
            },
            # Enhanced decode metadata
            'enhanced_decode_metadata': {
                'strategy_id': decode_result.strategy_id,
                'lexicon_terms_used': decode_result.lexicon_usage.get('priming_terms_used', 0),
                'lexicon_effectiveness': decode_result.lexicon_usage.get('lexicon_effectiveness', 0.0),
                'diversity_score': sum(decode_result.diversity_metrics.values()) / max(len(decode_result.diversity_metrics), 1) if decode_result.diversity_metrics else 0.0,
                'decode_parameters': decode_result.decode_parameters
            }
        }
        
        return candidate
    
    def _run_asr_variant_with_metadata(self, audio_path: str, diarization_data: Dict[str, Any], 
                                     config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ASR variant and create candidate with all metadata.
        Used for parallel processing.
        
        Args:
            audio_path: Path to audio file
            diarization_data: Diarization results
            config: ASR configuration parameters
            
        Returns:
            Complete candidate dictionary with ASR and diarization data
        """
        try:
            asr_result = self._run_asr_variant(audio_path, diarization_data, config)
            
            # Combine diarization and ASR data
            candidate = {
                'candidate_id': f"diar_{diarization_data['variant_id']}_asr_{config['variant_id']}",
                'diarization_variant_id': diarization_data['variant_id'],
                'asr_variant_id': config['variant_id'],
                'diarization_data': diarization_data,
                'asr_data': asr_result,
                'aligned_segments': self._align_asr_to_diarization(asr_result, diarization_data),
                'parameters': {
                    'diarization': diarization_data['parameters'],
                    'asr': config
                }
            }
            
            return candidate
            
        except Exception as e:
            # Re-raise with variant info for better error handling
            raise Exception(f"ASR variant {config['variant_id']} failed: {str(e)}")
    
    @trace_stage("asr_variant_processing")
    def _run_asr_variant(self, audio_path: str, diarization_data: Dict[str, Any], 
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single ASR variant with specific parameters.
        
        Args:
            audio_path: Path to audio file
            diarization_data: Diarization results
            config: ASR configuration parameters
            
        Returns:
            ASR transcription results
        """
        try:
            # Prepare parameters for OpenAI Whisper
            transcription_params = {
                'model': self.model,
                'response_format': config['response_format'],
                'temperature': config['temperature']
            }
            
            # Add optional parameters
            if config['language']:
                transcription_params['language'] = config['language']
            if config['prompt']:
                transcription_params['prompt'] = config['prompt']
            
            # Make transcription API call with tenacity retry and circuit breaker protection
            try:
                response = self._make_transcription_api_call(audio_path, **transcription_params)
            except CircuitBreakerOpenException as e:
                self.structured_logger.error(
                    f"Circuit breaker open for OpenAI service during ASR variant {config['variant_id']}",
                    context={'variant_id': config['variant_id'], 'error': str(e)}
                )
                raise Exception(f"OpenAI service temporarily unavailable: {str(e)}")
            except Exception as e:
                self.structured_logger.error(
                    f"ASR API call failed for variant {config['variant_id']} after retries",
                    context={'variant_id': config['variant_id'], 'error': str(e)}
                )
                raise
            
            # Extract word-level timestamps if available
            if hasattr(response, 'words') and response.words:
                words = [
                    {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'confidence': getattr(word, 'confidence', 0.9)  # Default confidence
                    }
                    for word in response.words
                ]
            else:
                # Fallback: create mock word timestamps from segments
                words = self._create_word_timestamps_from_segments(response)
            
            # Extract segments
            segments = []
            if hasattr(response, 'segments') and response.segments:
                segments = [
                    {
                        'id': seg.id,
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text,
                        'confidence': getattr(seg, 'avg_logprob', 0.0)  # Convert log prob to confidence
                    }
                    for seg in response.segments
                ]
            
            return {
                'variant_id': config['variant_id'],
                'text': response.text,
                'words': words,
                'segments': segments,
                'language': getattr(response, 'language', 'en'),
                'duration': getattr(response, 'duration', 0.0),
                'confidence_scores': self._calculate_confidence_metrics(words, segments)
            }
            
        except Exception as e:
            raise Exception(f"ASR transcription failed: {str(e)}")
    
    def _create_word_timestamps_from_segments(self, response) -> List[Dict[str, Any]]:
        """
        Create word-level timestamps from segment-level data when not available.
        
        Args:
            response: OpenAI Whisper response object
            
        Returns:
            List of word dictionaries with estimated timestamps
        """
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
                        'confidence': max(0.1, getattr(segment, 'avg_logprob', -1.0) + 1.0)  # Convert log prob
                    })
        
        return words
    
    def _calculate_confidence_metrics(self, words: List[Dict[str, Any]], 
                                    segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate confidence metrics for ASR results.
        
        Args:
            words: List of word-level results
            segments: List of segment-level results
            
        Returns:
            Dictionary of confidence metrics
        """
        if not words:
            return {'word_confidence_mean': 0.0, 'segment_confidence_mean': 0.0}
        
        # Word-level confidence
        word_confidences = [w['confidence'] for w in words if 'confidence' in w]
        word_confidence_mean = np.mean(word_confidences) if word_confidences else 0.0
        
        # Segment-level confidence  
        segment_confidences = [s['confidence'] for s in segments if 'confidence' in s]
        segment_confidence_mean = np.mean(segment_confidences) if segment_confidences else 0.0
        
        return {
            'word_confidence_mean': float(word_confidence_mean),
            'segment_confidence_mean': float(segment_confidence_mean),
            'word_count': len(words),
            'segment_count': len(segments)
        }
    
    def _align_asr_to_diarization(self, asr_data: Dict[str, Any], 
                                 diarization_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Align ASR word timestamps to diarization speaker segments with robust handling 
        of overlaps, gaps, and speaker transitions.
        
        Args:
            asr_data: ASR transcription results with word-level timestamps
            diarization_data: Speaker diarization results with segments
            
        Returns:
            List of aligned segments with enhanced speaker attribution and metrics
        """
        diar_segments = diarization_data.get('segments', [])
        words = asr_data.get('words', [])
        
        if not words or not diar_segments:
            return []
        
        # Sort by start time
        diar_segments = sorted(diar_segments, key=lambda x: x['start'])
        words = sorted(words, key=lambda x: x['start'])
        
        # Create word-to-speaker assignments with overlap handling
        word_assignments = self._assign_words_to_speakers(words, diar_segments)
        
        # Group words into coherent speaker segments
        aligned_segments = self._group_words_into_segments(word_assignments, diar_segments)
        
        # Calculate alignment quality metrics
        alignment_metrics = self._calculate_alignment_metrics(word_assignments, diar_segments)
        
        # Add metrics to each segment
        for segment in aligned_segments:
            segment['alignment_metrics'] = alignment_metrics
        
        return aligned_segments
    
    def _assign_words_to_speakers(self, words: List[Dict[str, Any]], 
                                 diar_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign each word to the most appropriate speaker using sophisticated overlap handling.
        
        Args:
            words: List of ASR words with timestamps
            diar_segments: List of diarization segments
            
        Returns:
            List of word assignments with speaker attribution and confidence
        """
        word_assignments = []
        tolerance_ms = 0.15  # 150ms tolerance for boundary alignment
        
        for word in words:
            word_start = word['start']
            word_end = word['end']
            word_center = (word_start + word_end) / 2
            
            # Find all overlapping diarization segments
            overlapping_segments = []
            for seg in diar_segments:
                # Check for any temporal overlap
                if (word_start <= seg['end'] + tolerance_ms and 
                    word_end >= seg['start'] - tolerance_ms):
                    
                    # Calculate overlap amount
                    overlap_start = max(word_start, seg['start'])
                    overlap_end = min(word_end, seg['end'])
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    overlapping_segments.append({
                        'segment': seg,
                        'overlap_duration': overlap_duration,
                        'overlap_ratio': overlap_duration / (word_end - word_start),
                        'center_distance': abs(word_center - (seg['start'] + seg['end']) / 2)
                    })
            
            # Determine best speaker assignment
            if overlapping_segments:
                # Sort by overlap ratio, then by center distance
                overlapping_segments.sort(
                    key=lambda x: (-x['overlap_ratio'], x['center_distance'])
                )
                
                best_match = overlapping_segments[0]
                assignment_confidence = min(1.0, best_match['overlap_ratio'] * 1.2)
                
                # Check for speaker conflicts (overlaps)
                speaker_conflict = len(overlapping_segments) > 1 and \
                                 overlapping_segments[1]['overlap_ratio'] > 0.3
                
                word_assignment = {
                    'word': word,
                    'speaker_id': best_match['segment']['speaker_id'],
                    'assignment_confidence': assignment_confidence,
                    'diarization_segment': best_match['segment'],
                    'overlap_ratio': best_match['overlap_ratio'],
                    'speaker_conflict': speaker_conflict,
                    'num_overlapping_speakers': len(overlapping_segments)
                }
            else:
                # Word falls in gap - assign to nearest speaker
                nearest_segment = min(
                    diar_segments,
                    key=lambda seg: min(
                        abs(word_start - seg['end']),  # Distance to segment end
                        abs(word_end - seg['start'])   # Distance to segment start
                    )
                )
                
                gap_distance = min(
                    abs(word_start - nearest_segment['end']),
                    abs(word_end - nearest_segment['start'])
                )
                
                # Lower confidence for gap assignments
                gap_confidence = max(0.1, 1.0 - (gap_distance / 2.0))  # 2s max penalty
                
                word_assignment = {
                    'word': word,
                    'speaker_id': nearest_segment['speaker_id'],
                    'assignment_confidence': gap_confidence,
                    'diarization_segment': nearest_segment,
                    'overlap_ratio': 0.0,
                    'speaker_conflict': False,
                    'num_overlapping_speakers': 0,
                    'gap_assignment': True,
                    'gap_distance': gap_distance
                }
            
            word_assignments.append(word_assignment)
        
        return word_assignments
    
    def _group_words_into_segments(self, word_assignments: List[Dict[str, Any]], 
                                  diar_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group word assignments into coherent speaker segments.
        
        Args:
            word_assignments: List of word-to-speaker assignments
            diar_segments: Original diarization segments
            
        Returns:
            List of aligned segments with speaker attribution
        """
        if not word_assignments:
            return []
        
        aligned_segments = []
        current_speaker = None
        current_words = []
        current_diar_segment = None
        
        for assignment in word_assignments:
            speaker_id = assignment['speaker_id']
            word = assignment['word']
            
            # Start new segment if speaker changed or significant time gap
            time_gap = 0
            if current_words:
                time_gap = word['start'] - current_words[-1]['end']
            
            should_start_new_segment = (
                current_speaker != speaker_id or
                time_gap > 2.0  # 2 second gap threshold
            )
            
            if should_start_new_segment and current_words and current_diar_segment:
                # Finish current segment
                aligned_segment = self._create_enhanced_aligned_segment(
                    current_diar_segment, current_words, word_assignments
                )
                if aligned_segment:  # Only add non-empty segments
                    aligned_segments.append(aligned_segment)
                current_words = []
            
            # Update current segment info
            current_speaker = speaker_id
            current_diar_segment = assignment['diarization_segment']
            current_words.append(word)
        
        # Add final segment
        if current_words and current_diar_segment:
            aligned_segment = self._create_enhanced_aligned_segment(
                current_diar_segment, current_words, word_assignments
            )
            if aligned_segment:  # Only add non-empty segments
                aligned_segments.append(aligned_segment)
        
        return aligned_segments
    
    def _create_enhanced_aligned_segment(self, diar_segment: Dict[str, Any], 
                                        words: List[Dict[str, Any]],
                                        all_assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create an enhanced aligned segment with comprehensive metadata.
        
        Args:
            diar_segment: Diarization segment data
            words: List of words in this segment
            all_assignments: All word assignments for context
            
        Returns:
            Enhanced aligned segment dictionary
        """
        if not words:
            return {}
        
        # Calculate segment boundaries from words
        segment_start = min(word['start'] for word in words)
        segment_end = max(word['end'] for word in words)
        
        # Join words into text
        segment_text = ' '.join(word.get('word', '').strip() for word in words)
        
        # Calculate confidence metrics
        word_confidences = [word.get('confidence', 0.0) for word in words]
        avg_word_confidence = float(np.mean(word_confidences)) if word_confidences else 0.0
        
        # Find assignment info for these words
        segment_assignments = [
            assignment for assignment in all_assignments 
            if any(w == assignment['word'] for w in words)
        ]
        
        # Calculate alignment quality metrics
        assignment_confidences = [a['assignment_confidence'] for a in segment_assignments]
        avg_assignment_confidence = float(np.mean(assignment_confidences)) if assignment_confidences else 0.0
        
        speaker_conflicts = sum(1 for a in segment_assignments if a.get('speaker_conflict', False))
        gap_assignments = sum(1 for a in segment_assignments if a.get('gap_assignment', False))
        
        # Calculate boundary alignment score
        boundary_score = self._calculate_boundary_alignment_score(
            segment_start, segment_end, diar_segment
        )
        
        return {
            'start': segment_start,
            'end': segment_end,
            'speaker_id': diar_segment['speaker_id'],
            'text': segment_text,
            'words': words,
            'confidence': avg_word_confidence,
            'word_count': len(words),
            'diarization_confidence': diar_segment.get('confidence', 0.0),
            'alignment_confidence': avg_assignment_confidence,
            'boundary_alignment_score': boundary_score,
            'speaker_conflicts': speaker_conflicts,
            'gap_assignments': gap_assignments,
            'temporal_coherence': self._calculate_temporal_coherence(words)
        }
    
    def _calculate_boundary_alignment_score(self, segment_start: float, segment_end: float, 
                                           diar_segment: Dict[str, Any]) -> float:
        """
        Calculate how well segment boundaries align with diarization boundaries.
        
        Args:
            segment_start: Start time of aligned segment
            segment_end: End time of aligned segment  
            diar_segment: Diarization segment
            
        Returns:
            Boundary alignment score (0.0-1.0)
        """
        diar_start = diar_segment['start']
        diar_end = diar_segment['end']
        
        # Calculate boundary offsets
        start_offset = abs(segment_start - diar_start)
        end_offset = abs(segment_end - diar_end)
        
        # Score based on offset (penalty for >200ms misalignment)
        start_score = max(0.0, 1.0 - (start_offset / 0.4))  # 400ms max penalty
        end_score = max(0.0, 1.0 - (end_offset / 0.4))
        
        return (start_score + end_score) / 2
    
    def _calculate_temporal_coherence(self, words: List[Dict[str, Any]]) -> float:
        """
        Calculate temporal coherence of word sequence.
        
        Args:
            words: List of words in segment
            
        Returns:
            Temporal coherence score (0.0-1.0)
        """
        if len(words) < 2:
            return 1.0
        
        violations = 0
        total_transitions = 0
        
        for i in range(len(words) - 1):
            current_end = words[i]['end']
            next_start = words[i + 1]['start']
            
            # Check for temporal violations
            if current_end > next_start + 0.05:  # 50ms tolerance
                violations += 1
            
            total_transitions += 1
        
        return 1.0 - (violations / max(total_transitions, 1))
    
    def _calculate_alignment_metrics(self, word_assignments: List[Dict[str, Any]], 
                                   diar_segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate comprehensive alignment quality metrics.
        
        Args:
            word_assignments: List of word-to-speaker assignments
            diar_segments: Diarization segments
            
        Returns:
            Dictionary of alignment quality metrics
        """
        if not word_assignments:
            return {
                'coverage_ratio': 0.0,
                'avg_assignment_confidence': 0.0,
                'boundary_precision': 0.0,
                'speaker_consistency': 0.0,
                'overlap_handling_score': 0.0
            }
        
        # Coverage ratio - fraction of words successfully aligned
        total_words = len(word_assignments)
        gap_assignments = sum(1 for a in word_assignments if a.get('gap_assignment', False))
        coverage_ratio = (total_words - gap_assignments) / total_words
        
        # Average assignment confidence
        confidences = [a['assignment_confidence'] for a in word_assignments]
        avg_assignment_confidence = float(np.mean(confidences))
        
        # Boundary precision - how well word boundaries align with speaker boundaries
        boundary_scores = []
        for assignment in word_assignments:
            if not assignment.get('gap_assignment', False):
                word = assignment['word']
                diar_seg = assignment['diarization_segment']
                score = self._calculate_boundary_alignment_score(
                    word['start'], word['end'], diar_seg
                )
                boundary_scores.append(score)
        
        boundary_precision = float(np.mean(boundary_scores)) if boundary_scores else 0.0
        
        # Speaker consistency - penalize rapid speaker changes
        speaker_changes = 0
        for i in range(len(word_assignments) - 1):
            if word_assignments[i]['speaker_id'] != word_assignments[i + 1]['speaker_id']:
                speaker_changes += 1
        
        speaker_consistency = max(0.0, 1.0 - (speaker_changes / max(total_words, 1)) * 2)
        
        # Overlap handling score - how well overlaps are managed
        conflict_words = sum(1 for a in word_assignments if a.get('speaker_conflict', False))
        overlap_handling_score = 1.0 - (conflict_words / max(total_words, 1))
        
        return {
            'coverage_ratio': coverage_ratio,
            'avg_assignment_confidence': avg_assignment_confidence,
            'boundary_precision': boundary_precision,
            'speaker_consistency': speaker_consistency,
            'overlap_handling_score': overlap_handling_score
        }
