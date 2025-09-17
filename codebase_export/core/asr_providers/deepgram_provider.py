"""
Deepgram ASR provider for API-based transcription with Nova models
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from .base import ASRProvider, ASRResult, ASRSegment, DecodeMode

logger = logging.getLogger(__name__)

class DeepgramProvider(ASRProvider):
    """
    Deepgram provider for high-quality API-based transcription
    
    Uses Deepgram Nova family models with smart formatting
    and meeting-optimized presets for US English meetings.
    """
    
    def __init__(self, provider_name: str = "deepgram", model_name: str = "nova-2", config: Dict[str, Any] = None):
        super().__init__(provider_name, model_name, config)
        self.api_key = self._get_api_key()
        self.client = None
        self._initialize_client()
    
    def _get_api_key(self) -> Optional[str]:
        """Get Deepgram API key from environment or config"""
        api_key = (
            self.config.get('api_key') or
            os.getenv('DEEPGRAM_API_KEY') or
            os.getenv('DG_API_KEY')
        )
        
        if not api_key:
            logger.warning("Deepgram API key not found. Set DEEPGRAM_API_KEY environment variable.")
        
        return api_key
    
    def _initialize_client(self) -> None:
        """Initialize Deepgram client"""
        if not self.api_key:
            logger.error("Cannot initialize Deepgram client: no API key")
            return
        
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
            
            self.client = DeepgramClient(self.api_key)
            self.PrerecordedOptions = PrerecordedOptions
            
            logger.info("Deepgram client initialized successfully")
            
        except ImportError:
            logger.error("deepgram-sdk not installed. Install with: pip install deepgram-sdk")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Deepgram client: {e}")
            self.client = None
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        decode_mode: DecodeMode = DecodeMode.DETERMINISTIC,
        language: str = "en-US",
        prompt: Optional[str] = None,
        **kwargs
    ) -> ASRResult:
        """
        Transcribe audio with Deepgram Nova models
        
        Args:
            audio_path: Path to audio file
            decode_mode: Quality vs speed tradeoff mode
            language: Language code (en-US for US English)
            prompt: Optional context prompt for better accuracy
            **kwargs: Additional Deepgram parameters
            
        Returns:
            ASRResult with segments and confidence scores
        """
        if not self.is_available():
            raise RuntimeError("Deepgram client not available")
        
        start_time = time.time()
        
        # Configure options based on decode mode
        options = self._get_decode_options(decode_mode, language, prompt)
        
        # Override with kwargs
        for key, value in kwargs.items():
            setattr(options, key, value)
        
        logger.info(f"Transcribing with Deepgram {self.model_name} in {decode_mode.value} mode")
        
        try:
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                buffer_data = audio_file.read()
            
            # Call Deepgram API
            response = self.client.listen.prerecorded.v("1").transcribe_file(
                buffer_data, options
            )
            
            # Parse response
            asr_segments = []
            full_text_parts = []
            
            if response.results and response.results.channels:
                for channel in response.results.channels:
                    for alternative in channel.alternatives:
                        if alternative.words:
                            # Group words into segments (Deepgram provides word-level timestamps)
                            segments = self._group_words_into_segments(alternative.words, alternative.confidence)
                            
                            for segment_data in segments:
                                calibrated_confidence = self.calibrate_confidence(
                                    segment_data['confidence'],
                                    segment_data['end'] - segment_data['start']
                                )
                                
                                asr_segment = ASRSegment(
                                    start=segment_data['start'],
                                    end=segment_data['end'],
                                    text=segment_data['text'],
                                    confidence=calibrated_confidence,
                                    words=segment_data['words']
                                )
                                
                                asr_segments.append(asr_segment)
                                full_text_parts.append(segment_data['text'])
                        
                        elif alternative.transcript:
                            # Fallback: single segment with full transcript
                            duration = self._estimate_duration(audio_path)
                            calibrated_confidence = self.calibrate_confidence(
                                alternative.confidence or 0.8,
                                duration
                            )
                            
                            asr_segment = ASRSegment(
                                start=0.0,
                                end=duration,
                                text=alternative.transcript,
                                confidence=calibrated_confidence
                            )
                            
                            asr_segments.append(asr_segment)
                            full_text_parts.append(alternative.transcript)
            
            processing_time = time.time() - start_time
            
            # Calculate overall confidence
            if asr_segments:
                avg_confidence = sum(seg.confidence for seg in asr_segments) / len(asr_segments)
            else:
                avg_confidence = 0.0
            
            # Create result
            result = ASRResult(
                segments=asr_segments,
                full_text=' '.join(full_text_parts),
                language=language,
                confidence=avg_confidence,
                calibrated_confidence=avg_confidence,
                processing_time=processing_time,
                provider=self.provider_name,
                decode_mode=decode_mode,
                model_name=self.model_name,
                metadata={
                    'request_id': getattr(response.metadata, 'request_id', None),
                    'model_info': getattr(response.metadata, 'model_info', {}),
                    'options': self._options_to_dict(options)
                }
            )
            
            logger.info(f"Deepgram transcription complete: {len(asr_segments)} segments, "
                       f"{avg_confidence:.3f} confidence, {processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            raise
    
    def _get_decode_options(self, decode_mode: DecodeMode, language: str, prompt: Optional[str]):
        """Get Deepgram options for specific decode mode"""
        base_options = {
            'model': self._get_model_for_mode(decode_mode),
            'language': language,
            'smart_format': True,
            'punctuate': True,
            'paragraphs': True,
            'utterances': True,
            'diarize': False,  # We handle diarization upstream
            'alternatives': 1,
            'profanity_filter': False,
            'redact': [],
            'search': [],
            'replace': [],
            'keywords': [],
            'interim_results': False,
            'endpointing': False
        }
        
        # Mode-specific options
        mode_options = {
            DecodeMode.CAREFUL: {
                'model': 'nova-2',
                'smart_format': True,
                'filler_words': False,
                'dictation': True
            },
            DecodeMode.DETERMINISTIC: {
                'model': 'nova-2',
                'smart_format': True,
                'filler_words': False
            },
            DecodeMode.EXPLORATORY: {
                'model': 'nova',
                'smart_format': True,
                'filler_words': True,
                'alternatives': 2
            },
            DecodeMode.FAST: {
                'model': 'base',
                'smart_format': False,
                'filler_words': False
            },
            DecodeMode.ENHANCED: {
                'model': 'nova-2',
                'smart_format': True,
                'filler_words': False,
                'dictation': True,
                'alternatives': 2
            }
        }
        
        # Merge options
        options_dict = base_options.copy()
        options_dict.update(mode_options.get(decode_mode, {}))
        
        # Add prompt as keywords if provided
        if prompt:
            options_dict['keywords'] = prompt.split()[:10]  # Limit keywords
        
        # Create PrerecordedOptions object
        options = self.PrerecordedOptions(**options_dict)
        
        return options
    
    def _get_model_for_mode(self, decode_mode: DecodeMode) -> str:
        """Get best Deepgram model for decode mode"""
        model_map = {
            DecodeMode.CAREFUL: 'nova-2',
            DecodeMode.DETERMINISTIC: 'nova-2', 
            DecodeMode.EXPLORATORY: 'nova',
            DecodeMode.FAST: 'base',
            DecodeMode.ENHANCED: 'nova-2'
        }
        return model_map.get(decode_mode, self.model_name)
    
    def _group_words_into_segments(self, words: List[Any], overall_confidence: float) -> List[Dict[str, Any]]:
        """Group word-level timestamps into logical segments"""
        if not words:
            return []
        
        segments = []
        current_segment = {
            'start': words[0].start,
            'words': [],
            'text_parts': []
        }
        
        for word in words:
            # Check for segment boundary (pause > 1 second or punctuation)
            if (current_segment['words'] and 
                (word.start - current_segment['words'][-1].end > 1.0 or
                 current_segment['words'][-1].word.endswith(('.', '!', '?')))):
                
                # Finish current segment
                current_segment['end'] = current_segment['words'][-1].end
                current_segment['text'] = ' '.join(current_segment['text_parts']).strip()
                current_segment['confidence'] = overall_confidence  # Use overall confidence
                
                segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    'start': word.start,
                    'words': [],
                    'text_parts': []
                }
            
            # Add word to current segment
            current_segment['words'].append({
                'word': word.word,
                'start': word.start,
                'end': word.end,
                'confidence': word.confidence if hasattr(word, 'confidence') else overall_confidence
            })
            current_segment['text_parts'].append(word.word)
        
        # Finish last segment
        if current_segment['words']:
            current_segment['end'] = current_segment['words'][-1]['end']
            current_segment['text'] = ' '.join(current_segment['text_parts']).strip()
            current_segment['confidence'] = overall_confidence
            segments.append(current_segment)
        
        return segments
    
    def _estimate_duration(self, audio_path: Union[str, Path]) -> float:
        """Estimate audio duration using ffprobe"""
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', str(audio_path)
            ], capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 60.0  # Default fallback
    
    def _options_to_dict(self, options) -> Dict[str, Any]:
        """Convert PrerecordedOptions to dict for metadata"""
        try:
            return {attr: getattr(options, attr) for attr in dir(options) 
                    if not attr.startswith('_') and not callable(getattr(options, attr))}
        except:
            return {}
    
    def is_available(self) -> bool:
        """Check if Deepgram client is available"""
        return self.client is not None and self.api_key is not None
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats"""
        return ['wav', 'mp3', 'mp4', 'flac', 'm4a', 'ogg', 'webm', 'aac']
    
    def get_max_file_size(self) -> int:
        """Get maximum file size for Deepgram"""
        return 2 * 1024 * 1024 * 1024  # 2GB
    
    def estimate_processing_time(self, audio_duration: float, decode_mode: DecodeMode) -> float:
        """Estimate processing time for Deepgram API"""
        # Deepgram is very fast, usually 0.1-0.3x real-time
        base_ratios = {
            DecodeMode.FAST: 0.05,
            DecodeMode.DETERMINISTIC: 0.1,
            DecodeMode.CAREFUL: 0.15,
            DecodeMode.EXPLORATORY: 0.2,
            DecodeMode.ENHANCED: 0.25
        }
        
        return audio_duration * base_ratios.get(decode_mode, 0.1)
    
    def _load_confidence_calibration(self) -> Dict[str, float]:
        """Load Deepgram-specific confidence calibration"""
        # Deepgram confidence tends to be well-calibrated but slightly optimistic
        return {
            'scale': 0.95,
            'offset': 0.0,
            'min_confidence': 0.2,
            'max_confidence': 0.98
        }