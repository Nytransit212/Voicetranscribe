"""
AssemblyAI Diarization Provider

Production-ready implementation of speaker diarization using AssemblyAI's API.
Supports large files, async processing, and robust error handling with fallback capabilities.
"""

import os
import time
import json
import hashlib
import requests
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .base import DiarizationProvider, DiarizationResult, DiarizationError, ProviderStatus
from utils.resilient_api import create_openai_retry_decorator
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.observability import trace_stage, track_cost
from utils.intelligent_cache import get_cache_manager, cached_operation
from core.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig


class AssemblyAIError(DiarizationError):
    """AssemblyAI-specific errors"""
    pass


class AssemblyAIDiarizationProvider(DiarizationProvider):
    """
    AssemblyAI Speaker Diarization Provider
    
    Provides production-ready speaker diarization using AssemblyAI's API with:
    - Support for large files (90+ minutes) via async processing
    - Robust error handling and retry logic
    - Rate limiting and cost management  
    - Progress tracking for long-running operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("assemblyai", config)
        
        # Initialize API configuration
        self.api_key = self._get_api_key()
        self.base_url = "https://api.assemblyai.com/v2"
        self.upload_url = f"{self.base_url}/upload"
        self.transcript_url = f"{self.base_url}/transcript"
        
        # Configure request session with headers
        self.session = requests.Session()
        self.session.headers.update({
            "authorization": self.api_key,
            "content-type": "application/json"
        })
        
        # Initialize enhanced logging and observability
        self.structured_logger = create_enhanced_logger("assemblyai_provider")
        self.cache_manager = get_cache_manager()
        
        # Configure retry decorator for AssemblyAI API calls
        self._api_retry = create_openai_retry_decorator("assemblyai")
        
        # Configure circuit breaker
        self.circuit_breaker = get_circuit_breaker(
            "assemblyai",
            CircuitBreakerConfig(
                service_name="assemblyai",
                failure_threshold=3,
                recovery_timeout=120,
                success_threshold=2,
                timeout_counts_as_failure=True
            )
        )
        
        # Processing configuration
        self.max_file_size_mb = config.get('max_file_size_mb', 512)  # 512MB limit
        self.max_duration_seconds = config.get('max_duration_seconds', 7200)  # 2 hours
        self.poll_interval = config.get('poll_interval', 5)  # Seconds between status checks
        self.max_wait_time = config.get('max_wait_time', 1800)  # 30 minutes max wait
        
        # Diarization parameters
        self.enable_speaker_diarization = True
        self.speakers_expected = config.get('default_speakers', None)
    
    def _get_api_key(self) -> str:
        """Get and validate AssemblyAI API key"""
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise AssemblyAIError("ASSEMBLYAI_API_KEY environment variable not set")
        
        if len(api_key) < 20:
            raise AssemblyAIError("Invalid ASSEMBLYAI_API_KEY format")
        
        return api_key
    
    @trace_stage("assemblyai_diarization")
    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        **kwargs
    ) -> DiarizationResult:
        """
        Perform speaker diarization using AssemblyAI API.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers 
            **kwargs: Additional parameters
            
        Returns:
            DiarizationResult with speaker segments
        """
        start_time = time.time()
        
        self.structured_logger.stage_start("assemblyai_diarization", 
                                         "Starting AssemblyAI speaker diarization",
                                         context={'audio_path': audio_path, 'min_speakers': min_speakers, 'max_speakers': max_speakers})
        
        try:
            # Validate file exists and size
            audio_file_path = Path(audio_path)
            if not audio_file_path.exists():
                raise AssemblyAIError(f"Audio file not found: {audio_path}")
            
            file_size_mb = audio_file_path.stat().st_size / (1024 * 1024)
            if not self.supports_file_size(file_size_mb):
                raise AssemblyAIError(f"File size {file_size_mb:.1f}MB exceeds limit {self.max_file_size_mb}MB")
            
            # Check cache first
            cache_key = self._generate_cache_key(audio_path, min_speakers, max_speakers, kwargs)
            cached_result = self.cache_manager.get("assemblyai_diarization", cache_key)
            if cached_result:
                self.structured_logger.info("Using cached AssemblyAI diarization result")
                return DiarizationResult(**cached_result)
            
            # Step 1: Upload audio file
            self.structured_logger.info("Uploading audio file to AssemblyAI")
            upload_url = self._upload_file(audio_path)
            
            # Step 2: Submit transcription request with diarization
            self.structured_logger.info("Submitting diarization request")
            transcript_id = self._submit_transcription_request(
                upload_url, min_speakers, max_speakers, **kwargs
            )
            
            # Step 3: Poll for completion
            self.structured_logger.info(f"Polling for completion, transcript_id: {transcript_id}")
            transcript_result = self._poll_for_completion(transcript_id)
            
            # Step 4: Parse diarization results
            diarization_result = self._parse_diarization_result(
                transcript_result, start_time, min_speakers, max_speakers
            )
            
            # Cache the result
            self.cache_manager.set("assemblyai_diarization", cache_key, diarization_result.__dict__)
            
            processing_time = time.time() - start_time
            self.structured_logger.stage_complete("assemblyai_diarization",
                                                f"AssemblyAI diarization completed: {len(diarization_result.segments)} segments, {diarization_result.total_speakers} speakers",
                                                duration=processing_time,
                                                metrics={
                                                    'segments_count': len(diarization_result.segments),
                                                    'total_speakers': diarization_result.total_speakers,
                                                    'confidence_score': diarization_result.confidence_score
                                                })
            
            return diarization_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.structured_logger.error(f"AssemblyAI diarization failed: {e}",
                                       duration=processing_time,
                                       context={'error_type': type(e).__name__})
            raise AssemblyAIError(f"Diarization failed: {e}") from e
    
    def _upload_file(self, audio_path: str) -> str:
        """Upload audio file to AssemblyAI and return upload URL"""
        @self._api_retry
        def _upload_request():
            with open(audio_path, 'rb') as audio_file:
                response = self.session.post(
                    self.upload_url,
                    files={'file': audio_file},
                    headers={'authorization': self.api_key}  # Override content-type header
                )
            return response
        
        response = _upload_request()
        
        if response.status_code != 200:
            raise AssemblyAIError(f"Upload failed: {response.status_code} - {response.text}")
        
        upload_data = response.json()
        return upload_data['upload_url']
    
    def _submit_transcription_request(
        self,
        upload_url: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        **kwargs
    ) -> str:
        """Submit transcription request with speaker diarization enabled"""
        
        # Build transcription configuration
        config = {
            "audio_url": upload_url,
            "speaker_labels": True,  # Enable speaker diarization
        }
        
        # Set speaker count if provided
        if min_speakers is not None and max_speakers is not None:
            # AssemblyAI uses speakers_expected parameter
            speakers_expected = kwargs.get('speakers_expected', (min_speakers + max_speakers) // 2)
            config["speakers_expected"] = speakers_expected
        elif self.speakers_expected:
            config["speakers_expected"] = self.speakers_expected
        
        # Additional configuration options
        config.update({
            "language_detection": True,  # Auto-detect language
            "punctuate": True,
            "format_text": True,
            "dual_channel": False,  # Assume mono or mixed stereo
        })
        
        # Apply any additional parameters
        extra_config = kwargs.get('assemblyai_config', {})
        config.update(extra_config)
        
        @self._api_retry
        def _submit_request():
            return self.session.post(self.transcript_url, json=config)
        
        response = _submit_request()
        
        if response.status_code != 200:
            raise AssemblyAIError(f"Transcription request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result['id']
    
    def _poll_for_completion(self, transcript_id: str) -> Dict[str, Any]:
        """Poll AssemblyAI API until transcription is complete"""
        start_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - start_time > self.max_wait_time:
                raise AssemblyAIError(f"Transcription timed out after {self.max_wait_time} seconds")
            
            # Get status
            status_response = self._get_transcription_status(transcript_id)
            status = status_response.get('status')
            
            self.structured_logger.debug(f"Transcription status: {status}")
            
            if status == 'completed':
                return status_response
            elif status == 'error':
                error_msg = status_response.get('error', 'Unknown error')
                raise AssemblyAIError(f"Transcription failed: {error_msg}")
            elif status in ['queued', 'processing']:
                # Continue polling
                time.sleep(self.poll_interval)
                continue
            else:
                raise AssemblyAIError(f"Unknown transcription status: {status}")
    
    def _get_transcription_status(self, transcript_id: str) -> Dict[str, Any]:
        """Get transcription status from AssemblyAI"""
        @self._api_retry
        def _status_request():
            return self.session.get(f"{self.transcript_url}/{transcript_id}")
        
        response = _status_request()
        
        if response.status_code != 200:
            raise AssemblyAIError(f"Status check failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    def _parse_diarization_result(
        self,
        transcript_result: Dict[str, Any],
        start_time: float,
        min_speakers: Optional[int],
        max_speakers: Optional[int]
    ) -> DiarizationResult:
        """Parse AssemblyAI transcript result into DiarizationResult format"""
        
        # Extract speaker segments
        segments = []
        speakers_found = set()
        
        # AssemblyAI returns utterances with speaker labels
        utterances = transcript_result.get('utterances', [])
        
        for utterance in utterances:
            start_ms = utterance.get('start', 0)
            end_ms = utterance.get('end', 0)
            speaker = utterance.get('speaker', 'SPEAKER_00')
            confidence = utterance.get('confidence', 0.5)
            
            # Convert milliseconds to seconds
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            
            # Ensure speaker ID format consistency
            if not speaker.startswith('SPEAKER_'):
                speaker = f"SPEAKER_{speaker}"
            
            speakers_found.add(speaker)
            
            segments.append({
                'start': start_sec,
                'end': end_sec,
                'speaker_id': speaker,
                'confidence': confidence,
                'text': utterance.get('text', '')
            })
        
        # Calculate overall metrics
        total_speakers = len(speakers_found)
        processing_time = time.time() - start_time
        
        # Calculate overall confidence from individual segment confidences
        if segments:
            confidence_score = sum(seg['confidence'] for seg in segments) / len(segments)
        else:
            confidence_score = 0.0
        
        # Provider metadata
        provider_metadata = {
            'transcript_id': transcript_result.get('id'),
            'language_detected': transcript_result.get('language_code'),
            'audio_duration': transcript_result.get('audio_duration'),
            'confidence': transcript_result.get('confidence'),
            'utterances_count': len(utterances),
            'processing_time_api': transcript_result.get('processing_time'),
            'api_version': 'v2'
        }
        
        return DiarizationResult(
            segments=segments,
            total_speakers=total_speakers,
            confidence_score=confidence_score,
            processing_time=processing_time,
            provider_metadata=provider_metadata
        )
    
    def _generate_cache_key(
        self, 
        audio_path: str, 
        min_speakers: Optional[int], 
        max_speakers: Optional[int], 
        kwargs: Dict[str, Any]
    ) -> str:
        """Generate cache key for diarization request"""
        # Create hash from file path + file modification time + parameters
        file_stat = Path(audio_path).stat()
        
        cache_data = {
            'audio_path': str(audio_path),
            'file_size': file_stat.st_size,
            'modified_time': file_stat.st_mtime,
            'min_speakers': min_speakers,
            'max_speakers': max_speakers,
            'config': {k: v for k, v in kwargs.items() if k != 'assemblyai_config'}
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def validate_config(self) -> bool:
        """Validate AssemblyAI configuration and API access"""
        try:
            # Test API access with a simple request
            response = self.session.get(f"{self.base_url}/transcript", params={'limit': 1})
            return response.status_code == 200
        except Exception as e:
            self.structured_logger.error(f"Config validation failed: {e}")
            return False
    
    def health_check(self) -> ProviderStatus:
        """Check AssemblyAI API health and availability"""
        try:
            # Simple health check request
            response = self.session.get(f"{self.base_url}/transcript", params={'limit': 1})
            
            if response.status_code == 200:
                return ProviderStatus.HEALTHY
            elif response.status_code == 429:
                return ProviderStatus.DEGRADED  # Rate limited
            elif 500 <= response.status_code < 600:
                return ProviderStatus.DEGRADED  # Server issues
            else:
                return ProviderStatus.ERROR
                
        except requests.exceptions.ConnectionError:
            return ProviderStatus.UNAVAILABLE
        except Exception as e:
            self.structured_logger.error(f"Health check failed: {e}")
            return ProviderStatus.ERROR
    
    def _get_variant_configurations(
        self, 
        base_params: Dict[str, Any], 
        num_variants: int
    ) -> List[Dict[str, Any]]:
        """
        Generate AssemblyAI-specific parameter configurations for variants.
        
        Leverages AssemblyAI's speaker diarization features for optimal results.
        """
        variants = []
        
        # Extract base parameters
        min_speakers = base_params.get('min_speakers', 2)
        max_speakers = base_params.get('max_speakers', 10)
        base_expected = (min_speakers + max_speakers) // 2
        
        # Variant 1: Conservative (fewer speakers expected)
        variants.append({
            **base_params,
            'speakers_expected': max(2, base_expected - 2),
            'variant_name': 'Conservative',
            'assemblyai_config': {
                'speaker_labels': True,
                'dual_channel': False
            }
        })
        
        # Variant 2: Balanced (expected speaker count)
        variants.append({
            **base_params,
            'speakers_expected': base_expected,
            'variant_name': 'Balanced',
            'assemblyai_config': {
                'speaker_labels': True,
                'dual_channel': False
            }
        })
        
        # Variant 3: Liberal (more speakers expected)
        variants.append({
            **base_params,
            'speakers_expected': min(20, base_expected + 3),  # AssemblyAI max ~20 speakers
            'variant_name': 'Liberal',
            'assemblyai_config': {
                'speaker_labels': True,
                'dual_channel': False
            }
        })
        
        # Variant 4: Multi-resolution (auto-detect speakers)
        variants.append({
            **base_params,
            # No speakers_expected - let AssemblyAI auto-detect
            'variant_name': 'Multi_Resolution',
            'assemblyai_config': {
                'speaker_labels': True,
                'dual_channel': False,
                'language_detection': True
            }
        })
        
        # Variant 5: Ensemble optimized
        variants.append({
            **base_params,
            'speakers_expected': base_expected,
            'variant_name': 'Ensemble_Optimized',
            'assemblyai_config': {
                'speaker_labels': True,
                'dual_channel': False,
                'punctuate': True,
                'format_text': True
            }
        })
        
        return variants[:num_variants]