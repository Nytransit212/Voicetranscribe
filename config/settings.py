import os
from typing import Dict, Any

class Settings:
    """Configuration settings for the ensemble transcription system"""
    
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # Audio Processing Settings
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1  # Mono
    MAX_AUDIO_DURATION = 5400  # 90 minutes in seconds
    
    # Diarization Settings
    DEFAULT_EXPECTED_SPEAKERS = 10
    MIN_SPEAKERS = 2
    MAX_SPEAKERS = 20
    
    DIARIZATION_VARIANTS = {
        'conservative': {
            'clustering_threshold': 0.7,
            'vad_onset': 0.5,
            'vad_offset': 0.4
        },
        'balanced': {
            'clustering_threshold': 0.6,
            'vad_onset': 0.4,
            'vad_offset': 0.3
        },
        'liberal': {
            'clustering_threshold': 0.5,
            'vad_onset': 0.3,
            'vad_offset': 0.2
        }
    }
    
    # ASR Settings
    ASR_MODEL = "whisper-1"
    ASR_VARIANTS = [
        {
            'temperature': 0.0,
            'language': 'en',
            'prompt': "The following is a recording of a meeting with multiple speakers."
        },
        {
            'temperature': 0.1,
            'language': 'en',
            'prompt': "This is a multi-speaker conversation with clear speaker changes."
        },
        {
            'temperature': 0.2,
            'language': 'en',
            'prompt': "Transcribe this meeting recording with multiple participants."
        },
        {
            'temperature': 0.0,
            'language': None,
            'prompt': ""
        },
        {
            'temperature': 0.3,
            'language': 'en',
            'prompt': "This recording contains multiple speakers in a meeting discussion."
        }
    ]
    
    # Confidence Scoring Weights
    CONFIDENCE_WEIGHTS = {
        'D': 0.28,  # Diarization consistency
        'A': 0.32,  # ASR alignment and confidence
        'L': 0.18,  # Linguistic quality
        'R': 0.12,  # Cross-run agreement
        'O': 0.10   # Overlap handling
    }
    
    # Quality Gates
    QUALITY_GATES = {
        'max_timestamp_regression_rate': 0.02,  # 2%
        'min_boundary_fit_rate': 0.92,  # 92%
        'min_overlap_plausibility': 0.30
    }
    
    # Output Settings
    SPEAKER_COLORS = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ]
    
    # File Settings
    SUPPORTED_VIDEO_FORMATS = ['.mp4']
    MAX_FILE_SIZE_MB = 2000  # 2GB
    TEMP_DIR_PREFIX = 'ensemble_transcription_'
    
    # Processing Settings
    MAX_CONCURRENT_ASR_REQUESTS = 3
    ASR_TIMEOUT_SECONDS = 300  # 5 minutes
    DIARIZATION_TIMEOUT_SECONDS = 600  # 10 minutes
    
    @classmethod
    def get_diarization_config(cls, variant: str, expected_speakers: int) -> Dict[str, Any]:
        """
        Get diarization configuration for a specific variant.
        
        Args:
            variant: Variant name ('conservative', 'balanced', 'liberal')
            expected_speakers: Expected number of speakers
            
        Returns:
            Configuration dictionary
        """
        base_config = cls.DIARIZATION_VARIANTS.get(variant, cls.DIARIZATION_VARIANTS['balanced'])
        
        if variant == 'conservative':
            min_speakers = max(cls.MIN_SPEAKERS, expected_speakers - 2)
            max_speakers = expected_speakers + 1
        elif variant == 'balanced':
            min_speakers = max(cls.MIN_SPEAKERS, expected_speakers - 1)
            max_speakers = expected_speakers + 2
        else:  # liberal
            min_speakers = max(cls.MIN_SPEAKERS, expected_speakers)
            max_speakers = expected_speakers + 3
        
        config = base_config.copy()
        config.update({
            'min_speakers': min_speakers,
            'max_speakers': min(max_speakers, cls.MAX_SPEAKERS)
        })
        
        return config
    
    @classmethod
    def validate_environment(cls) -> Dict[str, bool]:
        """
        Validate that required environment variables are set.
        
        Returns:
            Dictionary of validation results
        """
        return {
            'openai_api_key': bool(cls.OPENAI_API_KEY),
            'huggingface_token': bool(cls.HUGGINGFACE_TOKEN),
        }
    
    @classmethod
    def get_runtime_config(cls) -> Dict[str, Any]:
        """
        Get runtime configuration summary.
        
        Returns:
            Configuration summary
        """
        return {
            'audio_sample_rate': cls.AUDIO_SAMPLE_RATE,
            'max_audio_duration': cls.MAX_AUDIO_DURATION,
            'asr_model': cls.ASR_MODEL,
            'confidence_weights': cls.CONFIDENCE_WEIGHTS,
            'quality_gates': cls.QUALITY_GATES,
            'max_concurrent_requests': cls.MAX_CONCURRENT_ASR_REQUESTS
        }
