"""
ASR Provider Factory for intelligent provider selection and management
"""

from typing import Dict, List, Optional, Type, Union
from .base import ASRProvider, DecodeMode
from .openai_provider import OpenAIProvider
from .faster_whisper_provider import FasterWhisperProvider
from .deepgram_provider import DeepgramProvider
import logging

logger = logging.getLogger(__name__)

class ASRProviderFactory:
    """Factory for creating and managing ASR providers"""
    
    _providers: Dict[str, Type[ASRProvider]] = {
        'openai': OpenAIProvider,
        'faster-whisper': FasterWhisperProvider,
        'deepgram': DeepgramProvider
    }
    
    _provider_priority = [
        'faster-whisper',  # Local, cost-free, high quality
        'deepgram',        # API-based, excellent for meetings
        'openai'           # Fallback, good general quality
    ]
    
    @classmethod
    def create_provider(
        self,
        provider_name: str, 
        model_name: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> ASRProvider:
        """
        Create ASR provider instance
        
        Args:
            provider_name: Name of provider (openai, faster-whisper, deepgram)
            model_name: Specific model to use (provider-dependent)
            config: Provider-specific configuration
            
        Returns:
            Configured ASR provider instance
        """
        if provider_name not in self._providers:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {list(self._providers.keys())}")
        
        provider_class = self._providers[provider_name]
        
        # Set default models
        if model_name is None:
            model_name = self._get_default_model(provider_name)
        
        return provider_class(
            provider_name=provider_name,
            model_name=model_name, 
            config=config or {}
        )
    
    @classmethod
    def get_available_providers(self) -> List[str]:
        """Get list of available and configured providers"""
        available = []
        
        for provider_name in self._provider_priority:
            try:
                provider = self.create_provider(provider_name)
                if provider.is_available():
                    available.append(provider_name)
                    logger.info(f"Provider {provider_name} is available")
                else:
                    logger.warning(f"Provider {provider_name} is not available")
            except Exception as e:
                logger.error(f"Failed to check provider {provider_name}: {e}")
        
        return available
    
    @classmethod
    def create_ensemble(
        self,
        providers: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ) -> List[ASRProvider]:
        """
        Create ensemble of ASR providers for multi-engine transcription
        
        Args:
            providers: List of provider names (uses available if None)
            config: Global configuration for all providers
            
        Returns:
            List of configured ASR providers
        """
        if providers is None:
            providers = self.get_available_providers()
        
        ensemble = []
        for provider_name in providers:
            try:
                provider = self.create_provider(provider_name, config=config)
                if provider.is_available():
                    ensemble.append(provider)
                    logger.info(f"Added {provider_name} to ensemble")
                else:
                    logger.warning(f"Skipping unavailable provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to create provider {provider_name}: {e}")
        
        if not ensemble:
            logger.error("No ASR providers available for ensemble")
            raise RuntimeError("No ASR providers could be configured")
        
        return ensemble
    
    @classmethod
    def get_best_provider_for_mode(
        self, 
        decode_mode: DecodeMode,
        available_providers: Optional[List[str]] = None
    ) -> str:
        """
        Select best provider for specific decode mode
        
        Args:
            decode_mode: Target decode mode
            available_providers: List of available providers
            
        Returns:
            Best provider name for the mode
        """
        if available_providers is None:
            available_providers = self.get_available_providers()
        
        # Provider preferences by decode mode
        mode_preferences = {
            DecodeMode.CAREFUL: ['faster-whisper', 'openai', 'deepgram'],
            DecodeMode.DETERMINISTIC: ['faster-whisper', 'deepgram', 'openai'],
            DecodeMode.EXPLORATORY: ['faster-whisper', 'deepgram', 'openai'],
            DecodeMode.FAST: ['deepgram', 'faster-whisper', 'openai'],
            DecodeMode.ENHANCED: ['deepgram', 'faster-whisper', 'openai']
        }
        
        preferences = mode_preferences.get(decode_mode, self._provider_priority)
        
        for provider_name in preferences:
            if provider_name in available_providers:
                return provider_name
        
        # Fallback to first available
        if available_providers:
            return available_providers[0]
        
        raise RuntimeError("No ASR providers available")
    
    @classmethod
    def estimate_cost(
        self,
        provider_name: str,
        audio_duration: float,
        decode_mode: DecodeMode = DecodeMode.DETERMINISTIC
    ) -> float:
        """
        Estimate processing cost for provider and audio duration
        
        Args:
            provider_name: ASR provider name
            audio_duration: Audio duration in seconds
            decode_mode: Decode mode affects processing time
            
        Returns:
            Estimated cost in USD
        """
        # Cost estimates per minute (approximate)
        cost_per_minute = {
            'openai': 0.006,        # $0.006 per minute
            'deepgram': 0.0125,     # $0.0125 per minute for Nova
            'faster-whisper': 0.0   # Local processing, electricity only
        }
        
        base_cost = cost_per_minute.get(provider_name, 0.0)
        minutes = audio_duration / 60.0
        
        # Mode multipliers for API costs (more calls = higher cost)
        mode_multiplier = {
            DecodeMode.FAST: 1.0,
            DecodeMode.DETERMINISTIC: 1.0,
            DecodeMode.CAREFUL: 1.2,    # Higher beam = more compute
            DecodeMode.EXPLORATORY: 1.1,
            DecodeMode.ENHANCED: 1.3    # Premium models
        }.get(decode_mode, 1.0)
        
        return base_cost * minutes * mode_multiplier
    
    @classmethod
    def _get_default_model(self, provider_name: str) -> str:
        """Get default model for provider"""
        defaults = {
            'openai': 'whisper-1',
            'faster-whisper': 'large-v3',
            'deepgram': 'nova-2'
        }
        return defaults.get(provider_name, 'default')
    
    @classmethod
    def register_provider(self, name: str, provider_class: Type[ASRProvider]) -> None:
        """Register new ASR provider"""
        self._providers[name] = provider_class
        logger.info(f"Registered ASR provider: {name}")
    
    @classmethod
    def get_provider_info(self) -> Dict[str, Dict[str, any]]:
        """Get information about all registered providers"""
        info = {}
        
        for name, provider_class in self._providers.items():
            try:
                provider = self.create_provider(name)
                info[name] = {
                    'available': provider.is_available(),
                    'supported_formats': provider.get_supported_formats(),
                    'max_file_size': provider.get_max_file_size(),
                    'model': provider.model_name,
                    'estimated_cost_per_minute': self.estimate_cost(name, 60.0)
                }
            except Exception as e:
                info[name] = {
                    'available': False,
                    'error': str(e)
                }
        
        return info