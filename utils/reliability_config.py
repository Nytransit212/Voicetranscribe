"""
Reliability Configuration Utilities

Provides utilities for loading and accessing reliability configuration
from Hydra config files, with defaults and validation.
"""
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import os

from config.hydra_settings import HydraSettings


@dataclass
class TimeoutConfig:
    """Timeout configuration for different operations"""
    api_request: int = 300
    asr_variant: int = 180
    diarization: int = 600
    file_processing: int = 120


@dataclass
class ConcurrencyConfig:
    """Concurrency control configuration"""
    max_asr_requests: int = 3
    max_diarization: int = 2
    max_file_processors: int = 4
    thread_pool_queue_size: int = 50


@dataclass
class RetryConfig:
    """Retry configuration for a service"""
    max_attempts: int = 5
    initial_wait: float = 1.0
    max_wait: float = 60.0
    exponential_base: int = 2
    jitter: bool = True
    retry_on_status: Optional[list] = None


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for a service"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 2
    timeout_counts_as_failure: bool = True
    max_failure_history: int = 100


class ReliabilityConfigLoader:
    """
    Loads and provides access to reliability configuration from Hydra config.
    
    Handles defaults, service-specific overrides, and configuration validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._reliability_config = self.config.get('reliability', {})
    
    def _load_config(self, config_path: Optional[str] = None) -> DictConfig:
        """Load configuration using existing settings loader"""
        try:
            if config_path:
                config = OmegaConf.load(config_path)
                # Ensure we have a DictConfig, not a ListConfig
                if not isinstance(config, DictConfig):
                    raise ValueError(f"Expected dictionary config, got {type(config)}")
                return config
            else:
                # Try to load from Hydra settings
                try:
                    hydra_settings = HydraSettings()
                    return hydra_settings.config
                except:
                    # Fallback to loading config.yaml directly
                    config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
                    if os.path.exists(config_file):
                        config = OmegaConf.load(config_file)
                        # Ensure we have a DictConfig, not a ListConfig
                        if not isinstance(config, DictConfig):
                            raise ValueError(f"Expected dictionary config, got {type(config)}")
                        return config
                    else:
                        raise Exception("No config file found")
        except Exception as e:
            print(f"Warning: Could not load config, using defaults: {e}")
            return OmegaConf.create({
                'reliability': {
                    'timeouts': {},
                    'concurrency': {},
                    'retry': {'default': {}},
                    'circuit_breakers': {'default': {}}
                }
            })
    
    def get_timeout_config(self) -> TimeoutConfig:
        """Get timeout configuration with defaults and validation"""
        timeout_config = self._reliability_config.get('timeouts', {})
        
        # Handle new API timeout structure (dict with connect/read/total) or legacy single value
        api_request_config = timeout_config.get('api_request', {'connect': 3, 'read': 120, 'total': 180})
        if isinstance(api_request_config, dict):
            api_request_timeout = api_request_config  # Use the dict structure
        else:
            # Legacy single timeout value - convert to safe structure
            api_request_timeout = {
                'connect': 3,
                'read': min(api_request_config, 120),  # Cap at 2 minutes
                'total': min(api_request_config, 180)  # Cap at 3 minutes
            }
        
        # Get other timeouts with safe defaults
        asr_variant = timeout_config.get('asr_variant', 300)
        diarization = timeout_config.get('diarization', 600)
        file_processing = timeout_config.get('file_processing', 120)
        
        # Validate timeout hierarchy: API < stage < job wall-time
        total_api_timeout = api_request_timeout.get('total', 180) if isinstance(api_request_timeout, dict) else api_request_timeout
        
        if asr_variant <= total_api_timeout:
            print(f"⚠️ Warning: ASR variant timeout ({asr_variant}s) should be > API timeout ({total_api_timeout}s)")
            asr_variant = max(asr_variant, total_api_timeout + 60)  # Add 1 minute buffer
        
        if diarization <= asr_variant:
            print(f"⚠️ Warning: Diarization timeout ({diarization}s) should be > ASR variant timeout ({asr_variant}s)")
            diarization = max(diarization, asr_variant + 120)  # Add 2 minute buffer
        
        # Log timeout configuration for debugging
        print(f"✓ Timeout hierarchy validated: API={total_api_timeout}s, ASR={asr_variant}s, Diarization={diarization}s")
        
        return TimeoutConfig(
            api_request=api_request_timeout,
            asr_variant=asr_variant,
            diarization=diarization,
            file_processing=file_processing
        )
    
    def get_concurrency_config(self) -> ConcurrencyConfig:
        """Get concurrency configuration with defaults"""
        concurrency_config = self._reliability_config.get('concurrency', {})
        
        return ConcurrencyConfig(
            max_asr_requests=concurrency_config.get('max_asr_requests', 3),
            max_diarization=concurrency_config.get('max_diarization', 2),
            max_file_processors=concurrency_config.get('max_file_processors', 4),
            thread_pool_queue_size=concurrency_config.get('thread_pool_queue_size', 50)
        )
    
    def get_retry_config(self, service: str = 'default') -> RetryConfig:
        """
        Get retry configuration for a service.
        
        Args:
            service: Service name (e.g., 'openai', 'huggingface', 'subprocess')
            
        Returns:
            RetryConfig with service-specific overrides applied
        """
        retry_config = self._reliability_config.get('retry', {})
        
        # Start with default config
        default_config = retry_config.get('default', {})
        
        # Apply service-specific overrides
        service_config = retry_config.get(service, {})
        merged_config = {**default_config, **service_config}
        
        return RetryConfig(
            max_attempts=merged_config.get('max_attempts', 5),
            initial_wait=merged_config.get('initial_wait', 1.0),
            max_wait=merged_config.get('max_wait', 60.0),
            exponential_base=merged_config.get('exponential_base', 2),
            jitter=merged_config.get('jitter', True),
            retry_on_status=merged_config.get('retry_on_status', None)
        )
    
    def get_circuit_breaker_config(self, service: str = 'default') -> CircuitBreakerConfig:
        """
        Get circuit breaker configuration for a service.
        
        Args:
            service: Service name (e.g., 'openai', 'huggingface', 'ffmpeg')
            
        Returns:
            CircuitBreakerConfig with service-specific overrides applied
        """
        cb_config = self._reliability_config.get('circuit_breakers', {})
        
        # Start with default config
        default_config = cb_config.get('default', {})
        
        # Apply service-specific overrides
        service_config = cb_config.get(service, {})
        merged_config = {**default_config, **service_config}
        
        return CircuitBreakerConfig(
            failure_threshold=merged_config.get('failure_threshold', 5),
            recovery_timeout=merged_config.get('recovery_timeout', 60),
            success_threshold=merged_config.get('success_threshold', 2),
            timeout_counts_as_failure=merged_config.get('timeout_counts_as_failure', True),
            max_failure_history=merged_config.get('max_failure_history', 100)
        )
    
    def get_service_specific_configs(self, service: str) -> Dict[str, Any]:
        """
        Get all reliability configs for a specific service.
        
        Args:
            service: Service name
            
        Returns:
            Dictionary containing timeout, retry, and circuit breaker configs
        """
        return {
            'timeout': self.get_timeout_config(),
            'concurrency': self.get_concurrency_config(),
            'retry': self.get_retry_config(service),
            'circuit_breaker': self.get_circuit_breaker_config(service)
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate reliability configuration and return validation results.
        
        Returns:
            Dictionary with validation results and any issues found
        """
        issues = []
        warnings = []
        
        # Validate timeout values
        timeout_config = self.get_timeout_config()
        if timeout_config.api_request <= 0:
            issues.append("api_request timeout must be positive")
        if timeout_config.asr_variant > timeout_config.api_request:
            warnings.append("asr_variant timeout is longer than api_request timeout")
        
        # Validate concurrency values
        concurrency_config = self.get_concurrency_config()
        if concurrency_config.max_asr_requests <= 0:
            issues.append("max_asr_requests must be positive")
        if concurrency_config.thread_pool_queue_size <= 0:
            issues.append("thread_pool_queue_size must be positive")
        
        # Validate retry configs for known services
        for service in ['default', 'openai', 'huggingface', 'subprocess']:
            retry_config = self.get_retry_config(service)
            if retry_config.max_attempts <= 0:
                issues.append(f"{service} retry max_attempts must be positive")
            if retry_config.initial_wait <= 0:
                issues.append(f"{service} retry initial_wait must be positive")
            if retry_config.max_wait < retry_config.initial_wait:
                issues.append(f"{service} retry max_wait must be >= initial_wait")
        
        # Validate circuit breaker configs
        for service in ['default', 'openai', 'huggingface', 'ffmpeg']:
            cb_config = self.get_circuit_breaker_config(service)
            if cb_config.failure_threshold <= 0:
                issues.append(f"{service} circuit breaker failure_threshold must be positive")
            if cb_config.recovery_timeout <= 0:
                issues.append(f"{service} circuit breaker recovery_timeout must be positive")
            if cb_config.success_threshold <= 0:
                issues.append(f"{service} circuit breaker success_threshold must be positive")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'config_loaded': self._reliability_config is not None
        }


# Global config loader instance
reliability_config_loader = ReliabilityConfigLoader()


def get_timeout_config() -> TimeoutConfig:
    """Get global timeout configuration"""
    return reliability_config_loader.get_timeout_config()


def get_concurrency_config() -> ConcurrencyConfig:
    """Get global concurrency configuration"""
    return reliability_config_loader.get_concurrency_config()


def get_retry_config(service: str = 'default') -> RetryConfig:
    """Get retry configuration for a service"""
    return reliability_config_loader.get_retry_config(service)


def get_circuit_breaker_config(service: str = 'default') -> CircuitBreakerConfig:
    """Get circuit breaker configuration for a service"""
    return reliability_config_loader.get_circuit_breaker_config(service)


def get_service_configs(service: str) -> Dict[str, Any]:
    """Get all reliability configs for a service"""
    return reliability_config_loader.get_service_specific_configs(service)