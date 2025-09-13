"""
Resilient API Call Utilities with Tenacity and Circuit Breakers

Provides tenacity-based retry decorators with circuit breaker integration
and structured telemetry for all external API calls.
"""
import time
import functools
from typing import Any, Callable, Optional, Union, List, Type
from tenacity import (
    Retrying, 
    retry_if_exception_type, 
    retry_if_exception,
    stop_after_attempt, 
    wait_exponential,
    before_sleep_log,
    after_log,
    retry_if_result,
    retry_any,
    retry_if_not_exception_type
)
from tenacity.stop import stop_base
from tenacity.wait import wait_base
import logging

from openai import OpenAIError, RateLimitError, APITimeoutError, APIConnectionError
from core.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig, CircuitBreakerOpenException
from utils.reliability_config import get_retry_config, get_circuit_breaker_config
from utils.retry_telemetry import retry_telemetry, RetryEventType


def create_openai_retry_decorator(service_name: str = 'openai') -> Callable:
    """
    Create tenacity retry decorator for OpenAI API calls.
    
    Args:
        service_name: Name of service for config and telemetry
        
    Returns:
        Configured retry decorator
    """
    retry_config = get_retry_config(service_name)
    cb_config = get_circuit_breaker_config(service_name)
    
    # Create circuit breaker
    circuit_breaker = get_circuit_breaker(
        service_name, 
        CircuitBreakerConfig(
            service_name=service_name,
            failure_threshold=cb_config.failure_threshold,
            recovery_timeout=cb_config.recovery_timeout,
            success_threshold=cb_config.success_threshold,
            timeout_counts_as_failure=cb_config.timeout_counts_as_failure,
            max_failure_history=cb_config.max_failure_history
        )
    )
    
    # Define what exceptions to retry on
    retry_exceptions = (
        RateLimitError,        # 429 rate limit
        APITimeoutError,       # Request timeout
        APIConnectionError,    # Connection issues
    )
    
    # Also retry on 5xx server errors
    def should_retry_openai_error(exception):
        if isinstance(exception, OpenAIError):
            status_code = getattr(exception, 'status_code', None)
            if status_code and 500 <= status_code < 600:
                return True
            # Also check if it's in our configured retry status codes
            if hasattr(retry_config, 'retry_on_status') and retry_config.retry_on_status:
                return status_code in retry_config.retry_on_status
        return False
    
    # Custom retry condition that combines exception types and predicate
    def should_retry_exception(exception):
        # First check if it's in the direct retry exceptions
        if isinstance(exception, retry_exceptions):
            return True
        # Then check if it's an OpenAI error that should be retried
        return should_retry_openai_error(exception)
    
    retry_condition = retry_if_exception(should_retry_exception)
    
    def retry_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Track timing for telemetry
            start_time = time.time()
            attempt_count = 0
            
            # Before sleep callback with telemetry
            def before_sleep(retry_state):
                nonlocal attempt_count
                attempt_count += 1
                duration_ms = (time.time() - start_time) * 1000
                
                # Record retry attempt in telemetry
                retry_telemetry.record_retry_attempt(
                    service=service_name,
                    attempt=attempt_count,
                    total_attempts=retry_config.max_attempts,
                    error=retry_state.outcome.exception() if retry_state.outcome and retry_state.outcome.failed else None,
                    duration_ms=duration_ms,
                    backoff_duration=retry_state.next_action.sleep if hasattr(retry_state.next_action, 'sleep') else None
                )
                
                logging.getLogger(service_name).warning(
                    f"Retrying {func.__name__} attempt {attempt_count}/{retry_config.max_attempts} "
                    f"after {duration_ms:.1f}ms due to: {retry_state.outcome.exception()}"
                )
            
            # After attempt callback
            def after_attempt(retry_state):
                if not retry_state.outcome.failed:
                    duration_ms = (time.time() - start_time) * 1000
                    retry_telemetry.record_retry_attempt(
                        service=service_name,
                        attempt=attempt_count + 1,
                        total_attempts=retry_config.max_attempts,
                        error=None,
                        duration_ms=duration_ms
                    )
            
            # Configure retrying with circuit breaker integration
            retryer = Retrying(
                retry=retry_condition,
                stop=stop_after_attempt(retry_config.max_attempts),
                wait=wait_exponential(
                    multiplier=retry_config.initial_wait,
                    max=retry_config.max_wait,
                    exp_base=retry_config.exponential_base
                ),
                before_sleep=before_sleep,
                reraise=True
            )
            
            try:
                # Execute through circuit breaker
                def circuit_wrapped_call():
                    # Use retryer for the actual API call
                    for attempt in retryer:
                        with attempt:
                            result = func(*args, **kwargs)
                            after_attempt(attempt)  # Record success
                            return result
                    
                    # If we get here, retries exhausted
                    raise Exception(f"Max retries ({retry_config.max_attempts}) exceeded for {service_name}")
                
                return circuit_breaker.call(circuit_wrapped_call)
                
            except CircuitBreakerOpenException:
                # Circuit is open, record the event
                retry_telemetry.record_circuit_breaker_event(
                    service=service_name,
                    event_type=RetryEventType.CIRCUIT_OPEN
                )
                raise
            
            except Exception as e:
                # Record final failure
                duration_ms = (time.time() - start_time) * 1000
                retry_telemetry.record_retry_attempt(
                    service=service_name,
                    attempt=retry_config.max_attempts,
                    total_attempts=retry_config.max_attempts,
                    error=e,
                    duration_ms=duration_ms
                )
                raise
        
        return wrapper
    return retry_decorator


def create_huggingface_retry_decorator(service_name: str = 'huggingface') -> Callable:
    """
    Create tenacity retry decorator for HuggingFace API calls.
    
    Args:
        service_name: Name of service for config and telemetry
        
    Returns:
        Configured retry decorator
    """
    retry_config = get_retry_config(service_name)
    cb_config = get_circuit_breaker_config(service_name)
    
    # Create circuit breaker
    circuit_breaker = get_circuit_breaker(
        service_name,
        CircuitBreakerConfig(
            service_name=service_name,
            failure_threshold=cb_config.failure_threshold,
            recovery_timeout=cb_config.recovery_timeout,
            success_threshold=cb_config.success_threshold,
            timeout_counts_as_failure=cb_config.timeout_counts_as_failure,
            max_failure_history=cb_config.max_failure_history
        )
    )
    
    # HuggingFace-specific retry conditions
    def should_retry_hf_exception(exception):
        # Connection errors
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return True
        
        # HTTP errors with specific status codes
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            status_code = exception.response.status_code
            if status_code in [429, 500, 502, 503, 504]:
                return True
            if hasattr(retry_config, 'retry_on_status') and retry_config.retry_on_status:
                return status_code in retry_config.retry_on_status
        
        # Common network-related exceptions
        exception_types = (OSError, IOError, ConnectionResetError)
        return isinstance(exception, exception_types)
    
    def retry_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            attempt_count = 0
            
            def before_sleep(retry_state):
                nonlocal attempt_count
                attempt_count += 1
                duration_ms = (time.time() - start_time) * 1000
                
                retry_telemetry.record_retry_attempt(
                    service=service_name,
                    attempt=attempt_count,
                    total_attempts=retry_config.max_attempts,
                    error=retry_state.outcome.exception() if retry_state.outcome and retry_state.outcome.failed else None,
                    duration_ms=duration_ms,
                    backoff_duration=retry_state.next_action.sleep if hasattr(retry_state.next_action, 'sleep') else None
                )
                
                logging.getLogger(service_name).warning(
                    f"Retrying {func.__name__} attempt {attempt_count}/{retry_config.max_attempts}"
                )
            
            retryer = Retrying(
                retry=retry_if_exception(should_retry_hf_exception),
                stop=stop_after_attempt(retry_config.max_attempts),
                wait=wait_exponential(
                    multiplier=retry_config.initial_wait,
                    max=retry_config.max_wait,
                    exp_base=retry_config.exponential_base
                ),
                before_sleep=before_sleep,
                reraise=True
            )
            
            try:
                def circuit_wrapped_call():
                    for attempt in retryer:
                        with attempt:
                            result = func(*args, **kwargs)
                            # Record success
                            duration_ms = (time.time() - start_time) * 1000
                            retry_telemetry.record_retry_attempt(
                                service=service_name,
                                attempt=attempt_count + 1,
                                total_attempts=retry_config.max_attempts,
                                error=None,
                                duration_ms=duration_ms
                            )
                            return result
                    
                    raise Exception(f"Max retries ({retry_config.max_attempts}) exceeded for {service_name}")
                
                return circuit_breaker.call(circuit_wrapped_call)
                
            except CircuitBreakerOpenException:
                retry_telemetry.record_circuit_breaker_event(
                    service=service_name,
                    event_type=RetryEventType.CIRCUIT_OPEN
                )
                raise
            
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                retry_telemetry.record_retry_attempt(
                    service=service_name,
                    attempt=retry_config.max_attempts,
                    total_attempts=retry_config.max_attempts,
                    error=e,
                    duration_ms=duration_ms
                )
                raise
        
        return wrapper
    return retry_decorator


def create_subprocess_retry_decorator(service_name: str = 'subprocess') -> Callable:
    """
    Create tenacity retry decorator for subprocess calls (FFmpeg, etc.).
    
    Args:
        service_name: Name of service for config and telemetry
        
    Returns:
        Configured retry decorator  
    """
    retry_config = get_retry_config(service_name)
    cb_config = get_circuit_breaker_config('ffmpeg')  # Use ffmpeg config for subprocess
    
    circuit_breaker = get_circuit_breaker(
        service_name,
        CircuitBreakerConfig(
            service_name=service_name,
            failure_threshold=cb_config.failure_threshold,
            recovery_timeout=cb_config.recovery_timeout,
            success_threshold=cb_config.success_threshold,
            timeout_counts_as_failure=cb_config.timeout_counts_as_failure,
            max_failure_history=cb_config.max_failure_history
        )
    )
    
    # Subprocess-specific retry conditions
    def should_retry_subprocess_error(exception):
        import subprocess
        
        # Specific subprocess errors that are retriable
        if isinstance(exception, subprocess.TimeoutExpired):
            return True
        if isinstance(exception, subprocess.CalledProcessError):
            # Retry certain exit codes that might be transient
            return exception.returncode in [1, 2, 124, 125, 126, 127]  # Common transient errors
        if isinstance(exception, (OSError, FileNotFoundError)):
            return True
            
        return False
    
    def retry_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            attempt_count = 0
            
            def before_sleep(retry_state):
                nonlocal attempt_count
                attempt_count += 1
                duration_ms = (time.time() - start_time) * 1000
                
                retry_telemetry.record_retry_attempt(
                    service=service_name,
                    attempt=attempt_count,
                    total_attempts=retry_config.max_attempts,
                    error=retry_state.outcome.exception() if retry_state.outcome and retry_state.outcome.failed else None,
                    duration_ms=duration_ms,
                    backoff_duration=retry_state.next_action.sleep if hasattr(retry_state.next_action, 'sleep') else None
                )
                
                logging.getLogger(service_name).warning(
                    f"Retrying {func.__name__} attempt {attempt_count}/{retry_config.max_attempts}"
                )
            
            retryer = Retrying(
                retry=retry_if_exception(should_retry_subprocess_error),
                stop=stop_after_attempt(retry_config.max_attempts),
                wait=wait_exponential(
                    multiplier=retry_config.initial_wait,
                    max=retry_config.max_wait,
                    exp_base=retry_config.exponential_base
                ),
                before_sleep=before_sleep,
                reraise=True
            )
            
            try:
                def circuit_wrapped_call():
                    for attempt in retryer:
                        with attempt:
                            result = func(*args, **kwargs)
                            duration_ms = (time.time() - start_time) * 1000
                            retry_telemetry.record_retry_attempt(
                                service=service_name,
                                attempt=attempt_count + 1,
                                total_attempts=retry_config.max_attempts,
                                error=None,
                                duration_ms=duration_ms
                            )
                            return result
                    
                    raise Exception(f"Max retries ({retry_config.max_attempts}) exceeded for {service_name}")
                
                return circuit_breaker.call(circuit_wrapped_call)
                
            except CircuitBreakerOpenException:
                retry_telemetry.record_circuit_breaker_event(
                    service=service_name,
                    event_type=RetryEventType.CIRCUIT_OPEN
                )
                raise
            
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                retry_telemetry.record_retry_attempt(
                    service=service_name,
                    attempt=retry_config.max_attempts,
                    total_attempts=retry_config.max_attempts,
                    error=e,
                    duration_ms=duration_ms
                )
                raise
        
        return wrapper
    return retry_decorator


# Pre-configured decorators for common services
openai_retry = create_openai_retry_decorator()
huggingface_retry = create_huggingface_retry_decorator()
subprocess_retry = create_subprocess_retry_decorator()


def get_circuit_breaker_status(service_name: str) -> dict:
    """Get circuit breaker status for a service"""
    breaker = get_circuit_breaker(service_name)
    if breaker:
        return breaker.get_status()
    return {'service': service_name, 'status': 'not_found'}


def reset_circuit_breaker(service_name: str) -> bool:
    """Reset a circuit breaker manually"""
    breaker = get_circuit_breaker(service_name)
    if breaker:
        breaker.reset()
        retry_telemetry.record_circuit_breaker_event(
            service=service_name,
            event_type=RetryEventType.CIRCUIT_CLOSED
        )
        return True
    return False