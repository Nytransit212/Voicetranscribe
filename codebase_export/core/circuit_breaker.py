"""
Circuit Breaker Implementation for External Service Reliability

Provides circuit breaker patterns to prevent cascade failures when external services
(OpenAI, HuggingFace, etc.) become unavailable. Includes configurable thresholds,
automatic recovery, and structured telemetry.
"""
import time
import threading
from typing import Dict, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass
from collections import deque
import json

from utils.structured_logger import StructuredLogger


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Service unavailable, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    service_name: str
    failure_threshold: int = 5           # Consecutive failures to trip breaker
    recovery_timeout: int = 60           # Seconds before allowing test calls
    success_threshold: int = 2           # Successes needed to close breaker
    timeout_counts_as_failure: bool = True
    max_failure_history: int = 100       # Max failure events to track


@dataclass
class FailureEvent:
    """Record of a service failure"""
    timestamp: float
    error_type: str
    error_message: str
    duration: Optional[float] = None


class CircuitBreaker:
    """
    Circuit breaker implementation with automatic recovery and telemetry.
    
    Prevents cascade failures by temporarily blocking calls to failing services
    and providing graceful fallback behavior.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.failure_history: deque = deque(maxlen=config.max_failure_history)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Structured logging
        self.structured_logger = StructuredLogger(f"circuit_breaker_{config.service_name}")
        
        # State transition callbacks
        self._on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    
    def set_state_change_callback(self, callback: Callable[[CircuitState, CircuitState], None]):
        """Set callback for state transitions"""
        self._on_state_change = callback
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for circuit breaker"""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result if successful
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Original exception: If function fails and circuit remains closed
        """
        with self._lock:
            current_state = self.state
            
            # Check if circuit should allow the call
            if not self._should_allow_call():
                error_msg = f"Circuit breaker OPEN for {self.config.service_name}"
                self.structured_logger.error(
                    "Circuit breaker blocking call",
                    context={
                        'service': self.config.service_name,
                        'state': self.state.value,
                        'failure_count': self.failure_count,
                        'time_since_last_failure': time.time() - self.last_failure_time
                    }
                )
                raise CircuitBreakerOpenException(error_msg)
        
        # Execute the function with timing
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record success
            self._on_success(duration)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failure
            self._on_failure(e, duration)
            raise
    
    def _should_allow_call(self) -> bool:
        """Determine if circuit should allow the call based on current state"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self._transition_to_half_open()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        return False
    
    def _on_success(self, duration: float):
        """Handle successful function execution"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                # Check if we should close the circuit
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                    
                self.structured_logger.info(
                    "Circuit breaker success in half-open state",
                    context={
                        'service': self.config.service_name,
                        'success_count': self.success_count,
                        'duration': duration,
                        'required_successes': self.config.success_threshold
                    }
                )
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                if self.failure_count > 0:
                    self.structured_logger.info(
                        "Circuit breaker recovered from failures",
                        context={
                            'service': self.config.service_name,
                            'previous_failure_count': self.failure_count,
                            'duration': duration
                        }
                    )
                    self.failure_count = 0
    
    def _on_failure(self, error: Exception, duration: float):
        """Handle failed function execution"""
        with self._lock:
            # Record failure event
            failure_event = FailureEvent(
                timestamp=time.time(),
                error_type=type(error).__name__,
                error_message=str(error),
                duration=duration
            )
            self.failure_history.append(failure_event)
            
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Log failure
            self.structured_logger.error(
                "Circuit breaker recorded failure",
                context={
                    'service': self.config.service_name,
                    'error_type': failure_event.error_type,
                    'error_message': failure_event.error_message,
                    'failure_count': self.failure_count,
                    'duration': duration,
                    'threshold': self.config.failure_threshold
                }
            )
            
            # Check if we should trip the breaker
            if self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state transitions back to open
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.success_count = 0
        
        self.structured_logger.error(
            "Circuit breaker OPENED",
            context={
                'service': self.config.service_name,
                'previous_state': old_state.value,
                'failure_count': self.failure_count,
                'recovery_timeout': self.config.recovery_timeout
            }
        )
        
        if self._on_state_change:
            self._on_state_change(old_state, self.state)
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        
        self.structured_logger.info(
            "Circuit breaker HALF-OPEN - testing service recovery",
            context={
                'service': self.config.service_name,
                'previous_state': old_state.value,
                'time_since_last_failure': time.time() - self.last_failure_time
            }
        )
        
        if self._on_state_change:
            self._on_state_change(old_state, self.state)
    
    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        
        self.structured_logger.info(
            "Circuit breaker CLOSED - service recovered",
            context={
                'service': self.config.service_name,
                'previous_state': old_state.value,
                'recovery_duration': time.time() - self.last_failure_time
            }
        )
        
        if self._on_state_change:
            self._on_state_change(old_state, self.state)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status for monitoring"""
        with self._lock:
            return {
                'service': self.config.service_name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'time_since_last_failure': time.time() - self.last_failure_time if self.last_failure_time > 0 else None,
                'is_allowing_calls': self._should_allow_call(),
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold
                },
                'recent_failures': [
                    {
                        'timestamp': f.timestamp,
                        'error_type': f.error_type,
                        'error_message': f.error_message,
                        'duration': f.duration
                    }
                    for f in list(self.failure_history)[-5:]  # Last 5 failures
                ]
            }
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.failure_history.clear()
            
            self.structured_logger.info(
                "Circuit breaker manually reset",
                context={
                    'service': self.config.service_name,
                    'previous_state': old_state.value
                }
            )
            
            if self._on_state_change:
                self._on_state_change(old_state, self.state)


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services.
    
    Provides centralized configuration and monitoring for all circuit breakers
    in the system.
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        self.structured_logger = StructuredLogger("circuit_breaker_manager")
    
    def register_breaker(self, service_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Register a circuit breaker for a service.
        
        Args:
            service_name: Unique service identifier
            config: Optional custom configuration
            
        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if service_name in self.breakers:
                return self.breakers[service_name]
            
            if config is None:
                config = CircuitBreakerConfig(service_name=service_name)
            
            breaker = CircuitBreaker(config)
            breaker.set_state_change_callback(self._on_breaker_state_change)
            
            self.breakers[service_name] = breaker
            
            self.structured_logger.info(
                "Circuit breaker registered",
                context={
                    'service': service_name,
                    'failure_threshold': config.failure_threshold,
                    'recovery_timeout': config.recovery_timeout
                }
            )
            
            return breaker
    
    def get_breaker(self, service_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a service"""
        with self._lock:
            return self.breakers.get(service_name)
    
    def _on_breaker_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Handle circuit breaker state changes"""
        # This could be extended to trigger alerts, metrics, etc.
        pass
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        with self._lock:
            return {
                'breakers': {
                    name: breaker.get_status()
                    for name, breaker in self.breakers.items()
                },
                'total_breakers': len(self.breakers),
                'open_breakers': [
                    name for name, breaker in self.breakers.items()
                    if breaker.state == CircuitState.OPEN
                ],
                'timestamp': time.time()
            }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self.breakers.values():
                breaker.reset()
            
            self.structured_logger.info("All circuit breakers reset")


# Global circuit breaker manager instance
circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker(service_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a service.
    
    Convenience function for accessing circuit breakers throughout the application.
    
    Args:
        service_name: Service identifier
        config: Optional custom configuration
        
    Returns:
        Circuit breaker instance
    """
    return circuit_breaker_manager.register_breaker(service_name, config)