"""
Structured Retry Telemetry and Audit Trail

Provides comprehensive telemetry for retry events, circuit breaker state changes,
and service reliability metrics. Creates detailed audit trails for debugging
and optimization purposes.
"""
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import threading
from enum import Enum

from utils.structured_logger import StructuredLogger


class RetryEventType(Enum):
    """Types of retry events"""
    ATTEMPT = "attempt"
    SUCCESS = "success"
    FAILURE = "failure"
    EXHAUSTED = "exhausted"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSED = "circuit_closed"
    CIRCUIT_HALF_OPEN = "circuit_half_open"


@dataclass
class RetryEvent:
    """Individual retry event record"""
    timestamp: float
    service: str
    event_type: RetryEventType
    attempt_number: int
    total_attempts: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    backoff_duration: Optional[float] = None
    http_status_code: Optional[int] = None
    api_endpoint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return data


@dataclass
class ServiceMetrics:
    """Aggregated metrics for a service"""
    service_name: str
    total_attempts: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    circuit_trips: int = 0
    total_retry_time: float = 0.0
    max_attempts_used: int = 0
    common_errors: Optional[Dict[str, int]] = None
    avg_success_duration: float = 0.0
    
    def __post_init__(self):
        if self.common_errors is None:
            self.common_errors = {}


class RetryTelemetryCollector:
    """
    Collects and manages retry telemetry data.
    
    Provides both real-time metrics and persistent audit trails for
    debugging and performance optimization.
    """
    
    def __init__(self, audit_file_path: Optional[str] = None):
        self.events: List[RetryEvent] = []
        self.service_metrics: Dict[str, ServiceMetrics] = defaultdict(
            lambda: ServiceMetrics(service_name="unknown")
        )
        
        # Thread safety for concurrent access
        self._lock = threading.RLock()
        
        # Audit trail persistence
        self.audit_file_path = audit_file_path or "logs/retry_audit.jsonl"
        self._ensure_audit_directory()
        
        # Structured logging
        self.structured_logger = StructuredLogger("retry_telemetry")
        
        # Session tracking
        self.session_id = str(int(time.time()))
        self.session_start = time.time()
    
    def _ensure_audit_directory(self):
        """Ensure audit log directory exists"""
        audit_path = Path(self.audit_file_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
    
    def record_retry_attempt(self, service: str, attempt: int, total_attempts: int,
                           error: Optional[Exception] = None, duration_ms: Optional[float] = None,
                           backoff_duration: Optional[float] = None, http_status: Optional[int] = None,
                           endpoint: Optional[str] = None):
        """
        Record a retry attempt.
        
        Args:
            service: Service name (e.g., 'openai', 'huggingface')
            attempt: Current attempt number (1-based)
            total_attempts: Total max attempts allowed
            error: Exception that caused the retry (if any)
            duration_ms: Duration of the attempt in milliseconds
            backoff_duration: Backoff delay before next retry
            http_status: HTTP status code (if applicable)
            endpoint: API endpoint being called (if applicable)
        """
        with self._lock:
            # Determine event type
            if attempt == 1 and error is None:
                event_type = RetryEventType.SUCCESS
            elif error is not None and attempt < total_attempts:
                event_type = RetryEventType.ATTEMPT
            elif error is not None and attempt >= total_attempts:
                event_type = RetryEventType.EXHAUSTED
            else:
                event_type = RetryEventType.SUCCESS
            
            # Create event record
            event = RetryEvent(
                timestamp=time.time(),
                service=service,
                event_type=event_type,
                attempt_number=attempt,
                total_attempts=total_attempts,
                error_type=type(error).__name__ if error else None,
                error_message=str(error) if error else None,
                duration_ms=duration_ms,
                backoff_duration=backoff_duration,
                http_status_code=http_status,
                api_endpoint=endpoint
            )
            
            # Store event
            self.events.append(event)
            
            # Update service metrics
            self._update_service_metrics(service, event, error)
            
            # Log structured event
            self.structured_logger.info(
                f"Retry {event_type.value} for {service}",
                context={
                    'service': service,
                    'event_type': event_type.value,
                    'attempt': attempt,
                    'total_attempts': total_attempts,
                    'duration_ms': duration_ms,
                    'error_type': event.error_type,
                    'http_status': http_status,
                    'endpoint': endpoint
                }
            )
            
            # Persist to audit file
            self._persist_event(event)
    
    def record_circuit_breaker_event(self, service: str, event_type: RetryEventType,
                                   failure_count: Optional[int] = None,
                                   recovery_time: Optional[float] = None):
        """
        Record circuit breaker state change.
        
        Args:
            service: Service name
            event_type: Type of circuit breaker event
            failure_count: Number of failures that triggered the event
            recovery_time: Time until recovery (for OPEN events)
        """
        with self._lock:
            event = RetryEvent(
                timestamp=time.time(),
                service=service,
                event_type=event_type,
                attempt_number=failure_count or 0,
                total_attempts=0,
                error_message=f"Circuit breaker {event_type.value}, recovery_time={recovery_time}"
            )
            
            self.events.append(event)
            
            # Update circuit breaker metrics
            if event_type in [RetryEventType.CIRCUIT_OPEN, RetryEventType.CIRCUIT_CLOSED, 
                             RetryEventType.CIRCUIT_HALF_OPEN]:
                metrics = self.service_metrics[service]
                if event_type == RetryEventType.CIRCUIT_OPEN:
                    metrics.circuit_trips += 1
            
            # Log circuit breaker event
            self.structured_logger.warning(
                f"Circuit breaker {event_type.value} for {service}",
                context={
                    'service': service,
                    'event_type': event_type.value,
                    'failure_count': failure_count,
                    'recovery_time': recovery_time
                }
            )
            
            # Persist to audit file
            self._persist_event(event)
    
    def _update_service_metrics(self, service: str, event: RetryEvent, error: Optional[Exception]):
        """Update aggregated service metrics"""
        metrics = self.service_metrics[service]
        metrics.service_name = service
        metrics.total_attempts += 1
        
        if event.event_type == RetryEventType.SUCCESS:
            metrics.successful_calls += 1
            if event.duration_ms:
                # Update average success duration
                total_success_time = metrics.avg_success_duration * (metrics.successful_calls - 1)
                metrics.avg_success_duration = (total_success_time + event.duration_ms) / metrics.successful_calls
        elif event.event_type in [RetryEventType.FAILURE, RetryEventType.EXHAUSTED]:
            metrics.failed_calls += 1
        
        if event.backoff_duration:
            metrics.total_retry_time += event.backoff_duration
        
        metrics.max_attempts_used = max(metrics.max_attempts_used, event.attempt_number)
        
        # Track common errors
        if event.error_type:
            if metrics.common_errors is None:
                metrics.common_errors = {}
            metrics.common_errors[event.error_type] = metrics.common_errors.get(event.error_type, 0) + 1
    
    def _persist_event(self, event: RetryEvent):
        """Persist event to audit file"""
        try:
            with open(self.audit_file_path, 'a') as f:
                # Add session info to each event
                event_data = event.to_dict()
                event_data['session_id'] = self.session_id
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            self.structured_logger.error(f"Failed to persist retry event: {e}")
    
    def get_service_metrics(self, service: str) -> Optional[ServiceMetrics]:
        """Get metrics for a specific service"""
        with self._lock:
            return self.service_metrics.get(service)
    
    def get_all_metrics(self) -> Dict[str, ServiceMetrics]:
        """Get metrics for all services"""
        with self._lock:
            return dict(self.service_metrics)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        with self._lock:
            session_duration = time.time() - self.session_start
            
            # Calculate overall stats
            total_events = len(self.events)
            retry_events = len([e for e in self.events if e.event_type == RetryEventType.ATTEMPT])
            success_events = len([e for e in self.events if e.event_type == RetryEventType.SUCCESS])
            failure_events = len([e for e in self.events if e.event_type == RetryEventType.EXHAUSTED])
            circuit_events = len([e for e in self.events if e.event_type.value.startswith('circuit')])
            
            return {
                'session_id': self.session_id,
                'session_duration': session_duration,
                'total_events': total_events,
                'retry_events': retry_events,
                'success_events': success_events,
                'failure_events': failure_events,
                'circuit_events': circuit_events,
                'services_used': list(self.service_metrics.keys()),
                'service_metrics': {
                    name: asdict(metrics) for name, metrics in self.service_metrics.items()
                }
            }
    
    def create_audit_report(self, output_path: Optional[str] = None) -> str:
        """
        Create comprehensive audit report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Path to generated report
        """
        report_path = output_path or f"logs/retry_audit_report_{self.session_id}.json"
        
        with self._lock:
            report = {
                'metadata': {
                    'generated_at': time.time(),
                    'session_id': self.session_id,
                    'report_type': 'retry_audit_report',
                    'version': '1.0'
                },
                'session_summary': self.get_session_summary(),
                'events': [event.to_dict() for event in self.events],
                'service_analysis': self._analyze_services(),
                'recommendations': self._generate_recommendations()
            }
        
        # Save report
        report_path_obj = Path(report_path)
        report_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.structured_logger.info(
            f"Retry audit report generated: {report_path}",
            context={'report_path': report_path, 'session_id': self.session_id}
        )
        
        return report_path
    
    def _analyze_services(self) -> Dict[str, Any]:
        """Analyze service reliability patterns"""
        analysis = {}
        
        for service, metrics in self.service_metrics.items():
            if metrics.total_attempts == 0:
                continue
                
            success_rate = metrics.successful_calls / metrics.total_attempts
            avg_retries = (metrics.total_attempts - metrics.successful_calls - metrics.failed_calls) / max(1, metrics.successful_calls + metrics.failed_calls)
            
            analysis[service] = {
                'success_rate': success_rate,
                'average_retries_per_call': avg_retries,
                'most_common_error': max(metrics.common_errors.items(), key=lambda x: x[1]) if metrics.common_errors else None,
                'circuit_trips': metrics.circuit_trips,
                'avg_success_duration_ms': metrics.avg_success_duration,
                'total_retry_time_seconds': metrics.total_retry_time,
                'reliability_score': self._calculate_reliability_score(metrics)
            }
        
        return analysis
    
    def _calculate_reliability_score(self, metrics: ServiceMetrics) -> float:
        """Calculate a reliability score (0-100) for a service"""
        if metrics.total_attempts == 0:
            return 100.0
        
        success_rate = metrics.successful_calls / metrics.total_attempts
        circuit_penalty = min(metrics.circuit_trips * 10, 30)  # Max 30 point penalty
        retry_penalty = min((metrics.total_attempts - metrics.successful_calls - metrics.failed_calls) / metrics.total_attempts * 20, 20)
        
        score = (success_rate * 100) - circuit_penalty - retry_penalty
        return max(0.0, score)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on retry patterns"""
        recommendations = []
        
        for service, metrics in self.service_metrics.items():
            if metrics.total_attempts == 0:
                continue
            
            success_rate = metrics.successful_calls / metrics.total_attempts
            
            if success_rate < 0.95:
                recommendations.append(
                    f"Service '{service}' has low success rate ({success_rate:.1%}). "
                    f"Consider investigating service reliability or adjusting retry parameters."
                )
            
            if metrics.circuit_trips > 0:
                recommendations.append(
                    f"Service '{service}' had {metrics.circuit_trips} circuit breaker trips. "
                    f"Monitor service health and consider adjusting circuit breaker thresholds."
                )
            
            if metrics.total_retry_time > 60:  # More than 1 minute total retry time
                recommendations.append(
                    f"Service '{service}' spent {metrics.total_retry_time:.1f}s in retries. "
                    f"Consider optimizing retry intervals or investigating root cause."
                )
            
            if metrics.max_attempts_used >= 4:  # Using most of retry attempts
                recommendations.append(
                    f"Service '{service}' frequently uses maximum retry attempts. "
                    f"Consider increasing retry limits or improving error handling."
                )
        
        return recommendations
    
    def clear_session(self):
        """Clear current session data and start new session"""
        with self._lock:
            self.events.clear()
            self.service_metrics.clear()
            self.session_id = str(int(time.time()))
            self.session_start = time.time()
            
            self.structured_logger.info(f"Started new telemetry session: {self.session_id}")


# Global telemetry collector instance
retry_telemetry = RetryTelemetryCollector()


def record_retry_attempt(service: str, attempt: int, total_attempts: int, **kwargs):
    """Convenience function to record retry attempt"""
    retry_telemetry.record_retry_attempt(service, attempt, total_attempts, **kwargs)


def record_circuit_breaker_event(service: str, event_type: RetryEventType, **kwargs):
    """Convenience function to record circuit breaker event"""
    retry_telemetry.record_circuit_breaker_event(service, event_type, **kwargs)


def get_service_metrics(service: str) -> Optional[ServiceMetrics]:
    """Get metrics for a service"""
    return retry_telemetry.get_service_metrics(service)


def create_audit_report(output_path: Optional[str] = None) -> str:
    """Create audit report"""
    return retry_telemetry.create_audit_report(output_path)