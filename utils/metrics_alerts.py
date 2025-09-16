"""
Comprehensive metrics and alerts system for the Advanced Ensemble Transcription System.

This module provides:
- Structured logging with JSON format for parsing and analysis  
- Real-time metrics aggregation and threshold-based alerting
- CI/CD integration for build-time problem detection
- Performance regression detection and trend analysis
- Business metrics tracking and monitoring

Key Features:
- Thread-safe metrics collection for concurrent processing
- Integration with existing logging and telemetry systems
- Configurable alert thresholds and destinations
- Automated quality gate integration for CI pipelines
- Historical metrics comparison and trend analysis
"""

import os
import json
import time
import uuid
import threading
import statistics
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps

import orjson
import psutil
from loguru import logger

from .observability import get_observability_manager, EnhancedLogger


class AlertSeverity(Enum):
    """Alert severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Types of metrics being collected"""
    COUNTER = "counter"
    GAUGE = "gauge"  
    HISTOGRAM = "histogram"
    TIMER = "timer"
    PERCENTAGE = "percentage"


@dataclass
class MetricDefinition:
    """Definition of a metric including metadata and thresholds"""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    category: str  # performance, quality, system_health, business
    alert_thresholds: Dict[str, float]  # severity -> threshold value
    aggregation_method: str = "avg"  # avg, sum, max, min, p95, p99
    collection_interval: int = 30  # seconds
    retention_days: int = 30
    tags: Optional[Dict[str, str]] = None


@dataclass
class MetricValue:
    """Individual metric measurement"""
    name: str
    value: Union[float, int]
    timestamp: datetime
    tags: Dict[str, str]
    session_id: str
    component: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    """Alert generated from metric threshold violations"""
    id: str
    metric_name: str
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    session_id: str
    component: str
    context: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricAggregator:
    """Thread-safe metric aggregation with configurable windows"""
    
    def __init__(self, window_size: int = 300):  # 5 minute default window
        self.window_size = window_size
        self._data = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 points
        self._lock = threading.RLock()
    
    def add_value(self, metric_name: str, value: float, timestamp: datetime = None):
        """Add a metric value to the aggregation window"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        with self._lock:
            self._data[metric_name].append((timestamp, value))
    
    def get_aggregated_value(self, metric_name: str, method: str = "avg") -> Optional[float]:
        """Get aggregated value for a metric using specified method"""
        with self._lock:
            values = self._get_values_in_window(metric_name)
            if not values:
                return None
                
            if method == "avg":
                return statistics.mean(values)
            elif method == "sum":
                return sum(values)
            elif method == "max":
                return max(values)
            elif method == "min":
                return min(values)
            elif method == "p95":
                return statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
            elif method == "p99":
                return statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
            else:
                return statistics.mean(values)  # Default to average
    
    def _get_values_in_window(self, metric_name: str) -> List[float]:
        """Get values within the current time window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.window_size)
        values = []
        
        for timestamp, value in self._data[metric_name]:
            if timestamp >= cutoff_time:
                values.append(value)
                
        return values
    
    def get_trend(self, metric_name: str, points: int = 10) -> Optional[str]:
        """Analyze trend direction for a metric"""
        with self._lock:
            if len(self._data[metric_name]) < points:
                return None
                
            recent_values = list(self._data[metric_name])[-points:]
            values = [v[1] for v in recent_values]
            
            if len(values) < 2:
                return "stable"
                
            # Simple trend analysis using linear regression slope
            x = list(range(len(values)))
            n = len(values)
            
            slope = (n * sum(x[i] * values[i] for i in range(n)) - sum(x) * sum(values)) / \
                   (n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2)
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"


class AlertManager:
    """Manages alert generation, routing, and resolution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._active_alerts = {}  # alert_id -> Alert
        self._alert_history = deque(maxlen=10000)  # Keep last 10k alerts
        self._lock = threading.RLock()
        self._observers = []  # List of alert observer functions
        
        # Initialize alert destinations
        self._setup_alert_destinations()
    
    def _setup_alert_destinations(self):
        """Setup alert routing to configured destinations"""
        self.destinations = self.config.get('alert_destinations', {})
        
        # Setup file logging for alerts
        if self.destinations.get('file_logging', {}).get('enabled', True):
            alert_log_path = Path(self.destinations.get('file_logging', {}).get('path', 'logs/alerts'))
            alert_log_path.mkdir(parents=True, exist_ok=True)
            
            # Create dedicated alert logger
            self.alert_logger = logger.bind(alert_context=True)
            logger.add(
                sink=alert_log_path / "alerts_{time:YYYY-MM-DD}.jsonl",
                format=lambda record: orjson.dumps({
                    "timestamp": record["time"].isoformat(),
                    "level": "ALERT",
                    "alert_data": record["extra"].get("alert_data", {}),
                    "message": record["message"]
                }).decode() + "\n",
                level="INFO",
                filter=lambda record: "alert_context" in record["extra"],
                rotation="1 day",
                retention="90 days",
                compression="gz"
            )
    
    def generate_alert(self, 
                      metric_name: str, 
                      current_value: float,
                      threshold_value: float,
                      severity: AlertSeverity,
                      component: str,
                      session_id: str,
                      context: Dict[str, Any] = None) -> Alert:
        """Generate a new alert"""
        
        alert_id = f"{metric_name}_{component}_{int(time.time())}"
        message = f"{metric_name} threshold violation: {current_value} > {threshold_value}"
        
        alert = Alert(
            id=alert_id,
            metric_name=metric_name,
            severity=severity,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            component=component,
            context=context or {}
        )
        
        with self._lock:
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
        
        # Route alert to configured destinations
        self._route_alert(alert)
        
        # Notify observers
        for observer in self._observers:
            try:
                observer(alert)
            except Exception as e:
                logger.warning(f"Alert observer failed: {e}")
        
        return alert
    
    def _route_alert(self, alert: Alert):
        """Route alert to configured destinations"""
        
        # Always log to structured alert log
        self.alert_logger.info(
            f"ALERT: {alert.message}",
            alert_data=asdict(alert)
        )
        
        # Console output for development
        if self.destinations.get('console', {}).get('enabled', True):
            severity_color = {
                AlertSeverity.DEBUG: "blue",
                AlertSeverity.INFO: "green", 
                AlertSeverity.WARNING: "yellow",
                AlertSeverity.ERROR: "red",
                AlertSeverity.CRITICAL: "magenta"
            }
            
            print(f"\n🚨 [{alert.severity.value}] {alert.message}")
            print(f"   Component: {alert.component} | Session: {alert.session_id}")
            if alert.context:
                print(f"   Context: {json.dumps(alert.context, indent=2)}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        with self._lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].resolved = True
                self._active_alerts[alert_id].resolved_at = datetime.now(timezone.utc)
                del self._active_alerts[alert_id]
                return True
        return False
    
    def add_observer(self, observer_func: Callable[[Alert], None]):
        """Add alert observer function"""
        self._observers.append(observer_func)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts"""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history"""
        with self._lock:
            return list(self._alert_history)[-limit:]


class MetricsCollector:
    """Main metrics collection system with structured logging and alerting"""
    
    def __init__(self, config: Dict[str, Any], session_id: str = None):
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.enabled = config.get('enabled', True)
        
        if not self.enabled:
            logger.info("Metrics collection disabled by configuration")
            return
            
        # Initialize components
        self.aggregator = MetricAggregator(
            window_size=config.get('aggregation_window_seconds', 300)
        )
        self.alert_manager = AlertManager(config.get('alerting', {}))
        
        # Metric definitions registry
        self.metric_definitions = self._initialize_metric_definitions()
        
        # Thread-safe storage for raw metrics
        self._raw_metrics = deque(maxlen=100000)  # Keep last 100k raw metrics
        self._metrics_lock = threading.RLock()
        
        # Background processing
        self._background_thread = None
        self._stop_event = threading.Event()
        
        # Enhanced logger for structured output
        obs_manager = get_observability_manager()
        self.logger = obs_manager.get_enhanced_logger("metrics_collector")
        
        # CI integration state
        self.ci_mode = os.getenv('CI', '').lower() in ('true', '1', 'yes')
        self.ci_metrics = defaultdict(list)  # Special storage for CI metrics
        
        # Start background processing if enabled
        if config.get('enable_background_processing', True):
            self.start_background_processing()
        
        logger.info("Metrics collector initialized",
                   session_id=self.session_id,
                   ci_mode=self.ci_mode,
                   metrics_enabled=self.enabled)
    
    def _initialize_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """Initialize comprehensive metric definitions"""
        
        definitions = {}
        
        # Performance Metrics
        definitions.update({
            'processing_time_per_minute': MetricDefinition(
                name='processing_time_per_minute',
                metric_type=MetricType.TIMER,
                description='Audio processing time per minute of input',
                unit='seconds',
                category='performance',
                alert_thresholds={
                    'WARNING': 90.0,   # 1.5x expected 
                    'CRITICAL': 120.0  # 2x expected
                }
            ),
            'rtf_ratio': MetricDefinition(
                name='rtf_ratio',
                metric_type=MetricType.GAUGE,
                description='Real-time factor ratio (processing_time / audio_duration)',
                unit='ratio',
                category='performance',
                alert_thresholds={
                    'WARNING': 1.5,
                    'CRITICAL': 2.0
                }
            ),
            'memory_usage_peak_mb': MetricDefinition(
                name='memory_usage_peak_mb',
                metric_type=MetricType.GAUGE,
                description='Peak memory usage during processing',
                unit='MB',
                category='performance',
                alert_thresholds={
                    'WARNING': 6144,   # 6GB
                    'CRITICAL': 8192   # 8GB
                }
            ),
            'cache_hit_rate': MetricDefinition(
                name='cache_hit_rate',
                metric_type=MetricType.PERCENTAGE,
                description='Cache hit rate percentage',
                unit='%',
                category='performance',
                alert_thresholds={
                    'WARNING': 60.0,   # Below 60% is concerning
                    'INFO': 80.0       # Above 80% is good
                }
            )
        })
        
        # Quality Metrics  
        definitions.update({
            'word_error_rate': MetricDefinition(
                name='word_error_rate',
                metric_type=MetricType.PERCENTAGE,
                description='Word Error Rate percentage',
                unit='%', 
                category='quality',
                alert_thresholds={
                    'WARNING': 3.0,
                    'CRITICAL': 5.0
                }
            ),
            'diarization_error_rate': MetricDefinition(
                name='diarization_error_rate',
                metric_type=MetricType.PERCENTAGE,
                description='Diarization Error Rate percentage',
                unit='%',
                category='quality',
                alert_thresholds={
                    'WARNING': 8.0,
                    'CRITICAL': 15.0
                }
            ),
            'entity_accuracy': MetricDefinition(
                name='entity_accuracy',
                metric_type=MetricType.PERCENTAGE,
                description='Named entity recognition accuracy',
                unit='%',
                category='quality',
                alert_thresholds={
                    'WARNING': 85.0,   # Below 85% needs attention
                    'INFO': 95.0       # Above 95% is excellent
                }
            ),
            'confidence_score_avg': MetricDefinition(
                name='confidence_score_avg',
                metric_type=MetricType.GAUGE,
                description='Average confidence score across all predictions',
                unit='score',
                category='quality',
                alert_thresholds={
                    'WARNING': 0.7,    # Below 70% confidence
                    'INFO': 0.9        # Above 90% confidence
                }
            )
        })
        
        # System Health Metrics
        definitions.update({
            'error_rate': MetricDefinition(
                name='error_rate',
                metric_type=MetricType.PERCENTAGE,
                description='Overall error rate percentage',
                unit='%',
                category='system_health',
                alert_thresholds={
                    'WARNING': 5.0,
                    'CRITICAL': 10.0
                }
            ),
            'timeout_count': MetricDefinition(
                name='timeout_count',
                metric_type=MetricType.COUNTER,
                description='Number of processing timeouts',
                unit='count',
                category='system_health',
                alert_thresholds={
                    'WARNING': 5,
                    'CRITICAL': 10
                }
            ),
            'fallback_activation_rate': MetricDefinition(
                name='fallback_activation_rate',
                metric_type=MetricType.PERCENTAGE,
                description='Rate of fallback mechanism activation',
                unit='%',
                category='system_health',
                alert_thresholds={
                    'WARNING': 20.0,
                    'CRITICAL': 40.0
                }
            ),
            'resource_utilization_cpu': MetricDefinition(
                name='resource_utilization_cpu',
                metric_type=MetricType.PERCENTAGE,
                description='CPU utilization percentage',
                unit='%',
                category='system_health',
                alert_thresholds={
                    'WARNING': 80.0,
                    'CRITICAL': 95.0
                }
            )
        })
        
        # Business Metrics
        definitions.update({
            'files_processed_per_hour': MetricDefinition(
                name='files_processed_per_hour',
                metric_type=MetricType.GAUGE,
                description='Number of files processed per hour',
                unit='files/hour',
                category='business',
                alert_thresholds={
                    'INFO': 10  # Good throughput indicator
                }
            ),
            'success_rate': MetricDefinition(
                name='success_rate',
                metric_type=MetricType.PERCENTAGE,
                description='Overall processing success rate',
                unit='%',
                category='business',
                alert_thresholds={
                    'WARNING': 90.0,
                    'CRITICAL': 85.0
                }
            ),
            'user_satisfaction_proxy': MetricDefinition(
                name='user_satisfaction_proxy',
                metric_type=MetricType.GAUGE,
                description='Proxy metric for user satisfaction based on quality scores',
                unit='score',
                category='business',
                alert_thresholds={
                    'WARNING': 7.0,    # On scale of 1-10
                    'INFO': 8.5        # High satisfaction
                }
            )
        })
        
        return definitions
    
    def record_metric(self, 
                     name: str, 
                     value: Union[float, int],
                     component: str,
                     tags: Dict[str, str] = None,
                     context: Dict[str, Any] = None):
        """Record a metric value with structured logging"""
        
        if not self.enabled:
            return
            
        timestamp = datetime.now(timezone.utc)
        
        # Create metric value object
        metric_value = MetricValue(
            name=name,
            value=float(value),
            timestamp=timestamp,
            tags=tags or {},
            session_id=self.session_id,
            component=component,
            context=context
        )
        
        # Store raw metric
        with self._metrics_lock:
            self._raw_metrics.append(metric_value)
        
        # Add to aggregator for threshold checking
        self.aggregator.add_value(name, float(value), timestamp)
        
        # Check for threshold violations
        self._check_thresholds(metric_value)
        
        # Structured logging for the metric
        self.logger.info(f"Metric recorded: {name}",
                        metric_name=name,
                        metric_value=value,
                        metric_category=self._get_metric_category(name),
                        component=component,
                        tags=tags,
                        context=context)
        
        # CI mode special handling
        if self.ci_mode:
            self.ci_metrics[name].append(metric_value)
    
    def _check_thresholds(self, metric_value: MetricValue):
        """Check metric against defined thresholds and generate alerts"""
        
        metric_def = self.metric_definitions.get(metric_value.name)
        if not metric_def:
            return  # No thresholds defined
        
        value = metric_value.value
        
        # Check each threshold level
        for severity_str, threshold in metric_def.alert_thresholds.items():
            severity = AlertSeverity(severity_str)
            
            # Different comparison logic based on metric type and severity
            should_alert = False
            
            if severity_str in ['WARNING', 'CRITICAL', 'ERROR']:
                # Higher values are bad for most metrics
                if metric_value.name in ['word_error_rate', 'diarization_error_rate', 'error_rate', 
                                       'timeout_count', 'fallback_activation_rate', 'processing_time_per_minute',
                                       'rtf_ratio', 'memory_usage_peak_mb', 'resource_utilization_cpu']:
                    should_alert = value > threshold
                # Lower values are bad for some metrics  
                elif metric_value.name in ['cache_hit_rate', 'entity_accuracy', 'confidence_score_avg',
                                         'success_rate', 'user_satisfaction_proxy']:
                    should_alert = value < threshold
            elif severity_str == 'INFO':
                # INFO alerts for positive achievements
                if metric_value.name in ['cache_hit_rate', 'entity_accuracy', 'confidence_score_avg',
                                       'files_processed_per_hour', 'user_satisfaction_proxy']:
                    should_alert = value > threshold
            
            if should_alert:
                self.alert_manager.generate_alert(
                    metric_name=metric_value.name,
                    current_value=value,
                    threshold_value=threshold,
                    severity=severity,
                    component=metric_value.component,
                    session_id=metric_value.session_id,
                    context=metric_value.context
                )
                break  # Only generate one alert per metric per check
    
    def _get_metric_category(self, metric_name: str) -> str:
        """Get category for a metric name"""
        metric_def = self.metric_definitions.get(metric_name)
        return metric_def.category if metric_def else 'unknown'
    
    @contextmanager
    def timing_context(self, metric_name: str, component: str, **kwargs):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(metric_name, duration, component, **kwargs)
    
    def record_processing_stage(self, 
                              stage: str, 
                              duration: float, 
                              component: str,
                              quality_metrics: Dict[str, float] = None,
                              **context):
        """Record metrics for a complete processing stage"""
        
        if not self.enabled:
            return
            
        # Record basic timing
        self.record_metric(f'stage_duration_{stage}', duration, component, context=context)
        
        # Record quality metrics if provided
        if quality_metrics:
            for metric_name, value in quality_metrics.items():
                self.record_metric(metric_name, value, component, context=context)
        
        # Calculate derived metrics
        if 'audio_duration' in context and context['audio_duration'] > 0:
            rtf = duration / context['audio_duration']
            self.record_metric('rtf_ratio', rtf, component, context=context)
            
            processing_time_per_minute = (duration / context['audio_duration']) * 60
            self.record_metric('processing_time_per_minute', processing_time_per_minute, component, context=context)
    
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary for dashboards"""
        
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': self.session_id,
            'metrics': {},
            'alerts': {
                'active_count': len(self.alert_manager.get_active_alerts()),
                'recent_alerts': [asdict(alert) for alert in self.alert_manager.get_alert_history(10)]
            }
        }
        
        # Get aggregated values for key metrics
        for name, definition in self.metric_definitions.items():
            aggregated_value = self.aggregator.get_aggregated_value(name, definition.aggregation_method)
            if aggregated_value is not None:
                trend = self.aggregator.get_trend(name)
                summary['metrics'][name] = {
                    'value': aggregated_value,
                    'unit': definition.unit,
                    'category': definition.category,
                    'trend': trend,
                    'thresholds': definition.alert_thresholds
                }
        
        return summary
    
    def start_background_processing(self):
        """Start background thread for metric processing and alerts"""
        if self._background_thread and self._background_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._background_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._background_thread.start()
        logger.info("Background metrics processing started")
    
    def stop_background_processing(self):
        """Stop background processing thread"""
        if self._background_thread and self._background_thread.is_alive():
            self._stop_event.set()
            self._background_thread.join(timeout=5.0)
            logger.info("Background metrics processing stopped")
    
    def _background_worker(self):
        """Background worker for periodic metrics processing"""
        
        interval = self.config.get('background_processing_interval', 60)  # 60 seconds default
        
        while not self._stop_event.wait(interval):
            try:
                # Periodic system metrics collection
                self._collect_system_metrics()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Export metrics if configured
                self._export_metrics()
                
            except Exception as e:
                logger.error(f"Background metrics processing error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics automatically"""
        
        try:
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            self.record_metric('memory_usage_peak_mb', 
                             memory_info.rss / 1024 / 1024,
                             'system')
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.record_metric('resource_utilization_cpu', cpu_percent, 'system')
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        # Implementation would clean up old alerts based on retention policy
        pass
    
    def _export_metrics(self):
        """Export metrics to configured destinations"""
        # Implementation would export to monitoring systems like Prometheus, etc.
        pass
    
    # CI Integration Methods
    def get_ci_metrics_report(self) -> Dict[str, Any]:
        """Generate metrics report for CI pipeline integration"""
        
        if not self.ci_mode:
            return {}
        
        report = {
            'session_id': self.session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'ci_mode': True,
            'metrics_summary': {},
            'quality_gates': {},
            'alerts_generated': [],
            'regression_analysis': {}
        }
        
        # Aggregate CI metrics
        for metric_name, values in self.ci_metrics.items():
            if not values:
                continue
                
            numeric_values = [v.value for v in values]
            report['metrics_summary'][metric_name] = {
                'count': len(numeric_values),
                'avg': statistics.mean(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values),
                'last': numeric_values[-1] if numeric_values else None
            }
        
        # Quality gate checks
        report['quality_gates'] = self._evaluate_quality_gates()
        
        # Include all alerts for CI visibility
        report['alerts_generated'] = [asdict(alert) for alert in self.alert_manager.get_alert_history()]
        
        return report
    
    def _evaluate_quality_gates(self) -> Dict[str, Any]:
        """Evaluate quality gates for CI pipeline"""
        
        gates = {}
        gate_config = self.config.get('ci_quality_gates', {})
        
        # Default quality gates
        default_gates = {
            'word_error_rate': {'max': 5.0},
            'diarization_error_rate': {'max': 15.0},
            'error_rate': {'max': 10.0},
            'success_rate': {'min': 85.0}
        }
        
        # Merge with config
        gate_config = {**default_gates, **gate_config}
        
        for metric_name, criteria in gate_config.items():
            if metric_name in self.ci_metrics:
                values = [v.value for v in self.ci_metrics[metric_name]]
                if values:
                    latest_value = values[-1]
                    passed = True
                    
                    if 'max' in criteria and latest_value > criteria['max']:
                        passed = False
                    if 'min' in criteria and latest_value < criteria['min']:
                        passed = False
                    
                    gates[metric_name] = {
                        'value': latest_value,
                        'criteria': criteria,
                        'passed': passed
                    }
        
        return gates
    
    def export_ci_report(self, filepath: str = None) -> str:
        """Export CI metrics report to file"""
        
        report = self.get_ci_metrics_report()
        
        if filepath is None:
            filepath = f"ci_metrics_report_{self.session_id}_{int(time.time())}.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"CI metrics report exported to {filepath}")
        return str(filepath)


# Global instance management
_global_metrics_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector(config: Dict[str, Any] = None, session_id: str = None) -> MetricsCollector:
    """Get or create global metrics collector instance"""
    global _global_metrics_collector
    
    with _collector_lock:
        if _global_metrics_collector is None:
            if config is None:
                # Default configuration
                config = {
                    'enabled': True,
                    'aggregation_window_seconds': 300,
                    'enable_background_processing': True,
                    'background_processing_interval': 60,
                    'alerting': {
                        'alert_destinations': {
                            'console': {'enabled': True},
                            'file_logging': {'enabled': True, 'path': 'logs/alerts'}
                        }
                    }
                }
            _global_metrics_collector = MetricsCollector(config, session_id)
        
        return _global_metrics_collector


def initialize_metrics_system(config: Dict[str, Any], session_id: str = None) -> MetricsCollector:
    """Initialize the global metrics system"""
    global _global_metrics_collector
    
    with _collector_lock:
        if _global_metrics_collector:
            _global_metrics_collector.stop_background_processing()
        
        _global_metrics_collector = MetricsCollector(config, session_id)
        return _global_metrics_collector


# Decorator for automatic metrics collection
def track_performance(metric_name: str = None, component: str = None):
    """Decorator to automatically track performance metrics for functions"""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal metric_name, component
            
            if metric_name is None:
                metric_name = f"function_duration_{func.__name__}"
            if component is None:
                component = func.__module__
            
            collector = get_metrics_collector()
            
            with collector.timing_context(metric_name, component):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Context managers for stage tracking
@contextmanager
def track_processing_stage(stage_name: str, 
                          component: str,
                          audio_duration: float = None,
                          **context):
    """Context manager for tracking complete processing stages"""
    
    collector = get_metrics_collector()
    start_time = time.time()
    
    stage_context = {'stage': stage_name, **context}
    if audio_duration:
        stage_context['audio_duration'] = audio_duration
    
    collector.logger.info(f"Processing stage started: {stage_name}",
                         stage=stage_name,
                         component=component,
                         **stage_context)
    
    try:
        yield collector
    except Exception as e:
        collector.record_metric('stage_error_count', 1, component, context={'stage': stage_name, 'error': str(e)})
        raise
    finally:
        duration = time.time() - start_time
        collector.record_processing_stage(stage_name, duration, component, context=stage_context)
        
        collector.logger.info(f"Processing stage completed: {stage_name}",
                             stage=stage_name,
                             component=component,
                             duration=duration,
                             **stage_context)


# Utility functions
def record_quality_metrics(metrics: Dict[str, float], component: str, **context):
    """Convenience function to record multiple quality metrics"""
    collector = get_metrics_collector()
    
    for metric_name, value in metrics.items():
        collector.record_metric(metric_name, value, component, context=context)


def record_business_event(event_type: str, component: str, **context):
    """Record business-level events and metrics"""
    collector = get_metrics_collector()
    
    # Record the event
    collector.record_metric(f'business_event_{event_type}', 1, component, context=context)
    
    # Update relevant business metrics
    if event_type == 'file_processed':
        collector.record_metric('files_processed_per_hour', 1, component, context=context)
        
        success = context.get('success', True)
        collector.record_metric('success_rate', 100.0 if success else 0.0, component, context=context)


def get_current_system_health() -> Dict[str, Any]:
    """Get current system health metrics for monitoring dashboards"""
    collector = get_metrics_collector()
    return collector.get_current_metrics_summary()


def export_metrics_for_ci() -> str:
    """Export metrics report for CI/CD integration"""
    collector = get_metrics_collector()
    return collector.export_ci_report()