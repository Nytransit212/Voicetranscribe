"""
Comprehensive observability system with loguru, OpenTelemetry, cost tracking, and profiling.
This module provides enhanced structured logging, distributed tracing, and performance monitoring.
"""

import os
import time
import uuid
import tempfile
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timezone
from functools import wraps
from contextlib import contextmanager
from pathlib import Path

import orjson
import psutil
from loguru import logger

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter
)
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor


# Initialize global observability components
class ObservabilityManager:
    """Central manager for all observability features"""
    
    def __init__(self, service_name: str = "ensemble-transcription", 
                 jaeger_endpoint: Optional[str] = None,
                 enable_profiling: bool = True,
                 log_level: str = "INFO"):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.enable_profiling = enable_profiling
        self.session_id = str(uuid.uuid4())[:8]
        
        # Initialize cost tracking
        self.cost_tracker = CostTracker()
        
        # Initialize profiler if enabled
        self.profiler = ProfilerManager() if enable_profiling else None
        
        # Setup loguru
        self._setup_loguru(log_level)
        
        # Setup OpenTelemetry
        self._setup_opentelemetry()
        
        # Session metrics
        self.start_time = time.time()
        self.process = psutil.Process()
        
        logger.info("Observability system initialized", 
                   session_id=self.session_id,
                   service=service_name,
                   jaeger_enabled=bool(jaeger_endpoint),
                   profiling_enabled=enable_profiling)
    
    def _setup_loguru(self, log_level: str):
        """Configure loguru with structured JSON logging and orjson serialization"""
        
        # Remove default handler
        logger.remove()
        
        # Custom JSON serializer using orjson for performance
        def json_serializer(record):
            subset = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "module": record["name"],
                "function": record["function"],
                "line": record["line"]
            }
            
            # Add extra fields from record["extra"]
            if record["extra"]:
                subset.update(record["extra"])
            
            return orjson.dumps(subset).decode()
        
        # Console handler with JSON format
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format=lambda record: json_serializer(record) + "\n",
            level=log_level,
            enqueue=True  # Thread-safe logging
        )
        
        # File handler for persistent logs (with rotation)
        log_dir = Path("logs/observability")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            sink=log_dir / "ensemble_{time:YYYY-MM-DD}.log",
            format=lambda record: json_serializer(record) + "\n",
            level=log_level,
            rotation="1 day",
            retention="30 days",
            compression="gz",
            enqueue=True
        )
        
        # Add session context to all logs
        logger.configure(extra={"session_id": self.session_id})
    
    def _setup_opentelemetry(self):
        """Configure OpenTelemetry tracing and metrics"""
        
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        # Add span processors
        # Console exporter for development
        console_processor = BatchSpanProcessor(ConsoleSpanExporter())
        tracer_provider.add_span_processor(console_processor)  # type: ignore
        
        # Jaeger exporter if endpoint provided
        if self.jaeger_endpoint:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=14268,
                    collector_endpoint=self.jaeger_endpoint,
                )
                jaeger_processor = BatchSpanProcessor(jaeger_exporter)
                tracer_provider.add_span_processor(jaeger_processor)  # type: ignore
                logger.info("Jaeger exporter configured", endpoint=self.jaeger_endpoint)
            except Exception as e:
                logger.warning("Failed to configure Jaeger exporter", error=str(e))
        
        # Set up metrics provider
        metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
        metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
        
        # Get tracer and meter
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Auto-instrument OpenAI and requests
        OpenAIInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        
        # Create custom metrics
        self.api_duration_histogram = self.meter.create_histogram(
            name="api_call_duration",
            description="Duration of API calls",
            unit="s"
        )
        
        self.cost_counter = self.meter.create_counter(
            name="api_cost_total",
            description="Total API costs",
            unit="USD"
        )
        
        self.processing_duration_histogram = self.meter.create_histogram(
            name="processing_stage_duration",
            description="Duration of processing stages",
            unit="s"
        )
    
    def get_enhanced_logger(self, component: str) -> 'EnhancedLogger':
        """Get an enhanced logger for a specific component"""
        return EnhancedLogger(component, self)
    
    def create_span(self, name: str, **attributes) -> trace.Span:
        """Create a new span with standard attributes"""
        span = self.tracer.start_span(name)
        
        # Add standard attributes
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("session.id", self.session_id)
        
        # Add custom attributes
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
        
        return span
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        return {
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": cpu_percent,
            "session_duration": time.time() - self.start_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class EnhancedLogger:
    """Enhanced logger that combines loguru with OpenTelemetry context"""
    
    def __init__(self, component: str, obs_manager: ObservabilityManager):
        self.component = component
        self.obs_manager = obs_manager
        self.context_data = {"component": component}
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log with enhanced context and optional span correlation"""
        
        # Merge context data
        log_data = {**self.context_data, **kwargs}
        
        # Add span context if available
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            span_context = current_span.get_span_context()
            log_data.update({
                "trace_id": format(span_context.trace_id, "032x"),
                "span_id": format(span_context.span_id, "016x"),
            })
        
        # Add system metrics if requested
        if kwargs.get("include_system_metrics", False):
            log_data["system_metrics"] = self.obs_manager.get_system_metrics()
        
        # Log with loguru
        getattr(logger, level.lower())(message, **log_data)
    
    def info(self, message: str, **kwargs):
        self._log_with_context("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context("ERROR", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context("DEBUG", message, **kwargs)
    
    def stage_start(self, stage: str, message: str, **kwargs):
        """Log start of processing stage with span creation"""
        stage_span = self.obs_manager.create_span(f"stage.{stage}", stage=stage, **kwargs)
        self._log_with_context("INFO", f"STAGE START: {message}", 
                             stage=stage, stage_action="start", **kwargs)
        return stage_span
    
    def stage_complete(self, stage: str, message: str, duration: Optional[float] = None, 
                      stage_span: Optional[trace.Span] = None, **kwargs):
        """Log completion of processing stage"""
        
        metrics = kwargs.get('metrics', {})
        if duration is not None:
            metrics['duration_seconds'] = round(duration, 3)
            
            # Record metric
            self.obs_manager.processing_duration_histogram.record(
                duration, {"stage": stage, "component": self.component}
            )
        
        kwargs['metrics'] = metrics
        
        self._log_with_context("INFO", f"STAGE COMPLETE: {message}", 
                             stage=stage, stage_action="complete", **kwargs)
        
        # Close span if provided
        if stage_span and stage_span.is_recording():
            if duration:
                stage_span.set_attribute("duration.seconds", duration)
            stage_span.set_status(trace.Status(trace.StatusCode.OK))
            stage_span.end()


class CostTracker:
    """Track API costs and resource usage"""
    
    def __init__(self):
        self.costs = {}
        self.usage_metrics = {}
        self.lock = threading.Lock()
        
        # OpenAI pricing (as of 2024)
        self.pricing = {
            "whisper-1": {
                "cost_per_minute": 0.006  # $0.006 per minute
            },
            "gpt-4": {
                "input_cost_per_1k_tokens": 0.03,
                "output_cost_per_1k_tokens": 0.06
            },
            "gpt-3.5-turbo": {
                "input_cost_per_1k_tokens": 0.002,
                "output_cost_per_1k_tokens": 0.002
            }
        }
    
    def track_api_call(self, service: str, model: str, usage_data: Dict[str, Any], 
                      duration: float, run_id: Optional[str] = None) -> float:
        """Track API call cost and usage"""
        
        cost = 0.0
        
        with self.lock:
            if service == "openai_whisper":
                # Calculate based on audio duration
                audio_minutes = usage_data.get("audio_duration_seconds", 0) / 60
                cost = audio_minutes * self.pricing["whisper-1"]["cost_per_minute"]
                
            elif service == "openai_chat":
                # Calculate based on token usage
                input_tokens = usage_data.get("prompt_tokens", 0)
                output_tokens = usage_data.get("completion_tokens", 0)
                
                model_pricing = self.pricing.get(model, self.pricing["gpt-3.5-turbo"])
                cost = (
                    (input_tokens / 1000) * model_pricing["input_cost_per_1k_tokens"] +
                    (output_tokens / 1000) * model_pricing["output_cost_per_1k_tokens"]
                )
            
            # Track in aggregated costs
            key = f"{service}_{model}"
            if key not in self.costs:
                self.costs[key] = {
                    "total_cost": 0.0,
                    "total_calls": 0,
                    "total_duration": 0.0,
                    "total_tokens": 0,
                    "runs": set()
                }
            
            self.costs[key]["total_cost"] += cost
            self.costs[key]["total_calls"] += 1
            self.costs[key]["total_duration"] += duration
            self.costs[key]["total_tokens"] += usage_data.get("total_tokens", 0)
            
            if run_id:
                self.costs[key]["runs"].add(run_id)
        
        return cost
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary for current session"""
        
        with self.lock:
            total_cost = sum(data["total_cost"] for data in self.costs.values())
            total_calls = sum(data["total_calls"] for data in self.costs.values())
            total_duration = sum(data["total_duration"] for data in self.costs.values())
            
            return {
                "total_cost_usd": round(total_cost, 4),
                "total_api_calls": total_calls,
                "total_api_duration_seconds": round(total_duration, 2),
                "cost_breakdown": {
                    key: {
                        "cost": round(data["total_cost"], 4),
                        "calls": data["total_calls"],
                        "avg_cost_per_call": round(data["total_cost"] / max(data["total_calls"], 1), 4),
                        "total_duration": round(data["total_duration"], 2)
                    }
                    for key, data in self.costs.items()
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


class ProfilerManager:
    """Manage profiling with py-spy integration"""
    
    def __init__(self):
        self.profiling_active = False
        self.profile_data = []
        self.profile_output_dir = Path("artifacts/profiles")
        self.profile_output_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def profile_context(self, name: str):
        """Context manager for profiling specific operations"""
        profile_file = self.profile_output_dir / f"profile_{name}_{int(time.time())}.prof"
        
        # Start profiling (simplified - would need actual py-spy integration)
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield profile_file
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            # Record profile metadata
            profile_metadata = {
                "name": name,
                "duration": end_time - start_time,
                "memory_delta_mb": (end_memory - start_memory) / 1024 / 1024,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "profile_file": str(profile_file)
            }
            
            self.profile_data.append(profile_metadata)
            logger.info("Profile completed", **profile_metadata)
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling data"""
        return {
            "total_profiles": len(self.profile_data),
            "profiles": self.profile_data.copy()
        }


# Decorators for easy instrumentation

def trace_stage(stage_name: str):
    """Decorator to automatically create spans for processing stages"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get observability manager from self if available
            obs_manager = getattr(self, 'obs_manager', None)
            if obs_manager:
                with obs_manager.create_span(stage_name) as span:
                    span.set_attribute("function", func.__name__)
                    span.set_attribute("component", type(self).__name__)
                    
                    start_time = time.time()
                    try:
                        result = func(self, *args, **kwargs)
                        duration = time.time() - start_time
                        span.set_attribute("duration.seconds", duration)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        span.set_attribute("duration.seconds", duration)
                        span.set_attribute("error.message", str(e))
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


def track_cost(service: str, model: str):
    """Decorator to automatically track API call costs"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            obs_manager = getattr(self, 'obs_manager', None)
            
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                duration = time.time() - start_time
                
                # Extract usage data from result or kwargs
                usage_data = kwargs.get('usage_data', {})
                if hasattr(result, 'usage'):
                    # OpenAI response object
                    usage_data = {
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "total_tokens": result.usage.total_tokens
                    }
                
                if obs_manager:
                    cost = obs_manager.cost_tracker.track_api_call(
                        service, model, usage_data, duration
                    )
                    obs_manager.cost_counter.add(cost, {"service": service, "model": model})
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"API call failed: {e}", 
                           service=service, model=model, duration=duration)
                raise
        return wrapper
    return decorator


# Global observability instance (initialized when needed)
_global_obs_manager: Optional[ObservabilityManager] = None

def get_observability_manager(**kwargs) -> ObservabilityManager:
    """Get or create global observability manager"""
    global _global_obs_manager
    if _global_obs_manager is None:
        _global_obs_manager = ObservabilityManager(**kwargs)
    return _global_obs_manager


def initialize_observability(service_name: str = "ensemble-transcription", **kwargs):
    """Initialize global observability system"""
    global _global_obs_manager
    _global_obs_manager = ObservabilityManager(service_name=service_name, **kwargs)
    return _global_obs_manager