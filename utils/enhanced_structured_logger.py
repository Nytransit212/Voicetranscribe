"""
Enhanced structured logger that bridges the old StructuredLogger API with the new observability system.
This maintains compatibility while adding loguru, OpenTelemetry, and cost tracking features.
"""

import time
import uuid
from typing import Dict, Any, Optional, Callable
from functools import wraps

from opentelemetry import trace
from .observability import get_observability_manager, EnhancedLogger, trace_stage


class EnhancedStructuredLogger:
    """
    Enhanced structured logger that maintains compatibility with existing StructuredLogger API
    while adding loguru, OpenTelemetry tracing, and cost tracking capabilities.
    """
    
    def __init__(self, name: str, session_id: Optional[str] = None, run_id: Optional[str] = None):
        self.name = name
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.run_id = run_id
        
        # Get or initialize observability manager
        self.obs_manager = get_observability_manager()
        
        # Get enhanced logger for this component
        self.enhanced_logger = self.obs_manager.get_enhanced_logger(name)
        
        # Add run_id to context if provided
        if run_id:
            self.enhanced_logger.context_data["run_id"] = run_id
            
        # Track active spans for stage management
        self._active_spans = {}
        
        # Initialize cost tracking context
        self._stage_costs = {}
        self._stage_start_times = {}
    
    def _create_log_entry(self, 
                         level: str,
                         message: str,
                         stage: Optional[str] = None,
                         variant_id: Optional[str] = None,
                         process_id: Optional[str] = None,
                         metrics: Optional[Dict[str, Any]] = None,
                         context: Optional[Dict[str, Any]] = None,
                         **kwargs) -> Dict[str, Any]:
        """Create structured log entry (maintains compatibility)"""
        
        entry: Dict[str, Any] = {
            'session_id': self.session_id,
            'component': self.name,
            'level': level
        }
        
        if self.run_id:
            entry['run_id'] = self.run_id
        if stage:
            entry['stage'] = stage
        if variant_id:
            entry['variant_id'] = variant_id
        if process_id:
            entry['process_id'] = process_id
        if metrics is not None:
            entry['metrics'] = metrics
        if context is not None:
            entry['context'] = context
            
        # Add any additional kwargs
        entry.update(kwargs)
            
        return entry
    
    def info(self, message: str, **kwargs):
        """Log info message with enhanced features"""
        entry = self._create_log_entry('INFO', message, **kwargs)
        self.enhanced_logger.info(message, **entry)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with enhanced features"""
        entry = self._create_log_entry('WARNING', message, **kwargs)
        self.enhanced_logger.warning(message, **entry)
    
    def error(self, message: str, **kwargs):
        """Log error message with enhanced features"""
        entry = self._create_log_entry('ERROR', message, **kwargs)
        self.enhanced_logger.error(message, **entry)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with enhanced features"""
        entry = self._create_log_entry('DEBUG', message, **kwargs)
        self.enhanced_logger.debug(message, **entry)
    
    def stage_start(self, stage: str, message: str, **kwargs):
        """Log start of processing stage with OpenTelemetry span creation"""
        
        # Create span for this stage
        span = self.enhanced_logger.stage_start(stage, message, **kwargs)
        self._active_spans[stage] = span
        
        # Track stage start time for cost calculation
        self._stage_start_times[stage] = time.time()
        
        # Log traditional entry for compatibility
        entry = self._create_log_entry('INFO', f"STAGE START: {message}", stage=stage, **kwargs)
        self.enhanced_logger.info(f"STAGE START: {message}", stage_action="start", **entry)
    
    def stage_complete(self, stage: str, message: str, duration: Optional[float] = None, **kwargs):
        """Log completion of processing stage with span closure and cost tracking"""
        
        # Calculate duration if not provided
        if duration is None and stage in self._stage_start_times:
            duration = time.time() - self._stage_start_times[stage]
        
        # Get associated span
        span = self._active_spans.pop(stage, None)
        
        # Log stage completion with enhanced features
        self.enhanced_logger.stage_complete(stage, message, duration=duration, stage_span=span, **kwargs)
        
        # Add cost tracking if available
        if stage in self._stage_costs:
            cost_data = self._stage_costs.pop(stage)
            kwargs.setdefault('metrics', {}).update(cost_data)
        
        # Log traditional entry for compatibility
        metrics = kwargs.get('metrics', {})
        if duration is not None:
            metrics['duration_seconds'] = round(duration, 3)
        kwargs['metrics'] = metrics
        
        entry = self._create_log_entry('INFO', f"STAGE COMPLETE: {message}", stage=stage, **kwargs)
        self.enhanced_logger.info(f"STAGE COMPLETE: {message}", stage_action="complete", **entry)
        
        # Clean up tracking
        self._stage_start_times.pop(stage, None)
    
    def variant_start(self, variant_id: str, stage: str, message: str, **kwargs):
        """Log start of variant processing with enhanced tracing"""
        
        # Create a span for this variant
        span_name = f"variant.{stage}.{variant_id}"
        span = self.obs_manager.create_span(span_name, 
                                          variant_id=variant_id, 
                                          stage=stage,
                                          **kwargs)
        
        # Store span for later completion
        variant_key = f"{stage}_{variant_id}"
        self._active_spans[variant_key] = span
        self._stage_start_times[variant_key] = time.time()
        
        # Log with enhanced context
        entry = self._create_log_entry('INFO', f"VARIANT START: {message}", 
                                     stage=stage, variant_id=variant_id, **kwargs)
        self.enhanced_logger.info(f"VARIANT START: {message}", 
                                variant_id=variant_id, stage=stage, **entry)
    
    def variant_complete(self, variant_id: str, stage: str, message: str, **kwargs):
        """Log completion of variant processing with span closure"""
        
        variant_key = f"{stage}_{variant_id}"
        
        # Calculate duration
        duration = None
        if variant_key in self._stage_start_times:
            duration = time.time() - self._stage_start_times.pop(variant_key)
        
        # Close associated span
        span = self._active_spans.pop(variant_key, None)
        if span and span.is_recording():
            if duration:
                span.set_attribute("duration.seconds", duration)
            span.set_status(trace.Status(trace.StatusCode.OK))
            span.end()
        
        # Add duration to metrics
        metrics = kwargs.get('metrics', {})
        if duration is not None:
            metrics['duration_seconds'] = round(duration, 3)
        kwargs['metrics'] = metrics
        
        # Log with enhanced context
        entry = self._create_log_entry('INFO', f"VARIANT COMPLETE: {message}", 
                                     stage=stage, variant_id=variant_id, **kwargs)
        self.enhanced_logger.info(f"VARIANT COMPLETE: {message}", 
                                variant_id=variant_id, stage=stage, **entry)
    
    def repair_start(self, process_id: str, problem_type: str, segment_info: Dict[str, Any], **kwargs):
        """Log start of segment repair with enhanced tracking"""
        
        # Create span for repair process
        span = self.obs_manager.create_span(f"repair.{problem_type}", 
                                          process_id=process_id,
                                          problem_type=problem_type,
                                          **segment_info)
        
        self._active_spans[process_id] = span
        self._stage_start_times[process_id] = time.time()
        
        # Enhanced context
        context = {'problem_type': problem_type, 'segment': segment_info}
        entry = self._create_log_entry('INFO', f"REPAIR START: {problem_type} repair initiated", 
                                     stage='repair', process_id=process_id, context=context, **kwargs)
        
        self.enhanced_logger.info(f"REPAIR START: {problem_type} repair initiated", 
                                process_id=process_id, problem_type=problem_type,
                                segment_info=segment_info, **entry)
    
    def repair_complete(self, process_id: str, repair_results: Dict[str, Any], **kwargs):
        """Log completion of segment repair with enhanced metrics"""
        
        # Calculate duration
        duration = None
        if process_id in self._stage_start_times:
            duration = time.time() - self._stage_start_times.pop(process_id)
        
        # Close span
        span = self._active_spans.pop(process_id, None)
        if span and span.is_recording():
            if duration:
                span.set_attribute("duration.seconds", duration)
            span.set_attribute("candidates_generated", len(repair_results.get('candidates', [])))
            span.set_attribute("best_improvement", repair_results.get('best_improvement_score', 0))
            span.set_status(trace.Status(trace.StatusCode.OK))
            span.end()
        
        # Enhanced metrics
        metrics = {
            'candidates_generated': len(repair_results.get('candidates', [])),
            'best_improvement': repair_results.get('best_improvement_score', 0)
        }
        if duration:
            metrics['duration_seconds'] = round(duration, 3)
        
        entry = self._create_log_entry('INFO', f"REPAIR COMPLETE: Generated {metrics['candidates_generated']} repair candidates", 
                                     stage='repair', process_id=process_id, metrics=metrics, **kwargs)
        
        self.enhanced_logger.info(f"REPAIR COMPLETE: Generated {metrics['candidates_generated']} repair candidates", 
                                process_id=process_id, repair_results=repair_results, **entry)
    
    def track_api_cost(self, service: str, model: str, usage_data: Dict[str, Any], 
                      duration: float, stage: Optional[str] = None):
        """Track API call cost and associate with current stage"""
        
        cost = self.obs_manager.cost_tracker.track_api_call(
            service, model, usage_data, duration, self.run_id or "unknown"
        )
        
        # Record cost metrics
        self.obs_manager.cost_counter.add(cost, {
            "service": service, 
            "model": model, 
            "component": self.name,
            "stage": stage or "unknown"
        })
        
        # Store cost data for stage completion
        if stage and stage in self._stage_start_times:
            if stage not in self._stage_costs:
                self._stage_costs[stage] = {}
            
            self._stage_costs[stage].update({
                f"{service}_{model}_cost": round(cost, 6),
                f"{service}_{model}_duration": round(duration, 3),
                f"{service}_{model}_calls": self._stage_costs[stage].get(f"{service}_{model}_calls", 0) + 1
            })
        
        # Log cost tracking
        self.info(f"API cost tracked: ${cost:.6f}", 
                 service=service, model=model, cost_usd=cost, 
                 usage_data=usage_data, api_duration=duration, stage=stage)
        
        return cost
    
    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary for current session"""
        return self.obs_manager.cost_tracker.get_session_summary()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        return self.obs_manager.get_system_metrics()


# Enhanced decorator that works with both old and new logger
def enhanced_log_stage(stage_name: str):
    """Enhanced decorator to automatically log stage start and completion with tracing"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Try to get enhanced structured logger from self
            logger = getattr(self, 'structured_logger', None)
            
            if logger and hasattr(logger, 'stage_start'):
                # Enhanced logger with tracing support
                start_time = time.time()
                logger.stage_start(stage_name, f"Starting {func.__name__}")
                
                try:
                    result = func(self, *args, **kwargs)
                    duration = time.time() - start_time
                    logger.stage_complete(stage_name, f"Completed {func.__name__}", duration=duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"Stage {stage_name} failed in {func.__name__}: {str(e)}", 
                               stage=stage_name, metrics={'duration_seconds': duration})
                    raise
            elif logger:
                # Fallback to original structured logger
                start_time = time.time()
                logger.stage_start(stage_name, f"Starting {func.__name__}")
                
                try:
                    result = func(self, *args, **kwargs)
                    duration = time.time() - start_time
                    logger.stage_complete(stage_name, f"Completed {func.__name__}", duration=duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"Stage {stage_name} failed in {func.__name__}: {str(e)}", 
                               stage=stage_name, metrics={'duration_seconds': duration})
                    raise
            else:
                # No logger available
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


# Convenience function to create enhanced logger
def create_enhanced_logger(name: str, session_id: Optional[str] = None, run_id: Optional[str] = None) -> EnhancedStructuredLogger:
    """Create an enhanced structured logger instance"""
    return EnhancedStructuredLogger(name, session_id, run_id)