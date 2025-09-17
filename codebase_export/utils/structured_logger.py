import logging
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps

class StructuredLogger:
    """Structured logging utility for ensemble transcription pipeline"""
    
    def __init__(self, name: str, session_id: Optional[str] = None):
        self.name = name
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.logger = logging.getLogger(f"ensemble.{name}")
        
        # Configure logger if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _create_log_entry(self, 
                         level: str,
                         message: str,
                         stage: Optional[str] = None,
                         variant_id: Optional[str] = None,
                         process_id: Optional[str] = None,
                         metrics: Optional[Dict[str, Any]] = None,
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create structured log entry"""
        entry: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'component': self.name,
            'level': level,
            'message': message
        }
        
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
            
        return entry
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        entry = self._create_log_entry('INFO', message, **kwargs)
        self.logger.info(json.dumps(entry))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        entry = self._create_log_entry('WARNING', message, **kwargs)
        self.logger.warning(json.dumps(entry))
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        entry = self._create_log_entry('ERROR', message, **kwargs)
        self.logger.error(json.dumps(entry))
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        entry = self._create_log_entry('DEBUG', message, **kwargs)
        self.logger.debug(json.dumps(entry))
    
    def stage_start(self, stage: str, message: str, **kwargs):
        """Log start of processing stage"""
        self.info(f"STAGE START: {message}", stage=stage, **kwargs)
    
    def stage_complete(self, stage: str, message: str, duration: Optional[float] = None, **kwargs):
        """Log completion of processing stage"""
        metrics = kwargs.get('metrics', {})
        if duration is not None:
            metrics['duration_seconds'] = round(duration, 3)
        kwargs['metrics'] = metrics
        self.info(f"STAGE COMPLETE: {message}", stage=stage, **kwargs)
    
    def variant_start(self, variant_id: str, stage: str, message: str, **kwargs):
        """Log start of variant processing"""
        self.info(f"VARIANT START: {message}", stage=stage, variant_id=variant_id, **kwargs)
    
    def variant_complete(self, variant_id: str, stage: str, message: str, **kwargs):
        """Log completion of variant processing"""
        self.info(f"VARIANT COMPLETE: {message}", stage=stage, variant_id=variant_id, **kwargs)
    
    def repair_start(self, process_id: str, problem_type: str, segment_info: Dict[str, Any], **kwargs):
        """Log start of segment repair"""
        context = {'problem_type': problem_type, 'segment': segment_info}
        self.info(f"REPAIR START: {problem_type} repair initiated", 
                 stage='repair', process_id=process_id, context=context, **kwargs)
    
    def repair_complete(self, process_id: str, repair_results: Dict[str, Any], **kwargs):
        """Log completion of segment repair"""
        metrics = {
            'candidates_generated': len(repair_results.get('candidates', [])),
            'best_improvement': repair_results.get('best_improvement_score', 0)
        }
        self.info(f"REPAIR COMPLETE: Generated {metrics['candidates_generated']} repair candidates", 
                 stage='repair', process_id=process_id, metrics=metrics, **kwargs)

def log_stage(stage_name: str):
    """Decorator to automatically log stage start and completion"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Try to get structured logger from self
            logger = getattr(self, 'structured_logger', None)
            if logger:
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
                # Fallback to original function if no logger
                return func(self, *args, **kwargs)
        return wrapper
    return decorator