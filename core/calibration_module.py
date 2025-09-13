"""
Calibration Module for Advanced Ensemble Transcription System

Provides multiple calibration strategies for normalizing raw confidence scores
to consistently meaningful values across different processing contexts.
"""

import numpy as np
import json
import math
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.metrics_registry import MetricsRegistryManager, DimensionStats

@dataclass
class CalibrationResult:
    """Result from calibration processing"""
    calibrated_scores: List[float]
    calibration_method: str
    calibration_confidence: float
    calibration_metadata: Dict[str, Any]
    raw_scores: List[float]

class CalibrationStrategy(ABC):
    """Abstract base class for calibration strategies"""
    
    @abstractmethod
    def name(self) -> str:
        """Return strategy name"""
        pass
    
    @abstractmethod
    def calibrate_scores(self, scores: List[float], dimension: Optional[str] = None) -> CalibrationResult:
        """Calibrate scores using this strategy"""
        pass

class RawScoresStrategy(CalibrationStrategy):
    """Pass-through strategy: return raw scores without calibration"""
    
    def __init__(self):
        self.logger = create_enhanced_logger("calibration_raw")
    
    def name(self) -> str:
        return "raw_scores"
    
    def calibrate_scores(self, scores: List[float], dimension: Optional[str] = None) -> CalibrationResult:
        """Return raw scores without any calibration"""
        if not scores:
            return CalibrationResult(
                calibrated_scores=[],
                calibration_method=self.name(),
                calibration_confidence=1.0,
                calibration_metadata={'processing': 'none'},
                raw_scores=[]
            )
        
        # Simple clipping to 0-1 range
        clipped_scores = [max(0.0, min(1.0, score)) for score in scores]
        
        self.logger.info(f"Raw scores pass-through for {len(scores)} values", 
                        method='raw_scores', dimension=dimension)
        
        return CalibrationResult(
            calibrated_scores=clipped_scores,
            calibration_method=self.name(),
            calibration_confidence=1.0,
            calibration_metadata={
                'processing': 'clipping_only',
                'original_range': [min(scores), max(scores)],
                'clipped_count': sum(1 for s in scores if s < 0 or s > 1)
            },
            raw_scores=scores.copy()
        )

class RegistryBasedStrategy(CalibrationStrategy):
    """Registry-based calibration using historical statistics (current system)"""
    
    def __init__(self, domain: str = "general", speaker_count: int = 10, noise_level: str = "medium"):
        self.domain = domain
        self.speaker_count = speaker_count
        self.noise_level = noise_level
        self.logger = create_enhanced_logger("calibration_registry")
        
        # Initialize metrics registry
        try:
            self.metrics_registry = MetricsRegistryManager()
            self.calibration_stats = self.metrics_registry.get_calibration_stats(
                domain=domain, speaker_count=speaker_count, noise_level=noise_level
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize metrics registry: {e}", error=str(e))
            self.metrics_registry = None
            self.calibration_stats = None
    
    def name(self) -> str:
        return "registry_based"
    
    def calibrate_scores(self, scores: List[float], dimension: Optional[str] = None) -> CalibrationResult:
        """Calibrate scores using historical registry statistics"""
        if not scores:
            return CalibrationResult(
                calibrated_scores=[],
                calibration_method=self.name(),
                calibration_confidence=0.0,
                calibration_metadata={'error': 'no_scores'},
                raw_scores=[]
            )
        
        scores_array = np.array(scores)
        
        # Use registry calibration if available
        if (self.calibration_stats and dimension and dimension in self.calibration_stats):
            calibration_stat = self.calibration_stats[dimension]
            
            # Use historical mean and std for normalization
            target_mean = calibration_stat.mean
            target_std = max(calibration_stat.std, 0.01)  # Avoid division by zero
            
            # Robust z-score normalization with clipping
            current_mean = np.mean(scores_array)
            current_std = np.std(scores_array) if len(scores_array) > 1 else target_std
            
            if current_std > 0.001:
                # Standardize using current distribution
                z_scores = (scores_array - current_mean) / current_std
                
                # Transform to target distribution and apply sigmoid mapping
                calibrated_scores = z_scores * target_std + target_mean
                
                # Apply sigmoid to map to 0-1 range centered on historical performance
                def calibrated_sigmoid(x, center=target_mean, scale=target_std):
                    z = (x - center) / (scale * 2)  # 2-sigma range
                    return 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))  # Clip to prevent overflow
                
                normalized_scores = calibrated_sigmoid(calibrated_scores)
                confidence = 0.9  # High confidence with registry data
            else:
                # Fallback for uniform scores
                normalized_scores = np.full_like(scores_array, min(max(target_mean, 0.0), 1.0))
                confidence = 0.6  # Medium confidence for uniform fallback
                
            # Ensure valid range
            final_scores = np.clip(normalized_scores, 0.0, 1.0)
            
            self.logger.info(f"Registry calibration applied to {len(scores)} scores", 
                           context={'method': 'registry_based', 'dimension': dimension, 
                                   'target_mean': target_mean, 'target_std': target_std})
            
            return CalibrationResult(
                calibrated_scores=final_scores.tolist(),
                calibration_method=self.name(),
                calibration_confidence=confidence,
                calibration_metadata={
                    'registry_available': True,
                    'target_mean': target_mean,
                    'target_std': target_std,
                    'current_mean': current_mean,
                    'current_std': current_std,
                    'sample_count': calibration_stat.sample_count,
                    'domain_context': f"{self.domain}/{self.speaker_count}spk/{self.noise_level}"
                },
                raw_scores=scores.copy()
            )
        
        # Fallback to absolute calibration if no registry available
        return self._fallback_absolute_calibration(scores, dimension)
    
    def _fallback_absolute_calibration(self, scores: List[float], dimension: Optional[str]) -> CalibrationResult:
        """Fallback to absolute calibration ranges (current system fallback)"""
        calibration_ranges = {
            'D': {'min': 0.15, 'max': 0.95, 'median': 0.65},  # Diarization quality
            'A': {'min': 0.25, 'max': 0.98, 'median': 0.75},  # ASR confidence  
            'L': {'min': 0.30, 'max': 0.92, 'median': 0.70},  # Linguistic quality
            'R': {'min': 0.40, 'max': 0.90, 'median': 0.68},  # Cross-run agreement
            'O': {'min': 0.35, 'max': 0.88, 'median': 0.62}   # Overlap handling
        }
        
        if dimension and dimension in calibration_ranges:
            # Use calibrated absolute normalization
            cal_range = calibration_ranges[dimension]
            cal_min = cal_range['min']
            cal_max = cal_range['max']
            
            # Sigmoid-based normalization for better distribution
            normalized = []
            for score in scores:
                # Map to calibrated range first
                if cal_max > cal_min:
                    mapped_score = (score - cal_min) / (cal_max - cal_min)
                else:
                    mapped_score = 0.5
                
                # Apply sigmoid for smoother distribution around median
                sigmoid_score = 1.0 / (1.0 + math.exp(-6.0 * (mapped_score - 0.5)))
                normalized.append(sigmoid_score)
            
            # Clip to valid range
            final_scores = [max(0.0, min(1.0, score)) for score in normalized]
            
            self.logger.info(f"Absolute calibration fallback for {len(scores)} scores", 
                           context={'method': 'absolute_fallback', 'dimension': dimension})
            
            return CalibrationResult(
                calibrated_scores=final_scores,
                calibration_method=f"{self.name()}_absolute_fallback",
                calibration_confidence=0.7,
                calibration_metadata={
                    'registry_available': False,
                    'calibration_range': cal_range,
                    'fallback_reason': 'no_registry_data'
                },
                raw_scores=scores.copy()
            )
        
        else:
            # Final fallback to relative normalization
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                final_scores = [0.5] * len(scores)  # All scores are the same
            else:
                # Min-max normalization with gentle compression towards center
                final_scores = []
                for score in scores:
                    relative_score = (score - min_score) / (max_score - min_score)
                    # Apply slight compression towards 0.5 to avoid extreme values
                    compressed_score = 0.5 + 0.8 * (relative_score - 0.5)
                    final_scores.append(max(0.0, min(1.0, compressed_score)))
            
            self.logger.info(f"Relative normalization fallback for {len(scores)} scores", 
                           context={'method': 'relative_fallback', 'dimension': dimension})
            
            return CalibrationResult(
                calibrated_scores=final_scores,
                calibration_method=f"{self.name()}_relative_fallback",
                calibration_confidence=0.5,
                calibration_metadata={
                    'registry_available': False,
                    'fallback_reason': 'unknown_dimension',
                    'original_range': [min_score, max_score]
                },
                raw_scores=scores.copy()
            )

class IsotonicRegressionStrategy(CalibrationStrategy):
    """Isotonic regression calibration using pre-trained models"""
    
    def __init__(self, models_dir: str = "calibration_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = create_enhanced_logger("calibration_isotonic")
        
        # Load pre-trained isotonic regression models
        self.isotonic_models = self._load_isotonic_models()
    
    def name(self) -> str:
        return "isotonic_regression"
    
    def calibrate_scores(self, scores: List[float], dimension: Optional[str] = None) -> CalibrationResult:
        """Calibrate scores using isotonic regression models"""
        if not scores:
            return CalibrationResult(
                calibrated_scores=[],
                calibration_method=self.name(),
                calibration_confidence=0.0,
                calibration_metadata={'error': 'no_scores'},
                raw_scores=[]
            )
        
        scores_array = np.array(scores)
        
        # Use isotonic model if available for this dimension
        if dimension and dimension in self.isotonic_models:
            model = self.isotonic_models[dimension]
            
            try:
                # Apply isotonic regression transformation
                calibrated_scores = model.transform(scores_array.reshape(-1, 1))
                calibrated_scores = np.clip(calibrated_scores, 0.0, 1.0)
                
                confidence = 0.85  # High confidence with trained model
                
                self.logger.info(f"Isotonic regression applied to {len(scores)} scores", 
                               context={'method': 'isotonic_regression', 'dimension': dimension})
                
                return CalibrationResult(
                    calibrated_scores=calibrated_scores.tolist(),
                    calibration_method=self.name(),
                    calibration_confidence=confidence,
                    calibration_metadata={
                        'model_available': True,
                        'model_training_samples': getattr(model, 'training_samples_', 'unknown'),
                        'dimension': dimension
                    },
                    raw_scores=scores.copy()
                )
                
            except Exception as e:
                self.logger.warning(f"Isotonic model failed for dimension {dimension}: {e}")
        
        # Fallback to simple probability calibration if no model available
        return self._fallback_probability_calibration(scores, dimension)
    
    def _load_isotonic_models(self) -> Dict[str, IsotonicRegression]:
        """Load pre-trained isotonic regression models"""
        models = {}
        
        for dimension in ['D', 'A', 'L', 'R', 'O']:
            model_path = self.models_dir / f"isotonic_{dimension}.pkl"
            
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        models[dimension] = pickle.load(f)
                    self.logger.info(f"Loaded isotonic model for dimension {dimension}")
                except Exception as e:
                    self.logger.warning(f"Failed to load isotonic model for {dimension}: {e}")
            else:
                # Create a default isotonic regression model
                models[dimension] = self._create_default_isotonic_model(dimension)
                
        return models
    
    def _create_default_isotonic_model(self, dimension: str) -> IsotonicRegression:
        """Create a default isotonic regression model with reasonable calibration"""
        # Generate synthetic training data based on typical confidence score patterns
        np.random.seed(42 + ord(dimension))  # Deterministic but dimension-specific
        
        # Create synthetic raw scores (typically not well-calibrated)
        n_samples = 1000
        raw_scores = np.random.beta(2, 2, n_samples)  # Scores concentrated towards middle
        
        # Create corresponding "true" calibrated scores with some transformation
        dimension_bias = {'D': 0.72, 'A': 0.78, 'L': 0.68, 'R': 0.65, 'O': 0.70}[dimension]
        noise_level = 0.1
        
        true_scores = np.clip(
            dimension_bias + (raw_scores - 0.5) * 0.8 + np.random.normal(0, noise_level, n_samples),
            0.0, 1.0
        )
        
        # Train isotonic regression
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(raw_scores, true_scores)
        
        # Store training info
        model.training_samples_ = n_samples
        
        # Save model for future use
        model_path = self.models_dir / f"isotonic_{dimension}.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Created and saved default isotonic model for dimension {dimension}")
        except Exception as e:
            self.logger.warning(f"Failed to save isotonic model for {dimension}: {e}")
        
        return model
    
    def _fallback_probability_calibration(self, scores: List[float], dimension: Optional[str]) -> CalibrationResult:
        """Fallback probability calibration using Platt scaling"""
        scores_array = np.array(scores)
        
        # Simple Platt scaling: sigmoid transformation
        # Fit sigmoid to map scores to well-calibrated probabilities
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array) + 0.01  # Avoid zero std
        
        # Apply sigmoid transformation
        normalized_scores = (scores_array - mean_score) / std_score
        calibrated_scores = 1.0 / (1.0 + np.exp(-normalized_scores))
        calibrated_scores = np.clip(calibrated_scores, 0.01, 0.99)  # Avoid extreme probabilities
        
        self.logger.info(f"Probability calibration fallback for {len(scores)} scores", 
                        context={'method': 'platt_scaling_fallback', 'dimension': dimension})
        
        return CalibrationResult(
            calibrated_scores=calibrated_scores.tolist(),
            calibration_method=f"{self.name()}_platt_fallback",
            calibration_confidence=0.6,
            calibration_metadata={
                'model_available': False,
                'fallback_method': 'platt_scaling',
                'mean_score': mean_score,
                'std_score': std_score
            },
            raw_scores=scores.copy()
        )

class PerDomainStrategy(CalibrationStrategy):
    """Per-domain calibration with domain-specific adjustments"""
    
    def __init__(self, domain: str = "general", base_strategy: Optional[CalibrationStrategy] = None):
        self.domain = domain
        self.base_strategy = base_strategy or RegistryBasedStrategy(domain=domain)
        self.logger = create_enhanced_logger("calibration_per_domain")
        
        # Domain-specific adjustment factors
        self.domain_adjustments = {
            'meeting': {'D': 1.05, 'A': 0.95, 'L': 1.10, 'R': 1.08, 'O': 0.90},
            'interview': {'D': 1.02, 'A': 1.05, 'L': 1.05, 'R': 0.95, 'O': 1.15},
            'lecture': {'D': 0.95, 'A': 1.08, 'L': 1.15, 'R': 0.90, 'O': 0.85},
            'general': {'D': 1.00, 'A': 1.00, 'L': 1.00, 'R': 1.00, 'O': 1.00}
        }
    
    def name(self) -> str:
        return f"per_domain_{self.domain}"
    
    def calibrate_scores(self, scores: List[float], dimension: Optional[str] = None) -> CalibrationResult:
        """Calibrate scores with domain-specific adjustments"""
        if not scores:
            return CalibrationResult(
                calibrated_scores=[],
                calibration_method=self.name(),
                calibration_confidence=0.0,
                calibration_metadata={'error': 'no_scores'},
                raw_scores=[]
            )
        
        # First apply base calibration strategy
        base_result = self.base_strategy.calibrate_scores(scores, dimension)
        
        # Apply domain-specific adjustments
        adjustment_factor = 1.0
        if (self.domain in self.domain_adjustments and 
            dimension and dimension in self.domain_adjustments[self.domain]):
            adjustment_factor = self.domain_adjustments[self.domain][dimension]
        
        # Apply adjustment with bounds checking
        adjusted_scores = []
        for score in base_result.calibrated_scores:
            adjusted_score = score * adjustment_factor
            # Gentle clamping to preserve relative ordering
            if adjusted_score > 1.0:
                adjusted_score = 0.95 + 0.05 * (1.0 / (1.0 + (adjusted_score - 1.0)))
            elif adjusted_score < 0.0:
                adjusted_score = 0.05 * (1.0 - 1.0 / (1.0 + abs(adjusted_score)))
            
            adjusted_scores.append(adjusted_score)
        
        # Combine metadata
        combined_metadata = base_result.calibration_metadata.copy()
        combined_metadata.update({
            'domain': self.domain,
            'adjustment_factor': adjustment_factor,
            'base_strategy': self.base_strategy.name()
        })
        
        self.logger.info(f"Domain-specific calibration for {self.domain}", 
                        context={'method': self.name(), 'dimension': dimension, 
                                'adjustment_factor': adjustment_factor})
        
        return CalibrationResult(
            calibrated_scores=adjusted_scores,
            calibration_method=self.name(),
            calibration_confidence=base_result.calibration_confidence * 0.95,  # Slight reduction due to adjustment
            calibration_metadata=combined_metadata,
            raw_scores=base_result.raw_scores
        )

class CalibrationModule:
    """Main calibration processing module"""
    
    def __init__(self, default_strategy: str = "registry_based", 
                 domain: str = "general", speaker_count: int = 10, noise_level: str = "medium"):
        self.domain = domain
        self.speaker_count = speaker_count
        self.noise_level = noise_level
        
        # Initialize strategies
        self.strategies = {
            "raw_scores": RawScoresStrategy(),
            "registry_based": RegistryBasedStrategy(domain, speaker_count, noise_level),
            "isotonic_regression": IsotonicRegressionStrategy(),
            "per_domain": PerDomainStrategy(domain)
        }
        
        self.default_strategy = default_strategy
        self.logger = create_enhanced_logger("calibration_module")
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available calibration strategies"""
        return list(self.strategies.keys())
    
    def calibrate_scores(self, 
                        scores: List[float], 
                        dimension: Optional[str] = None,
                        strategy: Optional[str] = None,
                        strategy_params: Optional[Dict[str, Any]] = None) -> CalibrationResult:
        """
        Calibrate scores using specified strategy
        
        Args:
            scores: Raw scores to calibrate
            dimension: Scoring dimension ('D', 'A', 'L', 'R', 'O')
            strategy: Strategy name to use (defaults to instance default)
            strategy_params: Optional parameters for strategy
            
        Returns:
            CalibrationResult with calibrated scores and metadata
        """
        if not scores:
            return CalibrationResult(
                calibrated_scores=[],
                calibration_method="none",
                calibration_confidence=0.0,
                calibration_metadata={'error': 'no_scores'},
                raw_scores=[]
            )
        
        strategy_name = strategy or self.default_strategy
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(self.strategies.keys())}")
        
        calibration_strategy = self.strategies[strategy_name]
        
        # Apply strategy parameters if provided
        if strategy_params:
            self._apply_strategy_params(calibration_strategy, strategy_params)
        
        self.logger.info(f"Calibrating {len(scores)} scores with strategy: {strategy_name}", 
                        context={'scores_count': len(scores), 'dimension': dimension, 'strategy': strategy_name})
        
        try:
            result = calibration_strategy.calibrate_scores(scores, dimension)
            
            self.logger.info(f"Calibration completed successfully", 
                           context={'strategy': strategy_name, 'dimension': dimension,
                                   'confidence': result.calibration_confidence})
            
            return result
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}", 
                            context={'strategy': strategy_name, 'dimension': dimension, 'error': str(e)})
            
            # Fallback to raw scores if strategy fails
            if strategy_name != "raw_scores":
                self.logger.warning("Falling back to raw_scores strategy")
                fallback_strategy = self.strategies["raw_scores"]
                return fallback_strategy.calibrate_scores(scores, dimension)
            else:
                raise
    
    def _apply_strategy_params(self, strategy: CalibrationStrategy, params: Dict[str, Any]):
        """Apply parameters to strategy instance"""
        for key, value in params.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
    
    def compare_strategies(self, 
                          scores: List[float], 
                          dimension: Optional[str] = None,
                          strategies: Optional[List[str]] = None) -> Dict[str, CalibrationResult]:
        """
        Compare multiple calibration strategies
        
        Args:
            scores: Raw scores to calibrate
            dimension: Scoring dimension
            strategies: List of strategy names to compare (defaults to all)
            
        Returns:
            Dictionary mapping strategy names to calibration results
        """
        if strategies is None:
            strategies = list(self.strategies.keys())
        
        results = {}
        
        for strategy_name in strategies:
            if strategy_name in self.strategies:
                try:
                    result = self.calibrate_scores(scores, dimension=dimension, strategy=strategy_name)
                    results[strategy_name] = result
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy_name} failed during comparison: {e}")
        
        return results
    
    def train_isotonic_model(self, 
                           raw_scores: List[float], 
                           true_scores: List[float], 
                           dimension: str) -> bool:
        """
        Train and save an isotonic regression model for a specific dimension
        
        Args:
            raw_scores: Raw confidence scores
            true_scores: Ground truth calibrated scores
            dimension: Dimension to train model for
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            if len(raw_scores) != len(true_scores) or len(raw_scores) < 10:
                self.logger.error("Insufficient or mismatched training data")
                return False
            
            # Train isotonic regression model
            model = IsotonicRegression(out_of_bounds='clip')
            model.fit(np.array(raw_scores), np.array(true_scores))
            
            # Save model
            isotonic_strategy = self.strategies.get("isotonic_regression")
            if isinstance(isotonic_strategy, IsotonicRegressionStrategy):
                model_path = isotonic_strategy.models_dir / f"isotonic_{dimension}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Update strategy's loaded models
                isotonic_strategy.isotonic_models[dimension] = model
                
                self.logger.info(f"Successfully trained isotonic model for dimension {dimension}")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to train isotonic model for {dimension}: {e}")
        
        return False