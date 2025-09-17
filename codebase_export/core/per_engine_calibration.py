"""
Per-Engine Confidence Calibration System

Implements calibrated confidence scores for individual ASR providers (OpenAI, Deepgram, Faster-Whisper)
using isotonic regression and Platt scaling. This system addresses the fact that different ASR engines
have very different confidence score distributions and meanings.

Key Features:
- Per-engine calibration models (separate for each ASR provider)
- Isotonic regression and Platt scaling calibration methods
- Training on US English development sets with ground truth
- Reliability diagram generation and validation metrics
- Expected Calibration Error (ECE) and Brier Score computation
- Persistent model storage and loading
"""

import numpy as np
import pickle
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from .asr_providers.base import ASRResult, ASRSegment, ASRProvider
from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class CalibrationData:
    """Training data for calibration"""
    provider: str
    raw_confidences: List[float]
    ground_truth_labels: List[int]  # 1 for correct, 0 for incorrect
    segment_lengths: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CalibrationMetrics:
    """Calibration quality metrics"""
    expected_calibration_error: float
    brier_score: float
    reliability_score: float
    sharpness_score: float
    confidence_histogram: Dict[str, List[float]]
    reliability_diagram_data: Dict[str, List[float]]
    sample_count: int

@dataclass
class CalibratedResult:
    """Result of confidence calibration"""
    raw_confidence: float
    calibrated_confidence: float
    calibration_method: str
    provider: str
    confidence_delta: float
    calibration_metadata: Dict[str, Any] = field(default_factory=dict)

class ReliabilityDiagramGenerator:
    """Generates reliability diagrams for calibration assessment"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.logger = create_enhanced_logger("reliability_diagram")
    
    def generate_diagram_data(self, 
                            predicted_probs: np.ndarray, 
                            true_labels: np.ndarray) -> Dict[str, List[float]]:
        """
        Generate reliability diagram data
        
        Args:
            predicted_probs: Predicted probabilities/confidence scores
            true_labels: Ground truth binary labels
            
        Returns:
            Dictionary with bin data for reliability diagram
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = predicted_probs[in_bin].mean()
                count_in_bin = in_bin.sum()
            else:
                accuracy_in_bin = 0
                avg_confidence_in_bin = 0
                count_in_bin = 0
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(int(count_in_bin))
        
        return {
            'bin_boundaries': bin_boundaries.tolist(),
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }
    
    def calculate_expected_calibration_error(self, 
                                           predicted_probs: np.ndarray, 
                                           true_labels: np.ndarray) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(predicted_probs)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = predicted_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

class EngineCalibrationModel:
    """Calibration model for a specific ASR engine"""
    
    def __init__(self, 
                 provider_name: str,
                 calibration_method: str = "isotonic_regression",
                 models_dir: str = "calibration_models"):
        self.provider_name = provider_name
        self.calibration_method = calibration_method
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.logger = create_enhanced_logger(f"engine_calibration_{provider_name}")
        
        # Calibration models
        self.isotonic_model = None
        self.platt_model = None
        self.reliability_generator = ReliabilityDiagramGenerator()
        
        # Training metadata
        self.training_metadata = {}
        self.calibration_metrics = None
        
        # Load existing models if available
        self._load_models()
    
    def fit(self, calibration_data: CalibrationData) -> CalibrationMetrics:
        """
        Fit calibration models using training data
        
        Args:
            calibration_data: Training data with raw confidences and ground truth
            
        Returns:
            CalibrationMetrics with quality assessment
        """
        if len(calibration_data.raw_confidences) < 50:
            raise ValueError(f"Insufficient training data: {len(calibration_data.raw_confidences)} samples < 50")
        
        self.logger.info(f"Training calibration models for {self.provider_name}", 
                        context={'samples': len(calibration_data.raw_confidences)})
        
        raw_conf = np.array(calibration_data.raw_confidences)
        true_labels = np.array(calibration_data.ground_truth_labels)
        
        # Ensure valid ranges
        raw_conf = np.clip(raw_conf, 0.001, 0.999)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            raw_conf.reshape(-1, 1), true_labels, test_size=0.2, random_state=42
        )
        
        # Fit isotonic regression
        self.isotonic_model = IsotonicRegression(
            y_min=0.01, y_max=0.99, increasing=True, out_of_bounds='clip'
        )
        self.isotonic_model.fit(X_train.flatten(), y_train)
        
        # Fit Platt scaling (logistic regression)
        self.platt_model = LogisticRegression()
        self.platt_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_isotonic = self.isotonic_model.transform(X_val.flatten())
        val_platt = self.platt_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics for both methods
        isotonic_ece = self.reliability_generator.calculate_expected_calibration_error(val_isotonic, y_val)
        platt_ece = self.reliability_generator.calculate_expected_calibration_error(val_platt, y_val)
        
        isotonic_brier = brier_score_loss(y_val, val_isotonic)
        platt_brier = brier_score_loss(y_val, val_platt)
        
        # Choose better method
        if isotonic_ece <= platt_ece:
            best_method = "isotonic_regression"
            best_predictions = val_isotonic
            best_ece = isotonic_ece
            best_brier = isotonic_brier
        else:
            best_method = "platt_scaling"
            best_predictions = val_platt
            best_ece = platt_ece
            best_brier = platt_brier
        
        # Generate reliability diagram data
        reliability_data = self.reliability_generator.generate_diagram_data(best_predictions, y_val)
        
        # Calculate additional metrics
        reliability_score = 1.0 - best_ece  # Higher is better
        sharpness_score = np.std(best_predictions)  # Spread of predictions
        
        # Create calibration metrics
        self.calibration_metrics = CalibrationMetrics(
            expected_calibration_error=best_ece,
            brier_score=best_brier,
            reliability_score=reliability_score,
            sharpness_score=sharpness_score,
            confidence_histogram={'raw': raw_conf.tolist(), 'calibrated': best_predictions.tolist()},
            reliability_diagram_data=reliability_data,
            sample_count=len(calibration_data.raw_confidences)
        )
        
        # Update training metadata
        self.training_metadata = {
            'provider': self.provider_name,
            'training_samples': len(calibration_data.raw_confidences),
            'validation_samples': len(y_val),
            'best_method': best_method,
            'isotonic_ece': isotonic_ece,
            'platt_ece': platt_ece,
            'isotonic_brier': isotonic_brier,
            'platt_brier': platt_brier,
            'training_time': time.time(),
            'data_metadata': calibration_data.metadata
        }
        
        # Set the best method
        self.calibration_method = best_method
        
        # Save models
        self._save_models()
        
        self.logger.info(f"Calibration training complete for {self.provider_name}", 
                        context={
                            'best_method': best_method,
                            'ece': best_ece,
                            'brier_score': best_brier,
                            'samples': len(calibration_data.raw_confidences)
                        })
        
        return self.calibration_metrics
    
    def calibrate(self, raw_confidence: float) -> CalibratedResult:
        """
        Calibrate a raw confidence score
        
        Args:
            raw_confidence: Raw confidence from ASR provider
            
        Returns:
            CalibratedResult with calibrated confidence and metadata
        """
        if self.isotonic_model is None and self.platt_model is None:
            # No calibration available, return raw score
            return CalibratedResult(
                raw_confidence=raw_confidence,
                calibrated_confidence=np.clip(raw_confidence, 0.01, 0.99),
                calibration_method="none",
                provider=self.provider_name,
                confidence_delta=0.0,
                calibration_metadata={'reason': 'no_model_available'}
            )
        
        # Ensure valid input range
        clipped_raw = np.clip(raw_confidence, 0.001, 0.999)
        
        # Apply calibration based on method
        if self.calibration_method == "isotonic_regression" and self.isotonic_model is not None:
            calibrated = self.isotonic_model.transform([clipped_raw])[0]
        elif self.calibration_method == "platt_scaling" and self.platt_model is not None:
            calibrated = self.platt_model.predict_proba([[clipped_raw]])[0, 1]
        else:
            # Fallback to available method
            if self.isotonic_model is not None:
                calibrated = self.isotonic_model.transform([clipped_raw])[0]
            else:
                calibrated = self.platt_model.predict_proba([[clipped_raw]])[0, 1]
        
        # Ensure valid output range
        calibrated = np.clip(calibrated, 0.01, 0.99)
        
        confidence_delta = calibrated - raw_confidence
        
        return CalibratedResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=float(calibrated),
            calibration_method=self.calibration_method,
            provider=self.provider_name,
            confidence_delta=float(confidence_delta),
            calibration_metadata={
                'model_available': True,
                'training_samples': self.training_metadata.get('training_samples', 0),
                'model_ece': self.calibration_metrics.expected_calibration_error if self.calibration_metrics else None
            }
        )
    
    def _save_models(self):
        """Save calibration models to disk"""
        try:
            # Save isotonic model
            if self.isotonic_model is not None:
                isotonic_path = self.models_dir / f"{self.provider_name}_isotonic_calibration.pkl"
                with open(isotonic_path, 'wb') as f:
                    pickle.dump(self.isotonic_model, f)
            
            # Save Platt model
            if self.platt_model is not None:
                platt_path = self.models_dir / f"{self.provider_name}_platt_calibration.pkl"
                with open(platt_path, 'wb') as f:
                    pickle.dump(self.platt_model, f)
            
            # Save metadata
            metadata_path = self.models_dir / f"{self.provider_name}_calibration_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.training_metadata, f, indent=2)
            
            self.logger.info(f"Saved calibration models for {self.provider_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration models for {self.provider_name}: {e}")
    
    def _load_models(self):
        """Load existing calibration models from disk"""
        try:
            # Load isotonic model
            isotonic_path = self.models_dir / f"{self.provider_name}_isotonic_calibration.pkl"
            if isotonic_path.exists():
                with open(isotonic_path, 'rb') as f:
                    self.isotonic_model = pickle.load(f)
                self.logger.info(f"Loaded isotonic calibration model for {self.provider_name}")
            
            # Load Platt model
            platt_path = self.models_dir / f"{self.provider_name}_platt_calibration.pkl"
            if platt_path.exists():
                with open(platt_path, 'rb') as f:
                    self.platt_model = pickle.load(f)
                self.logger.info(f"Loaded Platt calibration model for {self.provider_name}")
            
            # Load metadata
            metadata_path = self.models_dir / f"{self.provider_name}_calibration_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.training_metadata = json.load(f)
                
                # Set calibration method from metadata
                if 'best_method' in self.training_metadata:
                    self.calibration_method = self.training_metadata['best_method']
            
        except Exception as e:
            self.logger.warning(f"Failed to load calibration models for {self.provider_name}: {e}")

class CalibrationEngine:
    """
    Main calibration engine managing per-provider calibration models
    
    Handles confidence calibration for OpenAI, Deepgram, and Faster-Whisper providers
    with separate models and training capabilities for each engine.
    """
    
    def __init__(self, 
                 models_dir: str = "calibration_models",
                 supported_providers: Optional[List[str]] = None):
        """
        Initialize calibration engine
        
        Args:
            models_dir: Directory to store calibration models
            supported_providers: List of supported ASR provider names
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.supported_providers = supported_providers or ['openai', 'deepgram', 'faster-whisper']
        
        self.logger = create_enhanced_logger("calibration_engine")
        
        # Per-engine calibration models
        self.engine_models: Dict[str, EngineCalibrationModel] = {}
        
        # Initialize calibration models for each provider
        for provider in self.supported_providers:
            self.engine_models[provider] = EngineCalibrationModel(
                provider_name=provider,
                models_dir=str(self.models_dir)
            )
        
        self.logger.info("CalibrationEngine initialized", 
                        context={'supported_providers': self.supported_providers,
                                'models_dir': str(self.models_dir)})
    
    def calibrate_confidence(self, 
                           raw_confidence: float, 
                           provider_name: str) -> CalibratedResult:
        """
        Calibrate confidence score for specific ASR provider
        
        Args:
            raw_confidence: Raw confidence from ASR provider
            provider_name: Name of ASR provider
            
        Returns:
            CalibratedResult with calibrated confidence
        """
        if provider_name not in self.engine_models:
            self.logger.warning(f"Unknown provider {provider_name}, using raw confidence")
            return CalibratedResult(
                raw_confidence=raw_confidence,
                calibrated_confidence=np.clip(raw_confidence, 0.01, 0.99),
                calibration_method="none",
                provider=provider_name,
                confidence_delta=0.0,
                calibration_metadata={'reason': 'unsupported_provider'}
            )
        
        return self.engine_models[provider_name].calibrate(raw_confidence)
    
    def train_provider_calibration(self, 
                                 provider_name: str, 
                                 calibration_data: CalibrationData) -> CalibrationMetrics:
        """
        Train calibration model for specific provider
        
        Args:
            provider_name: ASR provider name
            calibration_data: Training data with ground truth
            
        Returns:
            CalibrationMetrics with training results
        """
        if provider_name not in self.engine_models:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        self.logger.info(f"Training calibration for {provider_name}", 
                        context={'training_samples': len(calibration_data.raw_confidences)})
        
        return self.engine_models[provider_name].fit(calibration_data)
    
    def get_calibration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get calibration status for all providers"""
        status = {}
        
        for provider_name, model in self.engine_models.items():
            status[provider_name] = {
                'has_isotonic_model': model.isotonic_model is not None,
                'has_platt_model': model.platt_model is not None,
                'calibration_method': model.calibration_method,
                'training_metadata': model.training_metadata,
                'calibration_metrics': {
                    'ece': model.calibration_metrics.expected_calibration_error if model.calibration_metrics else None,
                    'brier_score': model.calibration_metrics.brier_score if model.calibration_metrics else None,
                    'sample_count': model.calibration_metrics.sample_count if model.calibration_metrics else 0
                } if model.calibration_metrics else None
            }
        
        return status
    
    def validate_calibration(self, 
                           provider_name: str, 
                           validation_data: CalibrationData) -> CalibrationMetrics:
        """
        Validate calibration quality on held-out data
        
        Args:
            provider_name: ASR provider name  
            validation_data: Validation data with ground truth
            
        Returns:
            CalibrationMetrics with validation results
        """
        if provider_name not in self.engine_models:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        model = self.engine_models[provider_name]
        
        if model.isotonic_model is None and model.platt_model is None:
            raise ValueError(f"No calibration model available for {provider_name}")
        
        # Get calibrated predictions
        calibrated_confidences = []
        for raw_conf in validation_data.raw_confidences:
            result = model.calibrate(raw_conf)
            calibrated_confidences.append(result.calibrated_confidence)
        
        calibrated_array = np.array(calibrated_confidences)
        true_labels = np.array(validation_data.ground_truth_labels)
        
        # Calculate validation metrics
        ece = model.reliability_generator.calculate_expected_calibration_error(calibrated_array, true_labels)
        brier_score = brier_score_loss(true_labels, calibrated_array)
        reliability_data = model.reliability_generator.generate_diagram_data(calibrated_array, true_labels)
        
        reliability_score = 1.0 - ece
        sharpness_score = np.std(calibrated_array)
        
        return CalibrationMetrics(
            expected_calibration_error=ece,
            brier_score=brier_score,
            reliability_score=reliability_score,
            sharpness_score=sharpness_score,
            confidence_histogram={
                'raw': validation_data.raw_confidences,
                'calibrated': calibrated_confidences
            },
            reliability_diagram_data=reliability_data,
            sample_count=len(validation_data.raw_confidences)
        )