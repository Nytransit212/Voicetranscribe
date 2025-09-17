"""
Calibration Training System for Per-Engine Confidence Calibration

Provides training infrastructure to fit calibration models on US English development sets,
generate reliability diagrams, and validate calibration quality across different ASR engines.
"""

import numpy as np
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

from .per_engine_calibration import CalibrationEngine, CalibrationData, CalibrationMetrics
from .asr_providers.factory import ASRProviderFactory
from .asr_providers.base import ASRProvider, ASRResult, DecodeMode
from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class TrainingDataItem:
    """Individual training sample with ground truth"""
    audio_path: str
    ground_truth_transcript: str
    ground_truth_words: List[Dict[str, Any]]  # Word-level ground truth with timings
    domain: str = "general"
    speaker_count: int = 1
    noise_level: str = "clean"
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CalibrationTrainingResult:
    """Results from calibration training session"""
    provider_metrics: Dict[str, CalibrationMetrics]
    training_summary: Dict[str, Any]
    validation_results: Dict[str, Any]
    total_training_time: float
    samples_processed: int

class CalibrationTrainer:
    """
    Trains and validates per-engine confidence calibration models
    
    Manages the process of collecting ASR results from different providers,
    computing ground truth comparisons, and fitting calibration models.
    """
    
    def __init__(self, 
                 calibration_engine: CalibrationEngine,
                 models_dir: str = "calibration_models",
                 dev_set_path: Optional[str] = None):
        """
        Initialize calibration trainer
        
        Args:
            calibration_engine: CalibrationEngine instance to train
            models_dir: Directory for storing calibration models
            dev_set_path: Path to US English development set
        """
        self.calibration_engine = calibration_engine
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.logger = create_enhanced_logger("calibration_trainer")
        
        # Development set configuration
        self.dev_set_path = dev_set_path
        
        # Training configuration
        self.supported_providers = ['openai', 'deepgram', 'faster-whisper']
        self.training_decode_modes = [DecodeMode.DETERMINISTIC, DecodeMode.CAREFUL]
        
        # Initialize ASR providers for training
        self.providers = self._initialize_training_providers()
        
        self.logger.info("CalibrationTrainer initialized", 
                        context={'models_dir': str(self.models_dir),
                                'dev_set_path': dev_set_path,
                                'available_providers': list(self.providers.keys())})
    
    def _initialize_training_providers(self) -> Dict[str, ASRProvider]:
        """Initialize ASR providers for training"""
        providers = {}
        
        for provider_name in self.supported_providers:
            try:
                provider = ASRProviderFactory.create_provider(provider_name)
                if provider.is_available():
                    providers[provider_name] = provider
                    self.logger.info(f"Training provider initialized: {provider_name}")
                else:
                    self.logger.warning(f"Provider {provider_name} not available for training")
            except Exception as e:
                self.logger.error(f"Failed to initialize training provider {provider_name}: {e}")
        
        return providers
    
    def train_all_providers(self, 
                          training_data: List[TrainingDataItem],
                          validation_split: float = 0.2) -> CalibrationTrainingResult:
        """
        Train calibration models for all available providers
        
        Args:
            training_data: List of training samples with ground truth
            validation_split: Fraction of data to use for validation
            
        Returns:
            CalibrationTrainingResult with training metrics
        """
        if not training_data:
            raise ValueError("No training data provided")
        
        self.logger.info(f"Starting calibration training for {len(self.supported_providers)} providers",
                        context={'training_samples': len(training_data),
                                'validation_split': validation_split})
        
        start_time = time.time()
        provider_metrics = {}
        validation_results = {}
        samples_processed = 0
        
        # Split data into train/validation
        random.shuffle(training_data)
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        self.logger.info(f"Data split: {len(train_data)} training, {len(val_data)} validation")
        
        # Train each provider
        for provider_name in self.supported_providers:
            if provider_name not in self.providers:
                self.logger.warning(f"Skipping {provider_name} - provider not available")
                continue
            
            try:
                self.logger.info(f"Training calibration for {provider_name}")
                
                # Collect training data for this provider
                provider_train_data = self._collect_provider_training_data(
                    provider_name, train_data
                )
                
                if not provider_train_data.raw_confidences:
                    self.logger.warning(f"No training data collected for {provider_name}")
                    continue
                
                # Train calibration models
                metrics = self.calibration_engine.train_provider_calibration(
                    provider_name, provider_train_data
                )
                provider_metrics[provider_name] = metrics
                samples_processed += len(provider_train_data.raw_confidences)
                
                # Validate on held-out data
                if val_data:
                    provider_val_data = self._collect_provider_training_data(
                        provider_name, val_data
                    )
                    
                    if provider_val_data.raw_confidences:
                        val_metrics = self.calibration_engine.validate_calibration(
                            provider_name, provider_val_data
                        )
                        validation_results[provider_name] = val_metrics
                
                self.logger.info(f"Calibration training complete for {provider_name}",
                               context={'ece': metrics.expected_calibration_error,
                                       'brier_score': metrics.brier_score,
                                       'samples': metrics.sample_count})
                
            except Exception as e:
                self.logger.error(f"Failed to train calibration for {provider_name}: {e}")
        
        training_time = time.time() - start_time
        
        # Create summary
        training_summary = {
            'total_providers_trained': len(provider_metrics),
            'total_training_time': training_time,
            'average_ece': np.mean([m.expected_calibration_error for m in provider_metrics.values()]),
            'average_brier_score': np.mean([m.brier_score for m in provider_metrics.values()]),
            'training_data_samples': len(train_data),
            'validation_data_samples': len(val_data),
            'timestamp': time.time()
        }
        
        self.logger.info("Calibration training complete",
                        context={
                            'providers_trained': len(provider_metrics),
                            'total_time': training_time,
                            'samples_processed': samples_processed
                        })
        
        return CalibrationTrainingResult(
            provider_metrics=provider_metrics,
            training_summary=training_summary,
            validation_results=validation_results,
            total_training_time=training_time,
            samples_processed=samples_processed
        )
    
    def _collect_provider_training_data(self, 
                                      provider_name: str, 
                                      training_samples: List[TrainingDataItem]) -> CalibrationData:
        """
        Collect training data for specific provider by running ASR and computing accuracy
        
        Args:
            provider_name: ASR provider name
            training_samples: Training data samples
            
        Returns:
            CalibrationData with raw confidences and ground truth labels
        """
        provider = self.providers.get(provider_name)
        if not provider:
            return CalibrationData(
                provider=provider_name,
                raw_confidences=[],
                ground_truth_labels=[],
                segment_lengths=[],
                metadata={'error': 'provider_not_available'}
            )
        
        raw_confidences = []
        ground_truth_labels = []
        segment_lengths = []
        
        self.logger.info(f"Collecting training data for {provider_name}",
                        context={'samples': len(training_samples)})
        
        for i, sample in enumerate(training_samples):
            try:
                # Run ASR with multiple decode modes
                for decode_mode in self.training_decode_modes:
                    result = provider.transcribe(
                        sample.audio_path,
                        decode_mode=decode_mode,
                        language="en"
                    )
                    
                    # Compare against ground truth and collect confidence/accuracy pairs
                    for segment in result.segments:
                        # Calculate accuracy by comparing with ground truth
                        accuracy = self._calculate_segment_accuracy(
                            segment.text,
                            sample.ground_truth_transcript,
                            segment.start,
                            segment.end,
                            sample.ground_truth_words
                        )
                        
                        # Use raw confidence before any calibration
                        raw_confidence = segment.confidence
                        
                        # Convert accuracy to binary label (1 for good, 0 for poor)
                        # Use threshold of 0.8 word-level accuracy
                        is_accurate = 1 if accuracy >= 0.8 else 0
                        
                        raw_confidences.append(raw_confidence)
                        ground_truth_labels.append(is_accurate)
                        segment_lengths.append(segment.end - segment.start)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(training_samples)} samples for {provider_name}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to process sample {i} for {provider_name}: {e}")
                continue
        
        self.logger.info(f"Training data collection complete for {provider_name}",
                        context={'confidence_values': len(raw_confidences),
                                'positive_labels': sum(ground_truth_labels),
                                'negative_labels': len(ground_truth_labels) - sum(ground_truth_labels)})
        
        return CalibrationData(
            provider=provider_name,
            raw_confidences=raw_confidences,
            ground_truth_labels=ground_truth_labels,
            segment_lengths=segment_lengths,
            metadata={
                'training_samples': len(training_samples),
                'decode_modes': [mode.value for mode in self.training_decode_modes],
                'accuracy_threshold': 0.8
            }
        )
    
    def _calculate_segment_accuracy(self,
                                  predicted_text: str,
                                  ground_truth_transcript: str,
                                  segment_start: float,
                                  segment_end: float,
                                  ground_truth_words: List[Dict[str, Any]]) -> float:
        """
        Calculate word-level accuracy for a transcription segment
        
        Args:
            predicted_text: ASR predicted text
            ground_truth_transcript: Full ground truth transcript
            segment_start: Segment start time
            segment_end: Segment end time
            ground_truth_words: Word-level ground truth with timings
            
        Returns:
            Word-level accuracy score (0.0 to 1.0)
        """
        # Extract ground truth words for this time segment
        segment_gt_words = []
        for word_data in ground_truth_words:
            word_start = word_data.get('start', 0.0)
            word_end = word_data.get('end', 0.0)
            
            # Check if word overlaps with segment
            if (word_start < segment_end and word_end > segment_start):
                segment_gt_words.append(word_data.get('word', '').lower().strip())
        
        # Tokenize predicted text
        predicted_words = predicted_text.lower().split()
        
        if not segment_gt_words or not predicted_words:
            return 0.0
        
        # Calculate word-level accuracy using simple overlap
        # More sophisticated metrics like WER could be used here
        gt_words_set = set(segment_gt_words)
        pred_words_set = set(predicted_words)
        
        if not gt_words_set:
            return 1.0 if not pred_words_set else 0.0
        
        # Jaccard similarity (intersection over union)
        intersection = len(gt_words_set.intersection(pred_words_set))
        union = len(gt_words_set.union(pred_words_set))
        
        accuracy = intersection / union if union > 0 else 0.0
        
        return accuracy
    
    def create_synthetic_dev_set(self, 
                               num_samples: int = 100,
                               duration_range: Tuple[float, float] = (10.0, 60.0)) -> List[TrainingDataItem]:
        """
        Create a synthetic development set for testing calibration training
        
        Args:
            num_samples: Number of synthetic samples to create
            duration_range: Range of audio durations in seconds
            
        Returns:
            List of synthetic TrainingDataItem objects
        """
        self.logger.info(f"Creating synthetic development set with {num_samples} samples")
        
        synthetic_data = []
        
        # Sample sentences for synthetic data
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "This is a test of the emergency broadcast system.",
            "Please transcribe this audio file with high accuracy.",
            "Meeting participants discussed quarterly financial results.",
            "The weather forecast predicts rain and thunderstorms.",
            "Technology companies are investing in artificial intelligence.",
            "Customer service representatives handle inquiries professionally.",
            "Project management requires clear communication and planning.",
            "Healthcare workers provide essential services to communities.",
            "Education systems adapt to digital learning environments."
        ]
        
        for i in range(num_samples):
            # Create synthetic sample
            text = random.choice(sample_texts)
            duration = random.uniform(duration_range[0], duration_range[1])
            
            # Create synthetic word-level timings
            words = text.split()
            word_duration = duration / len(words)
            word_timings = []
            
            for j, word in enumerate(words):
                start_time = j * word_duration
                end_time = (j + 1) * word_duration
                word_timings.append({
                    'word': word.lower().strip('.,!?'),
                    'start': start_time,
                    'end': end_time
                })
            
            synthetic_item = TrainingDataItem(
                audio_path=f"synthetic_audio_{i:04d}.wav",  # Placeholder path
                ground_truth_transcript=text,
                ground_truth_words=word_timings,
                domain="general",
                speaker_count=1,
                noise_level="clean",
                duration=duration,
                metadata={'synthetic': True, 'sample_id': i}
            )
            
            synthetic_data.append(synthetic_item)
        
        self.logger.info(f"Created {len(synthetic_data)} synthetic training samples")
        return synthetic_data
    
    def generate_calibration_report(self, 
                                  training_result: CalibrationTrainingResult,
                                  output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive calibration training report
        
        Args:
            training_result: Results from calibration training
            output_path: Optional path to save JSON report
            
        Returns:
            Dictionary with comprehensive calibration report
        """
        report = {
            'training_summary': training_result.training_summary,
            'provider_results': {},
            'validation_summary': {},
            'recommendations': []
        }
        
        # Provider-specific results
        for provider_name, metrics in training_result.provider_metrics.items():
            provider_report = {
                'expected_calibration_error': metrics.expected_calibration_error,
                'brier_score': metrics.brier_score,
                'reliability_score': metrics.reliability_score,
                'sharpness_score': metrics.sharpness_score,
                'sample_count': metrics.sample_count,
                'calibration_quality': self._assess_calibration_quality(metrics)
            }
            
            report['provider_results'][provider_name] = provider_report
        
        # Validation summary
        if training_result.validation_results:
            validation_eces = [v.expected_calibration_error for v in training_result.validation_results.values()]
            validation_briers = [v.brier_score for v in training_result.validation_results.values()]
            
            report['validation_summary'] = {
                'average_ece': np.mean(validation_eces),
                'average_brier_score': np.mean(validation_briers),
                'providers_validated': len(training_result.validation_results)
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_calibration_recommendations(training_result)
        
        # Save report if path provided
        if output_path:
            report_path = Path(output_path)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Calibration report saved to {report_path}")
        
        return report
    
    def _assess_calibration_quality(self, metrics: CalibrationMetrics) -> str:
        """Assess calibration quality based on metrics"""
        if metrics.expected_calibration_error < 0.05:
            return "excellent"
        elif metrics.expected_calibration_error < 0.10:
            return "good"
        elif metrics.expected_calibration_error < 0.15:
            return "fair"
        else:
            return "poor"
    
    def _generate_calibration_recommendations(self, 
                                            training_result: CalibrationTrainingResult) -> List[str]:
        """Generate recommendations based on training results"""
        recommendations = []
        
        avg_ece = training_result.training_summary.get('average_ece', 0.0)
        
        if avg_ece > 0.10:
            recommendations.append("Consider collecting more diverse training data to improve calibration")
        
        if avg_ece < 0.05:
            recommendations.append("Excellent calibration achieved - ready for production use")
        
        # Provider-specific recommendations
        for provider_name, metrics in training_result.provider_metrics.items():
            if metrics.expected_calibration_error > 0.15:
                recommendations.append(f"Provider {provider_name} shows poor calibration - investigate data quality")
        
        return recommendations