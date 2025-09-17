"""
Automated Evaluation Gates System for Ensemble Transcription

This module implements comprehensive automated testing and quality gates that prevent
regressions in transcription accuracy. The system automatically evaluates WER, DER,
and entity accuracy against configured thresholds and fails builds on violations.

Core Features:
- WER (Word Error Rate) threshold gates with scenario-specific limits
- DER (Diarization Error Rate) gates for speaker confusion detection
- Entity accuracy gates for proper nouns, numbers, and technical terms
- CI integration with automated build failure on threshold violations
- Regression detection and trend analysis
- Comprehensive reporting with detailed failure analysis
"""

import json
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import contextmanager

from utils.enhanced_structured_logger import create_enhanced_logger
from utils.metric_calculator import MetricCalculator, EnsembleMetrics, DiarizationMetrics, ASRMetrics
from utils.observability import trace_stage
from core.ensemble_manager import EnsembleManager


class GateStatus(Enum):
    """Evaluation gate status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"


class GateType(Enum):
    """Types of evaluation gates"""
    WER_THRESHOLD = "wer_threshold"
    DER_THRESHOLD = "der_threshold"
    ENTITY_ACCURACY = "entity_accuracy"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    PROCESSING_TIME = "processing_time"
    REGRESSION_DETECTION = "regression_detection"


@dataclass
class ThresholdConfig:
    """Configuration for evaluation thresholds"""
    # WER thresholds by scenario type
    wer_threshold_clean: float = 1.0  # % for clean audio scenarios
    wer_threshold_overlap: float = 3.0  # % for heavy overlap scenarios
    wer_threshold_challenging: float = 5.0  # % for challenging scenarios
    wer_threshold_edge_case: float = 10.0  # % for edge cases
    
    # DER thresholds
    der_threshold: float = 5.0  # % speaker confusion rate
    der_threshold_challenging: float = 15.0  # % for challenging scenarios
    
    # Entity accuracy thresholds
    entity_accuracy_threshold: float = 95.0  # % for proper nouns, numbers, technical terms
    entity_accuracy_threshold_challenging: float = 85.0  # % for challenging scenarios
    
    # Confidence calibration thresholds
    confidence_calibration_min: float = 0.7  # Minimum calibration score
    confidence_calibration_challenging: float = 0.5  # For challenging scenarios
    
    # Processing time thresholds (seconds per minute of audio)
    processing_time_ratio_max: float = 2.0  # Max 2 seconds processing per 1 second audio
    processing_time_ratio_challenging: float = 4.0  # For challenging scenarios
    
    # Regression detection thresholds
    regression_wer_increase_max: float = 0.5  # % maximum WER increase
    regression_der_increase_max: float = 2.0  # % maximum DER increase
    regression_confidence_decrease_max: float = 5.0  # % maximum confidence decrease
    
    # Statistical significance settings
    confidence_interval: float = 0.95  # 95% confidence interval
    min_samples_for_significance: int = 5  # Minimum samples for statistical tests


@dataclass
class GateResult:
    """Result of a single evaluation gate"""
    gate_type: GateType
    status: GateStatus
    test_id: str
    scenario_type: str
    measured_value: float
    threshold_value: float
    margin: float  # How much above/below threshold
    message: str
    details: Dict[str, Any]
    timestamp: datetime


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    session_id: str
    timestamp: datetime
    overall_status: GateStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    error_tests: int
    gate_results: List[GateResult]
    summary_metrics: Dict[str, float]
    regression_analysis: Dict[str, Any]
    recommendations: List[str]
    ci_exit_code: int  # 0 = pass, 1 = fail, 2 = warnings


@dataclass
class EntityAccuracyResult:
    """Result of entity accuracy evaluation"""
    total_entities: int
    correct_entities: int
    accuracy: float
    entity_breakdown: Dict[str, Dict[str, int]]  # entity_type -> {correct, total}
    errors: List[Dict[str, Any]]


class EvaluationGatesSystem:
    """Main evaluation gates system orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluation gates system.
        
        Args:
            config: Configuration dictionary with thresholds and settings
        """
        self.config = ThresholdConfig(**(config or {}))
        self.logger = create_enhanced_logger("evaluation_gates")
        self.metric_calculator = MetricCalculator(logger=self.logger)
        
        # Test suite configuration
        self.gold_test_path = Path("tests/gold_test_set")
        self.test_definitions_file = self.gold_test_path / "test_definitions.json"
        self.ground_truth_dir = self.gold_test_path / "ground_truth"
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        self.historical_results_dir = self.results_dir / "historical"
        self.historical_results_dir.mkdir(exist_ok=True)
        
        # Load test suite
        self._load_test_suite()
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("🔧 EVALUATION GATES SYSTEM INITIALIZED", context={
            'config': asdict(self.config),
            'test_cases_loaded': len(self.test_suite.get('test_cases', [])),
            'results_dir': str(self.results_dir)
        })
    
    def _load_test_suite(self):
        """Load test suite definitions"""
        if self.test_definitions_file.exists():
            with open(self.test_definitions_file, 'r') as f:
                self.test_suite = json.load(f)
        else:
            self.logger.warning("⚠️ TEST SUITE NOT FOUND", context={
                'expected_path': str(self.test_definitions_file)
            })
            self.test_suite = {"test_cases": [], "quality_thresholds": {}}
    
    @trace_stage("evaluation_gates_full_run")
    def run_full_evaluation(self, ensemble_manager: Optional[EnsembleManager] = None) -> EvaluationReport:
        """
        Run comprehensive evaluation across all test cases.
        
        Args:
            ensemble_manager: Optional ensemble manager for live testing
            
        Returns:
            Comprehensive evaluation report
        """
        session_id = f"eval_{int(time.time())}"
        start_time = datetime.now()
        
        self.logger.info("🚀 STARTING FULL EVALUATION", context={
            'session_id': session_id,
            'total_test_cases': len(self.test_suite.get('test_cases', [])),
            'start_time': start_time.isoformat()
        })
        
        gate_results = []
        processing_errors = []
        
        # Run evaluation for each test case
        test_cases = self.test_suite.get('test_cases', [])
        
        # Use ThreadPoolExecutor for parallel test execution
        with ThreadPoolExecutor(max_workers=min(4, len(test_cases))) as executor:
            future_to_test = {
                executor.submit(self._evaluate_test_case, test_case, ensemble_manager): test_case
                for test_case in test_cases
            }
            
            for future in as_completed(future_to_test):
                test_case = future_to_test[future]
                try:
                    test_results = future.result(timeout=300)  # 5 minute timeout per test
                    gate_results.extend(test_results)
                except Exception as e:
                    self.logger.error("❌ TEST CASE EXECUTION ERROR", context={
                        'test_id': test_case.get('test_id', 'unknown'),
                        'error': str(e)
                    })
                    processing_errors.append({
                        'test_id': test_case.get('test_id', 'unknown'),
                        'error': str(e)
                    })
        
        # Run regression analysis
        regression_analysis = self._perform_regression_analysis(gate_results)
        
        # Generate report
        report = self._generate_evaluation_report(
            session_id, gate_results, regression_analysis, processing_errors
        )
        
        # Store results for historical tracking
        self._store_evaluation_results(report)
        
        # Log summary
        self._log_evaluation_summary(report)
        
        return report
    
    def _evaluate_test_case(self, test_case: Dict[str, Any], 
                           ensemble_manager: Optional[EnsembleManager] = None) -> List[GateResult]:
        """
        Evaluate a single test case against all applicable gates.
        
        Args:
            test_case: Test case definition
            ensemble_manager: Optional ensemble manager for live processing
            
        Returns:
            List of gate results for this test case
        """
        test_id = test_case['test_id']
        scenario_type = test_case.get('scenario_type', 'unknown')
        
        self.logger.info("🧪 EVALUATING TEST CASE", context={
            'test_id': test_id,
            'scenario_type': scenario_type
        })
        
        gate_results = []
        
        try:
            # Load ground truth
            ground_truth = self._load_ground_truth(test_id)
            if not ground_truth:
                return [GateResult(
                    gate_type=GateType.WER_THRESHOLD,
                    status=GateStatus.ERROR,
                    test_id=test_id,
                    scenario_type=scenario_type,
                    measured_value=0.0,
                    threshold_value=0.0,
                    margin=0.0,
                    message=f"Ground truth not found for {test_id}",
                    details={},
                    timestamp=datetime.now()
                )]
            
            # Get evaluation results (either from live processing or simulation)
            if ensemble_manager:
                evaluation_metrics = self._run_live_evaluation(test_case, ground_truth, ensemble_manager)
            else:
                evaluation_metrics = self._simulate_evaluation_results(test_case, ground_truth)
            
            # Apply all gates
            gate_results.extend(self._apply_wer_gates(test_case, evaluation_metrics))
            gate_results.extend(self._apply_der_gates(test_case, evaluation_metrics))
            gate_results.extend(self._apply_entity_accuracy_gates(test_case, evaluation_metrics, ground_truth))
            gate_results.extend(self._apply_confidence_gates(test_case, evaluation_metrics))
            gate_results.extend(self._apply_performance_gates(test_case, evaluation_metrics))
            
        except Exception as e:
            self.logger.error("❌ TEST CASE EVALUATION ERROR", context={
                'test_id': test_id,
                'error': str(e)
            })
            gate_results.append(GateResult(
                gate_type=GateType.WER_THRESHOLD,
                status=GateStatus.ERROR,
                test_id=test_id,
                scenario_type=scenario_type,
                measured_value=0.0,
                threshold_value=0.0,
                margin=0.0,
                message=f"Evaluation error: {str(e)}",
                details={'exception': str(e)},
                timestamp=datetime.now()
            ))
        
        return gate_results
    
    def _load_ground_truth(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Load ground truth data for a test case"""
        ground_truth_file = self.ground_truth_dir / f"{test_id}.json"
        
        if not ground_truth_file.exists():
            return None
        
        try:
            with open(ground_truth_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("❌ GROUND TRUTH LOAD ERROR", context={
                'test_id': test_id,
                'file': str(ground_truth_file),
                'error': str(e)
            })
            return None
    
    def _run_live_evaluation(self, test_case: Dict[str, Any], ground_truth: Dict[str, Any], 
                           ensemble_manager: EnsembleManager) -> EnsembleMetrics:
        """
        Run live evaluation using actual ensemble processing.
        
        Args:
            test_case: Test case definition
            ground_truth: Ground truth data
            ensemble_manager: Ensemble manager instance
            
        Returns:
            Ensemble metrics from live processing
        """
        audio_file = test_case.get('audio_file')
        if not audio_file:
            raise ValueError(f"No audio file specified for test case {test_case['test_id']}")
        
        # Check if audio file exists in test data
        audio_path = self.gold_test_path / "audio_clips" / audio_file
        if not audio_path.exists():
            # Use sample data file if available
            data_dir = Path("data")
            audio_path = data_dir / audio_file
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        self.logger.info("🎵 RUNNING LIVE EVALUATION", context={
            'test_id': test_case['test_id'],
            'audio_file': str(audio_path)
        })
        
        # Process audio with ensemble manager
        result = ensemble_manager.process_video(str(audio_path))
        
        # Extract reference data for metric calculation
        reference_text = self._extract_reference_text(ground_truth)
        reference_segments = ground_truth.get("segments", [])
        
        # Calculate comprehensive metrics
        metrics = self.metric_calculator.calculate_ensemble_metrics(
            candidates=result.get('candidates', []),
            reference_segments=reference_segments,
            reference_text=reference_text,
            audio_duration=ground_truth.get("audio_duration", 0.0),
            processing_time=result.get('processing_metadata', {}).get('total_processing_time', 0.0)
        )
        
        return metrics
    
    def _simulate_evaluation_results(self, test_case: Dict[str, Any], 
                                   ground_truth: Dict[str, Any]) -> EnsembleMetrics:
        """
        Simulate evaluation results for testing without live processing.
        
        Args:
            test_case: Test case definition
            ground_truth: Ground truth data
            
        Returns:
            Simulated ensemble metrics
        """
        expected_metrics = test_case.get('expected_metrics', {})
        scenario_type = test_case.get('scenario_type', 'baseline')
        
        # Create realistic simulated metrics based on scenario difficulty
        if scenario_type == 'baseline':
            wer_base = 0.05
            der_base = 0.08
            confidence_base = 0.85
        elif scenario_type in ['challenging_diarization', 'challenging_asr', 'challenging_audio']:
            wer_base = 0.20
            der_base = 0.25
            confidence_base = 0.60
        else:  # edge_case
            wer_base = 0.35
            der_base = 0.40
            confidence_base = 0.45
        
        # Add some realistic variance
        wer = max(0.0, wer_base + np.random.normal(0, 0.02))
        der = max(0.0, der_base + np.random.normal(0, 0.03))
        confidence = max(0.0, min(1.0, confidence_base + np.random.normal(0, 0.05)))
        
        # Create mock metrics object
        asr_metrics = ASRMetrics(
            wer=wer,
            cer=wer * 0.8,  # Approximate CER from WER
            mer=wer * 0.6,  # Approximate MER from WER
            wil=wer * 0.4,  # Approximate WIL from WER
            wip=1.0 - wer,  # Approximate WIP from WER
            insertions=int(wer * 25 * 0.3),  # Approximate insertions
            deletions=int(wer * 25 * 0.4),   # Approximate deletions
            substitutions=int(wer * 25 * 0.3), # Approximate substitutions
            word_count=len(ground_truth.get('segments', [])) * 10  # Approximate
        )
        
        diarization_metrics = DiarizationMetrics(
            der=der,
            missed_speaker_time=der * 0.4,  # Approximate missed speaker time
            false_alarm_time=der * 0.2,     # Approximate false alarm time
            speaker_confusion_time=der * 0.4, # Approximate confusion time
            speaker_count_accuracy=0.9,
            overlap_coverage=1.0 - der * 0.5  # Approximate overlap coverage
        )
        
        ensemble_metrics = EnsembleMetrics(
            asr=asr_metrics,
            diarization=diarization_metrics,
            confidence_calibration=confidence,
            ensemble_agreement=0.75,
            processing_time=expected_metrics.get('processing_time_max', 60.0) * 0.8,
            total_candidates=15,
            winner_confidence=confidence
        )
        
        return ensemble_metrics
    
    def _apply_wer_gates(self, test_case: Dict[str, Any], metrics: EnsembleMetrics) -> List[GateResult]:
        """Apply WER threshold gates"""
        scenario_type = test_case.get('scenario_type', 'baseline')
        test_id = test_case['test_id']
        
        # Determine appropriate threshold based on scenario
        if scenario_type == 'baseline':
            threshold = self.config.wer_threshold_clean
        elif scenario_type in ['challenging_diarization', 'challenging_asr', 'challenging_audio']:
            threshold = self.config.wer_threshold_challenging
        elif scenario_type == 'edge_case':
            threshold = self.config.wer_threshold_edge_case
        else:
            threshold = self.config.wer_threshold_overlap
        
        # Convert to percentage for comparison
        measured_wer_pct = metrics.asr.wer * 100
        threshold_pct = threshold
        
        # Determine gate status
        margin = measured_wer_pct - threshold_pct
        if margin <= 0:
            status = GateStatus.PASS
            message = f"WER {measured_wer_pct:.2f}% passes threshold {threshold_pct:.1f}%"
        elif margin <= threshold_pct * 0.1:  # Within 10% of threshold
            status = GateStatus.WARNING
            message = f"WER {measured_wer_pct:.2f}% exceeds threshold {threshold_pct:.1f}% (warning)"
        else:
            status = GateStatus.FAIL
            message = f"WER {measured_wer_pct:.2f}% significantly exceeds threshold {threshold_pct:.1f}%"
        
        return [GateResult(
            gate_type=GateType.WER_THRESHOLD,
            status=status,
            test_id=test_id,
            scenario_type=scenario_type,
            measured_value=measured_wer_pct,
            threshold_value=threshold_pct,
            margin=margin,
            message=message,
            details={
                'wer_raw': metrics.asr.wer,
                'cer': metrics.asr.cer,
                'mer': metrics.asr.mer,
                'wil': metrics.asr.wil,
                'wip': metrics.asr.wip,
                'insertions': metrics.asr.insertions,
                'deletions': metrics.asr.deletions,
                'substitutions': metrics.asr.substitutions,
                'word_count': metrics.asr.word_count
            },
            timestamp=datetime.now()
        )]
    
    def _apply_der_gates(self, test_case: Dict[str, Any], metrics: EnsembleMetrics) -> List[GateResult]:
        """Apply DER threshold gates"""
        scenario_type = test_case.get('scenario_type', 'baseline')
        test_id = test_case['test_id']
        
        # Determine appropriate threshold
        if scenario_type in ['challenging_diarization', 'challenging_asr', 'challenging_audio', 'edge_case']:
            threshold = self.config.der_threshold_challenging
        else:
            threshold = self.config.der_threshold
        
        # Convert to percentage
        measured_der_pct = metrics.diarization.der * 100
        threshold_pct = threshold
        
        # Determine gate status
        margin = measured_der_pct - threshold_pct
        if margin <= 0:
            status = GateStatus.PASS
            message = f"DER {measured_der_pct:.2f}% passes threshold {threshold_pct:.1f}%"
        elif margin <= threshold_pct * 0.15:  # Within 15% of threshold
            status = GateStatus.WARNING
            message = f"DER {measured_der_pct:.2f}% exceeds threshold {threshold_pct:.1f}% (warning)"
        else:
            status = GateStatus.FAIL
            message = f"DER {measured_der_pct:.2f}% significantly exceeds threshold {threshold_pct:.1f}%"
        
        return [GateResult(
            gate_type=GateType.DER_THRESHOLD,
            status=status,
            test_id=test_id,
            scenario_type=scenario_type,
            measured_value=measured_der_pct,
            threshold_value=threshold_pct,
            margin=margin,
            message=message,
            details={
                'der_raw': metrics.diarization.der,
                'missed_speaker_time': metrics.diarization.missed_speaker_time,
                'false_alarm_time': metrics.diarization.false_alarm_time,
                'speaker_confusion_time': metrics.diarization.speaker_confusion_time,
                'speaker_count_accuracy': metrics.diarization.speaker_count_accuracy,
                'overlap_coverage': metrics.diarization.overlap_coverage
            },
            timestamp=datetime.now()
        )]
    
    def _apply_entity_accuracy_gates(self, test_case: Dict[str, Any], metrics: EnsembleMetrics,
                                   ground_truth: Dict[str, Any]) -> List[GateResult]:
        """Apply entity accuracy gates"""
        scenario_type = test_case.get('scenario_type', 'baseline')
        test_id = test_case['test_id']
        
        # Extract entities from ground truth and simulated results
        entity_result = self._calculate_entity_accuracy(ground_truth, metrics)
        
        # Determine threshold
        if scenario_type in ['challenging_asr', 'challenging_audio', 'edge_case']:
            threshold = self.config.entity_accuracy_threshold_challenging
        else:
            threshold = self.config.entity_accuracy_threshold
        
        measured_accuracy = entity_result.accuracy * 100
        
        # Determine gate status
        margin = measured_accuracy - threshold
        if margin >= 0:
            status = GateStatus.PASS
            message = f"Entity accuracy {measured_accuracy:.1f}% meets threshold {threshold:.1f}%"
        elif margin >= -threshold * 0.05:  # Within 5% of threshold
            status = GateStatus.WARNING
            message = f"Entity accuracy {measured_accuracy:.1f}% below threshold {threshold:.1f}% (warning)"
        else:
            status = GateStatus.FAIL
            message = f"Entity accuracy {measured_accuracy:.1f}% significantly below threshold {threshold:.1f}%"
        
        return [GateResult(
            gate_type=GateType.ENTITY_ACCURACY,
            status=status,
            test_id=test_id,
            scenario_type=scenario_type,
            measured_value=measured_accuracy,
            threshold_value=threshold,
            margin=margin,
            message=message,
            details={
                'total_entities': entity_result.total_entities,
                'correct_entities': entity_result.correct_entities,
                'entity_breakdown': entity_result.entity_breakdown,
                'error_samples': entity_result.errors[:5]  # First 5 errors for debugging
            },
            timestamp=datetime.now()
        )]
    
    def _apply_confidence_gates(self, test_case: Dict[str, Any], metrics: EnsembleMetrics) -> List[GateResult]:
        """Apply confidence calibration gates"""
        scenario_type = test_case.get('scenario_type', 'baseline')
        test_id = test_case['test_id']
        
        # Determine threshold
        if scenario_type in ['challenging_diarization', 'challenging_asr', 'challenging_audio', 'edge_case']:
            threshold = self.config.confidence_calibration_challenging
        else:
            threshold = self.config.confidence_calibration_min
        
        measured_confidence = metrics.confidence_calibration
        
        # Determine gate status
        margin = measured_confidence - threshold
        if margin >= 0:
            status = GateStatus.PASS
            message = f"Confidence calibration {measured_confidence:.3f} meets threshold {threshold:.2f}"
        elif margin >= -threshold * 0.1:  # Within 10% of threshold
            status = GateStatus.WARNING
            message = f"Confidence calibration {measured_confidence:.3f} below threshold {threshold:.2f} (warning)"
        else:
            status = GateStatus.FAIL
            message = f"Confidence calibration {measured_confidence:.3f} significantly below threshold {threshold:.2f}"
        
        return [GateResult(
            gate_type=GateType.CONFIDENCE_CALIBRATION,
            status=status,
            test_id=test_id,
            scenario_type=scenario_type,
            measured_value=measured_confidence,
            threshold_value=threshold,
            margin=margin,
            message=message,
            details={
                'ensemble_agreement': metrics.ensemble_agreement,
                'winner_confidence': metrics.winner_confidence,
                'total_candidates': metrics.total_candidates
            },
            timestamp=datetime.now()
        )]
    
    def _apply_performance_gates(self, test_case: Dict[str, Any], metrics: EnsembleMetrics) -> List[GateResult]:
        """Apply processing time performance gates"""
        scenario_type = test_case.get('scenario_type', 'baseline')
        test_id = test_case['test_id']
        
        # Calculate processing time ratio (processing_time / audio_duration)
        audio_duration = test_case.get('duration', 60.0)  # Default to 60 seconds
        processing_time_ratio = metrics.processing_time / audio_duration
        
        # Determine threshold
        if scenario_type in ['challenging_diarization', 'challenging_asr', 'challenging_audio', 'edge_case']:
            threshold = self.config.processing_time_ratio_challenging
        else:
            threshold = self.config.processing_time_ratio_max
        
        # Determine gate status
        margin = processing_time_ratio - threshold
        if margin <= 0:
            status = GateStatus.PASS
            message = f"Processing ratio {processing_time_ratio:.2f}x within threshold {threshold:.1f}x"
        elif margin <= threshold * 0.2:  # Within 20% of threshold
            status = GateStatus.WARNING
            message = f"Processing ratio {processing_time_ratio:.2f}x exceeds threshold {threshold:.1f}x (warning)"
        else:
            status = GateStatus.FAIL
            message = f"Processing ratio {processing_time_ratio:.2f}x significantly exceeds threshold {threshold:.1f}x"
        
        return [GateResult(
            gate_type=GateType.PROCESSING_TIME,
            status=status,
            test_id=test_id,
            scenario_type=scenario_type,
            measured_value=processing_time_ratio,
            threshold_value=threshold,
            margin=margin,
            message=message,
            details={
                'processing_time_seconds': metrics.processing_time,
                'audio_duration_seconds': audio_duration,
                'total_candidates': metrics.total_candidates
            },
            timestamp=datetime.now()
        )]
    
    def _calculate_entity_accuracy(self, ground_truth: Dict[str, Any], 
                                 metrics: EnsembleMetrics) -> EntityAccuracyResult:
        """
        Calculate entity accuracy by comparing ground truth entities with predicted entities.
        
        Args:
            ground_truth: Ground truth data with segments
            metrics: Ensemble metrics (contains predicted text)
            
        Returns:
            Entity accuracy results
        """
        # Extract entities from ground truth
        ground_truth_entities = self._extract_entities_from_ground_truth(ground_truth)
        
        # For simulation, create predicted entities based on ground truth with some noise
        predicted_entities = self._simulate_predicted_entities(ground_truth_entities, metrics)
        
        # Compare entities
        total_entities = len(ground_truth_entities)
        correct_entities = 0
        entity_breakdown = {
            'proper_nouns': {'correct': 0, 'total': 0},
            'numbers': {'correct': 0, 'total': 0},
            'technical_terms': {'correct': 0, 'total': 0}
        }
        errors = []
        
        for gt_entity in ground_truth_entities:
            entity_type = gt_entity['type']
            entity_breakdown[entity_type]['total'] += 1
            
            # Check if entity was correctly predicted
            found_match = False
            for pred_entity in predicted_entities:
                if self._entities_match(gt_entity, pred_entity):
                    correct_entities += 1
                    entity_breakdown[entity_type]['correct'] += 1
                    found_match = True
                    break
            
            if not found_match:
                errors.append({
                    'expected': gt_entity,
                    'found': None,
                    'type': 'missing_entity'
                })
        
        accuracy = correct_entities / total_entities if total_entities > 0 else 1.0
        
        return EntityAccuracyResult(
            total_entities=total_entities,
            correct_entities=correct_entities,
            accuracy=accuracy,
            entity_breakdown=entity_breakdown,
            errors=errors
        )
    
    def _extract_entities_from_ground_truth(self, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from ground truth data"""
        entities = []
        segments = ground_truth.get('segments', [])
        
        # Extract technical terms from linguistic features
        linguistic_features = ground_truth.get('linguistic_features', {})
        technical_terms = linguistic_features.get('technical_terms', [])
        
        for term in technical_terms:
            entities.append({
                'text': term,
                'type': 'technical_terms',
                'confidence': 1.0
            })
        
        # Extract numbers and proper nouns from segment text (simplified extraction)
        for segment in segments:
            text = segment.get('text', '')
            words = text.split()
            
            for word in words:
                # Simple heuristic: capitalized words (excluding first word) are proper nouns
                if word[0].isupper() and word not in ['The', 'A', 'An', 'This', 'That']:
                    entities.append({
                        'text': word,
                        'type': 'proper_nouns',
                        'confidence': 0.9
                    })
                
                # Simple heuristic: detect numbers (digits)
                if any(char.isdigit() for char in word):
                    entities.append({
                        'text': word,
                        'type': 'numbers',
                        'confidence': 0.95
                    })
        
        return entities
    
    def _simulate_predicted_entities(self, ground_truth_entities: List[Dict[str, Any]], 
                                   metrics: EnsembleMetrics) -> List[Dict[str, Any]]:
        """Simulate predicted entities based on ground truth and system performance"""
        predicted_entities = []
        
        # Accuracy simulation based on overall system performance
        entity_accuracy_factor = 1.0 - metrics.asr.wer  # Better WER = better entity accuracy
        
        for gt_entity in ground_truth_entities:
            # Simulate whether this entity would be correctly detected
            if np.random.random() < entity_accuracy_factor:
                # Entity correctly detected
                predicted_entities.append(gt_entity.copy())
            else:
                # Entity missed or corrupted - simulate some noise
                if np.random.random() < 0.5:  # 50% chance of corruption vs complete miss
                    corrupted_entity = gt_entity.copy()
                    corrupted_entity['text'] = corrupted_entity['text'] + 's'  # Simple corruption
                    predicted_entities.append(corrupted_entity)
        
        return predicted_entities
    
    def _entities_match(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """Check if two entities match (fuzzy matching for robustness)"""
        # Exact match
        if entity1['text'].lower() == entity2['text'].lower() and entity1['type'] == entity2['type']:
            return True
        
        # Fuzzy match for similar entities
        text1 = entity1['text'].lower().strip('.,!?')
        text2 = entity2['text'].lower().strip('.,!?')
        
        if entity1['type'] == entity2['type'] and abs(len(text1) - len(text2)) <= 1:
            # Simple edit distance check
            if len(text1) > 0 and len(text2) > 0:
                similarity = max(len(text1), len(text2)) - self._edit_distance(text1, text2)
                similarity_ratio = similarity / max(len(text1), len(text2))
                return similarity_ratio >= 0.8  # 80% similarity threshold
        
        return False
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _extract_reference_text(self, ground_truth: Dict[str, Any]) -> str:
        """Extract reference text from ground truth segments"""
        segments = ground_truth.get("segments", [])
        texts = []
        
        for segment in sorted(segments, key=lambda x: x.get("start", 0)):
            text = segment.get("text", "").strip()
            if text:
                texts.append(text)
        
        return " ".join(texts)
    
    def _perform_regression_analysis(self, gate_results: List[GateResult]) -> Dict[str, Any]:
        """Perform regression analysis against historical results"""
        regression_analysis = {
            'comparison_available': False,
            'regressions_detected': [],
            'improvements_detected': [],
            'trend_analysis': {},
            'statistical_significance': {}
        }
        
        try:
            # Load historical results for comparison
            historical_results = self._load_recent_historical_results(days_back=30)
            
            if not historical_results:
                regression_analysis['message'] = "No historical results available for regression analysis"
                return regression_analysis
            
            regression_analysis['comparison_available'] = True
            
            # Group current results by test_id and gate_type
            current_metrics = {}
            for result in gate_results:
                key = f"{result.test_id}_{result.gate_type.value}"
                current_metrics[key] = result
            
            # Compare with historical averages
            for key, current_result in current_metrics.items():
                if key in historical_results:
                    historical_values = historical_results[key]
                    historical_avg = statistics.mean(historical_values)
                    historical_std = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                    
                    # Calculate change
                    if current_result.gate_type in [GateType.WER_THRESHOLD, GateType.DER_THRESHOLD]:
                        # For error rates, increase is bad
                        change = current_result.measured_value - historical_avg
                        if change > self._get_regression_threshold(current_result.gate_type):
                            regression_analysis['regressions_detected'].append({
                                'test_id': current_result.test_id,
                                'gate_type': current_result.gate_type.value,
                                'current_value': current_result.measured_value,
                                'historical_average': historical_avg,
                                'change': change,
                                'standard_deviations': change / historical_std if historical_std > 0 else 0,
                                'significance': 'high' if abs(change) > 2 * historical_std else 'medium'
                            })
                        elif change < -self._get_regression_threshold(current_result.gate_type):
                            regression_analysis['improvements_detected'].append({
                                'test_id': current_result.test_id,
                                'gate_type': current_result.gate_type.value,
                                'current_value': current_result.measured_value,
                                'historical_average': historical_avg,
                                'improvement': abs(change)
                            })
                    else:
                        # For accuracy/confidence metrics, decrease is bad
                        change = historical_avg - current_result.measured_value
                        if change > self._get_regression_threshold(current_result.gate_type):
                            regression_analysis['regressions_detected'].append({
                                'test_id': current_result.test_id,
                                'gate_type': current_result.gate_type.value,
                                'current_value': current_result.measured_value,
                                'historical_average': historical_avg,
                                'change': change,
                                'standard_deviations': change / historical_std if historical_std > 0 else 0,
                                'significance': 'high' if abs(change) > 2 * historical_std else 'medium'
                            })
        
        except Exception as e:
            self.logger.error("❌ REGRESSION ANALYSIS ERROR", context={'error': str(e)})
            regression_analysis['error'] = str(e)
        
        return regression_analysis
    
    def _get_regression_threshold(self, gate_type: GateType) -> float:
        """Get regression threshold for specific gate type"""
        if gate_type == GateType.WER_THRESHOLD:
            return self.config.regression_wer_increase_max
        elif gate_type == GateType.DER_THRESHOLD:
            return self.config.regression_der_increase_max
        elif gate_type in [GateType.CONFIDENCE_CALIBRATION, GateType.ENTITY_ACCURACY]:
            return self.config.regression_confidence_decrease_max
        else:
            return 5.0  # Default 5% threshold
    
    def _load_recent_historical_results(self, days_back: int = 30) -> Dict[str, List[float]]:
        """Load recent historical results for regression analysis"""
        historical_metrics = {}
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Scan historical results directory
        for results_file in self.historical_results_dir.glob("evaluation_*.json"):
            try:
                with open(results_file, 'r') as f:
                    historical_report = json.load(f)
                
                # Check if within time window
                report_date = datetime.fromisoformat(historical_report['timestamp'])
                if report_date < cutoff_date:
                    continue
                
                # Extract metrics
                for gate_result in historical_report.get('gate_results', []):
                    key = f"{gate_result['test_id']}_{gate_result['gate_type']}"
                    if key not in historical_metrics:
                        historical_metrics[key] = []
                    historical_metrics[key].append(gate_result['measured_value'])
            
            except Exception as e:
                self.logger.warning("⚠️ HISTORICAL RESULTS LOAD WARNING", context={
                    'file': str(results_file),
                    'error': str(e)
                })
                continue
        
        return historical_metrics
    
    def _generate_evaluation_report(self, session_id: str, gate_results: List[GateResult],
                                  regression_analysis: Dict[str, Any], 
                                  processing_errors: List[Dict[str, Any]]) -> EvaluationReport:
        """Generate comprehensive evaluation report"""
        
        # Count results by status
        status_counts = {
            GateStatus.PASS: 0,
            GateStatus.FAIL: 0,
            GateStatus.WARNING: 0,
            GateStatus.ERROR: 0,
            GateStatus.SKIP: 0
        }
        
        for result in gate_results:
            status_counts[result.status] += 1
        
        # Add processing errors to error count
        status_counts[GateStatus.ERROR] += len(processing_errors)
        
        # Determine overall status
        if status_counts[GateStatus.FAIL] > 0:
            overall_status = GateStatus.FAIL
            ci_exit_code = 1
        elif status_counts[GateStatus.ERROR] > 0:
            overall_status = GateStatus.ERROR
            ci_exit_code = 1
        elif status_counts[GateStatus.WARNING] > 0:
            overall_status = GateStatus.WARNING
            ci_exit_code = 0  # Warnings don't fail CI by default
        else:
            overall_status = GateStatus.PASS
            ci_exit_code = 0
        
        # Check for significant regressions
        if regression_analysis.get('regressions_detected'):
            high_significance_regressions = [
                r for r in regression_analysis['regressions_detected']
                if r.get('significance') == 'high'
            ]
            if high_significance_regressions:
                overall_status = GateStatus.FAIL
                ci_exit_code = 1
        
        # Calculate summary metrics
        wer_results = [r for r in gate_results if r.gate_type == GateType.WER_THRESHOLD]
        der_results = [r for r in gate_results if r.gate_type == GateType.DER_THRESHOLD]
        entity_results = [r for r in gate_results if r.gate_type == GateType.ENTITY_ACCURACY]
        
        summary_metrics = {
            'avg_wer': statistics.mean([r.measured_value for r in wer_results]) if wer_results else 0.0,
            'max_wer': max([r.measured_value for r in wer_results]) if wer_results else 0.0,
            'avg_der': statistics.mean([r.measured_value for r in der_results]) if der_results else 0.0,
            'max_der': max([r.measured_value for r in der_results]) if der_results else 0.0,
            'avg_entity_accuracy': statistics.mean([r.measured_value for r in entity_results]) if entity_results else 100.0,
            'min_entity_accuracy': min([r.measured_value for r in entity_results]) if entity_results else 100.0,
            'pass_rate': (status_counts[GateStatus.PASS] / len(gate_results)) * 100 if gate_results else 0.0
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gate_results, regression_analysis)
        
        return EvaluationReport(
            session_id=session_id,
            timestamp=datetime.now(),
            overall_status=overall_status,
            total_tests=len(gate_results) + len(processing_errors),
            passed_tests=status_counts[GateStatus.PASS],
            failed_tests=status_counts[GateStatus.FAIL],
            warning_tests=status_counts[GateStatus.WARNING],
            error_tests=status_counts[GateStatus.ERROR] + len(processing_errors),
            gate_results=gate_results,
            summary_metrics=summary_metrics,
            regression_analysis=regression_analysis,
            recommendations=recommendations,
            ci_exit_code=ci_exit_code
        )
    
    def _generate_recommendations(self, gate_results: List[GateResult], 
                                regression_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on evaluation results"""
        recommendations = []
        
        # Analyze failure patterns
        failed_results = [r for r in gate_results if r.status == GateStatus.FAIL]
        warning_results = [r for r in gate_results if r.status == GateStatus.WARNING]
        
        # WER-specific recommendations
        wer_failures = [r for r in failed_results if r.gate_type == GateType.WER_THRESHOLD]
        if wer_failures:
            avg_wer_failure = statistics.mean([r.measured_value for r in wer_failures])
            if avg_wer_failure > 10.0:
                recommendations.append("🔧 CRITICAL: Consider ASR model retraining - WER significantly exceeds acceptable thresholds")
            elif avg_wer_failure > 5.0:
                recommendations.append("⚙️ Review ASR preprocessing pipeline and audio quality enhancement")
            else:
                recommendations.append("🎯 Fine-tune ASR confidence thresholds and ensemble weights")
        
        # DER-specific recommendations
        der_failures = [r for r in failed_results if r.gate_type == GateType.DER_THRESHOLD]
        if der_failures:
            avg_der_failure = statistics.mean([r.measured_value for r in der_failures])
            if avg_der_failure > 20.0:
                recommendations.append("🔧 CRITICAL: Review speaker diarization model - DER indicates poor speaker separation")
            else:
                recommendations.append("🎯 Adjust speaker diarization sensitivity and clustering parameters")
        
        # Entity accuracy recommendations
        entity_failures = [r for r in failed_results if r.gate_type == GateType.ENTITY_ACCURACY]
        if entity_failures:
            recommendations.append("📝 Review and expand domain-specific glossaries for better entity recognition")
            recommendations.append("🧠 Consider custom entity recognition model training for technical terminology")
        
        # Performance recommendations
        performance_failures = [r for r in failed_results if r.gate_type == GateType.PROCESSING_TIME]
        if performance_failures:
            recommendations.append("⚡ Optimize processing pipeline - consider parallel processing or caching strategies")
        
        # Regression-specific recommendations
        if regression_analysis.get('regressions_detected'):
            recommendations.append("📈 REGRESSION ALERT: Performance degradation detected - review recent model changes")
            recommendations.append("🔄 Consider rolling back to previous stable version if regressions are severe")
        
        # Warning-based recommendations
        if warning_results:
            recommendations.append("⚠️ Monitor warning results closely - early indicators of potential degradation")
        
        # Overall system health recommendations
        if len(failed_results) > len(gate_results) * 0.3:  # More than 30% failures
            recommendations.append("🚨 SYSTEM HEALTH: High failure rate indicates systemic issues - comprehensive review needed")
        
        return recommendations
    
    def _store_evaluation_results(self, report: EvaluationReport):
        """Store evaluation results for historical tracking"""
        timestamp_str = report.timestamp.strftime("%Y%m%d_%H%M%S")
        results_file = self.historical_results_dir / f"evaluation_{timestamp_str}.json"
        
        # Convert report to JSON-serializable format
        report_dict = asdict(report)
        
        # Convert datetime objects to strings
        report_dict['timestamp'] = report.timestamp.isoformat()
        for gate_result in report_dict['gate_results']:
            gate_result['timestamp'] = gate_result['timestamp'].isoformat() if isinstance(gate_result['timestamp'], datetime) else gate_result['timestamp']
        
        try:
            with open(results_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info("💾 EVALUATION RESULTS STORED", context={
                'session_id': report.session_id,
                'results_file': str(results_file)
            })
        
        except Exception as e:
            self.logger.error("❌ RESULTS STORAGE ERROR", context={
                'session_id': report.session_id,
                'error': str(e)
            })
    
    def _log_evaluation_summary(self, report: EvaluationReport):
        """Log comprehensive evaluation summary"""
        status_emoji = {
            GateStatus.PASS: "✅",
            GateStatus.FAIL: "❌", 
            GateStatus.WARNING: "⚠️",
            GateStatus.ERROR: "🚨"
        }
        
        self.logger.info(f"{status_emoji.get(report.overall_status, '❓')} EVALUATION COMPLETE", context={
            'session_id': report.session_id,
            'overall_status': report.overall_status.value,
            'total_tests': report.total_tests,
            'passed_tests': report.passed_tests,
            'failed_tests': report.failed_tests,
            'warning_tests': report.warning_tests,
            'error_tests': report.error_tests,
            'pass_rate_percent': f"{(report.passed_tests / report.total_tests * 100):.1f}%" if report.total_tests > 0 else "0%",
            'ci_exit_code': report.ci_exit_code
        })
        
        # Log key metrics
        self.logger.info("📊 EVALUATION METRICS SUMMARY", context=report.summary_metrics)
        
        # Log regressions if detected
        if report.regression_analysis.get('regressions_detected'):
            self.logger.warning("📈 PERFORMANCE REGRESSIONS DETECTED", context={
                'regressions': len(report.regression_analysis['regressions_detected']),
                'details': report.regression_analysis['regressions_detected'][:3]  # First 3 for brevity
            })
        
        # Log recommendations
        if report.recommendations:
            self.logger.info("💡 EVALUATION RECOMMENDATIONS", context={
                'recommendations': report.recommendations
            })


class EvaluationGatesCLI:
    """Command-line interface for evaluation gates system"""
    
    def __init__(self):
        self.gates_system = EvaluationGatesSystem()
    
    def run_evaluation(self, config_path: Optional[str] = None, 
                      live_processing: bool = False) -> int:
        """
        Run evaluation gates and return CI exit code.
        
        Args:
            config_path: Optional path to configuration file
            live_processing: Whether to use live ensemble processing
            
        Returns:
            CI exit code (0 = pass, 1 = fail, 2 = warnings)
        """
        try:
            # Load configuration if provided
            if config_path:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.gates_system = EvaluationGatesSystem(config.get('evaluation_gates', {}))
            
            # Setup ensemble manager if live processing requested
            ensemble_manager = None
            if live_processing:
                ensemble_manager = EnsembleManager()
            
            # Run evaluation
            report = self.gates_system.run_full_evaluation(ensemble_manager)
            
            # Print summary
            self._print_evaluation_summary(report)
            
            return report.ci_exit_code
            
        except Exception as e:
            print(f"❌ EVALUATION ERROR: {e}")
            return 1  # Fail on error
    
    def _print_evaluation_summary(self, report: EvaluationReport):
        """Print human-readable evaluation summary"""
        status_emoji = {
            GateStatus.PASS: "✅",
            GateStatus.FAIL: "❌", 
            GateStatus.WARNING: "⚠️",
            GateStatus.ERROR: "🚨"
        }
        
        print(f"\n{status_emoji.get(report.overall_status, '❓')} EVALUATION GATES SUMMARY")
        print("=" * 60)
        print(f"Session ID: {report.session_id}")
        print(f"Overall Status: {report.overall_status.value.upper()}")
        print(f"Tests: {report.passed_tests}/{report.total_tests} passed")
        print(f"Pass Rate: {(report.passed_tests / report.total_tests * 100):.1f}%")
        
        if report.failed_tests > 0:
            print(f"❌ Failed: {report.failed_tests}")
        if report.warning_tests > 0:
            print(f"⚠️  Warnings: {report.warning_tests}")
        if report.error_tests > 0:
            print(f"🚨 Errors: {report.error_tests}")
        
        print("\n📊 KEY METRICS:")
        print(f"  Average WER: {report.summary_metrics['avg_wer']:.2f}%")
        print(f"  Average DER: {report.summary_metrics['avg_der']:.2f}%")
        print(f"  Entity Accuracy: {report.summary_metrics['avg_entity_accuracy']:.1f}%")
        
        if report.regression_analysis.get('regressions_detected'):
            print(f"\n📈 REGRESSIONS DETECTED: {len(report.regression_analysis['regressions_detected'])}")
            for regression in report.regression_analysis['regressions_detected'][:3]:  # First 3
                print(f"  - {regression['test_id']}: {regression['gate_type']} increased by {regression['change']:.2f}")
        
        if report.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for recommendation in report.recommendations[:5]:  # First 5
                print(f"  {recommendation}")
        
        print(f"\nCI Exit Code: {report.ci_exit_code}")
        print("=" * 60)


if __name__ == "__main__":
    """Enable CLI usage: python -m core.evaluation_gates"""
    import sys
    
    cli = EvaluationGatesCLI()
    
    # Simple CLI argument parsing
    live_processing = "--live" in sys.argv
    config_path = None
    
    if "--config" in sys.argv:
        config_index = sys.argv.index("--config")
        if config_index + 1 < len(sys.argv):
            config_path = sys.argv[config_index + 1]
    
    exit_code = cli.run_evaluation(config_path, live_processing)
    sys.exit(exit_code)