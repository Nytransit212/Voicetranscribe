"""
Test suite for Automated Evaluation Gates System.

This module provides comprehensive tests for the evaluation gates system,
ensuring that WER, DER, and entity accuracy thresholds work correctly,
CI integration functions properly, and regression detection is accurate.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from dataclasses import asdict
from datetime import datetime, timedelta

# Internal imports
from core.evaluation_gates import (
    EvaluationGatesSystem, 
    ThresholdConfig, 
    GateResult, 
    GateStatus, 
    GateType, 
    EvaluationReport,
    EntityAccuracyResult,
    EvaluationGatesCLI
)
from core.ensemble_manager import EnsembleManager
from utils.metric_calculator import EnsembleMetrics, ASRMetrics, DiarizationMetrics


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create test structure
    gold_test_dir = temp_path / "tests" / "gold_test_set"
    gold_test_dir.mkdir(parents=True)
    ground_truth_dir = gold_test_dir / "ground_truth"
    ground_truth_dir.mkdir()
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_test_suite():
    """Sample test suite configuration"""
    return {
        "test_suite_version": "1.0.0",
        "created": "2025-09-16",
        "description": "Test suite for evaluation gates testing",
        "test_cases": [
            {
                "test_id": "test_clean_speech",
                "name": "Clean Speech Test",
                "audio_file": "clean_test.wav",
                "duration": 30.0,
                "scenario_type": "baseline",
                "description": "Clean audio with minimal noise",
                "expected_metrics": {
                    "der_max": 0.08,
                    "wer_max": 0.05,
                    "confidence_calibration_min": 0.8,
                    "processing_time_max": 60.0
                },
                "metadata": {
                    "speaker_count": 2,
                    "noise_level": "low",
                    "audio_quality": "high"
                }
            },
            {
                "test_id": "test_challenging_overlap", 
                "name": "Challenging Overlap Test",
                "audio_file": "overlap_test.wav",
                "duration": 45.0,
                "scenario_type": "challenging_diarization",
                "description": "Heavy speaker overlap with crosstalk",
                "expected_metrics": {
                    "der_max": 0.25,
                    "wer_max": 0.20,
                    "confidence_calibration_min": 0.6,
                    "processing_time_max": 90.0
                },
                "metadata": {
                    "speaker_count": 3,
                    "noise_level": "medium", 
                    "audio_quality": "medium",
                    "overlap_percentage": 25.0
                }
            }
        ],
        "quality_thresholds": {
            "regression_tolerance": {
                "der_increase_max": 0.05,
                "wer_increase_max": 0.05,
                "confidence_decrease_max": 0.1
            }
        }
    }


@pytest.fixture  
def sample_ground_truth():
    """Sample ground truth data"""
    return {
        "test_id": "test_clean_speech",
        "audio_duration": 30.0,
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "Speaker_A",
                "text": "Welcome to today's meeting about Project Alpha.",
                "confidence": 1.0
            },
            {
                "start": 5.5,
                "end": 12.0,
                "speaker": "Speaker_B", 
                "text": "Thank you. I have updates on the technical implementation using API version 2.1.",
                "confidence": 0.98
            },
            {
                "start": 12.5,
                "end": 18.0,
                "speaker": "Speaker_A",
                "text": "Excellent. The budget for Q3 is $50,000 for infrastructure improvements.",
                "confidence": 1.0
            }
        ],
        "linguistic_features": {
            "technical_terms": ["Project Alpha", "API", "infrastructure"],
            "content_domain": "business_meeting"
        }
    }


@pytest.fixture
def evaluation_gates_system(temp_test_dir, sample_test_suite):
    """Create evaluation gates system with test configuration"""
    # Setup test files
    gold_test_dir = temp_test_dir / "tests" / "gold_test_set"
    test_definitions_file = gold_test_dir / "test_definitions.json"
    
    with open(test_definitions_file, 'w') as f:
        json.dump(sample_test_suite, f, indent=2)
    
    # Create gates system
    config = {
        'wer_threshold_clean': 1.0,
        'wer_threshold_challenging': 5.0, 
        'der_threshold': 5.0,
        'entity_accuracy_threshold': 95.0,
        'confidence_calibration_min': 0.7
    }
    
    with patch.object(Path, 'cwd', return_value=temp_test_dir):
        gates_system = EvaluationGatesSystem(config)
        gates_system.gold_test_path = gold_test_dir
        gates_system.test_definitions_file = test_definitions_file
        gates_system.ground_truth_dir = gold_test_dir / "ground_truth"
        gates_system.results_dir = temp_test_dir / "evaluation_results"  
        gates_system.results_dir.mkdir(exist_ok=True)
        gates_system.historical_results_dir = gates_system.results_dir / "historical"
        gates_system.historical_results_dir.mkdir(exist_ok=True)
        gates_system._load_test_suite()
        
    return gates_system


class TestThresholdConfiguration:
    """Test threshold configuration and validation"""
    
    def test_threshold_config_defaults(self):
        """Test that threshold configuration has reasonable defaults"""
        config = ThresholdConfig()
        
        # WER thresholds should be increasing with difficulty
        assert config.wer_threshold_clean < config.wer_threshold_overlap
        assert config.wer_threshold_overlap < config.wer_threshold_challenging
        assert config.wer_threshold_challenging < config.wer_threshold_edge_case
        
        # Thresholds should be in reasonable ranges
        assert 0.5 <= config.wer_threshold_clean <= 2.0
        assert 10.0 <= config.wer_threshold_edge_case <= 20.0
        assert 2.0 <= config.der_threshold <= 10.0
        assert 80.0 <= config.entity_accuracy_threshold <= 100.0
    
    def test_threshold_config_custom(self):
        """Test custom threshold configuration"""
        custom_config = {
            'wer_threshold_clean': 0.8,
            'der_threshold': 3.0,
            'entity_accuracy_threshold': 97.0
        }
        
        config = ThresholdConfig(**custom_config)
        
        assert config.wer_threshold_clean == 0.8
        assert config.der_threshold == 3.0  
        assert config.entity_accuracy_threshold == 97.0


class TestEvaluationGatesSystem:
    """Test core evaluation gates system functionality"""
    
    def test_system_initialization(self, evaluation_gates_system):
        """Test that evaluation gates system initializes correctly"""
        gates_system = evaluation_gates_system
        
        assert gates_system is not None
        assert len(gates_system.test_suite.get('test_cases', [])) > 0
        assert gates_system.metric_calculator is not None
        assert gates_system.logger is not None
    
    def test_load_test_suite(self, evaluation_gates_system):
        """Test loading test suite definitions"""
        gates_system = evaluation_gates_system
        
        test_cases = gates_system.test_suite.get('test_cases', [])
        assert len(test_cases) == 2
        
        # Verify test case structure
        test_case = test_cases[0]
        assert 'test_id' in test_case
        assert 'scenario_type' in test_case
        assert 'expected_metrics' in test_case
        assert 'metadata' in test_case
    
    def test_load_ground_truth(self, evaluation_gates_system, sample_ground_truth, temp_test_dir):
        """Test loading ground truth data"""
        gates_system = evaluation_gates_system
        
        # Create ground truth file
        ground_truth_file = gates_system.ground_truth_dir / "test_clean_speech.json"
        with open(ground_truth_file, 'w') as f:
            json.dump(sample_ground_truth, f, indent=2)
        
        # Load ground truth
        loaded_gt = gates_system._load_ground_truth("test_clean_speech")
        
        assert loaded_gt is not None
        assert loaded_gt['test_id'] == 'test_clean_speech'
        assert len(loaded_gt['segments']) == 3
        assert 'linguistic_features' in loaded_gt
    
    def test_simulate_evaluation_results(self, evaluation_gates_system, sample_ground_truth):
        """Test simulation of evaluation results"""
        gates_system = evaluation_gates_system
        test_case = gates_system.test_suite['test_cases'][0]
        
        # Test baseline scenario simulation
        metrics = gates_system._simulate_evaluation_results(test_case, sample_ground_truth)
        
        assert isinstance(metrics, EnsembleMetrics)
        assert 0.0 <= metrics.asr.wer <= 0.15  # Should be low for baseline
        assert 0.0 <= metrics.diarization.der <= 0.20
        assert 0.5 <= metrics.confidence_calibration <= 1.0
    
    def test_wer_gates_pass(self, evaluation_gates_system, sample_ground_truth):
        """Test WER gates pass for good performance"""
        gates_system = evaluation_gates_system
        test_case = gates_system.test_suite['test_cases'][0]  # baseline scenario
        
        # Create metrics with low WER (should pass)
        asr_metrics = ASRMetrics(wer=0.008, cer=0.006, mer=0.005, wil=0.004, wip=0.996, insertions=1, deletions=0, substitutions=1, word_count=25)
        diarization_metrics = DiarizationMetrics(der=0.05, missed_speaker_time=0.02, false_alarm_time=0.01, speaker_confusion_time=0.02, speaker_count_accuracy=0.95, overlap_coverage=0.90)
        metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.85, ensemble_agreement=0.8, processing_time=45.0)
        
        gate_results = gates_system._apply_wer_gates(test_case, metrics)
        
        assert len(gate_results) == 1
        assert gate_results[0].status == GateStatus.PASS
        assert gate_results[0].gate_type == GateType.WER_THRESHOLD
        assert gate_results[0].measured_value < gate_results[0].threshold_value
    
    def test_wer_gates_fail(self, evaluation_gates_system, sample_ground_truth):
        """Test WER gates fail for poor performance"""
        gates_system = evaluation_gates_system 
        test_case = gates_system.test_suite['test_cases'][0]  # baseline scenario
        
        # Create metrics with high WER (should fail)
        asr_metrics = ASRMetrics(wer=0.025, cer=0.02, mer=0.015, wil=0.012, wip=0.988, insertions=3, deletions=2, substitutions=1, word_count=25)
        diarization_metrics = DiarizationMetrics(der=0.05, missed_speaker_time=0.02, false_alarm_time=0.01, speaker_confusion_time=0.02, speaker_count_accuracy=0.95, overlap_coverage=0.90)
        metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.85, ensemble_agreement=0.8, processing_time=45.0)
        
        gate_results = gates_system._apply_wer_gates(test_case, metrics)
        
        assert len(gate_results) == 1
        assert gate_results[0].status == GateStatus.FAIL
        assert gate_results[0].gate_type == GateType.WER_THRESHOLD
        assert gate_results[0].measured_value > gate_results[0].threshold_value
    
    def test_der_gates_pass(self, evaluation_gates_system):
        """Test DER gates pass for good diarization"""
        gates_system = evaluation_gates_system
        test_case = gates_system.test_suite['test_cases'][0]  # baseline scenario
        
        # Create metrics with low DER (should pass)
        asr_metrics = ASRMetrics(wer=0.01, cer=0.008, mer=0.005, wil=0.004, wip=0.996, insertions=1, deletions=0, substitutions=1, word_count=25)
        diarization_metrics = DiarizationMetrics(der=0.03, missed_speaker_time=0.015, false_alarm_time=0.005, speaker_confusion_time=0.01, speaker_count_accuracy=0.95, overlap_coverage=0.95)
        metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.85, ensemble_agreement=0.8, processing_time=45.0)
        
        gate_results = gates_system._apply_der_gates(test_case, metrics)
        
        assert len(gate_results) == 1
        assert gate_results[0].status == GateStatus.PASS
        assert gate_results[0].gate_type == GateType.DER_THRESHOLD
        assert gate_results[0].measured_value < gate_results[0].threshold_value
    
    def test_der_gates_fail(self, evaluation_gates_system):
        """Test DER gates fail for poor diarization"""
        gates_system = evaluation_gates_system
        test_case = gates_system.test_suite['test_cases'][0]  # baseline scenario
        
        # Create metrics with high DER (should fail)
        asr_metrics = ASRMetrics(wer=0.01, cer=0.008, mer=0.005, wil=0.004, wip=0.996, insertions=1, deletions=0, substitutions=1, word_count=25)
        diarization_metrics = DiarizationMetrics(der=0.08, missed_speaker_time=0.04, false_alarm_time=0.02, speaker_confusion_time=0.02, speaker_count_accuracy=0.8, overlap_coverage=0.85)
        metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.85, ensemble_agreement=0.8, processing_time=45.0)
        
        gate_results = gates_system._apply_der_gates(test_case, metrics)
        
        assert len(gate_results) == 1
        assert gate_results[0].status == GateStatus.FAIL
        assert gate_results[0].gate_type == GateType.DER_THRESHOLD
        assert gate_results[0].measured_value > gate_results[0].threshold_value
    
    def test_entity_accuracy_gates(self, evaluation_gates_system, sample_ground_truth):
        """Test entity accuracy gate evaluation"""
        gates_system = evaluation_gates_system
        test_case = gates_system.test_suite['test_cases'][0]
        
        # Mock high-performing metrics
        asr_metrics = ASRMetrics(wer=0.01, cer=0.008, bleu_score=0.95, semantic_similarity=0.9, word_count=25)
        diarization_metrics = DiarizationMetrics(der=0.03, speaker_count_accuracy=0.95, segment_boundary_accuracy=0.92, speaker_purity=0.97, speaker_coverage=0.97)
        metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.85, ensemble_agreement=0.8, processing_time=45.0, total_candidates=15, winner_confidence=0.85)
        
        gate_results = gates_system._apply_entity_accuracy_gates(test_case, metrics, sample_ground_truth)
        
        assert len(gate_results) == 1
        assert gate_results[0].gate_type == GateType.ENTITY_ACCURACY
        # Should pass with high-performing metrics and reasonable ground truth
        assert gate_results[0].status in [GateStatus.PASS, GateStatus.WARNING]
    
    def test_confidence_gates(self, evaluation_gates_system):
        """Test confidence calibration gates"""
        gates_system = evaluation_gates_system
        test_case = gates_system.test_suite['test_cases'][0]
        
        # High confidence metrics (should pass)
        asr_metrics = ASRMetrics(wer=0.01, cer=0.008, bleu_score=0.95, semantic_similarity=0.9, word_count=25)
        diarization_metrics = DiarizationMetrics(der=0.03, speaker_count_accuracy=0.95, segment_boundary_accuracy=0.92, speaker_purity=0.97, speaker_coverage=0.97)
        high_conf_metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.85, ensemble_agreement=0.8, processing_time=45.0, total_candidates=15, winner_confidence=0.85)
        
        gate_results = gates_system._apply_confidence_gates(test_case, high_conf_metrics)
        assert gate_results[0].status == GateStatus.PASS
        
        # Low confidence metrics (should fail)
        low_conf_metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.4, ensemble_agreement=0.3, processing_time=45.0, total_candidates=15, winner_confidence=0.4)
        
        gate_results = gates_system._apply_confidence_gates(test_case, low_conf_metrics)
        assert gate_results[0].status == GateStatus.FAIL
    
    def test_performance_gates(self, evaluation_gates_system):
        """Test processing time performance gates"""
        gates_system = evaluation_gates_system
        test_case = gates_system.test_suite['test_cases'][0]
        test_case['duration'] = 30.0  # 30 second audio
        
        asr_metrics = ASRMetrics(wer=0.01, cer=0.008, bleu_score=0.95, semantic_similarity=0.9, word_count=25)
        diarization_metrics = DiarizationMetrics(der=0.03, speaker_count_accuracy=0.95, segment_boundary_accuracy=0.92, speaker_purity=0.97, speaker_coverage=0.97)
        
        # Fast processing (should pass) - 45 seconds for 30 second audio = 1.5x ratio
        fast_metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.85, ensemble_agreement=0.8, processing_time=45.0, total_candidates=15, winner_confidence=0.85)
        
        gate_results = gates_system._apply_performance_gates(test_case, fast_metrics)
        assert gate_results[0].status == GateStatus.PASS
        
        # Slow processing (should fail) - 150 seconds for 30 second audio = 5x ratio
        slow_metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.85, ensemble_agreement=0.8, processing_time=150.0, total_candidates=15, winner_confidence=0.85)
        
        gate_results = gates_system._apply_performance_gates(test_case, slow_metrics) 
        assert gate_results[0].status == GateStatus.FAIL


class TestEntityAccuracy:
    """Test entity accuracy calculation and matching"""
    
    def test_extract_entities_from_ground_truth(self, evaluation_gates_system, sample_ground_truth):
        """Test entity extraction from ground truth"""
        gates_system = evaluation_gates_system
        
        entities = gates_system._extract_entities_from_ground_truth(sample_ground_truth)
        
        # Should extract technical terms, proper nouns, and numbers
        assert len(entities) > 0
        
        # Check for expected entities
        entity_texts = [e['text'] for e in entities]
        assert 'Project Alpha' in entity_texts or 'API' in entity_texts
        assert '$50,000' in entity_texts or '2.1' in entity_texts
        
        # Check entity types
        entity_types = set(e['type'] for e in entities)
        expected_types = {'technical_terms', 'proper_nouns', 'numbers'}
        assert entity_types.intersection(expected_types), f"Expected some of {expected_types}, got {entity_types}"
    
    def test_entity_matching_exact(self, evaluation_gates_system):
        """Test exact entity matching"""
        gates_system = evaluation_gates_system
        
        entity1 = {'text': 'Project Alpha', 'type': 'technical_terms', 'confidence': 1.0}
        entity2 = {'text': 'Project Alpha', 'type': 'technical_terms', 'confidence': 0.9}
        
        assert gates_system._entities_match(entity1, entity2)
    
    def test_entity_matching_fuzzy(self, evaluation_gates_system):
        """Test fuzzy entity matching"""
        gates_system = evaluation_gates_system
        
        entity1 = {'text': 'Project Alpha', 'type': 'technical_terms', 'confidence': 1.0}
        entity2 = {'text': 'project alpha', 'type': 'technical_terms', 'confidence': 0.9}  # Case difference
        
        assert gates_system._entities_match(entity1, entity2)
    
    def test_entity_matching_no_match(self, evaluation_gates_system):
        """Test entity matching failure"""
        gates_system = evaluation_gates_system
        
        entity1 = {'text': 'Project Alpha', 'type': 'technical_terms', 'confidence': 1.0}
        entity2 = {'text': 'Project Beta', 'type': 'technical_terms', 'confidence': 0.9}
        
        assert not gates_system._entities_match(entity1, entity2)
    
    def test_edit_distance_calculation(self, evaluation_gates_system):
        """Test edit distance calculation for fuzzy matching"""
        gates_system = evaluation_gates_system
        
        # Identical strings
        assert gates_system._edit_distance("hello", "hello") == 0
        
        # Single substitution
        assert gates_system._edit_distance("hello", "hallo") == 1
        
        # Single insertion
        assert gates_system._edit_distance("hello", "helloo") == 1
        
        # Single deletion
        assert gates_system._edit_distance("hello", "helo") == 1


class TestRegressionAnalysis:
    """Test regression detection and trend analysis"""
    
    def test_regression_detection_no_history(self, evaluation_gates_system):
        """Test regression detection when no historical data exists"""
        gates_system = evaluation_gates_system
        
        gate_results = []  # Empty results
        regression_analysis = gates_system._perform_regression_analysis(gate_results)
        
        assert not regression_analysis['comparison_available']
        assert len(regression_analysis['regressions_detected']) == 0
    
    def test_load_recent_historical_results(self, evaluation_gates_system):
        """Test loading recent historical results"""
        gates_system = evaluation_gates_system
        
        # Create sample historical results
        historical_report = {
            "session_id": "test_session",
            "timestamp": datetime.now().isoformat(),
            "gate_results": [
                {
                    "test_id": "test_clean_speech",
                    "gate_type": "wer_threshold", 
                    "measured_value": 0.8,  # 0.8% WER
                    "status": "pass"
                }
            ]
        }
        
        # Store historical result
        historical_file = gates_system.historical_results_dir / "evaluation_20250915_120000.json"
        with open(historical_file, 'w') as f:
            json.dump(historical_report, f)
        
        # Load historical results
        historical_metrics = gates_system._load_recent_historical_results(days_back=30)
        
        assert len(historical_metrics) > 0
        assert "test_clean_speech_wer_threshold" in historical_metrics
        assert 0.8 in historical_metrics["test_clean_speech_wer_threshold"]


class TestEvaluationReporting:
    """Test evaluation report generation"""
    
    def test_generate_evaluation_report(self, evaluation_gates_system):
        """Test evaluation report generation"""
        gates_system = evaluation_gates_system
        
        # Create sample gate results
        gate_results = [
            GateResult(
                gate_type=GateType.WER_THRESHOLD,
                status=GateStatus.PASS,
                test_id="test1",
                scenario_type="baseline",
                measured_value=0.8,
                threshold_value=1.0,
                margin=-0.2,
                message="WER passes threshold",
                details={},
                timestamp=datetime.now()
            ),
            GateResult(
                gate_type=GateType.DER_THRESHOLD,
                status=GateStatus.FAIL,
                test_id="test2", 
                scenario_type="challenging",
                measured_value=8.0,
                threshold_value=5.0,
                margin=3.0,
                message="DER exceeds threshold",
                details={},
                timestamp=datetime.now()
            )
        ]
        
        regression_analysis = {'regressions_detected': [], 'comparison_available': False}
        processing_errors = []
        
        report = gates_system._generate_evaluation_report(
            "test_session", gate_results, regression_analysis, processing_errors
        )
        
        assert isinstance(report, EvaluationReport)
        assert report.session_id == "test_session"
        assert report.total_tests == 2
        assert report.passed_tests == 1
        assert report.failed_tests == 1
        assert report.overall_status == GateStatus.FAIL  # Should fail due to DER failure
        assert report.ci_exit_code == 1  # Should fail CI
    
    def test_generate_recommendations(self, evaluation_gates_system):
        """Test recommendation generation based on failures"""
        gates_system = evaluation_gates_system
        
        # Create gate results with WER failure
        gate_results = [
            GateResult(
                gate_type=GateType.WER_THRESHOLD,
                status=GateStatus.FAIL,
                test_id="test1",
                scenario_type="baseline", 
                measured_value=12.0,  # High WER
                threshold_value=1.0,
                margin=11.0,
                message="High WER failure",
                details={},
                timestamp=datetime.now()
            )
        ]
        
        regression_analysis = {'regressions_detected': []}
        
        recommendations = gates_system._generate_recommendations(gate_results, regression_analysis)
        
        assert len(recommendations) > 0
        # Should recommend ASR model retraining for high WER
        wer_recommendations = [r for r in recommendations if 'ASR' in r and 'retraining' in r]
        assert len(wer_recommendations) > 0


class TestCLIIntegration:
    """Test command-line interface for CI integration"""
    
    def test_cli_initialization(self):
        """Test CLI initialization"""
        cli = EvaluationGatesCLI()
        assert cli.gates_system is not None
    
    @patch('core.evaluation_gates.EnsembleManager')
    @patch.object(EvaluationGatesSystem, 'run_full_evaluation')
    def test_cli_run_evaluation_pass(self, mock_run_eval, mock_ensemble):
        """Test CLI evaluation run that passes"""
        cli = EvaluationGatesCLI()
        
        # Mock successful evaluation
        mock_report = Mock()
        mock_report.ci_exit_code = 0
        mock_report.overall_status = GateStatus.PASS
        mock_report.session_id = "test_session"
        mock_report.total_tests = 5
        mock_report.passed_tests = 5
        mock_report.failed_tests = 0
        mock_report.warning_tests = 0
        mock_report.error_tests = 0
        mock_report.summary_metrics = {
            'avg_wer': 0.8,
            'avg_der': 3.2,
            'avg_entity_accuracy': 96.5
        }
        mock_report.regression_analysis = {'regressions_detected': []}
        mock_report.recommendations = []
        
        mock_run_eval.return_value = mock_report
        
        # Run CLI evaluation
        exit_code = cli.run_evaluation(live_processing=False)
        
        assert exit_code == 0
        mock_run_eval.assert_called_once()
    
    @patch.object(EvaluationGatesSystem, 'run_full_evaluation')
    def test_cli_run_evaluation_fail(self, mock_run_eval):
        """Test CLI evaluation run that fails"""
        cli = EvaluationGatesCLI()
        
        # Mock failed evaluation
        mock_report = Mock()
        mock_report.ci_exit_code = 1
        mock_report.overall_status = GateStatus.FAIL
        mock_report.session_id = "test_session"
        mock_report.total_tests = 5
        mock_report.passed_tests = 3
        mock_report.failed_tests = 2
        mock_report.warning_tests = 0
        mock_report.error_tests = 0
        mock_report.summary_metrics = {
            'avg_wer': 2.5,  # High WER
            'avg_der': 8.2,  # High DER
            'avg_entity_accuracy': 88.5  # Low entity accuracy
        }
        mock_report.regression_analysis = {'regressions_detected': []}
        mock_report.recommendations = ["Review ASR model performance"]
        
        mock_run_eval.return_value = mock_report
        
        # Run CLI evaluation
        exit_code = cli.run_evaluation(live_processing=False)
        
        assert exit_code == 1  # Should fail
        mock_run_eval.assert_called_once()


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases"""
    
    def test_mixed_gate_results(self, evaluation_gates_system, sample_ground_truth, temp_test_dir):
        """Test evaluation with mixed pass/fail results"""
        gates_system = evaluation_gates_system
        
        # Create ground truth file
        ground_truth_file = gates_system.ground_truth_dir / "test_clean_speech.json"
        with open(ground_truth_file, 'w') as f:
            json.dump(sample_ground_truth, f, indent=2)
        
        # Run single test case evaluation
        test_case = gates_system.test_suite['test_cases'][0]
        
        # Mock ensemble manager to return None (use simulation)
        gate_results = gates_system._evaluate_test_case(test_case, ensemble_manager=None)
        
        assert len(gate_results) > 0
        # Should have multiple gate types tested
        gate_types = set(r.gate_type for r in gate_results)
        expected_types = {GateType.WER_THRESHOLD, GateType.DER_THRESHOLD, GateType.ENTITY_ACCURACY}
        assert len(gate_types.intersection(expected_types)) > 0
    
    def test_error_handling_missing_ground_truth(self, evaluation_gates_system):
        """Test error handling when ground truth is missing"""
        gates_system = evaluation_gates_system
        
        # Test case with missing ground truth
        test_case = {
            'test_id': 'missing_test',
            'scenario_type': 'baseline',
            'expected_metrics': {}
        }
        
        gate_results = gates_system._evaluate_test_case(test_case, ensemble_manager=None)
        
        assert len(gate_results) == 1
        assert gate_results[0].status == GateStatus.ERROR
        assert "Ground truth not found" in gate_results[0].message
    
    @patch.object(EvaluationGatesSystem, '_simulate_evaluation_results')
    def test_error_handling_evaluation_exception(self, mock_simulate, evaluation_gates_system):
        """Test error handling when evaluation throws exception"""
        gates_system = evaluation_gates_system
        
        # Mock simulation to raise exception
        mock_simulate.side_effect = Exception("Simulation error")
        
        test_case = gates_system.test_suite['test_cases'][0]
        gate_results = gates_system._evaluate_test_case(test_case, ensemble_manager=None)
        
        assert len(gate_results) == 1
        assert gate_results[0].status == GateStatus.ERROR
        assert "Evaluation error" in gate_results[0].message
    
    def test_historical_results_storage(self, evaluation_gates_system):
        """Test storing and loading historical results"""
        gates_system = evaluation_gates_system
        
        # Create sample report
        report = EvaluationReport(
            session_id="test_storage",
            timestamp=datetime.now(),
            overall_status=GateStatus.PASS,
            total_tests=2,
            passed_tests=2,
            failed_tests=0,
            warning_tests=0,
            error_tests=0,
            gate_results=[],
            summary_metrics={},
            regression_analysis={},
            recommendations=[],
            ci_exit_code=0
        )
        
        # Store results
        gates_system._store_evaluation_results(report)
        
        # Verify file was created
        results_files = list(gates_system.historical_results_dir.glob("evaluation_*.json"))
        assert len(results_files) > 0
        
        # Verify content
        with open(results_files[0], 'r') as f:
            stored_data = json.load(f)
        
        assert stored_data['session_id'] == "test_storage"
        assert stored_data['overall_status'] == "pass"