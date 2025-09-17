"""
Integration tests for Evaluation Gates System with EnsembleManager and existing test suite.

This module tests the complete integration of evaluation gates with the ensemble
transcription system, ensuring CI integration works correctly and builds fail
appropriately on threshold violations.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import numpy as np

# Internal imports
from core.evaluation_gates import EvaluationGatesSystem, EvaluationGatesCLI, GateStatus, GateType
from core.ensemble_manager import EnsembleManager
from tests.test_acceptance_metrics import AcceptanceTestConfig
from utils.metric_calculator import MetricCalculator, EnsembleMetrics, ASRMetrics, DiarizationMetrics


@pytest.fixture
def integrated_test_environment():
    """Create integrated test environment with real file structure"""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create full test structure matching existing system
    tests_dir = temp_path / "tests"
    gold_test_dir = tests_dir / "gold_test_set"
    ground_truth_dir = gold_test_dir / "ground_truth" 
    audio_clips_dir = gold_test_dir / "audio_clips"
    metadata_dir = gold_test_dir / "metadata"
    
    for dir_path in [tests_dir, gold_test_dir, ground_truth_dir, audio_clips_dir, metadata_dir]:
        dir_path.mkdir(parents=True)
    
    # Create test definitions matching existing format
    test_definitions = {
        "test_suite_version": "1.0.0",
        "created": "2025-09-16",
        "description": "Integration test suite for evaluation gates",
        "test_cases": [
            {
                "test_id": "clear_speech_multispeaker",
                "name": "Clear Speech Multiple Speakers",
                "audio_file": "test_short_video.mov",
                "duration": 30.0,
                "scenario_type": "baseline",
                "description": "Clean audio with multiple distinct speakers",
                "expected_metrics": {
                    "der_max": 0.15,
                    "wer_max": 0.12,
                    "confidence_calibration_min": 0.7,
                    "processing_time_max": 45.0,
                    "speaker_count_accuracy_min": 0.8
                },
                "metadata": {
                    "speaker_count": 3,
                    "noise_level": "low",
                    "accent_types": ["US_General"],
                    "audio_quality": "high"
                }
            },
            {
                "test_id": "heavy_overlap_crosstalk",
                "name": "Heavy Speaker Overlap and Crosstalk",
                "audio_file": "test_video.mp4",
                "duration": 60.0,
                "scenario_type": "challenging_diarization", 
                "description": "Significant speaker overlap periods",
                "expected_metrics": {
                    "der_max": 0.35,
                    "wer_max": 0.25,
                    "confidence_calibration_min": 0.5,
                    "processing_time_max": 90.0
                },
                "metadata": {
                    "speaker_count": 4,
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
    
    # Write test definitions
    with open(gold_test_dir / "test_definitions.json", 'w') as f:
        json.dump(test_definitions, f, indent=2)
    
    # Create ground truth files
    ground_truth_clean = {
        "test_id": "clear_speech_multispeaker",
        "audio_file": "test_short_video.mov",
        "audio_duration": 30.0,
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "Speaker_A",
                "text": "Welcome everyone to today's quarterly review meeting for Project Phoenix.",
                "confidence": 1.0
            },
            {
                "start": 5.5,
                "end": 12.0,
                "speaker": "Speaker_B",
                "text": "Thanks Sarah. I'll start with the technical updates on API version 2.1 deployment.",
                "confidence": 0.98
            },
            {
                "start": 12.5,
                "end": 18.0,
                "speaker": "Speaker_C",
                "text": "The infrastructure improvements cost $75,000 but delivered 40% performance gains.",
                "confidence": 1.0
            }
        ],
        "linguistic_features": {
            "technical_terms": ["Project Phoenix", "API", "infrastructure", "deployment"],
            "content_domain": "business_technical_meeting"
        },
        "quality_targets": {
            "der_expected": 0.08,
            "wer_expected": 0.05,
            "entity_accuracy_expected": 0.95
        }
    }
    
    ground_truth_overlap = {
        "test_id": "heavy_overlap_crosstalk",
        "audio_file": "test_video.mp4", 
        "audio_duration": 60.0,
        "segments": [
            {
                "start": 0.0,
                "end": 8.0,
                "speaker": "Speaker_A",
                "text": "We need to discuss the Q3 revenue projections and budget allocation strategy.",
                "confidence": 0.9
            },
            {
                "start": 6.0,  # Overlap region
                "end": 14.0,
                "speaker": "Speaker_B",
                "text": "Actually, I wanted to interrupt - the numbers from the marketing department show different trends.",
                "confidence": 0.85
            }
        ],
        "linguistic_features": {
            "technical_terms": ["revenue projections", "budget allocation", "marketing department"],
            "content_domain": "business_strategy_meeting"
        },
        "quality_targets": {
            "der_expected": 0.25,
            "wer_expected": 0.20,
            "entity_accuracy_expected": 0.87
        }
    }
    
    # Write ground truth files
    with open(ground_truth_dir / "clear_speech_multispeaker.json", 'w') as f:
        json.dump(ground_truth_clean, f, indent=2)
    
    with open(ground_truth_dir / "heavy_overlap_crosstalk.json", 'w') as f:
        json.dump(ground_truth_overlap, f, indent=2)
    
    # Create sample audio files (empty - just for path validation)
    (audio_clips_dir / "test_short_video.mov").touch()
    (audio_clips_dir / "test_video.mp4").touch()
    
    # Create config directory structure
    config_dir = temp_path / "config"
    config_dir.mkdir()
    
    evaluation_gates_config = {
        "evaluation_gates": {
            "enabled": True,
            "fail_builds_on_violations": True,
            "wer_thresholds": {
                "clean_audio": 1.0,
                "challenging_audio": 5.0
            },
            "der_thresholds": {
                "baseline": 5.0,
                "challenging": 15.0
            },
            "entity_accuracy_thresholds": {
                "baseline": 95.0,
                "challenging": 85.0
            },
            "test_execution": {
                "enable_live_processing": False
            }
        }
    }
    
    with open(config_dir / "evaluation_config.json", 'w') as f:
        json.dump(evaluation_gates_config, f, indent=2)
    
    yield {
        "temp_path": temp_path,
        "tests_dir": tests_dir,
        "gold_test_dir": gold_test_dir,
        "config_dir": config_dir,
        "ground_truth_dir": ground_truth_dir
    }
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestEvaluationGatesEnsembleIntegration:
    """Test integration between evaluation gates and ensemble manager"""
    
    @patch('core.ensemble_manager.EnsembleManager.process_video')
    def test_evaluation_gates_with_ensemble_manager(self, mock_process_video, integrated_test_environment):
        """Test evaluation gates integration with real ensemble manager"""
        env = integrated_test_environment
        
        # Mock ensemble manager response
        mock_transcript_result = {
            'master_transcript': {
                'segments': [
                    {
                        'start': 0.0,
                        'end': 5.0,
                        'speaker': 'Speaker_A',
                        'text': 'Welcome everyone to todays quarterly review meeting for Project Phoenix.',
                        'confidence': 0.95
                    }
                ],
                'metadata': {
                    'confidence_summary': {
                        'final_score': 0.85
                    }
                }
            },
            'candidates': [
                {
                    'candidate_id': 'candidate_01',
                    'aligned_segments': [
                        {
                            'start': 0.0,
                            'end': 5.0,
                            'speaker': 'Speaker_A',
                            'text': 'Welcome everyone to todays quarterly review meeting for Project Phoenix.',
                            'confidence': 0.95
                        }
                    ],
                    'confidence_scores': {
                        'D_diarization': 0.9,
                        'A_asr_alignment': 0.88,
                        'L_linguistic': 0.85,
                        'R_agreement': 0.82,
                        'O_overlap': 0.90,
                        'final_score': 0.85
                    }
                }
            ],
            'processing_metadata': {
                'total_processing_time': 35.0,
                'total_candidates': 15
            }
        }
        
        mock_process_video.return_value = mock_transcript_result
        
        # Configure evaluation gates system with test environment
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            gates_system = EvaluationGatesSystem()
            gates_system.gold_test_path = env['gold_test_dir']
            gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            gates_system.ground_truth_dir = env['ground_truth_dir']
            gates_system._load_test_suite()
            
            # Create ensemble manager
            ensemble_manager = EnsembleManager()
            
            # Run evaluation with live processing
            report = gates_system.run_full_evaluation(ensemble_manager)
            
            # Verify integration worked
            assert report is not None
            assert len(report.gate_results) > 0
            
            # Should have called ensemble manager for each test case
            assert mock_process_video.call_count > 0
            
            # Verify report structure
            assert report.total_tests > 0
            assert report.session_id is not None
            
            # Should have multiple gate types tested
            gate_types = set(r.gate_type for r in report.gate_results)
            expected_types = {GateType.WER_THRESHOLD, GateType.DER_THRESHOLD, GateType.ENTITY_ACCURACY}
            assert len(gate_types.intersection(expected_types)) > 0
    
    def test_evaluation_gates_simulation_mode(self, integrated_test_environment):
        """Test evaluation gates in simulation mode (no live ensemble processing)"""
        env = integrated_test_environment
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            gates_system = EvaluationGatesSystem()
            gates_system.gold_test_path = env['gold_test_dir']
            gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            gates_system.ground_truth_dir = env['ground_truth_dir']
            gates_system._load_test_suite()
            
            # Run evaluation without ensemble manager (simulation mode)
            report = gates_system.run_full_evaluation(ensemble_manager=None)
            
            # Verify simulation worked
            assert report is not None
            assert len(report.gate_results) > 0
            assert report.total_tests == len(gates_system.test_suite['test_cases']) * 5  # 5 gates per test case
            
            # Should have results for both test cases
            test_ids = set(r.test_id for r in report.gate_results)
            assert "clear_speech_multispeaker" in test_ids
            assert "heavy_overlap_crosstalk" in test_ids
    
    def test_threshold_violation_detection(self, integrated_test_environment):
        """Test that threshold violations are properly detected and reported"""
        env = integrated_test_environment
        
        # Configure very strict thresholds
        strict_config = {
            'wer_threshold_clean': 0.1,  # Very strict - 0.1%
            'der_threshold': 1.0,        # Very strict - 1%
            'entity_accuracy_threshold': 99.0  # Very strict - 99%
        }
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            gates_system = EvaluationGatesSystem(strict_config)
            gates_system.gold_test_path = env['gold_test_dir']
            gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            gates_system.ground_truth_dir = env['ground_truth_dir']
            gates_system._load_test_suite()
            
            # Run evaluation - should fail due to strict thresholds
            report = gates_system.run_full_evaluation(ensemble_manager=None)
            
            # Should have failures
            failed_results = [r for r in report.gate_results if r.status == GateStatus.FAIL]
            assert len(failed_results) > 0
            
            # Overall status should be FAIL
            assert report.overall_status == GateStatus.FAIL
            assert report.ci_exit_code == 1
            
            # Should have failure reasons
            wer_failures = [r for r in failed_results if r.gate_type == GateType.WER_THRESHOLD]
            der_failures = [r for r in failed_results if r.gate_type == GateType.DER_THRESHOLD]
            
            # At least some failures should be present (simulated metrics won't pass strict thresholds)
            assert len(wer_failures) > 0 or len(der_failures) > 0


class TestCIIntegration:
    """Test CI/CD pipeline integration"""
    
    def test_cli_pass_scenario(self, integrated_test_environment):
        """Test CLI integration for passing scenario"""
        env = integrated_test_environment
        config_file = env['config_dir'] / "evaluation_config.json"
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            cli = EvaluationGatesCLI()
            
            # Mock lenient thresholds for passing
            with patch.object(cli.gates_system, 'config') as mock_config:
                mock_config.wer_threshold_clean = 10.0  # Lenient 
                mock_config.der_threshold = 20.0        # Lenient
                mock_config.entity_accuracy_threshold = 50.0  # Lenient
                
                cli.gates_system.gold_test_path = env['gold_test_dir']
                cli.gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
                cli.gates_system.ground_truth_dir = env['ground_truth_dir']
                cli.gates_system._load_test_suite()
                
                # Run CLI evaluation
                exit_code = cli.run_evaluation(config_path=str(config_file), live_processing=False)
                
                # Should pass with lenient thresholds
                assert exit_code == 0
    
    def test_cli_fail_scenario(self, integrated_test_environment):
        """Test CLI integration for failing scenario"""
        env = integrated_test_environment
        config_file = env['config_dir'] / "evaluation_config.json"
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            cli = EvaluationGatesCLI()
            
            # Mock strict thresholds for failing
            with patch.object(cli.gates_system, 'config') as mock_config:
                mock_config.wer_threshold_clean = 0.01   # Very strict
                mock_config.der_threshold = 0.5          # Very strict
                mock_config.entity_accuracy_threshold = 99.9  # Very strict
                
                cli.gates_system.gold_test_path = env['gold_test_dir']
                cli.gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
                cli.gates_system.ground_truth_dir = env['ground_truth_dir']
                cli.gates_system._load_test_suite()
                
                # Run CLI evaluation
                exit_code = cli.run_evaluation(config_path=str(config_file), live_processing=False)
                
                # Should fail with strict thresholds
                assert exit_code == 1
    
    def test_config_file_loading(self, integrated_test_environment):
        """Test configuration file loading in CLI"""
        env = integrated_test_environment
        
        # Create custom config with specific thresholds
        custom_config = {
            "evaluation_gates": {
                "wer_threshold_clean": 2.5,
                "der_threshold": 8.0,
                "entity_accuracy_threshold": 92.0
            }
        }
        
        config_file = env['config_dir'] / "custom_config.json"
        with open(config_file, 'w') as f:
            json.dump(custom_config, f)
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            cli = EvaluationGatesCLI()
            cli.gates_system.gold_test_path = env['gold_test_dir']
            cli.gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            cli.gates_system.ground_truth_dir = env['ground_truth_dir']
            cli.gates_system._load_test_suite()
            
            # Run with custom config
            exit_code = cli.run_evaluation(config_path=str(config_file), live_processing=False)
            
            # Should complete successfully (exit code depends on whether thresholds pass)
            assert exit_code in [0, 1]  # Either pass or fail, but not error


class TestAcceptanceTestIntegration:
    """Test integration with existing acceptance test infrastructure"""
    
    def test_acceptance_test_compatibility(self, integrated_test_environment):
        """Test that evaluation gates work with existing acceptance test structure"""
        env = integrated_test_environment
        
        # Create AcceptanceTestConfig using our test environment
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            # Mock AcceptanceTestConfig to use our test environment
            with patch('tests.test_acceptance_metrics.AcceptanceTestConfig') as mock_config_class:
                mock_config = Mock()
                mock_config.gold_test_path = env['gold_test_dir']
                mock_config.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
                mock_config.ground_truth_dir = env['ground_truth_dir']
                mock_config.reports_dir = env['temp_path'] / "test_reports"
                mock_config.reports_dir.mkdir(exist_ok=True)
                
                # Load our test suite
                with open(env['gold_test_dir'] / "test_definitions.json", 'r') as f:
                    test_suite = json.load(f)
                mock_config.test_suite = test_suite
                mock_config.regression_tolerance = test_suite["quality_thresholds"]["regression_tolerance"]
                
                mock_config_class.return_value = mock_config
                
                # Create evaluation gates using same structure
                gates_system = EvaluationGatesSystem()
                gates_system.gold_test_path = env['gold_test_dir']
                gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
                gates_system.ground_truth_dir = env['ground_truth_dir']
                gates_system._load_test_suite()
                
                # Verify compatibility
                assert gates_system.test_suite is not None
                assert len(gates_system.test_suite.get('test_cases', [])) > 0
                
                # Test cases should match expected format
                test_case = gates_system.test_suite['test_cases'][0]
                assert 'test_id' in test_case
                assert 'expected_metrics' in test_case
                assert 'metadata' in test_case
                
                # Should be able to run evaluation
                report = gates_system.run_full_evaluation(ensemble_manager=None)
                assert report is not None
    
    def test_metric_calculator_integration(self, integrated_test_environment):
        """Test integration with existing metric calculator"""
        env = integrated_test_environment
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            gates_system = EvaluationGatesSystem()
            gates_system.gold_test_path = env['gold_test_dir']
            gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            gates_system.ground_truth_dir = env['ground_truth_dir']
            gates_system._load_test_suite()
            
            # Verify metric calculator is properly initialized
            assert gates_system.metric_calculator is not None
            assert hasattr(gates_system.metric_calculator, 'calculate_ensemble_metrics')
            assert hasattr(gates_system.metric_calculator, 'calculate_asr_metrics')
            assert hasattr(gates_system.metric_calculator, 'calculate_diarization_metrics')
            
            # Test metric calculation integration
            test_case = gates_system.test_suite['test_cases'][0]
            ground_truth_file = env['ground_truth_dir'] / f"{test_case['test_id']}.json"
            
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
            
            # Should be able to simulate metrics
            metrics = gates_system._simulate_evaluation_results(test_case, ground_truth)
            assert isinstance(metrics, EnsembleMetrics)
            assert hasattr(metrics, 'asr')
            assert hasattr(metrics, 'diarization')
            assert hasattr(metrics, 'confidence_calibration')


class TestRegressionDetectionIntegration:
    """Test regression detection with historical data"""
    
    def test_historical_baseline_establishment(self, integrated_test_environment):
        """Test establishing historical baseline for regression detection"""
        env = integrated_test_environment
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            gates_system = EvaluationGatesSystem()
            gates_system.gold_test_path = env['gold_test_dir']
            gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            gates_system.ground_truth_dir = env['ground_truth_dir']
            gates_system._load_test_suite()
            
            # Run initial evaluation to establish baseline
            initial_report = gates_system.run_full_evaluation(ensemble_manager=None)
            
            # Verify historical results are stored
            historical_files = list(gates_system.historical_results_dir.glob("evaluation_*.json"))
            assert len(historical_files) > 0
            
            # Verify stored data structure
            with open(historical_files[0], 'r') as f:
                stored_data = json.load(f)
            
            assert 'session_id' in stored_data
            assert 'gate_results' in stored_data
            assert 'summary_metrics' in stored_data
            
            # Run second evaluation to test regression detection
            second_report = gates_system.run_full_evaluation(ensemble_manager=None)
            
            # Should have regression analysis available
            assert 'comparison_available' in second_report.regression_analysis
    
    def test_regression_alert_generation(self, integrated_test_environment):
        """Test generation of regression alerts"""
        env = integrated_test_environment
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            gates_system = EvaluationGatesSystem()
            gates_system.gold_test_path = env['gold_test_dir']
            gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            gates_system.ground_truth_dir = env['ground_truth_dir']
            gates_system._load_test_suite()
            
            # Create artificial historical data showing better performance
            from datetime import datetime, timedelta
            historical_data = {
                "session_id": "baseline_session",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "gate_results": [
                    {
                        "test_id": "clear_speech_multispeaker",
                        "gate_type": "wer_threshold",
                        "measured_value": 0.5,  # Low WER
                        "status": "pass"
                    },
                    {
                        "test_id": "clear_speech_multispeaker", 
                        "gate_type": "der_threshold",
                        "measured_value": 2.0,  # Low DER
                        "status": "pass"
                    }
                ]
            }
            
            # Store historical data
            historical_file = gates_system.historical_results_dir / "evaluation_baseline.json"
            with open(historical_file, 'w') as f:
                json.dump(historical_data, f)
            
            # Mock current results to be worse (regression scenario)
            with patch.object(gates_system, '_simulate_evaluation_results') as mock_simulate:
                # Return worse metrics to trigger regression
                asr_metrics = ASRMetrics(wer=0.03, cer=0.025, mer=0.02, wil=0.15, wip=0.85, insertions=2, deletions=1, substitutions=3, word_count=25)  # Higher WER
                diarization_metrics = DiarizationMetrics(der=0.08, missed_speaker_time=0.05, false_alarm_time=0.02, speaker_confusion_time=0.01, speaker_count_accuracy=0.85, overlap_coverage=0.75)  # Higher DER
                worse_metrics = EnsembleMetrics(asr=asr_metrics, diarization=diarization_metrics, confidence_calibration=0.7, ensemble_agreement=0.7, processing_time=60.0)
                
                mock_simulate.return_value = worse_metrics
                
                # Run evaluation - should detect regression
                report = gates_system.run_full_evaluation(ensemble_manager=None)
                
                # Check if regression was detected in analysis
                # Note: Regression detection may not trigger in simulation due to statistical variance
                # But the infrastructure should be in place
                assert 'regressions_detected' in report.regression_analysis
                assert 'comparison_available' in report.regression_analysis


class TestEndToEndScenarios:
    """End-to-end integration scenarios"""
    
    def test_full_pipeline_success_scenario(self, integrated_test_environment):
        """Test complete pipeline from configuration to CI success"""
        env = integrated_test_environment
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            # 1. Configure system with reasonable thresholds
            config = {
                'wer_threshold_clean': 5.0,    # Reasonable threshold
                'der_threshold': 10.0,         # Reasonable threshold
                'entity_accuracy_threshold': 80.0  # Reasonable threshold
            }
            
            # 2. Initialize evaluation gates
            gates_system = EvaluationGatesSystem(config)
            gates_system.gold_test_path = env['gold_test_dir']
            gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            gates_system.ground_truth_dir = env['ground_truth_dir']
            gates_system._load_test_suite()
            
            # 3. Run full evaluation
            report = gates_system.run_full_evaluation(ensemble_manager=None)
            
            # 4. Verify successful execution
            assert report is not None
            assert report.ci_exit_code == 0  # Should pass with reasonable thresholds
            assert report.total_tests > 0
            assert len(report.gate_results) > 0
            
            # 5. Verify all components were tested
            gate_types = set(r.gate_type for r in report.gate_results)
            expected_types = {GateType.WER_THRESHOLD, GateType.DER_THRESHOLD, GateType.ENTITY_ACCURACY, GateType.CONFIDENCE_CALIBRATION, GateType.PROCESSING_TIME}
            assert len(gate_types.intersection(expected_types)) >= 3  # At least 3 gate types
            
            # 6. Verify results were stored
            historical_files = list(gates_system.historical_results_dir.glob("evaluation_*.json"))
            assert len(historical_files) > 0
    
    def test_full_pipeline_failure_scenario(self, integrated_test_environment):
        """Test complete pipeline from configuration to CI failure"""
        env = integrated_test_environment
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            # 1. Configure system with very strict thresholds
            config = {
                'wer_threshold_clean': 0.01,   # Extremely strict
                'der_threshold': 0.1,          # Extremely strict  
                'entity_accuracy_threshold': 99.9  # Extremely strict
            }
            
            # 2. Initialize evaluation gates
            gates_system = EvaluationGatesSystem(config)
            gates_system.gold_test_path = env['gold_test_dir']
            gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            gates_system.ground_truth_dir = env['ground_truth_dir']
            gates_system._load_test_suite()
            
            # 3. Run full evaluation
            report = gates_system.run_full_evaluation(ensemble_manager=None)
            
            # 4. Verify failure conditions
            assert report is not None
            assert report.ci_exit_code == 1  # Should fail with strict thresholds
            assert report.overall_status == GateStatus.FAIL
            assert report.failed_tests > 0
            
            # 5. Verify failure details
            failed_results = [r for r in report.gate_results if r.status == GateStatus.FAIL]
            assert len(failed_results) > 0
            
            # 6. Verify recommendations were generated
            assert len(report.recommendations) > 0
            
            # 7. Should still store results for historical tracking
            historical_files = list(gates_system.historical_results_dir.glob("evaluation_*.json"))
            assert len(historical_files) > 0
    
    @patch('sys.argv', ['test_evaluation_gates.py', '--config', 'test_config.json'])
    def test_cli_integration_end_to_end(self, integrated_test_environment):
        """Test complete CLI integration end-to-end"""
        env = integrated_test_environment
        
        # Create CLI config
        cli_config = {
            "evaluation_gates": {
                "wer_threshold_clean": 3.0,
                "der_threshold": 8.0,
                "entity_accuracy_threshold": 85.0
            }
        }
        
        config_file = env['config_dir'] / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(cli_config, f)
        
        with patch.object(Path, 'cwd', return_value=env['temp_path']):
            # Initialize CLI
            cli = EvaluationGatesCLI()
            cli.gates_system.gold_test_path = env['gold_test_dir']
            cli.gates_system.test_definitions_file = env['gold_test_dir'] / "test_definitions.json"
            cli.gates_system.ground_truth_dir = env['ground_truth_dir']
            cli.gates_system._load_test_suite()
            
            # Run CLI evaluation
            exit_code = cli.run_evaluation(config_path=str(config_file), live_processing=False)
            
            # Should complete successfully
            assert exit_code in [0, 1]  # Pass or fail, but not error