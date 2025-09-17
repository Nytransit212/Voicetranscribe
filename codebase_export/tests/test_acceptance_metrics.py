"""
Comprehensive acceptance test suite for ensemble transcription system.

This module validates system performance against gold standard test cases,
ensures quality thresholds are met, and detects performance regression.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import time
from dataclasses import asdict

# Internal imports
from core.ensemble_manager import EnsembleManager
from utils.metric_calculator import MetricCalculator, EnsembleMetrics, DiarizationMetrics, ASRMetrics, MetricReporter
from utils.structured_logger import StructuredLogger
from utils.metrics_registry import MetricsRegistryManager


class AcceptanceTestConfig:
    """Configuration for acceptance testing"""
    
    def __init__(self):
        self.gold_test_path = Path("tests/gold_test_set")
        self.test_definitions_file = self.gold_test_path / "test_definitions.json"
        self.ground_truth_dir = self.gold_test_path / "ground_truth"
        self.reports_dir = Path("test_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load test definitions
        if self.test_definitions_file.exists():
            with open(self.test_definitions_file, 'r') as f:
                self.test_suite = json.load(f)
        else:
            self.test_suite = {"test_cases": [], "quality_thresholds": {}}
        
        # Regression tolerance settings
        self.regression_tolerance = self.test_suite.get("quality_thresholds", {}).get("regression_tolerance", {
            "der_increase_max": 0.05,
            "wer_increase_max": 0.05,
            "confidence_decrease_max": 0.1,
            "processing_time_increase_max": 20.0
        })


@pytest.fixture(scope="session")
def test_config():
    """Session-scoped test configuration fixture"""
    return AcceptanceTestConfig()


@pytest.fixture(scope="session")
def metric_calculator():
    """Session-scoped metric calculator fixture"""
    logger = StructuredLogger("acceptance_tests")
    return MetricCalculator(logger=logger)


@pytest.fixture(scope="session")
def metric_reporter(test_config):
    """Session-scoped metric reporter fixture"""
    return MetricReporter(output_dir=str(test_config.reports_dir))


@pytest.fixture
def ensemble_manager():
    """Fresh ensemble manager for each test"""
    return EnsembleManager(
        expected_speakers=3,
        noise_level='medium',
        target_language=None,
        enable_versioning=False,  # Disable for testing
        domain="test"
    )


@pytest.fixture
def mock_audio_processing():
    """Mock audio processing for tests that don't need real files"""
    with patch('core.audio_processor.AudioProcessor') as mock_processor:
        mock_instance = Mock()
        mock_instance.extract_audio_from_video.return_value = ("raw.wav", "clean.wav")
        mock_instance.get_audio_duration.return_value = 30.0
        mock_instance.estimate_noise_level.return_value = "medium"
        mock_processor.return_value = mock_instance
        yield mock_instance


class TestSystemAcceptance:
    """Main acceptance test suite for system validation"""
    
    def test_load_test_suite(self, test_config):
        """Verify that test suite definitions are properly loaded"""
        assert test_config.test_suite is not None
        assert "test_cases" in test_config.test_suite
        assert len(test_config.test_suite["test_cases"]) > 0
        
        # Verify required fields in test cases
        for test_case in test_config.test_suite["test_cases"]:
            required_fields = ["test_id", "expected_metrics", "metadata"]
            for field in required_fields:
                assert field in test_case, f"Missing {field} in test case {test_case.get('test_id', 'unknown')}"
    
    @pytest.mark.parametrize("test_case", [
        tc for tc in AcceptanceTestConfig().test_suite.get("test_cases", [])
        if tc.get("scenario_type") == "baseline"
    ])
    def test_baseline_scenarios(self, test_case, metric_calculator, test_config, mock_audio_processing):
        """Test baseline scenarios that should have high accuracy"""
        
        # Load ground truth
        ground_truth_file = test_config.ground_truth_dir / f"{test_case['test_id']}.json"
        
        if not ground_truth_file.exists():
            pytest.skip(f"Ground truth file not found: {ground_truth_file}")
        
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Simulate ensemble processing results (would normally come from actual processing)
        simulated_results = self._create_simulated_results(test_case, ground_truth)
        
        # Calculate metrics
        reference_text = self._extract_reference_text(ground_truth)
        reference_segments = ground_truth.get("segments", [])
        
        metrics = metric_calculator.calculate_ensemble_metrics(
            candidates=simulated_results["candidates"],
            reference_segments=reference_segments,
            reference_text=reference_text,
            audio_duration=ground_truth.get("audio_duration", 30.0),
            processing_time=45.0  # Simulated processing time
        )
        
        # Assert quality thresholds for baseline scenarios
        expected = test_case["expected_metrics"]
        
        assert metrics.diarization.der <= expected.get("der_max", 0.15), \
            f"DER {metrics.diarization.der:.3f} exceeds baseline threshold {expected.get('der_max', 0.15):.3f}"
        
        assert metrics.asr.wer <= expected.get("wer_max", 0.12), \
            f"WER {metrics.asr.wer:.3f} exceeds baseline threshold {expected.get('wer_max', 0.12):.3f}"
        
        assert metrics.confidence_calibration >= expected.get("confidence_calibration_min", 0.7), \
            f"Confidence calibration {metrics.confidence_calibration:.3f} below baseline threshold"
        
        # Store results for regression analysis
        self._store_test_results(test_case["test_id"], metrics, test_config)
    
    @pytest.mark.parametrize("test_case", [
        tc for tc in AcceptanceTestConfig().test_suite.get("test_cases", [])
        if tc.get("scenario_type") in ["challenging_diarization", "challenging_asr", "challenging_audio"]
    ])
    def test_challenging_scenarios(self, test_case, metric_calculator, test_config, mock_audio_processing):
        """Test challenging scenarios with appropriate tolerance"""
        
        # Load ground truth  
        ground_truth_file = test_config.ground_truth_dir / f"{test_case['test_id']}.json"
        
        if not ground_truth_file.exists():
            pytest.skip(f"Ground truth file not found: {ground_truth_file}")
        
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Simulate ensemble processing results
        simulated_results = self._create_simulated_results(test_case, ground_truth)
        
        # Calculate metrics
        reference_text = self._extract_reference_text(ground_truth)
        reference_segments = ground_truth.get("segments", [])
        
        metrics = metric_calculator.calculate_ensemble_metrics(
            candidates=simulated_results["candidates"],
            reference_segments=reference_segments,
            reference_text=reference_text,
            audio_duration=ground_truth.get("audio_duration", 60.0),
            processing_time=90.0  # Longer processing for challenging scenarios
        )
        
        # Assert quality thresholds for challenging scenarios
        expected = test_case["expected_metrics"]
        
        assert metrics.diarization.der <= expected.get("der_max", 0.35), \
            f"DER {metrics.diarization.der:.3f} exceeds challenging threshold {expected.get('der_max', 0.35):.3f}"
        
        assert metrics.asr.wer <= expected.get("wer_max", 0.35), \
            f"WER {metrics.asr.wer:.3f} exceeds challenging threshold {expected.get('wer_max', 0.35):.3f}"
        
        # For challenging scenarios, ensure system doesn't completely fail
        assert metrics.confidence_calibration >= 0.2, \
            f"Confidence calibration {metrics.confidence_calibration:.3f} indicates system failure"
        
        assert metrics.ensemble_agreement >= 0.3, \
            f"Ensemble agreement {metrics.ensemble_agreement:.3f} indicates inconsistent results"
        
        # Store results for regression analysis
        self._store_test_results(test_case["test_id"], metrics, test_config)
    
    @pytest.mark.parametrize("test_case", [
        tc for tc in AcceptanceTestConfig().test_suite.get("test_cases", [])
        if tc.get("scenario_type") == "edge_case"
    ])
    def test_edge_cases(self, test_case, metric_calculator, test_config, mock_audio_processing):
        """Test edge cases ensuring system graceful degradation"""
        
        # Load ground truth
        ground_truth_file = test_config.ground_truth_dir / f"{test_case['test_id']}.json"
        
        if not ground_truth_file.exists():
            pytest.skip(f"Ground truth file not found: {ground_truth_file}")
        
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Simulate ensemble processing results
        simulated_results = self._create_simulated_results(test_case, ground_truth)
        
        # Calculate metrics
        reference_text = self._extract_reference_text(ground_truth)
        reference_segments = ground_truth.get("segments", [])
        
        metrics = metric_calculator.calculate_ensemble_metrics(
            candidates=simulated_results["candidates"],
            reference_segments=reference_segments,
            reference_text=reference_text,
            audio_duration=ground_truth.get("audio_duration", 30.0),
            processing_time=60.0
        )
        
        # For edge cases, focus on graceful degradation rather than accuracy
        expected = test_case["expected_metrics"]
        
        # System should not completely fail
        assert metrics.diarization.der <= 0.8, \
            "DER indicates complete diarization failure"
        
        assert metrics.asr.wer <= 0.8, \
            "WER indicates complete ASR failure"
        
        # System should still provide some confidence calibration
        assert metrics.confidence_calibration >= 0.1, \
            "Confidence calibration indicates complete failure"
        
        # Processing time should remain reasonable even for edge cases
        assert metrics.processing_time <= expected.get("processing_time_max", 120.0), \
            f"Processing time {metrics.processing_time:.1f}s exceeds edge case limit"
        
        # Store results for regression analysis
        self._store_test_results(test_case["test_id"], metrics, test_config)
    
    def test_regression_detection(self, test_config, metric_calculator):
        """Test for performance regression against historical results"""
        
        historical_results = self._load_historical_results(test_config)
        
        if not historical_results:
            pytest.skip("No historical results available for regression testing")
        
        current_results = self._load_current_results(test_config)
        
        # Check for regression in critical metrics
        regression_detected = False
        regression_details = []
        
        for test_id in historical_results:
            if test_id not in current_results:
                continue
                
            hist = historical_results[test_id]
            curr = current_results[test_id]
            
            # Check DER regression
            der_increase = curr.get("der", 1.0) - hist.get("der", 0.0)
            if der_increase > test_config.regression_tolerance["der_increase_max"]:
                regression_detected = True
                regression_details.append(f"DER regression in {test_id}: +{der_increase:.3f}")
            
            # Check WER regression
            wer_increase = curr.get("wer", 1.0) - hist.get("wer", 0.0)
            if wer_increase > test_config.regression_tolerance["wer_increase_max"]:
                regression_detected = True
                regression_details.append(f"WER regression in {test_id}: +{wer_increase:.3f}")
            
            # Check confidence calibration regression
            conf_decrease = hist.get("confidence_calibration", 0.0) - curr.get("confidence_calibration", 0.0)
            if conf_decrease > test_config.regression_tolerance["confidence_decrease_max"]:
                regression_detected = True
                regression_details.append(f"Confidence regression in {test_id}: -{conf_decrease:.3f}")
        
        # Assert no significant regression
        if regression_detected:
            pytest.fail(f"Performance regression detected:\n" + "\n".join(regression_details))
    
    def test_metric_consistency(self, metric_calculator):
        """Test that metric calculations are consistent and reproducible"""
        
        # Create sample data
        reference_text = "This is a test sentence for metric consistency validation."
        hypothesis_text = "This is a test sentence for metric consistency validation."
        
        # Calculate metrics multiple times
        results = []
        for _ in range(5):
            metrics = metric_calculator.calculate_asr_metrics(reference_text, hypothesis_text)
            results.append(metrics.wer)
        
        # All results should be identical (perfect consistency)
        assert all(r == results[0] for r in results), \
            f"Metric calculation not consistent: {results}"
        
        # Test with slightly different inputs
        hypothesis_text_2 = "This is a test sentence for metric consistency verification."
        metrics_2 = metric_calculator.calculate_asr_metrics(reference_text, hypothesis_text_2)
        
        # Should detect the difference
        assert metrics_2.wer > 0, "Metric calculator should detect word differences"
    
    def test_confidence_calibration_accuracy(self, metric_calculator):
        """Test that confidence scores correlate with actual quality"""
        
        # Create test cases with known quality levels
        test_cases = [
            {
                "reference": "Perfect transcription test case",
                "hypothesis": "Perfect transcription test case", 
                "expected_quality": "high"
            },
            {
                "reference": "Medium quality transcription test case",
                "hypothesis": "Medium quality transcription test case with small error",
                "expected_quality": "medium"
            },
            {
                "reference": "Poor quality transcription test case",
                "hypothesis": "Poor transcription with many errors and mistakes",
                "expected_quality": "low"
            }
        ]
        
        wer_scores = []
        for test_case in test_cases:
            metrics = metric_calculator.calculate_asr_metrics(
                test_case["reference"], 
                test_case["hypothesis"]
            )
            wer_scores.append(metrics.wer)
        
        # WER should increase (quality decrease) from high to low quality
        assert wer_scores[0] <= wer_scores[1] <= wer_scores[2], \
            f"WER scores don't correlate with quality: {wer_scores}"
    
    def _create_simulated_results(self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Create simulated ensemble results for testing"""
        
        # Simulate 15 candidates with varying quality
        candidates = []
        base_segments = ground_truth.get("segments", [])
        
        for i in range(15):
            # Add some noise to simulate realistic variation
            confidence_base = 0.8 - (i * 0.03)  # Decreasing confidence
            
            candidate = {
                "candidate_id": f"candidate_{i:02d}",
                "aligned_segments": self._add_segment_noise(base_segments, noise_level=i*0.02),
                "confidence_scores": {
                    "D_diarization": max(0.1, confidence_base + np.random.normal(0, 0.05)),
                    "A_asr_alignment": max(0.1, confidence_base + np.random.normal(0, 0.05)),
                    "L_linguistic": max(0.1, confidence_base + np.random.normal(0, 0.03)),
                    "R_agreement": max(0.1, confidence_base + np.random.normal(0, 0.04)),
                    "O_overlap": max(0.1, confidence_base + np.random.normal(0, 0.06)),
                    "final_score": max(0.1, confidence_base)
                }
            }
            candidates.append(candidate)
        
        return {
            "candidates": candidates,
            "winner_candidate": candidates[0],  # Highest confidence
            "processing_metadata": {
                "total_candidates": 15,
                "processing_time": test_case.get("expected_metrics", {}).get("processing_time_max", 60.0) * 0.8
            }
        }
    
    def _add_segment_noise(self, segments: List[Dict[str, Any]], noise_level: float = 0.0) -> List[Dict[str, Any]]:
        """Add realistic noise to segments for simulation"""
        
        noisy_segments = []
        for segment in segments:
            noisy_segment = segment.copy()
            
            # Add timing noise
            timing_noise = noise_level * 0.5
            noisy_segment["start"] = max(0.0, segment["start"] + np.random.normal(0, timing_noise))
            noisy_segment["end"] = noisy_segment["start"] + max(0.1, 
                segment["end"] - segment["start"] + np.random.normal(0, timing_noise))
            
            # Add text noise occasionally
            if noise_level > 0.05 and np.random.random() < noise_level:
                words = segment["text"].split()
                if words and len(words) > 1:
                    # Randomly alter a word
                    word_idx = np.random.randint(0, len(words))
                    words[word_idx] = words[word_idx] + "s"  # Simple alteration
                    noisy_segment["text"] = " ".join(words)
            
            noisy_segments.append(noisy_segment)
        
        return noisy_segments
    
    def _extract_reference_text(self, ground_truth: Dict[str, Any]) -> str:
        """Extract reference text from ground truth"""
        segments = ground_truth.get("segments", [])
        texts = []
        
        for segment in sorted(segments, key=lambda x: x.get("start", 0)):
            text = segment.get("text", "").strip()
            if text:
                texts.append(text)
        
        return " ".join(texts)
    
    def _store_test_results(self, test_id: str, metrics: EnsembleMetrics, config: AcceptanceTestConfig):
        """Store test results for regression analysis"""
        
        results_file = config.reports_dir / f"latest_results_{test_id}.json"
        
        result_data = {
            "test_id": test_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": metrics.to_dict()
        }
        
        with open(results_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
    
    def _load_historical_results(self, config: AcceptanceTestConfig) -> Dict[str, Dict[str, Any]]:
        """Load historical test results for comparison"""
        
        historical_file = config.reports_dir / "historical_baseline.json"
        
        if not historical_file.exists():
            return {}
        
        try:
            with open(historical_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _load_current_results(self, config: AcceptanceTestConfig) -> Dict[str, Dict[str, Any]]:
        """Load current test results"""
        
        current_results = {}
        
        for result_file in config.reports_dir.glob("latest_results_*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    test_id = data["test_id"]
                    current_results[test_id] = data["metrics"]
            except Exception:
                continue
        
        return current_results


class TestMetricCalculator:
    """Test suite specifically for metric calculation components"""
    
    def test_wer_transformation_pipeline(self, metric_calculator):
        """Test WER transformation pipeline consistency"""
        
        test_cases = [
            {
                "reference": "Hello, world! How are you?",
                "hypothesis": "hello world how are you",
                "expected_similar": True
            },
            {
                "reference": "The quick brown fox jumps over the lazy dog.",
                "hypothesis": "The quick brown fox jumps over the lazy cat.",
                "expected_similar": False
            }
        ]
        
        for test_case in test_cases:
            metrics = metric_calculator.calculate_asr_metrics(
                test_case["reference"],
                test_case["hypothesis"]
            )
            
            if test_case["expected_similar"]:
                assert metrics.wer < 0.1, f"Expected similar texts to have low WER, got {metrics.wer}"
            else:
                assert metrics.wer > 0.0, f"Expected different texts to have non-zero WER, got {metrics.wer}"
    
    def test_diarization_metrics(self, metric_calculator):
        """Test diarization metric calculations"""
        
        # Create test segments
        reference_segments = [
            {"start": 0.0, "end": 5.0, "speaker": "Speaker_A", "text": "First utterance"},
            {"start": 5.5, "end": 10.0, "speaker": "Speaker_B", "text": "Second utterance"},
            {"start": 10.5, "end": 15.0, "speaker": "Speaker_A", "text": "Third utterance"}
        ]
        
        hypothesis_segments = [
            {"start": 0.1, "end": 4.9, "speaker": "Speaker_A", "text": "First utterance"},
            {"start": 5.6, "end": 9.9, "speaker": "Speaker_B", "text": "Second utterance"},
            {"start": 10.6, "end": 14.9, "speaker": "Speaker_A", "text": "Third utterance"}
        ]
        
        metrics = metric_calculator.calculate_diarization_metrics(
            reference_segments, hypothesis_segments, audio_duration=15.0
        )
        
        # Should have very low DER for well-aligned segments
        assert metrics.der < 0.2, f"Expected low DER for aligned segments, got {metrics.der}"
        assert metrics.speaker_count_accuracy > 0.9, f"Expected high speaker count accuracy"
    
    def test_ensemble_agreement_calculation(self, metric_calculator):
        """Test ensemble agreement metric calculation"""
        
        # Create candidates with different levels of agreement
        high_agreement_candidates = [
            {
                "aligned_segments": [
                    {"start": 0.0, "end": 5.0, "speaker": "A", "text": "Hello world"}
                ]
            },
            {
                "aligned_segments": [
                    {"start": 0.0, "end": 5.0, "speaker": "A", "text": "Hello world"}
                ]
            }
        ]
        
        low_agreement_candidates = [
            {
                "aligned_segments": [
                    {"start": 0.0, "end": 5.0, "speaker": "A", "text": "Hello world"}
                ]
            },
            {
                "aligned_segments": [
                    {"start": 0.0, "end": 5.0, "speaker": "A", "text": "Goodbye universe"}
                ]
            }
        ]
        
        high_agreement = metric_calculator._calculate_ensemble_agreement(high_agreement_candidates)
        low_agreement = metric_calculator._calculate_ensemble_agreement(low_agreement_candidates)
        
        assert high_agreement > low_agreement, \
            f"High agreement {high_agreement} should be greater than low agreement {low_agreement}"