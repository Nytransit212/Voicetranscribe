"""
Pytest configuration and shared fixtures for ensemble transcription tests.

This module provides session and function-scoped fixtures used across 
all test modules, including test data management, mock configurations,
and shared testing utilities.
"""

import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
import logging

# Set up test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_data_dir():
    """Session-scoped fixture providing path to test data directory"""
    return Path("tests/gold_test_set")


@pytest.fixture(scope="session")
def temp_work_dir():
    """Session-scoped temporary working directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="ensemble_test_")
    yield Path(temp_dir)
    # Cleanup after session
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing"""
    mock_env = {
        "OPENAI_API_KEY": "test-openai-key-12345",
        "HUGGINGFACE_TOKEN": "test-hf-token-67890"
    }
    
    with patch.dict(os.environ, mock_env):
        yield mock_env


@pytest.fixture
def sample_audio_metadata():
    """Sample audio metadata for testing"""
    return {
        "duration": 30.0,
        "sample_rate": 44100,
        "channels": 1,
        "format": "PCM_16",
        "estimated_speakers": 3,
        "noise_level": "medium"
    }


@pytest.fixture
def sample_segments():
    """Sample speaker segments for testing"""
    return [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "Speaker_A",
            "text": "Welcome to the meeting everyone.",
            "confidence": 0.95
        },
        {
            "start": 5.5,
            "end": 10.0,
            "speaker": "Speaker_B",
            "text": "Thank you for organizing this session.",
            "confidence": 0.92
        },
        {
            "start": 10.5,
            "end": 15.0,
            "speaker": "Speaker_C",
            "text": "I have some updates to share with the team.",
            "confidence": 0.88
        }
    ]


@pytest.fixture
def sample_confidence_scores():
    """Sample confidence scores for testing"""
    return {
        "D_diarization": 0.85,
        "A_asr_alignment": 0.90,
        "L_linguistic": 0.78,
        "R_agreement": 0.82,
        "O_overlap": 0.75,
        "final_score": 0.82
    }


@pytest.fixture
def mock_ensemble_results(sample_segments, sample_confidence_scores):
    """Mock ensemble processing results"""
    
    # Create 15 mock candidates with varying quality
    candidates = []
    for i in range(15):
        confidence_variation = 1.0 - (i * 0.05)  # Decreasing confidence
        
        candidate = {
            "candidate_id": f"candidate_{i:02d}",
            "aligned_segments": sample_segments.copy(),
            "confidence_scores": {
                k: max(0.1, v * confidence_variation) 
                for k, v in sample_confidence_scores.items()
            },
            "processing_metadata": {
                "asr_model": f"whisper_variant_{i % 5}",
                "diarization_model": f"pyannote_variant_{i // 3}",
                "processing_time": 30.0 + (i * 2.0)
            }
        }
        candidates.append(candidate)
    
    return {
        "candidates": candidates,
        "winner_candidate": candidates[0],  # Best confidence scores
        "processing_summary": {
            "total_candidates": 15,
            "total_processing_time": 120.0,
            "audio_duration": 30.0
        }
    }


@pytest.fixture
def disable_external_apis():
    """Disable external API calls during testing"""
    
    # Mock OpenAI API calls
    with patch('openai.Audio.transcribe') as mock_transcribe:
        mock_transcribe.return_value = Mock(text="Mocked transcription result")
        
        # Mock pyannote pipeline
        with patch('pyannote.audio.Pipeline.from_pretrained') as mock_pipeline:
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = {
                "segments": [
                    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
                    {"start": 5.5, "end": 10.0, "speaker": "SPEAKER_01"}
                ]
            }
            mock_pipeline.return_value = mock_pipeline_instance
            
            yield {
                "openai_transcribe": mock_transcribe,
                "pyannote_pipeline": mock_pipeline
            }


@pytest.fixture
def quality_thresholds():
    """Standard quality thresholds for testing"""
    return {
        "baseline_scenarios": {
            "der_max": 0.15,
            "wer_max": 0.12,
            "confidence_calibration_min": 0.7,
            "processing_time_max": 60.0
        },
        "challenging_scenarios": {
            "der_max": 0.35,
            "wer_max": 0.35,
            "confidence_calibration_min": 0.4,
            "processing_time_max": 120.0
        },
        "edge_cases": {
            "der_max": 0.50,
            "wer_max": 0.50,
            "confidence_calibration_min": 0.2,
            "processing_time_max": 180.0
        }
    }


@pytest.fixture
def regression_tolerances():
    """Regression tolerance settings for testing"""
    return {
        "der_increase_max": 0.05,
        "wer_increase_max": 0.05,
        "confidence_decrease_max": 0.10,
        "processing_time_increase_max": 20.0,
        "ensemble_agreement_decrease_max": 0.15
    }


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "acceptance: mark test as acceptance test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "regression: mark test as regression test"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file patterns"""
    
    for item in items:
        # Add markers based on test file names
        if "test_acceptance" in item.nodeid:
            item.add_marker(pytest.mark.acceptance)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if "test_ensemble_pipeline" in item.nodeid or "test_real_video" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# Session management
@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Automatic session setup for all tests"""
    logger.info("Starting test session for ensemble transcription system")
    
    # Ensure test directories exist
    test_dirs = [
        "test_reports",
        "tests/temp_files",
        "artifacts/test_outputs"
    ]
    
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    yield
    
    logger.info("Test session completed")


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Monitor test performance and resource usage"""
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    yield
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    duration = end_time - start_time
    memory_delta = end_memory - start_memory
    
    # Log performance if test took more than 10 seconds
    if duration > 10.0:
        logger.warning(f"Slow test detected: {duration:.2f}s, memory delta: {memory_delta/1024/1024:.1f}MB")


# Error handling and debugging fixtures
@pytest.fixture
def debug_on_failure():
    """Enhanced debugging information on test failure"""
    import traceback
    
    yield
    
    # This would typically be used with pytest-xdist for debugging
    # In case of failure, additional debug info can be collected here