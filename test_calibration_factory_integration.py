"""
Unit Test for CalibrationEngine Integration with ASRProviderFactory

Tests the complete calibration workflow from ASR provider creation,
calibration engine setting, to confidence calibration with proper
penalty logic ordering.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from core.asr_providers.factory import ASRProviderFactory
from core.asr_providers.base import ASRProvider, DecodeMode, ASRResult, ASRSegment
from core.per_engine_calibration import CalibrationEngine, CalibrationData, CalibratedResult


class MockASRProvider(ASRProvider):
    """Mock ASR provider for testing"""
    
    def __init__(self, provider_name: str, model_name: str, config: dict = None):
        super().__init__(provider_name, model_name, config)
        self.is_available_result = True
    
    def transcribe(self, audio_path, decode_mode=DecodeMode.DETERMINISTIC, 
                  language="en", prompt=None, **kwargs):
        # Mock transcription result
        segments = [ASRSegment(0.0, 1.0, "test text", 0.8)]
        return ASRResult(
            segments=segments,
            full_text="test text",
            language="en",
            confidence=0.8,
            calibrated_confidence=0.75,
            processing_time=1.0,
            provider=self.provider_name,
            decode_mode=decode_mode,
            model_name=self.model_name,
            metadata={}
        )
    
    def is_available(self) -> bool:
        return self.is_available_result
    
    def get_supported_formats(self) -> list:
        return ["wav", "mp3", "mp4"]
    
    def get_max_file_size(self) -> int:
        return 25 * 1024 * 1024  # 25MB


class TestCalibrationEngineIntegration:
    """Test CalibrationEngine integration with ASRProviderFactory"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Register mock provider with factory
        ASRProviderFactory.register_provider("test-provider", MockASRProvider)
        
        # Reset calibration engine
        ASRProviderFactory._calibration_engine = None
    
    def teardown_method(self):
        """Clean up after tests"""
        # Remove mock provider
        if "test-provider" in ASRProviderFactory._providers:
            del ASRProviderFactory._providers["test-provider"]
        
        # Reset calibration engine
        ASRProviderFactory._calibration_engine = None
    
    def test_factory_creates_calibration_engine(self):
        """Test that factory creates and sets CalibrationEngine correctly"""
        # Create provider - should initialize calibration engine
        provider = ASRProviderFactory.create_provider("test-provider")
        
        # Verify calibration engine was created
        assert ASRProviderFactory._calibration_engine is not None
        assert isinstance(ASRProviderFactory._calibration_engine, CalibrationEngine)
        
        # Verify provider received calibration engine
        assert provider._calibration_engine is not None
        assert provider._calibration_engine is ASRProviderFactory._calibration_engine
    
    def test_shared_calibration_engine_across_providers(self):
        """Test that all providers share the same CalibrationEngine instance"""
        # Create multiple providers
        provider1 = ASRProviderFactory.create_provider("test-provider")
        provider2 = ASRProviderFactory.create_provider("test-provider")
        
        # Verify they share the same calibration engine
        assert provider1._calibration_engine is provider2._calibration_engine
        assert provider1._calibration_engine is ASRProviderFactory._calibration_engine
    
    def test_calibration_engine_persistence(self):
        """Test that CalibrationEngine persists across factory operations"""
        # Create first provider
        provider1 = ASRProviderFactory.create_provider("test-provider")
        calibration_engine = ASRProviderFactory._calibration_engine
        
        # Create second provider
        provider2 = ASRProviderFactory.create_provider("test-provider")
        
        # Verify same engine instance is used
        assert ASRProviderFactory._calibration_engine is calibration_engine
        assert provider2._calibration_engine is calibration_engine
    
    def test_confidence_calibration_with_per_engine_enabled(self):
        """Test confidence calibration when per-engine calibration is enabled"""
        provider = ASRProviderFactory.create_provider(
            "test-provider", 
            config={"use_per_engine_calibration": True}
        )
        
        # Mock the calibration engine's calibrate_confidence method
        mock_result = CalibratedResult(
            raw_confidence=0.8,
            calibrated_confidence=0.75,
            calibration_method="isotonic_regression",
            provider="test-provider",
            confidence_delta=-0.05
        )
        
        with patch.object(provider._calibration_engine, 'calibrate_confidence', 
                         return_value=mock_result) as mock_calibrate:
            
            # Test calibration with normal segment
            calibrated = provider.calibrate_confidence(0.8, segment_length=1.0)
            
            # Verify calibration was called
            mock_calibrate.assert_called_once_with(0.8, "test-provider")
            
            # Should return calibrated value
            assert calibrated == 0.75
    
    def test_confidence_calibration_fallback_to_legacy(self):
        """Test confidence calibration falls back to legacy when per-engine fails"""
        provider = ASRProviderFactory.create_provider(
            "test-provider", 
            config={"use_per_engine_calibration": True}
        )
        
        # Mock calibration engine to raise exception
        with patch.object(provider._calibration_engine, 'calibrate_confidence', 
                         side_effect=Exception("Calibration failed")):
            
            # Test calibration - should fallback to legacy
            calibrated = provider.calibrate_confidence(0.8, segment_length=1.0)
            
            # Should use legacy calibration (scale=1.0, offset=0.0)
            assert calibrated == 0.8
    
    def test_penalty_logic_ordering_very_short_segments(self):
        """Test that very short segments (< 0.2s) get stronger penalty"""
        provider = ASRProviderFactory.create_provider("test-provider")
        
        # Mock calibration engine
        mock_result = CalibratedResult(
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            calibration_method="isotonic_regression",
            provider="test-provider",
            confidence_delta=0.0
        )
        
        with patch.object(provider._calibration_engine, 'calibrate_confidence', 
                         return_value=mock_result):
            
            # Very short segment should get 0.9 penalty multiplier
            calibrated = provider.calibrate_confidence(0.9, segment_length=0.1)
            expected = 0.9 * 0.9  # 0.81
            assert abs(calibrated - expected) < 0.001
    
    def test_penalty_logic_ordering_short_segments(self):
        """Test that short segments (< 0.5s but >= 0.2s) get moderate penalty"""
        provider = ASRProviderFactory.create_provider("test-provider")
        
        # Mock calibration engine
        mock_result = CalibratedResult(
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            calibration_method="isotonic_regression",
            provider="test-provider",
            confidence_delta=0.0
        )
        
        with patch.object(provider._calibration_engine, 'calibrate_confidence', 
                         return_value=mock_result):
            
            # Short segment should get 0.95 penalty multiplier
            calibrated = provider.calibrate_confidence(0.9, segment_length=0.3)
            expected = 0.9 * 0.95  # 0.855
            assert abs(calibrated - expected) < 0.001
    
    def test_penalty_logic_ordering_normal_segments(self):
        """Test that normal segments (>= 0.5s) get no penalty"""
        provider = ASRProviderFactory.create_provider("test-provider")
        
        # Mock calibration engine
        mock_result = CalibratedResult(
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            calibration_method="isotonic_regression",
            provider="test-provider",
            confidence_delta=0.0
        )
        
        with patch.object(provider._calibration_engine, 'calibrate_confidence', 
                         return_value=mock_result):
            
            # Normal segment should get no penalty
            calibrated = provider.calibrate_confidence(0.9, segment_length=1.0)
            assert calibrated == 0.9
    
    def test_legacy_penalty_logic_ordering(self):
        """Test legacy calibration penalty logic ordering"""
        provider = ASRProviderFactory.create_provider(
            "test-provider", 
            config={"use_per_engine_calibration": False}
        )
        
        # Test very short segment
        calibrated_very_short = provider.calibrate_confidence(0.9, segment_length=0.1)
        expected_very_short = 0.9 * 0.8  # 0.72
        assert abs(calibrated_very_short - expected_very_short) < 0.001
        
        # Test short segment
        calibrated_short = provider.calibrate_confidence(0.9, segment_length=0.3)
        expected_short = 0.9 * 0.9  # 0.81
        assert abs(calibrated_short - expected_short) < 0.001
        
        # Test normal segment
        calibrated_normal = provider.calibrate_confidence(0.9, segment_length=1.0)
        assert calibrated_normal == 0.9
    
    def test_ensemble_creation_with_calibration(self):
        """Test that ensemble creation properly sets calibration engines"""
        # Make provider available
        with patch.object(MockASRProvider, 'is_available', return_value=True):
            ensemble = ASRProviderFactory.create_ensemble(["test-provider"])
            
            # Verify ensemble has provider
            assert len(ensemble) == 1
            assert ensemble[0].provider_name == "test-provider"
            
            # Verify calibration engine is set
            assert ensemble[0]._calibration_engine is not None
            assert ensemble[0]._calibration_engine is ASRProviderFactory._calibration_engine
    
    def test_get_calibration_engine(self):
        """Test get_calibration_engine class method"""
        # Initially should be None
        assert ASRProviderFactory.get_calibration_engine() is None
        
        # Create provider to initialize engine
        ASRProviderFactory.create_provider("test-provider")
        
        # Now should return the engine
        engine = ASRProviderFactory.get_calibration_engine()
        assert engine is not None
        assert isinstance(engine, CalibrationEngine)
    
    def test_set_calibration_engine(self):
        """Test set_calibration_engine class method"""
        # Create custom calibration engine
        custom_engine = CalibrationEngine()
        
        # Set it via class method
        ASRProviderFactory.set_calibration_engine(custom_engine)
        
        # Verify it was set
        assert ASRProviderFactory.get_calibration_engine() is custom_engine
        
        # Create provider - should use the custom engine
        provider = ASRProviderFactory.create_provider("test-provider")
        assert provider._calibration_engine is custom_engine


def test_calibration_integration():
    """Main test function to run all integration tests"""
    print("🧪 Running CalibrationEngine Integration Tests...")
    
    # Run the test suite
    test_suite = TestCalibrationEngineIntegration()
    
    try:
        # Test each component
        test_suite.setup_method()
        test_suite.test_factory_creates_calibration_engine()
        test_suite.teardown_method()
        
        test_suite.setup_method()
        test_suite.test_shared_calibration_engine_across_providers()
        test_suite.teardown_method()
        
        test_suite.setup_method()
        test_suite.test_confidence_calibration_with_per_engine_enabled()
        test_suite.teardown_method()
        
        test_suite.setup_method()
        test_suite.test_penalty_logic_ordering_very_short_segments()
        test_suite.teardown_method()
        
        test_suite.setup_method()
        test_suite.test_penalty_logic_ordering_short_segments()
        test_suite.teardown_method()
        
        test_suite.setup_method()
        test_suite.test_legacy_penalty_logic_ordering()
        test_suite.teardown_method()
        
        print("✅ All CalibrationEngine integration tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_calibration_integration()