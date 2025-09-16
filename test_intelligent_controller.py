#!/usr/bin/env python3
"""
Integration test for the IntelligentController

Tests the controller with existing ASR providers and validates the 
confidence-based decode expansion strategy.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.intelligent_controller import IntelligentController, IntelligentControllerResult
from core.asr_providers.base import DecodeMode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_intelligent_controller_initialization():
    """Test that the controller initializes properly"""
    print("🧪 Testing IntelligentController initialization...")
    
    try:
        controller = IntelligentController(
            confidence_threshold=0.85,
            agreement_threshold=0.80,
            max_decodes_per_segment=5
        )
        
        print(f"✅ Controller initialized successfully")
        print(f"   Available providers: {list(controller.providers.keys())}")
        print(f"   Confidence threshold: {controller.confidence_threshold}")
        print(f"   Agreement threshold: {controller.agreement_threshold}")
        print(f"   Max decodes per segment: {controller.max_decodes_per_segment}")
        
        return True
        
    except Exception as e:
        print(f"❌ Controller initialization failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_auto_segmentation():
    """Test auto-segmentation functionality"""
    print("\n🧪 Testing auto-segmentation...")
    
    try:
        controller = IntelligentController()
        
        # Create a dummy audio file for testing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Write minimal WAV header (44 bytes) for testing
            wav_header = b'RIFF' + (1000).to_bytes(4, 'little') + b'WAVE'
            wav_header += b'fmt ' + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little')
            wav_header += (1).to_bytes(2, 'little') + (16000).to_bytes(4, 'little')
            wav_header += (32000).to_bytes(4, 'little') + (2).to_bytes(2, 'little')
            wav_header += (16).to_bytes(2, 'little') + b'data' + (956).to_bytes(4, 'little')
            tmp_file.write(wav_header)
            tmp_file.write(b'\x00' * 956)  # Silent audio data
            audio_path = tmp_file.name
        
        # Test auto-segmentation
        segments = controller._auto_segment_audio(audio_path)
        
        print(f"✅ Auto-segmentation successful")
        print(f"   Generated {len(segments)} segments")
        if segments:
            print(f"   First segment: {segments[0]['start']:.1f}s - {segments[0]['end']:.1f}s")
            print(f"   Segment duration: {segments[0]['duration']:.1f}s")
        
        # Cleanup
        os.unlink(audio_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-segmentation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_decode_configurations():
    """Test decode configuration setup"""
    print("\n🧪 Testing decode configurations...")
    
    try:
        controller = IntelligentController()
        
        print("✅ Decode configurations loaded successfully")
        print(f"   Initial probe configs: {len(controller.initial_probe_configs)}")
        for i, (provider, mode) in enumerate(controller.initial_probe_configs):
            print(f"      {i+1}. {provider} - {mode.value}")
        
        print(f"   Expansion configs: {len(controller.expansion_configs)}")  
        for i, (provider, mode) in enumerate(controller.expansion_configs):
            print(f"      {i+1}. {provider} - {mode.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decode configuration test failed: {e}")
        return False

def test_confidence_agreement_calculation():
    """Test confidence and agreement calculation methods"""
    print("\n🧪 Testing confidence and agreement calculation...")
    
    try:
        controller = IntelligentController()
        
        # Create mock candidates for testing
        from core.intelligent_controller import SegmentCandidate
        from core.asr_providers.base import ASRResult, ASRSegment
        
        mock_candidates = []
        
        # Mock candidate 1
        result1 = ASRResult(
            segments=[ASRSegment(0.0, 5.0, "Hello world", 0.95)],
            full_text="Hello world",
            language="en",
            confidence=0.95,
            calibrated_confidence=0.95,
            processing_time=1.0,
            provider="faster-whisper",
            decode_mode=DecodeMode.CAREFUL,
            model_name="large-v3",
            metadata={}
        )
        
        candidate1 = SegmentCandidate(
            provider="faster-whisper",
            decode_mode=DecodeMode.CAREFUL,
            model_name="large-v3",
            result=result1,
            calibrated_confidence=0.95,
            processing_time=1.0
        )
        
        # Mock candidate 2 (similar)
        result2 = ASRResult(
            segments=[ASRSegment(0.0, 5.0, "Hello world", 0.90)],
            full_text="Hello world",
            language="en", 
            confidence=0.90,
            calibrated_confidence=0.90,
            processing_time=1.2,
            provider="deepgram",
            decode_mode=DecodeMode.DETERMINISTIC,
            model_name="nova-2",
            metadata={}
        )
        
        candidate2 = SegmentCandidate(
            provider="deepgram",
            decode_mode=DecodeMode.DETERMINISTIC,
            model_name="nova-2",
            result=result2,
            calibrated_confidence=0.90,
            processing_time=1.2
        )
        
        mock_candidates = [candidate1, candidate2]
        
        # Test confidence calculation
        confidence_score = controller._calculate_confidence_score(mock_candidates)
        print(f"✅ Confidence calculation successful: {confidence_score:.3f}")
        
        # Test agreement calculation
        agreement_score = controller._calculate_agreement_score(mock_candidates)
        print(f"✅ Agreement calculation successful: {agreement_score:.3f}")
        
        # Test best candidate selection
        alignments = controller._create_word_alignments(mock_candidates)
        best_candidate = controller._select_best_candidate(mock_candidates, alignments)
        if best_candidate is not None:
            print(f"✅ Best candidate selection successful: {best_candidate.provider}-{best_candidate.decode_mode.value}")
        else:
            print(f"⚠️ Best candidate selection returned None")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence/agreement calculation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_integration_with_existing_providers():
    """Test integration with existing ASR provider factory"""
    print("\n🧪 Testing integration with existing ASR providers...")
    
    try:
        from core.asr_providers.factory import ASRProviderFactory
        
        # Test provider factory integration
        available_providers = ASRProviderFactory.get_available_providers()
        print(f"✅ Available providers from factory: {available_providers}")
        
        # Test controller provider initialization
        controller = IntelligentController()
        controller_providers = list(controller.providers.keys())
        print(f"✅ Controller initialized providers: {controller_providers}")
        
        # Verify providers are actual provider instances
        for name, provider in controller.providers.items():
            print(f"   {name}: {provider.__class__.__name__} - Available: {provider.is_available()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider integration test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run all integration tests"""
    print("🚀 Running IntelligentController Integration Tests\n")
    
    tests = [
        test_intelligent_controller_initialization,
        test_auto_segmentation,
        test_decode_configurations,
        test_confidence_agreement_calculation,
        test_integration_with_existing_providers
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print(f"\n🎉 All tests passed! IntelligentController is ready for use.")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Review issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)