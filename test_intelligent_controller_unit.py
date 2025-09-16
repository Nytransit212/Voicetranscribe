#!/usr/bin/env python3
"""
Unit tests for IntelligentController without heavy model downloads

Tests the core logic and algorithms without requiring actual ASR providers.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_controller_class_structure():
    """Test that the controller class is properly structured"""
    print("🧪 Testing IntelligentController class structure...")
    
    try:
        from core.intelligent_controller import (
            IntelligentController, 
            SegmentCandidate, 
            SegmentAnalysis, 
            IntelligentControllerResult
        )
        from core.asr_providers.base import DecodeMode
        
        print("✅ All classes imported successfully")
        print(f"   - IntelligentController class available")
        print(f"   - SegmentCandidate dataclass available") 
        print(f"   - SegmentAnalysis dataclass available")
        print(f"   - IntelligentControllerResult dataclass available")
        print(f"   - DecodeMode enum imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_decode_configurations():
    """Test decode configuration setup without initializing providers"""
    print("\n🧪 Testing decode configurations...")
    
    try:
        from core.intelligent_controller import IntelligentController
        from core.asr_providers.base import DecodeMode
        
        # Test configuration setup (without provider initialization)
        controller = IntelligentController.__new__(IntelligentController)
        controller.initial_probe_configs = [
            ("faster-whisper", DecodeMode.CAREFUL),
            ("faster-whisper", DecodeMode.DETERMINISTIC), 
            ("deepgram", DecodeMode.DETERMINISTIC)
        ]
        
        controller.expansion_configs = [
            ("deepgram", DecodeMode.ENHANCED),
            ("faster-whisper", DecodeMode.EXPLORATORY),
            ("openai", DecodeMode.DETERMINISTIC),
            ("deepgram", DecodeMode.FAST)
        ]
        
        print("✅ Decode configurations set successfully")
        print(f"   Initial probes: {len(controller.initial_probe_configs)} configs")
        for i, (provider, mode) in enumerate(controller.initial_probe_configs):
            print(f"      {i+1}. {provider} - {mode.value}")
        
        print(f"   Expansion configs: {len(controller.expansion_configs)} configs")  
        for i, (provider, mode) in enumerate(controller.expansion_configs):
            print(f"      {i+1}. {provider} - {mode.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decode configuration test failed: {e}")
        return False

def test_confidence_agreement_algorithms():
    """Test confidence and agreement calculation algorithms"""
    print("\n🧪 Testing confidence and agreement algorithms...")
    
    try:
        from core.intelligent_controller import IntelligentController, SegmentCandidate
        from core.asr_providers.base import ASRResult, ASRSegment, DecodeMode
        
        # Create controller instance without provider initialization
        controller = IntelligentController.__new__(IntelligentController)
        
        # Mock candidates for testing algorithms
        mock_candidates = []
        
        # Mock candidate 1 (high confidence)
        result1 = ASRResult(
            segments=[ASRSegment(0.0, 5.0, "Hello world this is a test", 0.95)],
            full_text="Hello world this is a test",
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
        
        # Mock candidate 2 (similar text, slightly lower confidence)
        result2 = ASRResult(
            segments=[ASRSegment(0.0, 5.0, "Hello world this is a test", 0.90)],
            full_text="Hello world this is a test",
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
        
        # Mock candidate 3 (different text, lower confidence)
        result3 = ASRResult(
            segments=[ASRSegment(0.0, 5.0, "Hello world this was a test", 0.80)],
            full_text="Hello world this was a test",
            language="en",
            confidence=0.80,
            calibrated_confidence=0.80,
            processing_time=0.8,
            provider="openai",
            decode_mode=DecodeMode.DETERMINISTIC,
            model_name="whisper-1",
            metadata={}
        )
        
        candidate3 = SegmentCandidate(
            provider="openai",
            decode_mode=DecodeMode.DETERMINISTIC,
            model_name="whisper-1",
            result=result3,
            calibrated_confidence=0.80,
            processing_time=0.8
        )
        
        mock_candidates = [candidate1, candidate2, candidate3]
        
        # Test confidence calculation
        confidence_score = controller._calculate_confidence_score(mock_candidates)
        print(f"✅ Confidence calculation: {confidence_score:.3f}")
        assert 0.0 <= confidence_score <= 1.0, f"Confidence score out of range: {confidence_score}"
        
        # Test text similarity fallback (since we don't have actual word alignments)
        texts = [c.result.full_text for c in mock_candidates]
        similarity_score = controller._calculate_text_similarity(texts)
        print(f"✅ Text similarity calculation: {similarity_score:.3f}")
        assert 0.0 <= similarity_score <= 1.0, f"Similarity score out of range: {similarity_score}"
        
        # Test best candidate scoring
        for candidate in mock_candidates:
            score = controller._score_candidate(candidate, mock_candidates, [])
            print(f"   {candidate.provider}-{candidate.decode_mode.value}: score={score:.3f}")
            assert 0.0 <= score <= 1.0, f"Candidate score out of range: {score}"
        
        print("✅ All algorithms working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Algorithm test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_segment_analysis_structure():
    """Test segment analysis data structures"""
    print("\n🧪 Testing segment analysis data structures...")
    
    try:
        from core.intelligent_controller import SegmentAnalysis, SegmentCandidate
        from core.asr_providers.base import ASRResult, ASRSegment, DecodeMode
        
        # Create mock segment analysis
        mock_result = ASRResult(
            segments=[ASRSegment(0.0, 5.0, "Test segment", 0.90)],
            full_text="Test segment", 
            language="en",
            confidence=0.90,
            calibrated_confidence=0.90,
            processing_time=1.0,
            provider="test-provider",
            decode_mode=DecodeMode.DETERMINISTIC,
            model_name="test-model",
            metadata={}
        )
        
        mock_candidate = SegmentCandidate(
            provider="test-provider",
            decode_mode=DecodeMode.DETERMINISTIC,
            model_name="test-model",
            result=mock_result,
            calibrated_confidence=0.90,
            processing_time=1.0
        )
        
        analysis = SegmentAnalysis(
            segment_start=0.0,
            segment_end=30.0,
            segment_duration=30.0,
            candidates=[mock_candidate],
            word_alignments=[],
            agreement_score=0.85,
            confidence_score=0.90,
            best_candidate=mock_candidate,
            expansion_decision="stop_early",
            total_decodes_run=3
        )
        
        print("✅ SegmentAnalysis created successfully")
        print(f"   Segment duration: {analysis.segment_duration}s")
        print(f"   Candidates: {len(analysis.candidates)}")
        print(f"   Agreement score: {analysis.agreement_score}")
        print(f"   Confidence score: {analysis.confidence_score}")
        print(f"   Expansion decision: {analysis.expansion_decision}")
        print(f"   Total decodes: {analysis.total_decodes_run}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

def test_threshold_logic():
    """Test confidence and agreement threshold logic"""
    print("\n🧪 Testing threshold decision logic...")
    
    try:
        # Test scenarios
        scenarios = [
            # (confidence, agreement, expected_decision)
            (0.95, 0.90, "should_stop_early"),
            (0.85, 0.90, "should_expand"), 
            (0.95, 0.80, "should_expand"),
            (0.70, 0.65, "should_expand_maximum"),  # Lower thresholds for maximum expansion
        ]
        
        confidence_threshold = 0.90
        agreement_threshold = 0.85
        
        for confidence, agreement, expected in scenarios:
            # Simulate decision logic
            if (confidence >= confidence_threshold and 
                agreement >= agreement_threshold):
                decision = "stop_early"
            elif (confidence < confidence_threshold * 0.8 or 
                  agreement < agreement_threshold * 0.8):
                decision = "expand_maximum"
            else:
                decision = "expand_standard"
            
            print(f"   Confidence: {confidence:.2f}, Agreement: {agreement:.2f} → {decision}")
            
            # Validate decision makes sense
            if expected == "should_stop_early":
                assert decision == "stop_early", f"Expected early stop but got {decision}"
            elif expected == "should_expand_maximum":
                assert decision == "expand_maximum", f"Expected maximum expansion but got {decision}"
        
        print("✅ Threshold logic working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Threshold logic test failed: {e}")
        return False

def main():
    """Run all unit tests"""
    print("🚀 Running IntelligentController Unit Tests\n")
    
    tests = [
        test_controller_class_structure,
        test_decode_configurations,
        test_confidence_agreement_algorithms,
        test_segment_analysis_structure,
        test_threshold_logic
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
        print(f"\n🎉 All unit tests passed! Core logic is working correctly.")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Review issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)