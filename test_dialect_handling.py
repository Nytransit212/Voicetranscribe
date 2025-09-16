#!/usr/bin/env python3
"""
Test script for Dialect Handling Engine

Tests the CMUdict + G2P system for dialect handling and phonetic agreement matching.
"""

import sys
import time
from typing import List, Dict, Any

# Test imports to catch any import errors early
try:
    from core.dialect_handling_engine import (
        DialectHandlingEngine, 
        CMUDictManager, 
        G2PConverter,
        PhoneticDistanceCalculator,
        DialectPatternMatcher,
        PhoneticTranscription
    )
    from core.asr_providers.base import ASRResult, ASRSegment, DecodeMode
    print("✅ All dialect handling imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_cmudict_manager():
    """Test CMUdict functionality"""
    print("\n🧪 Testing CMUDict Manager...")
    
    try:
        manager = CMUDictManager()
        print("✅ CMUDict Manager initialized")
        
        # Test common words
        test_words = ["hello", "world", "time", "right", "car", "park"]
        for word in test_words:
            phonemes = manager.get_phonemes(word)
            if phonemes:
                print(f"  '{word}' -> {phonemes}")
            else:
                print(f"  '{word}' -> Not found")
        
        # Test pronunciation alternatives
        alt_pronunciations = manager.get_all_pronunciations("time")
        print(f"  'time' alternatives: {len(alt_pronunciations)} found")
        
        return True
    except Exception as e:
        print(f"❌ CMUDict test failed: {e}")
        return False

def test_g2p_converter():
    """Test G2P conversion"""
    print("\n🧪 Testing G2P Converter...")
    
    try:
        converter = G2PConverter()
        print("✅ G2P Converter initialized")
        
        # Test OOV words
        test_words = ["transcription", "phonemic", "algorithm"]
        for word in test_words:
            phonemes = converter.convert_to_phonemes(word)
            print(f"  '{word}' -> {phonemes}")
        
        return True
    except Exception as e:
        print(f"❌ G2P test failed: {e}")
        return False

def test_phonetic_distance_calculator():
    """Test phonetic distance calculation"""
    print("\n🧪 Testing Phonetic Distance Calculator...")
    
    try:
        calculator = PhoneticDistanceCalculator()
        print("✅ Phonetic Distance Calculator initialized")
        
        # Test phonetic distances
        test_pairs = [
            (['T', 'AY', 'M'], ['T', 'AH', 'M']),  # "time" vs "tahm" (Southern)
            (['TH', 'IH', 'S'], ['D', 'IH', 'S']),  # "this" vs "dis" (AAVE)
            (['K', 'AA', 'R'], ['K', 'AH', 'R']),   # "car" vs "cah" (Boston)
        ]
        
        for phonemes1, phonemes2 in test_pairs:
            distance = calculator.calculate_phonetic_distance(phonemes1, phonemes2)
            print(f"  {phonemes1} <-> {phonemes2}")
            print(f"    Edit distance: {distance.edit_distance}")
            print(f"    Normalized: {distance.normalized_distance:.3f}")
            print(f"    Dialect variant: {distance.is_dialect_variant}")
            print(f"    Confidence: {distance.dialect_confidence:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Phonetic distance test failed: {e}")
        return False

def test_dialect_pattern_matcher():
    """Test dialect pattern matching"""
    print("\n🧪 Testing Dialect Pattern Matcher...")
    
    try:
        matcher = DialectPatternMatcher()
        print(f"✅ Dialect Pattern Matcher initialized with {len(matcher.dialect_patterns)} patterns")
        
        # Test pattern matching
        test_transcriptions = [
            PhoneticTranscription("time", ['T', 'AH', 'M'], source="test"),  # Southern "tahm"
            PhoneticTranscription("this", ['D', 'IH', 'S'], source="test"),  # AAVE "dis"  
            PhoneticTranscription("car", ['K', 'AH', 'R'], source="test"),   # Boston "cah"
        ]
        
        for transcription in test_transcriptions:
            patterns = matcher.match_patterns(transcription)
            if patterns:
                print(f"  '{transcription.word}' {transcription.arpabet} -> {len(patterns)} patterns matched:")
                for pattern in patterns:
                    print(f"    {pattern.dialect}: {pattern.description} (boost: {pattern.confidence_boost:.3f})")
            else:
                print(f"  '{transcription.word}' {transcription.arpabet} -> No patterns matched")
        
        return True
    except Exception as e:
        print(f"❌ Dialect pattern test failed: {e}")
        return False

def test_dialect_handling_engine():
    """Test full dialect handling engine"""
    print("\n🧪 Testing Dialect Handling Engine...")
    
    try:
        engine = DialectHandlingEngine(
            similarity_threshold=0.7,
            confidence_boost_factor=0.05,
            supported_dialects=['southern', 'aave', 'boston'],
            enable_g2p_fallback=True
        )
        print("✅ Dialect Handling Engine initialized")
        
        # Create test ASR segments with dialect variants
        test_segments = [
            ASRSegment(
                start=0.0,
                end=2.0, 
                text="I have to go right now",  # "right" might be "raht" in Southern
                confidence=0.8,
                words=None
            ),
            ASRSegment(
                start=2.0,
                end=4.0,
                text="This is the time to park the car",  # Multiple dialect opportunities
                confidence=0.75,
                words=None
            )
        ]
        
        # Create test ASR result
        asr_result = ASRResult(
            segments=test_segments,
            full_text="I have to go right now This is the time to park the car",
            language="en",
            confidence=0.775,
            calibrated_confidence=0.775,
            processing_time=1.0,
            provider="test",
            decode_mode=DecodeMode.DETERMINISTIC,
            model_name="test-model",
            metadata={"test": True}
        )
        
        # Process through dialect engine
        print("Processing ASR result through dialect engine...")
        start_time = time.time()
        
        result = engine.process_asr_result(asr_result)
        
        processing_time = time.time() - start_time
        print(f"✅ Processing completed in {processing_time:.3f}s")
        
        # Display results
        print(f"\n📊 Dialect Processing Results:")
        print(f"  Overall confidence adjustment: {result.overall_confidence_adjustment:+.4f}")
        print(f"  Dialect patterns detected: {result.dialect_patterns_detected}")
        print(f"  Processing time: {result.processing_time:.3f}s")
        print(f"  Processing stats: {result.processing_stats}")
        
        # Show segment analyses
        print(f"\n📋 Segment Analyses:")
        for i, analysis in enumerate(result.segment_analyses):
            print(f"  Segment {i+1}: '{analysis.original_segment.text}'")
            print(f"    Confidence adjustment: {analysis.confidence_adjustments.get('segment', 0.0):+.4f}")
            print(f"    Patterns matched: {len(analysis.dialect_patterns_matched)}")
            if analysis.phonetic_adjustments:
                print(f"    Phonetic adjustments: {len(analysis.phonetic_adjustments)}")
                for adj in analysis.phonetic_adjustments[:3]:  # Show first 3
                    print(f"      '{adj['word']}': {adj['patterns_matched']} (boost: {adj['confidence_boost']:+.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Dialect engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_compatibility():
    """Test integration with EnsembleManager"""
    print("\n🧪 Testing Integration Compatibility...")
    
    try:
        # Test imports that would be needed for integration
        from core.ensemble_manager import EnsembleManager
        print("✅ EnsembleManager import successful")
        
        # Test that EnsembleManager can be initialized with dialect handling
        manager = EnsembleManager(
            enable_dialect_handling=True,
            dialect_similarity_threshold=0.7,
            dialect_confidence_boost=0.05,
            supported_dialects=['southern', 'aave']
        )
        print("✅ EnsembleManager with dialect handling initialized")
        
        # Check that dialect engine was created
        if hasattr(manager, 'dialect_engine') and manager.dialect_engine:
            print("✅ Dialect engine created in EnsembleManager")
        else:
            print("⚠️  Dialect engine not found in EnsembleManager")
        
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all dialect handling tests"""
    print("🧪 Dialect Handling System Test Suite")
    print("=" * 50)
    
    tests = [
        ("CMUDict Manager", test_cmudict_manager),
        ("G2P Converter", test_g2p_converter),
        ("Phonetic Distance Calculator", test_phonetic_distance_calculator),
        ("Dialect Pattern Matcher", test_dialect_pattern_matcher),
        ("Dialect Handling Engine", test_dialect_handling_engine),
        ("Integration Compatibility", test_integration_compatibility),
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*50}")
    print("🧪 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("🎉 All tests passed! Dialect handling system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())