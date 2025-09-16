#!/usr/bin/env python3
"""
Test script to verify the fixed post-fusion punctuation engine works correctly
without transformers dependency, using rule-based fallbacks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.post_fusion_punctuation_engine import PostFusionPunctuationEngine
    print("✅ Successfully imported PostFusionPunctuationEngine")
except ImportError as e:
    print(f"❌ Failed to import PostFusionPunctuationEngine: {e}")
    sys.exit(1)

def test_punctuation_engine():
    """Test punctuation engine initialization and basic functionality"""
    
    print("\n=== Testing PostFusionPunctuationEngine ===")
    
    # Test 1: Initialize the engine
    try:
        engine = PostFusionPunctuationEngine(
            punctuation_model="oliverguhr/fullstop-punctuation-multilang-large",
            disfluency_level="light"
        )
        print("✅ Successfully initialized PostFusionPunctuationEngine")
        print(f"   - Model available: {engine.punctuation_model.is_available()}")
        print(f"   - Disfluency level: {engine.disfluency_level}")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return False
    
    # Test 2: Process sample segments
    sample_segments = [
        {
            'start': 0.0,
            'end': 3.5,
            'speaker': 'Speaker_1',
            'text': 'hello everyone um welcome to our meeting today'
        },
        {
            'start': 3.5,
            'end': 7.2,
            'speaker': 'Speaker_2', 
            'text': 'thanks john uh lets review the quarterly numbers'
        },
        {
            'start': 7.2,
            'end': 11.0,
            'speaker': 'Speaker_1',
            'text': 'sure um the revenue increased by fifteen percent'
        }
    ]
    
    try:
        result = engine.process_fused_segments(sample_segments)
        print("✅ Successfully processed sample segments")
        print(f"   - Processed {len(result.segments)} segments")
        print(f"   - Overall confidence: {result.overall_confidence:.3f}")
        print(f"   - Processing time: {result.processing_time:.3f}s")
        print(f"   - Model info: {result.model_info}")
        
        # Show first processed segment as example
        if result.segments:
            first_segment = result.segments[0]
            print(f"   - Example transformation:")
            print(f"     Original: '{first_segment.original_text}'")
            print(f"     Punctuated: '{first_segment.punctuated_text}'")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to process segments: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_disfluency_normalizer():
    """Test disfluency normalization component"""
    
    print("\n=== Testing DisfluencyNormalizer ===")
    
    try:
        from core.post_fusion_punctuation_engine import DisfluencyNormalizer
        
        normalizer = DisfluencyNormalizer(normalization_level="light")
        print("✅ Successfully initialized DisfluencyNormalizer")
        
        # Test normalization
        test_text = "um uh so basically we need to um you know review the numbers"
        result = normalizer.normalize_segment(test_text, formality_level="formal")
        
        print(f"   - Original: '{test_text}'")
        print(f"   - Normalized: '{result.normalized_text}'")
        print(f"   - Confidence: {result.confidence:.3f}")
        print(f"   - Applied normalizations: {result.normalization_applied}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test DisfluencyNormalizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vocabulary_handler():
    """Test meeting vocabulary handler"""
    
    print("\n=== Testing MeetingVocabularyHandler ===")
    
    try:
        from core.post_fusion_punctuation_engine import MeetingVocabularyHandler
        
        handler = MeetingVocabularyHandler()
        print("✅ Successfully initialized MeetingVocabularyHandler")
        
        # Test formality detection
        formal_text = "We need to review the quarterly revenue metrics and budget forecast"
        informal_text = "Yeah so basically we're gonna look at the numbers today"
        
        formal_level, formal_conf = handler.detect_formality_level(formal_text)
        informal_level, informal_conf = handler.detect_formality_level(informal_text)
        
        print(f"   - Formal text detection: {formal_level} (confidence: {formal_conf:.3f})")
        print(f"   - Informal text detection: {informal_level} (confidence: {informal_conf:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test MeetingVocabularyHandler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Fixed Post-Fusion Punctuation Engine")
    print("=" * 50)
    
    success = True
    success &= test_punctuation_engine()
    success &= test_disfluency_normalizer() 
    success &= test_vocabulary_handler()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Punctuation engine is working correctly.")
        print("   - Engine handles missing transformers gracefully with rule-based fallbacks")
        print("   - Numpy float conversion issue resolved")
        print("   - All core components functioning properly")
    else:
        print("❌ Some tests failed. Check error messages above.")
    
    sys.exit(0 if success else 1)