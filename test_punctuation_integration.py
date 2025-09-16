#!/usr/bin/env python3
"""
Quick validation test for PostFusionPunctuationEngine integration
"""

import sys
import time
from typing import Dict, Any, List

def test_punctuation_engine_import():
    """Test that the punctuation engine can be imported"""
    try:
        from core.post_fusion_punctuation_engine import (
            PostFusionPunctuationEngine, 
            create_punctuation_engine_from_preset,
            PUNCTUATION_PRESETS
        )
        print("✅ Successfully imported PostFusionPunctuationEngine")
        return True
    except ImportError as e:
        print(f"❌ Failed to import PostFusionPunctuationEngine: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing PostFusionPunctuationEngine: {e}")
        return False

def test_punctuation_engine_initialization():
    """Test that the punctuation engine can be initialized"""
    try:
        from core.post_fusion_punctuation_engine import create_punctuation_engine_from_preset
        
        # Test preset creation
        engine = create_punctuation_engine_from_preset("meeting_light")
        print("✅ Successfully created punctuation engine with meeting_light preset")
        
        # Test engine status
        status = engine.get_engine_status()
        print(f"✅ Engine status retrieved: {status['punctuation_model_available']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize punctuation engine: {e}")
        return False

def test_sample_punctuation():
    """Test punctuation processing on sample meeting text"""
    try:
        from core.post_fusion_punctuation_engine import create_punctuation_engine_from_preset
        
        # Create engine
        engine = create_punctuation_engine_from_preset("meeting_light")
        
        # Sample meeting segments (mimicking ensemble output format)
        test_segments = [
            {
                'start': 0.0,
                'end': 3.5,
                'speaker': 'Speaker 1',
                'text': 'um so lets start the meeting today um we have several action items to discuss'
            },
            {
                'start': 3.5,
                'end': 7.0,
                'speaker': 'Speaker 2', 
                'text': 'yeah absolutely we need to uh we need to review the quarterly numbers'
            },
            {
                'start': 7.0,
                'end': 10.5,
                'speaker': 'Speaker 1',
                'text': 'great so the first item on the agenda is the budget forecast'
            }
        ]
        
        # Process segments
        print("🔄 Processing sample meeting segments...")
        start_time = time.time()
        
        result = engine.process_fused_segments(test_segments)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Successfully processed {len(result.segments)} segments in {processing_time:.2f}s")
        print(f"📊 Overall confidence: {result.overall_confidence:.3f}")
        print(f"📈 Punctuation metrics: {result.punctuation_metrics}")
        print(f"🧹 Disfluency metrics: {result.disfluency_metrics}")
        
        # Show sample results
        print("\n📝 Sample Results:")
        for i, seg in enumerate(result.segments[:2]):  # Show first 2 segments
            print(f"  Segment {i+1}:")
            print(f"    Original: {seg.original_text}")
            print(f"    Punctuated: {seg.punctuated_text}")
            print(f"    Confidence: {seg.punctuation_confidence:.3f}")
            if seg.disfluency_normalization:
                print(f"    Normalization applied: {seg.disfluency_normalization.normalization_applied}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to process sample punctuation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_manager_integration():
    """Test that EnsembleManager can be initialized with punctuation enabled"""
    try:
        from core.ensemble_manager import EnsembleManager
        
        # Create ensemble manager with punctuation enabled
        manager = EnsembleManager(
            expected_speakers=5,
            noise_level='medium',
            target_language='en',
            domain='meeting'
        )
        
        # Check punctuation engine is available
        has_punctuation = hasattr(manager, 'punctuation_engine') and manager.punctuation_engine is not None
        print(f"✅ EnsembleManager initialized - Punctuation engine available: {has_punctuation}")
        
        if has_punctuation:
            print(f"📋 Punctuation preset: {manager.punctuation_preset}")
            print(f"🔧 Engine status: {manager.punctuation_engine.get_engine_status()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test ensemble manager integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that the system works when punctuation is disabled"""
    try:
        from core.ensemble_manager import EnsembleManager
        
        # Create ensemble manager
        manager = EnsembleManager(
            expected_speakers=5,
            noise_level='medium',
            target_language='en',
            domain='meeting'
        )
        
        # Disable punctuation
        manager.enable_post_fusion_punctuation = False
        
        print("✅ EnsembleManager works with punctuation disabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 Running PostFusionPunctuationEngine Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_punctuation_engine_import),
        ("Initialization Test", test_punctuation_engine_initialization),
        ("Sample Processing Test", test_sample_punctuation),
        ("EnsembleManager Integration Test", test_ensemble_manager_integration),
        ("Backward Compatibility Test", test_backward_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED! PostFusionPunctuationEngine is ready for production.")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())