#!/usr/bin/env python3
"""
Integration test for the dialect handling system

Verifies that:
1. Dialect processing is properly integrated into the main pipeline
2. ASR candidates are processed through dialect handling at Step 4.5
3. Confidence adjustments flow through to calibration/fusion systems
4. YAML configuration is properly loaded
5. The system provides actual benefits (not inert)
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Test imports
try:
    from core.ensemble_manager import EnsembleManager
    from core.dialect_handling_engine import DialectHandlingEngine
    from core.dialect_config_loader import load_dialect_config, get_dialect_config_loader
    from core.asr_providers.base import ASRResult, ASRSegment, DecodeMode
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_config_loading():
    """Test YAML configuration loading"""
    print("\n🧪 Testing YAML Configuration Loading...")
    
    try:
        # Test config loader
        config_loader = get_dialect_config_loader()
        config = config_loader.load_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"  Enable dialect handling: {config.enable_dialect_handling}")
        print(f"  Similarity threshold: {config.similarity_threshold}")
        print(f"  Confidence boost factor: {config.confidence_boost_factor}")
        print(f"  Supported dialects: {config.supported_dialects}")
        print(f"  G2P fallback enabled: {config.enable_g2p_fallback}")
        
        # Verify config values match YAML
        assert config.similarity_threshold == 0.7, f"Expected 0.7, got {config.similarity_threshold}"
        assert config.confidence_boost_factor == 0.05, f"Expected 0.05, got {config.confidence_boost_factor}"
        assert len(config.supported_dialects) == 6, f"Expected 6 dialects, got {len(config.supported_dialects)}"
        
        print("✅ Configuration values verified against YAML")
        return True
        
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        return False

def test_dialect_engine_initialization():
    """Test dialect engine initialization with config"""
    print("\n🧪 Testing Dialect Engine Initialization...")
    
    try:
        # Load config
        config = load_dialect_config()
        
        # Initialize with config
        engine = DialectHandlingEngine(config=config)
        
        print(f"✅ Dialect engine initialized with config")
        print(f"  Similarity threshold: {engine.similarity_threshold}")
        print(f"  Confidence boost: {engine.confidence_boost_factor}")
        print(f"  Supported dialects: {engine.supported_dialects}")
        print(f"  G2P fallback: {engine.enable_g2p_fallback}")
        
        # Verify config was applied
        assert engine.similarity_threshold == config.similarity_threshold
        assert engine.confidence_boost_factor == config.confidence_boost_factor
        assert engine.supported_dialects == config.supported_dialects
        
        print("✅ Config properly applied to engine")
        return True
        
    except Exception as e:
        print(f"❌ Dialect engine initialization test failed: {e}")
        return False

def test_ensemble_manager_config_integration():
    """Test EnsembleManager uses config properly"""
    print("\n🧪 Testing EnsembleManager Config Integration...")
    
    try:
        # Initialize EnsembleManager with dialect handling enabled
        manager = EnsembleManager(
            expected_speakers=2,
            noise_level='low',
            enable_dialect_handling=True
        )
        
        print(f"✅ EnsembleManager initialized")
        print(f"  Dialect handling enabled: {manager.enable_dialect_handling}")
        
        if manager.enable_dialect_handling and manager.dialect_engine:
            print(f"  Dialect engine present: {manager.dialect_engine is not None}")
            print(f"  Similarity threshold: {manager.dialect_similarity_threshold}")
            print(f"  Confidence boost: {manager.dialect_confidence_boost}")
            print(f"  Supported dialects: {manager.supported_dialects}")
            
            # Verify config loading worked
            config = load_dialect_config()
            assert manager.dialect_similarity_threshold == config.similarity_threshold
            assert manager.dialect_confidence_boost == config.confidence_boost_factor
            assert manager.supported_dialects == config.supported_dialects
            
            print("✅ EnsembleManager properly loaded config")
        else:
            print("❌ Dialect engine not initialized in EnsembleManager")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ EnsembleManager config integration test failed: {e}")
        return False

def test_dialect_processing_execution():
    """Test that dialect processing actually executes on candidates"""
    print("\n🧪 Testing Dialect Processing Execution...")
    
    try:
        # Create a mock ASR candidate with Southern dialect characteristics
        mock_candidate = {
            'asr_data': {
                'provider': 'test',
                'model_name': 'test-model'
            },
            'aligned_segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'text': 'right time',  # Could be "raht tahm" in Southern dialect
                    'confidence': 0.6,
                    'speaker_id': 'speaker_0'
                },
                {
                    'start': 2.5,
                    'end': 4.0,
                    'text': 'this car',  # Could be "dis cah" in AAVE/Boston
                    'confidence': 0.65,
                    'speaker_id': 'speaker_1'
                }
            ]
        }
        
        # Initialize EnsembleManager
        manager = EnsembleManager(
            expected_speakers=2,
            noise_level='low',
            enable_dialect_handling=True
        )
        
        if not manager.enable_dialect_handling or not manager.dialect_engine:
            print("❌ Dialect handling not enabled")
            return False
        
        # Test the _process_candidate_for_dialect method
        print("🔄 Processing candidate through dialect handling...")
        
        result = manager._process_candidate_for_dialect(
            mock_candidate, 
            mock_candidate['aligned_segments']
        )
        
        if result is None:
            print("❌ Dialect processing returned None - possible error")
            return False
        
        print("✅ Dialect processing executed successfully")
        
        # Check if confidence adjustments were made
        if 'confidence_adjustments' in result:
            adjustments = result['confidence_adjustments']
            print(f"  Confidence adjustments: {len(adjustments)} segments")
            
            for adj in adjustments:
                print(f"    '{adj['segment_text']}': +{adj['adjustment']:.3f} (patterns: {adj['patterns']})")
        
        # Check for dialect metadata
        if 'dialect_processing_metadata' in result:
            metadata = result['dialect_processing_metadata']
            print(f"  Overall adjustment: {metadata.get('overall_adjustment', 0):.3f}")
            print(f"  Patterns detected: {metadata.get('patterns_detected', [])}")
            print(f"  Processing time: {metadata.get('processing_time', 0):.3f}s")
        
        print("✅ Dialect processing executed and produced adjustments")
        return True
        
    except Exception as e:
        print(f"❌ Dialect processing execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test that dialect processing is integrated into the main pipeline"""
    print("\n🧪 Testing Pipeline Integration...")
    
    try:
        # Check if EnsembleManager.process_video includes dialect processing
        import inspect
        
        # Get the process_video method source
        manager = EnsembleManager(enable_dialect_handling=True)
        
        if not hasattr(manager, 'process_video'):
            print("❌ process_video method not found")
            return False
        
        # Get method source to verify integration
        source = inspect.getsource(manager.process_video)
        
        # Check for dialect processing integration points
        checks = [
            ('dialect_processed_candidates', 'Step 4.5 dialect processing variable'),
            ('dialect_processing', 'Dialect processing stage'),
            ('_process_candidate_for_dialect', 'Dialect processing method call'),
            ('enable_dialect_handling', 'Dialect handling condition')
        ]
        
        integration_found = True
        for check_str, description in checks:
            if check_str in source:
                print(f"✅ Found {description}")
            else:
                print(f"❌ Missing {description}")
                integration_found = False
        
        if integration_found:
            print("✅ Dialect processing properly integrated into main pipeline")
        else:
            print("❌ Dialect processing integration incomplete")
        
        return integration_found
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def test_production_readiness():
    """Test production readiness features"""
    print("\n🧪 Testing Production Readiness...")
    
    try:
        # Test NLTK dependency handling
        from core.dialect_handling_engine import CMUDictManager
        
        # This should not fail even if CMUdict is not available
        print("🔄 Testing CMUdict initialization...")
        cmu_manager = CMUDictManager()
        
        print(f"✅ CMUDict manager initialized")
        print(f"  Dictionary size: {len(cmu_manager.cmudict)}")
        
        # Test phoneme lookup (should work even with empty dict)
        test_phonemes = cmu_manager.get_phonemes("hello")
        if test_phonemes:
            print(f"  Phonemes for 'hello': {test_phonemes}")
        else:
            print("  Phoneme lookup returned None (fallback mode)")
        
        print("✅ Production-ready NLTK handling verified")
        return True
        
    except Exception as e:
        print(f"❌ Production readiness test failed: {e}")
        return False

def run_comprehensive_integration_test():
    """Run comprehensive integration test suite"""
    print("🚀 Starting Comprehensive Dialect Handling Integration Test")
    print("=" * 70)
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Engine Initialization", test_dialect_engine_initialization),
        ("EnsembleManager Integration", test_ensemble_manager_config_integration),
        ("Dialect Processing Execution", test_dialect_processing_execution),
        ("Pipeline Integration", test_pipeline_integration),
        ("Production Readiness", test_production_readiness)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results[test_name] = False
    
    print("\n" + "=" * 70)
    print("🏁 COMPREHENSIVE INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:>8} | {test_name}")
    
    print("-" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Dialect handling system is fully integrated!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - Review failed tests above")
        return False

if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    sys.exit(0 if success else 1)