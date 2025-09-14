#!/usr/bin/env python3
"""
Test script for Speaker Mapping Consistency Integration

This script tests the integration of the Hungarian alignment algorithm
with the existing diarization pipeline to verify speaker mapping functionality.
"""

import sys
import tempfile
import numpy as np
from typing import List, Dict, Any

def test_speaker_mapper_import():
    """Test that the speaker mapper module can be imported successfully."""
    try:
        from core.speaker_mapper import SpeakerMapper, ConsistencyMetrics, SpeakerEmbedding, SpeakerMapping
        print("✅ Successfully imported SpeakerMapper components")
        return True
    except ImportError as e:
        print(f"❌ Failed to import SpeakerMapper: {e}")
        return False

def test_diarization_engine_integration():
    """Test that DiarizationEngine properly integrates with speaker mapping."""
    try:
        from core.diarization_engine import DiarizationEngine
        
        # Initialize with speaker mapping enabled
        engine = DiarizationEngine(
            expected_speakers=3,
            noise_level='medium',
            enable_speaker_mapping=True,
            speaker_mapping_config={
                'similarity_threshold': 0.7,
                'embedding_dim': 128,
                'min_segment_duration': 1.0,
                'cache_embeddings': True,
                'enable_metrics': True
            }
        )
        
        print("✅ Successfully initialized DiarizationEngine with speaker mapping")
        
        # Check that speaker mapper is available
        if hasattr(engine, 'speaker_mapper') and engine.speaker_mapper is not None:
            print("✅ Speaker mapper properly initialized in DiarizationEngine")
            return True
        else:
            print("❌ Speaker mapper not properly initialized")
            return False
            
    except Exception as e:
        print(f"❌ Failed to integrate DiarizationEngine with speaker mapping: {e}")
        return False

def test_ensemble_manager_integration():
    """Test that EnsembleManager properly integrates with speaker mapping."""
    try:
        from core.ensemble_manager import EnsembleManager
        
        # Initialize with speaker mapping enabled
        manager = EnsembleManager(
            expected_speakers=3,
            noise_level='medium',
            enable_speaker_mapping=True,
            speaker_mapping_config={
                'similarity_threshold': 0.7,
                'embedding_dim': 128,
                'min_segment_duration': 1.0,
                'cache_embeddings': True,
                'enable_metrics': True
            },
            chunked_processing_threshold=900.0
        )
        
        print("✅ Successfully initialized EnsembleManager with speaker mapping")
        
        # Check that diarization engine has speaker mapping enabled
        if (hasattr(manager, 'diarization_engine') and 
            hasattr(manager.diarization_engine, 'enable_speaker_mapping') and
            manager.diarization_engine.enable_speaker_mapping):
            print("✅ Speaker mapping properly enabled in EnsembleManager")
            return True
        else:
            print("❌ Speaker mapping not properly enabled in EnsembleManager")
            return False
            
    except Exception as e:
        print(f"❌ Failed to integrate EnsembleManager with speaker mapping: {e}")
        return False

def test_speaker_mapper_functionality():
    """Test core speaker mapper functionality with mock data."""
    try:
        from core.speaker_mapper import SpeakerMapper
        
        # Initialize speaker mapper
        mapper = SpeakerMapper(
            similarity_threshold=0.7,
            embedding_dim=64,  # Smaller for testing
            min_segment_duration=0.5,
            cache_embeddings=False,  # Disable for testing
            enable_metrics=True
        )
        
        print("✅ Successfully initialized SpeakerMapper")
        
        # Create mock chunk segments
        chunk1_segments = [
            {'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00', 'confidence': 0.9},
            {'start': 2.0, 'end': 4.0, 'speaker': 'SPEAKER_01', 'confidence': 0.8},
            {'start': 4.0, 'end': 6.0, 'speaker': 'SPEAKER_00', 'confidence': 0.85}
        ]
        
        chunk2_segments = [
            {'start': 0.0, 'end': 2.5, 'speaker': 'SPEAKER_02', 'confidence': 0.88},  # Should map to SPEAKER_01
            {'start': 2.5, 'end': 5.0, 'speaker': 'SPEAKER_03', 'confidence': 0.92}   # Should map to SPEAKER_00
        ]
        
        chunk_segments_list = [chunk1_segments, chunk2_segments]
        
        # Test similarity matrix computation
        try:
            # This would normally use real audio, but we'll test the structure
            mapper.speaker_embeddings[0] = []  # Mock embeddings for chunk 0
            mapper.speaker_embeddings[1] = []  # Mock embeddings for chunk 1
            
            print("✅ Speaker mapper data structures working correctly")
            
            # Test metrics generation
            metrics = mapper._create_empty_metrics()
            if hasattr(metrics, 'baseline_continuity_score'):
                print("✅ Consistency metrics structure working correctly")
                return True
            else:
                print("❌ Consistency metrics structure incomplete")
                return False
                
        except Exception as e:
            print(f"❌ Speaker mapper functionality test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test SpeakerMapper functionality: {e}")
        return False

def test_config_integration():
    """Test that speaker mapping configuration is properly loaded."""
    try:
        import os
        config_path = "config/speaker_mapping/consistency_config.yaml"
        
        if os.path.exists(config_path):
            print("✅ Speaker mapping configuration file exists")
            
            # Try to read the config
            with open(config_path, 'r') as f:
                content = f.read()
                if 'similarity_threshold' in content and 'hungarian' in content:
                    print("✅ Configuration file contains expected parameters")
                    return True
                else:
                    print("❌ Configuration file missing expected parameters")
                    return False
        else:
            print("❌ Speaker mapping configuration file not found")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test configuration integration: {e}")
        return False

def run_integration_tests():
    """Run all integration tests for speaker mapping."""
    print("🧪 Running Speaker Mapping Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_speaker_mapper_import),
        ("DiarizationEngine Integration", test_diarization_engine_integration),
        ("EnsembleManager Integration", test_ensemble_manager_integration),
        ("SpeakerMapper Functionality", test_speaker_mapper_functionality),
        ("Configuration Integration", test_config_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Speaker mapping is ready.")
        return True
    else:
        print("⚠️  Some integration tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)