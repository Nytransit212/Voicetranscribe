"""
Test script for Turn Stabilization functionality

This script validates that the turn stabilization system works correctly
with the existing diarization pipeline.
"""

import os
import tempfile
import json
from core.diarization_engine import DiarizationEngine
from core.turn_stabilizer import TurnStabilizer, StabilizationConfig
import numpy as np
from typing import List, Dict, Any

def create_test_audio_file() -> str:
    """Create a test audio file for testing"""
    # Create a temporary audio file path
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        test_audio_path = f.name
    
    # Create a mock audio file (just an empty file for testing)
    with open(test_audio_path, 'wb') as f:
        f.write(b'RIFF' + b'\x00' * 40)  # Minimal WAV header
    
    return test_audio_path

def create_test_segments_with_rapid_transitions() -> List[Dict[str, Any]]:
    """Create test segments that include rapid transitions to test stabilization"""
    
    segments = [
        # Normal segment
        {'start': 0.0, 'end': 2.0, 'speaker_id': 'SPEAKER_00', 'confidence': 0.9},
        
        # Rapid transition sequence (should be stabilized)
        {'start': 2.0, 'end': 2.05, 'speaker_id': 'SPEAKER_01', 'confidence': 0.7},  # 50ms - rapid
        {'start': 2.05, 'end': 2.08, 'speaker_id': 'SPEAKER_00', 'confidence': 0.6}, # 30ms - rapid  
        {'start': 2.08, 'end': 2.12, 'speaker_id': 'SPEAKER_01', 'confidence': 0.65}, # 40ms - rapid
        {'start': 2.12, 'end': 2.15, 'speaker_id': 'SPEAKER_00', 'confidence': 0.7}, # 30ms - rapid
        
        # Normal segments again
        {'start': 2.15, 'end': 5.0, 'speaker_id': 'SPEAKER_00', 'confidence': 0.85},
        {'start': 5.0, 'end': 8.0, 'speaker_id': 'SPEAKER_01', 'confidence': 0.9},
        
        # Another short segment that should be merged
        {'start': 8.0, 'end': 8.2, 'speaker_id': 'SPEAKER_02', 'confidence': 0.6},  # 200ms - should be merged
        
        # Final normal segment
        {'start': 8.2, 'end': 10.0, 'speaker_id': 'SPEAKER_01', 'confidence': 0.88}
    ]
    
    return segments

def test_turn_stabilizer_standalone():
    """Test the TurnStabilizer class independently"""
    print("\n=== Testing TurnStabilizer Standalone ===")
    
    # Create test configuration
    config = StabilizationConfig(
        rapid_transition_threshold=0.1,  # 100ms
        median_window_size=5,
        min_turn_duration=0.5,  # 500ms
        enable_median_filtering=True,
        enable_min_duration_enforcement=True
    )
    
    # Initialize stabilizer
    stabilizer = TurnStabilizer(config)
    
    # Create test segments with rapid transitions
    test_segments = create_test_segments_with_rapid_transitions()
    
    print(f"Original segments: {len(test_segments)}")
    for i, seg in enumerate(test_segments):
        duration = seg['end'] - seg['start']
        print(f"  {i}: {seg['speaker_id']} [{seg['start']:.3f}-{seg['end']:.3f}] ({duration:.3f}s)")
    
    # Apply stabilization
    stabilized_segments, metrics = stabilizer.stabilize_segments(test_segments, "test_variant")
    
    print(f"\nStabilized segments: {len(stabilized_segments)}")
    for i, seg in enumerate(stabilized_segments):
        duration = seg['end'] - seg['start']
        print(f"  {i}: {seg['speaker_id']} [{seg['start']:.3f}-{seg['end']:.3f}] ({duration:.3f}s)")
    
    # Print metrics
    print(f"\nStabilization Metrics:")
    print(f"  - Original transitions: {metrics.original_transitions_count}")
    print(f"  - Stabilized transitions: {metrics.stabilized_transitions_count}")
    print(f"  - Transitions eliminated: {metrics.transitions_eliminated}")
    print(f"  - Rapid transitions eliminated: {metrics.rapid_transitions_eliminated}")
    print(f"  - Stability improvement: {metrics.stability_improvement_ratio:.3f}")
    
    # Generate report
    report = stabilizer.generate_stabilization_report(metrics)
    print(f"\nStabilization Report:")
    print(json.dumps(report, indent=2))
    
    return metrics.transitions_eliminated > 0

def test_diarization_engine_integration():
    """Test turn stabilization integration with DiarizationEngine"""
    print("\n=== Testing DiarizationEngine Integration ===")
    
    # Create test audio file
    test_audio_path = create_test_audio_file()
    
    try:
        # Create DiarizationEngine with turn stabilization enabled
        config = StabilizationConfig(
            rapid_transition_threshold=0.1,
            min_turn_duration=0.5,
            enable_median_filtering=True,
            enable_min_duration_enforcement=True
        )
        
        engine = DiarizationEngine(
            expected_speakers=3,
            enable_turn_stabilization=True,
            stabilization_config=config
        )
        
        print(f"DiarizationEngine created with turn stabilization: {engine.enable_turn_stabilization}")
        print(f"Turn stabilizer initialized: {engine.turn_stabilizer is not None}")
        
        # Create diarization variants (this will use mock pipeline)
        print("\nCreating diarization variants with turn stabilization...")
        variants = engine.create_diarization_variants(test_audio_path, use_voting_fusion=False)
        
        print(f"Generated {len(variants)} variants")
        
        # Check if variants have stabilization data
        stabilized_count = 0
        for i, variant in enumerate(variants):
            has_stabilization = variant.get('processed_with_stabilization', False)
            stabilization_metrics = variant.get('stabilization_metrics')
            
            print(f"\nVariant {i+1} ({variant.get('variant_name', 'Unknown')}):")
            print(f"  - Processed with stabilization: {has_stabilization}")
            print(f"  - Original segments: {variant.get('original_segment_count', 0)}")
            print(f"  - Stabilized segments: {variant.get('stabilized_segment_count', 0)}")
            
            if stabilization_metrics:
                print(f"  - Transitions eliminated: {stabilization_metrics.transitions_eliminated}")
                print(f"  - Rapid transitions eliminated: {stabilization_metrics.rapid_transitions_eliminated}")
                stabilized_count += 1
        
        print(f"\nSuccessfully stabilized {stabilized_count}/{len(variants)} variants")
        return stabilized_count == len(variants)
        
    finally:
        # Clean up test file
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)

def test_rapid_transition_detection():
    """Test rapid transition detection algorithm"""
    print("\n=== Testing Rapid Transition Detection ===")
    
    stabilizer = TurnStabilizer()
    test_segments = create_test_segments_with_rapid_transitions()
    
    rapid_transitions = stabilizer._detect_rapid_transitions(test_segments)
    
    print(f"Detected {len(rapid_transitions)} rapid transitions:")
    for transition in rapid_transitions:
        print(f"  - Index {transition['index']}: {transition['current_speaker']} -> {transition['next_speaker']} "
              f"(gap: {transition['gap_duration']:.3f}s)")
    
    # Should detect the rapid transitions in our test data
    expected_rapid_count = 3  # Based on our test data
    return len(rapid_transitions) >= expected_rapid_count

def test_median_filtering():
    """Test median filtering functionality"""
    print("\n=== Testing Median Filtering ===")
    
    config = StabilizationConfig(
        median_window_size=5,
        min_consensus_ratio=0.6,
        enable_median_filtering=True,
        enable_min_duration_enforcement=False  # Test only filtering
    )
    
    stabilizer = TurnStabilizer(config)
    test_segments = create_test_segments_with_rapid_transitions()
    
    # Apply only median filtering
    stabilized_segments, metrics = stabilizer.stabilize_segments(test_segments, "median_test")
    
    print(f"Median filtering results:")
    print(f"  - Original segments: {len(test_segments)}")
    print(f"  - Filtered segments: {len(stabilized_segments)}")
    print(f"  - Transitions eliminated: {metrics.transitions_eliminated}")
    
    return len(stabilized_segments) <= len(test_segments)

def run_comprehensive_test():
    """Run all tests and report results"""
    print("🎯 Starting Comprehensive Turn Stabilization Tests")
    print("=" * 60)
    
    tests = [
        ("Turn Stabilizer Standalone", test_turn_stabilizer_standalone),
        ("Rapid Transition Detection", test_rapid_transition_detection),
        ("Median Filtering", test_median_filtering),
        ("DiarizationEngine Integration", test_diarization_engine_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔧 Running: {test_name}")
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            print(f"✅ {test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status_emoji = "✅" if result == "PASS" else "❌"
        print(f"{status_emoji} {test_name}: {result}")
    
    passed_tests = sum(1 for result in results.values() if result == "PASS")
    total_tests = len(results)
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Turn stabilization is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)