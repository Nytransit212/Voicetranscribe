#!/usr/bin/env python3
"""
Test script for alignment-aware fusion integration
"""

import json
import sys
from typing import Dict, Any, List

# Import the modules to test
try:
    from core.consensus_module import ConsensusModule
    from core.alignment_fusion import AlignmentAwareFusionEngine
    print("✓ Successfully imported alignment fusion modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def create_mock_candidates() -> List[Dict[str, Any]]:
    """Create mock candidates for testing"""
    candidates = []
    
    for i in range(3):
        candidate = {
            'candidate_id': f'test_candidate_{i}',
            'diarization_variant_id': i % 3,
            'asr_variant_id': i,
            'confidence_scores': {
                'D_diarization': 0.8 + i * 0.05,
                'A_asr_alignment': 0.7 + i * 0.1,
                'L_linguistic': 0.75 + i * 0.05,
                'R_agreement': 0.6 + i * 0.1,
                'O_overlap': 0.65 + i * 0.08,
                'final_score': 0.7 + i * 0.08
            },
            'asr_data': {
                'text': f"This is test transcript {i} with some variation",
                'words': [
                    {'word': 'This', 'start': 0.0, 'end': 0.3, 'confidence': 0.9},
                    {'word': 'is', 'start': 0.3, 'end': 0.5, 'confidence': 0.8},
                    {'word': 'test', 'start': 0.5, 'end': 0.8, 'confidence': 0.85},
                    {'word': 'transcript', 'start': 0.8, 'end': 1.4, 'confidence': 0.9},
                    {'word': str(i), 'start': 1.4, 'end': 1.6, 'confidence': 0.7},  # Number variation
                    {'word': 'with', 'start': 1.6, 'end': 1.9, 'confidence': 0.8},
                    {'word': 'some', 'start': 1.9, 'end': 2.1, 'confidence': 0.85},
                    {'word': 'variation', 'start': 2.1, 'end': 2.7, 'confidence': 0.8}
                ],
                'segments': []
            },
            'aligned_segments': [
                {
                    'start': 0.0,
                    'end': 2.7,
                    'text': f"This is test transcript {i} with some variation",
                    'speaker_id': f'SPEAKER_0{i % 2}',
                    'words': [
                        {'word': 'This', 'start': 0.0, 'end': 0.3, 'confidence': 0.9},
                        {'word': 'is', 'start': 0.3, 'end': 0.5, 'confidence': 0.8},
                        {'word': 'test', 'start': 0.5, 'end': 0.8, 'confidence': 0.85},
                        {'word': 'transcript', 'start': 0.8, 'end': 1.4, 'confidence': 0.9},
                        {'word': str(i), 'start': 1.4, 'end': 1.6, 'confidence': 0.7},
                        {'word': 'with', 'start': 1.6, 'end': 1.9, 'confidence': 0.8},
                        {'word': 'some', 'start': 1.9, 'end': 2.1, 'confidence': 0.85},
                        {'word': 'variation', 'start': 2.1, 'end': 2.7, 'confidence': 0.8}
                    ]
                }
            ],
            'parameters': {
                'diarization': {'variant_id': i % 3},
                'asr': {'variant_id': i, 'temperature': 0.1 * i}
            }
        }
        candidates.append(candidate)
    
    return candidates

def test_consensus_module_integration():
    """Test that the alignment-aware fusion is properly integrated"""
    print("\n--- Testing ConsensusModule Integration ---")
    
    try:
        # Create consensus module
        consensus_module = ConsensusModule()
        
        # Check that the new strategy is available
        available_strategies = consensus_module.get_available_strategies()
        print(f"Available strategies: {available_strategies}")
        
        if "alignment_aware_fusion" in available_strategies:
            print("✓ Alignment-aware fusion strategy is registered")
        else:
            print("✗ Alignment-aware fusion strategy is NOT registered")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing consensus module: {e}")
        return False

def test_alignment_fusion_engine():
    """Test the alignment fusion engine directly"""
    print("\n--- Testing AlignmentAwareFusionEngine ---")
    
    try:
        # Create fusion engine
        fusion_engine = AlignmentAwareFusionEngine()
        print("✓ AlignmentAwareFusionEngine created successfully")
        
        # Create mock candidates
        candidates = create_mock_candidates()
        print(f"✓ Created {len(candidates)} mock candidates")
        
        # Test fusion
        fusion_result = fusion_engine.fuse_candidates_with_alignment(candidates)
        print("✓ Fusion completed successfully")
        
        # Check results
        print(f"  - Fused transcript: {fusion_result.fused_transcript[:100]}...")
        print(f"  - Word alignments: {len(fusion_result.word_alignments)}")
        print(f"  - Confusion sets: {len(fusion_result.confusion_sets)}")
        print(f"  - Fusion confidence: {fusion_result.confidence_weighted_score:.3f}")
        print(f"  - Fusion effectiveness: {fusion_result.alignment_metrics.fusion_effectiveness:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing fusion engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consensus_with_alignment_fusion():
    """Test full consensus processing with alignment-aware fusion"""
    print("\n--- Testing Full Consensus with Alignment Fusion ---")
    
    try:
        # Create consensus module
        consensus_module = ConsensusModule()
        
        # Create mock candidates
        candidates = create_mock_candidates()
        
        # Test alignment-aware fusion strategy
        result = consensus_module.process_consensus(
            candidates=candidates,
            strategy="alignment_aware_fusion"
        )
        
        print("✓ Consensus processing with alignment fusion completed")
        print(f"  - Method: {result.consensus_method}")
        print(f"  - Confidence: {result.consensus_confidence:.3f}")
        print(f"  - Winner ID: {result.winner_candidate.get('candidate_id', 'unknown')}")
        
        # Check if fusion metadata exists
        if 'fusion_metadata' in result.winner_candidate:
            print("✓ Fusion metadata found in winner candidate")
            fusion_meta = result.winner_candidate['fusion_metadata']
            print(f"  - Original candidates: {fusion_meta.get('original_candidates_count', 0)}")
            print(f"  - Fusion confidence: {fusion_meta.get('fusion_confidence', 0.0):.3f}")
        else:
            print("! No fusion metadata found (may be using fallback)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing full consensus: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Testing Alignment-Aware Fusion Integration ===")
    
    tests = [
        test_consensus_module_integration,
        test_alignment_fusion_engine,
        test_consensus_with_alignment_fusion
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASSED")
            else:
                failed += 1
                print("✗ FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ FAILED with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed! Integration is successful.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)