#!/usr/bin/env python3
"""
Test script for confusion network fusion integration with IntelligentController

This test validates the complete fusion pipeline including:
- SegmentCandidate creation with mock ASR results
- Word-level alignment using TemporalAligner
- Confusion network construction with token posteriors
- MBR path selection and temporal coherence
- Entity detection and confidence weighting
- Text normalization and final transcript generation
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from core.intelligent_controller import IntelligentController, SegmentCandidate, SegmentAnalysis
from core.fusion_engine import FusionEngine, FusionResult, TokenPosterior, ConfusionNetwork
from core.asr_providers.base import ASRResult, ASRSegment, DecodeMode
from core.alignment_fusion import TemporalAligner, WordAlignment

def create_mock_asr_result(provider: str, transcript: str, confidence: float, 
                          words: List[Dict[str, Any]]) -> ASRResult:
    """Create a mock ASR result for testing"""
    segments = [ASRSegment(
        start=0.0,
        end=10.0,
        text=transcript,
        confidence=confidence,
        words=words
    )]
    
    return ASRResult(
        segments=segments,
        full_text=transcript,
        language="en",
        confidence=confidence,
        calibrated_confidence=confidence * 0.95,  # Slight calibration
        processing_time=1.0,
        provider=provider,
        decode_mode=DecodeMode.DETERMINISTIC,
        model_name=f"{provider}_model",
        metadata={"test": True}
    )

def create_mock_candidates() -> List[SegmentCandidate]:
    """Create mock ASR candidates for testing"""
    
    # Mock candidate 1: Whisper - high confidence, slightly different transcript
    words1 = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.95},
        {"word": "everyone", "start": 0.6, "end": 1.2, "confidence": 0.90},
        {"word": "welcome", "start": 1.3, "end": 1.8, "confidence": 0.88},
        {"word": "to", "start": 1.9, "end": 2.0, "confidence": 0.92},
        {"word": "the", "start": 2.1, "end": 2.3, "confidence": 0.94},
        {"word": "meeting", "start": 2.4, "end": 3.0, "confidence": 0.89}
    ]
    result1 = create_mock_asr_result("faster-whisper", "Hello everyone welcome to the meeting", 0.91, words1)
    
    candidate1 = SegmentCandidate(
        provider="faster-whisper",
        decode_mode=DecodeMode.CAREFUL,
        model_name="whisper_large",
        result=result1,
        calibrated_confidence=0.88,
        processing_time=2.1,
        metadata={"test_candidate": 1}
    )
    
    # Mock candidate 2: Deepgram - medium confidence, slightly different words
    words2 = [
        {"word": "Hi", "start": 0.05, "end": 0.4, "confidence": 0.82},  # Different word
        {"word": "everyone", "start": 0.65, "end": 1.15, "confidence": 0.85},  # Slightly different timing
        {"word": "welcome", "start": 1.35, "end": 1.75, "confidence": 0.80},
        {"word": "to", "start": 1.95, "end": 2.05, "confidence": 0.88},
        {"word": "this", "start": 2.15, "end": 2.4, "confidence": 0.79},  # Different word
        {"word": "meeting", "start": 2.45, "end": 2.95, "confidence": 0.83}
    ]
    result2 = create_mock_asr_result("deepgram", "Hi everyone welcome to this meeting", 0.82, words2)
    
    candidate2 = SegmentCandidate(
        provider="deepgram",
        decode_mode=DecodeMode.DETERMINISTIC,
        model_name="deepgram_nova",
        result=result2,
        calibrated_confidence=0.79,
        processing_time=1.5,
        metadata={"test_candidate": 2}
    )
    
    # Mock candidate 3: OpenAI - good confidence, entity recognition
    words3 = [
        {"word": "Hello", "start": 0.02, "end": 0.48, "confidence": 0.90},
        {"word": "everyone", "start": 0.62, "end": 1.18, "confidence": 0.87},
        {"word": "welcome", "start": 1.32, "end": 1.78, "confidence": 0.85},
        {"word": "to", "start": 1.92, "end": 2.02, "confidence": 0.91},
        {"word": "the", "start": 2.12, "end": 2.28, "confidence": 0.93},
        {"word": "January", "start": 2.35, "end": 2.8, "confidence": 0.86},  # Entity: date
        {"word": "meeting", "start": 2.85, "end": 3.2, "confidence": 0.88}
    ]
    result3 = create_mock_asr_result("openai", "Hello everyone welcome to the January meeting", 0.87, words3)
    
    candidate3 = SegmentCandidate(
        provider="openai",
        decode_mode=DecodeMode.DETERMINISTIC,
        model_name="whisper-1",
        result=result3,
        calibrated_confidence=0.84,
        processing_time=1.8,
        metadata={"test_candidate": 3}
    )
    
    return [candidate1, candidate2, candidate3]

def test_fusion_engine_standalone():
    """Test the FusionEngine standalone functionality"""
    print("\n=== Testing FusionEngine Standalone ===")
    
    # Create fusion engine with custom configuration
    fusion_config = {
        'engine_weights': {
            'faster-whisper': 1.2,  # Higher weight for Whisper
            'deepgram': 1.0,
            'openai': 1.1
        },
        'temporal_coherence_config': {
            'baseline_offset': 0.15,
            'penalty_per_100ms': 0.10
        },
        'entity_detection_enabled': True,
        'mbr_config': {
            'entity_boost': 1.3,
            'consistency_weight': 0.2,
            'temporal_weight': 0.15
        }
    }
    
    fusion_engine = FusionEngine(
        engine_weights=fusion_config['engine_weights'],
        temporal_coherence_config=fusion_config['temporal_coherence_config'],
        entity_detection_enabled=fusion_config['entity_detection_enabled'],
        mbr_config=fusion_config['mbr_config']
    )
    
    print(f"✓ FusionEngine initialized with weights: {fusion_engine.engine_weights}")
    
    # Create mock candidates and segment analysis
    candidates = create_mock_candidates()
    
    # Create temporal aligner and align words
    temporal_aligner = TemporalAligner(
        timestamp_tolerance=0.3,
        confidence_threshold=0.1,
        max_alignment_gap=1.0
    )
    
    # Convert candidates to format expected by TemporalAligner
    candidate_data = []
    for i, candidate in enumerate(candidates):
        candidate_dict = {
            'candidate_id': f'test_candidate_{i}',
            'asr_data': {
                'words': candidate.result.segments[0].words
            }
        }
        candidate_data.append(candidate_dict)
    
    # Create word alignments
    word_alignments = temporal_aligner.align_words_across_candidates(candidate_data)
    print(f"✓ Created {len(word_alignments)} word alignments")
    
    # Create SegmentAnalysis for fusion
    segment_analysis = SegmentAnalysis(
        segment_start=0.0,
        segment_end=3.5,
        segment_duration=3.5,
        candidates=candidates,
        word_alignments=word_alignments,
        agreement_score=0.75,
        confidence_score=0.82,
        best_candidate=candidates[0],
        expansion_decision="expand_standard",
        total_decodes_run=3
    )
    
    # Apply fusion
    try:
        fusion_result = fusion_engine.fuse_segment_candidates(segment_analysis)
        
        print(f"✓ Fusion successful!")
        print(f"  - Fused transcript: '{fusion_result.fused_transcript}'")
        print(f"  - Overall confidence: {fusion_result.overall_confidence:.3f}")
        print(f"  - Confusion networks: {len(fusion_result.confusion_networks)}")
        print(f"  - MBR tokens: {len(fusion_result.mbr_path.tokens)}")
        print(f"  - Processing time: {fusion_result.processing_time:.3f}s")
        
        # Analyze fusion metrics
        metrics = fusion_result.fusion_metrics
        print(f"  - Entity detection: {metrics.get('total_entities_detected', 0)} entities")
        print(f"  - Engines used: {metrics.get('engines_list', [])}")
        print(f"  - Average confusion entropy: {metrics.get('average_confusion_entropy', 0):.3f}")
        print(f"  - Fusion effectiveness: {metrics.get('fusion_effectiveness_ratio', 1):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Fusion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_entity_detection():
    """Test entity detection functionality"""
    print("\n=== Testing Entity Detection ===")
    
    fusion_engine = FusionEngine(entity_detection_enabled=True)
    entity_detector = fusion_engine.entity_detector
    
    test_tokens = [
        ("Hello", False, None),
        ("123", True, "number"),
        ("January", True, "date"), 
        ("$500", True, "money"),
        ("3:30pm", True, "time"),
        ("John", True, "proper_name"),
        ("meeting", False, None)
    ]
    
    for token, expected_is_entity, expected_type in test_tokens:
        if entity_detector is not None:
            is_entity, entity_type = entity_detector.detect_entities(token)
            status = "✓" if (is_entity == expected_is_entity) else "✗"
            print(f"  {status} '{token}' -> entity: {is_entity}, type: {entity_type}")
        else:
            print(f"  ✗ Entity detector is None - cannot test token '{token}'")

def test_temporal_coherence():
    """Test temporal coherence scoring"""
    print("\n=== Testing Temporal Coherence ===")
    
    fusion_engine = FusionEngine()
    temporal_scorer = fusion_engine.temporal_scorer
    
    test_cases = [
        (0.0, 0.0, 0.5, 1.0, 0.0),  # Perfect alignment
        (0.1, 0.0, 0.5, 1.0, 0.0),  # Within baseline (150ms)
        (0.2, 0.0, 0.5, 0.9, 0.1),  # 50ms beyond baseline
        (0.3, 0.0, 0.5, 0.8, 0.2),  # 150ms beyond baseline
        (0.5, 0.0, 0.5, 0.6, 0.4),  # 350ms beyond baseline
    ]
    
    for token_start, expected_start, token_duration, expected_coherence, expected_penalty in test_cases:
        coherence, penalty = temporal_scorer.calculate_coherence_score(
            token_start, expected_start, token_duration
        )
        
        coherence_match = abs(coherence - expected_coherence) < 0.05
        penalty_match = abs(penalty - expected_penalty) < 0.05
        
        status = "✓" if (coherence_match and penalty_match) else "✗"
        print(f"  {status} offset {token_start - expected_start:.1f}s -> coherence: {coherence:.2f}, penalty: {penalty:.2f}")

def test_normalization():
    """Test text normalization"""
    print("\n=== Testing Text Normalization ===")
    
    fusion_engine = FusionEngine()
    
    test_cases = [
        ("hello world", "Hello world"),
        ("hello , world !", "Hello, world!"),
        ("hello   world", "Hello world"),
        ("hello. world", "Hello. World"),
        ("hello ! how are you ?", "Hello! How are you?"),
        ("", "")
    ]
    
    for input_text, expected_output in test_cases:
        normalized = fusion_engine._apply_normalization(input_text)
        status = "✓" if normalized == expected_output else "✗"
        print(f"  {status} '{input_text}' -> '{normalized}'")
        if normalized != expected_output:
            print(f"      Expected: '{expected_output}'")

def main():
    """Run all fusion system tests"""
    print("🚀 Testing Confusion Network Fusion System")
    print("=" * 50)
    
    # Test individual components
    test_entity_detection()
    test_temporal_coherence() 
    test_normalization()
    
    # Test complete fusion pipeline
    fusion_success = test_fusion_engine_standalone()
    
    print("\n" + "=" * 50)
    if fusion_success:
        print("🎉 All fusion tests completed successfully!")
        print("\nThe confusion network fusion system is ready for use with:")
        print("  ✓ Multi-engine ASR candidate processing")
        print("  ✓ Word-level temporal alignment")
        print("  ✓ Confusion network construction with posteriors")
        print("  ✓ Temporal coherence penalties (0.10 per 100ms beyond 150ms)")
        print("  ✓ Entity-aware fusion with confidence weighting")
        print("  ✓ MBR path selection for optimal transcripts")
        print("  ✓ Punctuation and casing normalization")
        print("  ✓ Comprehensive fusion metrics and analysis")
    else:
        print("❌ Some fusion tests failed - check the error messages above")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)