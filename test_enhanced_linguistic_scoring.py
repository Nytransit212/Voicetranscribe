#!/usr/bin/env python3
"""
Test script for enhanced linguistic scoring components in ConfidenceScorer.
Tests the sophisticated heuristics-based analysis for L1 scoring.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.confidence_scorer import ConfidenceScorer
import json

def test_enhanced_linguistic_scoring():
    """Test the enhanced linguistic scoring with various text samples"""
    
    scorer = ConfidenceScorer()
    
    # Test samples with different linguistic quality levels
    test_samples = [
        {
            "name": "high_quality_formal",
            "text": "Good morning, everyone. I would like to discuss the quarterly financial results. Our revenue increased by fifteen percent compared to the previous quarter. The marketing team implemented several effective strategies that contributed to this growth.",
            "expected_score_range": (0.7, 1.0)
        },
        {
            "name": "natural_conversation", 
            "text": "So, um, I was thinking about what you said yesterday. You know, about the project deadline. I think we can probably finish it on time if we, well, if we work together on the main components.",
            "expected_score_range": (0.5, 0.8)
        },
        {
            "name": "high_disfluency",
            "text": "Um, so, like, I was, uh, I was going to, you know, tell you about the, the thing that, um, that happened yesterday. It was, like, really, really important but I, uh, I forgot what I was gonna say.",
            "expected_score_range": (0.2, 0.5)
        },
        {
            "name": "poor_grammar",
            "text": "me go store yesterday. buy many thing. price very high. not happy with service. worker not help me good.",
            "expected_score_range": (0.1, 0.4)
        },
        {
            "name": "repetitive_text",
            "text": "The meeting meeting was good good. We talked talked about many many things things. Everyone everyone said said the same same thing thing over over and and over over again again.",
            "expected_score_range": (0.1, 0.4)
        },
        {
            "name": "good_punctuation",
            "text": "Hello Dr. Smith! How are you today? I wanted to ask about the project we discussed. Can you please review the proposal? It's quite important for our timeline.",
            "expected_score_range": (0.6, 0.9)
        },
        {
            "name": "poor_punctuation", 
            "text": "hello dr smith how are you today i wanted to ask about the project we discussed can you please review the proposal its quite important for our timeline",
            "expected_score_range": (0.2, 0.5)
        },
        {
            "name": "sophisticated_vocabulary",
            "text": "The implementation demonstrates remarkable sophistication in addressing multifaceted challenges. The comprehensive analysis reveals significant opportunities for optimization and enhancement of organizational effectiveness.",
            "expected_score_range": (0.6, 0.9)
        }
    ]
    
    print("Testing Enhanced Linguistic Scoring Components")
    print("=" * 60)
    
    results = []
    
    for sample in test_samples:
        # Create a simple candidate structure for testing
        candidates = [{
            'asr_data': {
                'text': sample['text'],
                'words': [{'word': word, 'confidence': 0.8} for word in sample['text'].split()]
            },
            'aligned_segments': [
                {
                    'text': sample['text'],
                    'start': 0.0,
                    'end': 10.0,
                    'speaker_id': 'speaker_1',
                    'words': [{'word': word, 'start': i, 'end': i+0.5} for i, word in enumerate(sample['text'].split())]
                }
            ]
        }]
        
        # Calculate linguistic scores
        linguistic_scores = scorer._calculate_linguistic_scores(candidates)
        score = linguistic_scores[0]
        
        # Test individual components
        language_model_score = scorer._calculate_language_model_score(sample['text'])
        punctuation_score = scorer._calculate_punctuation_score(sample['text'])
        disfluency_score = scorer._calculate_disfluency_score(sample['text'])
        
        # Store results
        result = {
            'name': sample['name'],
            'text_preview': sample['text'][:50] + "..." if len(sample['text']) > 50 else sample['text'],
            'overall_score': score,
            'language_model_score': language_model_score,
            'punctuation_score': punctuation_score,
            'disfluency_score': disfluency_score,
            'expected_range': sample['expected_score_range'],
            'in_expected_range': sample['expected_score_range'][0] <= score <= sample['expected_score_range'][1]
        }
        results.append(result)
        
        print(f"\nTest: {sample['name']}")
        print(f"Text: {result['text_preview']}")
        print(f"Overall L Score: {score:.3f}")
        print(f"  - Language Model: {language_model_score:.3f}")
        print(f"  - Punctuation: {punctuation_score:.3f}")
        print(f"  - Disfluency: {disfluency_score:.3f}")
        print(f"Expected Range: {sample['expected_score_range']}")
        print(f"In Range: {'✓' if result['in_expected_range'] else '✗'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['in_expected_range'])
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    # Detailed component analysis
    print("\nComponent Score Analysis:")
    print("-" * 30)
    
    for result in results:
        print(f"{result['name']:<20} | L:{result['language_model_score']:.2f} P:{result['punctuation_score']:.2f} D:{result['disfluency_score']:.2f} | Overall:{result['overall_score']:.2f}")
    
    # Test specific enhancements
    print("\n" + "=" * 60)
    print("SPECIFIC ENHANCEMENT TESTS")
    print("=" * 60)
    
    # Test vocabulary sophistication
    simple_vocab = "I go to the store. I buy food. I come home. I eat food."
    complex_vocab = "I ventured to the marketplace, procured sustenance, returned to my residence, and consumed the provisions."
    
    simple_lm_score = scorer._calculate_language_model_score(simple_vocab)
    complex_lm_score = scorer._calculate_language_model_score(complex_vocab)
    
    print(f"Vocabulary Sophistication Test:")
    print(f"  Simple: {simple_lm_score:.3f}")
    print(f"  Complex: {complex_lm_score:.3f}")
    print(f"  Enhancement Working: {'✓' if complex_lm_score > simple_lm_score else '✗'}")
    
    # Test punctuation detection
    good_punct = "Hello, Dr. Smith! How are you? I'm fine, thanks."
    bad_punct = "hello dr smith how are you im fine thanks"
    
    good_punct_score = scorer._calculate_punctuation_score(good_punct)
    bad_punct_score = scorer._calculate_punctuation_score(bad_punct)
    
    print(f"\nPunctuation Detection Test:")
    print(f"  Good Punctuation: {good_punct_score:.3f}")
    print(f"  Poor Punctuation: {bad_punct_score:.3f}")
    print(f"  Enhancement Working: {'✓' if good_punct_score > bad_punct_score else '✗'}")
    
    # Test disfluency detection
    fluent = "I want to discuss the project timeline and deliverables."
    disfluent = "Um, I, uh, I want to, like, discuss the, um, the project timeline and, you know, deliverables."
    
    fluent_disf_score = scorer._calculate_disfluency_score(fluent)
    disfluent_disf_score = scorer._calculate_disfluency_score(disfluent)
    
    print(f"\nDisfluency Detection Test:")
    print(f"  Fluent Speech: {fluent_disf_score:.3f}")
    print(f"  Disfluent Speech: {disfluent_disf_score:.3f}")
    print(f"  Enhancement Working: {'✓' if fluent_disf_score > disfluent_disf_score else '✗'}")
    
    # Final assessment
    enhancements_working = (
        complex_lm_score > simple_lm_score and
        good_punct_score > bad_punct_score and
        fluent_disf_score > disfluent_disf_score
    )
    
    print(f"\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    print(f"All Enhancement Tests Passed: {'✓ SUCCESS' if enhancements_working else '✗ NEEDS WORK'}")
    print(f"Overall Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if enhancements_working and passed_tests >= total_tests * 0.75:
        print("🎉 Enhanced linguistic scoring is working well!")
        return True
    else:
        print("⚠️  Some issues detected with enhanced scoring.")
        return False

if __name__ == "__main__":
    success = test_enhanced_linguistic_scoring()
    sys.exit(0 if success else 1)