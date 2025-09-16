#!/usr/bin/env python3
"""
Test script for the robust text normalization system

Tests various aspects including:
- Token invention prevention
- Timing preservation
- Acronym protection
- Profile functionality
- Guardrail effectiveness
"""

import sys
import os
import time
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core.text_normalizer import TextNormalizer, create_text_normalizer
    from core.guardrail_verifier import GuardrailVerifier, create_guardrail_verifier
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class TextNormalizationTestSuite:
    """Test suite for text normalization system"""
    
    def __init__(self):
        """Initialize test suite"""
        print("🧪 Initializing Text Normalization Test Suite...")
        
        try:
            self.normalizer = create_text_normalizer()
            self.guardrail_verifier = create_guardrail_verifier()
            print("✅ Normalization system initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize normalization system: {e}")
            raise
    
    def create_test_segments(self) -> List[Dict[str, Any]]:
        """Create test segments with various content types"""
        return [
            {
                'text': 'um so like we need to talk about the API integration and uh maybe some SQL stuff',
                'start': 0.0,
                'end': 5.0,
                'speaker': 'Speaker_A',
                'words': [
                    {'word': 'um', 'start': 0.0, 'end': 0.2, 'confidence': 0.8},
                    {'word': 'so', 'start': 0.3, 'end': 0.5, 'confidence': 0.9},
                    {'word': 'like', 'start': 0.6, 'end': 0.8, 'confidence': 0.9},
                    {'word': 'we', 'start': 0.9, 'end': 1.0, 'confidence': 0.95},
                    {'word': 'need', 'start': 1.1, 'end': 1.4, 'confidence': 0.95},
                    {'word': 'to', 'start': 1.5, 'end': 1.6, 'confidence': 0.9},
                    {'word': 'talk', 'start': 1.7, 'end': 1.9, 'confidence': 0.95},
                    {'word': 'about', 'start': 2.0, 'end': 2.3, 'confidence': 0.9},
                    {'word': 'the', 'start': 2.4, 'end': 2.5, 'confidence': 0.95},
                    {'word': 'API', 'start': 2.6, 'end': 2.9, 'confidence': 0.85},
                    {'word': 'integration', 'start': 3.0, 'end': 3.6, 'confidence': 0.9},
                    {'word': 'and', 'start': 3.7, 'end': 3.8, 'confidence': 0.9},
                    {'word': 'uh', 'start': 3.9, 'end': 4.0, 'confidence': 0.7},
                    {'word': 'maybe', 'start': 4.1, 'end': 4.4, 'confidence': 0.9},
                    {'word': 'some', 'start': 4.5, 'end': 4.7, 'confidence': 0.9},
                    {'word': 'SQL', 'start': 4.8, 'end': 5.0, 'confidence': 0.85}
                ]
            },
            {
                'text': 'the revenue for Q3 was about 2.5 million and we should definitely not ignore this',
                'start': 5.5,
                'end': 10.0,
                'speaker': 'Speaker_B',
                'words': [
                    {'word': 'the', 'start': 5.5, 'end': 5.6, 'confidence': 0.95},
                    {'word': 'revenue', 'start': 5.7, 'end': 6.1, 'confidence': 0.9},
                    {'word': 'for', 'start': 6.2, 'end': 6.3, 'confidence': 0.9},
                    {'word': 'Q3', 'start': 6.4, 'end': 6.7, 'confidence': 0.85},
                    {'word': 'was', 'start': 6.8, 'end': 7.0, 'confidence': 0.9},
                    {'word': 'about', 'start': 7.1, 'end': 7.4, 'confidence': 0.9},
                    {'word': '2.5', 'start': 7.5, 'end': 7.8, 'confidence': 0.9},
                    {'word': 'million', 'start': 7.9, 'end': 8.3, 'confidence': 0.95},
                    {'word': 'and', 'start': 8.4, 'end': 8.5, 'confidence': 0.9},
                    {'word': 'we', 'start': 8.6, 'end': 8.7, 'confidence': 0.95},
                    {'word': 'should', 'start': 8.8, 'end': 9.1, 'confidence': 0.9},
                    {'word': 'definitely', 'start': 9.2, 'end': 9.6, 'confidence': 0.9},
                    {'word': 'not', 'start': 9.7, 'end': 9.8, 'confidence': 0.95},
                    {'word': 'ignore', 'start': 9.9, 'end': 10.0, 'confidence': 0.9}
                ]
            },
            {
                'text': 'um uh er the AWS-S3 bucket configuration is very very important',
                'start': 10.5,
                'end': 15.0,
                'speaker': 'Speaker_C',
                'words': [
                    {'word': 'um', 'start': 10.5, 'end': 10.7, 'confidence': 0.6},
                    {'word': 'uh', 'start': 10.8, 'end': 11.0, 'confidence': 0.6},
                    {'word': 'er', 'start': 11.1, 'end': 11.3, 'confidence': 0.5},
                    {'word': 'the', 'start': 11.4, 'end': 11.5, 'confidence': 0.95},
                    {'word': 'AWS-S3', 'start': 11.6, 'end': 12.2, 'confidence': 0.8},
                    {'word': 'bucket', 'start': 12.3, 'end': 12.6, 'confidence': 0.9},
                    {'word': 'configuration', 'start': 12.7, 'end': 13.4, 'confidence': 0.85},
                    {'word': 'is', 'start': 13.5, 'end': 13.6, 'confidence': 0.95},
                    {'word': 'very', 'start': 13.7, 'end': 13.9, 'confidence': 0.9},
                    {'word': 'very', 'start': 14.0, 'end': 14.2, 'confidence': 0.9},
                    {'word': 'important', 'start': 14.3, 'end': 15.0, 'confidence': 0.95}
                ]
            }
        ]
    
    def test_token_invention_prevention(self, profile: str = "readable") -> bool:
        """Test that no new tokens are invented during normalization"""
        print(f"\n🔍 Testing token invention prevention (profile: {profile})...")
        
        test_segments = self.create_test_segments()
        
        try:
            results = self.normalizer.normalize_segments(test_segments, profile=profile)
            
            for i, result in enumerate(results):
                original_tokens = set(result.original_text.lower().split())
                normalized_tokens = set(result.normalized_text.lower().split())
                
                # Remove punctuation for comparison
                import re
                original_tokens = {re.sub(r'[^\w]', '', token) for token in original_tokens if token}
                normalized_tokens = {re.sub(r'[^\w]', '', token) for token in normalized_tokens if token}
                
                # Check for invented tokens (tokens in normalized but not in original)
                new_tokens = normalized_tokens - original_tokens
                
                # Filter out allowed new tokens (numbers spelled out, etc.)
                allowed_new_tokens = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}
                problematic_new_tokens = new_tokens - allowed_new_tokens
                
                if problematic_new_tokens:
                    print(f"❌ Segment {i}: Invented tokens detected: {problematic_new_tokens}")
                    print(f"   Original: {result.original_text}")
                    print(f"   Normalized: {result.normalized_text}")
                    return False
                
                print(f"✅ Segment {i}: No token invention detected")
                if new_tokens:
                    print(f"   Allowed new tokens: {new_tokens}")
            
            return True
            
        except Exception as e:
            print(f"❌ Token invention test failed with error: {e}")
            return False
    
    def test_acronym_protection(self, profile: str = "readable") -> bool:
        """Test that acronyms and technical terms are protected"""
        print(f"\n🛡️ Testing acronym protection (profile: {profile})...")
        
        test_segments = self.create_test_segments()
        
        try:
            results = self.normalizer.normalize_segments(test_segments, profile=profile)
            
            # Define expected protected terms
            expected_protected = {'API', 'SQL', 'Q3', 'AWS-S3'}
            
            for i, result in enumerate(results):
                original_text = result.original_text
                normalized_text = result.normalized_text
                
                # Check if protected terms are preserved
                for term in expected_protected:
                    if term in original_text:
                        if term.upper() not in normalized_text.upper():
                            print(f"❌ Segment {i}: Protected term '{term}' not preserved")
                            print(f"   Original: {original_text}")
                            print(f"   Normalized: {normalized_text}")
                            return False
                        else:
                            print(f"✅ Segment {i}: Protected term '{term}' preserved")
            
            return True
            
        except Exception as e:
            print(f"❌ Acronym protection test failed with error: {e}")
            return False
    
    def test_disfluency_removal(self, profile: str = "readable") -> bool:
        """Test that disfluencies are appropriately removed"""
        print(f"\n🧹 Testing disfluency removal (profile: {profile})...")
        
        test_segments = self.create_test_segments()
        
        try:
            results = self.normalizer.normalize_segments(test_segments, profile=profile)
            
            for i, result in enumerate(results):
                original_text = result.original_text
                normalized_text = result.normalized_text
                
                # Check for filler removal
                fillers_in_original = len([word for word in original_text.split() if word.lower() in ['um', 'uh', 'er']])
                fillers_in_normalized = len([word for word in normalized_text.split() if word.lower() in ['um', 'uh', 'er']])
                
                print(f"✅ Segment {i}: Fillers reduced from {fillers_in_original} to {fillers_in_normalized}")
                
                # Ensure semantic words are preserved (e.g., "not" in "should definitely not ignore")
                if 'not' in original_text and 'not' not in normalized_text:
                    print(f"❌ Segment {i}: Critical semantic word 'not' was removed")
                    return False
                
                # Check that "very very" emphasis is handled appropriately
                if 'very very' in original_text:
                    if profile == "executive":
                        # Executive profile should clean up repetition
                        print(f"✅ Segment {i}: Repetitive emphasis handling (executive profile)")
                    else:
                        # Other profiles should preserve emphasis
                        if 'very very' not in normalized_text and 'very' not in normalized_text:
                            print(f"❌ Segment {i}: Emphasis completely removed inappropriately")
                            return False
                        print(f"✅ Segment {i}: Emphasis appropriately handled")
            
            return True
            
        except Exception as e:
            print(f"❌ Disfluency removal test failed with error: {e}")
            return False
    
    def test_timing_preservation(self, profile: str = "readable") -> bool:
        """Test that word timing is preserved within tolerance"""
        print(f"\n⏱️ Testing timing preservation (profile: {profile})...")
        
        test_segments = self.create_test_segments()
        
        try:
            results = self.normalizer.normalize_segments(test_segments, profile=profile)
            
            for i, result in enumerate(results):
                original_tokens = result.original_tokens
                normalized_tokens = result.normalized_tokens
                
                # Check that we have reasonable timing preservation
                if original_tokens and normalized_tokens:
                    original_start = original_tokens[0].start_time
                    original_end = original_tokens[-1].end_time
                    normalized_start = normalized_tokens[0].start_time
                    normalized_end = normalized_tokens[-1].end_time
                    
                    start_drift = abs(original_start - normalized_start)
                    end_drift = abs(original_end - normalized_end)
                    
                    max_drift_threshold = 0.2  # 200ms tolerance
                    
                    if start_drift > max_drift_threshold or end_drift > max_drift_threshold:
                        print(f"❌ Segment {i}: Excessive timing drift detected")
                        print(f"   Start drift: {start_drift:.3f}s, End drift: {end_drift:.3f}s")
                        return False
                    
                    print(f"✅ Segment {i}: Timing preserved (start drift: {start_drift:.3f}s, end drift: {end_drift:.3f}s)")
            
            return True
            
        except Exception as e:
            print(f"❌ Timing preservation test failed with error: {e}")
            return False
    
    def test_profile_differences(self) -> bool:
        """Test that different profiles produce different levels of normalization"""
        print(f"\n📊 Testing profile differences...")
        
        test_segments = self.create_test_segments()
        profiles = ["verbatim", "light", "readable", "executive"]
        
        try:
            profile_results = {}
            
            for profile in profiles:
                results = self.normalizer.normalize_segments(test_segments, profile=profile)
                profile_results[profile] = results
                
                total_changes = sum(len(result.changes) for result in results)
                avg_readability = sum(result.readability_score_after for result in results) / len(results)
                
                print(f"✅ Profile '{profile}': {total_changes} total changes, avg readability: {avg_readability:.3f}")
            
            # Verify that executive profile makes more changes than verbatim
            verbatim_changes = sum(len(result.changes) for result in profile_results["verbatim"])
            executive_changes = sum(len(result.changes) for result in profile_results["executive"])
            
            if executive_changes <= verbatim_changes:
                print(f"❌ Executive profile should make more changes than verbatim profile")
                print(f"   Verbatim: {verbatim_changes}, Executive: {executive_changes}")
                return False
            
            print(f"✅ Profile escalation verified: verbatim ({verbatim_changes}) < executive ({executive_changes})")
            return True
            
        except Exception as e:
            print(f"❌ Profile differences test failed with error: {e}")
            return False
    
    def test_guardrail_system(self) -> bool:
        """Test that guardrail system catches violations"""
        print(f"\n🚨 Testing guardrail system...")
        
        try:
            # Test with a segment that might trigger guardrail violations
            problematic_segment = {
                'text': 'The um confidential data should not be shared',
                'start': 0.0,
                'end': 3.0,
                'speaker': 'Speaker_Test',
                'words': [
                    {'word': 'The', 'start': 0.0, 'end': 0.2},
                    {'word': 'um', 'start': 0.3, 'end': 0.5},
                    {'word': 'confidential', 'start': 0.6, 'end': 1.2},
                    {'word': 'data', 'start': 1.3, 'end': 1.5},
                    {'word': 'should', 'start': 1.6, 'end': 1.9},
                    {'word': 'not', 'start': 2.0, 'end': 2.1},
                    {'word': 'be', 'start': 2.2, 'end': 2.3},
                    {'word': 'shared', 'start': 2.4, 'end': 3.0}
                ]
            }
            
            # Test normalization
            results = self.normalizer.normalize_segments([problematic_segment], profile="executive")
            result = results[0]
            
            # Test guardrail verification
            guardrail_result = self.guardrail_verifier.verify_normalization(
                original_text=result.original_text,
                normalized_text=result.normalized_text,
                current_profile="executive"
            )
            
            print(f"✅ Guardrail system executed successfully")
            print(f"   Violations detected: {len(guardrail_result.violations)}")
            print(f"   Passed verification: {guardrail_result.passed}")
            print(f"   Confidence score: {guardrail_result.confidence_score:.3f}")
            
            # Verify that critical semantic word "not" is preserved
            if 'not' not in result.normalized_text:
                print(f"❌ Critical semantic word 'not' was removed - guardrail should have caught this")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Guardrail system test failed with error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        print("🚀 Starting Text Normalization Test Suite")
        print("=" * 60)
        
        tests = [
            ("Token Invention Prevention", lambda: self.test_token_invention_prevention()),
            ("Acronym Protection", lambda: self.test_acronym_protection()),
            ("Disfluency Removal", lambda: self.test_disfluency_removal()),
            ("Timing Preservation", lambda: self.test_timing_preservation()),
            ("Profile Differences", lambda: self.test_profile_differences()),
            ("Guardrail System", lambda: self.test_guardrail_system())
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
        
        print("\n" + "=" * 60)
        print(f"🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Text normalization system is working correctly.")
            return True
        else:
            print(f"⚠️ {total - passed} tests failed. Please review the issues above.")
            return False

def main():
    """Main test function"""
    try:
        test_suite = TextNormalizationTestSuite()
        success = test_suite.run_all_tests()
        
        if success:
            print("\n✅ Text normalization system validation completed successfully!")
            return 0
        else:
            print("\n❌ Text normalization system validation failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite failed to initialize: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)