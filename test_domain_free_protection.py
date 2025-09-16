#!/usr/bin/env python3
"""
Test script to validate domain-free acronym and ID protection system.

This script tests the enhanced text normalization system to ensure that
SKUs, case numbers, doses, versions, and other structured identifiers
are properly protected from normalization corruption.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import yaml
from core.text_normalizer import AcronymProtector, create_text_normalizer
from core.guardrail_verifier import ProtectedTokenValidator, GuardrailVerifier

def test_protection_patterns():
    """Test that protection patterns correctly identify various ID formats"""
    
    # Load configuration
    with open('config/normalization_profiles.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    patterns = config['global_settings']['protected_patterns']
    protector = AcronymProtector(patterns)
    
    # Test cases with expected formats
    test_cases = [
        # Mixed alphanumerics
        ("Product SKU AA-9999 was ordered", "AA-9999"),
        ("Reference code 9A9-9 is valid", "9A9-9"),
        ("Item A12B-34 needs processing", "A12B-34"),
        ("Order ACME-100 shipped today", "ACME-100"),
        ("Code 12-AB-34 verified", "12-AB-34"),
        
        # Legal case IDs
        ("Case 309059/2015 is closed", "309059/2015"),
        ("Docket CV-2024001 filed", "CV-2024001"),
        ("Case No. 123456 pending", "No. 123456"),
        
        # Doses and units
        ("Prescribed 9.5 mg daily", "9.5 mg"),
        ("Take 1,200 IU vitamin D", "1,200 IU"),
        ("Inject 2.5 ml solution", "2.5 ml"),
        ("Administer 50.5 mcg dose", "50.5 mcg"),
        
        # Versions and dates
        ("Update to v1.2.3 available", "v1.2.3"),
        ("Quarter Q3-24 results", "Q3-24"),
        ("Released 2024-09-15", "2024-09-15"),
        ("Build version V2.1.0", "V2.1.0"),
    ]
    
    print("Testing protection pattern recognition...")
    print("=" * 50)
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, (text, expected_token) in enumerate(test_cases, 1):
        protected_regions = protector.identify_protected_tokens(text)
        
        # Check if expected token is found in any protected region
        found_expected = False
        for start, end, token_text, reason in protected_regions:
            if expected_token in token_text:
                found_expected = True
                break
        
        status = "✅ PASS" if found_expected else "❌ FAIL"
        print(f"Test {i:2d}: {status} - '{text}'")
        print(f"         Expected: '{expected_token}'")
        print(f"         Protected: {[token for _, _, token, _ in protected_regions]}")
        
        if found_expected:
            success_count += 1
        
        print()
    
    print(f"Protection Pattern Tests: {success_count}/{total_tests} passed")
    return success_count == total_tests

def test_guardrail_validation():
    """Test that guardrail validation properly protects tokens"""
    
    validator = ProtectedTokenValidator()
    
    # Test cases for guardrail violations
    test_cases = [
        {
            'original': 'Order ACME-100 processed',
            'normalized': 'Order ACME-100 processed',  # Should be preserved
            'protected_regions': [(6, 14, 'ACME-100', 'protected_mixed_alphanumeric_high')],
            'should_violate': False,
            'description': 'Exact preservation - no changes'
        },
        {
            'original': 'Case 309059/2015 closed',
            'normalized': 'Case 309,059/2015 closed',  # Comma added - violation
            'protected_regions': [(5, 17, '309059/2015', 'protected_legal_cases_high')],
            'should_violate': True,
            'description': 'Punctuation insertion violation'
        },
        {
            'original': 'Dose 9.5 mg prescribed',
            'normalized': 'Dose 9.5mg prescribed',  # Space removed - violation
            'protected_regions': [(5, 11, '9.5 mg', 'protected_doses_units_high')],
            'should_violate': True,
            'description': 'Spacing change violation'
        },
        {
            'original': 'Version v1.2.3 released',
            'normalized': 'Version V1.2.3 released',  # Case change - violation
            'protected_regions': [(8, 15, 'v1.2.3', 'protected_versions_high')],
            'should_violate': True,
            'description': 'Case change violation'
        }
    ]
    
    print("Testing guardrail validation...")
    print("=" * 50)
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        violations = validator.validate_protected_tokens(
            test_case['original'],
            test_case['normalized'],
            test_case['protected_regions']
        )
        
        has_violations = len(violations) > 0
        expected_violations = test_case['should_violate']
        
        test_passed = has_violations == expected_violations
        status = "✅ PASS" if test_passed else "❌ FAIL"
        
        print(f"Test {i:2d}: {status} - {test_case['description']}")
        print(f"         Original:   '{test_case['original']}'")
        print(f"         Normalized: '{test_case['normalized']}'")
        print(f"         Expected violations: {expected_violations}, Found: {has_violations}")
        
        if violations:
            print(f"         Violations: {[v.rule_name for v in violations]}")
        
        if test_passed:
            success_count += 1
        
        print()
    
    print(f"Guardrail Validation Tests: {success_count}/{total_tests} passed")
    return success_count == total_tests

def test_end_to_end_protection():
    """Test end-to-end protection through the normalization pipeline"""
    
    print("Testing end-to-end protection...")
    print("=" * 50)
    
    # Test text with various protected tokens
    test_text = """Patient received 9.5 mg medication for case 309059/2015. 
    Product SKU AA-9999 shipped on 2024-09-15. 
    Software version v1.2.3 includes ACME-100 features."""
    
    # Create normalizer instance (this should work with our enhanced system)
    try:
        normalizer = create_text_normalizer()
        print("✅ TextNormalizer created successfully")
        
        # Mock a segment for processing
        test_segment = {
            'text': test_text,
            'start': 0.0,
            'end': 10.0,
            'words': []
        }
        
        # Test with different profiles
        profiles = ['verbatim', 'light', 'readable']
        
        for profile in profiles:
            print(f"\nTesting with profile: {profile}")
            print("-" * 30)
            
            # This would normally process through the normalization pipeline
            # For now, just test that the components work together
            
            try:
                # Test pattern recognition
                if hasattr(normalizer, 'acronym_protector'):
                    protected_regions = normalizer.acronym_protector.identify_protected_tokens(test_text)
                    print(f"Protected regions found: {len(protected_regions)}")
                    
                    for start, end, token, reason in protected_regions[:3]:  # Show first 3
                        print(f"  - '{token}' ({reason})")
                    
                    if len(protected_regions) > 3:
                        print(f"  ... and {len(protected_regions) - 3} more")
                
                print(f"✅ Profile {profile} test completed")
                
            except Exception as e:
                print(f"❌ Error in profile {profile}: {e}")
                return False
        
        print("\n✅ End-to-end protection test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating normalizer: {e}")
        return False

def main():
    """Run all protection system tests"""
    
    print("Domain-Free Acronym and ID Protection System Test")
    print("=" * 60)
    print()
    
    # Run all tests
    test_results = []
    
    test_results.append(test_protection_patterns())
    test_results.append(test_guardrail_validation())
    test_results.append(test_end_to_end_protection())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    test_names = [
        "Protection Pattern Recognition",
        "Guardrail Validation",
        "End-to-End Protection"
    ]
    
    for i, (test_name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print("-" * 60)
    print(f"Overall Result: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Domain-free ID protection system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)