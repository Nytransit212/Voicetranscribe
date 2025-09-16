#!/usr/bin/env python3
"""
Test audio format validation functionality
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_audio_validation():
    """Test audio format validation on test files"""
    try:
        from utils.audio_format_validator import get_audio_validator
        
        validator = get_audio_validator()
        test_files = ['data/test_short_video.mov', 'data/test_video.mp4']
        
        print("🔍 Testing audio format validation...")
        results = validator.validate_test_audio_files(test_files)
        
        print("\n📊 Validation Results:")
        for file_path, result in results.items():
            print(f"\n📁 {Path(file_path).name}:")
            print(f"   ✅ Valid: {result.valid}")
            print(f"   🔄 Conversion needed: {result.conversion_needed}")
            if result.error_message:
                print(f"   ❌ Error: {result.error_message}")
            if result.normalized_path:
                print(f"   📄 Normalized path: {result.normalized_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Audio validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_capability_manager():
    """Test capability manager functionality"""
    try:
        from utils.capability_manager import check_system_capabilities
        
        print("\n🔍 Testing capability manager...")
        capability_report = check_system_capabilities()
        
        print(f"\n📊 System Status: {capability_report.system_status.value}")
        print(f"✅ Available features: {len(capability_report.available_features)}")
        print(f"⚠️ Degraded features: {len(capability_report.degraded_features)}")
        print(f"🔄 Fallback features: {len(capability_report.fallback_features)}")
        print(f"❌ Unavailable features: {len(capability_report.unavailable_features)}")
        
        if capability_report.critical_missing:
            print(f"🚨 Critical missing: {capability_report.critical_missing}")
        
        return capability_report
        
    except Exception as e:
        print(f"❌ Capability manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 TESTING PRODUCTION READINESS FIXES")
    
    # Test capability manager
    capability_report = test_capability_manager()
    
    # Test audio validation
    validation_results = test_audio_validation()
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    if capability_report:
        print(f"✅ Capability manager working: {capability_report.system_status.value}")
    else:
        print("❌ Capability manager failed")
    
    if validation_results:
        valid_files = sum(1 for r in validation_results.values() if r.valid)
        total_files = len(validation_results)
        print(f"✅ Audio validation working: {valid_files}/{total_files} files valid")
    else:
        print("❌ Audio validation failed")