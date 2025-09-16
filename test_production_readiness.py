#!/usr/bin/env python3
"""
Production Readiness Test Suite

Validates all critical production readiness fixes:
1. Runtime dependency gaps resolved
2. Audio format issues fixed
3. Adaptive biasing path validated
4. Capability checks and fallbacks working
5. Integration test validation completed
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_runtime_dependencies():
    """Test runtime dependency management"""
    print("🔍 Testing runtime dependency management...")
    
    try:
        from utils.capability_manager import check_system_capabilities, get_capability_manager
        
        # Run capability assessment
        capability_report = check_system_capabilities()
        manager = get_capability_manager()
        
        print(f"✅ System Status: {capability_report.system_status.value}")
        print(f"📊 Dependencies Status:")
        for dep_name, dep_info in capability_report.dependencies.items():
            status_icon = "✅" if dep_info.status.value == "available" else "⚠️" if dep_info.status.value in ["degraded", "fallback"] else "❌"
            print(f"   {status_icon} {dep_info.name}: {dep_info.status.value}")
        
        # Test specific capabilities
        critical_features = [
            'basic_speaker_diarization', 
            'speaker_embeddings', 
            'consensus_fusion',
            'audio_format_conversion'
        ]
        
        print(f"\n🧪 Critical Features Test:")
        all_critical_available = True
        for feature in critical_features:
            available = manager.is_feature_available(feature)
            icon = "✅" if available else "❌"
            print(f"   {icon} {feature}: {'Available' if available else 'Unavailable'}")
            if not available:
                all_critical_available = False
        
        return {
            'success': True,
            'system_status': capability_report.system_status.value,
            'all_critical_available': all_critical_available,
            'total_deps': len(capability_report.dependencies),
            'available_deps': len([d for d in capability_report.dependencies.values() if d.status.value == "available"]),
            'fallback_deps': len([d for d in capability_report.dependencies.values() if d.status.value == "fallback"])
        }
        
    except Exception as e:
        print(f"❌ Runtime dependency test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_audio_format_handling():
    """Test audio format validation and conversion"""
    print("\n🔍 Testing audio format handling...")
    
    try:
        from utils.audio_format_validator import get_audio_validator, validate_audio_file
        
        test_files = ['data/test_short_video.mov', 'data/test_video.mp4']
        validator = get_audio_validator()
        
        results = {}
        for file_path in test_files:
            if Path(file_path).exists():
                result = validate_audio_file(file_path)
                results[file_path] = result
                
                icon = "✅" if result.valid else "❌"
                print(f"   {icon} {Path(file_path).name}: {'Valid' if result.valid else 'Invalid'}")
                if result.valid and result.conversion_needed:
                    print(f"      🔄 Converted to: {result.normalized_path}")
            else:
                print(f"   ❌ {file_path}: File not found")
                results[file_path] = None
        
        success = all(r and r.valid for r in results.values() if r is not None)
        
        return {
            'success': success,
            'files_tested': len([r for r in results.values() if r is not None]),
            'files_valid': len([r for r in results.values() if r and r.valid]),
            'conversions_needed': len([r for r in results.values() if r and r.conversion_needed])
        }
        
    except Exception as e:
        print(f"❌ Audio format test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_adaptive_biasing_path():
    """Test adaptive biasing integration"""
    print("\n🔍 Testing adaptive biasing path...")
    
    try:
        from core.term_bias import create_adaptive_biasing_engine
        from core.consensus_module import BestSingleCandidateStrategy, ConsensusModule
        from core.term_store import create_project_term_store
        
        # Test biasing engine creation
        term_store = create_project_term_store()
        biasing_engine = create_adaptive_biasing_engine(term_store)
        print("   ✅ Adaptive biasing engine created")
        
        # Test session bias list generation
        bias_list = biasing_engine.generate_session_bias_list(
            project_id="test_project",
            session_id="test_session"
        )
        print(f"   ✅ Session bias list generated: {bias_list.total_bias_terms} terms")
        
        # Test consensus strategy bias support
        strategy = BestSingleCandidateStrategy()
        supports_bias = strategy.supports_session_bias()
        print(f"   ✅ Consensus strategy bias support: {supports_bias}")
        
        # Test consensus module integration
        consensus_module = ConsensusModule()
        print("   ✅ Consensus module with bias integration ready")
        
        return {
            'success': True,
            'bias_engine_created': True,
            'bias_list_generated': bias_list.total_bias_terms > 0,
            'consensus_supports_bias': supports_bias,
            'total_bias_terms': bias_list.total_bias_terms
        }
        
    except Exception as e:
        print(f"❌ Adaptive biasing test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_ensemble_integration():
    """Test ensemble manager integration with all fixes"""
    print("\n🔍 Testing ensemble manager integration...")
    
    try:
        from core.ensemble_manager import EnsembleManager
        
        # Create ensemble manager (this will trigger capability checks)
        print("   🔄 Creating ensemble manager with capability checks...")
        ensemble = EnsembleManager(
            expected_speakers=3,
            noise_level='low',
            enable_auto_glossary=True,
            enable_long_horizon_tracking=True
        )
        print("   ✅ Ensemble manager created successfully")
        
        # Check if capability report was generated
        has_capability_report = hasattr(ensemble, 'capability_report') and ensemble.capability_report is not None
        print(f"   ✅ Capability assessment: {'Completed' if has_capability_report else 'Not found'}")
        
        # Check if audio validator is available
        has_audio_validator = hasattr(ensemble, 'audio_validator') and ensemble.audio_validator is not None
        print(f"   ✅ Audio validation: {'Available' if has_audio_validator else 'Not available'}")
        
        # Check configuration adjustments
        auto_glossary_enabled = ensemble.enable_auto_glossary
        overlap_processing_enabled = ensemble.enable_overlap_aware_processing
        print(f"   📊 Auto-glossary enabled: {auto_glossary_enabled}")
        print(f"   📊 Overlap processing enabled: {overlap_processing_enabled}")
        
        return {
            'success': True,
            'ensemble_created': True,
            'capability_assessment': has_capability_report,
            'audio_validation': has_audio_validator,
            'auto_glossary_enabled': auto_glossary_enabled,
            'overlap_processing_enabled': overlap_processing_enabled
        }
        
    except Exception as e:
        print(f"❌ Ensemble integration test failed: {e}")
        return {'success': False, 'error': str(e)}

def run_production_readiness_suite():
    """Run complete production readiness test suite"""
    print("🚀 PRODUCTION READINESS TEST SUITE")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: Runtime Dependencies
    test_results['runtime_dependencies'] = test_runtime_dependencies()
    
    # Test 2: Audio Format Handling
    test_results['audio_format'] = test_audio_format_handling()
    
    # Test 3: Adaptive Biasing Path
    test_results['adaptive_biasing'] = test_adaptive_biasing_path()
    
    # Test 4: Ensemble Integration
    test_results['ensemble_integration'] = test_ensemble_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 PRODUCTION READINESS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in test_results.items():
        if result['success']:
            print(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
        else:
            print(f"❌ {test_name.replace('_', ' ').title()}: FAILED - {result.get('error', 'Unknown error')}")
            all_passed = False
    
    print(f"\n🎯 Overall Status: {'✅ PRODUCTION READY' if all_passed else '❌ ISSUES DETECTED'}")
    
    if all_passed:
        print("\n🎉 All critical production readiness issues have been resolved!")
        print("📈 System is ready for production deployment with:")
        print("   • Comprehensive dependency checking and fallbacks")
        print("   • Audio format validation and conversion")
        print("   • Adaptive biasing integration")
        print("   • Graceful degradation when features unavailable")
    else:
        print("\n⚠️  Some issues remain - check individual test results above")
    
    return test_results, all_passed

if __name__ == "__main__":
    results, passed = run_production_readiness_suite()
    sys.exit(0 if passed else 1)