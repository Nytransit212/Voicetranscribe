#!/usr/bin/env python3
"""
Comprehensive test suite for cache correctness system.

This test verifies that the cache correctness implementation works as expected,
including cache key generation, validation, atomic operations, and integration
with the existing IntelligentCacheManager.
"""

import os
import sys
import tempfile
import time
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cache_key import CacheKey, CacheScope, generate_cache_key_for_stage, validate_cached_data
from utils.cache_operations import AtomicCacheOperations, get_cache_operations
from utils.intelligent_cache import IntelligentCacheManager
from core.run_context import RunContext
from utils.enhanced_structured_logger import create_enhanced_logger


def test_cache_key_generation():
    """Test comprehensive cache key generation"""
    print("Testing cache key generation...")
    
    # Create test data
    test_cache_key = CacheKey(
        media_sha256="test_media_hash_12345",
        stage_name="asr",
        component_name="openai_whisper",
        config_snapshot_id="config_hash_67890",
        model_version_fingerprint="model_hash_abcde",
        sample_rate=16000,
        language_hint="en",
        quality_profile="high",
        project_id="test_project",
        session_id="test_session",
        run_id="test_run"
    )
    
    # Generate cache key
    cache_key = test_cache_key.generate_cache_key()
    
    print(f"Generated cache key: {cache_key}")
    assert len(cache_key) == 64, f"Cache key should be 64 characters, got {len(cache_key)}"
    
    # Test deterministic generation
    cache_key2 = test_cache_key.generate_cache_key()
    assert cache_key == cache_key2, "Cache key generation should be deterministic"
    
    # Test cache key changes with different parameters
    modified_key = CacheKey(
        media_sha256="different_media_hash",
        stage_name="asr",
        component_name="openai_whisper",
        config_snapshot_id="config_hash_67890",
        model_version_fingerprint="model_hash_abcde",
        sample_rate=16000,
        language_hint="en",
        quality_profile="high",
        project_id="test_project",
        session_id="test_session",
        run_id="test_run"
    )
    
    modified_cache_key = modified_key.generate_cache_key()
    assert cache_key != modified_cache_key, "Cache key should change when media hash changes"
    
    print("✅ Cache key generation test passed")


def test_cache_validation():
    """Test cache validation against RunContext"""
    print("Testing cache validation...")
    
    # Create test RunContext
    config_dict = {
        "asr": {"provider": "openai", "model": "whisper-1"},
        "quality_profile": "high"
    }
    
    model_versions = {
        "asr": {"openai_whisper": "v1.0"},
        "diarization": {"pyannote": "v2.1"}
    }
    
    run_context = RunContext.create_deterministic_context(
        media_sha256="test_media_hash_12345",
        config_dict=config_dict,
        model_versions=model_versions,
        session_id="test_session",
        project_id="test_project"
    )
    
    # Create matching cache key
    matching_cache_key = CacheKey(
        media_sha256=run_context.media_sha256,
        stage_name="asr",
        component_name="openai_whisper",
        config_snapshot_id=run_context.config_snapshot_id,
        model_version_fingerprint=run_context.model_version_fingerprint,
        project_id=run_context.project_id
    )
    
    # Test validation
    is_valid = validate_cached_data(matching_cache_key, run_context)
    assert is_valid, "Matching cache key should validate successfully"
    
    # Create mismatched cache key (different media)
    mismatched_cache_key = CacheKey(
        media_sha256="different_media_hash",
        stage_name="asr",
        component_name="openai_whisper",
        config_snapshot_id=run_context.config_snapshot_id,
        model_version_fingerprint=run_context.model_version_fingerprint,
        project_id=run_context.project_id
    )
    
    is_valid = validate_cached_data(mismatched_cache_key, run_context)
    assert not is_valid, "Mismatched cache key should fail validation"
    
    print("✅ Cache validation test passed")


def test_atomic_cache_operations():
    """Test atomic cache operations"""
    print("Testing atomic cache operations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create cache operations instance
        cache_ops = AtomicCacheOperations(temp_dir)
        
        # Create test data
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        
        # Create test cache key
        test_cache_key = CacheKey(
            media_sha256="test_media_hash",
            stage_name="test_stage",
            component_name="test_component",
            config_snapshot_id="test_config",
            model_version_fingerprint="test_models",
            scope=CacheScope.PROJECT,
            project_id="test_project"
        )
        
        # Create fake run context for validation
        config_dict = {"test": "config"}
        model_versions = {"test": {"model": "v1"}}
        
        run_context = RunContext.create_deterministic_context(
            media_sha256="test_media_hash",
            config_dict=config_dict,
            model_versions=model_versions
        )
        
        # Update cache key with run context values
        test_cache_key = test_cache_key.update_from_run_context(run_context)
        
        # Test cache set
        success = cache_ops.cache_set(
            test_cache_key,
            test_data,
            creating_component="test_component",
            component_version="1.0.0"
        )
        assert success, "Cache set should succeed"
        
        # Test cache get
        retrieved_data = cache_ops.cache_get(test_cache_key, run_context)
        assert retrieved_data is not None, "Cache get should return data"
        assert retrieved_data == test_data, "Retrieved data should match original"
        
        # Test cache invalidation
        success = cache_ops.cache_invalidate(test_cache_key)
        assert success, "Cache invalidation should succeed"
        
        # Verify data is gone
        retrieved_data = cache_ops.cache_get(test_cache_key, run_context)
        assert retrieved_data is None, "Data should be gone after invalidation"
        
        print("✅ Atomic cache operations test passed")


def test_intelligent_cache_integration():
    """Test integration with IntelligentCacheManager"""
    print("Testing IntelligentCacheManager integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create enhanced cache manager with cache correctness enabled
        cache_manager = IntelligentCacheManager(
            cache_dir=temp_dir,
            use_cache_correctness=True
        )
        
        # Create test RunContext
        config_dict = {"test": "config"}
        model_versions = {"test": {"model": "v1"}}
        
        run_context = RunContext.create_deterministic_context(
            media_sha256="test_media_hash",
            config_dict=config_dict,
            model_versions=model_versions
        )
        
        # Set global run context for the test
        from core.run_context import set_global_run_context
        set_global_run_context(run_context)
        
        # Test data
        test_data = {"result": "test_output", "score": 0.95}
        
        # Test cache miss
        result = cache_manager.get(
            "test_operation",
            component_name="test_component",
            stage_params={"quality": "high"}
        )
        assert result is None, "Should get cache miss initially"
        
        # Test cache set
        success = cache_manager.set(
            "test_operation",
            test_data,
            component_name="test_component",
            stage_params={"quality": "high"}
        )
        assert success, "Cache set should succeed"
        
        # Test cache hit
        result = cache_manager.get(
            "test_operation",
            component_name="test_component",
            stage_params={"quality": "high"}
        )
        assert result is not None, "Should get cache hit"
        assert result == test_data, "Retrieved data should match original"
        
        # Test cache stats
        stats = cache_manager.get_stats()
        assert stats['cache_correctness_enabled'], "Cache correctness should be enabled"
        assert stats['validation']['validated_hits'] > 0, "Should have validated hits"
        
        print("✅ IntelligentCacheManager integration test passed")


def test_cache_audit_functionality():
    """Test cache audit functionality"""
    print("Testing cache audit functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Import here to avoid circular imports during module loading
        from scripts.cache_audit import CacheAuditor
        
        # Create cache operations and add some test data
        cache_ops = AtomicCacheOperations(temp_dir)
        
        # Create test entries
        for i in range(3):
            test_cache_key = CacheKey(
                media_sha256=f"test_media_hash_{i}",
                stage_name="test_stage",
                component_name="test_component",
                config_snapshot_id="test_config",
                model_version_fingerprint="test_models",
                scope=CacheScope.PROJECT,
                project_id="test_project"
            )
            
            test_data = {"test_data": f"value_{i}"}
            
            cache_ops.cache_set(
                test_cache_key,
                test_data,
                creating_component="test_component"
            )
        
        # Create cache auditor
        auditor = CacheAuditor(cache_ops=cache_ops)
        
        # Run integrity scan
        stats = auditor.scan_cache_integrity()
        
        assert stats.valid_entries == 3, f"Should have 3 valid entries, got {stats.valid_entries}"
        assert stats.corrupted_metadata_files == 0, "Should have no corrupted metadata"
        assert stats.orphaned_data_files == 0, "Should have no orphaned data files"
        
        print("✅ Cache audit functionality test passed")


def run_all_tests():
    """Run all cache correctness tests"""
    print("🧪 Running Cache Correctness System Tests")
    print("=" * 50)
    
    try:
        test_cache_key_generation()
        test_cache_validation()
        test_atomic_cache_operations()
        test_intelligent_cache_integration()
        test_cache_audit_functionality()
        
        print("=" * 50)
        print("🎉 All cache correctness tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)