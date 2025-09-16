#!/usr/bin/env python3
"""
Atomic I/O Integration Test Script

This script validates the atomic I/O and temp hygiene system to ensure:
1. Core atomic operations work correctly
2. Integration with FileHandler works
3. Run temp directory management functions properly
4. Collision prevention and cleanup work as expected
5. No breaking changes to existing functionality
"""

import os
import json
import time
import tempfile
import shutil
from pathlib import Path

# Import the atomic I/O system
from utils.atomic_io import (
    get_atomic_io_manager,
    atomic_write,
    TempDirectoryScope,
    create_run_temp_directory,
    get_run_temp_subdir,
    open_temp_for,
    commit_temp,
    safe_remove
)
from utils.file_handler import FileHandler
from core.run_context import RunContext, create_run_context, set_global_run_context

def test_atomic_operations():
    """Test core atomic operations"""
    print("🧪 Testing core atomic operations...")
    
    # Test 1: Basic atomic write
    test_dir = Path("/tmp/atomic_io_test")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test_atomic.json"
    
    test_data = {"test": "atomic write", "timestamp": time.time()}
    
    try:
        with atomic_write(test_file) as f:
            json.dump(test_data, f, indent=2)
        
        # Verify file exists and contains correct data
        assert test_file.exists(), "Atomic write failed - file doesn't exist"
        
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data, "Atomic write failed - data mismatch"
        print("✅ Basic atomic write test passed")
        
    except Exception as e:
        print(f"❌ Basic atomic write test failed: {e}")
        return False
    
    # Test 2: Manual atomic operations
    try:
        test_file2 = test_dir / "test_manual.txt"
        temp_file, temp_path = open_temp_for(test_file2)
        
        test_content = "Manual atomic operation test"
        temp_file.write(test_content)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        
        commit_temp(temp_path, test_file2)
        
        # Verify file exists and contains correct data
        assert test_file2.exists(), "Manual atomic operation failed - file doesn't exist"
        
        with open(test_file2, 'r') as f:
            loaded_content = f.read()
        
        assert loaded_content == test_content, "Manual atomic operation failed - content mismatch"
        print("✅ Manual atomic operations test passed")
        
    except Exception as e:
        print(f"❌ Manual atomic operations test failed: {e}")
        return False
    
    # Test 3: Safe remove
    try:
        success = safe_remove(test_file)
        assert success, "Safe remove failed"
        assert not test_file.exists(), "Safe remove failed - file still exists"
        print("✅ Safe remove test passed")
        
    except Exception as e:
        print(f"❌ Safe remove test failed: {e}")
        return False
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    return True

def test_temp_directory_management():
    """Test per-run temp directory management"""
    print("🧪 Testing temp directory management...")
    
    try:
        # Create a test run context
        test_run_id = f"test_run_{int(time.time())}"
        
        run_context = RunContext(
            run_id=test_run_id,
            session_id="test_session",
            project_id="test_project"
        )
        set_global_run_context(run_context)
        
        # Test 1: Create run temp directory
        temp_dir = create_run_temp_directory(
            run_id=test_run_id,
            stage_name="test_stage",
            project_id="test_project",
            session_id="test_session"
        )
        
        assert temp_dir.exists(), "Run temp directory creation failed"
        assert temp_dir.name == test_run_id, "Run temp directory has wrong name"
        print("✅ Run temp directory creation test passed")
        
        # Test 2: Get temp subdirectories
        for scope in TempDirectoryScope:
            subdir = get_run_temp_subdir(test_run_id, scope)
            assert subdir.exists(), f"Temp subdirectory creation failed for {scope}"
            assert subdir.name == scope.value, f"Temp subdirectory has wrong name for {scope}"
        
        print("✅ Temp subdirectory creation test passed")
        
        # Test 3: Write files to temp subdirectories
        test_data = {"scope_test": True, "timestamp": time.time()}
        
        artifacts_dir = get_run_temp_subdir(test_run_id, TempDirectoryScope.ARTIFACTS)
        test_file = artifacts_dir / "test_artifact.json"
        
        with atomic_write(test_file) as f:
            json.dump(test_data, f, indent=2)
        
        assert test_file.exists(), "Temp subdir file write failed"
        print("✅ Temp subdirectory file operations test passed")
        
        # Test 4: Mark run as aborted
        atomic_io = get_atomic_io_manager()
        atomic_io.mark_run_aborted(test_run_id, "test_stage", "Test abortion")
        
        breadcrumb_file = temp_dir / "ABORTED"
        assert breadcrumb_file.exists(), "Breadcrumb file creation failed"
        
        with open(breadcrumb_file, 'r') as f:
            breadcrumb_data = json.load(f)
        
        assert breadcrumb_data['run_id'] == test_run_id, "Breadcrumb data mismatch"
        assert breadcrumb_data['stage_name'] == "test_stage", "Breadcrumb stage mismatch"
        print("✅ Run abortion and breadcrumb test passed")
        
        # Test 5: Cleanup temp directory
        success = atomic_io.cleanup_run_temp_directory(test_run_id, force=True)
        assert success, "Temp directory cleanup failed"
        assert not temp_dir.exists(), "Temp directory still exists after cleanup"
        print("✅ Temp directory cleanup test passed")
        
    except Exception as e:
        print(f"❌ Temp directory management test failed: {e}")
        return False
    
    return True

def test_file_handler_integration():
    """Test FileHandler integration with atomic I/O"""
    print("🧪 Testing FileHandler integration...")
    
    try:
        # Test 1: FileHandler with atomic I/O enabled
        test_run_id = f"filehandler_test_{int(time.time())}"
        
        file_handler = FileHandler(
            use_atomic_io=True,
            run_id=test_run_id,
            session_id="test_session"
        )
        
        # FileHandler should auto-create run temp directory
        assert file_handler.atomic_io is not None, "Atomic I/O manager not initialized"
        print("✅ FileHandler atomic I/O initialization test passed")
        
        # Test 2: Create session directory (now uses run temp directory)
        session_dir = file_handler.create_run_temp_directory(
            run_id=test_run_id,
            stage_name="filehandler_test"
        )
        
        assert Path(session_dir).exists(), "FileHandler session directory creation failed"
        print("✅ FileHandler session directory creation test passed")
        
        # Test 3: Save JSON with atomic operations
        test_data = {"filehandler": "integration_test", "timestamp": time.time()}
        json_path = file_handler.save_json(test_data, "test_json", "transcripts")
        
        assert Path(json_path).exists(), "FileHandler atomic JSON save failed"
        
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data, "FileHandler JSON save data mismatch"
        print("✅ FileHandler atomic JSON save test passed")
        
        # Test 4: Save text with atomic operations
        test_text = "FileHandler atomic text save test"
        text_path = file_handler.save_text(test_text, "test.txt", "transcripts")
        
        assert Path(text_path).exists(), "FileHandler atomic text save failed"
        
        with open(text_path, 'r') as f:
            loaded_text = f.read()
        
        assert loaded_text == test_text, "FileHandler text save content mismatch"
        print("✅ FileHandler atomic text save test passed")
        
        # Test 5: Test fallback to legacy mode
        file_handler_legacy = FileHandler(use_atomic_io=False)
        legacy_session_dir = file_handler_legacy.create_session_directory("legacy_test")
        
        assert Path(legacy_session_dir).exists(), "FileHandler legacy mode failed"
        print("✅ FileHandler legacy mode compatibility test passed")
        
        # Cleanup
        file_handler.cleanup_session(force=True)
        file_handler_legacy.cleanup_session()
        
    except Exception as e:
        print(f"❌ FileHandler integration test failed: {e}")
        return False
    
    return True

def test_collision_prevention():
    """Test filename collision prevention system"""
    print("🧪 Testing collision prevention...")
    
    try:
        test_dir = Path("/tmp/collision_test")
        test_dir.mkdir(exist_ok=True)
        
        # Test 1: Different cache keys should produce different temp file names
        test_file = test_dir / "collision_test.txt"
        
        # Get atomic I/O manager
        atomic_io = get_atomic_io_manager()
        
        # Create multiple temp files with different cache keys
        temp_files = []
        cache_keys = ["key1", "key2", "key3"]
        
        for cache_key in cache_keys:
            temp_file, temp_path = atomic_io.open_temp_for(test_file, cache_key=cache_key)
            temp_file.write(f"Content for {cache_key}")
            temp_file.close()
            temp_files.append(temp_path)
        
        # Verify all temp files have different names
        temp_names = [Path(tp).name for tp in temp_files]
        assert len(set(temp_names)) == len(temp_names), "Collision prevention failed - duplicate temp names"
        print("✅ Collision prevention test passed")
        
        # Cleanup temp files
        for temp_path in temp_files:
            safe_remove(temp_path)
        
        shutil.rmtree(test_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ Collision prevention test failed: {e}")
        return False
    
    return True

def test_statistics_and_telemetry():
    """Test statistics and telemetry functionality"""
    print("🧪 Testing statistics and telemetry...")
    
    try:
        atomic_io = get_atomic_io_manager()
        
        # Get initial statistics
        initial_stats = atomic_io.get_statistics()
        assert isinstance(initial_stats, dict), "Statistics not returned as dict"
        assert 'atomic_commits_count' in initial_stats, "Missing atomic_commits_count in stats"
        print("✅ Statistics retrieval test passed")
        
        # Perform some operations to update stats
        test_dir = Path("/tmp/stats_test")
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / "stats_test.txt"
        with atomic_write(test_file) as f:
            f.write("Statistics test content")
        
        # Get updated statistics
        updated_stats = atomic_io.get_statistics()
        assert updated_stats['atomic_commits_count'] > initial_stats['atomic_commits_count'], "Statistics not updated"
        print("✅ Statistics update test passed")
        
        # Test janitor cleanup (dry run)
        cleanup_stats = atomic_io.run_janitor_cleanup()
        assert isinstance(cleanup_stats, dict), "Cleanup stats not returned as dict"
        assert 'directories_scanned' in cleanup_stats, "Missing directories_scanned in cleanup stats"
        print("✅ Janitor cleanup test passed")
        
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ Statistics and telemetry test failed: {e}")
        return False
    
    return True

def main():
    """Run all atomic I/O integration tests"""
    print("🚀 Starting Atomic I/O Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Core Atomic Operations", test_atomic_operations),
        ("Temp Directory Management", test_temp_directory_management),
        ("FileHandler Integration", test_file_handler_integration),
        ("Collision Prevention", test_collision_prevention),
        ("Statistics and Telemetry", test_statistics_and_telemetry)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} tests...")
        try:
            if test_func():
                print(f"✅ {test_name} tests PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} tests FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} tests FAILED with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All atomic I/O integration tests PASSED!")
        print("🔒 Atomic I/O system is ready for production use")
        return True
    else:
        print("⚠️  Some tests failed - please review implementation")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)