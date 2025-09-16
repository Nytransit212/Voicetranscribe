#!/usr/bin/env python3
"""Simple atomic I/O validation test"""

import json
import tempfile
from pathlib import Path
from utils.atomic_io import atomic_write, get_atomic_io_manager

def test_basic_atomic_operations():
    """Test basic atomic operations"""
    print("🧪 Testing basic atomic operations...")
    
    # Use /tmp which should be writable
    test_dir = Path("/tmp/simple_atomic_test")
    test_dir.mkdir(exist_ok=True)
    
    # Test 1: Atomic write
    test_file = test_dir / "test.json"
    test_data = {"test": "success", "atomic": True}
    
    try:
        with atomic_write(test_file) as f:
            json.dump(test_data, f, indent=2)
        
        # Verify file exists and has correct content
        if test_file.exists():
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
            
            if loaded_data == test_data:
                print("✅ Atomic write test passed")
                return True
            else:
                print("❌ Data mismatch")
                return False
        else:
            print("❌ File doesn't exist")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        test_dir.rmdir()

def test_atomic_io_manager():
    """Test atomic I/O manager initialization"""
    print("🧪 Testing atomic I/O manager...")
    
    try:
        manager = get_atomic_io_manager()
        
        if manager is not None:
            stats = manager.get_statistics()
            if isinstance(stats, dict) and 'atomic_commits_count' in stats:
                print("✅ Atomic I/O manager test passed")
                return True
            else:
                print("❌ Invalid statistics")
                return False
        else:
            print("❌ Manager not initialized")
            return False
            
    except Exception as e:
        print(f"❌ Manager test failed: {e}")
        return False

def main():
    """Run simple validation tests"""
    print("🚀 Running Simple Atomic I/O Validation")
    print("=" * 40)
    
    tests = [
        test_basic_atomic_operations,
        test_atomic_io_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🏁 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed - Atomic I/O is working!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)