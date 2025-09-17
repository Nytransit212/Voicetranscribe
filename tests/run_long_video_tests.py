#!/usr/bin/env python3
"""
Test runner for long video processing integration tests.

This script runs the comprehensive test suite for long video processing,
providing detailed reporting and validation of the transcription system's
ability to handle long-form content without truncation.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_test_suite(test_level: str = "quick", verbose: bool = True, coverage: bool = False) -> Dict[str, Any]:
    """
    Run the long video processing test suite
    
    Args:
        test_level: "quick", "comprehensive", or "all"
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        
    Returns:
        Dictionary with test results
    """
    print("🎬 Long Video Processing Test Suite")
    print("=" * 50)
    
    # Define test configurations
    test_configs = {
        "quick": {
            "description": "Quick validation tests (< 2 minutes)",
            "files": [
                "tests/test_duration_validation.py",
            ],
            "markers": ["not slow", "not memory_intensive"]
        },
        "comprehensive": {
            "description": "Comprehensive tests including memory monitoring (< 10 minutes)",
            "files": [
                "tests/test_duration_validation.py",
                "tests/test_memory_monitoring.py",
                "tests/test_error_handling_long_videos.py"
            ],
            "markers": ["not slow"]
        },
        "all": {
            "description": "Full test suite including long video simulations (< 30 minutes)",
            "files": [
                "tests/test_duration_validation.py",
                "tests/test_memory_monitoring.py", 
                "tests/test_error_handling_long_videos.py",
                "tests/test_long_video_integration.py"
            ],
            "markers": []
        }
    }
    
    if test_level not in test_configs:
        raise ValueError(f"Invalid test level: {test_level}. Choose from: {list(test_configs.keys())}")
    
    config = test_configs[test_level]
    print(f"Running: {config['description']}")
    print(f"Test files: {len(config['files'])}")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test files
    for test_file in config['files']:
        if Path(test_file).exists():
            cmd.append(test_file)
        else:
            print(f"⚠️  Warning: Test file not found: {test_file}")
    
    # Add markers
    if config['markers']:
        for marker in config['markers']:
            cmd.extend(["-m", marker])
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=core", "--cov=utils", "--cov-report=html", "--cov-report=term-missing"])
    
    # Add timeout for safety (if timeout plugin is available)
    try:
        import pytest_timeout
        cmd.extend(["--timeout=1800"])  # 30 minute timeout
    except ImportError:
        pass  # Skip timeout if plugin not available
    
    # Add detailed output
    cmd.extend(["--tb=short", "--strict-markers"])
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run tests
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)  # 40 minute timeout
        duration = time.time() - start_time
        
        # Parse results
        success = result.returncode == 0
        
        print("\n" + "=" * 50)
        print(f"🏁 Test Results ({duration:.1f}s)")
        print("=" * 50)
        
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
        
        print(f"\nReturn code: {result.returncode}")
        
        # Print stdout (test results)
        if result.stdout:
            print("\n📊 Test Output:")
            print("-" * 30)
            print(result.stdout)
        
        # Print stderr (errors/warnings)
        if result.stderr:
            print("\n⚠️  Warnings/Errors:")
            print("-" * 30)
            print(result.stderr)
        
        return {
            "success": success,
            "duration": duration,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"\n⏰ Tests timed out after {duration:.1f}s")
        return {
            "success": False,
            "duration": duration,
            "return_code": -1,
            "error": "Timeout"
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n💥 Error running tests: {e}")
        return {
            "success": False,
            "duration": duration,
            "return_code": -1,
            "error": str(e)
        }


def validate_test_environment() -> bool:
    """Validate that the test environment is properly set up"""
    print("🔍 Validating test environment...")
    
    required_files = [
        "core/audio_processor.py",
        "core/ensemble_manager.py",
        "config/config.yaml",
        "tests/conftest.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    # Check for pytest
    try:
        result = subprocess.run(["python", "-m", "pytest", "--version"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ pytest not available")
            return False
    except Exception:
        print("❌ Error checking pytest availability")
        return False
    
    print("✅ Test environment validated")
    return True


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run long video processing tests")
    parser.add_argument("--level", choices=["quick", "comprehensive", "all"], 
                       default="quick", help="Test level to run")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output")
    parser.add_argument("--coverage", action="store_true",
                       help="Enable coverage reporting")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate environment, don't run tests")
    
    args = parser.parse_args()
    
    # Validate environment
    if not validate_test_environment():
        print("\n❌ Environment validation failed")
        sys.exit(1)
    
    if args.validate_only:
        print("\n✅ Environment validation complete")
        sys.exit(0)
    
    # Run tests
    try:
        results = run_test_suite(
            test_level=args.level,
            verbose=args.verbose,
            coverage=args.coverage
        )
        
        # Print summary
        print("\n" + "=" * 50)
        print("📋 Summary")
        print("=" * 50)
        print(f"Test Level: {args.level}")
        print(f"Duration: {results['duration']:.1f}s")
        print(f"Success: {'✅ Yes' if results['success'] else '❌ No'}")
        
        if results['success']:
            print("\n🎉 All long video processing tests passed!")
            print("The system successfully handles long videos without truncation.")
        else:
            print("\n⚠️  Some tests failed. Please review the output above.")
            print("This may indicate issues with long video processing.")
        
        sys.exit(0 if results['success'] else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()