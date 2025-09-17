#!/usr/bin/env python3
"""
Test script to verify StageCompletionManager fixes are working correctly.

This test verifies:
1. StageCompletionManager initializes without NameError
2. Atomic write operations work correctly
3. Stage completion marking functions properly
4. Resume detection works end-to-end
"""

import os
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from core.stage_completion_manager import (
    StageCompletionManager, 
    ProcessingStage, 
    get_stage_completion_manager,
    StageCompletionError
)

def test_stage_completion_manager_initialization():
    """Test that StageCompletionManager initializes without NameError"""
    print("🧪 Testing StageCompletionManager initialization...")
    
    try:
        # Test direct initialization
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StageCompletionManager(base_work_dir=temp_dir)
            print("✅ Direct initialization successful")
            
            # Test factory function
            manager2 = get_stage_completion_manager(base_work_dir=temp_dir)
            print("✅ Factory function initialization successful")
            
            # Test manifest manager setup
            manager.set_manifest_manager(
                session_dir=temp_dir,
                session_id="test_session",
                project_id="test_project", 
                run_id="test_run"
            )
            print("✅ Manifest manager setup successful")
            
            return True
            
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_atomic_stage_completion():
    """Test atomic stage completion operations"""
    print("\n🧪 Testing atomic stage completion...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StageCompletionManager(base_work_dir=temp_dir)
            
            # Test data
            run_id = "test_run_123"
            session_id = "test_session_456"
            project_id = "test_project_789"
            
            inputs_sha256 = ["abc123", "def456"]
            config_snapshot = {"model": "whisper-1", "temperature": 0.0}
            model_versions = {"whisper": "1.0", "pyannote": "3.1"}
            stage_outputs = [{"type": "audio", "path": "/tmp/test.wav", "sha256": "xyz789"}]
            stage_metadata = {"duration": 120.5, "speakers": 2}
            processing_duration = 45.2
            
            # Test marking stage complete
            marker = manager.mark_stage_complete(
                stage=ProcessingStage.AUDIO_EXTRACTION,
                run_id=run_id,
                session_id=session_id,
                project_id=project_id,
                inputs_sha256=inputs_sha256,
                config_snapshot=config_snapshot,
                model_versions=model_versions,
                stage_outputs=stage_outputs,
                stage_metadata=stage_metadata,
                processing_duration=processing_duration
            )
            
            print("✅ Stage completion marking successful")
            
            # Verify marker was written
            completed_stages = manager.get_completed_stages(run_id)
            assert ProcessingStage.AUDIO_EXTRACTION in completed_stages
            print("✅ Stage completion verification successful")
            
            # Test resume detection
            next_stage = manager.get_next_stage_to_process(run_id)
            assert next_stage == ProcessingStage.DIARIZATION
            print("✅ Resume detection successful")
            
            return True
            
    except Exception as e:
        print(f"❌ Stage completion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_manager_integration():
    """Test EnsembleManager integration with StageCompletionManager"""
    print("\n🧪 Testing EnsembleManager integration...")
    
    try:
        from core.ensemble_manager import EnsembleManager
        
        # Test safe creation
        manager = EnsembleManager.create_safe(expected_speakers=2)
        
        if manager.stage_completion_manager is None:
            print("❌ EnsembleManager failed to initialize StageCompletionManager")
            return False
        
        print("✅ EnsembleManager has working StageCompletionManager")
        
        # Check if initialization warnings exist
        if hasattr(manager, '_initialization_warnings'):
            warnings = manager._initialization_warnings
            if warnings:
                print(f"⚠️ Initialization warnings: {warnings}")
            else:
                print("✅ No initialization warnings")
        
        return True
        
    except Exception as e:
        print(f"❌ EnsembleManager integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting StageCompletionManager fix verification tests...\n")
    
    tests = [
        test_stage_completion_manager_initialization,
        test_atomic_stage_completion,
        test_ensemble_manager_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED! StageCompletionManager fixes are working!")
        return 0
    else:
        print("\n💥 SOME TESTS FAILED! Additional fixes needed.")
        return 1

if __name__ == "__main__":
    exit(main())