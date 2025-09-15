#!/usr/bin/env python3
"""
Test processing a single chunk through the ensemble pipeline
"""

import os
import sys
import json
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ensemble_manager import EnsembleManager

def main():
    """Test processing a single chunk"""
    
    # Configuration for 10 speakers
    expected_speakers = 10
    noise_level = 'medium'
    
    # Initialize ensemble manager with proper configuration
    print("🚀 Initializing Ensemble Manager...")
    
    ensemble = EnsembleManager(
        expected_speakers=expected_speakers,
        noise_level=noise_level,
        enable_versioning=True,
        domain="general",
        consensus_strategy="best_single_candidate",
        calibration_method="registry_based",
        enable_speaker_mapping=True
    )
    
    # Test with first chunk
    chunk_file = "artifacts/20min_processing/chunks/chunk_01_0-240s.mp4"
    
    print(f"🎬 Processing test chunk: {chunk_file}")
    
    try:
        start_time = time.time()
        
        # Process the chunk (correct method call with only video path)
        result = ensemble.process_video(chunk_file)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.1f}s")
        
        # Save result
        output_file = "artifacts/20min_processing/test_chunk_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Result saved to: {output_file}")
        
        # Print summary
        if 'best_transcript' in result:
            transcript_preview = result['best_transcript'][:200] + "..." if len(result['best_transcript']) > 200 else result['best_transcript']
            print(f"📝 Transcript preview: {transcript_preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test failed!")