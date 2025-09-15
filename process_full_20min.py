#!/usr/bin/env python3
"""
Process the full 20-minute video file directly through AssemblyAI ensemble pipeline
"""

import os
import sys
import json
import time
from pathlib import Path

def main():
    """Process the full 20-minute file directly"""
    
    print("🚀 Processing Full 20-Minute Video")
    print("=" * 50)
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Import ensemble manager
        from core.ensemble_manager import EnsembleManager
        
        print("✅ Imports successful")
        
        # Configure for 10 speakers
        print("🔧 Initializing EnsembleManager for 10 speakers...")
        ensemble = EnsembleManager(
            expected_speakers=10,
            noise_level='medium',
            enable_versioning=True,
            domain="general",
            consensus_strategy="best_single_candidate",
            calibration_method="registry_based",
            enable_speaker_mapping=True,
            chunked_processing_threshold=900.0  # 15 minutes - allows chunked processing for our 20min file
        )
        
        print("✅ EnsembleManager initialized")
        
        # Process the full 20-minute file
        video_path = "artifacts/20min_processing/first_20min.mp4"
        
        print(f"🎬 Processing 20-minute video: {Path(video_path).name}")
        print("📊 Configuration:")
        print(f"  - Expected speakers: 10")
        print(f"  - Noise level: medium")
        print(f"  - Domain: general")
        print(f"  - Consensus: best_single_candidate")
        print(f"  - Enable chunked processing for files >15min")
        
        start_time = time.time()
        
        # Process the video
        result = ensemble.process_video(video_path)
        
        processing_time = time.time() - start_time
        print(f"✅ Processing completed in {processing_time:.1f}s ({processing_time/60:.1f} minutes)")
        
        # Save result
        results_dir = Path("artifacts/20min_processing/results")
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / "full_20min_result.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Full result saved: {result_file}")
        
        # Show transcript preview
        if 'best_transcript' in result:
            transcript = result['best_transcript']
            preview = transcript[:300] + "..." if len(transcript) > 300 else transcript
            print(f"\n📝 Transcript preview:")
            print("-" * 40)
            print(preview)
            print("-" * 40)
            print(f"Total transcript length: {len(transcript)} characters")
        elif 'transcript' in result:
            transcript = result['transcript']
            preview = transcript[:300] + "..." if len(transcript) > 300 else transcript
            print(f"\n📝 Transcript preview:")
            print("-" * 40) 
            print(preview)
            print("-" * 40)
            print(f"Total transcript length: {len(transcript)} characters")
        
        # Generate additional output formats
        print("\n🔄 Generating additional output formats...")
        
        # TXT format
        if 'best_transcript' in result:
            txt_file = results_dir / "full_20min_transcript.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(result['best_transcript'])
            print(f"📄 TXT transcript: {txt_file}")
        
        # Summary information
        print(f"\n📊 Processing Summary:")
        print(f"  - Processing time: {processing_time:.1f}s")
        print(f"  - Results directory: {results_dir}")
        print(f"  - Full result JSON: {result_file}")
        
        if 'speakers' in result:
            print(f"  - Detected speakers: {len(result['speakers'])}")
        
        if 'confidence_scores' in result:
            print(f"  - Confidence scores available: Yes")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 20-Minute Processing Completed Successfully!")
        print("📁 Check artifacts/20min_processing/results/ for all outputs")
    else:
        print("\n💥 Processing failed!")