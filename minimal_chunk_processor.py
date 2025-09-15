#!/usr/bin/env python3
"""
Minimal Chunk Processor - Process chunks one at a time with minimal complexity
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path

def process_single_chunk_minimal(chunk_path: str, chunk_index: int):
    """Process a single chunk with minimal setup"""
    
    print(f"\n{'='*50}")
    print(f"Processing Chunk {chunk_index}: {Path(chunk_path).name}")
    print(f"{'='*50}")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Import with minimal dependencies
        from core.ensemble_manager import EnsembleManager
        
        print("✅ Imports successful")
        
        # Minimal configuration
        print("🔧 Initializing EnsembleManager...")
        ensemble = EnsembleManager(
            expected_speakers=10,
            noise_level='medium',
            enable_versioning=False,  # Disable to reduce complexity
            domain="general",
            consensus_strategy="best_single_candidate", 
            calibration_method="registry_based",
            enable_speaker_mapping=True
        )
        
        print("✅ EnsembleManager initialized")
        
        # Process the chunk
        print(f"🎬 Processing video file...")
        start_time = time.time()
        
        result = ensemble.process_video(chunk_path)
        
        processing_time = time.time() - start_time
        print(f"✅ Processing completed in {processing_time:.1f}s")
        
        # Save result
        results_dir = Path("artifacts/20min_processing/results")
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / f"chunk_{chunk_index:02d}_result.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Result saved: {result_file}")
        
        # Show preview
        if 'best_transcript' in result:
            transcript = result['best_transcript']
            preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
            print(f"📝 Transcript preview: {preview}")
        elif 'transcript' in result:
            transcript = result['transcript']  
            preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
            print(f"📝 Transcript preview: {preview}")
        
        return {
            'status': 'success',
            'chunk_index': chunk_index,
            'chunk_name': Path(chunk_path).name,
            'processing_time': processing_time,
            'result_file': str(result_file),
            'result': result
        }
        
    except Exception as e:
        error_msg = f"Failed to process chunk {chunk_index}: {e}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'chunk_index': chunk_index,
            'chunk_name': Path(chunk_path).name,
            'processing_time': 0,
            'result_file': None,
            'error': str(e)
        }

def main():
    """Main processing loop"""
    print("🚀 Starting Minimal Chunk Processing")
    
    # Find all chunk files
    chunks_dir = Path("artifacts/20min_processing/chunks")
    chunk_files = sorted(chunks_dir.glob("chunk_*.mp4"))
    
    if not chunk_files:
        print("❌ No chunk files found!")
        return
    
    print(f"📁 Found {len(chunk_files)} chunks to process")
    
    # Process each chunk
    results = []
    successful_chunks = 0
    
    for i, chunk_file in enumerate(chunk_files, 1):
        print(f"\n🔄 Processing chunk {i}/{len(chunk_files)}")
        
        result = process_single_chunk_minimal(str(chunk_file), i)
        results.append(result)
        
        if result['status'] == 'success':
            successful_chunks += 1
            print(f"✅ Chunk {i} completed successfully")
        else:
            print(f"❌ Chunk {i} failed")
        
        # Small delay between chunks to avoid overwhelming the system
        time.sleep(2)
    
    # Generate summary
    print(f"\n{'='*60}")
    print(f"📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunk_files)}")
    print(f"Successful: {successful_chunks}")
    print(f"Failed: {len(chunk_files) - successful_chunks}")
    
    # Save processing summary
    summary_file = Path("artifacts/20min_processing/results/processing_summary.json")
    summary = {
        "total_chunks": len(chunk_files),
        "successful_chunks": successful_chunks,
        "failed_chunks": len(chunk_files) - successful_chunks,
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Summary saved: {summary_file}")
    
    if successful_chunks > 0:
        print(f"\n🎉 Successfully processed {successful_chunks} chunks!")
        print(f"📁 Check results in: artifacts/20min_processing/results/")
    else:
        print(f"\n💥 No chunks processed successfully")

if __name__ == "__main__":
    main()