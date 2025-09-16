#!/usr/bin/env python3
"""
Quick script to process Google Drive file and get outputs
"""

import sys
import os
sys.path.append('.')

from utils.google_drive_handler import GoogleDriveHandler
from core.ensemble_manager import EnsembleManager

def main():
    # Initialize handlers
    print('🚀 Initializing ensemble transcription system...')
    
    drive_handler = GoogleDriveHandler()
    ensemble_manager = EnsembleManager(
        expected_speakers=10,
        noise_level='medium', 
        enable_versioning=True,
        consensus_strategy='best_single_candidate'
    )

    # Google Drive file ID from URL
    file_id = '1VLXvaXYa0LMokEHPp-58ERX1wFcysWAu'
    
    print('🚀 Starting processing of Google Drive file...')
    print(f'📁 File ID: {file_id}')

    try:
        # Download from Google Drive
        print('📥 Downloading file from Google Drive...')
        local_path = drive_handler.download_file(file_id, target_name='user_video.mp4')
        print(f'✅ Downloaded to: {local_path}')
        
        # Get file info
        file_info = drive_handler.get_file_info(file_id)
        size_mb = int(file_info.get('size', 0)) / (1024 * 1024)
        print(f'📊 File size: {size_mb:.1f} MB')
        
        # Process with ensemble
        print('🎯 Processing with ensemble transcription system...')
        
        def progress_callback(stage, percent, message):
            print(f'[{stage}] {percent}% - {message}')
        
        results = ensemble_manager.process_video(
            local_path, 
            progress_callback=progress_callback
        )
        
        print('\n🎉 Processing completed successfully!')
        
        # Show output files info
        if 'output_files' in results:
            output_info = results['output_files']
            print(f'\n📂 Outputs saved to: {output_info["directory"]}')
            print(f'📄 Files created: {len(output_info["files"])}')
            for file_path in output_info['files']:
                print(f'  - {file_path}')
        
        # Show summary
        print(f'\n📊 PROCESSING SUMMARY:')
        print(f'💰 Total cost: ${results["cost_summary"].get("total_cost_usd", 0.0):.2f}')
        print(f'⏱️ Processing time: {results["processing_time"]:.1f}s')
        print(f'🎯 Confidence score: {results["winner_score"]:.3f}')
        print(f'👥 Detected speakers: {results["detected_speakers"]}')
        print(f'📝 Transcript segments: {results["session_metadata"]["candidates_generated"]}')
        
        # Show transcript preview
        if 'transcript_preview' in results:
            print(f'\n📖 TRANSCRIPT PREVIEW:')
            for item in results['transcript_preview'][:3]:
                print(f'  [{item["timestamp"]}] {item["speaker"]}: {item["text"]}')
        
        print(f'\n✅ All outputs ready for download!')
        
    except Exception as e:
        print(f'❌ Error processing file: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()