#!/usr/bin/env python3
"""
Direct processing with cache priority for fastest results
"""
import sys
import os
sys.path.append('.')

from utils.google_drive_handler import GoogleDriveHandler
from core.ensemble_manager import EnsembleManager

def main():
    # File ID from Google Drive URL
    file_id = '1VLXvaXYa0LMokEHPp-58ERX1wFcysWAu'
    
    print('🚀 Processing with cache priority for fastest assembly...')
    print(f'📁 File ID: {file_id}')
    
    try:
        # Initialize with caching enabled
        drive_handler = GoogleDriveHandler()
        ensemble_manager = EnsembleManager(
            expected_speakers=10,
            noise_level='medium',
            enable_versioning=True,
            consensus_strategy='best_single_candidate'
        )
        
        print('📥 Downloading from Google Drive...')
        local_path = drive_handler.download_file(file_id, target_name='cached_video.mp4')
        print(f'✅ Downloaded: {local_path}')
        
        # Get file size for info
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f'📊 File size: {size_mb:.1f} MB')
        
        print('🎯 Processing (cache-first approach)...')
        
        def progress_update(stage, percent, message):
            if 'cache' in message.lower() or 'cached' in message.lower():
                print(f'⚡ CACHE HIT: {message}')
            else:
                print(f'[{stage}] {percent:3d}% - {message}')
        
        # Process with cache priority
        results = ensemble_manager.process_video(
            local_path,
            progress_callback=progress_update
        )
        
        print('\n🎉 PROCESSING COMPLETE!')
        
        # Show key metrics
        cost = results.get('cost_summary', {}).get('total_cost_usd', 0)
        time_taken = results.get('processing_time', 0)
        score = results.get('winner_score', 0)
        speakers = results.get('detected_speakers', 0)
        
        print(f'💰 Cost: ${cost:.2f}')
        print(f'⏱️  Time: {time_taken:.1f}s')
        print(f'🎯 Confidence: {score:.3f}')
        print(f'👥 Speakers: {speakers}')
        
        # Show output files (from our fix)
        if 'output_files' in results:
            output_info = results['output_files']
            print(f'\n📂 OUTPUTS SAVED TO:')
            print(f'   {output_info["directory"]}')
            print(f'📄 Files created: {len(output_info["files"])}')
            for filepath in output_info['files']:
                filename = os.path.basename(filepath)
                print(f'   ✓ {filename}')
            print(f'\n🆔 Run ID: {output_info["run_id"]}')
        else:
            print('\n⚠️  Output files location not found in results')
        
        print('\n✅ ALL TRANSCRIPT FILES READY!')
        
        # Show transcript preview
        if 'transcript_preview' in results:
            print('\n📖 TRANSCRIPT PREVIEW:')
            for item in results['transcript_preview'][:2]:
                print(f'   [{item["timestamp"]}] {item["speaker"]}: {item["text"][:100]}...')
        
        return results
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()