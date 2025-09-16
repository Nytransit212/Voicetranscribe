#!/usr/bin/env python3
import sys
sys.path.append('.')
from utils.google_drive_handler import GoogleDriveHandler  
from core.ensemble_manager import EnsembleManager

# Extract file ID from Google Drive URL
file_id = '1VLXvaXYa0LMokEHPp-58ERX1wFcysWAu'

try:
    print('🚀 Processing your Google Drive file...')
    
    # Initialize systems
    drive_handler = GoogleDriveHandler()
    ensemble_manager = EnsembleManager(
        expected_speakers=10,
        noise_level='medium', 
        enable_versioning=True
    )
    
    # Download file
    print('📥 Downloading from Google Drive...')
    local_path = drive_handler.download_file(file_id, target_name='transcription_video.mp4')
    print(f'✅ Downloaded: {local_path}')
    
    # Process with progress tracking
    def show_progress(stage, percent, message):
        print(f'[{stage}] {percent:3d}% - {message}')
    
    print('🎯 Starting ensemble transcription...')
    results = ensemble_manager.process_video(local_path, progress_callback=show_progress)
    
    print('\n🎉 TRANSCRIPTION COMPLETE!')
    print(f'💰 Cost: ${results["cost_summary"].get("total_cost_usd", 0):.2f}')
    print(f'⏱️  Time: {results["processing_time"]:.1f}s') 
    print(f'🎯 Score: {results["winner_score"]:.3f}')
    print(f'👥 Speakers: {results["detected_speakers"]}')
    
    # Show output file locations
    if 'output_files' in results:
        output_info = results['output_files']
        print(f'\n📂 OUTPUTS SAVED TO: {output_info["directory"]}')
        print(f'📄 Files created: {len(output_info["files"])}')
        for filepath in output_info['files']:
            print(f'  ✓ {filepath}')
        print(f'\n🆔 Run ID: {output_info["run_id"]}')
    
    print('\n✅ All transcript files ready for download!')
    
except Exception as e:
    print(f'❌ Error: {str(e)}')
    import traceback
    traceback.print_exc()