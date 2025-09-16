#!/usr/bin/env python3
"""
Process Google Drive file using service account authentication
"""
import sys
import os
sys.path.append('.')

from core.ensemble_manager import EnsembleManager
from utils.google_drive_handler import GoogleDriveHandler

def main():
    file_id = '1VLXvaXYa0LMokEHPp-58ERX1wFcysWAu'
    
    print('🚀 Processing with service account authentication...')
    print(f'📁 File ID: {file_id}')
    
    try:
        # Initialize with service account credentials
        print('🔑 Initializing Google Drive handler with service account...')
        drive_handler = GoogleDriveHandler()
        
        # Check if service account is properly initialized
        if not hasattr(drive_handler, 'service') or drive_handler.service is None:
            print('❌ Service account not properly initialized')
            
            # Check if credentials are available
            if 'GOOGLE_SERVICE_ACCOUNT_KEY' in os.environ:
                print('✅ GOOGLE_SERVICE_ACCOUNT_KEY found in environment')
            else:
                print('❌ GOOGLE_SERVICE_ACCOUNT_KEY not found in environment')
            
            return None
        
        print('✅ Service account initialized successfully')
        
        # Download file using service account
        print('📥 Downloading file using service account...')
        local_path = '/tmp/cincinnati_swing.mp4'
        downloaded_path = drive_handler.download_file(file_id, local_path)
        
        if not local_path or not os.path.exists(local_path):
            print('❌ File download failed')
            return None
            
        # Check file size and type
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f'✅ Downloaded: {local_path}')
        print(f'📊 File size: {size_mb:.1f} MB')
        
        # Verify it's actually a video file, not HTML
        with open(local_path, 'rb') as f:
            first_bytes = f.read(100)
            if b'<!DOCTYPE html>' in first_bytes:
                print('❌ Downloaded file is HTML, not video - authentication failed')
                return None
        
        print('✅ File is valid video format')
        
        # Initialize ensemble manager with chunked processing for large files
        print('🎯 Initializing ensemble manager for large file processing...')
        ensemble_manager = EnsembleManager(
            expected_speakers=10,
            noise_level='medium',
            enable_versioning=True,
            chunked_processing_threshold=1200,  # 20-minute chunks
            consensus_strategy='best_single_candidate'
        )
        
        def progress_callback(stage, percent, message):
            if 'chunk' in message.lower():
                print(f'🔄 CHUNK: {message}')
            elif 'cache' in message.lower():
                print(f'⚡ CACHE: {message}')
            else:
                print(f'[{stage}] {percent:3d}% - {message}')
        
        print('⚡ Starting ensemble processing with chunked approach...')
        print('📋 This will create 25 candidate transcripts using 5 diarization + 5 ASR variants')
        
        results = ensemble_manager.process_video(
            local_path,
            progress_callback=progress_callback
        )
        
        print('\n🎉 PROCESSING COMPLETE!')
        
        # Display results summary
        cost = results.get('cost_summary', {}).get('total_cost_usd', 0)
        time_taken = results.get('processing_time', 0)
        score = results.get('winner_score', 0)
        speakers = results.get('detected_speakers', 0)
        
        print(f'💰 Total cost: ${cost:.2f}')
        print(f'⏱️  Processing time: {time_taken:.1f}s')
        print(f'🎯 Confidence score: {score:.3f}')
        print(f'👥 Speakers detected: {speakers}')
        
        # Show output files location
        if 'output_files' in results:
            output_info = results['output_files']
            print(f'\n📂 OUTPUT FILES SAVED TO:')
            print(f'   📁 Directory: {output_info["directory"]}')
            print(f'   🆔 Run ID: {output_info["run_id"]}')
            print(f'   📄 Files created: {len(output_info["files"])}')
            
            for filepath in output_info['files']:
                filename = os.path.basename(filepath)
                file_size = os.path.getsize(filepath) / 1024
                print(f'     ✓ {filename} ({file_size:.1f} KB)')
        else:
            print('\n⚠️  Output files location not found in results')
        
        # Show transcript preview
        if 'transcript_preview' in results:
            print('\n📖 TRANSCRIPT PREVIEW:')
            for i, item in enumerate(results['transcript_preview'][:3]):
                print(f'   [{item["timestamp"]}] {item["speaker"]}: {item["text"][:80]}...')
        
        print('\n✅ ALL TRANSCRIPT FILES READY FOR DOWNLOAD!')
        
        # Clean up downloaded file
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f'🧹 Cleaned up temporary file: {local_path}')
        
        return results
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()