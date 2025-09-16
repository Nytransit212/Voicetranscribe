#!/usr/bin/env python3
"""
Process public Google Drive file without authentication
"""
import sys
import os
import requests
import tempfile
sys.path.append('.')

from core.ensemble_manager import EnsembleManager

def download_public_drive_file(file_id, output_name):
    """Download file from public Google Drive link"""
    print(f'📥 Downloading public file: {file_id}')
    
    # Try direct download first
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    with requests.session() as session:
        response = session.get(url, stream=True)
        
        # Handle confirmation token for large files
        if 'confirm=' in response.text:
            token = None
            for line in response.text.split('\n'):
                if 'confirm=' in line:
                    # Extract token
                    token = line.split('confirm=')[1].split('"')[0]
                    break
            
            if token:
                url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
                response = session.get(url, stream=True)
        
        # Save file
        output_path = f"/tmp/{output_name}"
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024*1024*10) == 0:  # Every 10MB
                            print(f'  📦 {progress:.1f}% ({downloaded/1024/1024:.1f}MB)')
        
        print(f'✅ Downloaded: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f}MB)')
        return output_path

def main():
    file_id = '1VLXvaXYa0LMokEHPp-58ERX1wFcysWAu'
    
    try:
        print('🚀 Processing public Google Drive file...')
        
        # Download file
        video_path = download_public_drive_file(file_id, 'public_video.mp4')
        
        # Initialize with chunked processing for 85+ minute files
        print('🎯 Initializing chunked processing for long-form audio...')
        ensemble_manager = EnsembleManager(
            expected_speakers=10,
            noise_level='medium',
            enable_versioning=True,
            chunked_processing_threshold=1200  # 20 minutes max per chunk
        )
        
        def progress_callback(stage, percent, message):
            if 'chunk' in message.lower():
                print(f'🔄 CHUNK: {message}')
            else:
                print(f'[{stage}] {percent:3d}% - {message}')
        
        print('⚡ Starting chunked ensemble processing...')
        results = ensemble_manager.process_video(
            video_path,
            progress_callback=progress_callback
        )
        
        print('\n🎉 PROCESSING COMPLETE!')
        
        # Show results
        cost = results.get('cost_summary', {}).get('total_cost_usd', 0)
        time_taken = results.get('processing_time', 0)  
        score = results.get('winner_score', 0)
        speakers = results.get('detected_speakers', 0)
        
        print(f'💰 Cost: ${cost:.2f}')
        print(f'⏱️  Time: {time_taken:.1f}s')
        print(f'🎯 Confidence: {score:.3f}')
        print(f'👥 Speakers: {speakers}')
        
        # Show output files
        if 'output_files' in results:
            output_info = results['output_files']
            print(f'\n📂 OUTPUTS SAVED TO:')
            print(f'   {output_info["directory"]}')
            print(f'📄 Files: {len(output_info["files"])}')
            for filepath in output_info['files']:
                print(f'   ✓ {os.path.basename(filepath)}')
        
        print('\n✅ ALL TRANSCRIPT FILES READY!')
        
        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        if 'access denied' in str(e).lower() or '403' in str(e):
            print('\n🔑 File is not publicly accessible.')
            print('To fix: Open your Google Drive file → Share → Change to "Anyone with the link can view"')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()