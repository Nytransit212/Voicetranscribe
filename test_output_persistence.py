#!/usr/bin/env python3
"""
Test the output persistence functionality with minimal processing
"""
import sys
import os
sys.path.append('.')

from core.ensemble_manager import EnsembleManager
from pathlib import Path
import tempfile
import time

def create_short_test_audio():
    """Create a 10-second test audio file"""
    test_audio_path = '/tmp/test_audio_10sec.wav'
    
    # Create a simple 10-second silent audio file for testing
    os.system(f'''
    ffmpeg -f lavfi -i "anullsrc=channel_layout=mono:sample_rate=16000" -t 10 -y {test_audio_path} 2>/dev/null
    ''')
    
    return test_audio_path

def test_minimal_processing():
    """Test minimal processing to verify output persistence"""
    
    print('🧪 Testing output persistence with 10-second audio...')
    
    try:
        # Create test audio
        test_audio = create_short_test_audio()
        
        if not os.path.exists(test_audio):
            print('❌ Failed to create test audio')
            return False
        
        print(f'✅ Test audio created: {test_audio}')
        
        # Initialize manager with minimal settings
        manager = EnsembleManager(
            expected_speakers=2,  # Minimal speakers
            noise_level='low',
            enable_versioning=True,
            chunked_processing_threshold=3600  # Don't chunk 10 seconds
        )
        
        def progress_callback(stage, percent, message):
            print(f'[TEST] {stage} {percent}% - {message}')
        
        print('🎯 Starting minimal processing...')
        
        # Process with timeout
        results = manager.process_video(
            test_audio,
            progress_callback=progress_callback
        )
        
        print('\n📊 RESULTS:')
        
        if results:
            # Check for output files
            if 'output_files' in results:
                output_info = results['output_files']
                print(f'✅ Output directory: {output_info["directory"]}')
                print(f'✅ Files created: {len(output_info["files"])}')
                
                for filepath in output_info['files']:
                    filename = os.path.basename(filepath)
                    if os.path.exists(filepath):
                        size = os.path.getsize(filepath)
                        print(f'   ✓ {filename} ({size} bytes)')
                    else:
                        print(f'   ❌ {filename} (NOT FOUND)')
                
                return True
            else:
                print('❌ No output_files in results')
                print(f'Result keys: {list(results.keys())}')
                return False
        else:
            print('❌ No results returned')
            return False
        
    except Exception as e:
        print(f'❌ Test failed: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_minimal_processing()
    if success:
        print('\n✅ OUTPUT PERSISTENCE TEST PASSED!')
    else:
        print('\n❌ OUTPUT PERSISTENCE TEST FAILED!')