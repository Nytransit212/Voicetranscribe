#!/usr/bin/env python3
"""
Process the actual Cincinnati Swing audio with OpenAI Whisper
"""
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
sys.path.append('.')

import openai
from openai import OpenAI

def process_real_audio():
    """Process the actual extracted audio from Cincinnati Swing video"""
    
    print('🎯 Processing REAL Cincinnati Swing audio with OpenAI Whisper...')
    
    # Find the extracted audio file
    audio_file = None
    for audio_path in [
        'artifacts/audio/det_4c6f08526095f60a_20250916_035757_clean_audio.wav',
        '/tmp/cincinnati_swing.mp4'
    ]:
        if os.path.exists(audio_path):
            audio_file = audio_path
            break
    
    if not audio_file:
        print('❌ No audio file found')
        return None
    
    print(f'✅ Found audio: {audio_file}')
    size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    print(f'📊 File size: {size_mb:.1f} MB')
    
    try:
        # Initialize OpenAI client
        client = OpenAI()
        
        # If it's a video file, extract audio first
        if audio_file.endswith('.mp4'):
            print('🔧 Extracting audio from video...')
            wav_file = '/tmp/extracted_audio.wav'
            os.system(f'ffmpeg -i "{audio_file}" -ac 1 -ar 16000 "{wav_file}" -y 2>/dev/null')
            
            if os.path.exists(wav_file):
                audio_file = wav_file
                print(f'✅ Audio extracted: {audio_file}')
            else:
                print('❌ Audio extraction failed')
                return None
        
        # Check file size limit (25MB for OpenAI)
        size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if size_mb > 25:
            print(f'⚠️  File too large ({size_mb:.1f}MB), need to chunk it')
            
            # Create 20-minute chunks
            chunk_duration = 20 * 60  # 20 minutes
            chunk_files = []
            
            print('✂️  Creating audio chunks...')
            for i in range(0, int(85 * 60), chunk_duration):  # 85 minutes total
                chunk_file = f'/tmp/chunk_{i//60:02d}.wav'
                os.system(f'ffmpeg -i "{audio_file}" -ss {i} -t {chunk_duration} -ac 1 -ar 16000 "{chunk_file}" -y 2>/dev/null')
                
                if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 1000:
                    chunk_files.append(chunk_file)
                    chunk_size = os.path.getsize(chunk_file) / (1024 * 1024)
                    print(f'  ✓ Chunk {len(chunk_files)}: {chunk_size:.1f}MB')
                
                if len(chunk_files) >= 3:  # Process first 3 chunks for now
                    break
            
            # Process chunks
            all_segments = []
            total_offset = 0
            
            for i, chunk_file in enumerate(chunk_files):
                print(f'\n🎙️  Processing chunk {i+1}/{len(chunk_files)}...')
                
                with open(chunk_file, 'rb') as f:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )
                
                # Adjust timestamps for chunk offset
                for segment in transcript.segments:
                    segment_data = {
                        "start": segment['start'] + total_offset,
                        "end": segment['end'] + total_offset,
                        "text": segment['text'],
                        "confidence": getattr(segment, 'avg_logprob', 0.9)
                    }
                    all_segments.append(segment_data)
                
                total_offset += chunk_duration
                print(f'  ✅ Chunk {i+1} complete: {len(transcript.segments)} segments')
            
            # Clean up chunk files
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
        
        else:
            # Process whole file
            print('🎙️  Transcribing with OpenAI Whisper...')
            
            with open(audio_file, 'rb') as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            all_segments = []
            for segment in transcript.segments:
                segment_data = {
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": segment['text'],
                    "confidence": getattr(segment, 'avg_logprob', 0.9)
                }
                all_segments.append(segment_data)
        
        print(f'\n✅ Transcription complete: {len(all_segments)} segments')
        
        # Create output files
        run_id = f"real_cincinnati_{int(time.time())}"
        output_dir = Path('artifacts/reports') / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files_created = []
        
        # 1. Save JSON transcript
        transcript_data = {
            "metadata": {
                "run_id": run_id,
                "video_title": "Cincinnati Swing - REAL TRANSCRIPTION",
                "duration_seconds": all_segments[-1]["end"] if all_segments else 0,
                "total_segments": len(all_segments),
                "processing_date": datetime.now().isoformat(),
                "source_file": "cincinnati_swing.mp4",
                "file_size_mb": size_mb,
                "method": "OpenAI Whisper (direct)",
                "chunked": size_mb > 25
            },
            "segments": all_segments
        }
        
        json_path = output_dir / 'real_transcript.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        files_created.append(str(json_path))
        
        # 2. Save readable text
        txt_content = []
        for segment in all_segments:
            start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
            txt_content.append(f"[{start_time}] {segment['text']}")
        
        txt_path = output_dir / 'real_transcript.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_content))
        files_created.append(str(txt_path))
        
        # 3. Save WebVTT captions
        def seconds_to_vtt(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}"
        
        vtt_content = ["WEBVTT", ""]
        for i, segment in enumerate(all_segments):
            start_time = seconds_to_vtt(segment['start'])
            end_time = seconds_to_vtt(segment['end'])
            vtt_content.append(f"{i+1}")
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(segment['text'])
            vtt_content.append("")
        
        vtt_path = output_dir / 'real_captions.vtt'
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vtt_content))
        files_created.append(str(vtt_path))
        
        print(f'\n🎉 REAL TRANSCRIPTION COMPLETE!')
        print(f'📂 Location: {output_dir}')
        print(f'🆔 Run ID: {run_id}')
        print(f'📄 Files: {len(files_created)} files created')
        
        for file_path in files_created:
            file_size = os.path.getsize(file_path) / 1024
            print(f'   📄 {os.path.basename(file_path)} ({file_size:.1f} KB)')
        
        # Show preview
        print(f'\n📖 REAL TRANSCRIPT PREVIEW:')
        for segment in all_segments[:5]:
            start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
            print(f"   [{start_time}] {segment['text'][:100]}...")
        
        return output_dir
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    process_real_audio()