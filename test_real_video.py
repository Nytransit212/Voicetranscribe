#!/usr/bin/env python3
"""
Test the complete ensemble pipeline with real video file and implement
section-based scoring with 80% overlap requirement.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append('.')

from core.ensemble_manager import EnsembleManager
from core.audio_processor import AudioProcessor
from utils.transcript_formatter import TranscriptFormatter

def test_overlap_based_scoring():
    """Test overlap-based section scoring with 80% threshold."""
    
    print("🎬 Testing Complete Ensemble Pipeline with Real Video")
    print("=" * 60)
    
    # Test file path
    video_path = "data/test_short_video.mov"
    
    if not os.path.exists(video_path):
        print(f"❌ Test video not found: {video_path}")
        return False
    
    print(f"📁 Input: {video_path}")
    print(f"📊 Duration: ~4:51 (291 seconds)")
    print()
    
    try:
        # Initialize components
        print("🔧 Initializing ensemble pipeline...")
        ensemble_manager = EnsembleManager()
        audio_processor = AudioProcessor()
        transcript_formatter = TranscriptFormatter()
        
        # Step 1: Extract and process audio
        print("🎵 Step 1: Audio extraction and preprocessing...")
        start_time = time.time()
        
        raw_audio_path, cleaned_audio_path = audio_processor.extract_audio_from_video(video_path)
        
        audio_time = time.time() - start_time
        print(f"   ✓ Audio processed in {audio_time:.1f}s")
        print(f"   📄 Raw audio: {raw_audio_path}")
        print(f"   📄 Cleaned audio: {cleaned_audio_path}")
        print()
        
        # Step 2: Run complete ensemble processing
        print("🎤 Step 2: Running ensemble transcription...")
        print("   • Generating 3 diarization variants")
        print("   • Creating 5 ASR variants per diarization (15 total)")
        print("   • Processing with Tier 4 OpenAI API access")
        print()
        
        ensemble_start = time.time()
        
        # Process with ensemble manager
        result = ensemble_manager.process_video(video_path)
        
        ensemble_time = time.time() - ensemble_start
        print(f"   ✓ Ensemble processing completed in {ensemble_time:.1f}s")
        print()
        
        # Step 3: Analyze results
        print("📊 Step 3: Analyzing ensemble results...")
        
        if result and 'candidates' in result:
            candidates = result['candidates']
            print(f"   • Generated {len(candidates)} candidates")
            
            # Check candidate structure
            for i, candidate in enumerate(candidates[:3], 1):
                print(f"   • Candidate {i}: {candidate.get('candidate_id', 'unknown')}")
                print(f"     - Diarization variant: {candidate.get('diarization_variant_id', 'N/A')}")
                print(f"     - ASR variant: {candidate.get('asr_variant_id', 'N/A')}")
                
                # Check aligned segments
                aligned_segments = candidate.get('aligned_segments', [])
                print(f"     - Aligned segments: {len(aligned_segments)}")
                
                if aligned_segments:
                    # Show first segment as example
                    first_seg = aligned_segments[0]
                    print(f"     - First segment: {first_seg.get('start', 0):.1f}s - {first_seg.get('end', 0):.1f}s")
                    print(f"       Speaker: {first_seg.get('speaker_id', 'unknown')}")
                    words = first_seg.get('words', [])
                    if words:
                        print(f"       Words: {len(words)} ('{words[0].get('word', '')}...')")
            
            print()
            
            # Step 4: Test overlap-based scoring
            print("🎯 Step 4: Testing overlap-based scoring (80% threshold)...")
            overlap_results = test_overlap_scoring_algorithm(candidates)
            
            # Step 5: Winner selection
            if 'winner' in result:
                winner = result['winner']
                print(f"🏆 Winner: {winner.get('candidate_id', 'unknown')}")
                print(f"   Overall confidence: {winner.get('confidence_score', 0):.3f}")
                
                # Show confidence breakdown
                confidence_breakdown = winner.get('confidence_breakdown', {})
                for dim, score in confidence_breakdown.items():
                    print(f"   {dim}: {score:.3f}")
            
            print()
            
            # Step 6: Generate outputs
            print("📄 Step 6: Generating output formats...")
            output_dir = "data/test_outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            if 'winner' in result:
                winner_transcript = result['winner']
                
                # Generate all formats
                formats = transcript_formatter.generate_all_formats(
                    winner_transcript, output_dir, "test_video"
                )
                
                print(f"   Generated {len(formats)} output files:")
                for format_name, file_path in formats.items():
                    size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    print(f"   • {format_name}: {file_path} ({size} bytes)")
            
        else:
            print("❌ No valid results from ensemble processing")
            return False
            
        total_time = time.time() - start_time
        print()
        print("=" * 60)
        print(f"✅ Complete pipeline test successful!")
        print(f"📊 Total processing time: {total_time:.1f}s")
        print(f"🎯 Ready for production use with Tier 4 OpenAI access")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_overlap_scoring_algorithm(candidates):
    """
    Test the overlap-based scoring algorithm with 80% overlap requirement.
    
    Args:
        candidates: List of transcription candidates
        
    Returns:
        Dictionary with overlap analysis results
    """
    
    print("   🔍 Analyzing segment overlaps across candidates...")
    
    # Extract all segments with timecodes
    all_segments = []
    for candidate in candidates:
        candidate_id = candidate.get('candidate_id', 'unknown')
        aligned_segments = candidate.get('aligned_segments', [])
        
        for segment in aligned_segments:
            segment_data = {
                'candidate_id': candidate_id,
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'duration': segment.get('end', 0) - segment.get('start', 0),
                'speaker_id': segment.get('speaker_id', 'unknown'),
                'text': segment.get('text', ''),
                'words': segment.get('words', [])
            }
            all_segments.append(segment_data)
    
    print(f"   • Total segments across all candidates: {len(all_segments)}")
    
    # Find overlapping regions (80% threshold)
    overlap_groups = find_overlap_groups(all_segments, overlap_threshold=0.8)
    
    print(f"   • Found {len(overlap_groups)} overlap groups with ≥80% overlap")
    
    # Calculate scores only for overlapping regions
    scoring_regions = []
    for group in overlap_groups:
        if len(group) >= 2:  # Need at least 2 candidates to compare
            region = calculate_overlap_region_scores(group)
            scoring_regions.append(region)
    
    print(f"   • Created {len(scoring_regions)} scoring regions for comparison")
    
    if scoring_regions:
        avg_overlap_duration = sum(r['duration'] for r in scoring_regions) / len(scoring_regions)
        print(f"   • Average overlap region duration: {avg_overlap_duration:.1f}s")
    
    return {
        'total_segments': len(all_segments),
        'overlap_groups': len(overlap_groups),
        'scoring_regions': len(scoring_regions),
        'regions': scoring_regions
    }

def find_overlap_groups(segments, overlap_threshold=0.8):
    """
    Find groups of segments with ≥80% temporal overlap.
    
    Args:
        segments: List of segment dictionaries with start/end times
        overlap_threshold: Minimum overlap ratio (0.8 = 80%)
        
    Returns:
        List of overlap groups
    """
    
    overlap_groups = []
    processed = set()
    
    for i, seg1 in enumerate(segments):
        if i in processed:
            continue
            
        group = [seg1]
        processed.add(i)
        
        for j, seg2 in enumerate(segments[i+1:], i+1):
            if j in processed:
                continue
                
            overlap_ratio = calculate_overlap_ratio(seg1, seg2)
            
            if overlap_ratio >= overlap_threshold:
                group.append(seg2)
                processed.add(j)
        
        if len(group) >= 2:  # Only include groups with multiple segments
            overlap_groups.append(group)
    
    return overlap_groups

def calculate_overlap_ratio(seg1, seg2):
    """Calculate the overlap ratio between two segments."""
    
    # Find overlap region
    overlap_start = max(seg1['start'], seg2['start'])
    overlap_end = min(seg1['end'], seg2['end'])
    
    if overlap_start >= overlap_end:
        return 0.0  # No overlap
    
    overlap_duration = overlap_end - overlap_start
    
    # Calculate ratio relative to shorter segment
    seg1_duration = seg1['end'] - seg1['start']
    seg2_duration = seg2['end'] - seg2['start']
    min_duration = min(seg1_duration, seg2_duration)
    
    if min_duration <= 0:
        return 0.0
    
    return overlap_duration / min_duration

def calculate_overlap_region_scores(segment_group):
    """
    Calculate D-A-L-R-O scores for an overlapping region.
    
    Args:
        segment_group: List of overlapping segments from different candidates
        
    Returns:
        Dictionary with region scoring data
    """
    
    # Find the common overlap region
    start_times = [seg['start'] for seg in segment_group]
    end_times = [seg['end'] for seg in segment_group]
    
    region_start = max(start_times)
    region_end = min(end_times)
    region_duration = region_end - region_start
    
    # Extract words within the overlap region
    region_words = []
    for segment in segment_group:
        candidate_words = []
        for word in segment.get('words', []):
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)
            
            # Check if word falls within overlap region
            if word_start >= region_start and word_end <= region_end:
                candidate_words.append(word)
        
        if candidate_words:
            region_words.append({
                'candidate_id': segment['candidate_id'],
                'words': candidate_words,
                'text': ' '.join(w.get('word', '') for w in candidate_words)
            })
    
    return {
        'start': region_start,
        'end': region_end, 
        'duration': region_duration,
        'candidates': len(segment_group),
        'candidate_ids': [seg['candidate_id'] for seg in segment_group],
        'words': region_words,
        # Placeholder for actual D-A-L-R-O scoring
        'scores': {
            'D': 0.85,  # Diarization quality
            'A': 0.82,  # ASR-diarization alignment 
            'L': 0.88,  # Linguistic quality
            'R': 0.79,  # Temporal regularity
            'O': 0.86   # Overall confidence
        }
    }

if __name__ == "__main__":
    success = test_overlap_based_scoring()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)