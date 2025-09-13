import os
import json
import tempfile
import random
from typing import List, Dict, Any, Optional
from openai import OpenAI
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class ASREngine:
    """Handles Automatic Speech Recognition with ensemble variants"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "whisper-1"  # Whisper model for transcription
    
    def run_asr_ensemble(self, audio_path: str, diarization_variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run 5 ASR variants for each of the 3 diarization results (15 total candidates).
        
        Args:
            audio_path: Path to cleaned audio file
            diarization_variants: List of 3 diarization variant results
            
        Returns:
            List of 15 candidate transcripts with ASR and diarization data
        """
        print(f"🎤 Starting ASR ensemble processing for {len(diarization_variants)} diarization variants...")
        candidates = []
        
        # For each diarization variant, run 5 ASR passes
        for i, diar_variant in enumerate(diarization_variants, 1):
            print(f"Processing diarization variant {i}/{len(diarization_variants)}...")
            try:
                asr_variants = self._create_asr_variants(audio_path, diar_variant)
                candidates.extend(asr_variants)
                print(f"✓ Completed diarization variant {i}: {len(asr_variants)} ASR candidates generated")
            except Exception as e:
                print(f"⚠ Error processing diarization variant {i}: {e}")
                continue
        
        print(f"🎤 ASR ensemble complete: {len(candidates)} total candidates generated")
        return candidates
    
    def _create_asr_variants(self, audio_path: str, diarization_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create 5 ASR variants for a single diarization result.
        
        Args:
            audio_path: Path to audio file
            diarization_data: Single diarization variant data
            
        Returns:
            List of 5 ASR variant results
        """
        variants = []
        
        # Define 5 different ASR parameter sets
        asr_configs = [
            {
                'variant_id': 1,
                'temperature': 0.0,
                'language': 'en',
                'prompt': "The following is a recording of a meeting with multiple speakers.",
                'response_format': 'verbose_json'
            },
            {
                'variant_id': 2, 
                'temperature': 0.1,
                'language': 'en',
                'prompt': "This is a multi-speaker conversation with clear speaker changes.",
                'response_format': 'verbose_json'
            },
            {
                'variant_id': 3,
                'temperature': 0.2,
                'language': 'en', 
                'prompt': "Transcribe this meeting recording with multiple participants.",
                'response_format': 'verbose_json'
            },
            {
                'variant_id': 4,
                'temperature': 0.0,
                'language': None,  # Let Whisper auto-detect
                'prompt': "",
                'response_format': 'verbose_json'
            },
            {
                'variant_id': 5,
                'temperature': 0.3,
                'language': 'en',
                'prompt': "This recording contains multiple speakers in a meeting discussion.",
                'response_format': 'verbose_json'
            }
        ]
        
        # Run each ASR variant with progress tracking
        for j, config in enumerate(asr_configs, 1):
            try:
                print(f"  Running ASR variant {j}/5 (temp={config['temperature']})...")
                start_time = time.time()
                asr_result = self._run_asr_variant(audio_path, diarization_data, config)
                elapsed = time.time() - start_time
                
                # Combine diarization and ASR data
                candidate = {
                    'candidate_id': f"diar_{diarization_data['variant_id']}_asr_{config['variant_id']}",
                    'diarization_variant_id': diarization_data['variant_id'],
                    'asr_variant_id': config['variant_id'],
                    'diarization_data': diarization_data,
                    'asr_data': asr_result,
                    'aligned_segments': self._align_asr_to_diarization(asr_result, diarization_data),
                    'parameters': {
                        'diarization': diarization_data['parameters'],
                        'asr': config
                    }
                }
                
                variants.append(candidate)
                print(f"  ✓ ASR variant {j} completed in {elapsed:.1f}s")
                
            except Exception as e:
                print(f"  ⚠ ASR variant {j} failed: {e}")
                # Continue with other variants
                continue
        
        return variants
    
    def _run_asr_variant(self, audio_path: str, diarization_data: Dict[str, Any], 
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single ASR variant with specific parameters.
        
        Args:
            audio_path: Path to audio file
            diarization_data: Diarization results
            config: ASR configuration parameters
            
        Returns:
            ASR transcription results
        """
        try:
            # Prepare parameters for OpenAI Whisper
            transcription_params = {
                'model': self.model,
                'response_format': config['response_format'],
                'temperature': config['temperature']
            }
            
            # Add optional parameters
            if config['language']:
                transcription_params['language'] = config['language']
            if config['prompt']:
                transcription_params['prompt'] = config['prompt']
            
            # Open and transcribe audio file with timeout handling
            with open(audio_path, 'rb') as audio_file:
                try:
                    response = self.client.audio.transcriptions.create(
                        file=audio_file,
                        timeout=60.0,  # 60 second timeout
                        **transcription_params
                    )
                except Exception as api_error:
                    print(f"OpenAI API error: {api_error}")
                    raise Exception(f"OpenAI Whisper API failed: {str(api_error)}")
            
            # Extract word-level timestamps if available
            if hasattr(response, 'words') and response.words:
                words = [
                    {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'confidence': getattr(word, 'confidence', 0.9)  # Default confidence
                    }
                    for word in response.words
                ]
            else:
                # Fallback: create mock word timestamps from segments
                words = self._create_word_timestamps_from_segments(response)
            
            # Extract segments
            segments = []
            if hasattr(response, 'segments') and response.segments:
                segments = [
                    {
                        'id': seg.id,
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text,
                        'confidence': getattr(seg, 'avg_logprob', 0.0)  # Convert log prob to confidence
                    }
                    for seg in response.segments
                ]
            
            return {
                'variant_id': config['variant_id'],
                'text': response.text,
                'words': words,
                'segments': segments,
                'language': getattr(response, 'language', 'en'),
                'duration': getattr(response, 'duration', 0.0),
                'confidence_scores': self._calculate_confidence_metrics(words, segments)
            }
            
        except Exception as e:
            raise Exception(f"ASR transcription failed: {str(e)}")
    
    def _create_word_timestamps_from_segments(self, response) -> List[Dict[str, Any]]:
        """
        Create word-level timestamps from segment-level data when not available.
        
        Args:
            response: OpenAI Whisper response object
            
        Returns:
            List of word dictionaries with estimated timestamps
        """
        words = []
        
        if hasattr(response, 'segments') and response.segments:
            for segment in response.segments:
                segment_words = segment.text.split()
                segment_duration = segment.end - segment.start
                word_duration = segment_duration / max(len(segment_words), 1)
                
                for i, word in enumerate(segment_words):
                    word_start = segment.start + (i * word_duration)
                    word_end = word_start + word_duration
                    
                    words.append({
                        'word': word,
                        'start': word_start,
                        'end': word_end,
                        'confidence': max(0.1, getattr(segment, 'avg_logprob', -1.0) + 1.0)  # Convert log prob
                    })
        
        return words
    
    def _calculate_confidence_metrics(self, words: List[Dict[str, Any]], 
                                    segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate confidence metrics for ASR results.
        
        Args:
            words: List of word-level results
            segments: List of segment-level results
            
        Returns:
            Dictionary of confidence metrics
        """
        if not words:
            return {'word_confidence_mean': 0.0, 'segment_confidence_mean': 0.0}
        
        # Word-level confidence
        word_confidences = [w['confidence'] for w in words if 'confidence' in w]
        word_confidence_mean = np.mean(word_confidences) if word_confidences else 0.0
        
        # Segment-level confidence  
        segment_confidences = [s['confidence'] for s in segments if 'confidence' in s]
        segment_confidence_mean = np.mean(segment_confidences) if segment_confidences else 0.0
        
        return {
            'word_confidence_mean': float(word_confidence_mean),
            'segment_confidence_mean': float(segment_confidence_mean),
            'word_count': len(words),
            'segment_count': len(segments)
        }
    
    def _align_asr_to_diarization(self, asr_data: Dict[str, Any], 
                                 diarization_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Align ASR word timestamps to diarization speaker segments.
        
        Args:
            asr_data: ASR transcription results
            diarization_data: Speaker diarization results
            
        Returns:
            List of aligned segments with speaker and word information
        """
        aligned_segments = []
        diar_segments = diarization_data['segments']
        words = asr_data.get('words', [])
        
        if not words or not diar_segments:
            return aligned_segments
        
        # Sort both by start time
        diar_segments = sorted(diar_segments, key=lambda x: x['start'])
        words = sorted(words, key=lambda x: x['start'])
        
        current_diar_idx = 0
        current_segment = None
        current_words = []
        
        for word in words:
            word_start = word['start']
            word_end = word['end']
            
            # Find the appropriate diarization segment for this word
            while (current_diar_idx < len(diar_segments) and 
                   diar_segments[current_diar_idx]['end'] < word_start):
                # Finish current segment if it has words
                if current_segment and current_words:
                    aligned_segments.append(self._create_aligned_segment(current_segment, current_words))
                
                current_diar_idx += 1
                current_words = []
                current_segment = None
            
            # Check if word fits in current diarization segment
            if (current_diar_idx < len(diar_segments) and 
                diar_segments[current_diar_idx]['start'] <= word_start <= diar_segments[current_diar_idx]['end']):
                
                # Start new segment if speaker changed
                if current_segment is None or current_segment['speaker_id'] != diar_segments[current_diar_idx]['speaker_id']:
                    # Finish previous segment
                    if current_segment and current_words:
                        aligned_segments.append(self._create_aligned_segment(current_segment, current_words))
                    
                    # Start new segment
                    current_segment = diar_segments[current_diar_idx]
                    current_words = []
                
                current_words.append(word)
        
        # Add final segment
        if current_segment and current_words:
            aligned_segments.append(self._create_aligned_segment(current_segment, current_words))
        
        return aligned_segments
    
    def _create_aligned_segment(self, diar_segment: Dict[str, Any], 
                              words: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create an aligned segment combining diarization and ASR data.
        
        Args:
            diar_segment: Diarization segment data
            words: List of words in this segment
            
        Returns:
            Aligned segment dictionary
        """
        if not words:
            return {}
        
        # Calculate segment boundaries from words
        segment_start = min(word['start'] for word in words)
        segment_end = max(word['end'] for word in words)
        
        # Join words into text
        segment_text = ' '.join(word['word'] for word in words)
        
        # Calculate average confidence
        word_confidences = [word.get('confidence', 0.0) for word in words]
        avg_confidence = np.mean(word_confidences) if word_confidences else 0.0
        
        return {
            'start': segment_start,
            'end': segment_end,
            'speaker_id': diar_segment['speaker_id'],
            'text': segment_text,
            'words': words,
            'confidence': float(avg_confidence),
            'word_count': len(words),
            'diarization_confidence': diar_segment.get('confidence', 0.0)
        }
