#!/usr/bin/env python3
"""
Comprehensive test script for the ensemble transcription pipeline.
Tests all components individually and the complete end-to-end integration.
"""

import os
import sys
import tempfile
import numpy as np
import json
import time
from typing import Dict, Any, List
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ensemble_manager import EnsembleManager
from core.audio_processor import AudioProcessor
from core.diarization_engine import DiarizationEngine
from core.asr_engine import ASREngine
from core.confidence_scorer import ConfidenceScorer
from utils.transcript_formatter import TranscriptFormatter
from utils.file_handler import FileHandler

class EnsembleTestSuite:
    """Comprehensive test suite for the ensemble transcription pipeline"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_files = []
        
    def create_mock_audio_file(self, duration_seconds: float = 30.0, sample_rate: int = 16000) -> str:
        """Create a mock WAV file for testing"""
        # Generate mock audio: sine wave with some noise
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        
        # Create a simple multi-frequency audio signal
        frequency1 = 440  # A4 note
        frequency2 = 880  # A5 note
        
        audio_data = (
            0.3 * np.sin(2 * np.pi * frequency1 * t) +
            0.2 * np.sin(2 * np.pi * frequency2 * t) +
            0.1 * np.random.normal(0, 0.1, len(t))
        )
        
        # Save as WAV file
        import soundfile as sf
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio_data, sample_rate)
        temp_file.close()
        
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def create_mock_video_file(self, duration_seconds: float = 30.0) -> str:
        """Create a mock MP4 file for testing using ffmpeg"""
        try:
            import ffmpeg
            
            # Create audio stream
            audio_file = self.create_mock_audio_file(duration_seconds)
            
            # Create a simple black video with the audio
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_video.close()
            
            (
                ffmpeg
                .input('color=c=black:s=640x480:r=1', f='lavfi', t=duration_seconds)
                .output(
                    temp_video.name,
                    **{'c:v': 'libx264', 'c:a': 'aac', 'shortest': None}
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Add audio track
            (
                ffmpeg
                .output(
                    ffmpeg.input(temp_video.name)['v'],
                    ffmpeg.input(audio_file)['a'],
                    temp_video.name + '_final.mp4',
                    **{'c:v': 'copy', 'c:a': 'aac'}
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Clean up intermediate files
            os.unlink(temp_video.name)
            final_video = temp_video.name + '_final.mp4'
            self.temp_files.append(final_video)
            
            return final_video
            
        except Exception as e:
            print(f"Warning: Could not create mock video file: {e}")
            # Fallback: return the audio file for testing
            return self.create_mock_audio_file(duration_seconds)
    
    def test_audio_processor(self) -> Dict[str, Any]:
        """Test AudioProcessor component"""
        print("🎵 Testing AudioProcessor...")
        
        try:
            processor = AudioProcessor()
            test_audio = self.create_mock_audio_file(30.0)
            
            # Test audio loading and preprocessing
            duration = processor.get_audio_duration(test_audio)
            noise_level = processor.estimate_noise_level(test_audio)
            
            # Test preprocessing
            import librosa
            audio_data, sr = librosa.load(test_audio, sr=16000)
            processed_audio = processor._preprocess_audio(audio_data, sr)
            
            result = {
                'status': 'success',
                'duration': duration,
                'noise_level': noise_level,
                'original_samples': len(audio_data),
                'processed_samples': len(processed_audio),
                'sample_rate': sr
            }
            
            print(f"  ✅ Audio duration: {duration:.2f}s")
            print(f"  ✅ Noise level: {noise_level}")
            print(f"  ✅ Preprocessing: {len(audio_data)} → {len(processed_audio)} samples")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"  ❌ AudioProcessor test failed: {e}")
        
        return result
    
    def test_diarization_engine(self) -> Dict[str, Any]:
        """Test DiarizationEngine component"""
        print("👥 Testing DiarizationEngine...")
        
        try:
            engine = DiarizationEngine(expected_speakers=3, noise_level='medium')
            test_audio = self.create_mock_audio_file(60.0)  # 1 minute
            
            # Test diarization variant creation
            variants = engine.create_diarization_variants(test_audio)
            
            result = {
                'status': 'success',
                'num_variants': len(variants),
                'variants': []
            }
            
            for i, variant in enumerate(variants):
                variant_info = {
                    'variant_id': variant['variant_id'],
                    'num_segments': len(variant['segments']),
                    'unique_speakers': len(set(seg['speaker'] for seg in variant['segments'])),
                    'total_duration': sum(seg['end'] - seg['start'] for seg in variant['segments'])
                }
                result['variants'].append(variant_info)
                
                print(f"  ✅ Variant {i+1}: {variant_info['num_segments']} segments, {variant_info['unique_speakers']} speakers")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"  ❌ DiarizationEngine test failed: {e}")
        
        return result
    
    def test_asr_engine(self) -> Dict[str, Any]:
        """Test ASREngine component with OpenAI Whisper"""
        print("🎤 Testing ASREngine...")
        
        try:
            engine = ASREngine()
            test_audio = self.create_mock_audio_file(10.0)  # Short audio for testing
            
            # Create mock diarization data
            mock_diarization = {
                'variant_id': 1,
                'segments': [
                    {'start': 0.0, 'end': 5.0, 'speaker': 'SPEAKER_00'},
                    {'start': 5.0, 'end': 10.0, 'speaker': 'SPEAKER_01'}
                ],
                'parameters': {'test': True}
            }
            
            # Test single ASR variant
            asr_config = {
                'variant_id': 1,
                'temperature': 0.0,
                'language': 'en',
                'prompt': "Test audio transcription.",
                'response_format': 'verbose_json'
            }
            
            asr_result = engine._run_asr_variant(test_audio, mock_diarization, asr_config)
            
            result = {
                'status': 'success',
                'asr_segments': len(asr_result.get('segments', [])),
                'has_words': bool(asr_result.get('words')),
                'language': asr_result.get('language', 'unknown'),
                'duration': asr_result.get('duration', 0)
            }
            
            print(f"  ✅ ASR completed: {result['asr_segments']} segments")
            print(f"  ✅ Language detected: {result['language']}")
            print(f"  ✅ Duration: {result['duration']:.2f}s")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"  ❌ ASREngine test failed: {e}")
        
        return result
    
    def create_mock_candidates(self) -> List[Dict[str, Any]]:
        """Create mock candidate data for testing confidence scoring"""
        candidates = []
        
        for diar_id in [1, 2, 3]:
            for asr_id in [1, 2, 3, 4, 5]:
                candidate = {
                    'candidate_id': f"diar_{diar_id}_asr_{asr_id}",
                    'diarization_variant_id': diar_id,
                    'asr_variant_id': asr_id,
                    'aligned_segments': [
                        {
                            'start': 0.0, 'end': 5.0, 'speaker': 'SPEAKER_00',
                            'text': 'Hello everyone, welcome to our meeting today.',
                            'words': [
                                {'start': 0.0, 'end': 0.5, 'word': 'Hello'},
                                {'start': 0.5, 'end': 1.0, 'word': 'everyone'},
                                {'start': 1.5, 'end': 2.0, 'word': 'welcome'},
                                {'start': 2.5, 'end': 2.8, 'word': 'to'},
                                {'start': 3.0, 'end': 3.3, 'word': 'our'},
                                {'start': 3.5, 'end': 4.0, 'word': 'meeting'},
                                {'start': 4.2, 'end': 4.5, 'word': 'today'}
                            ]
                        },
                        {
                            'start': 5.5, 'end': 10.0, 'speaker': 'SPEAKER_01',
                            'text': 'Thank you for that introduction. Let me share my screen.',
                            'words': [
                                {'start': 5.5, 'end': 6.0, 'word': 'Thank'},
                                {'start': 6.0, 'end': 6.2, 'word': 'you'},
                                {'start': 6.5, 'end': 6.8, 'word': 'for'},
                                {'start': 7.0, 'end': 7.3, 'word': 'that'},
                                {'start': 7.5, 'end': 8.2, 'word': 'introduction'},
                                {'start': 8.5, 'end': 8.8, 'word': 'Let'},
                                {'start': 9.0, 'end': 9.2, 'word': 'me'},
                                {'start': 9.3, 'end': 9.6, 'word': 'share'},
                                {'start': 9.7, 'end': 9.9, 'word': 'my'},
                                {'start': 9.9, 'end': 10.0, 'word': 'screen'}
                            ]
                        }
                    ],
                    'asr_data': {
                        'segments': [],
                        'language': 'en',
                        'duration': 10.0
                    },
                    'diarization_data': {
                        'variant_id': diar_id,
                        'segments': [
                            {'start': 0.0, 'end': 5.0, 'speaker': 'SPEAKER_00'},
                            {'start': 5.5, 'end': 10.0, 'speaker': 'SPEAKER_01'}
                        ]
                    }
                }
                
                # Add some variation to make scoring meaningful
                if asr_id == 5:  # Lower quality variant
                    candidate['aligned_segments'][0]['text'] = 'Hello everyone welcome to our meeting'  # Missing comma
                    candidate['aligned_segments'][1]['text'] = 'Thank you for that intro. Let me share screen.'  # Truncated
                
                candidates.append(candidate)
        
        return candidates
    
    def test_confidence_scorer(self) -> Dict[str, Any]:
        """Test ConfidenceScorer component"""
        print("🎯 Testing ConfidenceScorer...")
        
        try:
            scorer = ConfidenceScorer()
            candidates = self.create_mock_candidates()
            
            # Test scoring all candidates
            scored_candidates = scorer.score_all_candidates(candidates)
            
            # Test winner selection
            winner = scorer.select_winner(scored_candidates)
            
            result = {
                'status': 'success',
                'num_candidates': len(scored_candidates),
                'winner_id': winner['candidate_id'],
                'winner_score': winner['confidence_scores']['final_score'],
                'score_dimensions': list(winner['confidence_scores'].keys()),
                'score_breakdown': winner['confidence_scores']
            }
            
            print(f"  ✅ Scored {len(scored_candidates)} candidates")
            print(f"  ✅ Winner: {winner['candidate_id']} (score: {winner['confidence_scores']['final_score']:.3f})")
            print(f"  ✅ Score dimensions: {len(result['score_dimensions'])} metrics")
            
            # Display top 3 candidates
            sorted_candidates = sorted(scored_candidates, 
                                     key=lambda x: x['confidence_scores']['final_score'], 
                                     reverse=True)
            print("  📊 Top 3 candidates:")
            for i, candidate in enumerate(sorted_candidates[:3]):
                score = candidate['confidence_scores']['final_score']
                print(f"    {i+1}. {candidate['candidate_id']}: {score:.3f}")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"  ❌ ConfidenceScorer test failed: {e}")
        
        return result
    
    def test_transcript_formatter(self) -> Dict[str, Any]:
        """Test TranscriptFormatter component"""
        print("📄 Testing TranscriptFormatter...")
        
        try:
            formatter = TranscriptFormatter()
            
            # Create mock master transcript
            mock_transcript = {
                'segments': [
                    {
                        'start': 0.0, 'end': 5.0, 'speaker': 'Alice',
                        'text': 'Hello everyone, welcome to our meeting today.'
                    },
                    {
                        'start': 5.5, 'end': 10.0, 'speaker': 'Bob',
                        'text': 'Thank you for that introduction. Let me share my screen.'
                    }
                ],
                'speaker_map': {
                    'SPEAKER_00': 'Alice',
                    'SPEAKER_01': 'Bob'
                },
                'metadata': {
                    'total_duration': 10.0,
                    'total_segments': 2,
                    'speaker_count': 2,
                    'confidence_summary': {
                        'final_score': 0.875,
                        'D_diarization': 0.92,
                        'A_asr_alignment': 0.88,
                        'L_linguistic': 0.85,
                        'R_agreement': 0.80,
                        'O_overlap': 0.90
                    }
                }
            }
            
            # Test all format generations
            txt_transcript = formatter.create_txt_transcript(mock_transcript)
            vtt_captions = formatter.create_vtt_captions(mock_transcript)
            srt_captions = formatter.create_srt_captions(mock_transcript)
            ass_captions = formatter.create_ass_captions(mock_transcript)
            
            result = {
                'status': 'success',
                'txt_length': len(txt_transcript),
                'vtt_length': len(vtt_captions),
                'srt_length': len(srt_captions),
                'ass_length': len(ass_captions),
                'formats_generated': 4
            }
            
            print(f"  ✅ TXT transcript: {result['txt_length']} characters")
            print(f"  ✅ VTT captions: {result['vtt_length']} characters")
            print(f"  ✅ SRT captions: {result['srt_length']} characters")
            print(f"  ✅ ASS captions: {result['ass_length']} characters")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"  ❌ TranscriptFormatter test failed: {e}")
        
        return result
    
    def test_ensemble_manager(self) -> Dict[str, Any]:
        """Test complete EnsembleManager pipeline"""
        print("🎯 Testing Complete EnsembleManager Pipeline...")
        
        try:
            # Create mock video file
            test_video = self.create_mock_video_file(15.0)  # 15 seconds for faster testing
            
            # Initialize ensemble manager
            ensemble_manager = EnsembleManager(
                expected_speakers=3,
                noise_level='medium'
            )
            
            # Track progress
            progress_steps = []
            def progress_callback(step, progress, message):
                progress_steps.append({
                    'step': step,
                    'progress': progress,
                    'message': message,
                    'timestamp': time.time()
                })
                print(f"  📊 Step {step} ({progress}%): {message}")
            
            # Run complete pipeline
            start_time = time.time()
            results = ensemble_manager.process_video(test_video, progress_callback)
            end_time = time.time()
            
            result = {
                'status': 'success',
                'processing_time': end_time - start_time,
                'progress_steps': len(progress_steps),
                'results_keys': list(results.keys()),
                'winner_score': results.get('winner_score', 0),
                'detected_speakers': results.get('detected_speakers', 0),
                'transcript_preview_segments': len(results.get('transcript_preview', [])),
                'ensemble_audit_candidates': len(results.get('ensemble_audit', {}).get('all_candidates', [])),
                'output_formats': {
                    'json': bool(results.get('winner_transcript')),
                    'txt': bool(results.get('winner_transcript_txt')),
                    'vtt': bool(results.get('captions_vtt')),
                    'srt': bool(results.get('captions_srt')),
                    'ass': bool(results.get('captions_ass'))
                }
            }
            
            print(f"  ✅ Pipeline completed in {result['processing_time']:.2f}s")
            print(f"  ✅ Winner score: {result['winner_score']:.3f}")
            print(f"  ✅ Detected speakers: {result['detected_speakers']}")
            print(f"  ✅ Output formats: {sum(result['output_formats'].values())}/5")
            print(f"  ✅ Audit candidates: {result['ensemble_audit_candidates']}")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"  ❌ EnsembleManager test failed: {e}")
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        print("🚀 Starting Ensemble Transcription Pipeline Test Suite")
        print("=" * 60)
        
        # Run individual component tests
        self.test_results['audio_processor'] = self.test_audio_processor()
        self.test_results['diarization_engine'] = self.test_diarization_engine()
        self.test_results['asr_engine'] = self.test_asr_engine()
        self.test_results['confidence_scorer'] = self.test_confidence_scorer()
        self.test_results['transcript_formatter'] = self.test_transcript_formatter()
        
        # Run complete pipeline test
        self.test_results['ensemble_manager'] = self.test_ensemble_manager()
        
        # Generate summary
        self.test_results['summary'] = self.generate_test_summary()
        
        return self.test_results
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate test summary and statistics"""
        total_tests = len(self.test_results) - 1  # Exclude summary itself
        passed_tests = sum(1 for key, result in self.test_results.items() 
                          if key != 'summary' and result.get('status') == 'success')
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'test_details': {}
        }
        
        for test_name, result in self.test_results.items():
            if test_name != 'summary':
                summary['test_details'][test_name] = {
                    'status': result.get('status', 'unknown'),
                    'error': result.get('error') if result.get('status') == 'error' else None
                }
        
        return summary
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not clean up {temp_file}: {e}")
    
    def print_final_report(self):
        """Print final test report"""
        summary = self.test_results.get('summary', {})
        
        print("\n" + "=" * 60)
        print("🏁 FINAL TEST REPORT")
        print("=" * 60)
        
        print(f"📊 Tests Run: {summary.get('total_tests', 0)}")
        print(f"✅ Passed: {summary.get('passed_tests', 0)}")
        print(f"❌ Failed: {summary.get('failed_tests', 0)}")
        print(f"📈 Success Rate: {summary.get('success_rate', 0):.1%}")
        
        print("\n📋 Test Details:")
        for test_name, details in summary.get('test_details', {}).items():
            status_emoji = "✅" if details['status'] == 'success' else "❌"
            print(f"  {status_emoji} {test_name}: {details['status']}")
            if details.get('error'):
                print(f"    Error: {details['error']}")
        
        print("\n" + "=" * 60)

def main():
    """Main test execution function"""
    test_suite = EnsembleTestSuite()
    
    try:
        # Run all tests
        results = test_suite.run_all_tests()
        
        # Print final report
        test_suite.print_final_report()
        
        # Save results to file
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: test_results.json")
        
        # Return exit code based on success
        summary = results.get('summary', {})
        if summary.get('success_rate', 0) == 1.0:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed. Check the report above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite execution failed: {e}")
        print(traceback.format_exc())
        return 1
        
    finally:
        # Always clean up
        test_suite.cleanup()

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)