import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, List, Optional, Tuple
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.asr_engine import ASREngine
from core.diarization_engine import DiarizationEngine
from core.confidence_scorer import ConfidenceScorer
from utils.structured_logger import StructuredLogger

class RepairEngine:
    """Handles targeted reprocessing and repair of transcript segments"""
    
    def __init__(self):
        self.asr_engine = ASREngine()
        self.diarization_engine = DiarizationEngine()
        self.confidence_scorer = ConfidenceScorer()
        
        # Initialize structured logging
        self.structured_logger = StructuredLogger("repair_engine")
        
        # Repair-specific ASR configurations for different problem types
        self.repair_configs = {
            'low_confidence': [
                {
                    'temperature': 0.0,
                    'language': 'en',
                    'prompt': "This is a high-quality recording. Please transcribe carefully with attention to detail.",
                    'response_format': 'verbose_json'
                },
                {
                    'temperature': 0.05,
                    'language': 'en', 
                    'prompt': "Transcribe this clear audio segment with precise word timing.",
                    'response_format': 'verbose_json'
                }
            ],
            'unclear_audio': [
                {
                    'temperature': 0.1,
                    'language': 'en',
                    'prompt': "This audio may have background noise. Focus on speech content.",
                    'response_format': 'verbose_json'
                },
                {
                    'temperature': 0.2,
                    'language': 'en',
                    'prompt': "Transcribe speech from noisy audio, ignore background sounds.",
                    'response_format': 'verbose_json'
                }
            ],
            'speaker_confusion': [
                {
                    'temperature': 0.0,
                    'language': 'en',
                    'prompt': "This is a conversation between multiple speakers. Pay attention to speaker changes.",
                    'response_format': 'verbose_json'
                },
                {
                    'temperature': 0.1,
                    'language': None,  # Auto-detect
                    'prompt': "Multi-speaker conversation with clear turn-taking.",
                    'response_format': 'verbose_json'
                }
            ]
        }
    
    def repair_segment(self, audio_path: str, segment: Dict[str, Any], 
                      problem_type: str = 'low_confidence',
                      context_segments: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Repair a specific segment using targeted reprocessing.
        
        Args:
            audio_path: Path to original audio file
            segment: Segment to repair
            problem_type: Type of problem (low_confidence, unclear_audio, speaker_confusion)
            context_segments: Surrounding segments for context
            
        Returns:
            List of repair candidates for the segment
        """
        print(f"🔧 Repairing segment [{segment['start']:.1f}-{segment['end']:.1f}s] for {problem_type}...")
        
        try:
            # Extract segment audio
            segment_audio_path = self._extract_segment_audio(
                audio_path, segment['start'], segment['end']
            )
            
            # Get appropriate repair configs
            configs = self.repair_configs.get(problem_type, self.repair_configs['low_confidence'])
            
            # Generate repair candidates
            repair_candidates = []
            
            for i, config in enumerate(configs):
                try:
                    print(f"  Generating repair candidate {i+1}/{len(configs)}...")
                    
                    # Add context to prompt if available
                    enhanced_config = self._enhance_config_with_context(config, context_segments)
                    
                    # Run ASR with repair-specific configuration
                    asr_result = self.asr_engine._run_asr_variant(
                        segment_audio_path, 
                        {'segments': [segment]}, 
                        enhanced_config
                    )
                    
                    # Create repair candidate
                    repair_candidate = {
                        'repair_id': f"repair_{problem_type}_{i+1}",
                        'original_segment': segment,
                        'repaired_segment': self._create_repaired_segment(segment, asr_result),
                        'repair_config': enhanced_config,
                        'problem_type': problem_type,
                        'improvement_score': self._calculate_improvement_score(segment, asr_result),
                        'repair_confidence': self._calculate_repair_confidence(asr_result, problem_type)
                    }
                    
                    repair_candidates.append(repair_candidate)
                    print(f"  ✓ Repair candidate {i+1} generated (confidence: {repair_candidate['repair_confidence']:.3f})")
                    
                except Exception as e:
                    print(f"  ⚠ Repair candidate {i+1} failed: {e}")
                    continue
            
            # Clean up temporary audio file
            if os.path.exists(segment_audio_path):
                os.unlink(segment_audio_path)
            
            print(f"🔧 Segment repair complete: {len(repair_candidates)} candidates generated")
            return repair_candidates
            
        except Exception as e:
            print(f"❌ Segment repair failed: {e}")
            return []
    
    def batch_repair_segments(self, audio_path: str, segments_to_repair: List[Dict[str, Any]], 
                            all_segments: List[Dict[str, Any]], 
                            max_workers: int = 3) -> Dict[int, List[Dict[str, Any]]]:
        """
        Repair multiple segments in parallel.
        
        Args:
            audio_path: Path to original audio file
            segments_to_repair: List of segments needing repair
            all_segments: Complete list of transcript segments for context
            max_workers: Maximum number of parallel repair processes
            
        Returns:
            Dictionary mapping segment indices to repair candidates
        """
        print(f"🔧 Starting batch repair of {len(segments_to_repair)} segments...")
        
        repair_results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit repair tasks
            future_to_segment = {}
            
            for repair_info in segments_to_repair:
                segment_idx = repair_info['segment_index']
                segment = repair_info['segment']
                problem_type = repair_info.get('problem_type', 'low_confidence')
                
                # Get context segments (before and after)
                context_segments = self._get_context_segments(all_segments, segment_idx)
                
                future = executor.submit(
                    self.repair_segment,
                    audio_path,
                    segment,
                    problem_type,
                    context_segments
                )
                future_to_segment[future] = segment_idx
            
            # Collect results
            for future in as_completed(future_to_segment):
                segment_idx = future_to_segment[future]
                try:
                    repair_candidates = future.result()
                    repair_results[segment_idx] = repair_candidates
                    print(f"✅ Completed repair for segment {segment_idx}")
                except Exception as e:
                    print(f"❌ Repair failed for segment {segment_idx}: {e}")
                    repair_results[segment_idx] = []
        
        print(f"🔧 Batch repair complete: {len(repair_results)} segments processed")
        return repair_results
    
    def apply_segment_repair(self, master_transcript: Dict[str, Any], 
                           segment_index: int, repair_candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a repair candidate to the master transcript.
        
        Args:
            master_transcript: Current master transcript
            segment_index: Index of segment to repair
            repair_candidate: Repair candidate to apply
            
        Returns:
            Updated master transcript with repair applied
        """
        updated_transcript = master_transcript.copy()
        segments = updated_transcript['segments'].copy()
        
        if 0 <= segment_index < len(segments):
            # Replace segment with repaired version
            old_segment = segments[segment_index]
            new_segment = repair_candidate['repaired_segment']
            
            segments[segment_index] = new_segment
            updated_transcript['segments'] = segments
            
            # Update metadata
            if 'repair_history' not in updated_transcript:
                updated_transcript['repair_history'] = []
            
            repair_log = {
                'timestamp': time.time(),
                'segment_index': segment_index,
                'repair_id': repair_candidate['repair_id'],
                'problem_type': repair_candidate['problem_type'],
                'old_confidence': old_segment.get('confidence', 0.0),
                'new_confidence': new_segment.get('confidence', 0.0),
                'improvement_score': repair_candidate['improvement_score'],
                'old_text': old_segment.get('text', ''),
                'new_text': new_segment.get('text', ''),
                'repair_config': repair_candidate['repair_config']
            }
            
            updated_transcript['repair_history'].append(repair_log)
            
            # Recalculate overall confidence if needed
            self._update_overall_confidence(updated_transcript)
            
            print(f"✅ Applied repair to segment {segment_index}: {old_segment.get('confidence', 0):.3f} → {new_segment.get('confidence', 0):.3f}")
        
        return updated_transcript
    
    def reprocess_entire_segment_range(self, audio_path: str, start_time: float, end_time: float,
                                     diarization_data: Dict[str, Any],
                                     enhanced_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Reprocess an entire time range with enhanced parameters.
        
        Args:
            audio_path: Path to original audio file
            start_time: Start time of range to reprocess
            end_time: End time of range to reprocess
            diarization_data: Diarization data for the range
            enhanced_params: Enhanced ASR parameters
            
        Returns:
            Reprocessed segment data
        """
        print(f"🔄 Reprocessing range [{start_time:.1f}-{end_time:.1f}s] with enhanced parameters...")
        
        try:
            # Extract audio range
            range_audio_path = self._extract_segment_audio(audio_path, start_time, end_time)
            
            # Use enhanced parameters or defaults
            if not enhanced_params:
                enhanced_params = {
                    'temperature': 0.0,
                    'language': 'en',
                    'prompt': "High-quality transcription with careful attention to detail and timing.",
                    'response_format': 'verbose_json'
                }
            
            # Run enhanced ASR
            asr_result = self.asr_engine._run_asr_variant(
                range_audio_path,
                diarization_data,
                enhanced_params
            )
            
            # Clean up temporary file
            if os.path.exists(range_audio_path):
                os.unlink(range_audio_path)
            
            # Process result into segments
            processed_segments = self._process_reprocessed_range(
                asr_result, diarization_data, start_time, end_time
            )
            
            return {
                'start_time': start_time,
                'end_time': end_time,
                'reprocessed_segments': processed_segments,
                'asr_result': asr_result,
                'parameters': enhanced_params,
                'processing_time': time.time()
            }
            
        except Exception as e:
            print(f"❌ Range reprocessing failed: {e}")
            return {}
    
    def _extract_segment_audio(self, audio_path: str, start_time: float, end_time: float) -> str:
        """Extract audio segment to temporary file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Calculate sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract segment
            segment_audio = y[start_sample:end_sample]
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, segment_audio, sr)
            
            return temp_file.name
            
        except Exception as e:
            raise Exception(f"Audio extraction failed: {e}")
    
    def _enhance_config_with_context(self, config: Dict[str, Any], 
                                   context_segments: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Enhance ASR config with context from surrounding segments"""
        enhanced_config = config.copy()
        
        if context_segments:
            # Extract context text
            context_text = " ".join([seg.get('text', '') for seg in context_segments if seg.get('text')])
            
            if context_text:
                # Add context to prompt
                original_prompt = enhanced_config.get('prompt', '')
                enhanced_prompt = f"{original_prompt} Context: {context_text[:200]}..."
                enhanced_config['prompt'] = enhanced_prompt
        
        return enhanced_config
    
    def _create_repaired_segment(self, original_segment: Dict[str, Any], 
                               asr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create repaired segment from ASR result"""
        repaired_segment = original_segment.copy()
        
        # Update with new ASR data
        repaired_segment['text'] = asr_result.get('text', '').strip()
        repaired_segment['words'] = asr_result.get('words', [])
        repaired_segment['confidence'] = asr_result.get('confidence_scores', {}).get('word_confidence_mean', 0.0)
        repaired_segment['word_count'] = len(asr_result.get('words', []))
        
        # Add repair metadata
        repaired_segment['repair_applied'] = True
        repaired_segment['repair_timestamp'] = time.time()
        repaired_segment['original_confidence'] = original_segment.get('confidence', 0.0)
        
        return repaired_segment
    
    def _calculate_improvement_score(self, original_segment: Dict[str, Any], 
                                   asr_result: Dict[str, Any]) -> float:
        """Calculate improvement score for repair"""
        original_confidence = original_segment.get('confidence', 0.0)
        new_confidence = asr_result.get('confidence_scores', {}).get('word_confidence_mean', 0.0)
        
        improvement = new_confidence - original_confidence
        return max(0.0, min(1.0, improvement))
    
    def _calculate_repair_confidence(self, asr_result: Dict[str, Any], problem_type: str) -> float:
        """Calculate confidence in the repair quality"""
        base_confidence = asr_result.get('confidence_scores', {}).get('word_confidence_mean', 0.0)
        
        # Adjust based on problem type
        problem_adjustments = {
            'low_confidence': 0.0,      # No adjustment
            'unclear_audio': -0.1,      # Slightly lower confidence for unclear audio
            'speaker_confusion': -0.05   # Slight adjustment for speaker issues
        }
        
        adjustment = problem_adjustments.get(problem_type, 0.0)
        repair_confidence = base_confidence + adjustment
        
        return max(0.0, min(1.0, repair_confidence))
    
    def _get_context_segments(self, all_segments: List[Dict[str, Any]], 
                            target_index: int, context_window: int = 2) -> List[Dict[str, Any]]:
        """Get context segments around target segment"""
        context_segments = []
        
        # Get segments before
        for i in range(max(0, target_index - context_window), target_index):
            if i < len(all_segments):
                context_segments.append(all_segments[i])
        
        # Get segments after
        for i in range(target_index + 1, min(len(all_segments), target_index + context_window + 1)):
            context_segments.append(all_segments[i])
        
        return context_segments
    
    def _update_overall_confidence(self, transcript: Dict[str, Any]):
        """Update overall confidence scores after repair"""
        segments = transcript.get('segments', [])
        
        if segments:
            # Recalculate mean confidence
            segment_confidences = [seg.get('confidence', 0.0) for seg in segments]
            mean_confidence = np.mean(segment_confidences)
            
            # Update metadata
            metadata = transcript.get('metadata', {})
            if 'confidence_summary' in metadata:
                # Keep original dimension scores, update derived metrics
                metadata['confidence_summary']['mean_segment_confidence'] = float(mean_confidence)
                metadata['confidence_summary']['updated_after_repair'] = True
                metadata['confidence_summary']['repair_timestamp'] = time.time()
    
    def _process_reprocessed_range(self, asr_result: Dict[str, Any], 
                                 diarization_data: Dict[str, Any],
                                 start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Process reprocessed range into segments"""
        # This would involve re-aligning ASR results with diarization
        # For now, return a single segment
        processed_segments = [{
            'start': start_time,
            'end': end_time,
            'text': asr_result.get('text', ''),
            'words': asr_result.get('words', []),
            'confidence': asr_result.get('confidence_scores', {}).get('word_confidence_mean', 0.0),
            'reprocessed': True,
            'processing_timestamp': time.time()
        }]
        
        return processed_segments
    
    def generate_repair_report(self, repair_results: Dict[int, List[Dict[str, Any]]], 
                             original_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive repair report"""
        report = {
            'timestamp': time.time(),
            'segments_repaired': len(repair_results),
            'total_segments': len(original_segments),
            'repair_summary': {},
            'improvement_metrics': {},
            'recommendations': []
        }
        
        # Calculate improvement metrics
        total_improvement = 0.0
        successful_repairs = 0
        
        for segment_idx, candidates in repair_results.items():
            if candidates:
                best_candidate = max(candidates, key=lambda x: x['improvement_score'])
                if best_candidate['improvement_score'] > 0.05:  # Minimum 5% improvement
                    total_improvement += best_candidate['improvement_score']
                    successful_repairs += 1
        
        report['improvement_metrics'] = {
            'average_improvement': total_improvement / max(successful_repairs, 1),
            'successful_repairs': successful_repairs,
            'success_rate': successful_repairs / max(len(repair_results), 1)
        }
        
        # Generate recommendations
        if report['improvement_metrics']['success_rate'] < 0.3:
            report['recommendations'].append("Low repair success rate - consider manual review")
        
        if report['improvement_metrics']['average_improvement'] > 0.2:
            report['recommendations'].append("Significant improvements achieved - consider applying repairs")
        
        return report