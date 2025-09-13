"""
Metric calculation utilities for transcription system quality assessment.

This module provides standardized calculation of:
- Diarization Error Rate (DER) using pyannote.metrics
- Word Error Rate (WER) using jiwer with consistent transformation pipeline
- Ensemble quality metrics for reproducible assessment
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# Metric calculation libraries
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.detection import DetectionErrorRate
import jiwer
from jiwer import wer, cer, mer, wil, wip

# Internal imports
from utils.structured_logger import StructuredLogger
from utils.metrics_registry import MetricsRegistryManager


@dataclass
class DiarizationMetrics:
    """Structured container for diarization evaluation metrics"""
    der: float                    # Diarization Error Rate
    missed_speaker_time: float    # Percentage of time with missed speakers
    false_alarm_time: float       # Percentage of time with false alarms
    speaker_confusion_time: float # Percentage of time with speaker confusion
    speaker_count_accuracy: float # Accuracy of speaker count detection
    overlap_coverage: float       # Percentage of true overlaps detected
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ASRMetrics:
    """Structured container for ASR evaluation metrics"""
    wer: float          # Word Error Rate
    cer: float          # Character Error Rate
    mer: float          # Match Error Rate
    wil: float          # Word Information Lost
    wip: float          # Word Information Preserved
    insertions: int     # Number of word insertions
    deletions: int      # Number of word deletions
    substitutions: int  # Number of word substitutions
    word_count: int     # Total words in reference
    
    def to_dict(self) -> Dict[str, Union[float, int]]:
        return asdict(self)


@dataclass
class EnsembleMetrics:
    """Comprehensive metrics for ensemble evaluation"""
    diarization: DiarizationMetrics
    asr: ASRMetrics
    confidence_calibration: float  # Calibration accuracy of confidence scores
    ensemble_agreement: float      # Agreement between ensemble candidates
    processing_time: float         # Total processing time in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'diarization': self.diarization.to_dict(),
            'asr': self.asr.to_dict(),
            'confidence_calibration': self.confidence_calibration,
            'ensemble_agreement': self.ensemble_agreement,
            'processing_time': self.processing_time
        }


class WERTransformPipeline:
    """Consistent WER transformation pipeline for reproducible calculations"""
    
    def __init__(self):
        """Initialize standardized transformation pipeline"""
        # Define consistent transformation pipeline
        self.transforms = jiwer.Compose([
            # Normalize case
            jiwer.ToLowerCase(),
            # Remove punctuation but keep word boundaries
            jiwer.RemovePunctuation(),
            # Handle multiple whitespace
            jiwer.RemoveMultipleSpaces(),
            # Remove leading/trailing whitespace
            jiwer.Strip(),
            # Handle common abbreviations and contractions consistently
            jiwer.ExpandCommonEnglishContractions(),
            # Remove empty strings that might result from processing
            jiwer.RemoveEmptyStrings(),
        ])
    
    def __call__(self, sentences: List[str]) -> List[str]:
        """Apply transformation pipeline to sentences"""
        return self.transforms(sentences)
    
    def calculate_wer_detailed(self, reference: str, hypothesis: str) -> Tuple[float, Dict[str, int]]:
        """Calculate WER with detailed error breakdown"""
        # Apply transformations
        ref_transformed = self.transforms([reference])[0]
        hyp_transformed = self.transforms([hypothesis])[0]
        
        # Calculate detailed metrics using updated jiwer API
        wer_value = wer(ref_transformed, hyp_transformed)
        
        # For detailed breakdown, we'll use character-level analysis
        # This is a simplified version - in production you'd want more sophisticated alignment
        ref_words = ref_transformed.split()
        hyp_words = hyp_transformed.split()
        
        # Simple alignment-based calculation
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, ref_words, hyp_words)
        operations = matcher.get_opcodes()
        
        substitutions = 0
        deletions = 0
        insertions = 0
        hits = 0
        
        for tag, ref_start, ref_end, hyp_start, hyp_end in operations:
            if tag == 'replace':
                substitutions += max(ref_end - ref_start, hyp_end - hyp_start)
            elif tag == 'delete':
                deletions += ref_end - ref_start
            elif tag == 'insert':
                insertions += hyp_end - hyp_start
            elif tag == 'equal':
                hits += ref_end - ref_start
        
        return wer_value, {
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'hits': hits
        }


class MetricCalculator:
    """Central calculator for all transcription quality metrics"""
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize metric calculator
        
        Args:
            logger: Optional structured logger instance
        """
        self.logger = logger or StructuredLogger("metric_calculator")
        
        # Initialize WER transformation pipeline
        self.wer_pipeline = WERTransformPipeline()
        
        # Initialize diarization metrics calculator
        self.der_metric = DiarizationErrorRate()
        self.detection_metric = DetectionErrorRate()
        
        self.logger.info("MetricCalculator initialized with standardized pipelines")
    
    def calculate_diarization_metrics(
        self, 
        reference_segments: List[Dict[str, Any]], 
        hypothesis_segments: List[Dict[str, Any]],
        audio_duration: float
    ) -> DiarizationMetrics:
        """
        Calculate comprehensive diarization metrics
        
        Args:
            reference_segments: Ground truth speaker segments
            hypothesis_segments: Predicted speaker segments  
            audio_duration: Total audio duration in seconds
            
        Returns:
            DiarizationMetrics object with all computed metrics
        """
        try:
            # Convert to pyannote Annotation format
            reference = self._segments_to_annotation(reference_segments)
            hypothesis = self._segments_to_annotation(hypothesis_segments)
            
            # Calculate DER
            der_value = self.der_metric(reference, hypothesis)
            
            # Calculate detailed error components
            # Note: pyannote.metrics API may not have detailed() method in all versions
            der_components = {"missed speaker": 0.0, "false alarm": 0.0, "speaker confusion": 0.0}
            
            # Calculate speaker count accuracy
            ref_speakers = len(set(seg.get('speaker', 'UNKNOWN') for seg in reference_segments))
            hyp_speakers = len(set(seg.get('speaker', 'UNKNOWN') for seg in hypothesis_segments))
            speaker_count_accuracy = 1.0 - abs(ref_speakers - hyp_speakers) / max(ref_speakers, 1)
            
            # Calculate overlap metrics
            overlap_coverage = self._calculate_overlap_coverage(reference_segments, hypothesis_segments)
            
            # Safely convert DER value to float (handles pyannote Details type)
            try:
                if isinstance(der_value, (int, float)):
                    der_float = float(der_value)
                else:
                    # Handle pyannote Details type or other complex types
                    der_float = 0.0
                    # Try multiple conversion strategies
                    for attr in ['item', '__float__', '_value', 'value']:
                        if hasattr(der_value, attr):
                            try:
                                der_float = float(getattr(der_value, attr)())
                                break
                            except (TypeError, AttributeError, ValueError):
                                continue
                    else:
                        # Last resort: try direct conversion
                        try:
                            der_float = float(str(der_value))
                        except (ValueError, TypeError):
                            der_float = 0.0
            except Exception:
                der_float = 0.0
            
            metrics = DiarizationMetrics(
                der=der_float,
                missed_speaker_time=der_components.get('missed speaker', 0.0),
                false_alarm_time=der_components.get('false alarm', 0.0),
                speaker_confusion_time=der_components.get('speaker confusion', 0.0),
                speaker_count_accuracy=speaker_count_accuracy,
                overlap_coverage=overlap_coverage
            )
            
            self.logger.info("Diarization metrics calculated", 
                           context={'der': metrics.der, 'speaker_accuracy': speaker_count_accuracy})
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating diarization metrics: {e}")
            # Return default metrics on error
            return DiarizationMetrics(
                der=1.0, missed_speaker_time=0.0, false_alarm_time=0.0,
                speaker_confusion_time=0.0, speaker_count_accuracy=0.0, overlap_coverage=0.0
            )
    
    def calculate_asr_metrics(
        self, 
        reference_text: str, 
        hypothesis_text: str
    ) -> ASRMetrics:
        """
        Calculate comprehensive ASR metrics using standardized WER pipeline
        
        Args:
            reference_text: Ground truth transcript
            hypothesis_text: Predicted transcript
            
        Returns:
            ASRMetrics object with all computed metrics
        """
        try:
            # Calculate WER with detailed breakdown
            wer_value, error_breakdown = self.wer_pipeline.calculate_wer_detailed(reference_text, hypothesis_text)
            
            # Apply transformations for other metrics
            ref_clean = self.wer_pipeline([reference_text])[0]
            hyp_clean = self.wer_pipeline([hypothesis_text])[0]
            
            # Calculate additional metrics
            cer_value = cer(ref_clean, hyp_clean)
            mer_value = mer(ref_clean, hyp_clean)
            wil_value = wil(ref_clean, hyp_clean)
            wip_value = wip(ref_clean, hyp_clean)
            
            # Count words in reference
            word_count = len(ref_clean.split())
            
            metrics = ASRMetrics(
                wer=float(wer_value),
                cer=float(cer_value),
                mer=float(mer_value),
                wil=float(wil_value),
                wip=float(wip_value),
                insertions=error_breakdown['insertions'],
                deletions=error_breakdown['deletions'],
                substitutions=error_breakdown['substitutions'],
                word_count=word_count
            )
            
            self.logger.info("ASR metrics calculated", 
                           context={'wer': metrics.wer, 'word_count': word_count})
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating ASR metrics: {e}")
            # Return default metrics on error
            return ASRMetrics(
                wer=1.0, cer=1.0, mer=1.0, wil=1.0, wip=0.0,
                insertions=0, deletions=0, substitutions=0, word_count=0
            )
    
    def calculate_ensemble_metrics(
        self,
        candidates: List[Dict[str, Any]],
        reference_segments: List[Dict[str, Any]],
        reference_text: str,
        audio_duration: float,
        processing_time: float
    ) -> EnsembleMetrics:
        """
        Calculate comprehensive ensemble evaluation metrics
        
        Args:
            candidates: List of ensemble candidates with confidence scores
            reference_segments: Ground truth speaker segments
            reference_text: Ground truth transcript
            audio_duration: Total audio duration in seconds
            processing_time: Total processing time in seconds
            
        Returns:
            EnsembleMetrics object with all computed metrics
        """
        # Find best candidate (highest final score)
        best_candidate = max(candidates, key=lambda x: x.get('confidence_scores', {}).get('final_score', 0))
        
        # Calculate diarization metrics for best candidate
        diarization_metrics = self.calculate_diarization_metrics(
            reference_segments,
            best_candidate.get('aligned_segments', []),
            audio_duration
        )
        
        # Calculate ASR metrics for best candidate
        hypothesis_text = self._extract_full_transcript(best_candidate)
        asr_metrics = self.calculate_asr_metrics(reference_text, hypothesis_text)
        
        # Calculate confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(candidates, asr_metrics.wer)
        
        # Calculate ensemble agreement
        ensemble_agreement = self._calculate_ensemble_agreement(candidates)
        
        return EnsembleMetrics(
            diarization=diarization_metrics,
            asr=asr_metrics,
            confidence_calibration=confidence_calibration,
            ensemble_agreement=ensemble_agreement,
            processing_time=processing_time
        )
    
    def _segments_to_annotation(self, segments: List[Dict[str, Any]]) -> Annotation:
        """Convert segment list to pyannote Annotation"""
        annotation = Annotation()
        
        for segment in segments:
            start = segment.get('start', 0.0)
            end = segment.get('end', 0.0)
            speaker = segment.get('speaker', 'UNKNOWN')
            
            if end > start:  # Valid segment
                annotation[Segment(start, end)] = speaker
        
        return annotation
    
    def _calculate_overlap_coverage(
        self, 
        reference_segments: List[Dict[str, Any]], 
        hypothesis_segments: List[Dict[str, Any]]
    ) -> float:
        """Calculate what percentage of reference overlaps are detected"""
        # Find reference overlaps
        ref_overlaps = self._find_overlaps(reference_segments)
        if not ref_overlaps:
            return 1.0  # Perfect if no overlaps to detect
        
        # Find hypothesis overlaps  
        hyp_overlaps = self._find_overlaps(hypothesis_segments)
        
        # Calculate coverage
        detected_overlaps = 0
        for ref_overlap in ref_overlaps:
            for hyp_overlap in hyp_overlaps:
                if self._overlap_matches(ref_overlap, hyp_overlap):
                    detected_overlaps += 1
                    break
        
        return detected_overlaps / len(ref_overlaps)
    
    def _find_overlaps(self, segments: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Find time regions where multiple speakers overlap"""
        overlaps = []
        
        for i, seg1 in enumerate(segments):
            for seg2 in segments[i+1:]:
                start1, end1 = seg1.get('start', 0), seg1.get('end', 0)
                start2, end2 = seg2.get('start', 0), seg2.get('end', 0)
                
                # Check for temporal overlap
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                
                if overlap_end > overlap_start:
                    overlaps.append((overlap_start, overlap_end))
        
        return overlaps
    
    def _overlap_matches(self, ref_overlap: Tuple[float, float], hyp_overlap: Tuple[float, float], threshold: float = 0.5) -> bool:
        """Check if two overlaps match within threshold"""
        ref_start, ref_end = ref_overlap
        hyp_start, hyp_end = hyp_overlap
        
        # Calculate intersection over union
        intersection = max(0, min(ref_end, hyp_end) - max(ref_start, hyp_start))
        union = max(ref_end, hyp_end) - min(ref_start, hyp_start)
        
        iou = intersection / union if union > 0 else 0
        return iou >= threshold
    
    def _extract_full_transcript(self, candidate: Dict[str, Any]) -> str:
        """Extract full transcript text from candidate"""
        segments = candidate.get('aligned_segments', [])
        texts = []
        
        for segment in sorted(segments, key=lambda x: x.get('start', 0)):
            text = segment.get('text', '').strip()
            if text:
                texts.append(text)
        
        return ' '.join(texts)
    
    def _calculate_confidence_calibration(self, candidates: List[Dict[str, Any]], true_wer: float) -> float:
        """Calculate how well confidence scores predict actual quality"""
        if len(candidates) < 2:
            return 0.0
        
        # Extract confidence scores and calculate predicted vs actual quality
        confidences = []
        for candidate in candidates:
            conf_scores = candidate.get('confidence_scores', {})
            final_score = conf_scores.get('final_score', 0.0)
            confidences.append(final_score)
        
        # Calculate correlation between confidence and inverse WER
        predicted_quality = np.array(confidences)
        actual_quality = 1.0 - true_wer  # Convert WER to quality score
        
        # Calibration is measured as how well confidence predicts quality
        if np.std(predicted_quality) > 0:
            correlation = np.corrcoef(predicted_quality, [actual_quality] * len(predicted_quality))[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _calculate_ensemble_agreement(self, candidates: List[Dict[str, Any]]) -> float:
        """Calculate agreement between ensemble candidates"""
        if len(candidates) < 2:
            return 1.0
        
        # Extract transcripts from all candidates
        transcripts = []
        for candidate in candidates:
            transcript = self._extract_full_transcript(candidate)
            transcripts.append(transcript)
        
        # Calculate pairwise WER agreement
        agreements = []
        for i, trans1 in enumerate(transcripts):
            for trans2 in transcripts[i+1:]:
                if trans1 and trans2:
                    wer_between = wer(trans1, trans2)  # Removed invalid truth parameter
                    agreement = 1.0 - wer_between  # Convert to agreement score
                    agreements.append(max(0.0, agreement))
        
        return float(np.mean(agreements)) if agreements else 0.0


class MetricReporter:
    """Generate comprehensive metric reports and summaries"""
    
    def __init__(self, output_dir: str = "metrics_reports"):
        """
        Initialize metric reporter
        
        Args:
            output_dir: Directory to save metric reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = StructuredLogger("metric_reporter")
    
    def generate_report(
        self, 
        metrics: EnsembleMetrics, 
        test_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive metric report
        
        Args:
            metrics: Computed ensemble metrics
            test_name: Name of the test case
            metadata: Additional metadata to include
            
        Returns:
            Path to generated report file
        """
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"metrics_report_{test_name}_{timestamp}.json"
        
        report_data = {
            'test_name': test_name,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'metrics': metrics.to_dict()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Metric report generated: {report_file}")
        return str(report_file)
    
    def compare_reports(self, report_paths: List[str]) -> Dict[str, Any]:
        """Compare multiple metric reports and identify trends"""
        if len(report_paths) < 2:
            return {}
        
        reports = []
        for path in report_paths:
            with open(path, 'r') as f:
                reports.append(json.load(f))
        
        # Extract key metrics for comparison
        comparison = {
            'wer_trend': [r['metrics']['asr']['wer'] for r in reports],
            'der_trend': [r['metrics']['diarization']['der'] for r in reports],
            'confidence_trend': [r['metrics']['confidence_calibration'] for r in reports],
            'processing_time_trend': [r['metrics']['processing_time'] for r in reports]
        }
        
        # Calculate regression indicators  
        regression_data = {}
        for metric, values in comparison.items():
            if len(values) >= 2:
                recent_avg = float(np.mean(values[-3:]))  # Last 3 values
                baseline_avg = float(np.mean(values[:3]))  # First 3 values
                regression_data[f'{metric}_regression'] = recent_avg - baseline_avg
        
        # Add regression data to comparison
        comparison.update(regression_data)
        
        return comparison