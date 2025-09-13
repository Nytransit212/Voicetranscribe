import numpy as np
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import statistics

class QualityLevel(Enum):
    """Quality levels for different use cases"""
    DRAFT = "draft"        # Lower standards for initial review
    REVIEW = "review"      # Medium standards for internal review
    FINAL = "final"        # High standards for publication
    PREMIUM = "premium"    # Highest standards for critical use

class IssueType(Enum):
    """Types of quality issues detected"""
    LOW_CONFIDENCE = "low_confidence"
    SPEAKER_CONFUSION = "speaker_confusion"
    UNCLEAR_AUDIO = "unclear_audio"
    TRANSCRIPTION_ERROR = "transcription_error"
    ALIGNMENT_ISSUE = "alignment_issue"
    LINGUISTIC_ISSUE = "linguistic_issue"
    OVERLAP_ISSUE = "overlap_issue"

class IssueSeverity(Enum):
    """Severity levels for issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class QualityThresholds:
    """Quality threshold configuration for different levels"""
    level: QualityLevel
    final_score_min: float
    D_diarization_min: float
    A_asr_alignment_min: float
    L_linguistic_min: float
    R_agreement_min: float
    O_overlap_min: float
    segment_confidence_min: float
    word_confidence_min: float
    outlier_std_threshold: float  # Standard deviations below mean for outlier detection

@dataclass
class QualityIssue:
    """Represents a quality issue found in a segment"""
    segment_index: int
    issue_type: IssueType
    severity: IssueSeverity
    confidence_score: float
    description: str
    suggested_action: str
    context: Dict[str, Any]

@dataclass
class QualityAssessment:
    """Complete quality assessment for a transcript"""
    overall_grade: QualityLevel
    passes_quality_gate: bool
    total_segments: int
    flagged_segments: int
    quality_score: float
    issues: List[QualityIssue]
    metrics: Dict[str, float]
    recommendations: List[str]

class QualityController:
    """Comprehensive quality control system for ensemble transcription"""
    
    def __init__(self):
        # Define quality thresholds for different levels
        self.thresholds = {
            QualityLevel.DRAFT: QualityThresholds(
                level=QualityLevel.DRAFT,
                final_score_min=0.45,
                D_diarization_min=0.35,
                A_asr_alignment_min=0.40,
                L_linguistic_min=0.30,
                R_agreement_min=0.25,
                O_overlap_min=0.25,
                segment_confidence_min=0.30,
                word_confidence_min=0.25,
                outlier_std_threshold=1.5
            ),
            QualityLevel.REVIEW: QualityThresholds(
                level=QualityLevel.REVIEW,
                final_score_min=0.60,
                D_diarization_min=0.50,
                A_asr_alignment_min=0.55,
                L_linguistic_min=0.45,
                R_agreement_min=0.40,
                O_overlap_min=0.40,
                segment_confidence_min=0.45,
                word_confidence_min=0.40,
                outlier_std_threshold=1.25
            ),
            QualityLevel.FINAL: QualityThresholds(
                level=QualityLevel.FINAL,
                final_score_min=0.75,
                D_diarization_min=0.65,
                A_asr_alignment_min=0.70,
                L_linguistic_min=0.60,
                R_agreement_min=0.55,
                O_overlap_min=0.55,
                segment_confidence_min=0.60,
                word_confidence_min=0.55,
                outlier_std_threshold=1.0
            ),
            QualityLevel.PREMIUM: QualityThresholds(
                level=QualityLevel.PREMIUM,
                final_score_min=0.85,
                D_diarization_min=0.75,
                A_asr_alignment_min=0.80,
                L_linguistic_min=0.70,
                R_agreement_min=0.65,
                O_overlap_min=0.65,
                segment_confidence_min=0.70,
                word_confidence_min=0.65,
                outlier_std_threshold=0.8
            )
        }
        
    def assess_transcript_quality(self, master_transcript: Dict[str, Any], 
                                target_level: QualityLevel = QualityLevel.REVIEW) -> QualityAssessment:
        """
        Perform comprehensive quality assessment of a transcript.
        
        Args:
            master_transcript: Master transcript data with segments and confidence scores
            target_level: Target quality level for assessment
            
        Returns:
            Complete quality assessment with issues and recommendations
        """
        print(f"🔍 Performing quality assessment at {target_level.value} level...")
        
        segments = master_transcript.get('segments', [])
        confidence_summary = master_transcript.get('metadata', {}).get('confidence_summary', {})
        thresholds = self.thresholds[target_level]
        
        # Analyze segments for issues
        issues = []
        flagged_segment_count = 0
        
        # Overall quality metrics
        overall_final_score = confidence_summary.get('final_score', 0.0)
        
        # Segment-level analysis
        segment_scores = []
        for i, segment in enumerate(segments):
            segment_confidence = segment.get('confidence', 0.0)
            segment_scores.append(segment_confidence)
            
            # Check for low confidence
            if segment_confidence < thresholds.segment_confidence_min:
                severity = self._determine_severity(segment_confidence, thresholds.segment_confidence_min)
                issues.append(QualityIssue(
                    segment_index=i,
                    issue_type=IssueType.LOW_CONFIDENCE,
                    severity=severity,
                    confidence_score=segment_confidence,
                    description=f"Segment confidence ({segment_confidence:.3f}) below minimum ({thresholds.segment_confidence_min:.3f})",
                    suggested_action="Consider re-processing with alternative ASR parameters",
                    context={'segment': segment}
                ))
                flagged_segment_count += 1
            
            # Check for speaker assignment issues
            if hasattr(segment, 'speaker_conflicts') and segment.get('speaker_conflicts', 0) > 0:
                issues.append(QualityIssue(
                    segment_index=i,
                    issue_type=IssueType.SPEAKER_CONFUSION,
                    severity=IssueSeverity.HIGH,
                    confidence_score=segment_confidence,
                    description=f"Speaker assignment conflicts detected ({segment.get('speaker_conflicts', 0)} conflicts)",
                    suggested_action="Review speaker diarization for this segment",
                    context={'segment': segment}
                ))
            
            # Check for alignment issues
            alignment_confidence = segment.get('alignment_confidence', 1.0)
            if alignment_confidence < 0.6:
                issues.append(QualityIssue(
                    segment_index=i,
                    issue_type=IssueType.ALIGNMENT_ISSUE,
                    severity=IssueSeverity.MEDIUM,
                    confidence_score=alignment_confidence,
                    description=f"Poor ASR-diarization alignment ({alignment_confidence:.3f})",
                    suggested_action="Check segment boundaries and speaker transitions",
                    context={'segment': segment}
                ))
        
        # Statistical outlier detection
        if segment_scores:
            mean_score = statistics.mean(segment_scores)
            std_score = statistics.stdev(segment_scores) if len(segment_scores) > 1 else 0.0
            outlier_threshold = mean_score - (thresholds.outlier_std_threshold * std_score)
            
            for i, score in enumerate(segment_scores):
                if score < outlier_threshold and score >= thresholds.segment_confidence_min:
                    # This segment is a statistical outlier but not flagged by absolute threshold
                    issues.append(QualityIssue(
                        segment_index=i,
                        issue_type=IssueType.TRANSCRIPTION_ERROR,
                        severity=IssueSeverity.MEDIUM,
                        confidence_score=score,
                        description=f"Statistical outlier: {thresholds.outlier_std_threshold:.1f}σ below mean ({score:.3f} vs {mean_score:.3f}±{std_score:.3f})",
                        suggested_action="Review for potential transcription errors",
                        context={'segment': segments[i], 'mean_score': mean_score, 'std_score': std_score}
                    ))
        
        # Check dimension-specific issues
        dimension_issues = self._check_dimension_specific_issues(confidence_summary, thresholds)
        issues.extend(dimension_issues)
        
        # Calculate overall quality metrics
        quality_score = self._calculate_overall_quality_score(confidence_summary, segment_scores)
        passes_gate = self._check_quality_gate(confidence_summary, segment_scores, thresholds)
        overall_grade = self._determine_quality_grade(quality_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, confidence_summary, target_level)
        
        # Compile metrics
        metrics = {
            'overall_final_score': overall_final_score,
            'mean_segment_confidence': mean_score if segment_scores else 0.0,
            'std_segment_confidence': std_score if segment_scores else 0.0,
            'flagged_segments_ratio': flagged_segment_count / max(len(segments), 1),
            'total_issues': len(issues),
            'critical_issues': sum(1 for issue in issues if issue.severity == IssueSeverity.CRITICAL),
            'high_issues': sum(1 for issue in issues if issue.severity == IssueSeverity.HIGH),
            'medium_issues': sum(1 for issue in issues if issue.severity == IssueSeverity.MEDIUM),
            'low_issues': sum(1 for issue in issues if issue.severity == IssueSeverity.LOW)
        }
        
        assessment = QualityAssessment(
            overall_grade=overall_grade,
            passes_quality_gate=passes_gate,
            total_segments=len(segments),
            flagged_segments=flagged_segment_count,
            quality_score=quality_score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )
        
        print(f"✅ Quality assessment complete:")
        print(f"   Grade: {overall_grade.value.upper()}")
        print(f"   Passes {target_level.value} gate: {passes_gate}")
        print(f"   Issues found: {len(issues)} ({metrics['critical_issues']} critical)")
        print(f"   Flagged segments: {flagged_segment_count}/{len(segments)}")
        
        return assessment
    
    def get_repair_candidates(self, master_transcript: Dict[str, Any], 
                            all_candidates: List[Dict[str, Any]], 
                            assessment: QualityAssessment) -> Dict[int, List[Dict[str, Any]]]:
        """
        Find alternative candidates for segments that need repair.
        
        Args:
            master_transcript: Current master transcript
            all_candidates: All 15 original candidates from ensemble
            assessment: Quality assessment with identified issues
            
        Returns:
            Dictionary mapping segment indices to alternative candidate segments
        """
        print(f"🔧 Finding repair candidates for {len(assessment.issues)} issues...")
        
        segments = master_transcript.get('segments', [])
        repair_candidates = {}
        
        # Get segments that need repair
        problem_segments = set()
        for issue in assessment.issues:
            if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]:
                problem_segments.add(issue.segment_index)
        
        # For each problem segment, find alternative candidates
        for segment_idx in problem_segments:
            if segment_idx >= len(segments):
                continue
                
            problem_segment = segments[segment_idx]
            segment_start = problem_segment['start']
            segment_end = problem_segment['end']
            
            # Find overlapping segments from other candidates
            alternatives = []
            
            for candidate in all_candidates:
                candidate_segments = candidate.get('aligned_segments', [])
                
                # Find segments that overlap with the problem segment
                for cand_segment in candidate_segments:
                    if self._segments_overlap(problem_segment, cand_segment):
                        # Calculate overlap ratio
                        overlap_ratio = self._calculate_overlap_ratio(problem_segment, cand_segment)
                        
                        # Only consider segments with significant overlap (>50%)
                        if overlap_ratio > 0.5:
                            alternatives.append({
                                'candidate_id': candidate['candidate_id'],
                                'segment': cand_segment,
                                'overlap_ratio': overlap_ratio,
                                'confidence': cand_segment.get('confidence', 0.0),
                                'alternative_score': self._score_alternative_candidate(
                                    problem_segment, cand_segment, candidate
                                )
                            })
            
            # Sort alternatives by score (best first)
            alternatives.sort(key=lambda x: x['alternative_score'], reverse=True)
            
            # Keep top 5 alternatives
            repair_candidates[segment_idx] = alternatives[:5]
            
            print(f"   Segment {segment_idx}: Found {len(alternatives)} alternatives")
        
        return repair_candidates
    
    def create_repair_plan(self, assessment: QualityAssessment, 
                          repair_candidates: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Create a comprehensive repair plan for the transcript.
        
        Args:
            assessment: Quality assessment with identified issues
            repair_candidates: Alternative candidates for problematic segments
            
        Returns:
            Detailed repair plan with actions and priorities
        """
        repair_actions = []
        
        # Prioritize issues by severity
        critical_issues = [i for i in assessment.issues if i.severity == IssueSeverity.CRITICAL]
        high_issues = [i for i in assessment.issues if i.severity == IssueSeverity.HIGH]
        medium_issues = [i for i in assessment.issues if i.severity == IssueSeverity.MEDIUM]
        low_issues = [i for i in assessment.issues if i.severity == IssueSeverity.LOW]
        
        # Create repair actions for critical and high priority issues
        for issue in critical_issues + high_issues:
            segment_idx = issue.segment_index
            
            action = {
                'segment_index': segment_idx,
                'issue': issue,
                'priority': 'high' if issue.severity == IssueSeverity.CRITICAL else 'medium',
                'repair_type': self._determine_repair_type(issue),
                'alternatives': repair_candidates.get(segment_idx, []),
                'estimated_improvement': self._estimate_repair_improvement(issue, repair_candidates.get(segment_idx, []))
            }
            
            repair_actions.append(action)
        
        # Add medium priority issues if they have good alternatives
        for issue in medium_issues:
            segment_idx = issue.segment_index
            alternatives = repair_candidates.get(segment_idx, [])
            
            # Only include if we have good alternatives
            if alternatives and alternatives[0]['alternative_score'] > 0.7:
                action = {
                    'segment_index': segment_idx,
                    'issue': issue,
                    'priority': 'low',
                    'repair_type': self._determine_repair_type(issue),
                    'alternatives': alternatives,
                    'estimated_improvement': self._estimate_repair_improvement(issue, alternatives)
                }
                repair_actions.append(action)
        
        # Sort by priority and estimated improvement
        repair_actions.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x['priority']],
            x['estimated_improvement']
        ), reverse=True)
        
        repair_plan = {
            'total_issues': len(assessment.issues),
            'repairable_issues': len(repair_actions),
            'estimated_time_minutes': len(repair_actions) * 2,  # 2 minutes per repair
            'repair_actions': repair_actions,
            'summary': {
                'critical_repairs': len([a for a in repair_actions if a['priority'] == 'high']),
                'medium_repairs': len([a for a in repair_actions if a['priority'] == 'medium']),
                'low_repairs': len([a for a in repair_actions if a['priority'] == 'low']),
                'projected_quality_improvement': self._calculate_projected_improvement(repair_actions, assessment)
            }
        }
        
        return repair_plan
    
    def apply_segment_repair(self, master_transcript: Dict[str, Any], 
                           segment_index: int, alternative_segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a repair by replacing a segment with an alternative.
        
        Args:
            master_transcript: Current master transcript
            segment_index: Index of segment to replace
            alternative_segment: Alternative segment data
            
        Returns:
            Updated master transcript
        """
        updated_transcript = master_transcript.copy()
        segments = updated_transcript['segments'].copy()
        
        if 0 <= segment_index < len(segments):
            # Replace the segment
            old_segment = segments[segment_index]
            segments[segment_index] = alternative_segment['segment'].copy()
            
            # Update metadata
            updated_transcript['segments'] = segments
            
            # Log the repair
            repair_log = {
                'timestamp': time.time(),
                'segment_index': segment_index,
                'old_confidence': old_segment.get('confidence', 0.0),
                'new_confidence': alternative_segment['segment'].get('confidence', 0.0),
                'improvement': alternative_segment['segment'].get('confidence', 0.0) - old_segment.get('confidence', 0.0),
                'alternative_source': alternative_segment.get('candidate_id', 'unknown')
            }
            
            # Add to repair history
            if 'repair_history' not in updated_transcript:
                updated_transcript['repair_history'] = []
            updated_transcript['repair_history'].append(repair_log)
            
            print(f"✅ Applied repair to segment {segment_index}: {old_segment.get('confidence', 0):.3f} → {alternative_segment['segment'].get('confidence', 0):.3f}")
        
        return updated_transcript
    
    def _determine_severity(self, score: float, threshold: float) -> IssueSeverity:
        """Determine issue severity based on how far below threshold"""
        ratio = score / threshold if threshold > 0 else 0
        
        if ratio < 0.5:
            return IssueSeverity.CRITICAL
        elif ratio < 0.7:
            return IssueSeverity.HIGH
        elif ratio < 0.85:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
    
    def _check_dimension_specific_issues(self, confidence_summary: Dict[str, float], 
                                       thresholds: QualityThresholds) -> List[QualityIssue]:
        """Check for issues in specific confidence dimensions"""
        issues = []
        
        dimension_checks = [
            ('D_diarization', thresholds.D_diarization_min, IssueType.SPEAKER_CONFUSION, 
             "Poor speaker diarization quality"),
            ('A_asr_alignment', thresholds.A_asr_alignment_min, IssueType.ALIGNMENT_ISSUE,
             "Poor ASR-diarization alignment"),
            ('L_linguistic', thresholds.L_linguistic_min, IssueType.LINGUISTIC_ISSUE,
             "Poor linguistic quality"),
            ('R_agreement', thresholds.R_agreement_min, IssueType.TRANSCRIPTION_ERROR,
             "Low cross-run agreement"),
            ('O_overlap', thresholds.O_overlap_min, IssueType.OVERLAP_ISSUE,
             "Poor overlap handling")
        ]
        
        for dimension, min_threshold, issue_type, description in dimension_checks:
            score = confidence_summary.get(dimension, 0.0)
            if score < min_threshold:
                severity = self._determine_severity(score, min_threshold)
                issues.append(QualityIssue(
                    segment_index=-1,  # Overall issue
                    issue_type=issue_type,
                    severity=severity,
                    confidence_score=score,
                    description=f"{description} ({score:.3f} < {min_threshold:.3f})",
                    suggested_action=f"Review {issue_type.value} settings and parameters",
                    context={'dimension': dimension, 'score': score, 'threshold': min_threshold}
                ))
        
        return issues
    
    def _calculate_overall_quality_score(self, confidence_summary: Dict[str, float], 
                                       segment_scores: List[float]) -> float:
        """Calculate overall quality score combining multiple factors"""
        final_score = confidence_summary.get('final_score', 0.0)
        mean_segment = statistics.mean(segment_scores) if segment_scores else 0.0
        
        # Weighted combination
        overall_score = (0.7 * final_score) + (0.3 * mean_segment)
        
        return min(1.0, max(0.0, overall_score))
    
    def _check_quality_gate(self, confidence_summary: Dict[str, float], 
                          segment_scores: List[float], thresholds: QualityThresholds) -> bool:
        """Check if transcript passes the quality gate"""
        final_score = confidence_summary.get('final_score', 0.0)
        mean_segment = statistics.mean(segment_scores) if segment_scores else 0.0
        
        # All conditions must pass
        passes = (
            final_score >= thresholds.final_score_min and
            mean_segment >= thresholds.segment_confidence_min and
            confidence_summary.get('D_diarization', 0.0) >= thresholds.D_diarization_min and
            confidence_summary.get('A_asr_alignment', 0.0) >= thresholds.A_asr_alignment_min and
            confidence_summary.get('L_linguistic', 0.0) >= thresholds.L_linguistic_min
        )
        
        return passes
    
    def _determine_quality_grade(self, quality_score: float) -> QualityLevel:
        """Determine quality grade based on score"""
        if quality_score >= 0.85:
            return QualityLevel.PREMIUM
        elif quality_score >= 0.75:
            return QualityLevel.FINAL
        elif quality_score >= 0.60:
            return QualityLevel.REVIEW
        else:
            return QualityLevel.DRAFT
    
    def _generate_recommendations(self, issues: List[QualityIssue], 
                                confidence_summary: Dict[str, float], 
                                target_level: QualityLevel) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Count issues by type
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        # Generate recommendations based on issue patterns
        if issue_counts.get(IssueType.LOW_CONFIDENCE, 0) > 3:
            recommendations.append("Consider re-processing with higher quality ASR settings")
        
        if issue_counts.get(IssueType.SPEAKER_CONFUSION, 0) > 2:
            recommendations.append("Review speaker diarization parameters - may need manual speaker labeling")
        
        if issue_counts.get(IssueType.ALIGNMENT_ISSUE, 0) > 2:
            recommendations.append("Check ASR-diarization alignment - consider adjusting segmentation boundaries")
        
        if confidence_summary.get('final_score', 0.0) < 0.6:
            recommendations.append("Overall quality is low - consider re-processing entire transcript")
        
        # Level-specific recommendations
        if target_level in [QualityLevel.FINAL, QualityLevel.PREMIUM]:
            recommendations.append("For production use, manually review all flagged segments")
            recommendations.append("Consider human validation of speaker assignments")
        
        if not recommendations:
            recommendations.append("Quality is acceptable for current standards")
        
        return recommendations
    
    def _segments_overlap(self, seg1: Dict[str, Any], seg2: Dict[str, Any]) -> bool:
        """Check if two segments have temporal overlap"""
        return seg1['start'] < seg2['end'] and seg2['start'] < seg1['end']
    
    def _calculate_overlap_ratio(self, seg1: Dict[str, Any], seg2: Dict[str, Any]) -> float:
        """Calculate overlap ratio between two segments"""
        overlap_start = max(seg1['start'], seg2['start'])
        overlap_end = min(seg1['end'], seg2['end'])
        overlap_duration = max(0, overlap_end - overlap_start)
        
        seg1_duration = seg1['end'] - seg1['start']
        
        return overlap_duration / seg1_duration if seg1_duration > 0 else 0.0
    
    def _score_alternative_candidate(self, problem_segment: Dict[str, Any], 
                                   candidate_segment: Dict[str, Any], 
                                   candidate: Dict[str, Any]) -> float:
        """Score how good an alternative candidate is"""
        # Base score from confidence
        confidence_score = candidate_segment.get('confidence', 0.0)
        
        # Bonus for better overlap
        overlap_ratio = self._calculate_overlap_ratio(problem_segment, candidate_segment)
        overlap_bonus = min(0.2, overlap_ratio - 0.5)  # Up to 0.2 bonus for >50% overlap
        
        # Bonus for higher overall candidate quality
        candidate_final_score = candidate.get('confidence_scores', {}).get('final_score', 0.0)
        candidate_bonus = min(0.1, (candidate_final_score - 0.5) * 0.2)  # Up to 0.1 bonus
        
        total_score = confidence_score + overlap_bonus + candidate_bonus
        
        return min(1.0, max(0.0, total_score))
    
    def _determine_repair_type(self, issue: QualityIssue) -> str:
        """Determine the type of repair needed"""
        if issue.issue_type == IssueType.LOW_CONFIDENCE:
            return "alternative_candidate"
        elif issue.issue_type == IssueType.SPEAKER_CONFUSION:
            return "speaker_reassignment"
        elif issue.issue_type == IssueType.ALIGNMENT_ISSUE:
            return "boundary_adjustment"
        elif issue.issue_type == IssueType.TRANSCRIPTION_ERROR:
            return "text_correction"
        else:
            return "manual_review"
    
    def _estimate_repair_improvement(self, issue: QualityIssue, 
                                   alternatives: List[Dict[str, Any]]) -> float:
        """Estimate how much improvement a repair would provide"""
        if not alternatives:
            return 0.1  # Minimal improvement without alternatives
        
        current_score = issue.confidence_score
        best_alternative_score = max(alt['confidence'] for alt in alternatives)
        
        improvement = best_alternative_score - current_score
        return max(0.0, min(1.0, improvement))
    
    def _calculate_projected_improvement(self, repair_actions: List[Dict[str, Any]], 
                                       assessment: QualityAssessment) -> float:
        """Calculate projected quality improvement from repair plan"""
        if not repair_actions:
            return 0.0
        
        total_improvement = sum(action['estimated_improvement'] for action in repair_actions)
        base_score = assessment.quality_score
        
        # Conservative estimate: 50% of theoretical improvement
        projected_improvement = (total_improvement * 0.5) / len(repair_actions)
        
        return min(0.3, projected_improvement)  # Cap at 30% improvement