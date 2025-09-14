"""
Segment Worklist System for Ensemble Transcription
U7 Upgrade: Tracks low-confidence windows for targeted reprocessing
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import hashlib

# Configure logging
worklist_logger = logging.getLogger(__name__)

@dataclass
class SegmentFlag:
    """Represents a flagged segment for reprocessing"""
    segment_id: str
    start_time: float
    end_time: float
    current_confidence: float
    flag_reason: str
    flag_source: str  # 'automatic' or 'manual'
    flag_timestamp: str
    processing_priority: int  # 1-10, higher = more important
    reprocess_count: int = 0
    last_reprocess_timestamp: Optional[str] = None
    original_transcript: str = ""
    improved_transcript: Optional[str] = None
    improvement_score: Optional[float] = None
    notes: str = ""

@dataclass
class WorklistEntry:
    """Represents a complete file's segment worklist"""
    file_path: str
    file_hash: str
    run_id: str
    creation_timestamp: str
    last_update_timestamp: str
    total_segments_flagged: int
    segments_processed: int
    segments_improved: int
    average_improvement: float
    confidence_threshold_used: float
    flagged_segments: List[SegmentFlag]

class SegmentWorklistManager:
    """
    Manages segment worklists for tracking and reprocessing low-confidence segments.
    Provides intelligent prioritization and progress tracking.
    """
    
    def __init__(self, worklist_dir: Optional[str] = None, 
                 confidence_threshold: float = 0.65,
                 auto_flag_enabled: bool = True):
        """
        Initialize segment worklist manager.
        
        Args:
            worklist_dir: Directory for storing worklist files
            confidence_threshold: Threshold below which segments are auto-flagged
            auto_flag_enabled: Whether to automatically flag low-confidence segments
        """
        if worklist_dir is None:
            worklist_dir = os.path.join(os.getcwd(), "artifacts", "worklists")
        
        self.worklist_dir = Path(worklist_dir)
        self.worklist_dir.mkdir(parents=True, exist_ok=True)
        
        self.confidence_threshold = confidence_threshold
        self.auto_flag_enabled = auto_flag_enabled
        
        # Statistics tracking
        self.stats = {
            'total_files_processed': 0,
            'total_segments_flagged': 0,
            'total_segments_reprocessed': 0,
            'total_segments_improved': 0,
            'average_improvement_score': 0.0
        }
        
        worklist_logger.info(f"Initialized segment worklist manager: {worklist_dir}")
        worklist_logger.info(f"Confidence threshold: {confidence_threshold}, Auto-flag: {auto_flag_enabled}")
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate hash for file to use as worklist identifier."""
        try:
            stat = os.stat(file_path)
            file_info = f"{stat.st_size}_{stat.st_mtime}_{Path(file_path).name}"
            return hashlib.md5(file_info.encode()).hexdigest()[:16]
        except Exception as e:
            worklist_logger.warning(f"Failed to hash file {file_path}: {e}")
            return hashlib.md5(str(file_path).encode()).hexdigest()[:16]
    
    def _get_worklist_path(self, file_path: str, run_id: str) -> Path:
        """Get path for worklist file."""
        file_hash = self._generate_file_hash(file_path)
        filename = f"worklist_{file_hash}_{run_id}.json"
        return self.worklist_dir / filename
    
    def create_worklist_from_candidates(self, file_path: str, run_id: str,
                                      candidates: List[Dict[str, Any]],
                                      winner_candidate: Dict[str, Any]) -> WorklistEntry:
        """
        Create a segment worklist from processing candidates.
        
        Args:
            file_path: Path to original file
            run_id: Processing run identifier
            candidates: List of all candidate transcripts
            winner_candidate: Selected winner candidate
            
        Returns:
            Created worklist entry
        """
        current_time = datetime.now().isoformat()
        file_hash = self._generate_file_hash(file_path)
        
        # Extract segments from winner candidate
        segments = winner_candidate.get('segments', [])
        confidence_scores = winner_candidate.get('confidence_scores', {})
        
        # Create flagged segments list
        flagged_segments = []
        
        for i, segment in enumerate(segments):
            segment_id = f"seg_{i:04d}"
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', 0.0)
            transcript = segment.get('text', '')
            
            # Get confidence score (try multiple sources)
            confidence = self._extract_segment_confidence(segment, confidence_scores, i)
            
            # Auto-flag if below threshold
            if self.auto_flag_enabled and confidence < self.confidence_threshold:
                flag_reason = f"Low confidence ({confidence:.3f} < {self.confidence_threshold})"
                priority = self._calculate_priority(confidence, end_time - start_time)
                
                flagged_segments.append(SegmentFlag(
                    segment_id=segment_id,
                    start_time=start_time,
                    end_time=end_time,
                    current_confidence=confidence,
                    flag_reason=flag_reason,
                    flag_source="automatic",
                    flag_timestamp=current_time,
                    processing_priority=priority,
                    original_transcript=transcript
                ))
        
        # Create worklist entry
        worklist_entry = WorklistEntry(
            file_path=file_path,
            file_hash=file_hash,
            run_id=run_id,
            creation_timestamp=current_time,
            last_update_timestamp=current_time,
            total_segments_flagged=len(flagged_segments),
            segments_processed=0,
            segments_improved=0,
            average_improvement=0.0,
            confidence_threshold_used=self.confidence_threshold,
            flagged_segments=flagged_segments
        )
        
        # Save to disk
        self._save_worklist(worklist_entry)
        
        # Update statistics
        self.stats['total_files_processed'] += 1
        self.stats['total_segments_flagged'] += len(flagged_segments)
        
        worklist_logger.info(f"Created worklist for {file_path}: {len(flagged_segments)} segments flagged")
        return worklist_entry
    
    def _extract_segment_confidence(self, segment: Dict[str, Any], 
                                  confidence_scores: Dict[str, Any], 
                                  index: int) -> float:
        """Extract confidence score for a segment from various sources."""
        # Try segment-level confidence first
        if 'confidence' in segment:
            return float(segment['confidence'])
        
        # Try confidence scores by index
        if 'segment_confidences' in confidence_scores and index < len(confidence_scores['segment_confidences']):
            return float(confidence_scores['segment_confidences'][index])
        
        # Try overall confidence
        if 'final_score' in confidence_scores:
            return float(confidence_scores['final_score'])
        
        # Default fallback
        return 0.5
    
    def _calculate_priority(self, confidence: float, duration: float) -> int:
        """
        Calculate processing priority for a segment.
        
        Args:
            confidence: Current confidence score (0-1)
            duration: Segment duration in seconds
            
        Returns:
            Priority level (1-10, higher = more important)
        """
        # Base priority on confidence (lower confidence = higher priority)
        conf_priority = max(1, int((1.0 - confidence) * 7) + 1)
        
        # Adjust for segment duration (longer segments get slight priority boost)
        if duration > 10.0:  # Long segments
            conf_priority = min(10, conf_priority + 1)
        elif duration < 2.0:  # Very short segments might be less important
            conf_priority = max(1, conf_priority - 1)
        
        return conf_priority
    
    def manually_flag_segment(self, file_path: str, run_id: str,
                            start_time: float, end_time: float,
                            reason: str, priority: int = 5) -> bool:
        """
        Manually flag a segment for reprocessing.
        
        Args:
            file_path: Path to original file
            run_id: Processing run identifier
            start_time: Segment start time
            end_time: Segment end time
            reason: Reason for flagging
            priority: Processing priority (1-10)
            
        Returns:
            True if successfully flagged, False otherwise
        """
        try:
            worklist_path = self._get_worklist_path(file_path, run_id)
            
            if not worklist_path.exists():
                worklist_logger.error(f"Worklist not found: {worklist_path}")
                return False
            
            # Load existing worklist
            worklist = self._load_worklist(worklist_path)
            if not worklist:
                return False
            
            # Create new flag
            segment_id = f"manual_{int(start_time)}_{int(end_time)}"
            current_time = datetime.now().isoformat()
            
            new_flag = SegmentFlag(
                segment_id=segment_id,
                start_time=start_time,
                end_time=end_time,
                current_confidence=0.0,  # Manual flags assume low confidence
                flag_reason=reason,
                flag_source="manual",
                flag_timestamp=current_time,
                processing_priority=priority,
                notes=f"Manually flagged: {reason}"
            )
            
            # Add to worklist
            worklist.flagged_segments.append(new_flag)
            worklist.total_segments_flagged += 1
            worklist.last_update_timestamp = current_time
            
            # Save updated worklist
            self._save_worklist(worklist)
            
            worklist_logger.info(f"Manually flagged segment {start_time}-{end_time} in {file_path}")
            return True
            
        except Exception as e:
            worklist_logger.error(f"Error manually flagging segment: {e}")
            return False
    
    def get_prioritized_segments(self, file_path: str, run_id: str,
                               max_segments: Optional[int] = None) -> List[SegmentFlag]:
        """
        Get prioritized list of segments for reprocessing.
        
        Args:
            file_path: Path to original file
            run_id: Processing run identifier
            max_segments: Maximum number of segments to return
            
        Returns:
            List of segments sorted by priority
        """
        worklist_path = self._get_worklist_path(file_path, run_id)
        
        if not worklist_path.exists():
            return []
        
        worklist = self._load_worklist(worklist_path)
        if not worklist:
            return []
        
        # Filter unprocessed segments
        unprocessed_segments = [
            seg for seg in worklist.flagged_segments 
            if seg.reprocess_count == 0
        ]
        
        # Sort by priority (highest first), then by confidence (lowest first)
        sorted_segments = sorted(
            unprocessed_segments,
            key=lambda s: (-s.processing_priority, s.current_confidence)
        )
        
        if max_segments:
            sorted_segments = sorted_segments[:max_segments]
        
        return sorted_segments
    
    def mark_segment_processed(self, file_path: str, run_id: str,
                             segment_id: str, new_transcript: str,
                             new_confidence: float) -> bool:
        """
        Mark a segment as processed with results.
        
        Args:
            file_path: Path to original file
            run_id: Processing run identifier
            segment_id: Segment identifier
            new_transcript: Improved transcript
            new_confidence: New confidence score
            
        Returns:
            True if successfully updated, False otherwise
        """
        try:
            worklist_path = self._get_worklist_path(file_path, run_id)
            
            if not worklist_path.exists():
                return False
            
            worklist = self._load_worklist(worklist_path)
            if not worklist:
                return False
            
            # Find and update segment
            for segment in worklist.flagged_segments:
                if segment.segment_id == segment_id:
                    segment.improved_transcript = new_transcript
                    segment.reprocess_count += 1
                    segment.last_reprocess_timestamp = datetime.now().isoformat()
                    
                    # Calculate improvement score
                    improvement = new_confidence - segment.current_confidence
                    segment.improvement_score = improvement
                    
                    # Update worklist statistics
                    worklist.segments_processed += 1
                    if improvement > 0.05:  # Significant improvement threshold
                        worklist.segments_improved += 1
                    
                    # Recalculate average improvement
                    total_improvements = sum(
                        seg.improvement_score for seg in worklist.flagged_segments
                        if seg.improvement_score is not None
                    )
                    worklist.average_improvement = total_improvements / max(1, worklist.segments_processed)
                    
                    worklist.last_update_timestamp = datetime.now().isoformat()
                    
                    # Save updated worklist
                    self._save_worklist(worklist)
                    
                    # Update global statistics
                    self.stats['total_segments_reprocessed'] += 1
                    if improvement > 0.05:
                        self.stats['total_segments_improved'] += 1
                    
                    worklist_logger.info(f"Marked segment {segment_id} as processed (improvement: {improvement:+.3f})")
                    return True
            
            worklist_logger.warning(f"Segment {segment_id} not found in worklist")
            return False
            
        except Exception as e:
            worklist_logger.error(f"Error marking segment as processed: {e}")
            return False
    
    def get_worklist_summary(self, file_path: str, run_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of worklist status."""
        worklist_path = self._get_worklist_path(file_path, run_id)
        
        if not worklist_path.exists():
            return None
        
        worklist = self._load_worklist(worklist_path)
        if not worklist:
            return None
        
        return {
            'file_path': worklist.file_path,
            'run_id': worklist.run_id,
            'total_flagged': worklist.total_segments_flagged,
            'processed': worklist.segments_processed,
            'improved': worklist.segments_improved,
            'average_improvement': worklist.average_improvement,
            'pending_count': len([s for s in worklist.flagged_segments if s.reprocess_count == 0]),
            'creation_time': worklist.creation_timestamp,
            'last_update': worklist.last_update_timestamp
        }
    
    def _save_worklist(self, worklist: WorklistEntry):
        """Save worklist to disk."""
        worklist_path = self._get_worklist_path(worklist.file_path, worklist.run_id)
        
        try:
            with open(worklist_path, 'w') as f:
                json.dump(asdict(worklist), f, indent=2)
            worklist_logger.debug(f"Saved worklist: {worklist_path}")
        except Exception as e:
            worklist_logger.error(f"Error saving worklist: {e}")
    
    def _load_worklist(self, worklist_path: Path) -> Optional[WorklistEntry]:
        """Load worklist from disk."""
        try:
            with open(worklist_path, 'r') as f:
                data = json.load(f)
            
            # Convert flagged_segments back to SegmentFlag objects
            flagged_segments = [SegmentFlag(**seg) for seg in data['flagged_segments']]
            data['flagged_segments'] = flagged_segments
            
            return WorklistEntry(**data)
        except Exception as e:
            worklist_logger.error(f"Error loading worklist {worklist_path}: {e}")
            return None
    
    def list_all_worklists(self) -> List[Dict[str, Any]]:
        """Get summary of all worklists."""
        worklists = []
        
        for worklist_file in self.worklist_dir.glob("worklist_*.json"):
            try:
                worklist = self._load_worklist(worklist_file)
                if worklist:
                    summary = {
                        'file_path': worklist.file_path,
                        'run_id': worklist.run_id,
                        'file_hash': worklist.file_hash,
                        'total_flagged': worklist.total_segments_flagged,
                        'processed': worklist.segments_processed,
                        'improved': worklist.segments_improved,
                        'pending': len([s for s in worklist.flagged_segments if s.reprocess_count == 0]),
                        'creation_time': worklist.creation_timestamp,
                        'last_update': worklist.last_update_timestamp
                    }
                    worklists.append(summary)
            except Exception as e:
                worklist_logger.warning(f"Error reading worklist {worklist_file}: {e}")
        
        # Sort by creation time (newest first)
        return sorted(worklists, key=lambda x: x['creation_time'], reverse=True)
    
    def cleanup_old_worklists(self, days_old: int = 30) -> int:
        """
        Clean up worklists older than specified days.
        
        Args:
            days_old: Number of days after which to remove worklists
            
        Returns:
            Number of worklists removed
        """
        cutoff_time = datetime.now() - timedelta(days=days_old)
        removed_count = 0
        
        for worklist_file in self.worklist_dir.glob("worklist_*.json"):
            try:
                worklist = self._load_worklist(worklist_file)
                if worklist:
                    creation_time = datetime.fromisoformat(worklist.creation_timestamp)
                    if creation_time < cutoff_time:
                        worklist_file.unlink()
                        removed_count += 1
                        worklist_logger.info(f"Removed old worklist: {worklist_file}")
            except Exception as e:
                worklist_logger.warning(f"Error during cleanup of {worklist_file}: {e}")
        
        return removed_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive worklist statistics."""
        return {
            'global': self.stats.copy(),
            'current_session': {
                'total_files_processed': self.stats['total_files_processed'],
                'total_segments_flagged': self.stats['total_segments_flagged'],
                'average_confidence_threshold': self.confidence_threshold
            }
        }
    
    def list_available_worklists(self) -> List[Dict[str, Any]]:
        """List all available worklists with summary information."""
        worklists = []
        
        for worklist_file in self.worklist_dir.glob("worklist_*.json"):
            try:
                worklist = self._load_worklist(worklist_file)
                if worklist:
                    summary = {
                        'file_path': worklist.file_path,
                        'run_id': worklist.run_id,
                        'total_flagged': worklist.total_segments_flagged,
                        'processed': worklist.segments_processed,
                        'improved': worklist.segments_improved,
                        'pending': len([s for s in worklist.flagged_segments if s.reprocess_count == 0]),
                        'creation_time': worklist.creation_timestamp,
                        'last_update': worklist.last_update_timestamp
                    }
                    worklists.append(summary)
            except Exception as e:
                worklist_logger.warning(f"Error reading worklist {worklist_file}: {e}")
        
        return sorted(worklists, key=lambda x: x['creation_time'], reverse=True)
    
    def load_worklist(self, file_path: str, run_id: str) -> Optional[WorklistEntry]:
        """Load a specific worklist by file path and run ID."""
        worklist_path = self._get_worklist_path(file_path, run_id)
        
        if not worklist_path.exists():
            return None
        
        return self._load_worklist(worklist_path)
    
    def unflag_segment(self, file_path: str, run_id: str, segment_id: str) -> bool:
        """Remove a flag from a specific segment."""
        try:
            worklist_path = self._get_worklist_path(file_path, run_id)
            
            if not worklist_path.exists():
                return False
            
            worklist = self._load_worklist(worklist_path)
            if not worklist:
                return False
            
            # Find and remove the segment
            original_count = len(worklist.flagged_segments)
            worklist.flagged_segments = [s for s in worklist.flagged_segments if s.segment_id != segment_id]
            
            if len(worklist.flagged_segments) < original_count:
                worklist.total_segments_flagged = len(worklist.flagged_segments)
                worklist.last_update_timestamp = datetime.now().isoformat()
                self._save_worklist(worklist)
                
                worklist_logger.info(f"Unflagged segment {segment_id} from {file_path}")
                return True
            
            return False
        except Exception as e:
            worklist_logger.error(f"Error unflagging segment {segment_id}: {e}")
            return False
    
    def flag_segment_manually(self, file_path: str, run_id: str, start_time: float, 
                            end_time: float, reason: str, priority: int = 5) -> bool:
        """Manually flag a segment for reprocessing."""
        return self.manually_flag_segment(file_path, run_id, start_time, end_time, reason, priority)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all worklists."""
        return self.stats.copy()


# Global worklist manager instance
_worklist_manager: Optional[SegmentWorklistManager] = None

def get_worklist_manager() -> SegmentWorklistManager:
    """Get or create global worklist manager instance."""
    global _worklist_manager
    if _worklist_manager is None:
        _worklist_manager = SegmentWorklistManager()
    return _worklist_manager