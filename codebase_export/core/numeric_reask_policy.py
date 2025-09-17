"""
Numeric Re-Ask Policy - Phase 4.3 Implementation
Entity-focused targeted re-decoding with cross-candidate agreement thresholds

Implements smart re-ask system for critical data accuracy insurance:
- Numeric/entity span detection
- Cross-candidate agreement thresholds  
- Targeted prompt engineering for re-ask
- Critical data validation
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum
import numpy as np

from .asr_providers.base import ASRProvider, ASRResult, ASRSegment, DecodeMode
from .asr_providers.factory import ASRProviderFactory
from .intelligent_controller import SegmentCandidate, SegmentAnalysis
from utils.enhanced_structured_logger import create_enhanced_logger

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Critical entity types requiring high accuracy"""
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    TIME = "time"
    NUMBER = "number"
    ADDRESS = "address"
    ALPHANUMERIC_CODE = "alphanumeric_code"
    PROPER_NAME = "proper_name"

@dataclass
class EntitySpan:
    """Detected entity span in transcript"""
    entity_type: EntityType
    text: str
    start_char: int
    end_char: int
    confidence: float
    detection_pattern: str
    context_before: str
    context_after: str
    validation_score: float = 0.0

@dataclass
class CrossCandidateAnalysis:
    """Analysis of entity consistency across multiple candidates"""
    entity_span: EntitySpan
    candidate_variations: List[Tuple[str, float]]  # (text, confidence) pairs
    agreement_score: float
    variation_count: int
    dominant_version: str
    conflict_severity: float
    requires_reask: bool
    reask_priority: int  # 1 (highest) to 5 (lowest)

@dataclass
class ReAskRequest:
    """Request for targeted re-decoding"""
    segment_id: str
    original_segment: SegmentAnalysis
    target_entities: List[EntitySpan]
    specialized_prompt: str
    decode_parameters: Dict[str, Any]
    priority: int
    max_attempts: int = 3

@dataclass
class ReAskResult:
    """Result from targeted re-decoding"""
    request: ReAskRequest
    improved_entities: List[EntitySpan]
    final_transcript: str
    confidence_improvement: float
    agreement_improvement: float
    attempts_used: int
    cost_incurred: float
    processing_time: float

class EntityDetector:
    """Advanced entity detection with pattern matching and validation"""
    
    def __init__(self):
        self.patterns = {
            EntityType.PHONE_NUMBER: [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',
                r'\b\+\d{1,3}\s?\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            ],
            EntityType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            EntityType.CURRENCY: [
                r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?dollars?\b',
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?USD\b'
            ],
            EntityType.PERCENTAGE: [
                r'\b\d{1,3}(?:\.\d{1,2})?\s?%\b',
                r'\b\d{1,3}(?:\.\d{1,2})?\s?percent\b'
            ],
            EntityType.DATE: [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
            ],
            EntityType.TIME: [
                r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b',
                r'\b(?:at\s+)?\d{1,2}\s?(?:o\'clock|oclock)\b'
            ],
            EntityType.NUMBER: [
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',
                r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b'
            ],
            EntityType.ALPHANUMERIC_CODE: [
                r'\b[A-Z]\d+[A-Z]?\b',
                r'\b\d+[A-Z]+\d*\b',
                r'\b[A-Z]{2,}\d{2,}\b'
            ]
        }
    
    def detect_entities(self, text: str, confidence: float = 1.0) -> List[EntitySpan]:
        """Detect all critical entities in text"""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract context
                    context_start = max(0, match.start() - 20)
                    context_end = min(len(text), match.end() + 20)
                    context_before = text[context_start:match.start()].strip()
                    context_after = text[match.end():context_end].strip()
                    
                    # Validate entity
                    validation_score = self._validate_entity(
                        entity_type, match.group(), context_before, context_after
                    )
                    
                    entity = EntitySpan(
                        entity_type=entity_type,
                        text=match.group(),
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=confidence,
                        detection_pattern=pattern,
                        context_before=context_before,
                        context_after=context_after,
                        validation_score=validation_score
                    )
                    entities.append(entity)
        
        return sorted(entities, key=lambda x: x.start_char)
    
    def _validate_entity(self, entity_type: EntityType, text: str, 
                        context_before: str, context_after: str) -> float:
        """Validate detected entity using context and format rules"""
        score = 0.5  # Base score
        
        if entity_type == EntityType.PHONE_NUMBER:
            # Check for common phone number contexts
            if any(word in context_before.lower() for word in 
                   ['call', 'phone', 'number', 'contact', 'reach']):
                score += 0.3
            if len(re.findall(r'\d', text)) == 10:  # US phone number
                score += 0.2
                
        elif entity_type == EntityType.CURRENCY:
            # Check for financial contexts
            if any(word in context_before.lower() for word in 
                   ['cost', 'price', 'pay', 'spend', 'budget', 'revenue']):
                score += 0.3
                
        elif entity_type == EntityType.PERCENTAGE:
            # Check for statistical contexts
            if any(word in context_before.lower() for word in 
                   ['rate', 'growth', 'increase', 'decrease', 'percent']):
                score += 0.3
                
        return min(1.0, score)

class NumericReAskPolicy:
    """
    Main Numeric Re-Ask Policy engine implementing Phase 4.3 functionality
    
    Provides entity-focused targeted re-decoding with intelligent gating
    """
    
    def __init__(self,
                 agreement_threshold: float = 0.7,
                 confidence_threshold: float = 0.8,
                 max_reask_attempts: int = 3,
                 enable_cost_tracking: bool = True):
        """
        Initialize Numeric Re-Ask Policy
        
        Args:
            agreement_threshold: Minimum agreement across candidates to skip re-ask
            confidence_threshold: Minimum confidence to skip re-ask
            max_reask_attempts: Maximum re-ask attempts per entity
            enable_cost_tracking: Whether to track re-ask costs
        """
        self.agreement_threshold = agreement_threshold
        self.confidence_threshold = confidence_threshold
        self.max_reask_attempts = max_reask_attempts
        self.enable_cost_tracking = enable_cost_tracking
        
        self.entity_detector = EntityDetector()
        self.asr_factory = ASRProviderFactory()
        self.logger = create_enhanced_logger("numeric_reask_policy")
        
        # Cost tracking
        self.total_reask_cost = 0.0
        self.reask_statistics = defaultdict(int)
        
        # Specialized prompts for entity types
        self.entity_prompts = {
            EntityType.PHONE_NUMBER: "Focus on phone numbers. Spell out digits clearly.",
            EntityType.EMAIL: "Focus on email addresses. Spell out @ and domain names clearly.",
            EntityType.CURRENCY: "Focus on monetary amounts. Include currency symbols and decimals.",
            EntityType.PERCENTAGE: "Focus on percentages. Include decimal points and percent signs.",
            EntityType.DATE: "Focus on dates. Include full month names and year.",
            EntityType.TIME: "Focus on times. Include AM/PM indicators.",
            EntityType.NUMBER: "Focus on numbers. Spell out large numbers clearly.",
            EntityType.ALPHANUMERIC_CODE: "Focus on codes. Spell out each character clearly."
        }
    
    def analyze_segment_entities(self, segment_analysis: SegmentAnalysis) -> List[CrossCandidateAnalysis]:
        """
        Analyze entities across all candidates in a segment
        
        Returns list of entities that may require re-asking
        """
        if not segment_analysis.candidates:
            return []
        
        # Extract entities from all candidates
        all_entities = {}
        for candidate in segment_analysis.candidates:
            entities = self.entity_detector.detect_entities(
                candidate.result.full_text, 
                candidate.calibrated_confidence
            )
            all_entities[candidate.provider] = entities
        
        # Group entities by position and type
        entity_groups = self._group_overlapping_entities(all_entities)
        
        # Analyze cross-candidate agreement for each entity group
        cross_analyses = []
        for group in entity_groups:
            analysis = self._analyze_entity_agreement(group, segment_analysis)
            if analysis:
                cross_analyses.append(analysis)
        
        return cross_analyses
    
    def generate_reask_requests(self, 
                               segment_analysis: SegmentAnalysis,
                               entity_analyses: List[CrossCandidateAnalysis]) -> List[ReAskRequest]:
        """Generate targeted re-ask requests for low-agreement entities"""
        
        reask_requests = []
        
        # Filter entities that need re-asking
        critical_entities = [
            analysis for analysis in entity_analyses 
            if analysis.requires_reask and analysis.reask_priority <= 3
        ]
        
        if not critical_entities:
            return reask_requests
        
        # Group entities by type for efficient re-asking
        entities_by_type = defaultdict(list)
        for analysis in critical_entities:
            entities_by_type[analysis.entity_span.entity_type].append(analysis)
        
        # Create re-ask requests per entity type
        for entity_type, entity_list in entities_by_type.items():
            if len(entity_list) > 0:
                request = self._create_reask_request(
                    segment_analysis, entity_type, entity_list
                )
                reask_requests.append(request)
        
        # Sort by priority (highest first)
        reask_requests.sort(key=lambda x: x.priority)
        
        return reask_requests
    
    def execute_reask(self, request: ReAskRequest) -> ReAskResult:
        """Execute targeted re-decoding for specific entities"""
        start_time = time.time()
        best_result = None
        attempts = 0
        total_cost = 0.0
        
        # Get original audio segment
        segment = request.original_segment
        
        while attempts < request.max_attempts:
            attempts += 1
            
            try:
                # Configure specialized ASR parameters
                asr_params = {
                    **request.decode_parameters,
                    'prompt': request.specialized_prompt,
                    'temperature': 0.0,  # Deterministic for critical data
                    'language': 'en'
                }
                
                # Use highest accuracy provider (OpenAI Whisper)
                provider = self.asr_factory.create_provider('openai')
                
                # Perform targeted re-decode
                # Note: In practice, would need audio segment extraction and re-transcription
                # For now, simulate improvement by using the best existing candidate
                if not segment.best_candidate:
                    continue
                    
                result = segment.best_candidate.result
                
                # Extract entities from new result
                new_entities = self.entity_detector.detect_entities(
                    result.full_text, result.confidence
                )
                
                # Calculate improvement metrics
                improvement = self._calculate_improvement(
                    request.target_entities, new_entities
                )
                
                if improvement['confidence_improvement'] > 0.1:
                    best_result = {
                        'improved_entities': new_entities,
                        'transcript': result.full_text,
                        'confidence_improvement': improvement['confidence_improvement'],
                        'agreement_improvement': improvement['agreement_improvement']
                    }
                    break
                    
                # Track costs
                if self.enable_cost_tracking:
                    segment_duration = segment.segment_duration
                    cost = segment_duration * 0.006  # OpenAI Whisper pricing
                    total_cost += cost
                    
            except Exception as e:
                self.logger.error(f"Re-ask attempt {attempts} failed: {str(e)}")
                continue
        
        processing_time = time.time() - start_time
        
        # Create result
        if best_result:
            result = ReAskResult(
                request=request,
                improved_entities=best_result['improved_entities'],
                final_transcript=best_result['transcript'],
                confidence_improvement=best_result['confidence_improvement'],
                agreement_improvement=best_result['agreement_improvement'],
                attempts_used=attempts,
                cost_incurred=total_cost,
                processing_time=processing_time
            )
        else:
            # No improvement found
            result = ReAskResult(
                request=request,
                improved_entities=request.target_entities,
                final_transcript=segment.best_candidate.result.full_text if segment.best_candidate else "",
                confidence_improvement=0.0,
                agreement_improvement=0.0,
                attempts_used=attempts,
                cost_incurred=total_cost,
                processing_time=processing_time
            )
        
        # Update statistics
        self.reask_statistics['total_reasks'] += 1
        self.total_reask_cost += total_cost
        
        return result
    
    def _group_overlapping_entities(self, all_entities: Dict[str, List[EntitySpan]]) -> List[List[Tuple[str, EntitySpan]]]:
        """Group overlapping entities across candidates"""
        groups = []
        processed = set()
        
        for provider, entities in all_entities.items():
            for entity in entities:
                if (provider, entity.start_char, entity.end_char) in processed:
                    continue
                
                # Find overlapping entities from other candidates
                group = [(provider, entity)]
                processed.add((provider, entity.start_char, entity.end_char))
                
                for other_provider, other_entities in all_entities.items():
                    if other_provider == provider:
                        continue
                    
                    for other_entity in other_entities:
                        if (other_provider, other_entity.start_char, other_entity.end_char) in processed:
                            continue
                        
                        # Check for overlap (allow some tolerance)
                        if self._entities_overlap(entity, other_entity):
                            group.append((other_provider, other_entity))
                            processed.add((other_provider, other_entity.start_char, other_entity.end_char))
                
                if len(group) > 1:  # Only include groups with multiple candidates
                    groups.append(group)
        
        return groups
    
    def _entities_overlap(self, entity1: EntitySpan, entity2: EntitySpan, tolerance: int = 10) -> bool:
        """Check if two entities overlap within tolerance"""
        if entity1.entity_type != entity2.entity_type:
            return False
        
        return not (entity1.end_char + tolerance < entity2.start_char or 
                   entity2.end_char + tolerance < entity1.start_char)
    
    def _analyze_entity_agreement(self, entity_group: List[Tuple[str, EntitySpan]], 
                                segment_analysis: SegmentAnalysis) -> Optional[CrossCandidateAnalysis]:
        """Analyze agreement for a group of overlapping entities"""
        
        if len(entity_group) < 2:
            return None
        
        # Extract entity texts and confidences
        entity_texts = [entity.text for _, entity in entity_group]
        entity_confidences = [entity.confidence for _, entity in entity_group]
        
        # Calculate agreement score
        text_counter = Counter(entity_texts)
        most_common_text, most_common_count = text_counter.most_common(1)[0]
        agreement_score = most_common_count / len(entity_texts)
        
        # Calculate variation metrics
        variation_count = len(set(entity_texts))
        conflict_severity = 1.0 - agreement_score
        
        # Determine if re-ask is needed
        avg_confidence = float(np.mean(entity_confidences))
        requires_reask = bool(agreement_score < self.agreement_threshold or 
                         avg_confidence < self.confidence_threshold)
        
        # Calculate priority (1 = highest, 5 = lowest)
        priority = self._calculate_reask_priority(
            entity_group[0][1].entity_type, 
            agreement_score, 
            avg_confidence, 
            conflict_severity
        )
        
        # Use the first entity as representative
        representative_entity = entity_group[0][1]
        
        return CrossCandidateAnalysis(
            entity_span=representative_entity,
            candidate_variations=list(zip(entity_texts, entity_confidences)),
            agreement_score=agreement_score,
            variation_count=variation_count,
            dominant_version=most_common_text,
            conflict_severity=conflict_severity,
            requires_reask=requires_reask,
            reask_priority=priority
        )
    
    def _calculate_reask_priority(self, entity_type: EntityType, agreement_score: float, 
                                confidence: float, conflict_severity: float) -> int:
        """Calculate re-ask priority based on entity importance and disagreement"""
        
        # Base priority by entity type (lower = higher priority)
        type_priorities = {
            EntityType.PHONE_NUMBER: 1,
            EntityType.EMAIL: 1,
            EntityType.CURRENCY: 2,
            EntityType.PERCENTAGE: 2,
            EntityType.DATE: 2,
            EntityType.TIME: 3,
            EntityType.ALPHANUMERIC_CODE: 1,
            EntityType.NUMBER: 3,
            EntityType.PROPER_NAME: 4,
            EntityType.ADDRESS: 2
        }
        
        base_priority = type_priorities.get(entity_type, 5)
        
        # Adjust based on agreement and confidence
        if agreement_score < 0.3 or confidence < 0.5:
            base_priority = max(1, base_priority - 2)  # Boost priority
        elif agreement_score > 0.8 and confidence > 0.9:
            base_priority = min(5, base_priority + 1)  # Lower priority
        
        return base_priority
    
    def _create_reask_request(self, segment_analysis: SegmentAnalysis,
                            entity_type: EntityType, 
                            entity_analyses: List[CrossCandidateAnalysis]) -> ReAskRequest:
        """Create specialized re-ask request for entity type"""
        
        # Get specialized prompt
        base_prompt = self.entity_prompts.get(entity_type, "Focus on accuracy.")
        
        # Add context from problematic entities
        entity_contexts = []
        for analysis in entity_analyses:
            context = f"{analysis.entity_span.context_before} {analysis.entity_span.text} {analysis.entity_span.context_after}".strip()
            entity_contexts.append(context)
        
        specialized_prompt = f"{base_prompt} Pay special attention to: {' | '.join(entity_contexts[:3])}"
        
        # Configure decode parameters for maximum accuracy
        decode_params = {
            'beam_size': 5,
            'best_of': 5,
            'temperature': 0.0,
            'compression_ratio_threshold': 2.4,
            'log_prob_threshold': -1.0,
            'word_timestamps': True
        }
        
        # Calculate overall priority (minimum of individual priorities)
        min_priority = min(analysis.reask_priority for analysis in entity_analyses)
        
        # Extract target entities
        target_entities = [analysis.entity_span for analysis in entity_analyses]
        
        return ReAskRequest(
            segment_id=f"{segment_analysis.segment_start}_{segment_analysis.segment_end}",
            original_segment=segment_analysis,
            target_entities=target_entities,
            specialized_prompt=specialized_prompt,
            decode_parameters=decode_params,
            priority=min_priority,
            max_attempts=self.max_reask_attempts
        )
    
    def _calculate_improvement(self, original_entities: List[EntitySpan], 
                             new_entities: List[EntitySpan]) -> Dict[str, float]:
        """Calculate improvement metrics from re-asking"""
        
        if not original_entities:
            return {'confidence_improvement': 0.0, 'agreement_improvement': 0.0}
        
        # Calculate average confidence improvement
        original_confidence = np.mean([e.confidence for e in original_entities])
        new_confidence = np.mean([e.confidence for e in new_entities]) if new_entities else 0.0
        confidence_improvement = new_confidence - original_confidence
        
        # Calculate agreement improvement (simplified)
        # In practice, this would compare against other candidates
        agreement_improvement = confidence_improvement * 0.5  # Simplified metric
        
        return {
            'confidence_improvement': float(confidence_improvement),
            'agreement_improvement': float(agreement_improvement)
        }
    
    def get_reask_statistics(self) -> Dict[str, Any]:
        """Get comprehensive re-ask statistics"""
        return {
            'total_reasks_performed': self.reask_statistics['total_reasks'],
            'total_cost_incurred': self.total_reask_cost,
            'average_cost_per_reask': (self.total_reask_cost / max(1, self.reask_statistics['total_reasks'])),
            'reask_statistics': dict(self.reask_statistics)
        }

# Integration example for use with IntelligentController
def integrate_with_intelligent_controller():
    """
    Example integration with existing IntelligentController
    This shows how to add numeric re-ask capability to the existing pipeline
    """
    
    # Initialize the policy
    reask_policy = NumericReAskPolicy(
        agreement_threshold=0.7,
        confidence_threshold=0.8,
        max_reask_attempts=2
    )
    
    def enhanced_segment_processing(segment_analysis: SegmentAnalysis) -> SegmentAnalysis:
        """Enhanced processing with numeric re-ask policy"""
        
        # Analyze entities in the segment
        entity_analyses = reask_policy.analyze_segment_entities(segment_analysis)
        
        # Generate re-ask requests for problematic entities
        reask_requests = reask_policy.generate_reask_requests(segment_analysis, entity_analyses)
        
        # Execute re-asks for high-priority entities
        for request in reask_requests[:2]:  # Limit to top 2 requests
            if request.priority <= 2:  # Only high priority
                reask_result = reask_policy.execute_reask(request)
                
                # Update segment with improved entities if significant improvement
                if reask_result.confidence_improvement > 0.1:
                    # In practice, would integrate the improved result
                    logger.info(f"Improved entity accuracy for segment {request.segment_id}")
        
        return segment_analysis
    
    return enhanced_segment_processing

if __name__ == "__main__":
    # Example usage and testing
    reask_policy = NumericReAskPolicy()
    
    # Test entity detection
    test_text = "Please call me at 555-123-4567 or email john@company.com. The cost is $1,234.56 which is 15.5% above budget."
    entities = reask_policy.entity_detector.detect_entities(test_text)
    
    print("Detected Entities:")
    for entity in entities:
        print(f"  {entity.entity_type.value}: '{entity.text}' (confidence: {entity.confidence:.3f})")