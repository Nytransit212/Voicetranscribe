#!/usr/bin/env python3
"""
Smoke Check Script for Determinism Validation

This script validates that the ensemble transcription system produces identical
outputs for identical inputs and configurations, ensuring complete reproducibility.
"""

import os
import sys
import json
import time
import hashlib
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ensemble_manager import EnsembleManager
from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class DeterminismTestResult:
    """Result of determinism validation test"""
    test_name: str
    run1_manifest_path: str
    run2_manifest_path: str
    identical_words: bool
    timestamp_drift_ms: float
    manifest_differences: List[str]
    test_passed: bool
    processing_time_1: float
    processing_time_2: float
    error_message: Optional[str] = None

class DeterminismValidator:
    """Validates deterministic processing across multiple runs"""
    
    def __init__(self, test_sample_path: str, config_path: str = "config/config.yaml"):
        """
        Initialize determinism validator
        
        Args:
            test_sample_path: Path to test audio sample
            config_path: Path to configuration file
        """
        self.test_sample_path = Path(test_sample_path)
        self.config_path = Path(config_path)
        self.logger = create_enhanced_logger("determinism_validator")
        
        # Validation configuration
        self.timestamp_tolerance_ms = 5.0  # 5ms tolerance for timestamps
        self.max_processing_time_ratio = 2.0  # Processing times within 2x of each other
        
        if not self.test_sample_path.exists():
            raise FileNotFoundError(f"Test sample not found: {self.test_sample_path}")
        
        self.logger.info(f"Initialized determinism validator with sample: {self.test_sample_path}")
    
    def run_determinism_test(self, test_name: str = "smoke_check") -> DeterminismTestResult:
        """
        Run complete determinism test with two identical runs
        
        Args:
            test_name: Name of the test for logging
            
        Returns:
            DeterminismTestResult with validation results
        """
        self.logger.info(f"Starting determinism test: {test_name}")
        
        # Create temporary directories for each run
        with tempfile.TemporaryDirectory(prefix=f"determinism_test_{test_name}_") as temp_dir:
            temp_path = Path(temp_dir)
            
            run1_dir = temp_path / "run1"
            run2_dir = temp_path / "run2"
            run1_dir.mkdir()
            run2_dir.mkdir()
            
            try:
                # Run first transcription
                self.logger.info("Executing first run...")
                start_time_1 = time.time()
                result1 = self._run_transcription(run1_dir, test_name + "_run1")
                processing_time_1 = time.time() - start_time_1
                
                # Small delay to ensure different timestamps if non-deterministic
                time.sleep(0.1)
                
                # Run second transcription
                self.logger.info("Executing second run...")
                start_time_2 = time.time()
                result2 = self._run_transcription(run2_dir, test_name + "_run2")
                processing_time_2 = time.time() - start_time_2
                
                # Compare results
                comparison_result = self._compare_results(result1, result2)
                
                # Create test result
                test_result = DeterminismTestResult(
                    test_name=test_name,
                    run1_manifest_path=str(result1['manifest_path']),
                    run2_manifest_path=str(result2['manifest_path']),
                    identical_words=comparison_result['identical_words'],
                    timestamp_drift_ms=comparison_result['max_timestamp_drift_ms'],
                    manifest_differences=comparison_result['manifest_differences'],
                    test_passed=comparison_result['test_passed'],
                    processing_time_1=processing_time_1,
                    processing_time_2=processing_time_2
                )
                
                # Log results
                self._log_test_results(test_result)
                
                return test_result
                
            except Exception as e:
                self.logger.error(f"Determinism test failed with exception: {e}")
                return DeterminismTestResult(
                    test_name=test_name,
                    run1_manifest_path="",
                    run2_manifest_path="",
                    identical_words=False,
                    timestamp_drift_ms=float('inf'),
                    manifest_differences=[f"Test failed with exception: {e}"],
                    test_passed=False,
                    processing_time_1=0.0,
                    processing_time_2=0.0,
                    error_message=str(e)
                )
    
    def _run_transcription(self, output_dir: Path, session_id: str) -> Dict[str, Any]:
        """Run a single transcription and return results"""
        
        # Create EnsembleManager with deterministic configuration
        manager = EnsembleManager(
            expected_speakers=2,
            noise_level='low',
            target_language='en',
            enable_versioning=True,
            domain='test',
            consensus_strategy='best_single_candidate',
            calibration_method='registry_based'
        )
        
        # Run transcription
        result = manager.process_video(
            video_path=str(self.test_sample_path)
        )
        
        # Find manifest file
        manifest_path = output_dir / "run_manifest.json"
        if not manifest_path.exists():
            # Look for manifest in subdirectory
            manifest_files = list(output_dir.rglob("run_manifest.json"))
            if manifest_files:
                manifest_path = manifest_files[0]
            else:
                raise FileNotFoundError(f"No manifest found in {output_dir}")
        
        return {
            'result': result,
            'manifest_path': manifest_path,
            'output_dir': output_dir
        }
    
    def _compare_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two transcription results for determinism"""
        
        try:
            # Load manifests
            with open(result1['manifest_path'], 'r') as f:
                manifest1 = json.load(f)
            
            with open(result2['manifest_path'], 'r') as f:
                manifest2 = json.load(f)
            
            # Compare word sequences
            words1 = self._extract_word_sequence(result1['result'])
            words2 = self._extract_word_sequence(result2['result'])
            
            identical_words = words1 == words2
            
            # Compare timestamps
            timestamps1 = self._extract_timestamps(result1['result'])
            timestamps2 = self._extract_timestamps(result2['result'])
            
            max_timestamp_drift = self._calculate_max_timestamp_drift(timestamps1, timestamps2)
            
            # Compare manifest contents (excluding timestamps)
            manifest_differences = self._compare_manifests(manifest1, manifest2)
            
            # Determine if test passed
            timestamp_acceptable = max_timestamp_drift <= self.timestamp_tolerance_ms
            manifest_acceptable = len(manifest_differences) == 0
            
            test_passed = identical_words and timestamp_acceptable and manifest_acceptable
            
            self.logger.info(
                f"Determinism comparison - Words identical: {identical_words}, "
                f"Max timestamp drift: {max_timestamp_drift:.2f}ms, "
                f"Manifest differences: {len(manifest_differences)}"
            )
            
            return {
                'identical_words': identical_words,
                'max_timestamp_drift_ms': max_timestamp_drift,
                'manifest_differences': manifest_differences,
                'test_passed': test_passed,
                'words1': words1,
                'words2': words2,
                'timestamps1': timestamps1,
                'timestamps2': timestamps2
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compare results: {e}")
            return {
                'identical_words': False,
                'max_timestamp_drift_ms': float('inf'),
                'manifest_differences': [f"Comparison failed: {e}"],
                'test_passed': False,
                'words1': [],
                'words2': [],
                'timestamps1': [],
                'timestamps2': []
            }
    
    def _extract_word_sequence(self, result: Dict[str, Any]) -> List[str]:
        """Extract word sequence from transcription result"""
        words = []
        
        try:
            if 'transcript' in result:
                # Simple text transcript
                text = result['transcript']
                words = text.split()
            elif 'segments' in result:
                # Segmented transcript
                for segment in result['segments']:
                    if 'text' in segment:
                        words.extend(segment['text'].split())
            elif 'fused_transcript' in result:
                # Fused transcript
                text = result['fused_transcript']
                words = text.split()
            
            # Normalize words (lowercase, remove punctuation)
            normalized_words = []
            for word in words:
                clean_word = ''.join(c.lower() for c in word if c.isalnum())
                if clean_word:
                    normalized_words.append(clean_word)
            
            return normalized_words
            
        except Exception as e:
            self.logger.error(f"Failed to extract word sequence: {e}")
            return []
    
    def _extract_timestamps(self, result: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Extract timing information from transcription result"""
        timestamps = []
        
        try:
            if 'segments' in result:
                for segment in result['segments']:
                    start = segment.get('start', 0.0)
                    end = segment.get('end', 0.0)
                    timestamps.append((start, end))
            elif 'words' in result:
                for word in result['words']:
                    start = word.get('start', 0.0)
                    end = word.get('end', 0.0)
                    timestamps.append((start, end))
            
            return timestamps
            
        except Exception as e:
            self.logger.error(f"Failed to extract timestamps: {e}")
            return []
    
    def _calculate_max_timestamp_drift(self, timestamps1: List[Tuple[float, float]], 
                                     timestamps2: List[Tuple[float, float]]) -> float:
        """Calculate maximum timestamp drift between two runs in milliseconds"""
        
        if len(timestamps1) != len(timestamps2):
            self.logger.warning("Timestamp sequences have different lengths")
            return float('inf')
        
        max_drift_ms = 0.0
        
        for (start1, end1), (start2, end2) in zip(timestamps1, timestamps2):
            start_drift = abs(start1 - start2) * 1000  # Convert to ms
            end_drift = abs(end1 - end2) * 1000
            
            max_drift_ms = max(max_drift_ms, start_drift, end_drift)
        
        return max_drift_ms
    
    def _compare_manifests(self, manifest1: Dict[str, Any], manifest2: Dict[str, Any]) -> List[str]:
        """Compare manifests excluding timestamp fields"""
        differences = []
        
        # Fields to exclude from comparison (timing-related)
        excluded_fields = {
            'started_at', 'completed_at', 'processing_duration_seconds',
            'last_validated_at', 'created_at', 'run_id', 'manifest_sha256'
        }
        
        try:
            # Compare top-level fields
            for key in set(manifest1.keys()) | set(manifest2.keys()):
                if key in excluded_fields:
                    continue
                
                if key not in manifest1:
                    differences.append(f"Field '{key}' missing from first manifest")
                elif key not in manifest2:
                    differences.append(f"Field '{key}' missing from second manifest")
                elif manifest1[key] != manifest2[key]:
                    # Special handling for complex fields
                    if key == 'artifacts':
                        artifact_diffs = self._compare_artifacts(manifest1[key], manifest2[key])
                        differences.extend(artifact_diffs)
                    else:
                        differences.append(f"Field '{key}' differs: {manifest1[key]} vs {manifest2[key]}")
            
            return differences
            
        except Exception as e:
            self.logger.error(f"Failed to compare manifests: {e}")
            return [f"Manifest comparison failed: {e}"]
    
    def _compare_artifacts(self, artifacts1: List[Dict[str, Any]], 
                          artifacts2: List[Dict[str, Any]]) -> List[str]:
        """Compare artifact lists excluding timing fields"""
        differences = []
        
        if len(artifacts1) != len(artifacts2):
            differences.append(f"Different number of artifacts: {len(artifacts1)} vs {len(artifacts2)}")
            return differences
        
        # Sort artifacts by path for consistent comparison
        sorted_artifacts1 = sorted(artifacts1, key=lambda x: x.get('path', ''))
        sorted_artifacts2 = sorted(artifacts2, key=lambda x: x.get('path', ''))
        
        excluded_artifact_fields = {'created_at'}
        
        for i, (art1, art2) in enumerate(zip(sorted_artifacts1, sorted_artifacts2)):
            for key in set(art1.keys()) | set(art2.keys()):
                if key in excluded_artifact_fields:
                    continue
                
                if key not in art1:
                    differences.append(f"Artifact {i}: field '{key}' missing from first")
                elif key not in art2:
                    differences.append(f"Artifact {i}: field '{key}' missing from second")
                elif art1[key] != art2[key]:
                    differences.append(f"Artifact {i}: field '{key}' differs: {art1[key]} vs {art2[key]}")
        
        return differences
    
    def _log_test_results(self, result: DeterminismTestResult):
        """Log detailed test results"""
        
        if result.test_passed:
            self.logger.info(f"✅ Determinism test PASSED: {result.test_name}")
        else:
            self.logger.error(f"❌ Determinism test FAILED: {result.test_name}")
        
        self.logger.info(f"Word sequences identical: {result.identical_words}")
        self.logger.info(f"Max timestamp drift: {result.timestamp_drift_ms:.2f}ms (tolerance: {self.timestamp_tolerance_ms}ms)")
        self.logger.info(f"Manifest differences: {len(result.manifest_differences)}")
        self.logger.info(f"Processing times: {result.processing_time_1:.2f}s vs {result.processing_time_2:.2f}s")
        
        if result.manifest_differences:
            self.logger.info("Manifest differences:")
            for diff in result.manifest_differences[:10]:  # Limit to first 10
                self.logger.info(f"  - {diff}")
        
        # Log structured telemetry
        self.logger.info("Determinism test complete",
            event_type="determinism_test_complete",
            metrics={
                "test_name": result.test_name,
                "test_passed": result.test_passed,
                "identical_words": result.identical_words,
                "timestamp_drift_ms": result.timestamp_drift_ms,
                "manifest_differences_count": len(result.manifest_differences),
                "processing_time_1": result.processing_time_1,
                "processing_time_2": result.processing_time_2,
                "processing_time_ratio": result.processing_time_2 / result.processing_time_1 if result.processing_time_1 > 0 else 0,
                "error_message": result.error_message
            })


def main():
    """Main entry point for smoke check script"""
    parser = argparse.ArgumentParser(description="Run determinism smoke check")
    parser.add_argument("--sample", "-s", required=True, help="Path to test audio sample")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--test-name", "-n", default="smoke_check", help="Name of the test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Initialize validator
        validator = DeterminismValidator(args.sample, args.config)
        
        # Run test
        result = validator.run_determinism_test(args.test_name)
        
        # Print summary
        print("\n" + "="*60)
        print(f"DETERMINISM TEST RESULTS: {result.test_name}")
        print("="*60)
        print(f"Overall: {'✅ PASSED' if result.test_passed else '❌ FAILED'}")
        print(f"Word sequences identical: {result.identical_words}")
        print(f"Max timestamp drift: {result.timestamp_drift_ms:.2f}ms")
        print(f"Manifest differences: {len(result.manifest_differences)}")
        print(f"Processing time ratio: {result.processing_time_2/result.processing_time_1:.2f}x")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        print("="*60)
        
        # Exit with appropriate code
        sys.exit(0 if result.test_passed else 1)
        
    except Exception as e:
        print(f"❌ Smoke check failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()