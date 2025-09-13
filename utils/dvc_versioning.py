"""
DVC-based versioning and artifact tracking for ensemble pipeline.
Handles tracking of all processing artifacts and provides content-based verification.
"""

import os
import json
import hashlib
import shutil
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import subprocess
from dataclasses import dataclass, asdict
from utils.structured_logger import StructuredLogger

@dataclass
class ArtifactInfo:
    """Information about a versioned artifact"""
    path: str
    hash: str
    size: int
    timestamp: str
    stage: str
    type: str  # 'input', 'intermediate', 'output'

@dataclass
class RunManifest:
    """Complete manifest for a processing run"""
    run_id: str
    timestamp: str
    input_files: Dict[str, ArtifactInfo]
    intermediate_artifacts: Dict[str, ArtifactInfo]
    output_artifacts: Dict[str, ArtifactInfo]
    processing_config: Dict[str, Any]
    metrics_registry_version: str
    total_processing_time: float
    pipeline_version: str

class DVCVersioningManager:
    """Manages DVC-based versioning for ensemble pipeline artifacts"""
    
    def __init__(self, base_artifacts_dir: str = "artifacts"):
        self.base_artifacts_dir = Path(base_artifacts_dir)
        self.logger = StructuredLogger("dvc_versioning")
        
        # Ensure artifact directories exist
        self.input_dir = self.base_artifacts_dir / "inputs"
        self.audio_dir = self.base_artifacts_dir / "audio"
        self.diarization_dir = self.base_artifacts_dir / "diarization"
        self.asr_dir = self.base_artifacts_dir / "asr"
        self.manifest_dir = self.base_artifacts_dir / "manifests"
        
        for dir_path in [self.input_dir, self.audio_dir, self.diarization_dir, 
                        self.asr_dir, self.manifest_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # DVC availability check
        self.dvc_available = self._check_dvc_available()
        
    def _check_dvc_available(self) -> bool:
        """Check if DVC is available and properly configured"""
        try:
            result = subprocess.run(['dvc', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def create_artifact_info(self, file_path: Union[str, Path], stage: str, 
                           artifact_type: str) -> ArtifactInfo:
        """Create artifact information record"""
        file_path = Path(file_path)
        
        try:
            stat_info = file_path.stat()
            return ArtifactInfo(
                path=str(file_path),
                hash=self.calculate_file_hash(file_path),
                size=stat_info.st_size,
                timestamp=datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                stage=stage,
                type=artifact_type
            )
        except Exception as e:
            self.logger.error(f"Failed to create artifact info for {file_path}: {e}")
            return ArtifactInfo(
                path=str(file_path),
                hash="",
                size=0,
                timestamp=datetime.now().isoformat(),
                stage=stage,
                type=artifact_type
            )
    
    def track_input_file(self, input_path: str, run_id: str) -> Tuple[str, ArtifactInfo]:
        """
        Track input video file and create versioned copy.
        
        Args:
            input_path: Path to input MP4 file
            run_id: Unique run identifier
            
        Returns:
            Tuple of (tracked_path, artifact_info)
        """
        input_path_obj = Path(input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create tracked filename with hash for uniqueness
        file_hash = self.calculate_file_hash(input_path_obj)[:8]
        tracked_filename = f"{run_id}_{timestamp}_{file_hash}{input_path_obj.suffix}"
        tracked_path = str(self.input_dir / tracked_filename)
        
        # Copy input file to tracked location
        try:
            shutil.copy2(input_path_obj, tracked_path)
            
            # Create DVC tracking file if available
            if self.dvc_available:
                self._create_dvc_file(Path(tracked_path))
            
            artifact_info = self.create_artifact_info(tracked_path, "input", "input")
            
            self.logger.info(f"Tracked input file: {tracked_path}", 
                           context={'original_path': str(input_path_obj), 'hash': file_hash})
            
            return str(tracked_path), artifact_info
            
        except Exception as e:
            self.logger.error(f"Failed to track input file {input_path_obj}: {e}")
            # Fallback: create artifact info for original file
            artifact_info = self.create_artifact_info(input_path_obj, "input", "input")
            return str(input_path_obj), artifact_info
    
    def track_audio_artifacts(self, raw_audio_path: str, clean_audio_path: str, 
                            run_id: str) -> Dict[str, ArtifactInfo]:
        """Track audio extraction artifacts"""
        artifacts = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for audio_type, audio_path in [("raw", raw_audio_path), ("clean", clean_audio_path)]:
            if not audio_path or not os.path.exists(audio_path):
                continue
                
            tracked_filename = f"{run_id}_{timestamp}_{audio_type}_audio.wav"
            tracked_path = self.audio_dir / tracked_filename
            
            try:
                shutil.copy2(audio_path, tracked_path)
                
                if self.dvc_available:
                    self._create_dvc_file(tracked_path)
                
                artifacts[f"{audio_type}_audio"] = self.create_artifact_info(
                    tracked_path, "audio_extraction", "intermediate"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to track {audio_type} audio: {e}")
                # Fallback to original file
                artifacts[f"{audio_type}_audio"] = self.create_artifact_info(
                    audio_path, "audio_extraction", "intermediate"
                )
        
        return artifacts
    
    def track_diarization_artifacts(self, diarization_variants: List[Dict[str, Any]], 
                                  run_id: str) -> Dict[str, ArtifactInfo]:
        """Track diarization variant outputs"""
        artifacts = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, variant in enumerate(diarization_variants):
            # Save diarization result as JSON
            variant_filename = f"{run_id}_{timestamp}_diarization_variant_{i}.json"
            variant_path = self.diarization_dir / variant_filename
            
            try:
                with open(variant_path, 'w') as f:
                    json.dump(variant, f, indent=2)
                
                if self.dvc_available:
                    self._create_dvc_file(variant_path)
                
                artifacts[f"diarization_variant_{i}"] = self.create_artifact_info(
                    variant_path, "diarization", "intermediate"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to track diarization variant {i}: {e}")
        
        return artifacts
    
    def track_asr_artifacts(self, candidates: List[Dict[str, Any]], 
                          run_id: str) -> Dict[str, ArtifactInfo]:
        """Track ASR candidate outputs"""
        artifacts = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all candidates as single JSON file
        candidates_filename = f"{run_id}_{timestamp}_asr_candidates.json"
        candidates_path = self.asr_dir / candidates_filename
        
        try:
            with open(candidates_path, 'w') as f:
                json.dump(candidates, f, indent=2)
            
            if self.dvc_available:
                self._create_dvc_file(candidates_path)
            
            artifacts["asr_candidates"] = self.create_artifact_info(
                candidates_path, "asr_ensemble", "intermediate"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to track ASR candidates: {e}")
        
        return artifacts
    
    def track_output_artifacts(self, results: Dict[str, Any], 
                             run_id: str) -> Dict[str, ArtifactInfo]:
        """Track final output artifacts"""
        artifacts = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create temporary files for different output formats
        output_files = {
            'winner_transcript': (f"{run_id}_{timestamp}_winner_transcript.json", 
                                results.get('winner_transcript', {})),
            'transcript_txt': (f"{run_id}_{timestamp}_transcript.txt", 
                             results.get('winner_transcript_txt', '')),
            'captions_vtt': (f"{run_id}_{timestamp}_captions.vtt", 
                           results.get('captions_vtt', '')),
            'captions_srt': (f"{run_id}_{timestamp}_captions.srt", 
                           results.get('captions_srt', '')),
            'ensemble_audit': (f"{run_id}_{timestamp}_ensemble_audit.json", 
                             results.get('ensemble_audit', {})),
            'confidence_breakdown': (f"{run_id}_{timestamp}_confidence_breakdown.json", 
                                   results.get('confidence_breakdown', {}))
        }
        
        for output_type, (filename, content) in output_files.items():
            output_path = self.manifest_dir / filename
            
            try:
                if filename.endswith('.json'):
                    with open(output_path, 'w') as f:
                        json.dump(content, f, indent=2)
                else:
                    with open(output_path, 'w') as f:
                        f.write(str(content))
                
                if self.dvc_available:
                    self._create_dvc_file(output_path)
                
                artifacts[output_type] = self.create_artifact_info(
                    output_path, "output_generation", "output"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to track output {output_type}: {e}")
        
        return artifacts
    
    def _create_dvc_file(self, file_path: Union[str, Path]):
        """Create DVC tracking file for artifact"""
        file_path = Path(file_path)
        try:
            dvc_file = str(file_path.with_suffix(file_path.suffix + '.dvc'))
            
            # Create basic DVC file structure
            dvc_content = {
                'outs': [{'path': file_path.name, 'cache': True}]
            }
            
            with open(dvc_file, 'w') as f:
                import yaml
                yaml.dump(dvc_content, f)
                
        except Exception as e:
            self.logger.error(f"Failed to create DVC file for {file_path}: {e}")
    
    def create_run_manifest(self, run_id: str, input_artifacts: Dict[str, ArtifactInfo],
                          intermediate_artifacts: Dict[str, ArtifactInfo],
                          output_artifacts: Dict[str, ArtifactInfo],
                          processing_config: Dict[str, Any],
                          processing_time: float,
                          metrics_registry_version: str = "v1.0") -> RunManifest:
        """Create comprehensive run manifest"""
        
        manifest = RunManifest(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            input_files=input_artifacts,
            intermediate_artifacts=intermediate_artifacts,
            output_artifacts=output_artifacts,
            processing_config=processing_config,
            metrics_registry_version=metrics_registry_version,
            total_processing_time=processing_time,
            pipeline_version="ensemble_v1.0"
        )
        
        return manifest
    
    def save_run_manifest(self, manifest: RunManifest) -> str:
        """Save run manifest to file and return path"""
        manifest_filename = f"run_manifest_{manifest.run_id}.json"
        manifest_path = self.manifest_dir / manifest_filename
        
        try:
            with open(manifest_path, 'w') as f:
                json.dump(asdict(manifest), f, indent=2)
            
            if self.dvc_available:
                self._create_dvc_file(manifest_path)
            
            self.logger.info(f"Saved run manifest: {manifest_path}")
            return str(manifest_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save run manifest: {e}")
            return ""
    
    def verify_artifact_integrity(self, artifact_info: ArtifactInfo) -> bool:
        """Verify artifact integrity using stored hash"""
        if not os.path.exists(artifact_info.path):
            return False
        
        current_hash = self.calculate_file_hash(artifact_info.path)
        return current_hash == artifact_info.hash
    
    def get_artifact_lineage(self, run_id: str) -> Optional[RunManifest]:
        """Retrieve complete artifact lineage for a run"""
        manifest_filename = f"run_manifest_{run_id}.json"
        manifest_path = self.manifest_dir / manifest_filename
        
        try:
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                return RunManifest(**manifest_data)
        except Exception as e:
            self.logger.error(f"Failed to load run manifest {run_id}: {e}")
        
        return None