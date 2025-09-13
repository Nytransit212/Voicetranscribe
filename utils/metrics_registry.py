"""
Metrics registry system for tracking historical confidence scoring statistics.
Enables calibrated scoring by maintaining historical means and standard deviations.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from statistics import mean, stdev
from utils.structured_logger import StructuredLogger

@dataclass
class DimensionStats:
    """Statistical data for a scoring dimension"""
    mean: float
    std: float
    min_value: float
    max_value: float
    sample_count: int
    last_updated: str

@dataclass
class DomainStats:
    """Statistics for a specific domain (e.g., meeting, interview, lecture)"""
    domain_name: str
    speaker_range: str  # e.g., "2-5", "6-10", "11+"
    noise_level: str    # "low", "medium", "high"
    stats: Dict[str, DimensionStats]  # D, A, L, R, O dimensions

@dataclass
class MetricsRegistry:
    """Complete metrics registry with historical calibration data"""
    version: str
    created_date: str
    last_updated: str
    global_stats: Dict[str, DimensionStats]  # Overall stats across all domains
    domain_stats: List[DomainStats]
    calibration_runs: int
    registry_schema_version: str = "1.0"

class MetricsRegistryManager:
    """Manages historical metrics for confidence score calibration"""
    
    def __init__(self, registry_dir: str = "metrics_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        self.logger = StructuredLogger("metrics_registry")
        
        # Registry file paths
        self.current_registry_path = self.registry_dir / "current_registry.json"
        self.historical_registry_dir = self.registry_dir / "historical"
        self.historical_registry_dir.mkdir(exist_ok=True)
        
        # Load or create current registry
        self.current_registry = self._load_or_create_registry()
    
    def _load_or_create_registry(self) -> MetricsRegistry:
        """Load existing registry or create new one with defaults"""
        if self.current_registry_path.exists():
            try:
                with open(self.current_registry_path, 'r') as f:
                    registry_data = json.load(f)
                return MetricsRegistry(**registry_data)
            except Exception as e:
                self.logger.error(f"Failed to load registry, creating new: {e}")
        
        return self._create_default_registry()
    
    def _create_default_registry(self) -> MetricsRegistry:
        """Create default registry with baseline calibration values"""
        
        # Default dimension statistics based on typical confidence score distributions
        default_dimension_stats = {
            'D': DimensionStats(
                mean=0.72, std=0.18, min_value=0.0, max_value=1.0,
                sample_count=0, last_updated=datetime.now().isoformat()
            ),
            'A': DimensionStats(
                mean=0.78, std=0.15, min_value=0.0, max_value=1.0,
                sample_count=0, last_updated=datetime.now().isoformat()
            ),
            'L': DimensionStats(
                mean=0.68, std=0.22, min_value=0.0, max_value=1.0,
                sample_count=0, last_updated=datetime.now().isoformat()
            ),
            'R': DimensionStats(
                mean=0.65, std=0.25, min_value=0.0, max_value=1.0,
                sample_count=0, last_updated=datetime.now().isoformat()
            ),
            'O': DimensionStats(
                mean=0.70, std=0.20, min_value=0.0, max_value=1.0,
                sample_count=0, last_updated=datetime.now().isoformat()
            )
        }
        
        # Create domain-specific baselines
        domain_stats = []
        
        for domain in ["meeting", "interview", "lecture", "general"]:
            for speaker_range in ["2-5", "6-10", "11+"]:
                for noise_level in ["low", "medium", "high"]:
                    # Adjust stats based on domain characteristics
                    adjusted_stats = {}
                    for dim, base_stats in default_dimension_stats.items():
                        # Apply domain-specific adjustments
                        noise_penalty = {"low": 0.0, "medium": -0.05, "high": -0.12}[noise_level]
                        speaker_penalty = {"2-5": 0.02, "6-10": 0.0, "11+": -0.08}[speaker_range]
                        
                        adjusted_mean = max(0.1, min(0.95, base_stats.mean + noise_penalty + speaker_penalty))
                        adjusted_std = base_stats.std * (1.0 + abs(noise_penalty) * 0.5)
                        
                        adjusted_stats[dim] = DimensionStats(
                            mean=adjusted_mean,
                            std=adjusted_std,
                            min_value=base_stats.min_value,
                            max_value=base_stats.max_value,
                            sample_count=0,
                            last_updated=datetime.now().isoformat()
                        )
                    
                    domain_stats.append(DomainStats(
                        domain_name=domain,
                        speaker_range=speaker_range,
                        noise_level=noise_level,
                        stats=adjusted_stats
                    ))
        
        registry = MetricsRegistry(
            version="v1.0",
            created_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            global_stats=default_dimension_stats,
            domain_stats=domain_stats,
            calibration_runs=0
        )
        
        self._save_registry(registry)
        return registry
    
    def get_calibration_stats(self, domain: str = "general", 
                            speaker_count: int = 10, 
                            noise_level: str = "medium") -> Dict[str, DimensionStats]:
        """
        Get calibration statistics for specific processing context.
        
        Args:
            domain: Domain type (meeting, interview, lecture, general)
            speaker_count: Number of speakers
            noise_level: Noise level (low, medium, high)
            
        Returns:
            Dictionary of dimension statistics for calibration
        """
        
        # Determine speaker range
        if speaker_count <= 5:
            speaker_range = "2-5"
        elif speaker_count <= 10:
            speaker_range = "6-10"
        else:
            speaker_range = "11+"
        
        # Find matching domain stats
        for domain_stat in self.current_registry.domain_stats:
            if (domain_stat.domain_name == domain and 
                domain_stat.speaker_range == speaker_range and
                domain_stat.noise_level == noise_level):
                return domain_stat.stats
        
        # Fallback to global stats if no specific domain found
        self.logger.warning(f"No specific stats found for {domain}/{speaker_range}/{noise_level}, using global")
        return self.current_registry.global_stats
    
    def update_registry_with_run(self, confidence_scores: List[Dict[str, float]], 
                               domain: str = "general",
                               speaker_count: int = 10,
                               noise_level: str = "medium"):
        """
        Update registry with new run data.
        
        Args:
            confidence_scores: List of confidence score dictionaries from ensemble run
            domain: Domain type
            speaker_count: Number of speakers
            noise_level: Noise level
        """
        
        if not confidence_scores:
            return
        
        # Extract scores by dimension
        dimension_values = {dim: [] for dim in ['D', 'A', 'L', 'R', 'O']}
        
        for score_dict in confidence_scores:
            for key, value in score_dict.items():
                if key.startswith('D_'):
                    dimension_values['D'].append(value)
                elif key.startswith('A_'):
                    dimension_values['A'].append(value)
                elif key.startswith('L_'):
                    dimension_values['L'].append(value)
                elif key.startswith('R_'):
                    dimension_values['R'].append(value)
                elif key.startswith('O_'):
                    dimension_values['O'].append(value)
        
        # Update global stats
        self._update_dimension_stats(self.current_registry.global_stats, dimension_values)
        
        # Update domain-specific stats
        speaker_range = "2-5" if speaker_count <= 5 else "6-10" if speaker_count <= 10 else "11+"
        
        for domain_stat in self.current_registry.domain_stats:
            if (domain_stat.domain_name == domain and 
                domain_stat.speaker_range == speaker_range and
                domain_stat.noise_level == noise_level):
                self._update_dimension_stats(domain_stat.stats, dimension_values)
                break
        
        # Update registry metadata
        self.current_registry.last_updated = datetime.now().isoformat()
        self.current_registry.calibration_runs += 1
        
        # Save updated registry
        self._save_registry(self.current_registry)
        
        self.logger.info(f"Updated registry with run data", 
                        context={'domain': domain, 'speaker_count': speaker_count, 
                               'noise_level': noise_level, 'scores_count': len(confidence_scores)})
    
    def _update_dimension_stats(self, stats_dict: Dict[str, DimensionStats], 
                              dimension_values: Dict[str, List[float]]):
        """Update statistics for each dimension with new values"""
        
        for dimension, values in dimension_values.items():
            if not values or dimension not in stats_dict:
                continue
            
            current_stats = stats_dict[dimension]
            
            # Combine with existing data for recalculation
            if current_stats.sample_count > 0:
                # Use existing mean/std to approximate old values, then combine
                old_total = current_stats.mean * current_stats.sample_count
                new_total = sum(values)
                combined_total = old_total + new_total
                new_count = current_stats.sample_count + len(values)
                new_mean = combined_total / new_count
                
                # For std deviation, use incremental formula
                if new_count > 1:
                    # Simplified: recalculate with all available data
                    all_values = values  # In production, would maintain more history
                    new_std = stdev(all_values) if len(all_values) > 1 else current_stats.std
                else:
                    new_std = current_stats.std
                    
            else:
                # First time calculation
                new_mean = mean(values)
                new_std = stdev(values) if len(values) > 1 else 0.1
                new_count = len(values)
            
            # Update stats
            stats_dict[dimension] = DimensionStats(
                mean=new_mean,
                std=new_std,
                min_value=min(current_stats.min_value, min(values)),
                max_value=max(current_stats.max_value, max(values)),
                sample_count=new_count,
                last_updated=datetime.now().isoformat()
            )
    
    def _save_registry(self, registry: MetricsRegistry):
        """Save registry to current file and create historical backup"""
        
        try:
            # Save current registry
            with open(self.current_registry_path, 'w') as f:
                json.dump(asdict(registry), f, indent=2)
            
            # Create historical backup
            historical_filename = f"registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            historical_path = self.historical_registry_dir / historical_filename
            
            with open(historical_path, 'w') as f:
                json.dump(asdict(registry), f, indent=2)
            
            # Cleanup old historical files (keep last 30 days)
            self._cleanup_historical_registries()
            
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def _cleanup_historical_registries(self):
        """Remove historical registry files older than 30 days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for file_path in self.historical_registry_dir.glob("registry_backup_*.json"):
                try:
                    # Extract date from filename
                    date_str = file_path.stem.split('_')[2:4]  # ['20240101', '120000']
                    file_date = datetime.strptime(f"{date_str[0]}_{date_str[1]}", "%Y%m%d_%H%M%S")
                    
                    if file_date < cutoff_date:
                        file_path.unlink()
                        
                except (ValueError, IndexError):
                    # Skip files with unexpected naming
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup historical registries: {e}")
    
    def get_registry_version(self) -> str:
        """Get current registry version"""
        return self.current_registry.version
    
    def export_registry_summary(self) -> Dict[str, Any]:
        """Export registry summary for reporting"""
        return {
            'version': self.current_registry.version,
            'last_updated': self.current_registry.last_updated,
            'calibration_runs': self.current_registry.calibration_runs,
            'global_stats_summary': {
                dim: {
                    'mean': stats.mean,
                    'std': stats.std,
                    'samples': stats.sample_count
                }
                for dim, stats in self.current_registry.global_stats.items()
            },
            'domain_count': len(self.current_registry.domain_stats)
        }