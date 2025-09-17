"""
Observability reporting system that generates comprehensive CSV and profiling reports
alongside transcript outputs for batch analysis and optimization.
"""

import os
import csv
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger


class ObservabilityReporter:
    """Generates comprehensive reports for observability data analysis"""
    
    def __init__(self, output_dir: str = "artifacts/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report templates
        self.cost_report_fields = [
            'run_id', 'session_id', 'timestamp', 'service', 'model', 
            'cost_usd', 'api_calls', 'duration_seconds', 'tokens_total',
            'audio_duration_seconds', 'language', 'temperature'
        ]
        
        self.performance_report_fields = [
            'run_id', 'session_id', 'stage', 'stage_duration_seconds',
            'memory_delta_mb', 'memory_peak_mb', 'cpu_peak_percent',
            'candidates_generated', 'winner_score', 'success'
        ]
        
        self.pipeline_report_fields = [
            'run_id', 'session_id', 'start_time', 'end_time', 'total_duration_seconds',
            'total_cost_usd', 'total_api_calls', 'candidates_generated', 'winner_score',
            'detected_speakers', 'audio_duration_seconds', 'expected_speakers',
            'noise_level', 'target_language', 'instrumentation_enabled',
            'profiling_enabled', 'peak_memory_mb', 'success', 'error_message'
        ]
        
        logger.info("Observability reporter initialized", output_dir=str(self.output_dir))
    
    def generate_comprehensive_report(self, results: Dict[str, Any], 
                                    profiling_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate comprehensive observability reports from processing results.
        
        Args:
            results: Complete processing results with observability data
            profiling_data: Optional profiling data from ProfilingManager
            
        Returns:
            Dictionary of generated report file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = results.get('observability_metadata', {}).get('run_id', 'unknown')
        
        report_files = {}
        
        try:
            # 1. Generate cost analysis report
            cost_report_path = self._generate_cost_report(results, timestamp, run_id)
            if cost_report_path:
                report_files['cost_analysis'] = cost_report_path
            
            # 2. Generate performance analysis report
            performance_report_path = self._generate_performance_report(results, timestamp, run_id)
            if performance_report_path:
                report_files['performance_analysis'] = performance_report_path
            
            # 3. Generate pipeline summary report
            pipeline_report_path = self._generate_pipeline_report(results, timestamp, run_id)
            if pipeline_report_path:
                report_files['pipeline_summary'] = pipeline_report_path
            
            # 4. Generate profiling report if data available
            if profiling_data:
                profiling_report_path = self._generate_profiling_report(profiling_data, timestamp, run_id)
                if profiling_report_path:
                    report_files['profiling_analysis'] = profiling_report_path
            
            # 5. Generate master observability summary
            master_report_path = self._generate_master_summary(results, report_files, timestamp, run_id)
            if master_report_path:
                report_files['master_summary'] = master_report_path
            
            logger.info("Comprehensive observability reports generated",
                       run_id=run_id,
                       report_count=len(report_files),
                       reports=list(report_files.keys()))
            
        except Exception as e:
            logger.error(f"Failed to generate observability reports: {e}", run_id=run_id)
        
        return report_files
    
    def _generate_cost_report(self, results: Dict[str, Any], timestamp: str, run_id: str) -> Optional[str]:
        """Generate detailed cost analysis CSV report"""
        
        cost_summary = results.get('cost_summary', {})
        if not cost_summary:
            return None
        
        cost_breakdown = cost_summary.get('cost_breakdown', {})
        obs_metadata = results.get('observability_metadata', {})
        
        report_path = self.output_dir / f"cost_analysis_{run_id}_{timestamp}.csv"
        
        try:
            with open(report_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.cost_report_fields)
                writer.writeheader()
                
                session_id = obs_metadata.get('session_id', 'unknown')
                report_timestamp = datetime.now().isoformat()
                
                # Write cost breakdown rows
                for service, data in cost_breakdown.items():
                    writer.writerow({
                        'run_id': run_id,
                        'session_id': session_id,
                        'timestamp': report_timestamp,
                        'service': service.split('_')[0],  # e.g., 'openai' from 'openai_whisper'
                        'model': service.split('_')[1] if '_' in service else 'unknown',
                        'cost_usd': data.get('cost', 0.0),
                        'api_calls': data.get('calls', 0),
                        'duration_seconds': data.get('total_duration', 0.0),
                        'tokens_total': 0,  # Would need to be tracked separately
                        'audio_duration_seconds': 0,  # Would be calculated from usage
                        'language': 'auto',  # Default
                        'temperature': 0.0  # Default
                    })
            
            logger.info(f"Cost analysis report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate cost report: {e}")
            return None
    
    def _generate_performance_report(self, results: Dict[str, Any], timestamp: str, run_id: str) -> Optional[str]:
        """Generate performance analysis CSV report"""
        
        system_metrics = results.get('system_metrics', {})
        obs_metadata = results.get('observability_metadata', {})
        
        if not system_metrics:
            return None
        
        report_path = self.output_dir / f"performance_analysis_{run_id}_{timestamp}.csv"
        
        try:
            with open(report_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.performance_report_fields)
                writer.writeheader()
                
                session_id = obs_metadata.get('session_id', 'unknown')
                
                # Write performance data for each pipeline stage
                pipeline_stages = obs_metadata.get('pipeline_stages', [])
                processing_time = results.get('processing_time', 0.0)
                
                for i, stage in enumerate(pipeline_stages):
                    # Estimate stage duration (simplified)
                    stage_duration = processing_time / len(pipeline_stages)
                    
                    writer.writerow({
                        'run_id': run_id,
                        'session_id': session_id,
                        'stage': stage,
                        'stage_duration_seconds': stage_duration,
                        'memory_delta_mb': system_metrics.get('memory_rss_mb', 0) / len(pipeline_stages),
                        'memory_peak_mb': system_metrics.get('memory_rss_mb', 0),
                        'cpu_peak_percent': system_metrics.get('cpu_percent', 0),
                        'candidates_generated': 15 if 'asr' in stage else 0,
                        'winner_score': results.get('winner_score', 0.0) if i == len(pipeline_stages) - 1 else 0.0,
                        'success': True
                    })
            
            logger.info(f"Performance analysis report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return None
    
    def _generate_pipeline_report(self, results: Dict[str, Any], timestamp: str, run_id: str) -> Optional[str]:
        """Generate pipeline summary CSV report"""
        
        report_path = self.output_dir / f"pipeline_summary_{run_id}_{timestamp}.csv"
        
        try:
            with open(report_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.pipeline_report_fields)
                writer.writeheader()
                
                cost_summary = results.get('cost_summary', {})
                system_metrics = results.get('system_metrics', {})
                obs_metadata = results.get('observability_metadata', {})
                session_metadata = results.get('session_metadata', {})
                
                # Calculate start/end times
                processing_time = results.get('processing_time', 0.0)
                end_time = datetime.now()
                start_time = datetime.fromtimestamp(end_time.timestamp() - processing_time)
                
                writer.writerow({
                    'run_id': run_id,
                    'session_id': obs_metadata.get('session_id', 'unknown'),
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_duration_seconds': processing_time,
                    'total_cost_usd': cost_summary.get('total_cost_usd', 0.0),
                    'total_api_calls': cost_summary.get('total_api_calls', 0),
                    'candidates_generated': session_metadata.get('candidates_generated', 15),
                    'winner_score': results.get('winner_score', 0.0),
                    'detected_speakers': results.get('detected_speakers', 0),
                    'audio_duration_seconds': session_metadata.get('audio_duration', 0.0),
                    'expected_speakers': session_metadata.get('expected_speakers', 10),
                    'noise_level': session_metadata.get('estimated_noise_level', 'medium'),
                    'target_language': 'auto',  # Default
                    'instrumentation_enabled': obs_metadata.get('instrumentation_enabled', False),
                    'profiling_enabled': obs_metadata.get('profiling_enabled', False),
                    'peak_memory_mb': obs_metadata.get('peak_memory_mb', 0),
                    'success': True,
                    'error_message': ''
                })
            
            logger.info(f"Pipeline summary report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate pipeline report: {e}")
            return None
    
    def _generate_profiling_report(self, profiling_data: Dict[str, Any], timestamp: str, run_id: str) -> Optional[str]:
        """Generate profiling analysis CSV report"""
        
        if not profiling_data or not profiling_data.get('profiles'):
            return None
        
        report_path = self.output_dir / f"profiling_analysis_{run_id}_{timestamp}.csv"
        
        try:
            profiles = profiling_data.get('profiles', [])
            
            # Convert profiling data to DataFrame for easy CSV export
            profile_rows = []
            for profile in profiles:
                profile_rows.append({
                    'run_id': run_id,
                    'profile_id': profile.get('profile_id', ''),
                    'operation_name': profile.get('operation_name', ''),
                    'duration_seconds': profile.get('duration_seconds', 0.0),
                    'memory_delta_mb': profile.get('memory_delta_mb', 0.0),
                    'cpu_peak_percent': profile.get('cpu_peak_percent', 0.0),
                    'start_time': profile.get('start_time', ''),
                    'profile_file': profile.get('profile_file', ''),
                    'flame_graph_file': profile.get('flame_graph_file', '')
                })
            
            if profile_rows:
                df = pd.DataFrame(profile_rows)
                df.to_csv(report_path, index=False)
                
                logger.info(f"Profiling analysis report generated: {report_path}",
                           profile_count=len(profile_rows))
                return str(report_path)
                
        except Exception as e:
            logger.error(f"Failed to generate profiling report: {e}")
        
        return None
    
    def _generate_master_summary(self, results: Dict[str, Any], report_files: Dict[str, str], 
                                timestamp: str, run_id: str) -> Optional[str]:
        """Generate master observability summary JSON"""
        
        summary_path = self.output_dir / f"observability_summary_{run_id}_{timestamp}.json"
        
        try:
            summary = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'processing_summary': {
                    'total_duration': results.get('processing_time', 0.0),
                    'winner_score': results.get('winner_score', 0.0),
                    'detected_speakers': results.get('detected_speakers', 0),
                    'candidates_generated': results.get('session_metadata', {}).get('candidates_generated', 15)
                },
                'cost_summary': results.get('cost_summary', {}),
                'system_metrics': results.get('system_metrics', {}),
                'observability_metadata': results.get('observability_metadata', {}),
                'generated_reports': report_files,
                'report_generation': {
                    'timestamp': datetime.now().isoformat(),
                    'reports_generated': len(report_files),
                    'total_size_bytes': sum(
                        Path(path).stat().st_size 
                        for path in report_files.values() 
                        if Path(path).exists()
                    )
                }
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Master observability summary generated: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            logger.error(f"Failed to generate master summary: {e}")
            return None
    
    def aggregate_reports(self, report_pattern: str = "*.csv") -> Dict[str, str]:
        """Aggregate multiple reports into combined datasets"""
        
        try:
            csv_files = list(self.output_dir.glob(report_pattern))
            
            if not csv_files:
                return {}
            
            # Group files by report type
            report_types = {}
            for csv_file in csv_files:
                report_type = csv_file.name.split('_')[0]  # e.g., 'cost', 'performance', 'pipeline'
                if report_type not in report_types:
                    report_types[report_type] = []
                report_types[report_type].append(csv_file)
            
            # Combine reports by type
            combined_reports = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for report_type, files in report_types.items():
                if len(files) > 1:
                    combined_path = self.output_dir / f"{report_type}_combined_{timestamp}.csv"
                    
                    # Read and combine CSV files
                    all_data = []
                    for file_path in files:
                        try:
                            df = pd.read_csv(file_path)
                            all_data.append(df)
                        except Exception as e:
                            logger.warning(f"Failed to read {file_path}: {e}")
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        combined_df.to_csv(combined_path, index=False)
                        combined_reports[f"{report_type}_combined"] = str(combined_path)
                        
                        logger.info(f"Combined {len(files)} {report_type} reports into {combined_path}")
            
            return combined_reports
            
        except Exception as e:
            logger.error(f"Failed to aggregate reports: {e}")
            return {}


# Global reporter instance
_global_reporter: Optional[ObservabilityReporter] = None

def get_observability_reporter() -> ObservabilityReporter:
    """Get or create global observability reporter"""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = ObservabilityReporter()
    return _global_reporter