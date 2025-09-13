"""
Advanced profiling manager with py-spy integration and performance analysis.
Provides context managers for profiling specific operations and generating performance reports.
"""

import os
import time
import subprocess
import tempfile
import psutil
from typing import Dict, Any, Optional, List, Tuple, Union
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger


class ProfilingManager:
    """Manages performance profiling with py-spy and system metrics collection"""
    
    def __init__(self, enable_py_spy: bool = True, profile_output_dir: str = "artifacts/profiles"):
        self.enable_py_spy = enable_py_spy
        self.profile_output_dir = Path(profile_output_dir)
        self.profile_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Active profiling sessions
        self.active_profiles = {}
        self.profile_data = []
        
        # System monitoring
        self.process = psutil.Process()
        self.baseline_metrics = self._get_system_metrics()
        
        logger.info("Profiling manager initialized", 
                   py_spy_enabled=enable_py_spy,
                   output_dir=str(self.profile_output_dir))
    
    @contextmanager
    def profile_operation(self, operation_name: str, include_flame_graph: bool = True):
        """
        Profile a specific operation with py-spy and system metrics.
        
        Args:
            operation_name: Name of the operation being profiled
            include_flame_graph: Whether to generate flame graph output
            
        Yields:
            Profile metadata dictionary
        """
        profile_id = f"{operation_name}_{int(time.time())}"
        
        # Prepare output files
        profile_data_file = self.profile_output_dir / f"{profile_id}.prof"
        flame_graph_file = self.profile_output_dir / f"{profile_id}_flame.svg"
        speedscope_file = self.profile_output_dir / f"{profile_id}_speedscope.json"
        
        # Start system metrics collection
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        # Start py-spy profiling if enabled
        py_spy_process = None
        if self.enable_py_spy and self._check_py_spy_available():
            try:
                py_spy_args = [
                    "py-spy", "record",
                    "--pid", str(os.getpid()),
                    "--output", str(profile_data_file),
                    "--format", "speedscope",
                    "--duration", "0",  # Record until stopped
                    "--rate", "100",     # 100Hz sampling rate
                ]
                
                py_spy_process = subprocess.Popen(
                    py_spy_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                logger.info(f"Started py-spy profiling for {operation_name}", 
                           profile_id=profile_id, pid=os.getpid())
                
            except Exception as e:
                logger.warning(f"Failed to start py-spy profiling: {e}")
                py_spy_process = None
        
        # Profile metadata
        profile_metadata = {
            "profile_id": profile_id,
            "operation_name": operation_name,
            "start_time": start_time,
            "start_metrics": start_metrics,
            "profile_data_file": str(profile_data_file),
            "flame_graph_file": str(flame_graph_file) if include_flame_graph else None,
            "speedscope_file": str(speedscope_file)
        }
        
        self.active_profiles[profile_id] = {
            "py_spy_process": py_spy_process,
            "metadata": profile_metadata
        }
        
        try:
            yield profile_metadata
            
        except Exception as e:
            logger.error(f"Error during profiled operation {operation_name}: {e}",
                        profile_id=profile_id)
            raise
            
        finally:
            # Stop profiling and collect results
            end_time = time.time()
            end_metrics = self._get_system_metrics()
            
            # Stop py-spy if it was running
            if py_spy_process:
                try:
                    py_spy_process.terminate()
                    py_spy_process.wait(timeout=5)
                    logger.info(f"Stopped py-spy profiling for {operation_name}",
                               profile_id=profile_id)
                except Exception as e:
                    logger.warning(f"Error stopping py-spy: {e}")
                    try:
                        py_spy_process.kill()
                    except:
                        pass
            
            # Calculate metrics delta
            duration = end_time - start_time
            metrics_delta = self._calculate_metrics_delta(start_metrics, end_metrics)
            
            # Generate flame graph if requested and py-spy was successful
            if include_flame_graph and profile_data_file.exists():
                self._generate_flame_graph(profile_data_file, flame_graph_file)
            
            # Store profile results
            profile_result = {
                "profile_id": profile_id,
                "operation_name": operation_name,
                "duration_seconds": duration,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "metrics_delta": metrics_delta,
                "files": {
                    "profile_data": str(profile_data_file) if profile_data_file.exists() else None,
                    "flame_graph": str(flame_graph_file) if flame_graph_file.exists() else None,
                    "speedscope": str(speedscope_file) if speedscope_file.exists() else None
                },
                "py_spy_enabled": bool(py_spy_process),
                "success": True
            }
            
            self.profile_data.append(profile_result)
            
            # Clean up active profiles
            self.active_profiles.pop(profile_id, None)
            
            logger.info(f"Profiling completed for {operation_name}",
                       profile_id=profile_id,
                       duration=duration,
                       memory_delta_mb=metrics_delta.get("memory_delta_mb", 0),
                       cpu_peak_percent=metrics_delta.get("cpu_peak_percent", 0))
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        memory_info = self.process.memory_info()
        
        return {
            "timestamp": time.time(),
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": self.process.cpu_percent(),
            "num_threads": self.process.num_threads(),
            "open_files": len(self.process.open_files()),
            "connections": len(self.process.connections())
        }
    
    def _calculate_metrics_delta(self, start_metrics: Dict[str, Any], 
                               end_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the difference between start and end metrics"""
        return {
            "duration_seconds": end_metrics["timestamp"] - start_metrics["timestamp"],
            "memory_delta_mb": end_metrics["memory_rss_mb"] - start_metrics["memory_rss_mb"],
            "memory_peak_mb": end_metrics["memory_rss_mb"],
            "cpu_peak_percent": max(start_metrics["cpu_percent"], end_metrics["cpu_percent"]),
            "threads_delta": end_metrics["num_threads"] - start_metrics["num_threads"],
            "files_delta": end_metrics["open_files"] - start_metrics["open_files"]
        }
    
    def _check_py_spy_available(self) -> bool:
        """Check if py-spy is available and executable"""
        try:
            result = subprocess.run(["py-spy", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _generate_flame_graph(self, speedscope_file: Path, flame_graph_file: Path):
        """Generate flame graph from speedscope data"""
        try:
            # Convert speedscope to flame graph format
            # This is a simplified conversion - in practice you might use
            # speedscope's export functionality or other tools
            
            if speedscope_file.exists():
                # For now, just create a placeholder
                # In production, integrate with flamegraph.pl or similar tools
                flame_graph_file.write_text(
                    f"<!-- Flame graph for {speedscope_file.name} -->\n"
                    f"<!-- Generated at {datetime.now().isoformat()} -->\n"
                    "<svg>Flame graph placeholder - integrate with flamegraph tools</svg>"
                )
                
                logger.debug(f"Generated flame graph placeholder: {flame_graph_file}")
                
        except Exception as e:
            logger.warning(f"Failed to generate flame graph: {e}")
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all profiling sessions"""
        if not self.profile_data:
            return {
                "total_sessions": 0,
                "total_duration": 0.0,
                "operations": []
            }
        
        # Calculate aggregate statistics
        total_duration = sum(p["duration_seconds"] for p in self.profile_data)
        operations = list(set(p["operation_name"] for p in self.profile_data))
        
        # Per-operation statistics
        operation_stats = {}
        for op in operations:
            op_profiles = [p for p in self.profile_data if p["operation_name"] == op]
            durations = [p["duration_seconds"] for p in op_profiles]
            memory_deltas = [p["metrics_delta"]["memory_delta_mb"] for p in op_profiles]
            
            operation_stats[op] = {
                "count": len(op_profiles),
                "total_duration": sum(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
                "max_memory_delta_mb": max(memory_deltas)
            }
        
        return {
            "total_sessions": len(self.profile_data),
            "total_duration": total_duration,
            "operations": operations,
            "operation_statistics": operation_stats,
            "latest_profiles": self.profile_data[-5:] if len(self.profile_data) >= 5 else self.profile_data,
            "profile_output_dir": str(self.profile_output_dir)
        }
    
    def export_profiling_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Export profiling data to CSV for analysis"""
        if not self.profile_data:
            logger.warning("No profiling data available for export")
            return ""
        
        # Flatten profiling data for CSV export
        csv_data = []
        for profile in self.profile_data:
            row = {
                "profile_id": profile["profile_id"],
                "operation_name": profile["operation_name"],
                "duration_seconds": profile["duration_seconds"],
                "start_time": profile["start_time"],
                "memory_delta_mb": profile["metrics_delta"]["memory_delta_mb"],
                "memory_peak_mb": profile["metrics_delta"]["memory_peak_mb"],
                "cpu_peak_percent": profile["metrics_delta"]["cpu_peak_percent"],
                "threads_delta": profile["metrics_delta"]["threads_delta"],
                "py_spy_enabled": profile["py_spy_enabled"],
                "profile_data_file": profile["files"]["profile_data"],
                "flame_graph_file": profile["files"]["flame_graph"]
            }
            csv_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(csv_data)
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.profile_output_dir / f"profiling_report_{timestamp}.csv")
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"Profiling report exported to {output_path}",
                   total_profiles=len(csv_data))
        
        return str(output_path)
    
    def cleanup_old_profiles(self, max_age_days: int = 7):
        """Clean up old profile files to save disk space"""
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            
            deleted_count = 0
            for file_path in self.profile_output_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old profile files",
                       max_age_days=max_age_days)
                       
        except Exception as e:
            logger.warning(f"Failed to cleanup old profiles: {e}")


# Global profiling manager instance
_global_profiling_manager: Optional[ProfilingManager] = None


def get_profiling_manager() -> ProfilingManager:
    """Get or create global profiling manager"""
    global _global_profiling_manager
    if _global_profiling_manager is None:
        _global_profiling_manager = ProfilingManager()
    return _global_profiling_manager


def profile_operation(operation_name: str, include_flame_graph: bool = True):
    """Decorator for profiling function operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiling_manager()
            with profiler.profile_operation(operation_name, include_flame_graph):
                return func(*args, **kwargs)
        return wrapper
    return decorator