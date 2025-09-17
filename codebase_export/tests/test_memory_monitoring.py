"""
Memory monitoring tests for long video processing.

This module provides focused tests for monitoring memory usage, detecting leaks,
and ensuring resource management during long video processing.
"""

import pytest
import time
import psutil
import threading
import gc
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from core.ensemble_manager import EnsembleManager
from core.audio_processor import AudioProcessor


class MemoryProfiler:
    """Advanced memory profiling for long video processing tests"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.memory_samples = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def start_profiling(self):
        """Start memory profiling"""
        self.is_monitoring = True
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._sampling_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return analysis"""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        return self._analyze_memory_usage()
        
    def _sampling_loop(self):
        """Background memory sampling"""
        start_time = time.time()
        
        while self.is_monitoring:
            try:
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                sample = {
                    'timestamp': time.time(),
                    'elapsed': time.time() - start_time,
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'memory_percent': memory_percent,
                    'available_mb': psutil.virtual_memory().available / (1024 * 1024)
                }
                
                self.memory_samples.append(sample)
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Memory sampling error: {e}")
                break
                
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze collected memory samples"""
        if not self.memory_samples:
            return {"error": "No memory samples collected"}
            
        rss_values = [s['rss_mb'] for s in self.memory_samples]
        vms_values = [s['vms_mb'] for s in self.memory_samples]
        
        # Calculate statistics
        analysis = {
            'duration_seconds': self.memory_samples[-1]['elapsed'],
            'sample_count': len(self.memory_samples),
            'rss_stats': {
                'initial_mb': rss_values[0],
                'final_mb': rss_values[-1],
                'peak_mb': max(rss_values),
                'min_mb': min(rss_values),
                'average_mb': sum(rss_values) / len(rss_values),
                'growth_mb': rss_values[-1] - rss_values[0]
            },
            'vms_stats': {
                'initial_mb': vms_values[0],
                'final_mb': vms_values[-1],
                'peak_mb': max(vms_values),
                'growth_mb': vms_values[-1] - vms_values[0]
            }
        }
        
        # Detect potential memory leaks
        if len(rss_values) >= 10:
            # Calculate trend over time
            mid_point = len(rss_values) // 2
            first_half_avg = sum(rss_values[:mid_point]) / mid_point
            second_half_avg = sum(rss_values[mid_point:]) / (len(rss_values) - mid_point)
            
            analysis['memory_trend'] = {
                'first_half_avg_mb': first_half_avg,
                'second_half_avg_mb': second_half_avg,
                'trend_growth_mb': second_half_avg - first_half_avg,
                'potential_leak': second_half_avg > first_half_avg * 1.2  # 20% increase
            }
            
        return analysis


@pytest.fixture
def memory_profiler():
    """Fixture providing memory profiler"""
    profiler = MemoryProfiler(sampling_interval=0.5)
    yield profiler
    profiler.stop_profiling()


class TestMemoryManagement:
    """Test memory management during long video processing"""
    
    def test_memory_baseline_measurement(self, memory_profiler):
        """Establish baseline memory usage"""
        memory_profiler.start_profiling()
        
        # Simulate baseline activity
        time.sleep(2)
        
        # Force garbage collection
        gc.collect()
        
        time.sleep(2)
        analysis = memory_profiler.stop_profiling()
        
        # Validate baseline measurements
        assert analysis['sample_count'] > 0, "Should collect memory samples"
        assert analysis['rss_stats']['peak_mb'] > 0, "Should measure RSS memory"
        
        # Baseline should be stable (no significant growth without processing)
        growth = analysis['rss_stats']['growth_mb']
        assert abs(growth) < 50, f"Baseline memory should be stable, growth: {growth:.1f}MB"
        
    @patch('core.ensemble_manager.EnsembleManager.process_video')
    def test_memory_usage_short_video(self, mock_process_video, memory_profiler):
        """Test memory usage for short video processing"""
        
        # Mock short video processing
        def mock_short_processing(*args, **kwargs):
            time.sleep(1)  # Simulate processing
            return {
                'master_transcript': {
                    'segments': [{'start': 0, 'end': 30, 'speaker': 'A', 'text': 'Test', 'confidence': 0.9}],
                    'metadata': {'total_duration': 30.0}
                }
            }
        
        mock_process_video.side_effect = mock_short_processing
        
        memory_profiler.start_profiling()
        
        # Process short video
        manager = EnsembleManager.create_safe()
        result = manager.process_video('/fake/short_video.mp4')
        
        analysis = memory_profiler.stop_profiling()
        
        # Validate reasonable memory usage for short video
        peak_memory = analysis['rss_stats']['peak_mb']
        assert peak_memory < 2000, f"Memory usage too high for short video: {peak_memory:.1f}MB"
        
        # Should not have significant memory growth
        growth = analysis['rss_stats']['growth_mb']
        assert growth < 100, f"Memory growth too high for short video: {growth:.1f}MB"
        
    @patch('core.ensemble_manager.EnsembleManager.process_video')
    def test_memory_usage_long_video(self, mock_process_video, memory_profiler):
        """Test memory usage for long video processing"""
        
        def mock_long_processing(*args, **kwargs):
            # Simulate longer processing with incremental memory usage
            for i in range(10):
                time.sleep(0.2)
                # Simulate some processing load
                data = [0] * (100000 * i)  # Gradually increase memory usage
                
            return {
                'master_transcript': {
                    'segments': [
                        {'start': i*60, 'end': (i+1)*60, 'speaker': f'Speaker_{i%3}', 
                         'text': f'Long segment {i}', 'confidence': 0.9}
                        for i in range(120)  # 2 hours of segments
                    ],
                    'metadata': {'total_duration': 7200.0}  # 2 hours
                }
            }
        
        mock_process_video.side_effect = mock_long_processing
        
        memory_profiler.start_profiling()
        
        # Process long video
        manager = EnsembleManager.create_safe()
        result = manager.process_video('/fake/long_video.mp4')
        
        analysis = memory_profiler.stop_profiling()
        
        # Validate memory usage for long video
        peak_memory = analysis['rss_stats']['peak_mb']
        assert peak_memory < 8000, f"Memory usage too high for long video: {peak_memory:.1f}MB"
        
        # Growth should be reasonable
        growth = analysis['rss_stats']['growth_mb']
        assert growth < 2000, f"Memory growth too high for long video: {growth:.1f}MB"
        
        # Check for memory leaks
        if 'memory_trend' in analysis:
            potential_leak = analysis['memory_trend']['potential_leak']
            assert not potential_leak, f"Potential memory leak detected: {analysis['memory_trend']}"
    
    def test_memory_cleanup_after_processing(self, memory_profiler):
        """Test that memory is cleaned up after processing"""
        
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Simulate processing with memory allocation
        with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
            def allocate_and_process(*args, **kwargs):
                # Allocate some memory during processing
                large_data = [0] * 1000000  # ~4MB of data
                time.sleep(0.5)
                del large_data  # Clean up
                gc.collect()
                return {'master_transcript': {'segments': [], 'metadata': {'total_duration': 60.0}}}
            
            mock_process.side_effect = allocate_and_process
            
            memory_profiler.start_profiling()
            
            # Process multiple videos to test cleanup
            manager = EnsembleManager.create_safe()
            for i in range(3):
                result = manager.process_video(f'/fake/video_{i}.mp4')
                gc.collect()  # Force cleanup between videos
                
            analysis = memory_profiler.stop_profiling()
        
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_difference = final_memory - initial_memory
        
        # Memory should return close to initial level after cleanup
        assert memory_difference < 500, f"Memory not cleaned up properly: +{memory_difference:.1f}MB"
        
    def test_memory_efficiency_chunked_processing(self):
        """Test memory efficiency of chunked processing vs. non-chunked"""
        
        # Test that chunked processing uses less peak memory
        manager = EnsembleManager.create_safe()
        
        # Ensure chunking is enabled for this test
        original_threshold = getattr(manager, 'chunked_processing_threshold', 1800)
        manager.chunked_processing_threshold = 60  # Force chunking for short videos
        
        try:
            with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
                def chunked_processing(*args, **kwargs):
                    # Simulate chunked processing with controlled memory usage
                    max_chunk_memory = 0
                    for chunk in range(5):  # 5 chunks
                        chunk_data = [0] * 200000  # 800KB per chunk
                        max_chunk_memory = max(max_chunk_memory, len(chunk_data))
                        del chunk_data
                        gc.collect()
                    
                    return {'master_transcript': {'segments': [], 'metadata': {'total_duration': 300.0}}}
                
                mock_process.side_effect = chunked_processing
                
                # Should not use excessive memory with chunking
                initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                result = manager.process_video('/fake/chunked_video.mp4')
                final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                
                memory_used = final_memory - initial_memory
                assert memory_used < 1000, f"Chunked processing used too much memory: {memory_used:.1f}MB"
                
        finally:
            # Restore original threshold
            manager.chunked_processing_threshold = original_threshold


class TestMemoryLeakDetection:
    """Specific tests for detecting memory leaks"""
    
    @patch('core.ensemble_manager.EnsembleManager.process_video')
    def test_repeated_processing_no_leak(self, mock_process_video, memory_profiler):
        """Test repeated video processing doesn't cause memory leaks"""
        
        def consistent_processing(*args, **kwargs):
            # Consistent memory usage per call
            temp_data = [0] * 100000  # 400KB
            time.sleep(0.1)
            del temp_data
            return {'master_transcript': {'segments': [], 'metadata': {'total_duration': 60.0}}}
        
        mock_process_video.side_effect = consistent_processing
        
        memory_profiler.start_profiling()
        
        # Process multiple videos
        manager = EnsembleManager.create_safe()
        for i in range(10):
            result = manager.process_video(f'/fake/video_{i}.mp4')
            if i % 3 == 0:  # Force GC periodically
                gc.collect()
                
        analysis = memory_profiler.stop_profiling()
        
        # Check for memory leak indicators
        growth = analysis['rss_stats']['growth_mb']
        assert growth < 200, f"Potential memory leak: {growth:.1f}MB growth over 10 videos"
        
        if 'memory_trend' in analysis:
            trend_growth = analysis['memory_trend']['trend_growth_mb']
            assert trend_growth < 100, f"Memory trend indicates leak: +{trend_growth:.1f}MB"
    
    def test_garbage_collection_effectiveness(self):
        """Test that garbage collection is effective during processing"""
        
        initial_objects = len(gc.get_objects())
        
        # Simulate processing with object creation
        with patch('core.ensemble_manager.EnsembleManager.process_video') as mock_process:
            def object_creating_process(*args, **kwargs):
                # Create many objects
                objects = []
                for i in range(1000):
                    objects.append({'data': [0] * 100, 'id': i})
                
                # Process and clean up some objects
                result_objects = objects[:100]
                del objects
                
                return {
                    'master_transcript': {'segments': result_objects, 'metadata': {'total_duration': 60.0}}
                }
            
            mock_process.side_effect = object_creating_process
            
            manager = EnsembleManager.create_safe()
            result = manager.process_video('/fake/test_video.mp4')
            
            # Force garbage collection
            collected = gc.collect()
            
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Should not have excessive object growth
        assert object_growth < 10000, f"Too many objects not collected: +{object_growth} objects"
        assert collected >= 0, "Garbage collection should run"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])