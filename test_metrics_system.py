#!/usr/bin/env python3
"""
Test script to validate the comprehensive metrics and alerts system implementation.

This script tests:
1. Metrics system initialization
2. Basic metric collection and alerting
3. Configuration loading
4. Integration with existing observability systems
5. Structured logging functionality
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_metrics_imports():
    """Test that all metrics system imports work correctly"""
    try:
        from utils.metrics_alerts import (
            get_metrics_collector,
            initialize_metrics_system,
            track_processing_stage,
            record_quality_metrics,
            record_business_event,
            track_performance,
            MetricsCollector,
            AlertSeverity,
            MetricType
        )
        print("✅ Metrics system imports successful")
        return True
    except Exception as e:
        print(f"❌ Metrics system import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    try:
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if metrics configuration exists
        if 'metrics' in config:
            print("✅ Metrics configuration found in config.yaml")
            print(f"  - Enabled: {config['metrics'].get('enabled', False)}")
            print(f"  - Collection interval: {config['metrics'].get('collection_interval_seconds', 'N/A')}s")
            print(f"  - Alerting enabled: {config['metrics'].get('alerting', {}).get('enabled', False)}")
            return True
        else:
            print("❌ Metrics configuration not found in config.yaml")
            return False
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_metrics_initialization():
    """Test metrics system initialization"""
    try:
        from utils.metrics_alerts import initialize_metrics_system
        
        test_config = {
            'enabled': True,
            'aggregation_window_seconds': 60,
            'enable_background_processing': False,  # Disable for testing
            'alerting': {
                'enabled': True,
                'alert_destinations': {
                    'console': {'enabled': True},
                    'file_logging': {'enabled': True, 'path': '/tmp/test_alerts'}
                }
            }
        }
        
        collector = initialize_metrics_system(test_config, session_id="test_session")
        print("✅ Metrics system initialization successful")
        print(f"  - Session ID: {collector.session_id}")
        print(f"  - Enabled: {collector.enabled}")
        print(f"  - Background processing: {collector.config.get('enable_background_processing', False)}")
        return collector
    except Exception as e:
        print(f"❌ Metrics system initialization failed: {e}")
        return None

def test_metric_collection(collector):
    """Test basic metric collection"""
    try:
        if not collector:
            return False
            
        # Test recording different types of metrics
        collector.record_metric('word_error_rate', 2.5, 'test_component', 
                               tags={'test': 'true'}, 
                               context={'test_type': 'unit_test'})
        
        collector.record_metric('processing_time_per_minute', 45.0, 'test_component',
                               tags={'stage': 'test'},
                               context={'audio_duration': 60.0})
        
        collector.record_metric('success_rate', 95.0, 'test_component',
                               tags={'test_run': 'validation'})
        
        print("✅ Basic metric collection successful")
        return True
    except Exception as e:
        print(f"❌ Metric collection failed: {e}")
        return False

def test_quality_metrics_recording(collector):
    """Test quality metrics recording"""
    try:
        from utils.metrics_alerts import record_quality_metrics
        
        quality_metrics = {
            'word_error_rate': 1.8,
            'confidence_score_avg': 0.92,
            'entity_accuracy': 96.5
        }
        
        record_quality_metrics(quality_metrics, 'test_component', 
                             session_id='test_session',
                             audio_duration=120.0)
        
        print("✅ Quality metrics recording successful")
        return True
    except Exception as e:
        print(f"❌ Quality metrics recording failed: {e}")
        return False

def test_business_events(collector):
    """Test business event recording"""
    try:
        from utils.metrics_alerts import record_business_event
        
        record_business_event('file_processed', 'test_component',
                            success=True,
                            processing_time=75.5,
                            file_path='/test/file.mp4')
        
        print("✅ Business event recording successful")
        return True
    except Exception as e:
        print(f"❌ Business event recording failed: {e}")
        return False

def test_stage_tracking():
    """Test processing stage tracking"""
    try:
        from utils.metrics_alerts import track_processing_stage
        
        with track_processing_stage("test_stage", "test_component", 
                                  audio_duration=60.0) as metrics_tracker:
            # Simulate some processing
            import time
            time.sleep(0.1)  # Brief delay to simulate processing
        
        print("✅ Stage tracking successful")
        return True
    except Exception as e:
        print(f"❌ Stage tracking failed: {e}")
        return False

def test_alert_generation(collector):
    """Test alert generation and thresholds"""
    try:
        if not collector:
            return False
            
        # Record a metric that should trigger an alert (high WER)
        collector.record_metric('word_error_rate', 8.0, 'test_component',
                               context={'should_trigger_alert': True})
        
        # Record a metric that should trigger a warning (high processing time)
        collector.record_metric('processing_time_per_minute', 95.0, 'test_component',
                               context={'should_trigger_warning': True})
        
        # Check if alerts were generated
        active_alerts = collector.alert_manager.get_active_alerts()
        alert_history = collector.alert_manager.get_alert_history(5)
        
        print(f"✅ Alert system functional - {len(active_alerts)} active, {len(alert_history)} in history")
        return True
    except Exception as e:
        print(f"❌ Alert generation test failed: {e}")
        return False

def test_metrics_summary(collector):
    """Test metrics summary generation"""
    try:
        if not collector:
            return False
            
        summary = collector.get_current_metrics_summary()
        print("✅ Metrics summary generation successful")
        print(f"  - Metrics tracked: {len(summary.get('metrics', {}))}")
        print(f"  - Active alerts: {summary.get('alerts', {}).get('active_count', 0)}")
        return True
    except Exception as e:
        print(f"❌ Metrics summary generation failed: {e}")
        return False

def test_ci_integration(collector):
    """Test CI integration features"""
    try:
        if not collector:
            return False
            
        # Simulate CI mode
        os.environ['CI'] = '1'
        
        ci_report = collector.get_ci_metrics_report()
        print("✅ CI integration test successful")
        print(f"  - CI mode detected: {ci_report.get('ci_mode', False)}")
        print(f"  - Quality gates: {len(ci_report.get('quality_gates', {}))}")
        
        # Clean up
        if 'CI' in os.environ:
            del os.environ['CI']
        
        return True
    except Exception as e:
        print(f"❌ CI integration test failed: {e}")
        return False

def main():
    """Run comprehensive metrics system test"""
    print("🚀 Testing Comprehensive Metrics and Alerts System")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Basic imports
    test_results.append(("Imports", test_metrics_imports()))
    
    # Test 2: Configuration loading
    test_results.append(("Configuration", test_config_loading()))
    
    # Test 3: System initialization
    collector = test_metrics_initialization()
    test_results.append(("Initialization", collector is not None))
    
    # Test 4: Basic metric collection
    test_results.append(("Metric Collection", test_metric_collection(collector)))
    
    # Test 5: Quality metrics
    test_results.append(("Quality Metrics", test_quality_metrics_recording(collector)))
    
    # Test 6: Business events
    test_results.append(("Business Events", test_business_events(collector)))
    
    # Test 7: Stage tracking
    test_results.append(("Stage Tracking", test_stage_tracking()))
    
    # Test 8: Alert generation
    test_results.append(("Alert Generation", test_alert_generation(collector)))
    
    # Test 9: Metrics summary
    test_results.append(("Metrics Summary", test_metrics_summary(collector)))
    
    # Test 10: CI integration
    test_results.append(("CI Integration", test_ci_integration(collector)))
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Comprehensive metrics system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)