"""
Calibration Validation and Monitoring System

Provides comprehensive validation and monitoring of per-engine confidence calibration
including reliability diagrams, Expected Calibration Error tracking, and real-time
calibration quality monitoring for production use.
"""

import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict

from .per_engine_calibration import CalibrationEngine, CalibrationData, CalibrationMetrics, CalibratedResult
from utils.enhanced_structured_logger import create_enhanced_logger

@dataclass
class CalibrationMonitoringData:
    """Real-time calibration monitoring data"""
    provider: str
    timestamp: float
    raw_confidence: float
    calibrated_confidence: float
    ground_truth_accuracy: Optional[float] = None
    segment_length: float = 0.0
    decode_mode: str = "unknown"
    session_id: str = ""

@dataclass
class CalibrationAlert:
    """Alert for calibration quality degradation"""
    alert_type: str  # "ece_degradation", "bias_drift", "coverage_gap"
    provider: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    metrics: Dict[str, float]
    timestamp: float
    recommendations: List[str] = field(default_factory=list)

@dataclass
class CalibrationDashboardData:
    """Data for calibration monitoring dashboard"""
    overall_metrics: Dict[str, Any]
    provider_metrics: Dict[str, Dict[str, Any]]
    reliability_diagrams: Dict[str, Dict[str, List[float]]]
    recent_alerts: List[CalibrationAlert]
    trend_data: Dict[str, List[Tuple[float, float]]]  # timestamp, metric_value pairs

class ReliabilityDiagramPlotter:
    """Creates and saves reliability diagram visualizations"""
    
    def __init__(self, output_dir: str = "calibration_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = create_enhanced_logger("reliability_plotter")
    
    def create_reliability_diagram(self, 
                                 predicted_probs: np.ndarray,
                                 true_labels: np.ndarray,
                                 provider_name: str,
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create and save reliability diagram
        
        Args:
            predicted_probs: Predicted confidence scores
            true_labels: Ground truth binary labels
            provider_name: Name of ASR provider
            save_path: Optional path to save plot
            
        Returns:
            Dictionary with reliability diagram data and statistics
        """
        try:
            # Set up the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Reliability diagram
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = true_labels[in_bin].mean()
                    avg_confidence_in_bin = predicted_probs[in_bin].mean()
                    count_in_bin = in_bin.sum()
                else:
                    accuracy_in_bin = 0
                    avg_confidence_in_bin = 0
                    count_in_bin = 0
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(count_in_bin)
            
            # Plot reliability diagram
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            ax1.bar(bin_confidences, bin_accuracies, width=0.08, alpha=0.7, 
                   edgecolor='black', label='Accuracy in bin')
            
            # Add bin counts as text
            for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
                if count > 0:
                    ax1.text(conf, acc + 0.02, str(count), ha='center', va='bottom', fontsize=8)
            
            ax1.set_xlabel('Mean Predicted Confidence')
            ax1.set_ylabel('Fraction of Positives')
            ax1.set_title(f'Reliability Diagram - {provider_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            
            # Confidence histogram
            ax2.hist(predicted_probs, bins=20, alpha=0.7, density=True, edgecolor='black')
            ax2.set_xlabel('Predicted Confidence')
            ax2.set_ylabel('Density')
            ax2.set_title(f'Confidence Distribution - {provider_name}')
            ax2.grid(True, alpha=0.3)
            
            # Calculate calibration metrics
            ece = self._calculate_ece(predicted_probs, true_labels, n_bins=n_bins)
            mce = max([abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences) if conf > 0] + [0])
            
            # Add metrics to plot
            metrics_text = f'ECE: {ece:.3f}\nMCE: {mce:.3f}\nSamples: {len(predicted_probs)}'
            ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Reliability diagram saved to {save_path}")
            else:
                save_path = self.output_dir / f"reliability_diagram_{provider_name}_{int(time.time())}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            plt.close()
            
            return {
                'ece': ece,
                'mce': mce,
                'bin_accuracies': bin_accuracies,
                'bin_confidences': bin_confidences,
                'bin_counts': bin_counts,
                'sample_count': len(predicted_probs),
                'plot_path': str(save_path)
            }
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting - skipping visualization")
            # Return data without plot
            ece = self._calculate_ece(predicted_probs, true_labels)
            return {
                'ece': ece,
                'sample_count': len(predicted_probs),
                'plot_path': None
            }
        except Exception as e:
            self.logger.error(f"Failed to create reliability diagram for {provider_name}: {e}")
            return {
                'ece': 0.0,
                'sample_count': len(predicted_probs),
                'error': str(e)
            }
    
    def _calculate_ece(self, predicted_probs: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(predicted_probs)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = predicted_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

class CalibrationMonitor:
    """
    Real-time calibration quality monitoring system
    
    Tracks calibration performance in production, generates alerts for degradation,
    and provides monitoring dashboard data for operational visibility.
    """
    
    def __init__(self, 
                 calibration_engine: CalibrationEngine,
                 monitoring_window: int = 1000,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize calibration monitor
        
        Args:
            calibration_engine: CalibrationEngine instance to monitor
            monitoring_window: Number of recent samples to track
            alert_thresholds: Thresholds for generating alerts
        """
        self.calibration_engine = calibration_engine
        self.monitoring_window = monitoring_window
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'ece_degradation_threshold': 0.15,  # ECE above this triggers alert
            'bias_drift_threshold': 0.10,       # Bias drift above this triggers alert  
            'coverage_gap_threshold': 0.05,     # Coverage gap above this triggers alert
            'sample_count_minimum': 50          # Minimum samples for reliable monitoring
        }
        
        self.logger = create_enhanced_logger("calibration_monitor")
        
        # Monitoring data storage
        self.monitoring_data: Dict[str, List[CalibrationMonitoringData]] = defaultdict(list)
        self.alert_history: List[CalibrationAlert] = []
        
        # Reliability diagram plotter
        self.plotter = ReliabilityDiagramPlotter()
        
        self.logger.info("CalibrationMonitor initialized",
                        context={'monitoring_window': monitoring_window,
                                'alert_thresholds': self.alert_thresholds})
    
    def record_calibration_event(self, 
                                provider: str,
                                raw_confidence: float,
                                calibrated_confidence: float,
                                ground_truth_accuracy: Optional[float] = None,
                                segment_length: float = 0.0,
                                decode_mode: str = "unknown",
                                session_id: str = "") -> None:
        """
        Record a calibration event for monitoring
        
        Args:
            provider: ASR provider name
            raw_confidence: Raw confidence from ASR provider
            calibrated_confidence: Calibrated confidence score
            ground_truth_accuracy: Known accuracy if available
            segment_length: Segment duration
            decode_mode: Decode mode used
            session_id: Session identifier
        """
        monitoring_event = CalibrationMonitoringData(
            provider=provider,
            timestamp=time.time(),
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            ground_truth_accuracy=ground_truth_accuracy,
            segment_length=segment_length,
            decode_mode=decode_mode,
            session_id=session_id
        )
        
        # Add to monitoring data and maintain window size
        self.monitoring_data[provider].append(monitoring_event)
        if len(self.monitoring_data[provider]) > self.monitoring_window:
            self.monitoring_data[provider].pop(0)
        
        # Check for alerts if we have enough data
        if len(self.monitoring_data[provider]) >= self.alert_thresholds['sample_count_minimum']:
            self._check_calibration_alerts(provider)
    
    def _check_calibration_alerts(self, provider: str) -> None:
        """Check for calibration quality alerts"""
        recent_data = self.monitoring_data[provider]
        
        # Extract data for analysis
        calibrated_scores = [d.calibrated_confidence for d in recent_data]
        ground_truth_labels = [d.ground_truth_accuracy for d in recent_data if d.ground_truth_accuracy is not None]
        
        if len(ground_truth_labels) < self.alert_thresholds['sample_count_minimum']:
            return  # Not enough ground truth data for reliable alerts
        
        # Convert accuracy scores to binary labels (threshold at 0.8)
        binary_labels = [1 if acc >= 0.8 else 0 for acc in ground_truth_labels]
        
        # Calculate current ECE
        if len(binary_labels) == len(calibrated_scores):
            current_ece = self.plotter._calculate_ece(
                np.array(calibrated_scores), 
                np.array(binary_labels)
            )
            
            # Check ECE degradation
            if current_ece > self.alert_thresholds['ece_degradation_threshold']:
                alert = CalibrationAlert(
                    alert_type="ece_degradation",
                    provider=provider,
                    severity="high" if current_ece > 0.20 else "medium",
                    message=f"Calibration quality degraded for {provider}: ECE = {current_ece:.3f}",
                    metrics={'ece': current_ece, 'threshold': self.alert_thresholds['ece_degradation_threshold']},
                    timestamp=time.time(),
                    recommendations=[
                        "Review recent training data quality",
                        "Consider retraining calibration models",
                        "Check for distribution shift in inputs"
                    ]
                )
                
                self._add_alert(alert)
        
        # Check for bias drift
        mean_calibrated = np.mean(calibrated_scores)
        mean_accuracy = np.mean(ground_truth_labels)
        bias_drift = abs(mean_calibrated - mean_accuracy)
        
        if bias_drift > self.alert_thresholds['bias_drift_threshold']:
            alert = CalibrationAlert(
                alert_type="bias_drift",
                provider=provider,
                severity="medium",
                message=f"Confidence bias drift detected for {provider}: drift = {bias_drift:.3f}",
                metrics={'bias_drift': bias_drift, 'mean_confidence': mean_calibrated, 'mean_accuracy': mean_accuracy},
                timestamp=time.time(),
                recommendations=[
                    "Investigate systematic bias in confidence scores",
                    "Review calibration model assumptions",
                    "Consider bias correction techniques"
                ]
            )
            
            self._add_alert(alert)
    
    def _add_alert(self, alert: CalibrationAlert) -> None:
        """Add alert to history and log"""
        self.alert_history.append(alert)
        
        # Maintain alert history size
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)
        
        # Log alert
        self.logger.warning(f"Calibration alert: {alert.message}",
                           context={
                               'alert_type': alert.alert_type,
                               'provider': alert.provider,
                               'severity': alert.severity,
                               'metrics': alert.metrics
                           })
    
    def get_monitoring_dashboard_data(self) -> CalibrationDashboardData:
        """
        Get comprehensive monitoring dashboard data
        
        Returns:
            CalibrationDashboardData for dashboard visualization
        """
        overall_metrics = {}
        provider_metrics = {}
        reliability_diagrams = {}
        trend_data = {}
        
        # Calculate overall metrics
        total_events = sum(len(events) for events in self.monitoring_data.values())
        overall_metrics['total_calibration_events'] = total_events
        overall_metrics['active_providers'] = len(self.monitoring_data)
        overall_metrics['recent_alerts_count'] = len([a for a in self.alert_history if time.time() - a.timestamp < 3600])
        
        # Provider-specific metrics
        for provider, events in self.monitoring_data.items():
            if not events:
                continue
            
            recent_events = events[-100:]  # Last 100 events
            calibrated_scores = [e.calibrated_confidence for e in recent_events]
            raw_scores = [e.raw_confidence for e in recent_events]
            
            provider_metrics[provider] = {
                'total_events': len(events),
                'recent_events': len(recent_events),
                'mean_calibrated_confidence': np.mean(calibrated_scores),
                'std_calibrated_confidence': np.std(calibrated_scores),
                'mean_raw_confidence': np.mean(raw_scores),
                'confidence_adjustment': np.mean(calibrated_scores) - np.mean(raw_scores),
                'last_event_timestamp': events[-1].timestamp
            }
            
            # Generate reliability diagram data if ground truth available
            ground_truth_events = [e for e in recent_events if e.ground_truth_accuracy is not None]
            if len(ground_truth_events) >= 20:
                calibrated_scores_with_gt = [e.calibrated_confidence for e in ground_truth_events]
                binary_labels = [1 if e.ground_truth_accuracy >= 0.8 else 0 for e in ground_truth_events]
                
                reliability_data = self.plotter.create_reliability_diagram(
                    np.array(calibrated_scores_with_gt),
                    np.array(binary_labels),
                    provider,
                    save_path=None  # Don't save for dashboard data
                )
                
                reliability_diagrams[provider] = reliability_data
            
            # Trend data (ECE over time)
            if len(ground_truth_events) >= 50:
                trend_data[provider] = self._calculate_trend_data(ground_truth_events)
        
        # Recent alerts (last 24 hours)
        recent_alerts = [a for a in self.alert_history if time.time() - a.timestamp < 86400]
        
        return CalibrationDashboardData(
            overall_metrics=overall_metrics,
            provider_metrics=provider_metrics,
            reliability_diagrams=reliability_diagrams,
            recent_alerts=recent_alerts,
            trend_data=trend_data
        )
    
    def _calculate_trend_data(self, events: List[CalibrationMonitoringData]) -> List[Tuple[float, float]]:
        """Calculate ECE trend data over time"""
        # Split events into time windows
        window_size = 50  # Events per window
        trend_points = []
        
        for i in range(0, len(events) - window_size + 1, 10):  # Sliding window with step
            window_events = events[i:i + window_size]
            
            calibrated_scores = [e.calibrated_confidence for e in window_events]
            binary_labels = [1 if e.ground_truth_accuracy >= 0.8 else 0 for e in window_events]
            
            timestamp = np.mean([e.timestamp for e in window_events])
            ece = self.plotter._calculate_ece(np.array(calibrated_scores), np.array(binary_labels))
            
            trend_points.append((timestamp, ece))
        
        return trend_points
    
    def generate_monitoring_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive calibration monitoring report
        
        Args:
            output_path: Optional path to save JSON report
            
        Returns:
            Dictionary with monitoring report
        """
        dashboard_data = self.get_monitoring_dashboard_data()
        
        report = {
            'report_timestamp': time.time(),
            'monitoring_summary': dashboard_data.overall_metrics,
            'provider_analysis': dashboard_data.provider_metrics,
            'calibration_quality': {},
            'alert_summary': {
                'total_alerts': len(self.alert_history),
                'recent_alerts': len(dashboard_data.recent_alerts),
                'alert_types': {}
            },
            'recommendations': []
        }
        
        # Calibration quality summary
        for provider, reliability_data in dashboard_data.reliability_diagrams.items():
            report['calibration_quality'][provider] = {
                'ece': reliability_data.get('ece', 0.0),
                'quality_assessment': self._assess_calibration_quality(reliability_data.get('ece', 0.0)),
                'sample_count': reliability_data.get('sample_count', 0)
            }
        
        # Alert type summary
        alert_types = defaultdict(int)
        for alert in self.alert_history:
            alert_types[alert.alert_type] += 1
        report['alert_summary']['alert_types'] = dict(alert_types)
        
        # Generate recommendations
        report['recommendations'] = self._generate_monitoring_recommendations(dashboard_data)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Monitoring report saved to {output_path}")
        
        return report
    
    def _assess_calibration_quality(self, ece: float) -> str:
        """Assess calibration quality based on ECE"""
        if ece < 0.05:
            return "excellent"
        elif ece < 0.10:
            return "good"
        elif ece < 0.15:
            return "fair"
        else:
            return "poor"
    
    def _generate_monitoring_recommendations(self, dashboard_data: CalibrationDashboardData) -> List[str]:
        """Generate recommendations based on monitoring data"""
        recommendations = []
        
        # Check for providers with poor calibration
        for provider, reliability_data in dashboard_data.reliability_diagrams.items():
            ece = reliability_data.get('ece', 0.0)
            if ece > 0.15:
                recommendations.append(f"Provider {provider} shows poor calibration (ECE={ece:.3f}) - consider retraining")
        
        # Check for recent alerts
        if len(dashboard_data.recent_alerts) > 5:
            recommendations.append("High alert frequency detected - investigate system stability")
        
        # Check for insufficient monitoring data
        low_sample_providers = []
        for provider, metrics in dashboard_data.provider_metrics.items():
            if metrics['recent_events'] < 50:
                low_sample_providers.append(provider)
        
        if low_sample_providers:
            recommendations.append(f"Insufficient monitoring data for: {', '.join(low_sample_providers)}")
        
        return recommendations