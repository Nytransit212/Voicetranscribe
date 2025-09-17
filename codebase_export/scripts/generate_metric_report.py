#!/usr/bin/env python3
"""
Comprehensive metric reporting script for ensemble transcription system.

This script generates detailed reports from acceptance test results,
performs trend analysis, and creates visualizations for metric tracking.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metric_calculator import MetricReporter, MetricCalculator
from utils.structured_logger import StructuredLogger


class ComprehensiveMetricReporter:
    """Generate comprehensive metric reports with visualizations and trends"""
    
    def __init__(self, reports_dir: str = "test_reports"):
        """
        Initialize comprehensive metric reporter
        
        Args:
            reports_dir: Directory containing test reports
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        self.logger = StructuredLogger("metric_reporter")
        self.metric_calculator = MetricCalculator()
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive metric report with trends and analysis
        
        Returns:
            Path to generated HTML report
        """
        # Collect all test results
        test_results = self._collect_test_results()
        
        if not test_results:
            self.logger.warning("No test results found for reporting")
            return ""
        
        # Generate analysis
        trend_analysis = self._analyze_trends(test_results)
        regression_analysis = self._detect_regressions(test_results)
        quality_summary = self._summarize_quality_metrics(test_results)
        
        # Create visualizations
        trend_plots = self._create_trend_plots(test_results)
        quality_dashboard = self._create_quality_dashboard(test_results)
        
        # Generate HTML report
        report_path = self._generate_html_report(
            test_results, trend_analysis, regression_analysis, 
            quality_summary, trend_plots, quality_dashboard
        )
        
        self.logger.info(f"Comprehensive metric report generated: {report_path}")
        return str(report_path)
    
    def _collect_test_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect all test results from report files"""
        
        results = {}
        
        # Collect latest results
        for result_file in self.reports_dir.glob("latest_results_*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    test_id = data["test_id"]
                    
                    if test_id not in results:
                        results[test_id] = []
                    
                    results[test_id].append(data)
                    
            except Exception as e:
                self.logger.warning(f"Failed to load result file {result_file}: {e}")
        
        # Collect historical results
        for result_file in self.reports_dir.glob("metrics_report_*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    test_id = data.get("test_name", "unknown")
                    
                    if test_id not in results:
                        results[test_id] = []
                    
                    # Convert to consistent format
                    converted_data = {
                        "test_id": test_id,
                        "timestamp": data.get("timestamp", ""),
                        "metrics": data.get("metrics", {})
                    }
                    
                    results[test_id].append(converted_data)
                    
            except Exception as e:
                self.logger.warning(f"Failed to load historical file {result_file}: {e}")
        
        return results
    
    def _analyze_trends(self, test_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze metric trends over time"""
        
        trends = {
            "overall_trend": "stable",
            "improving_metrics": [],
            "degrading_metrics": [],
            "stable_metrics": [],
            "trend_details": {}
        }
        
        for test_id, results in test_results.items():
            if len(results) < 2:
                continue
            
            # Sort by timestamp
            sorted_results = sorted(results, key=lambda x: x.get("timestamp", ""))
            
            # Extract metric series
            der_series = []
            wer_series = []
            confidence_series = []
            
            for result in sorted_results:
                metrics = result.get("metrics", {})
                
                if isinstance(metrics, dict):
                    # Handle nested structure
                    diar_metrics = metrics.get("diarization", {})
                    asr_metrics = metrics.get("asr", {})
                    
                    der_series.append(diar_metrics.get("der", 0.0))
                    wer_series.append(asr_metrics.get("wer", 0.0))
                    confidence_series.append(metrics.get("confidence_calibration", 0.0))
            
            # Calculate trends
            if len(der_series) >= 2:
                der_trend = self._calculate_trend(der_series)
                wer_trend = self._calculate_trend(wer_series)
                conf_trend = self._calculate_trend(confidence_series)
                
                trends["trend_details"][test_id] = {
                    "der_trend": der_trend,
                    "wer_trend": wer_trend,
                    "confidence_trend": conf_trend
                }
                
                # Classify trends (lower DER/WER is better, higher confidence is better)
                if der_trend < -0.02:  # Improving DER
                    trends["improving_metrics"].append(f"{test_id}_DER")
                elif der_trend > 0.02:  # Degrading DER
                    trends["degrading_metrics"].append(f"{test_id}_DER")
                else:
                    trends["stable_metrics"].append(f"{test_id}_DER")
        
        # Determine overall trend
        if len(trends["degrading_metrics"]) > len(trends["improving_metrics"]):
            trends["overall_trend"] = "degrading"
        elif len(trends["improving_metrics"]) > len(trends["degrading_metrics"]):
            trends["overall_trend"] = "improving"
        
        return trends
    
    def _calculate_trend(self, series: List[float]) -> float:
        """Calculate linear trend slope for a metric series"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = np.array(series)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _detect_regressions(self, test_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Detect performance regressions"""
        
        regressions = {
            "has_regression": False,
            "regression_count": 0,
            "regression_details": [],
            "regression_summary": {}
        }
        
        # Define regression thresholds
        thresholds = {
            "der_increase_max": 0.05,
            "wer_increase_max": 0.05,
            "confidence_decrease_max": 0.10
        }
        
        for test_id, results in test_results.items():
            if len(results) < 2:
                continue
            
            # Compare latest vs baseline (earliest)
            sorted_results = sorted(results, key=lambda x: x.get("timestamp", ""))
            baseline = sorted_results[0]
            latest = sorted_results[-1]
            
            baseline_metrics = baseline.get("metrics", {})
            latest_metrics = latest.get("metrics", {})
            
            # Check for regressions
            regressions_found = []
            
            # DER regression
            baseline_der = self._extract_metric(baseline_metrics, "der", 0.0)
            latest_der = self._extract_metric(latest_metrics, "der", 0.0)
            der_increase = latest_der - baseline_der
            
            if der_increase > thresholds["der_increase_max"]:
                regressions_found.append(f"DER increased by {der_increase:.3f}")
                regressions["has_regression"] = True
                regressions["regression_count"] += 1
            
            # WER regression
            baseline_wer = self._extract_metric(baseline_metrics, "wer", 0.0)
            latest_wer = self._extract_metric(latest_metrics, "wer", 0.0)
            wer_increase = latest_wer - baseline_wer
            
            if wer_increase > thresholds["wer_increase_max"]:
                regressions_found.append(f"WER increased by {wer_increase:.3f}")
                regressions["has_regression"] = True
                regressions["regression_count"] += 1
            
            # Confidence regression
            baseline_conf = baseline_metrics.get("confidence_calibration", 0.0)
            latest_conf = latest_metrics.get("confidence_calibration", 0.0)
            conf_decrease = baseline_conf - latest_conf
            
            if conf_decrease > thresholds["confidence_decrease_max"]:
                regressions_found.append(f"Confidence decreased by {conf_decrease:.3f}")
                regressions["has_regression"] = True
                regressions["regression_count"] += 1
            
            if regressions_found:
                regressions["regression_details"].append({
                    "test_id": test_id,
                    "regressions": regressions_found
                })
        
        return regressions
    
    def _extract_metric(self, metrics: Dict[str, Any], metric_name: str, default: float) -> float:
        """Extract metric value from nested structure"""
        
        # Try direct access
        if metric_name in metrics:
            return float(metrics[metric_name])
        
        # Try nested access
        if "diarization" in metrics and metric_name == "der":
            return float(metrics["diarization"].get("der", default))
        
        if "asr" in metrics and metric_name == "wer":
            return float(metrics["asr"].get("wer", default))
        
        return default
    
    def _summarize_quality_metrics(self, test_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Summarize current quality metrics"""
        
        summary = {
            "total_tests": len(test_results),
            "average_der": 0.0,
            "average_wer": 0.0,
            "average_confidence": 0.0,
            "best_performing_test": "",
            "worst_performing_test": "",
            "quality_distribution": {}
        }
        
        all_der = []
        all_wer = []
        all_confidence = []
        test_scores = {}
        
        for test_id, results in test_results.items():
            if not results:
                continue
            
            # Use latest result
            latest = sorted(results, key=lambda x: x.get("timestamp", ""))[-1]
            metrics = latest.get("metrics", {})
            
            der = self._extract_metric(metrics, "der", 1.0)
            wer = self._extract_metric(metrics, "wer", 1.0)
            confidence = metrics.get("confidence_calibration", 0.0)
            
            all_der.append(der)
            all_wer.append(wer)
            all_confidence.append(confidence)
            
            # Calculate composite score (lower is better for DER/WER, higher for confidence)
            composite_score = (1.0 - der) * 0.4 + (1.0 - wer) * 0.4 + confidence * 0.2
            test_scores[test_id] = composite_score
        
        if all_der:
            summary["average_der"] = float(np.mean(all_der))
            summary["average_wer"] = float(np.mean(all_wer))
            summary["average_confidence"] = float(np.mean(all_confidence))
            
            # Find best and worst performing tests
            if test_scores:
                summary["best_performing_test"] = max(test_scores.keys(), key=lambda k: test_scores[k])
                summary["worst_performing_test"] = min(test_scores.keys(), key=lambda k: test_scores[k])
            
            # Quality distribution
            excellent_count = sum(1 for der in all_der if der < 0.15)
            good_count = sum(1 for der in all_der if 0.15 <= der < 0.25)
            fair_count = sum(1 for der in all_der if 0.25 <= der < 0.35)
            poor_count = sum(1 for der in all_der if der >= 0.35)
            
            summary["quality_distribution"] = {
                "excellent": excellent_count,
                "good": good_count,
                "fair": fair_count,
                "poor": poor_count
            }
        
        return summary
    
    def _create_trend_plots(self, test_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Create trend visualization plots"""
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('DER Trends', 'WER Trends', 'Confidence Trends', 'Processing Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (test_id, results) in enumerate(test_results.items()):
            if len(results) < 2:
                continue
                
            color = colors[i % len(colors)]
            
            # Sort by timestamp
            sorted_results = sorted(results, key=lambda x: x.get("timestamp", ""))
            
            timestamps = [r.get("timestamp", "") for r in sorted_results]
            der_values = [self._extract_metric(r.get("metrics", {}), "der", 0.0) for r in sorted_results]
            wer_values = [self._extract_metric(r.get("metrics", {}), "wer", 0.0) for r in sorted_results]
            conf_values = [r.get("metrics", {}).get("confidence_calibration", 0.0) for r in sorted_results]
            time_values = [r.get("metrics", {}).get("processing_time", 0.0) for r in sorted_results]
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=timestamps, y=der_values, name=f"{test_id} DER", 
                          line=dict(color=color), showlegend=False),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=wer_values, name=f"{test_id} WER",
                          line=dict(color=color), showlegend=False),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=conf_values, name=f"{test_id} Confidence",
                          line=dict(color=color), showlegend=False),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=time_values, name=f"{test_id} Time",
                          line=dict(color=color), showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Metric Trends Over Time",
            height=600,
            showlegend=False
        )
        
        # Save plot
        plot_path = self.reports_dir / "metric_trends.html"
        fig.write_html(str(plot_path))
        
        return str(plot_path)
    
    def _create_quality_dashboard(self, test_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Create quality dashboard visualization"""
        
        # Extract latest metrics for each test
        test_names = []
        der_values = []
        wer_values = []
        confidence_values = []
        
        for test_id, results in test_results.items():
            if not results:
                continue
                
            latest = sorted(results, key=lambda x: x.get("timestamp", ""))[-1]
            metrics = latest.get("metrics", {})
            
            test_names.append(test_id)
            der_values.append(self._extract_metric(metrics, "der", 0.0))
            wer_values.append(self._extract_metric(metrics, "wer", 0.0))
            confidence_values.append(metrics.get("confidence_calibration", 0.0))
        
        # Create dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('DER by Test', 'WER by Test', 'Confidence by Test', 'Quality Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # DER bar chart
        fig.add_trace(
            go.Bar(x=test_names, y=der_values, name="DER", marker_color="red"),
            row=1, col=1
        )
        
        # WER bar chart
        fig.add_trace(
            go.Bar(x=test_names, y=wer_values, name="WER", marker_color="orange"),
            row=1, col=2
        )
        
        # Confidence bar chart
        fig.add_trace(
            go.Bar(x=test_names, y=confidence_values, name="Confidence", marker_color="green"),
            row=2, col=1
        )
        
        # Quality score scatter
        quality_scores = [(1.0 - der) * 0.5 + (1.0 - wer) * 0.3 + conf * 0.2 
                         for der, wer, conf in zip(der_values, wer_values, confidence_values)]
        
        fig.add_trace(
            go.Scatter(x=test_names, y=quality_scores, mode="markers+lines",
                      name="Quality Score", marker=dict(size=10, color="blue")),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Current Quality Dashboard",
            height=600,
            showlegend=False
        )
        
        # Save dashboard
        dashboard_path = self.reports_dir / "quality_dashboard.html"
        fig.write_html(str(dashboard_path))
        
        return str(dashboard_path)
    
    def _generate_html_report(self, test_results, trend_analysis, regression_analysis, 
                             quality_summary, trend_plots, quality_dashboard) -> str:
        """Generate comprehensive HTML report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ensemble Transcription System - Metric Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
                .regression {{ color: red; font-weight: bold; }}
                .improvement {{ color: green; font-weight: bold; }}
                .stable {{ color: blue; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Ensemble Transcription System - Metric Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Total Test Cases: {quality_summary['total_tests']}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Average DER:</strong> {quality_summary['average_der']:.3f}
                </div>
                <div class="metric">
                    <strong>Average WER:</strong> {quality_summary['average_wer']:.3f}
                </div>
                <div class="metric">
                    <strong>Average Confidence:</strong> {quality_summary['average_confidence']:.3f}
                </div>
                <div class="metric">
                    <strong>Overall Trend:</strong> 
                    <span class="{'improvement' if trend_analysis['overall_trend'] == 'improving' else 'regression' if trend_analysis['overall_trend'] == 'degrading' else 'stable'}">
                        {trend_analysis['overall_trend'].title()}
                    </span>
                </div>
            </div>
            
            <div class="section">
                <h2>Regression Analysis</h2>
                {'<div class="regression">⚠️ REGRESSIONS DETECTED</div>' if regression_analysis['has_regression'] else '<div class="improvement">✅ No regressions detected</div>'}
                <p>Regression Count: {regression_analysis['regression_count']}</p>
        """
        
        if regression_analysis['regression_details']:
            html_content += """
                <h3>Regression Details</h3>
                <ul>
            """
            for detail in regression_analysis['regression_details']:
                html_content += f"<li><strong>{detail['test_id']}:</strong> {', '.join(detail['regressions'])}</li>"
            html_content += "</ul>"
        
        html_content += f"""
            </div>
            
            <div class="section">
                <h2>Quality Distribution</h2>
                <table>
                    <tr>
                        <th>Quality Level</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    <tr>
                        <td>Excellent (DER < 0.15)</td>
                        <td>{quality_summary['quality_distribution']['excellent']}</td>
                        <td>{quality_summary['quality_distribution']['excellent'] / max(quality_summary['total_tests'], 1) * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Good (DER 0.15-0.25)</td>
                        <td>{quality_summary['quality_distribution']['good']}</td>
                        <td>{quality_summary['quality_distribution']['good'] / max(quality_summary['total_tests'], 1) * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Fair (DER 0.25-0.35)</td>
                        <td>{quality_summary['quality_distribution']['fair']}</td>
                        <td>{quality_summary['quality_distribution']['fair'] / max(quality_summary['total_tests'], 1) * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Poor (DER > 0.35)</td>
                        <td>{quality_summary['quality_distribution']['poor']}</td>
                        <td>{quality_summary['quality_distribution']['poor'] / max(quality_summary['total_tests'], 1) * 100:.1f}%</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Best and Worst Performing Tests</h2>
                <p><strong>Best:</strong> {quality_summary.get('best_performing_test', 'N/A')}</p>
                <p><strong>Worst:</strong> {quality_summary.get('worst_performing_test', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p><a href="{Path(trend_plots).name}" target="_blank">📈 View Metric Trends</a></p>
                <p><a href="{Path(quality_dashboard).name}" target="_blank">📊 View Quality Dashboard</a></p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Add dynamic recommendations
        if regression_analysis['has_regression']:
            html_content += "<li>🔴 Investigate and fix identified regressions before deployment</li>"
        
        if trend_analysis['overall_trend'] == 'degrading':
            html_content += "<li>🔴 Address declining performance trends</li>"
        
        if quality_summary['average_der'] > 0.25:
            html_content += "<li>🔶 Consider improvements to diarization accuracy</li>"
        
        if quality_summary['average_wer'] > 0.25:
            html_content += "<li>🔶 Consider improvements to ASR accuracy</li>"
        
        if quality_summary['average_confidence'] < 0.6:
            html_content += "<li>🔶 Improve confidence calibration accuracy</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = self.reports_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)


def main():
    """Main entry point for metric reporting script"""
    
    parser = argparse.ArgumentParser(description="Generate comprehensive metric report")
    parser.add_argument("--reports-dir", default="test_reports", 
                       help="Directory containing test reports")
    parser.add_argument("--output-format", choices=["html", "json", "both"], default="html",
                       help="Output format for report")
    
    args = parser.parse_args()
    
    reporter = ComprehensiveMetricReporter(args.reports_dir)
    
    if args.output_format in ["html", "both"]:
        html_report = reporter.generate_comprehensive_report()
        print(f"HTML report generated: {html_report}")
    
    if args.output_format in ["json", "both"]:
        # Could add JSON export functionality here
        print("JSON export not yet implemented")


if __name__ == "__main__":
    main()