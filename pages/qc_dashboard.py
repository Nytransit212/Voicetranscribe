import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
import numpy as np

from core.quality_controller import QualityController, QualityLevel, IssueSeverity, IssueType
from core.repair_engine import RepairEngine
from utils.transcript_formatter import TranscriptFormatter

def render_qc_dashboard():
    """Render the Quality Control dashboard page"""
    st.title("🔍 Quality Control Dashboard")
    st.markdown("Review and improve transcript quality with automated QC and targeted repairs")
    
    # Initialize QC components
    if 'quality_controller' not in st.session_state:
        st.session_state.quality_controller = QualityController()
    if 'repair_engine' not in st.session_state:
        st.session_state.repair_engine = RepairEngine()
    
    # Check if we have results to analyze
    if 'results' not in st.session_state or not st.session_state.results:
        st.warning("⚠️ No transcript results available. Please process a video first from the main page.")
        return
    
    results = st.session_state.results
    master_transcript = results['winner_transcript']
    
    # QC Configuration
    st.header("⚙️ Quality Control Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_quality_level = st.selectbox(
            "Target Quality Level",
            options=[level.value for level in QualityLevel],
            index=1,  # Default to 'review'
            help="Quality standards to apply for assessment"
        )
        quality_level = QualityLevel(target_quality_level)
    
    with col2:
        auto_repair = st.checkbox(
            "Enable Auto-Repair",
            value=False,
            help="Automatically apply best repair candidates"
        )
    
    with col3:
        if st.button("🔍 Run Quality Assessment", type="primary"):
            with st.spinner("Performing quality assessment..."):
                assessment = st.session_state.quality_controller.assess_transcript_quality(
                    master_transcript, quality_level
                )
                st.session_state.qc_assessment = assessment
                st.success("Quality assessment completed!")
                st.rerun()
    
    # Display Quality Assessment Results
    if 'qc_assessment' in st.session_state:
        assessment = st.session_state.qc_assessment
        
        # Quality Overview
        st.header("📊 Quality Assessment Overview")
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            grade_color = {
                QualityLevel.PREMIUM: "green",
                QualityLevel.FINAL: "blue", 
                QualityLevel.REVIEW: "orange",
                QualityLevel.DRAFT: "red"
            }[assessment.overall_grade]
            
            st.metric(
                "Quality Grade",
                assessment.overall_grade.value.upper(),
                help="Overall quality assessment grade"
            )
            st.markdown(f"<div style='color: {grade_color}'>●</div>", unsafe_allow_html=True)
        
        with col2:
            gate_status = "✅ PASS" if assessment.passes_quality_gate else "❌ FAIL"
            st.metric(
                "Quality Gate",
                gate_status,
                help=f"Passes {quality_level.value} quality standards"
            )
        
        with col3:
            st.metric(
                "Quality Score",
                f"{assessment.quality_score:.3f}",
                help="Overall quality score (0.0-1.0)"
            )
        
        with col4:
            st.metric(
                "Issues Found",
                f"{len(assessment.issues)}",
                help="Total number of quality issues detected"
            )
        
        with col5:
            flagged_ratio = assessment.flagged_segments / assessment.total_segments * 100
            st.metric(
                "Flagged Segments",
                f"{assessment.flagged_segments}/{assessment.total_segments}",
                delta=f"{flagged_ratio:.1f}%"
            )
        
        # Quality Charts
        st.subheader("📈 Quality Metrics Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Issue severity distribution
            severity_counts = {
                'Critical': assessment.metrics['critical_issues'],
                'High': assessment.metrics['high_issues'],
                'Medium': assessment.metrics['medium_issues'],
                'Low': assessment.metrics['low_issues']
            }
            
            fig_severity = px.pie(
                values=list(severity_counts.values()),
                names=list(severity_counts.keys()),
                title="Issues by Severity",
                color_discrete_map={
                    'Critical': '#ff4444',
                    'High': '#ff8800',
                    'Medium': '#ffaa00',
                    'Low': '#ffdd00'
                }
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            # Confidence distribution histogram
            segments = master_transcript.get('segments', [])
            segment_confidences = [seg.get('confidence', 0) for seg in segments]
            
            fig_hist = px.histogram(
                x=segment_confidences,
                nbins=20,
                title="Segment Confidence Distribution",
                labels={'x': 'Confidence Score', 'y': 'Number of Segments'}
            )
            fig_hist.add_vline(
                x=assessment.metrics['mean_segment_confidence'],
                line_dash="dash",
                annotation_text="Mean"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Issues Table
        st.subheader("🚨 Quality Issues")
        
        if assessment.issues:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=[s.value for s in IssueSeverity],
                    default=[IssueSeverity.CRITICAL.value, IssueSeverity.HIGH.value]
                )
            
            with col2:
                issue_type_filter = st.multiselect(
                    "Filter by Issue Type",
                    options=[t.value for t in IssueType],
                    default=[]
                )
            
            with col3:
                show_all_segments = st.checkbox("Show all segment issues", value=False)
            
            # Filter issues
            filtered_issues = []
            for issue in assessment.issues:
                if severity_filter and issue.severity.value not in severity_filter:
                    continue
                if issue_type_filter and issue.issue_type.value not in issue_type_filter:
                    continue
                if not show_all_segments and issue.segment_index == -1:
                    continue
                filtered_issues.append(issue)
            
            # Display issues
            if filtered_issues:
                issues_data = []
                for issue in filtered_issues:
                    segment_text = ""
                    if issue.segment_index >= 0 and issue.segment_index < len(segments):
                        segment_text = segments[issue.segment_index].get('text', '')[:100]
                        if len(segment_text) == 100:
                            segment_text += "..."
                    
                    issues_data.append({
                        'Segment': issue.segment_index if issue.segment_index >= 0 else 'Overall',
                        'Severity': issue.severity.value.upper(),
                        'Type': issue.issue_type.value.replace('_', ' ').title(),
                        'Score': f"{issue.confidence_score:.3f}",
                        'Description': issue.description,
                        'Text': segment_text,
                        'Action': issue.suggested_action
                    })
                
                issues_df = pd.DataFrame(issues_data)
                st.dataframe(issues_df, use_container_width=True)
                
                # Repair Actions
                st.subheader("🔧 Repair Actions")
                
                # Find repair candidates
                if st.button("🔍 Find Repair Candidates"):
                    with st.spinner("Analyzing repair options..."):
                        # Get all candidates from results
                        all_candidates = results.get('ensemble_audit', {}).get('all_candidates', [])
                        
                        repair_candidates = st.session_state.quality_controller.get_repair_candidates(
                            master_transcript, all_candidates, assessment
                        )
                        
                        st.session_state.repair_candidates = repair_candidates
                        st.success(f"Found repair candidates for {len(repair_candidates)} segments")
                        st.rerun()
                
                # Display repair candidates
                if 'repair_candidates' in st.session_state:
                    render_repair_interface(master_transcript, st.session_state.repair_candidates)
            
            else:
                st.info("No issues match the current filters.")
        
        else:
            st.success("🎉 No quality issues found! This transcript meets all quality standards.")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        for i, recommendation in enumerate(assessment.recommendations, 1):
            st.write(f"{i}. {recommendation}")
        
        # Export Quality Report
        st.subheader("📋 Export Quality Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Generate Quality Report"):
                quality_report = generate_quality_report(assessment, master_transcript)
                st.session_state.quality_report = quality_report
                st.success("Quality report generated!")
        
        with col2:
            if 'quality_report' in st.session_state:
                st.download_button(
                    label="💾 Download Quality Report",
                    data=st.session_state.quality_report,
                    file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def render_repair_interface(master_transcript: Dict[str, Any], 
                          repair_candidates: Dict[int, List[Dict[str, Any]]]):
    """Render the repair interface for problematic segments"""
    st.subheader("🛠️ Segment Repair Interface")
    
    if not repair_candidates:
        st.info("No repair candidates available.")
        return
    
    # Segment selector
    segment_indices = list(repair_candidates.keys())
    selected_segment = st.selectbox(
        "Select segment to repair",
        options=segment_indices,
        format_func=lambda x: f"Segment {x} ({master_transcript['segments'][x]['start']:.1f}s)"
    )
    
    if selected_segment in repair_candidates:
        candidates = repair_candidates[selected_segment]
        original_segment = master_transcript['segments'][selected_segment]
        
        # Display original segment
        st.write("**Original Segment:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Time:** {original_segment['start']:.1f}s - {original_segment['end']:.1f}s")
        with col2:
            st.write(f"**Speaker:** {original_segment.get('speaker', 'Unknown')}")
        with col3:
            st.write(f"**Confidence:** {original_segment.get('confidence', 0):.3f}")
        
        st.write(f"**Text:** {original_segment.get('text', '')}")
        
        # Display repair candidates
        if candidates:
            st.write("**Repair Candidates:**")
            
            candidate_data = []
            for i, candidate in enumerate(candidates):
                candidate_data.append({
                    'Rank': i + 1,
                    'Source': candidate['candidate_id'],
                    'Confidence': f"{candidate['confidence']:.3f}",
                    'Score': f"{candidate['alternative_score']:.3f}",
                    'Overlap': f"{candidate['overlap_ratio']:.1%}",
                    'Text': candidate['segment'].get('text', '')[:150] + ('...' if len(candidate['segment'].get('text', '')) > 150 else '')
                })
            
            candidates_df = pd.DataFrame(candidate_data)
            
            # Make the dataframe selectable
            selected_rows = st.dataframe(
                candidates_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            # Apply repair button
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Apply Best Candidate"):
                    if candidates:
                        best_candidate = candidates[0]  # Already sorted by score
                        updated_transcript = st.session_state.quality_controller.apply_segment_repair(
                            master_transcript, selected_segment, best_candidate
                        )
                        
                        # Update session state
                        st.session_state.results['winner_transcript'] = updated_transcript
                        st.success(f"Applied repair to segment {selected_segment}")
                        st.rerun()
            
            with col2:
                if st.button("🔄 Reprocess Segment"):
                    with st.spinner("Reprocessing segment..."):
                        # Get audio path if available
                        if 'uploaded_file' in st.session_state and st.session_state.uploaded_file:
                            # This would require audio extraction - simplified for demo
                            st.info("Segment reprocessing would be implemented here")
            
            with col3:
                if st.button("✏️ Manual Edit"):
                    st.session_state.show_manual_edit = selected_segment
                    st.rerun()
        
        # Manual edit interface
        if 'show_manual_edit' in st.session_state and st.session_state.show_manual_edit == selected_segment:
            render_manual_edit_interface(master_transcript, selected_segment)

def render_manual_edit_interface(master_transcript: Dict[str, Any], segment_index: int):
    """Render manual edit interface for a segment"""
    st.subheader("✏️ Manual Edit")
    
    segment = master_transcript['segments'][segment_index]
    
    with st.form(f"edit_segment_{segment_index}"):
        # Editable fields
        new_text = st.text_area(
            "Transcript Text",
            value=segment.get('text', ''),
            height=100
        )
        
        new_speaker = st.text_input(
            "Speaker",
            value=segment.get('speaker', '')
        )
        
        new_confidence = st.slider(
            "Manual Confidence Override",
            min_value=0.0,
            max_value=1.0,
            value=segment.get('confidence', 0.0),
            step=0.01
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("💾 Save Changes"):
                # Update segment
                updated_segment = segment.copy()
                updated_segment['text'] = new_text
                updated_segment['speaker'] = new_speaker
                updated_segment['confidence'] = new_confidence
                updated_segment['manually_edited'] = True
                updated_segment['edit_timestamp'] = time.time()
                
                # Update transcript
                updated_transcript = master_transcript.copy()
                updated_transcript['segments'][segment_index] = updated_segment
                
                # Update session state
                st.session_state.results['winner_transcript'] = updated_transcript
                
                # Clear edit mode
                if 'show_manual_edit' in st.session_state:
                    del st.session_state.show_manual_edit
                
                st.success("Segment updated successfully!")
                st.rerun()
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                if 'show_manual_edit' in st.session_state:
                    del st.session_state.show_manual_edit
                st.rerun()

def render_quality_timeline(master_transcript: Dict[str, Any], assessment):
    """Render quality timeline showing confidence over time"""
    segments = master_transcript.get('segments', [])
    
    if not segments:
        return
    
    # Prepare timeline data
    timeline_data = []
    for i, segment in enumerate(segments):
        timeline_data.append({
            'segment_index': i,
            'start_time': segment['start'],
            'end_time': segment['end'],
            'confidence': segment.get('confidence', 0),
            'speaker': segment.get('speaker', 'Unknown'),
            'text_preview': segment.get('text', '')[:50] + ('...' if len(segment.get('text', '')) > 50 else '')
        })
    
    df = pd.DataFrame(timeline_data)
    
    # Create timeline chart
    fig = px.scatter(
        df,
        x='start_time',
        y='confidence',
        color='speaker',
        size='end_time',
        hover_data=['segment_index', 'text_preview'],
        title="Confidence Timeline",
        labels={'start_time': 'Time (seconds)', 'confidence': 'Confidence Score'}
    )
    
    # Add quality threshold line
    fig.add_hline(
        y=0.6,  # Example threshold
        line_dash="dash",
        annotation_text="Quality Threshold"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def generate_quality_report(assessment, master_transcript: Dict[str, Any]) -> str:
    """Generate comprehensive quality report"""
    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'report_version': '1.0',
            'assessment_level': assessment.overall_grade.value
        },
        'quality_summary': {
            'overall_grade': assessment.overall_grade.value,
            'passes_quality_gate': assessment.passes_quality_gate,
            'quality_score': assessment.quality_score,
            'total_segments': assessment.total_segments,
            'flagged_segments': assessment.flagged_segments
        },
        'metrics': assessment.metrics,
        'issues': [
            {
                'segment_index': issue.segment_index,
                'issue_type': issue.issue_type.value,
                'severity': issue.severity.value,
                'confidence_score': issue.confidence_score,
                'description': issue.description,
                'suggested_action': issue.suggested_action
            }
            for issue in assessment.issues
        ],
        'recommendations': assessment.recommendations,
        'transcript_metadata': master_transcript.get('metadata', {})
    }
    
    return json.dumps(report, indent=2)

def render_batch_operations():
    """Render batch operations interface"""
    st.subheader("🔄 Batch Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 Auto-Repair All Critical Issues"):
            st.info("Auto-repair functionality would be implemented here")
    
    with col2:
        if st.button("📊 Regenerate Quality Assessment"):
            if 'qc_assessment' in st.session_state:
                del st.session_state.qc_assessment
            st.rerun()
    
    with col3:
        if st.button("💾 Export Repaired Transcript"):
            if 'results' in st.session_state:
                transcript_data = st.session_state.results['winner_transcript']
                st.download_button(
                    label="📄 Download Repaired Transcript",
                    data=json.dumps(transcript_data, indent=2),
                    file_name=f"repaired_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )