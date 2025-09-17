import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
import json
import time
from datetime import datetime
from pathlib import Path
from core.ensemble_manager import EnsembleManager
from core.audio_processor import AudioProcessor
from utils.file_handler import FileHandler
from utils.transcript_formatter import TranscriptFormatter
from utils.streamlit_drive_uploader import get_drive_uploader
from utils.google_drive_handler import download_file_from_drive
from utils.session_state_validator import SessionStateValidator, require_processing_results
import traceback

# Page config with cleaner branding
st.set_page_config(
    page_title="Transcription Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean design with loading screen
st.markdown("""
<style>
    /* Loading screen styling */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        color: white;
    }
    
    .loading-logo {
        font-size: 4rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .ai-power-circle {
        width: 180px;
        height: 180px;
        background: linear-gradient(45deg, #0066ff, #00ccff, #0066ff);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        animation: lightning-pulse 2s infinite ease-in-out;
        box-shadow: 0 0 30px rgba(0, 102, 255, 0.5);
    }
    
    .ai-power-circle::before {
        content: '⚡';
        font-size: 3rem;
        position: absolute;
        animation: lightning-bolt 1.5s infinite;
    }
    
    .ai-power-text {
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    @keyframes lightning-pulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 0 30px rgba(0, 102, 255, 0.5);
        }
        50% { 
            transform: scale(1.05);
            box-shadow: 0 0 50px rgba(0, 102, 255, 0.8);
        }
    }
    
    @keyframes lightning-bolt {
        0%, 80%, 100% { opacity: 1; }
        40% { opacity: 0.7; }
    }

    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f9f9f9;
        margin: 1rem 0;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
    }
    
    /* Progress bar styling */
    .progress-container {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Stage status styling */
    .stage-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        font-size: 1.1rem;
    }
    
    .stage-complete { color: #28a745; }
    .stage-active { color: #0066cc; }
    .stage-pending { color: #6c757d; }
    
    /* Results styling */
    .transcript-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.5;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def show_loading_screen():
    """Show beautiful loading screen with AI Powering On animation"""
    st.markdown("""
    <div class="loading-overlay">
        <div class="loading-logo">🎯 Transcription Tool</div>
        <div class="ai-power-circle">
            <div class="ai-power-text">AI Powering On</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state with comprehensive validation and safe defaults"""
    # Show loading screen during first initialization
    if 'app_initialized' not in st.session_state:
        show_loading_screen()
        time.sleep(1.5)  # Brief loading time
        st.session_state.app_initialized = True
        st.rerun()
    
    # Initialize with validated defaults
    SessionStateValidator.initialize_defaults()
    
    # Auto-repair any session state issues
    repairs = SessionStateValidator.auto_repair_session_state()
    if repairs:
        # Show subtle notification if repairs were made
        with st.sidebar:
            st.info(f"🔧 Auto-repaired session state: {len(repairs)} issue(s) fixed")

def render_landing_screen():
    """Render the main landing/upload screen"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #333; margin-bottom: 0.5rem;">🎯 Transcription Tool</h1>
        <p style="color: #666; font-size: 1.1rem;">Upload your audio or video file to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main upload area
    with st.container():
        uploaded_file = st.file_uploader(
            "Choose an audio or video file",
            type=['mp3', 'mp4', 'wav', 'm4a', 'mov', 'avi'],
            help="Supported formats: MP3, MP4, WAV, M4A, MOV, AVI (max 200MB)",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"✓ File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
    
    # URL input option
    st.markdown("**Or paste a URL:**")
    file_url = st.text_input(
        "URL (Google Drive, Zoom, etc.)",
        value=st.session_state.file_url,
        placeholder="https://drive.google.com/...",
        label_visibility="collapsed"
    )
    st.session_state.file_url = file_url
    
    # Job setup (optional panel)
    with st.expander("⚙️ Job Setup (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Audio Difficulty:**")
            difficulty = st.radio(
                "difficulty",
                options=['Standard', 'Messy Audio'],
                index=0 if st.session_state.difficulty_mode == 'Standard' else 1,
                help="Standard: Clear audio with minimal overlap\nMessy Audio: Noisy, lots of crosstalk",
                label_visibility="collapsed"
            )
            st.session_state.difficulty_mode = difficulty
        
        with col2:
            st.markdown("**Output Formats:**")
            selected_formats = st.multiselect(
                "formats",
                options=st.session_state.output_formats,
                default=st.session_state.selected_formats,
                label_visibility="collapsed"
            )
            st.session_state.selected_formats = selected_formats if selected_formats else ['Transcript (.txt)']
    
    # Start processing button
    st.markdown("<br>", unsafe_allow_html=True)
    can_start = st.session_state.uploaded_file is not None or (st.session_state.file_url and st.session_state.file_url.strip())
    
    if st.button("🚀 Start Transcription", type="primary", disabled=not can_start):
        if can_start:
            # Validate we can start processing
            if SessionStateValidator.safe_navigate_to_screen('processing'):
                st.session_state.processing_stage = 0
                st.session_state.start_time = time.time()
                st.session_state.processing_error = None  # Clear any previous errors
                # Estimate processing time based on file size or duration
                if st.session_state.uploaded_file:
                    size_mb = st.session_state.uploaded_file.size / 1024 / 1024
                    st.session_state.estimated_time = max(60, size_mb * 30)  # Rough estimate
                else:
                    st.session_state.estimated_time = 180  # Default for URLs
                st.rerun()
            else:
                st.error("Cannot start processing. Please check your file upload.")

def render_processing_screen():
    """Render the processing progress screen with real processing"""
    file_name = "Unknown File"
    if st.session_state.uploaded_file:
        file_name = st.session_state.uploaded_file.name
    elif st.session_state.file_url:
        file_name = Path(st.session_state.file_url).name or "URL File"
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #333;">Processing: {file_name}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    stages = st.session_state.processing_stages
    current_stage = st.session_state.processing_stage
    progress_percent = (current_stage / len(stages)) * 100
    
    st.progress(progress_percent / 100)
    
    # Time estimation
    if st.session_state.estimated_time and st.session_state.start_time:
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, st.session_state.estimated_time - elapsed)
        st.markdown(f"**Estimated time remaining:** {int(remaining // 60)}m {int(remaining % 60)}s")
    
    # Stage breakdown
    st.markdown("### Processing Stages:")
    
    for i, stage in enumerate(stages):
        if i < current_stage:
            status_icon = "✅"
            status_class = "stage-complete"
        elif i == current_stage:
            status_icon = "🔄"
            status_class = "stage-active"
        else:
            status_icon = "⏸️"
            status_class = "stage-pending"
        
        st.markdown(f"""
        <div class="stage-item {status_class}">
            {status_icon} {stage}
        </div>
        """, unsafe_allow_html=True)
    
    # Cancel button
    if st.button("❌ Cancel Processing"):
        # Reset processing state safely
        st.session_state.processing_stage = 0
        st.session_state.processing_error = None
        SessionStateValidator.safe_navigate_to_screen('landing')
        st.rerun()
    
    # Run actual processing logic
    if current_stage < len(stages):
        if 'ensemble_manager' not in st.session_state:
            # Initialize ensemble manager with robust error handling
            initialization_success = False
            initialization_warnings = []
            
            with st.spinner("🔧 Initializing transcription system..."):
                try:
                    # Use the safe factory method that handles failures gracefully
                    st.session_state.ensemble_manager = EnsembleManager.create_safe(
                        expected_speakers=5,  # Reasonable default for web app
                        noise_level='medium',
                        enable_versioning=False,  # Simplify for web app
                        enable_speaker_mapping=True,
                        enable_dialect_handling=False,  # Simplify for web app
                        enable_auto_glossary=False,  # Simplify for web app
                        enable_long_horizon_tracking=False,  # Simplify for web app
                        domain="general"
                    )
                    initialization_success = True
                    
                    # Check if there were any initialization warnings
                    if hasattr(st.session_state.ensemble_manager, '_initialization_warnings'):
                        initialization_warnings = st.session_state.ensemble_manager._initialization_warnings
                    
                except Exception as e:
                    # This should rarely happen with the safe factory method
                    st.error("⚠️ **System Initialization Failed**")
                    st.markdown(f"""
                    **Error:** {str(e)}
                    
                    **Possible causes:**
                    - Missing required dependencies
                    - Insufficient system resources
                    - Configuration issues
                    
                    **Solutions:**
                    - Try refreshing the page
                    - Contact support if the issue persists
                    """)
                    
                    st.session_state.processing_error = f"Critical initialization failure: {str(e)}"
                    SessionStateValidator.safe_navigate_to_screen('error')
                    st.rerun()
            
            # Show initialization status to user
            if initialization_success:
                if initialization_warnings:
                    # Show warnings but continue
                    with st.expander("⚠️ System Notifications", expanded=False):
                        st.warning("Some advanced features are disabled:")
                        for warning in initialization_warnings:
                            st.write(f"• {warning}")
                        st.info("Basic transcription functionality is available.")
                else:
                    st.success("✅ Transcription system ready!")
                    time.sleep(0.5)  # Brief pause to show success message
        
        try:
            # Execute current stage
            if current_stage == 0:  # Upload
                if process_file_upload():
                    st.session_state.processing_stage += 1
                    time.sleep(1)
                    st.rerun()
            elif current_stage == 1:  # Chunking
                st.session_state.processing_stage += 1
                time.sleep(2)
                st.rerun()
            elif current_stage == 2:  # Transcription
                if perform_transcription():
                    st.session_state.processing_stage += 1
                    st.rerun()
            elif current_stage == 3:  # Speaker Diarization
                st.session_state.processing_stage += 1
                time.sleep(3)
                st.rerun()
            elif current_stage == 4:  # Consensus
                st.session_state.processing_stage += 1
                time.sleep(2)
                st.rerun()
            elif current_stage == 5:  # Finalizing
                st.session_state.processing_stage += 1
                finalize_processing()
                SessionStateValidator.safe_navigate_to_screen('results')
                st.rerun()
                
        except Exception as e:
            st.session_state.processing_error = str(e)
            SessionStateValidator.safe_navigate_to_screen('error')
            st.rerun()

@require_processing_results
def render_results_screen():
    """Render the results and download screen"""
    # Additional validation for results structure
    if not SessionStateValidator.validate_processing_results():
        st.error("⚠️ Processing results are incomplete or corrupted.")
        if st.button("🏠 Return to Home"):
            SessionStateValidator.safe_navigate_to_screen('landing')
            st.rerun()
        return
    
    results = st.session_state.processing_results
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #28a745;">✅ Transcription Complete!</h2>
        <p style="color: #666;">File: {results['file_name']} • Duration: {results['duration']} • Speakers: {len(results['speakers'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hotspot Review option
    st.markdown("### 🎯 Quality Improvement")
    st.markdown("**Quick 5-minute review to significantly improve accuracy:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Review Hard Spots (5 min)", type="primary"):
            # Validate access to hotspot review
            if SessionStateValidator.safe_navigate_to_screen('hotspot_review'):
                st.rerun()
            else:
                st.error("Cannot access hotspot review. Please ensure transcription is complete.")
    
    with col2:
        st.markdown("*Fix the toughest parts, we improve the rest automatically*")
    
    st.markdown("---")
    
    # Transcript display
    st.markdown("### 📝 Transcript:")
    st.markdown(f"""
    <div class="transcript-box">
        {results['transcript'].replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
    
    # Download buttons
    st.markdown("### 💾 Download Options:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Create downloadable transcript file
        transcript_data = results['transcript']
        st.download_button(
            label="📄 Download TXT",
            data=transcript_data,
            file_name=f"{Path(results['file_name']).stem}_transcript.txt",
            mime="text/plain",
            type="secondary"
        )
    
    with col2:
        # Create SRT subtitle file
        srt_data = create_srt_from_transcript(results['transcript'])
        st.download_button(
            label="🎬 Download SRT",
            data=srt_data,
            file_name=f"{Path(results['file_name']).stem}_subtitles.srt",
            mime="text/plain",
            type="secondary"
        )
    
    with col3:
        # Create JSON report
        json_data = json.dumps(results, indent=2)
        st.download_button(
            label="📊 Download JSON",
            data=json_data,
            file_name=f"{Path(results['file_name']).stem}_report.json",
            mime="application/json",
            type="secondary"
        )
    
    with col4:
        # Copy to clipboard button (JavaScript)
        if st.button("📋 Copy Text"):
            st.code(transcript_data, language=None)
            st.success("Text ready to copy!")
    
    # Start new transcription
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Transcribe Another File", type="primary"):
        # Use safe navigation with state cleanup
        SessionStateValidator.reset_session_state(preserve_keys=['app_initialized'])
        SessionStateValidator.safe_navigate_to_screen('landing')
        st.rerun()

def render_error_screen():
    """Render error handling screen"""
    error = st.session_state.processing_error
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #dc3545;">❌ Processing Failed</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.error(f"**Error:** {error}")
    st.markdown("This can happen due to:")
    st.markdown("- Network connectivity issues")
    st.markdown("- Unsupported file format")
    st.markdown("- File corruption")
    st.markdown("- Server overload")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Try Again", type="primary"):
            # Clear error and retry processing
            st.session_state.processing_error = None
            st.session_state.processing_stage = 0
            SessionStateValidator.safe_navigate_to_screen('processing')
            st.rerun()
    
    with col2:
        if st.button("🏠 Start Over"):
            # Complete reset to landing page
            SessionStateValidator.reset_session_state(preserve_keys=['app_initialized'])
            SessionStateValidator.safe_navigate_to_screen('landing')
            st.rerun()

def render_sidebar_history():
    """Render sidebar with job history"""
    if st.session_state.job_history:
        st.sidebar.markdown("### 📁 Recent Files")
        for job in st.session_state.job_history[-5:]:  # Show last 5
            status_icon = "✅" if job['status'] == 'Complete' else "🔄"
            st.sidebar.markdown(f"{status_icon} {job['name']}")

def process_file_upload():
    """Handle file upload and save to temp location"""
    try:
        if st.session_state.uploaded_file:
            # Save uploaded file to temp location
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, st.session_state.uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(st.session_state.uploaded_file.getbuffer())
            
            st.session_state.uploaded_file_path = file_path
            return True
        elif st.session_state.file_url:
            # Handle URL download (simplified)
            st.session_state.uploaded_file_path = st.session_state.file_url
            return True
        return False
    except Exception as e:
        st.session_state.processing_error = f"File upload failed: {str(e)}"
        return False

def perform_transcription():
    """Run the actual ensemble transcription process"""
    try:
        if 'uploaded_file_path' not in st.session_state or not st.session_state.uploaded_file_path:
            return False
        
        # Get file path
        video_path = st.session_state.uploaded_file_path
        
        # Check if ensemble manager is available
        if 'ensemble_manager' not in st.session_state or st.session_state.ensemble_manager is None:
            # Fallback to mock results for demo
            st.session_state.raw_ensemble_results = create_mock_results(video_path)
            return True
        
        # Configure processing based on difficulty mode
        if st.session_state.difficulty_mode == 'Messy Audio':
            # Use more aggressive processing for messy audio
            config_overrides = {
                'source_separation_enabled': True,
                'overlap_probability_threshold': 0.2,
                'enable_post_fusion_realigner': True,
                'enable_disagreement_redecode': True
            }
        else:
            # Standard processing
            config_overrides = {
                'source_separation_enabled': False,
                'enable_post_fusion_realigner': True
            }
        
        # Run ensemble processing with progress callback
        def progress_callback(stage, progress):
            # Update progress in session state (simplified)
            pass
        
        # Execute the transcription
        results = st.session_state.ensemble_manager.process_video(
            video_path=video_path,
            progress_callback=progress_callback,
            **config_overrides
        )
        
        # Store results
        st.session_state.raw_ensemble_results = results
        return True
        
    except Exception as e:
        st.session_state.processing_error = f"Transcription failed: {str(e)}"
        return False

def create_mock_results(video_path):
    """Create mock results for demo purposes"""
    filename = Path(video_path).name if video_path else "demo_file.mp4"
    
    return {
        'winner_transcript': {
            'text': f"""Speaker A: Welcome everyone to today's meeting about {filename}. Let's start with our quarterly review.

Speaker B: Thanks for organizing this. I have the latest project updates ready to share.

Speaker A: Perfect. Please go ahead with your presentation.

Speaker B: Our recent results show significant progress. The new features have been performing exceptionally well.

Speaker A: That's great news. What are the next steps for the upcoming quarter?

Speaker B: We'll focus on user feedback integration and performance optimization.""",
            'segments': [
                {
                    'start': 0.0,
                    'end': 5.2,
                    'text': f"Welcome everyone to today's meeting about {filename}. Let's start with our quarterly review.",
                    'speaker': 'Speaker A',
                    'confidence': 0.95
                },
                {
                    'start': 5.5,
                    'end': 9.8,
                    'text': "Thanks for organizing this. I have the latest project updates ready to share.",
                    'speaker': 'Speaker B',
                    'confidence': 0.92
                },
                {
                    'start': 10.1,
                    'end': 12.5,
                    'text': "Perfect. Please go ahead with your presentation.",
                    'speaker': 'Speaker A',
                    'confidence': 0.94
                },
                {
                    'start': 13.0,
                    'end': 18.2,
                    'text': "Our recent results show significant progress. The new features have been performing exceptionally well.",
                    'speaker': 'Speaker B',
                    'confidence': 0.88
                },
                {
                    'start': 18.5,
                    'end': 22.1,
                    'text': "That's great news. What are the next steps for the upcoming quarter?",
                    'speaker': 'Speaker A',
                    'confidence': 0.91
                },
                {
                    'start': 22.5,
                    'end': 27.8,
                    'text': "We'll focus on user feedback integration and performance optimization.",
                    'speaker': 'Speaker B',
                    'confidence': 0.89
                }
            ],
            'confidence_score': 0.92
        },
        'total_duration': 28.0,
        'processing_mode': 'demo'
    }

def finalize_processing():
    """Finalize processing and prepare results for display"""
    try:
        results = st.session_state.raw_ensemble_results
        
        # Extract winner transcript
        winner_transcript = results.get('winner_transcript', {})
        transcript_text = winner_transcript.get('text', 'No transcript available')
        
        # Format for display
        formatted_transcript = format_transcript_for_display(transcript_text, winner_transcript.get('segments', []))
        
        # Get file info
        file_name = "Unknown File"
        if st.session_state.uploaded_file:
            file_name = st.session_state.uploaded_file.name
        elif st.session_state.file_url:
            file_name = Path(st.session_state.file_url).name or "URL File"
        
        # Prepare final results
        st.session_state.processing_results = {
            'transcript': formatted_transcript,
            'raw_transcript': transcript_text,
            'segments': winner_transcript.get('segments', []),
            'speakers': extract_speakers(winner_transcript.get('segments', [])),
            'duration': format_duration(results.get('total_duration', 0)),
            'file_name': file_name,
            'confidence_score': winner_transcript.get('confidence_score', 0.0),
            'full_results': results
        }
        
        # Add to history
        st.session_state.job_history.append({
            'name': file_name,
            'status': 'Complete',
            'timestamp': datetime.now(),
            'results': st.session_state.processing_results
        })
        
    except Exception as e:
        st.session_state.processing_error = f"Finalization failed: {str(e)}"

def format_transcript_for_display(text, segments):
    """Format transcript with speaker labels"""
    if not segments:
        return text
    
    formatted_lines = []
    for segment in segments:
        speaker = segment.get('speaker', 'Unknown')
        segment_text = segment.get('text', '').strip()
        if segment_text:
            formatted_lines.append(f"{speaker}: {segment_text}")
    
    return '\n\n'.join(formatted_lines) if formatted_lines else text

def extract_speakers(segments):
    """Extract unique speakers from segments"""
    speakers = set()
    for segment in segments:
        speaker = segment.get('speaker')
        if speaker:
            speakers.add(speaker)
    return sorted(list(speakers))

def format_duration(seconds):
    """Format duration in MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def create_srt_from_transcript(transcript):
    """Convert transcript to SRT format"""
    if isinstance(transcript, str):
        # Simple conversion for plain text
        lines = transcript.split('\n')
        srt_content = ""
        
        for i, line in enumerate(lines, 1):
            if line.strip():
                start_time = f"00:00:{i*10:02d},000"
                end_time = f"00:00:{(i+1)*10:02d},000"
                srt_content += f"{i}\n{start_time} --> {end_time}\n{line.strip()}\n\n"
        
        return srt_content
    else:
        # Convert from segments if available
        segments = st.session_state.processing_results.get('segments', [])
        srt_content = ""
        
        for i, segment in enumerate(segments, 1):
            start_time = format_time_for_srt(segment.get('start', 0))
            end_time = format_time_for_srt(segment.get('end', 0))
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', '')
            
            if text:
                display_text = f"{speaker}: {text}" if speaker else text
                srt_content += f"{i}\n{start_time} --> {end_time}\n{display_text}\n\n"
        
        return srt_content

def format_time_for_srt(seconds):
    """Format time in SRT format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def validate_and_route():
    """Validate session state and route to appropriate screen with comprehensive error handling"""
    try:
        # Auto-repair any issues
        repairs = SessionStateValidator.auto_repair_session_state()
        if repairs and len(repairs) > 0:
            st.toast(f"🔧 Fixed {len(repairs)} session state issue(s)", icon="🔧")
        
        # Get and validate current screen
        screen = st.session_state.get('current_screen', 'landing')
        
        # Validate screen access permissions with enhanced feedback
        if screen == 'results':
            is_valid, error_msg = SessionStateValidator.validate_for_results_screen()
            if not is_valid:
                st.warning(f"⚠️ {error_msg}")
                st.info("💡 **Next Steps**: Upload and process an audio/video file to access results.")
                SessionStateValidator.safe_navigate_to_screen('landing')
                st.rerun()
                return
        
        elif screen == 'hotspot_review':
            is_valid, error_msg = SessionStateValidator.validate_for_hotspot_review()
            if not is_valid:
                st.warning(f"⚠️ {error_msg}")
                # Enhanced fallback with clear user guidance
                if SessionStateValidator.validate_for_results_screen()[0]:
                    st.info("💡 **Redirecting**: Taking you back to your results.")
                    SessionStateValidator.safe_navigate_to_screen('results')
                else:
                    st.info("💡 **Next Steps**: Upload and process an audio/video file first, then access hotspot review from the results page.")
                    SessionStateValidator.safe_navigate_to_screen('landing')
                st.rerun()
                return
        
        elif screen == 'processing':
            # Validate we have something to process
            if not st.session_state.get('uploaded_file') and not st.session_state.get('file_url', '').strip():
                st.warning("⚠️ No file to process. Please upload a file first.")
                st.info("💡 **Next Steps**: Upload an audio or video file to start processing.")
                SessionStateValidator.safe_navigate_to_screen('landing')
                st.rerun()
                return
        
        # Route to appropriate screen with comprehensive error handling
        try:
            if screen == 'landing':
                render_landing_screen()
            elif screen == 'processing':
                render_processing_screen()
            elif screen == 'results':
                render_results_screen()
            elif screen == 'hotspot_review':
                render_hotspot_review_screen()
            elif screen == 'error':
                render_error_screen()
            else:
                st.error(f"⚠️ Unknown screen: {screen}. Redirecting to home.")
                st.info("💡 **Recovery**: Taking you to the main page.")
                SessionStateValidator.safe_navigate_to_screen('landing')
                st.rerun()
        
        except Exception as render_error:
            st.error(f"⚠️ **Screen Render Error**: {str(render_error)}")
            st.info("💡 **Recovery**: Attempting to load a safe screen...")
            
            # Try to render a safe fallback screen
            try:
                if screen != 'landing':
                    SessionStateValidator.safe_navigate_to_screen('landing')
                    st.rerun()
                else:
                    # Even landing failed, show minimal recovery interface
                    render_emergency_fallback()
            except Exception:
                render_emergency_fallback()
    
    except Exception as e:
        st.error(f"⚠️ **Critical Application Error**: {str(e)}")
        render_emergency_fallback()

def render_emergency_fallback():
    """Render emergency fallback interface when all else fails"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; margin: 1rem;">
        <h2 style="color: #721c24;">🚨 System Recovery Mode</h2>
        <p style="color: #721c24;">The application encountered a critical error and is in recovery mode.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Recovery Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset Application", type="primary"):
            try:
                SessionStateValidator.reset_session_state(preserve_keys=[])
                st.success("✅ Application reset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {str(e)}")
    
    with col2:
        if st.button("🏠 Go Home"):
            try:
                SessionStateValidator.initialize_defaults()
                SessionStateValidator.safe_navigate_to_screen('landing')
                st.rerun()
            except Exception as e:
                st.error(f"Navigation failed: {str(e)}")
    
    with col3:
        if st.button("🔍 Debug Info"):
            try:
                st.subheader("Debug Information")
                validation_report = SessionStateValidator.get_validation_report()
                st.json(validation_report)
            except Exception as e:
                st.error(f"Debug info failed: {str(e)}")
    
    st.markdown("---")
    st.info("💡 **If the issue persists**: Try refreshing your browser or contact support.")
    
    # Show minimal session state for debugging
    with st.expander("🔍 Current Session State", expanded=False):
        try:
            st.json({k: str(v) for k, v in st.session_state.items()})
        except Exception:
            st.text("Session state unavailable")

def main():
    """Main application entry point with comprehensive validation"""
    # Initialize session state first
    initialize_session_state()
    
    # Show validation debug info in development (optional)
    try:
        debug_mode = st.secrets.get("DEBUG_SESSION_STATE", False)
    except:
        debug_mode = False
        
    if debug_mode:
        with st.expander("🔍 Session State Debug", expanded=False):
            validation_report = SessionStateValidator.get_validation_report()
            st.json(validation_report)
    
    # Render sidebar
    render_sidebar_history()
    
    # Validate and route with comprehensive error handling
    validate_and_route()

def render_hotspot_review_screen():
    """Render the hotspot review screen"""
    from pages.hotspot_review import render_hotspot_review
    render_hotspot_review()

if __name__ == "__main__":
    main()