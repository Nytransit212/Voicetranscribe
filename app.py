import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from core.ensemble_manager import EnsembleManager
from core.audio_processor import AudioProcessor
from core.diarization_engine import DiarizationEngine
from utils.file_handler import FileHandler
from utils.transcript_formatter import TranscriptFormatter
from pages.qc_dashboard import render_qc_dashboard
import traceback

st.set_page_config(
    page_title="Advanced Ensemble Transcription System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    
    # Page selection
    page_options = {
        'main': '🏠 Main Processing',
        'qc': '🔍 Quality Control'
    }
    
    selected_page = st.sidebar.radio(
        "Select Page",
        options=list(page_options.keys()),
        format_func=lambda x: page_options[x],
        index=0 if st.session_state.current_page == 'main' else 1
    )
    
    st.session_state.current_page = selected_page
    
    # Results status in sidebar
    if st.session_state.results:
        st.sidebar.success("✅ Results Available")
        winner_score = st.session_state.results.get('winner_score', 0)
        st.sidebar.metric("Winner Score", f"{winner_score:.3f}")
        
        processing_time = st.session_state.results.get('processing_time', 0)
        st.sidebar.metric("Processing Time", f"{processing_time:.1f}s")
    else:
        st.sidebar.info("ℹ️ No results yet")
    
    # Route to appropriate page
    if selected_page == 'main':
        render_main_page()
    elif selected_page == 'qc':
        render_qc_dashboard()

def render_main_page():
    st.title("🎯 Advanced Ensemble Transcription System")
    st.markdown("Generate 15 candidate transcripts with multi-dimensional confidence scoring")

    # Check for required API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    # Check FFmpeg availability
    ffmpeg_available, ffmpeg_info = AudioProcessor.check_ffmpeg_availability()
    if not ffmpeg_available:
        st.error(f"⚠️ FFmpeg is required but not available: {ffmpeg_info}")
        with st.expander("📥 FFmpeg Installation Instructions", expanded=True):
            st.markdown(AudioProcessor.get_ffmpeg_install_instructions())
        st.stop()
    else:
        st.success(f"✅ FFmpeg available: {ffmpeg_info}")
    
    # Check diarization capability
    diarization_engine = DiarizationEngine()
    if hasattr(diarization_engine, 'pipeline') and hasattr(diarization_engine.pipeline, '__class__'):
        pipeline_class = diarization_engine.pipeline.__class__.__name__
        if "Mock" in pipeline_class or not hasattr(diarization_engine, '_validate_hf_token') or not diarization_engine._validate_hf_token(os.getenv("HUGGINGFACE_TOKEN")):
            st.warning("⚠️ **Using Mock Diarization Pipeline**")
            st.info("""
            **Important Notice:**
            - Real speaker diarization is not available (pyannote.audio not properly configured)
            - Using mock pipeline with synthetic speaker boundaries
            - Transcription quality may be reduced
            - For production use, configure pyannote.audio with valid HUGGINGFACE_TOKEN
            """)
        else:
            st.success("✅ Real diarization pipeline available")
    else:
        st.warning("⚠️ Diarization pipeline status unknown")

    # File upload section
    st.header("📁 Upload Video File")
    
    # Language selection
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an MP4 video file (up to 90 minutes)",
            type=['mp4'],
            help="Upload your video file for ensemble transcription processing"
        )
    
    with col2:
        language_options = {
            'auto': 'Auto-detect',
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'zh': 'Chinese',
            'ko': 'Korean',
            'ar': 'Arabic'
        }
        
        selected_language = st.selectbox(
            "Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0,  # Default to auto-detect
            help="Select the primary language or use auto-detect"
        )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📄 File: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Processing parameters
        st.header("⚙️ Processing Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            expected_speakers = st.slider(
                "Expected number of speakers",
                min_value=2,
                max_value=20,
                value=10,
                help="Approximate number of participants in the recording"
            )
            
        with col2:
            noise_level = st.selectbox(
                "Room noise level",
                ["Low", "Medium", "High"],
                index=1,
                help="Background noise level in the recording environment"
            )

        # Process button
        if st.button("🚀 Start Ensemble Processing", disabled=st.session_state.processing):
            process_video(uploaded_file, expected_speakers, noise_level, selected_language)

    # Display processing status
    if st.session_state.processing:
        display_processing_status()
    
    # Display results
    if st.session_state.results:
        display_results()
        
        # QC Navigation
        st.header("🔍 Quality Control")
        st.markdown("Review and improve transcript quality with automated QC and targeted repairs")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Open Quality Control Dashboard", type="primary"):
                st.session_state.current_page = 'qc'
                st.rerun()
        
        with col2:
            st.info("💡 Use QC Dashboard to review flagged segments and apply repairs")

def process_video(uploaded_file, expected_speakers, noise_level, selected_language):
    """Process the uploaded video through the ensemble pipeline"""
    st.session_state.processing = True
    st.session_state.results = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    tmp_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Initialize ensemble manager
        ensemble_manager = EnsembleManager(
            expected_speakers=expected_speakers,
            noise_level=noise_level.lower(),
            target_language=selected_language if selected_language != 'auto' else None
        )
        
        # Process through ensemble pipeline
        def update_progress(step, progress, message):
            progress_bar.progress(progress)
            status_text.text(f"Step {step}: {message}")
        
        results = ensemble_manager.process_video(tmp_path, update_progress)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Store results
        st.session_state.results = results
        st.session_state.processing = False
        
        progress_bar.progress(100)
        status_text.text("✅ Processing completed successfully!")
        
        st.success("🎉 Ensemble transcription completed! Check the results below.")
        st.rerun()
        
    except Exception as e:
        st.session_state.processing = False
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Processing failed: {str(e)}")
        st.error("📋 Error details:")
        st.code(traceback.format_exc())
        
        # Clean up temporary file if it exists
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except:
            pass

def display_processing_status():
    """Display current processing status"""
    st.header("🔄 Processing Status")
    st.info("Ensemble processing is running... This may take several minutes for long videos.")
    
    with st.expander("📊 Processing Steps", expanded=True):
        st.markdown("""
        1. **Audio Extraction** - Converting MP4 to 16kHz mono WAV
        2. **Audio Preprocessing** - Noise reduction and normalization
        3. **Diarization Variants** - Creating 3 speaker diarization variants
        4. **ASR Ensemble** - Running 5 ASR passes per diarization (15 total)
        5. **Confidence Scoring** - Evaluating candidates across 5 dimensions
        6. **Winner Selection** - Selecting best transcript using weighted formula
        7. **Output Generation** - Creating final transcripts and subtitles
        """)

def display_results():
    """Display processing results and download options"""
    results = st.session_state.results
    
    st.header("🏆 Ensemble Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Winner Score",
            f"{results['winner_score']:.3f}",
            help="Final confidence score of the winning transcript"
        )
    
    with col2:
        st.metric(
            "Candidates Generated",
            "15",
            help="Total number of candidate transcripts evaluated"
        )
    
    with col3:
        st.metric(
            "Processing Time",
            f"{results['processing_time']:.1f}s",
            help="Total time spent processing the video"
        )
    
    with col4:
        st.metric(
            "Speaker Count",
            results['detected_speakers'],
            help="Number of unique speakers detected"
        )

    # Confidence breakdown
    st.subheader("📈 Confidence Score Breakdown")
    
    confidence_data = results['confidence_breakdown']
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Score Components:**")
        for component, score in confidence_data.items():
            st.write(f"- **{component}**: {score:.3f}")
    
    with col2:
        st.markdown("**Score Weights:**")
        weights = {
            'Diarization (D)': '0.28',
            'ASR Alignment (A)': '0.32', 
            'Linguistic Quality (L)': '0.18',
            'Cross-run Agreement (R)': '0.12',
            'Overlap Handling (O)': '0.10'
        }
        for component, weight in weights.items():
            st.write(f"- **{component}**: {weight}")

    # Transcript preview
    st.subheader("📄 Winning Transcript Preview")
    
    transcript_preview = results['transcript_preview']
    with st.expander("View transcript excerpt (first 10 segments)", expanded=True):
        for segment in transcript_preview:
            timestamp = segment['timestamp']
            speaker = segment['speaker']
            text = segment['text']
            confidence = segment['confidence']
            
            st.write(f"**[{timestamp}] {speaker}** (confidence: {confidence:.2f})")
            st.write(f"*{text}*")
            st.write("---")

    # Ensemble audit
    st.subheader("🔍 Ensemble Audit")
    
    audit_data = results['ensemble_audit']
    
    with st.expander("View candidate rankings", expanded=False):
        st.markdown("**Top 5 Candidates:**")
        for i, candidate in enumerate(audit_data['top_candidates'][:5]):
            rank = i + 1
            score = candidate['final_score']
            variant = candidate['variant_info']
            
            st.write(f"{rank}. **Score: {score:.3f}** - {variant}")

    # Download section
    st.header("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Transcripts")
        
        # JSON transcript
        if st.download_button(
            label="📄 Download JSON Transcript",
            data=json.dumps(results['winner_transcript'], indent=2),
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        ):
            st.success("JSON transcript downloaded!")
        
        # TXT transcript
        if st.download_button(
            label="📝 Download TXT Transcript", 
            data=results['winner_transcript_txt'],
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        ):
            st.success("TXT transcript downloaded!")

    with col2:
        st.subheader("🎬 Subtitles")
        
        # VTT captions
        if st.download_button(
            label="📺 Download WebVTT Captions",
            data=results['captions_vtt'],
            file_name=f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vtt",
            mime="text/vtt"
        ):
            st.success("VTT captions downloaded!")
        
        # SRT captions  
        if st.download_button(
            label="📺 Download SRT Captions",
            data=results['captions_srt'],
            file_name=f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
            mime="text/plain"
        ):
            st.success("SRT captions downloaded!")

    with col3:
        st.subheader("📊 Reports")
        
        # Ensemble audit
        if st.download_button(
            label="📈 Download Ensemble Audit",
            data=json.dumps(results['ensemble_audit'], indent=2),
            file_name=f"ensemble_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        ):
            st.success("Audit report downloaded!")
        
        # Confidence report
        if st.download_button(
            label="🎯 Download Confidence Report",
            data=json.dumps(results['confidence_breakdown'], indent=2),
            file_name=f"confidence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        ):
            st.success("Confidence report downloaded!")

    # Reset button
    st.header("🔄 Process Another File")
    if st.button("🗑️ Clear Results and Start Over"):
        st.session_state.results = None
        st.session_state.uploaded_file = None
        st.session_state.processing = False
        st.rerun()

if __name__ == "__main__":
    main()
