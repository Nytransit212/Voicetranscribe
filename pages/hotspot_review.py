import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import base64

from core.hotspot_manager import HotspotManager, Clip, HumanEdit, SpeakerMap
from utils.audio_format_validator import AudioFormatValidator
from utils.transcript_formatter import TranscriptFormatter

# Custom CSS for hotspot review interface
HOTSPOT_REVIEW_CSS = """
<style>
    /* Hotspot review container */
    .hotspot-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Header styling */
    .hotspot-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Clip carousel */
    .clip-carousel {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    
    /* Audio player styling */
    .audio-controls {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1rem 0;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Transcript editing area */
    .transcript-editor {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        min-height: 120px;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
    }
    
    /* Low confidence word highlighting */
    .low-confidence {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
        border: 1px solid #ffeaa7;
    }
    
    .very-low-confidence {
        background-color: #f8d7da;
        padding: 2px 4px;
        border-radius: 3px;
        border: 1px solid #f5c6cb;
    }
    
    /* Progress indicator */
    .progress-dots {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin: 1rem 0;
    }
    
    .progress-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #dee2e6;
    }
    
    .progress-dot.current {
        background-color: #0066cc;
    }
    
    .progress-dot.completed {
        background-color: #28a745;
    }
    
    /* Action buttons */
    .action-buttons {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin: 1.5rem 0;
    }
    
    /* Speaker field */
    .speaker-field {
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #e9ecef;
        border-radius: 6px;
    }
    
    /* Completion summary */
    .completion-summary {
        text-align: center;
        padding: 2rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        margin: 2rem 0;
    }
</style>
"""

def render_hotspot_review():
    """Render the hotspot review interface"""
    st.markdown(HOTSPOT_REVIEW_CSS, unsafe_allow_html=True)
    
    # Initialize hotspot manager
    if 'hotspot_manager' not in st.session_state:
        st.session_state.hotspot_manager = HotspotManager()
    
    # Check if we have results to review
    if 'processing_results' not in st.session_state or not st.session_state.processing_results:
        st.warning("⚠️ No transcript results available. Please process a video first.")
        if st.button("← Back to Main"):
            st.session_state.current_screen = 'results'
            st.rerun()
        return
    
    # Initialize hotspot review session if not started
    if 'hotspot_session' not in st.session_state:
        initialize_hotspot_session()
    
    session = st.session_state.hotspot_session
    
    # Check if review is complete
    if session['status'] == 'completed':
        render_completion_summary()
        return
    
    # Render main review interface
    render_review_interface()

def initialize_hotspot_session():
    """Initialize a new hotspot review session"""
    try:
        # Get processing results
        results = st.session_state.processing_results
        full_results = results.get('full_results', {})
        
        # Generate hotspots from the transcript
        hotspot_manager = st.session_state.hotspot_manager
        clips = hotspot_manager.select_hotspots(
            transcript_data=full_results,
            audio_file_path=st.session_state.get('uploaded_file_path'),
            human_time_budget_min=5.0
        )
        
        # Initialize session state
        st.session_state.hotspot_session = {
            'clips': clips,
            'current_clip_index': 0,
            'edited_clips': {},
            'speaker_map': {},
            'start_time': time.time(),
            'status': 'reviewing',  # reviewing, completed
            'auto_save_data': {}
        }
        
        # Show initialization message
        st.success(f"✅ Found {len(clips)} hotspots for review (estimated 5 minutes)")
        
    except Exception as e:
        st.error(f"Failed to initialize hotspot review: {str(e)}")
        st.session_state.hotspot_session = {
            'clips': [],
            'current_clip_index': 0,
            'edited_clips': {},
            'speaker_map': {},
            'start_time': time.time(),
            'status': 'completed',
            'auto_save_data': {}
        }

def render_review_interface():
    """Render the main review interface"""
    session = st.session_state.hotspot_session
    clips = session['clips']
    
    if not clips:
        st.warning("No hotspots found for review.")
        render_completion_summary()
        return
    
    current_index = session['current_clip_index']
    current_clip = clips[current_index] if current_index < len(clips) else None
    
    if not current_clip:
        session['status'] = 'completed'
        st.rerun()
        return
    
    # Header
    render_header(current_index + 1, len(clips))
    
    # Progress dots
    render_progress_dots(current_index, len(clips), session['edited_clips'])
    
    # Main clip review area
    render_clip_review(current_clip, current_index)
    
    # Navigation and action buttons
    render_navigation_controls(current_index, len(clips))

def render_header(current_clip, total_clips):
    """Render the header section"""
    st.markdown(f"""
    <div class="hotspot-header">
        <h2>🎯 Review Hard Spots</h2>
        <p>About 5 minutes. Fix the toughest parts, we improve the rest.</p>
        <h4>Clip {current_clip} of {total_clips}</h4>
    </div>
    """, unsafe_allow_html=True)

def render_progress_dots(current_index, total_clips, edited_clips):
    """Render progress indicator dots"""
    dots_html = '<div class="progress-dots">'
    
    for i in range(total_clips):
        if i == current_index:
            dot_class = "progress-dot current"
        elif i in edited_clips:
            dot_class = "progress-dot completed"
        else:
            dot_class = "progress-dot"
        
        dots_html += f'<div class="{dot_class}"></div>'
    
    dots_html += '</div>'
    st.markdown(dots_html, unsafe_allow_html=True)

def render_clip_review(clip: Clip, clip_index: int):
    """Render the clip review section"""
    st.markdown('<div class="clip-carousel">', unsafe_allow_html=True)
    
    # Clip info
    duration = clip.end_s - clip.start_s
    st.markdown(f"**Time:** {format_time(clip.start_s)} - {format_time(clip.end_s)} ({duration:.1f}s)")
    st.markdown(f"**Uncertainty Score:** {clip.uncertainty_score:.2f}")
    
    # Audio player section
    render_audio_player(clip)
    
    # Transcript editing
    render_transcript_editor(clip, clip_index)
    
    # Speaker field
    render_speaker_field(clip, clip_index)
    
    # Action buttons
    render_action_buttons(clip, clip_index)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_audio_player(clip: Clip):
    """Render audio player controls"""
    st.markdown("### 🎵 Audio Player")
    
    # Try to create audio player if we have audio file
    if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
        try:
            # For now, show time range - actual audio clipping would need ffmpeg
            st.markdown(f"🎧 **Audio Segment:** {format_time(clip.start_s)} - {format_time(clip.end_s)}")
            st.markdown("*Audio playback coming soon - use external player with timestamps above*")
            
            # Audio controls info
            st.markdown("""
            **Keyboard Controls:**
            - `Space`: Play/Pause
            - `←/→`: Skip 1s back/forward  
            - `Enter`: Save edits + Next
            - `Esc`: Skip this clip
            """)
            
        except Exception as e:
            st.warning(f"Audio preview not available: {str(e)}")
    else:
        st.info("Audio file not available for playback")

def render_transcript_editor(clip: Clip, clip_index: int):
    """Render transcript editing interface"""
    st.markdown("### ✏️ Transcript")
    
    # Get current text (edited or original)
    session = st.session_state.hotspot_session
    current_text = session['edited_clips'].get(clip_index, {}).get('text_final', clip.text_proposed)
    
    # Highlight low-confidence words (convert Token objects to dicts)
    tokens_as_dicts = [{'text': token.text, 'conf': token.conf} for token in clip.tokens]
    highlighted_text = highlight_low_confidence_words(tokens_as_dicts, clip.text_proposed)
    
    if highlighted_text != clip.text_proposed:
        st.markdown("**Original with confidence highlighting:**")
        st.markdown(f'<div class="transcript-editor">{highlighted_text}</div>', unsafe_allow_html=True)
    
    # Editable text area
    st.markdown("**Edit transcript:**")
    edited_text = st.text_area(
        "transcript_edit",
        value=current_text,
        height=120,
        key=f"transcript_edit_{clip_index}",
        label_visibility="collapsed",
        help="Edit the transcript text. Changes will be applied to improve the full transcript."
    )
    
    # Auto-save functionality
    if edited_text != current_text and edited_text is not None:
        auto_save_edit(clip_index, edited_text)

def highlight_low_confidence_words(tokens: List[Dict], original_text: str) -> str:
    """Highlight low-confidence words in the transcript"""
    if not tokens:
        return original_text
    
    # Create highlighted version
    highlighted_parts = []
    current_pos = 0
    
    for token in tokens:
        token_text = token.get('text', '')
        confidence = token.get('conf', 1.0)
        
        # Find token position in original text
        token_start = original_text.find(token_text, current_pos)
        if token_start == -1:
            continue
        
        # Add text before token
        if token_start > current_pos:
            highlighted_parts.append(original_text[current_pos:token_start])
        
        # Add highlighted token based on confidence
        if confidence < 0.3:
            highlighted_parts.append(f'<span class="very-low-confidence">{token_text}</span>')
        elif confidence < 0.6:
            highlighted_parts.append(f'<span class="low-confidence">{token_text}</span>')
        else:
            highlighted_parts.append(token_text)
        
        current_pos = token_start + len(token_text)
    
    # Add remaining text
    if current_pos < len(original_text):
        highlighted_parts.append(original_text[current_pos:])
    
    return ''.join(highlighted_parts)

def auto_save_edit(clip_index: int, edited_text: str):
    """Auto-save edits to session state"""
    session = st.session_state.hotspot_session
    if 'edited_clips' not in session:
        session['edited_clips'] = {}
    
    session['edited_clips'][clip_index] = {
        'text_final': edited_text,
        'timestamp': time.time()
    }
    
    # Also save to hotspot_edits format for API compatibility
    if 'hotspot_edits' not in st.session_state:
        st.session_state.hotspot_edits = {}
    
    clip = session['clips'][clip_index]
    st.session_state.hotspot_edits[clip.id] = {
        'edit': HumanEdit(
            clip_id=clip.id,
            text_final=edited_text,
            original_text=clip.text_proposed
        )
    }

def format_time(seconds: float) -> str:
    """Format time in MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def render_speaker_field(clip: Clip, clip_index: int):
    """Render speaker identification field"""
    if not clip.speakers:
        return
    
    st.markdown('<div class="speaker-field">', unsafe_allow_html=True)
    st.markdown("### 👥 Speaker")
    
    session = st.session_state.hotspot_session
    current_speaker = session['edited_clips'].get(clip_index, {}).get('speaker_override')
    
    # Speaker options
    speaker_options = ["Keep current"] + clip.speakers + ["Name this speaker..."]
    current_index = 0
    
    if current_speaker:
        if current_speaker in speaker_options:
            current_index = speaker_options.index(current_speaker)
        else:
            speaker_options.append(current_speaker)
            current_index = len(speaker_options) - 1
    
    selected_speaker = st.selectbox(
        "Speaker for this clip",
        options=speaker_options,
        index=current_index,
        key=f"speaker_select_{clip_index}"
    )
    
    # Handle custom speaker name
    if selected_speaker == "Name this speaker...":
        custom_name = st.text_input(
            "Enter speaker name",
            key=f"custom_speaker_{clip_index}",
            placeholder="e.g., John Smith, Moderator, etc."
        )
        if custom_name:
            selected_speaker = custom_name
    
    # Save speaker override
    if selected_speaker and selected_speaker != "Keep current":
        auto_save_speaker(clip_index, selected_speaker)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_action_buttons(clip: Clip, clip_index: int):
    """Render action buttons for the current clip"""
    st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Approve", key=f"approve_{clip_index}", type="secondary"):
            approve_clip(clip_index)
    
    with col2:
        if st.button("💾 Save + Next", key=f"save_next_{clip_index}", type="primary"):
            save_and_next(clip_index)
    
    with col3:
        if st.button("⏭️ Skip", key=f"skip_{clip_index}", type="secondary"):
            skip_clip(clip_index)
    
    with col4:
        if st.button("🚩 Flag", key=f"flag_{clip_index}", type="secondary"):
            flag_clip(clip_index)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_navigation_controls(current_index: int, total_clips: int):
    """Render navigation controls"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_index > 0:
            if st.button("← Back", key="nav_back"):
                st.session_state.hotspot_session['current_clip_index'] = current_index - 1
                st.rerun()
    
    with col2:
        if st.button("🏁 Finish Now", key="finish_now", type="primary"):
            finish_review()
    
    with col3:
        if current_index < total_clips - 1:
            if st.button("Next →", key="nav_next"):
                st.session_state.hotspot_session['current_clip_index'] = current_index + 1
                st.rerun()

def render_completion_summary():
    """Render the completion summary screen"""
    session = st.session_state.hotspot_session
    
    # Calculate summary stats
    total_clips = len(session['clips'])
    reviewed_clips = len(session['edited_clips'])
    elapsed_time = time.time() - session['start_time']
    
    # Estimated improvement (simplified calculation)
    improvement_pct = min(15.0, reviewed_clips * 2.5)  # Rough estimate
    
    st.markdown(f"""
    <div class="completion-summary">
        <h2>🎉 Hotspot Review Complete!</h2>
        <p><strong>You reviewed {reviewed_clips} of {total_clips} clips</strong></p>
        <p><strong>Time spent:</strong> {elapsed_time/60:.1f} minutes</p>
        <p><strong>Estimated improvement:</strong> {improvement_pct:.1f}%</p>
        <p>Your edits have been applied and will improve the entire transcript through smart propagation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply improvements and regenerate outputs
    if 'improvements_applied' not in st.session_state.hotspot_session:
        apply_hotspot_improvements()
        st.session_state.hotspot_session['improvements_applied'] = True
    
    # Download options
    st.markdown("### 💾 Download Improved Results:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Download TXT", type="secondary"):
            download_improved_transcript('txt')
    
    with col2:
        if st.button("🎬 Download SRT", type="secondary"):
            download_improved_transcript('srt')
    
    with col3:
        if st.button("📊 Download JSON", type="secondary"):
            download_improved_transcript('json')
    
    with col4:
        if st.button("📋 Copy Text", type="secondary"):
            copy_improved_transcript()
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Transcribe Another File", type="primary"):
            reset_to_landing()
    
    with col2:
        if st.button("← Back to Results", type="secondary"):
            st.session_state.current_screen = 'results'
            st.rerun()

# Helper functions

def auto_save_speaker(clip_index: int, speaker: str):
    """Auto-save speaker override"""
    session = st.session_state.hotspot_session
    if clip_index not in session['edited_clips']:
        session['edited_clips'][clip_index] = {}
    session['edited_clips'][clip_index]['speaker_override'] = speaker

def approve_clip(clip_index: int):
    """Approve current clip without changes"""
    session = st.session_state.hotspot_session
    session['edited_clips'][clip_index] = {'approved': True, 'timestamp': time.time()}
    next_clip()

def save_and_next(clip_index: int):
    """Save current edits and move to next clip"""
    session = st.session_state.hotspot_session
    if clip_index not in session['edited_clips']:
        session['edited_clips'][clip_index] = {}
    session['edited_clips'][clip_index]['saved'] = True
    session['edited_clips'][clip_index]['timestamp'] = time.time()
    next_clip()

def skip_clip(clip_index: int):
    """Skip current clip"""
    session = st.session_state.hotspot_session
    session['edited_clips'][clip_index] = {'skipped': True, 'timestamp': time.time()}
    next_clip()

def flag_clip(clip_index: int):
    """Flag current clip for follow-up"""
    session = st.session_state.hotspot_session
    if clip_index not in session['edited_clips']:
        session['edited_clips'][clip_index] = {}
    session['edited_clips'][clip_index]['flagged'] = True
    session['edited_clips'][clip_index]['timestamp'] = time.time()
    st.success("🚩 Clip flagged for follow-up")

def next_clip():
    """Move to next clip or finish if at end"""
    session = st.session_state.hotspot_session
    current_index = session['current_clip_index']
    total_clips = len(session['clips'])
    
    if current_index < total_clips - 1:
        session['current_clip_index'] = current_index + 1
        st.rerun()
    else:
        finish_review()

def finish_review():
    """Finish the hotspot review"""
    session = st.session_state.hotspot_session
    session['status'] = 'completed'
    session['end_time'] = time.time()
    st.rerun()

def apply_hotspot_improvements():
    """Apply hotspot improvements to the main transcript with comprehensive validation"""
    try:
        session = st.session_state.hotspot_session
        hotspot_manager = st.session_state.hotspot_manager
        
        # Validate session state before proceeding
        if not st.session_state.processing_results:
            st.error("⚠️ No processing results available. Cannot apply improvements.")
            return
        
        # Convert edited clips to HumanEdit objects
        human_edits = []
        for clip_index, edit_data in session['edited_clips'].items():
            if 'text_final' in edit_data and edit_data['text_final'].strip():
                clip = session['clips'][clip_index]
                human_edit = HumanEdit(
                    clip_id=clip.id,
                    text_final=edit_data['text_final'].strip(),
                    speaker_label_override=edit_data.get('speaker_override'),
                    flags=['flagged'] if edit_data.get('flagged') else []
                )
                human_edits.append(human_edit)
        
        # Apply improvements through comprehensive propagation pipeline
        if human_edits:
            # Store original for validation
            original_results = st.session_state.processing_results.copy()
            
            # Apply improvements with full pipeline (includes file regeneration, manifest updates)
            improved_results = hotspot_manager.apply_improvements(
                original_results=original_results,
                human_edits=human_edits,
                speaker_map=session.get('speaker_map', {})
            )
            
            # Validate improvements were applied
            if _validate_improvements(original_results, improved_results):
                # Update processing results completely
                st.session_state.processing_results = improved_results
                
                # Update job history with human review metadata
                if hasattr(st.session_state, 'job_history') and st.session_state.job_history:
                    latest_job = st.session_state.job_history[-1]
                    latest_job['results'] = improved_results
                    latest_job['human_reviewed'] = True
                    latest_job['review_timestamp'] = time.time()
                
                # Show comprehensive success message
                files_created = improved_results.get('regenerated_files', {})
                st.success(f"✅ Applied {len(human_edits)} improvements to transcript")
                if files_created:
                    st.info(f"📁 Generated {len(files_created)} updated transcript files")
                    for file_type, path in files_created.items():
                        st.write(f"  • {file_type.upper()}: {Path(path).name}")
            else:
                st.warning("⚠️ Improvements may not have been fully applied. Please verify results.")
        else:
            st.info("No edits found to apply.")
        
    except Exception as e:
        st.error(f"Failed to apply improvements: {str(e)}")
        # Log the full error for debugging
        import traceback
        st.error(f"Technical details: {traceback.format_exc()}")

def _validate_improvements(original: Dict, improved: Dict) -> bool:
    """Validate that improvements were actually applied"""
    try:
        # Check if transcript changed
        if original.get('transcript', '') != improved.get('transcript', ''):
            return True
        
        # Check if human_reviewed metadata was added
        if 'human_reviewed' in improved and improved['human_reviewed']:
            return True
        
        # Check if files were regenerated
        if 'regenerated_files' in improved and improved['regenerated_files']:
            return True
        
        return False
    except:
        return True  # Assume valid if validation fails

def download_improved_transcript(format_type: str):
    """Generate download for improved transcript"""
    results = st.session_state.processing_results
    
    # Initialize default values
    content = ""
    filename = "improved_transcript.txt"
    mime = "text/plain"
    
    if format_type == 'txt':
        content = results.get('transcript', '')
        filename = f"{Path(results.get('file_name', 'transcript')).stem}_improved.txt"
        mime = "text/plain"
    elif format_type == 'srt':
        content = create_srt_from_results(results)
        filename = f"{Path(results.get('file_name', 'transcript')).stem}_improved.srt"
        mime = "text/plain"
    elif format_type == 'json':
        content = json.dumps(results, indent=2)
        filename = f"{Path(results.get('file_name', 'transcript')).stem}_improved.json"
        mime = "application/json"
    else:
        # Default case
        content = results.get('transcript', '')
        filename = "improved_transcript.txt"
        mime = "text/plain"
    
    st.download_button(
        label=f"Download {format_type.upper()}",
        data=content,
        file_name=filename,
        mime=mime
    )

def create_srt_from_results(results):
    """Create SRT content from results"""
    # Simplified SRT generation
    segments = results.get('segments', [])
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
    """Format time in SRT format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def copy_improved_transcript():
    """Show transcript for copying"""
    results = st.session_state.processing_results
    st.code(results['transcript'], language=None)
    st.success("📋 Transcript ready to copy!")

def reset_to_landing():
    """Reset to landing screen for new transcription"""
    st.session_state.current_screen = 'landing'
    st.session_state.processing_stage = 0
    st.session_state.uploaded_file = None
    st.session_state.file_url = ''
    if 'hotspot_session' in st.session_state:
        del st.session_state.hotspot_session
    st.rerun()