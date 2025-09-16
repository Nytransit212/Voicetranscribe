import streamlit as st
import streamlit.components.v1 as components
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
from utils.intelligent_cache import get_cache_manager
from utils.segment_worklist import get_worklist_manager
from utils.selective_asr import get_selective_asr_processor
from utils.streamlit_drive_uploader import get_drive_uploader
from utils.google_drive_handler import download_file_from_drive
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
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'drive_file_info' not in st.session_state:
        st.session_state.drive_file_info = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'
    # Audio processing paths for new two-step approach
    if 'pristine_audio_path' not in st.session_state:
        st.session_state.pristine_audio_path = None
    if 'asr_wav_path' not in st.session_state:
        st.session_state.asr_wav_path = None
    if 'audio_processing_complete' not in st.session_state:
        st.session_state.audio_processing_complete = False
    
    # Initialize scoring weights with defaults
    if 'scoring_weights' not in st.session_state:
        st.session_state.scoring_weights = {
            'D': 0.28,  # Diarization consistency
            'A': 0.32,  # ASR alignment and confidence  
            'L': 0.18,  # Linguistic quality
            'R': 0.12,  # Cross-run agreement
            'O': 0.10   # Overlap handling
        }
    
    # Initialize consensus and calibration settings
    if 'consensus_strategy' not in st.session_state:
        st.session_state.consensus_strategy = 'best_single_candidate'
    if 'calibration_method' not in st.session_state:
        st.session_state.calibration_method = 'registry_based'
    if 'enable_ab_testing' not in st.session_state:
        st.session_state.enable_ab_testing = False
    if 'ab_comparison_methods' not in st.session_state:
        st.session_state.ab_comparison_methods = {
            'consensus': ['best_single_candidate', 'weighted_voting'],
            'calibration': ['registry_based', 'raw_scores']
        }
    
    # Initialize post-fusion punctuation settings
    if 'punctuation_enabled' not in st.session_state:
        st.session_state.punctuation_enabled = True
    if 'punctuation_preset' not in st.session_state:
        st.session_state.punctuation_preset = 'meeting_light'
    if 'disfluency_level' not in st.session_state:
        st.session_state.disfluency_level = 'light'
    if 'meeting_vocabulary_enabled' not in st.session_state:
        st.session_state.meeting_vocabulary_enabled = True
    if 'capitalization_enabled' not in st.session_state:
        st.session_state.capitalization_enabled = True
    
    # Initialize source separation settings
    if 'source_separation_enabled' not in st.session_state:
        st.session_state.source_separation_enabled = True
    if 'overlap_probability_threshold' not in st.session_state:
        st.session_state.overlap_probability_threshold = 0.25
    if 'source_separation_providers' not in st.session_state:
        st.session_state.source_separation_providers = ['demucs', 'vocal_isolation', 'mixed_approach']
    if 'demucs_model_name' not in st.session_state:
        st.session_state.demucs_model_name = 'htdemucs_6s'
    
    # Initialize speaker mapping settings  
    if 'speaker_mapping_enabled' not in st.session_state:
        st.session_state.speaker_mapping_enabled = True
    if 'enable_ecapa_tdnn' not in st.session_state:
        st.session_state.enable_ecapa_tdnn = True
    if 'ecapa_embedding_dim' not in st.session_state:
        st.session_state.ecapa_embedding_dim = 192
    if 'enable_speaker_backtracking' not in st.session_state:
        st.session_state.enable_speaker_backtracking = True
    if 'backtracking_segments' not in st.session_state:
        st.session_state.backtracking_segments = 5
    if 'speaker_consistency_threshold' not in st.session_state:
        st.session_state.speaker_consistency_threshold = 0.7
    
    # Initialize dialect handling settings
    if 'dialect_handling_enabled' not in st.session_state:
        st.session_state.dialect_handling_enabled = True
    if 'supported_dialects' not in st.session_state:
        st.session_state.supported_dialects = ['general_american', 'southern_us', 'northeast_us', 'canadian_english', 'general_british']
    if 'primary_dialect' not in st.session_state:
        st.session_state.primary_dialect = 'general_american'
    if 'cmudict_g2p_enabled' not in st.session_state:
        st.session_state.cmudict_g2p_enabled = True
    if 'phonetic_agreement_threshold' not in st.session_state:
        st.session_state.phonetic_agreement_threshold = 0.8
    if 'dialect_confidence_boost' not in st.session_state:
        st.session_state.dialect_confidence_boost = 0.1

    # U7 Upgrade: Initialize U7 system settings
    if 'u7_enable_caching' not in st.session_state:
        st.session_state.u7_enable_caching = True
    if 'u7_enable_selective_reprocessing' not in st.session_state:
        st.session_state.u7_enable_selective_reprocessing = True
    if 'u7_confidence_threshold' not in st.session_state:
        st.session_state.u7_confidence_threshold = 0.65
    if 'u7_max_segments_for_reprocessing' not in st.session_state:
        st.session_state.u7_max_segments_for_reprocessing = 10
    if 'u7_enable_deterministic_processing' not in st.session_state:
        st.session_state.u7_enable_deterministic_processing = True
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    
    # Page selection
    page_options = {
        'main': '🏠 Main Processing',
        'qc': '🔍 Quality Control',
        'u7': '⚡ U7 System Management'
    }
    
    current_page_index = 0
    if st.session_state.current_page == 'qc':
        current_page_index = 1
    elif st.session_state.current_page == 'u7':
        current_page_index = 2
    
    selected_page = st.sidebar.radio(
        "Select Page",
        options=list(page_options.keys()),
        format_func=lambda x: page_options[x],
        index=current_page_index
    )
    
    st.session_state.current_page = selected_page
    
    # Scoring weights configuration
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚖️ Scoring Weights Configuration", expanded=False):
        st.markdown("**Adjust confidence scoring emphasis:**")
        
        # Create sliders for each weight
        d_weight = st.slider(
            "D - Diarization Consistency",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.scoring_weights['D'],
            step=0.01,
            help="Speaker boundary accuracy and turn stability"
        )
        
        a_weight = st.slider(
            "A - ASR Alignment & Confidence", 
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.scoring_weights['A'],
            step=0.01,
            help="Speech recognition quality and word-level confidence"
        )
        
        l_weight = st.slider(
            "L - Linguistic Quality",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.scoring_weights['L'],
            step=0.01,
            help="Grammar, coherence, and language model agreement"
        )
        
        r_weight = st.slider(
            "R - Cross-run Agreement",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.scoring_weights['R'],
            step=0.01,
            help="Consistency across multiple transcription runs"
        )
        
        o_weight = st.slider(
            "O - Overlap Handling",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.scoring_weights['O'],
            step=0.01,
            help="Quality of simultaneous speech detection"
        )
        
        # Auto-normalize weights to sum to 1.0
        total_weight = d_weight + a_weight + l_weight + r_weight + o_weight
        
        if total_weight > 0:
            # Normalize and update session state
            st.session_state.scoring_weights = {
                'D': d_weight / total_weight,
                'A': a_weight / total_weight,
                'L': l_weight / total_weight,
                'R': r_weight / total_weight,
                'O': o_weight / total_weight
            }
            
            # Show normalized weights
            st.markdown("**Normalized weights (sum = 1.0):**")
            for key, desc in [('D', 'Diarization'), ('A', 'ASR'), ('L', 'Linguistic'), ('R', 'Agreement'), ('O', 'Overlap')]:
                normalized_val = st.session_state.scoring_weights[key]
                st.text(f"{key}: {normalized_val:.2f} ({desc})")
        else:
            st.error("All weights cannot be zero!")
        
        # Reset to defaults button
        if st.button("🔄 Reset to Defaults"):
            st.session_state.scoring_weights = {
                'D': 0.28, 'A': 0.32, 'L': 0.18, 'R': 0.12, 'O': 0.10
            }
            st.rerun()
    
    # Consensus Strategy Configuration
    st.sidebar.markdown("---")
    with st.sidebar.expander("🤝 Consensus Strategy Configuration", expanded=False):
        st.markdown("**Select consensus approach for final transcript:**")
        
        consensus_options = {
            'best_single_candidate': '🏆 Best Single Candidate (Default)',
            'weighted_voting': '🗳️ Weighted Voting',
            'multidimensional_consensus': '📊 Multi-dimensional Consensus',
            'confidence_based': '📈 Confidence-based Selection'
        }
        
        selected_consensus = st.selectbox(
            "Consensus Strategy",
            options=list(consensus_options.keys()),
            format_func=lambda x: consensus_options[x],
            index=list(consensus_options.keys()).index(st.session_state.consensus_strategy),
            help="Choose how multiple candidate transcripts are combined into final result"
        )
        
        st.session_state.consensus_strategy = selected_consensus
        
        # Show strategy description
        descriptions = {
            'best_single_candidate': 'Select highest-scoring candidate with tie-breaking rules (current system)',
            'weighted_voting': 'Weighted combination of top-performing candidates',
            'multidimensional_consensus': 'Selection based on dimensional excellence patterns',
            'confidence_based': 'Selection based on confidence distribution analysis'
        }
        st.info(descriptions[selected_consensus])
    
    # Calibration Method Configuration
    st.sidebar.markdown("---")
    with st.sidebar.expander("📏 Calibration Method Configuration", expanded=False):
        st.markdown("**Select score calibration approach:**")
        
        calibration_options = {
            'registry_based': '📊 Registry-based (Default)',
            'isotonic_regression': '📈 Isotonic Regression',
            'per_domain': '🏷️ Per-domain Calibration',
            'raw_scores': '🔢 Raw Scores (No Calibration)'
        }
        
        selected_calibration = st.selectbox(
            "Calibration Method",
            options=list(calibration_options.keys()),
            format_func=lambda x: calibration_options[x],
            index=list(calibration_options.keys()).index(st.session_state.calibration_method),
            help="Choose how raw confidence scores are normalized for consistency"
        )
        
        st.session_state.calibration_method = selected_calibration
        
        # Show calibration description
        cal_descriptions = {
            'registry_based': 'Use historical statistics for z-score normalization',
            'isotonic_regression': 'Use trained isotonic regression models',
            'per_domain': 'Domain-specific adjustments with registry baseline',
            'raw_scores': 'Pass through raw scores with simple clipping'
        }
        st.info(cal_descriptions[selected_calibration])
    
    # Post-Fusion Punctuation Configuration
    st.sidebar.markdown("---")
    with st.sidebar.expander("✍️ Post-Fusion Punctuation Configuration", expanded=False):
        st.markdown("**Configure punctuation and disfluency normalization:**")
        
        # Enable punctuation processing
        punctuation_enabled = st.checkbox(
            "Enable Post-Fusion Punctuation",
            value=st.session_state.punctuation_enabled,
            help="Apply punctuation and capitalization after fusion consensus"
        )
        st.session_state.punctuation_enabled = punctuation_enabled
        
        if punctuation_enabled:
            # Punctuation preset selection
            punctuation_presets = {
                'meeting_light': '🤝 Meeting Light (Default)',
                'meeting_moderate': '🤝 Meeting Moderate', 
                'meeting_aggressive': '🤝 Meeting Aggressive',
                'general_light': '📝 General Purpose Light'
            }
            
            selected_preset = st.selectbox(
                "Punctuation Preset",
                options=list(punctuation_presets.keys()),
                format_func=lambda x: punctuation_presets[x],
                index=list(punctuation_presets.keys()).index(st.session_state.punctuation_preset),
                help="Choose punctuation configuration optimized for different contexts"
            )
            st.session_state.punctuation_preset = selected_preset
            
            # Disfluency normalization level
            disfluency_levels = {
                'light': '🌿 Light - Remove excessive fillers only',
                'moderate': '⚖️ Moderate - More aggressive cleaning', 
                'aggressive': '💪 Aggressive - Maximum polish'
            }
            
            selected_disfluency = st.selectbox(
                "Disfluency Normalization Level",
                options=list(disfluency_levels.keys()),
                format_func=lambda x: disfluency_levels[x],
                index=list(disfluency_levels.keys()).index(st.session_state.disfluency_level),
                help="Choose how aggressively to remove fillers and false starts"
            )
            st.session_state.disfluency_level = selected_disfluency
            
            # Meeting vocabulary handling
            meeting_vocab_enabled = st.checkbox(
                "Enable Meeting Vocabulary",
                value=st.session_state.meeting_vocabulary_enabled,
                help="Use meeting-specific vocabulary for better punctuation context"
            )
            st.session_state.meeting_vocabulary_enabled = meeting_vocab_enabled
            
            # Capitalization  
            capitalization_enabled = st.checkbox(
                "Enable Smart Capitalization",
                value=st.session_state.capitalization_enabled,
                help="Apply intelligent capitalization for proper nouns and sentence boundaries"
            )
            st.session_state.capitalization_enabled = capitalization_enabled
            
            # Show preset description
            preset_descriptions = {
                'meeting_light': 'Light punctuation with minimal disfluency removal. Preserves natural speech patterns while improving readability.',
                'meeting_moderate': 'Balanced punctuation with moderate disfluency cleaning. Good for most business meetings.',
                'meeting_aggressive': 'Maximum punctuation polish with aggressive disfluency removal. Creates highly polished transcripts.',
                'general_light': 'Basic punctuation for general content without meeting-specific optimizations.'
            }
            st.info(f"**{punctuation_presets[selected_preset].split(' ', 1)[1]}**: {preset_descriptions[selected_preset]}")
            
            # Disfluency level descriptions
            disfluency_descriptions = {
                'light': 'Removes 3+ consecutive fillers (um um um), obvious false starts, and multiple pause markers.',
                'moderate': 'Removes 2+ consecutive fillers, word repetitions, and context-dependent single fillers.',
                'aggressive': 'Removes all fillers, discourse markers (basically, actually), and partial words for maximum polish.'
            }
            st.info(f"**Normalization**: {disfluency_descriptions[selected_disfluency]}")
        else:
            st.info("💡 Enable punctuation processing to improve transcript readability and accuracy")
    
    # Source Separation Configuration
    st.sidebar.markdown("---")
    with st.sidebar.expander("🎵 Source Separation Configuration", expanded=False):
        st.markdown("**Configure overlap detection and audio source separation:**")
        
        # Enable source separation
        source_separation_enabled = st.checkbox(
            "Enable Source Separation",
            value=st.session_state.source_separation_enabled,
            help="Use Demucs neural source separation for overlapping speech scenarios"
        )
        st.session_state.source_separation_enabled = source_separation_enabled
        
        if source_separation_enabled:
            # Overlap probability threshold
            overlap_threshold = st.slider(
                "Overlap Detection Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.overlap_probability_threshold,
                step=0.05,
                help="Probability threshold for detecting overlapping speech segments (0.25 = conservative, 0.75 = aggressive)"
            )
            st.session_state.overlap_probability_threshold = overlap_threshold
            
            # Demucs model selection
            demucs_models = {
                'htdemucs': '🎯 HTDemucs (Fast, Good Quality)',
                'htdemucs_6s': '🎯 HTDemucs 6-Source (Default)',
                'htdemucs_ft': '🎯 HTDemucs Fine-tuned (High Quality)', 
                'mdx_extra': '📊 MDX Extra (Experimental)'
            }
            
            selected_demucs = st.selectbox(
                "Demucs Model",
                options=list(demucs_models.keys()),
                format_func=lambda x: demucs_models[x],
                index=list(demucs_models.keys()).index(st.session_state.demucs_model_name) if st.session_state.demucs_model_name in demucs_models else 1,
                help="Choose Demucs model variant for source separation quality vs speed tradeoff"
            )
            st.session_state.demucs_model_name = selected_demucs
            
            # Source separation providers
            provider_options = {
                'demucs': '🎵 Demucs Neural Network',
                'vocal_isolation': '🗣️ Vocal Isolation',
                'mixed_approach': '🔀 Mixed Approach',
                'spectral_subtraction': '📊 Spectral Subtraction'
            }
            
            selected_providers = st.multiselect(
                "Source Separation Providers",
                options=list(provider_options.keys()),
                default=st.session_state.source_separation_providers,
                format_func=lambda x: provider_options[x],
                help="Select multiple providers for ensemble source separation"
            )
            st.session_state.source_separation_providers = selected_providers
            
            # Show configuration summary
            st.info(f"**Active**: {demucs_models[selected_demucs].split(' ', 1)[1]} with {len(selected_providers)} providers at {overlap_threshold:.2f} threshold")
        else:
            st.warning("⚠️ Source separation disabled - overlapping speech may reduce accuracy")
    
    # Speaker Mapping Configuration
    st.sidebar.markdown("---")  
    with st.sidebar.expander("👥 Speaker Mapping Configuration", expanded=False):
        st.markdown("**Configure speaker identity robustness and tracking:**")
        
        # Enable speaker mapping
        speaker_mapping_enabled = st.checkbox(
            "Enable Speaker Mapping",
            value=st.session_state.speaker_mapping_enabled,
            help="Use advanced speaker identity tracking with ECAPA-TDNN embeddings"
        )
        st.session_state.speaker_mapping_enabled = speaker_mapping_enabled
        
        if speaker_mapping_enabled:
            # ECAPA-TDNN settings
            ecapa_enabled = st.checkbox(
                "Enable ECAPA-TDNN Embeddings",
                value=st.session_state.enable_ecapa_tdnn,
                help="Use state-of-the-art ECAPA-TDNN neural network for speaker embeddings"
            )
            st.session_state.enable_ecapa_tdnn = ecapa_enabled
            
            if ecapa_enabled:
                # ECAPA embedding dimension
                embedding_dims = {
                    192: '192-dim (Fast, Good Quality)',
                    512: '512-dim (Standard, High Quality)',
                    1024: '1024-dim (Slow, Highest Quality)'
                }
                
                selected_dim = st.selectbox(
                    "ECAPA Embedding Dimension",
                    options=list(embedding_dims.keys()),
                    format_func=lambda x: embedding_dims[x],
                    index=list(embedding_dims.keys()).index(st.session_state.ecapa_embedding_dim) if st.session_state.ecapa_embedding_dim in embedding_dims else 0,
                    help="Higher dimensions provide better speaker discrimination but slower processing"
                )
                st.session_state.ecapa_embedding_dim = selected_dim
            
            # Speaker backtracking
            backtracking_enabled = st.checkbox(
                "Enable Speaker Backtracking", 
                value=st.session_state.enable_speaker_backtracking,
                help="Retroactively correct speaker assignments based on voice consistency patterns"
            )
            st.session_state.enable_speaker_backtracking = backtracking_enabled
            
            if backtracking_enabled:
                # Backtracking segments
                backtrack_segments = st.slider(
                    "Backtracking Segment Window",
                    min_value=3,
                    max_value=15,
                    value=st.session_state.backtracking_segments,
                    step=1,
                    help="Number of segments to analyze for speaker consistency correction"
                )
                st.session_state.backtracking_segments = backtrack_segments
                
                # Speaker consistency threshold
                consistency_threshold = st.slider(
                    "Speaker Consistency Threshold",
                    min_value=0.5,
                    max_value=0.95,
                    value=st.session_state.speaker_consistency_threshold,
                    step=0.05,
                    help="Minimum voice consistency required to maintain speaker assignment"
                )
                st.session_state.speaker_consistency_threshold = consistency_threshold
            
            # Show speaker mapping status
            status_components = []
            if ecapa_enabled:
                status_components.append(f"ECAPA-TDNN ({st.session_state.ecapa_embedding_dim}D)")
            if backtracking_enabled:
                status_components.append(f"Backtracking ({st.session_state.backtracking_segments} segments)")
            
            if status_components:
                st.success(f"**Active**: {', '.join(status_components)}")
            else:
                st.info("Basic speaker mapping without advanced features")
        else:
            st.warning("⚠️ Speaker mapping disabled - speaker boundaries may be less accurate")
    
    # Dialect Handling Configuration
    st.sidebar.markdown("---")
    with st.sidebar.expander("🌍 Dialect Handling Configuration", expanded=False):
        st.markdown("**Configure dialect-aware processing and phonetic agreement:**")
        
        # Enable dialect handling
        dialect_handling_enabled = st.checkbox(
            "Enable Dialect Handling",
            value=st.session_state.dialect_handling_enabled,
            help="Use dialect-specific processing with CMUdict and G2P phonetic agreement"
        )
        st.session_state.dialect_handling_enabled = dialect_handling_enabled
        
        if dialect_handling_enabled:
            # Primary dialect selection
            dialect_options = {
                'general_american': '🇺🇸 General American (Default)',
                'southern_us': '🏛️ Southern US',
                'northeast_us': '🏙️ Northeast US', 
                'midwest_us': '🌾 Midwest US',
                'canadian_english': '🇨🇦 Canadian English',
                'general_british': '🇬🇧 General British',
                'australian_english': '🇦🇺 Australian English'
            }
            
            selected_dialect = st.selectbox(
                "Primary Dialect",
                options=list(dialect_options.keys()),
                format_func=lambda x: dialect_options[x],
                index=list(dialect_options.keys()).index(st.session_state.primary_dialect) if st.session_state.primary_dialect in dialect_options else 0,
                help="Select the primary dialect for phonetic processing and confidence scoring"
            )
            st.session_state.primary_dialect = selected_dialect
            
            # Supported dialects (multi-select)
            supported_dialects = st.multiselect(
                "Supported Dialects",
                options=list(dialect_options.keys()),
                default=st.session_state.supported_dialects,
                format_func=lambda x: dialect_options[x],
                help="Select all dialects to support for multi-dialect processing"
            )
            st.session_state.supported_dialects = supported_dialects
            
            # CMUdict + G2P settings
            cmudict_g2p_enabled = st.checkbox(
                "Enable CMUdict + G2P Phonetic Agreement",
                value=st.session_state.cmudict_g2p_enabled,
                help="Use CMUdict pronunciation dictionary with G2P fallback for phonetic validation"
            )
            st.session_state.cmudict_g2p_enabled = cmudict_g2p_enabled
            
            if cmudict_g2p_enabled:
                # Phonetic agreement threshold
                phonetic_threshold = st.slider(
                    "Phonetic Agreement Threshold", 
                    min_value=0.5,
                    max_value=0.95,
                    value=st.session_state.phonetic_agreement_threshold,
                    step=0.05,
                    help="Minimum phonetic similarity required for dialect confidence boost"
                )
                st.session_state.phonetic_agreement_threshold = phonetic_threshold
                
                # Dialect confidence boost
                confidence_boost = st.slider(
                    "Dialect Confidence Boost",
                    min_value=0.0,
                    max_value=0.3,
                    value=st.session_state.dialect_confidence_boost,
                    step=0.01,
                    help="Confidence boost applied when phonetic agreement exceeds threshold"
                )
                st.session_state.dialect_confidence_boost = confidence_boost
            
            # Show dialect configuration summary
            dialect_summary = f"Primary: {dialect_options[selected_dialect].split(' ', 1)[1]}"
            if len(supported_dialects) > 1:
                dialect_summary += f" + {len(supported_dialects)-1} others"
            if cmudict_g2p_enabled:
                dialect_summary += f", G2P@{phonetic_threshold:.2f}"
            
            st.success(f"**Active**: {dialect_summary}")
        else:
            st.warning("⚠️ Dialect handling disabled - may reduce accuracy for non-standard pronunciations")
    
    # A/B Testing Configuration
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧪 A/B Testing Configuration", expanded=False):
        st.markdown("**Enable methodology comparison:**")
        
        enable_ab = st.checkbox(
            "Enable A/B Testing",
            value=st.session_state.enable_ab_testing,
            help="Compare multiple consensus and calibration methods simultaneously"
        )
        st.session_state.enable_ab_testing = enable_ab
        
        if enable_ab:
            st.markdown("**Consensus Methods to Compare:**")
            consensus_compare = st.multiselect(
                "Select consensus strategies",
                options=list(consensus_options.keys()),
                default=st.session_state.ab_comparison_methods['consensus'],
                format_func=lambda x: consensus_options[x]
            )
            st.session_state.ab_comparison_methods['consensus'] = consensus_compare
            
            st.markdown("**Calibration Methods to Compare:**")
            calibration_compare = st.multiselect(
                "Select calibration methods",
                options=list(calibration_options.keys()),
                default=st.session_state.ab_comparison_methods['calibration'],
                format_func=lambda x: calibration_options[x]
            )
            st.session_state.ab_comparison_methods['calibration'] = calibration_compare
            
            if consensus_compare and calibration_compare:
                total_combinations = len(consensus_compare) * len(calibration_compare)
                st.success(f"Will test {total_combinations} method combinations")
            else:
                st.warning("Select at least one method from each category for A/B testing")
    
    # U7 System Configuration  
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚡ U7 System Configuration", expanded=False):
        st.markdown("**Advanced system controls:**")
        
        # Caching controls
        enable_caching = st.checkbox(
            "Enable Intelligent Caching",
            value=st.session_state.u7_enable_caching,
            help="Cache expensive operations for significant performance improvement"
        )
        st.session_state.u7_enable_caching = enable_caching
        
        # Deterministic processing
        enable_deterministic = st.checkbox(
            "Enable Deterministic Processing",
            value=st.session_state.u7_enable_deterministic_processing,
            help="Ensure identical results for same inputs using fixed seeds"
        )
        st.session_state.u7_enable_deterministic_processing = enable_deterministic
        
        # Selective reprocessing
        enable_selective = st.checkbox(
            "Enable Selective Reprocessing",
            value=st.session_state.u7_enable_selective_reprocessing,
            help="Automatically reprocess low-confidence segments for quality improvement"
        )
        st.session_state.u7_enable_selective_reprocessing = enable_selective
        
        if enable_selective:
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence Threshold for Flagging",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.u7_confidence_threshold,
                step=0.01,
                help="Segments below this confidence will be flagged for reprocessing"
            )
            st.session_state.u7_confidence_threshold = confidence_threshold
            
            # Max segments for reprocessing
            max_segments = st.number_input(
                "Max Segments for Reprocessing",
                min_value=1,
                max_value=50,
                value=st.session_state.u7_max_segments_for_reprocessing,
                help="Maximum number of flagged segments to reprocess automatically"
            )
            st.session_state.u7_max_segments_for_reprocessing = max_segments
        
        if st.button("🔄 Reset U7 to Defaults"):
            st.session_state.u7_enable_caching = True
            st.session_state.u7_enable_selective_reprocessing = True
            st.session_state.u7_confidence_threshold = 0.65
            st.session_state.u7_max_segments_for_reprocessing = 10
            st.session_state.u7_enable_deterministic_processing = True
            st.rerun()
    
    # Results status in sidebar
    if st.session_state.results:
        st.sidebar.success("✅ Results Available")
        winner_score = st.session_state.results.get('winner_score', 0)
        st.sidebar.metric("Winner Score", f"{winner_score:.2f}")
        
        processing_time = st.session_state.results.get('processing_time', 0)
        st.sidebar.metric("Processing Time", f"{processing_time:.1f}s")
    else:
        st.sidebar.info("ℹ️ No results yet")
    
    # Route to appropriate page
    if selected_page == 'main':
        render_main_page()
    elif selected_page == 'qc':
        render_qc_dashboard()
    elif selected_page == 'u7':
        render_u7_system_management()

def render_main_page():
    st.title("🎯 Advanced Ensemble Transcription System")
    st.markdown("Generate 15 candidate transcripts with multi-dimensional confidence scoring")

    # Check for required API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    # Check FFmpeg availability
    audio_processor = AudioProcessor()
    ffmpeg_available, ffmpeg_info = audio_processor.check_ffmpeg_availability()
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

    # Google Drive Upload Section
    drive_uploader = get_drive_uploader()
    upload_result = drive_uploader.render_upload_interface(
        accept_types=['.mp4', '.avi', '.mov', '.mkv', '.webm'],
        max_size_mb=200.0
    )
    
    # Show recent uploads for quick access
    recent_file_result = drive_uploader.show_recent_uploads(max_files=5)
    if recent_file_result:
        upload_result = recent_file_result
    
    if upload_result and upload_result['status'] == 'success':
        st.session_state.drive_file_info = upload_result
        
        # Show file info
        file_size_mb = upload_result['size'] / (1024 * 1024)
        st.success(f"✅ **{upload_result['filename']}** ready for processing ({file_size_mb:.1f}MB)")
        
        # Show file source info
        if upload_result['source'] == 'uploaded_to_drive':
            st.info("📤 File uploaded to Google Drive")
        else:
            st.info("🔗 Using existing Google Drive file")
    
    if st.session_state.drive_file_info:
        
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
        if st.button("🚀 Start Ensemble Processing", disabled=st.session_state.processing, key="start_processing"):
            st.session_state.processing = True
            try:
                process_video_from_drive(st.session_state.drive_file_info, expected_speakers, noise_level, st.session_state.scoring_weights)
            except Exception as e:
                st.session_state.processing = False
                st.error(f"Processing error: {e}")
                st.code(traceback.format_exc())

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


def process_video_from_path(video_file_path, expected_speakers, noise_level, scoring_weights):
    """Process the video file from local path through the ensemble pipeline"""
    return process_video_from_local_path(video_file_path, expected_speakers, noise_level, scoring_weights)

def process_video_from_local_path(video_file_path, expected_speakers, noise_level, scoring_weights, progress_bar=None, status_text=None, start_progress=0):
    """Process the video/audio file from local path through the ensemble pipeline
    
    Note: video_file_path can now be either a video file (for direct processing)
    or an ASR-ready WAV file (when called from the new audio processing pipeline)
    """
    if not progress_bar:
        progress_bar = st.progress(0)
    if not status_text:
        status_text = st.empty()
    
    if not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.results = None
    
    try:
        # File is available locally, use the path directly
        tmp_path = video_file_path
        
        status_text.text("⚙️ Initializing ensemble manager...")
        progress_bar.progress(start_progress + 10)
        
        # Initialize ensemble manager with versioning enabled, selected methods, and U7 settings
        ensemble_manager = EnsembleManager(
            expected_speakers=expected_speakers,
            noise_level=noise_level.lower(),
            target_language="en",  # English only
            scoring_weights=scoring_weights,
            enable_versioning=True,
            consensus_strategy=st.session_state.consensus_strategy,
            calibration_method=st.session_state.calibration_method,
            domain="meeting",  # Default domain for UI uploads
            # Supported parameters only
            enable_speaker_mapping=st.session_state.speaker_mapping_enabled,
            enable_dialect_handling=st.session_state.dialect_handling_enabled,
            supported_dialects=st.session_state.supported_dialects,
            dialect_confidence_boost=st.session_state.dialect_confidence_boost
        )
        
        # Configure source separation settings
        if hasattr(ensemble_manager, 'source_separation_engine') and ensemble_manager.source_separation_engine:
            ensemble_manager.enable_source_separation = st.session_state.source_separation_enabled
            ensemble_manager.overlap_probability_threshold = st.session_state.overlap_probability_threshold
            if hasattr(ensemble_manager.source_separation_engine, 'configure'):
                source_sep_config = {
                    'providers': st.session_state.source_separation_providers,
                    'demucs_model_name': st.session_state.demucs_model_name,
                    'overlap_threshold': st.session_state.overlap_probability_threshold
                }
                ensemble_manager.source_separation_engine.configure(source_sep_config)
        
        # Configure detailed speaker mapping settings
        if hasattr(ensemble_manager, 'speaker_mapper') and ensemble_manager.speaker_mapper:
            detailed_speaker_config = {
                'enable_ecapa_tdnn': st.session_state.enable_ecapa_tdnn,
                'ecapa_embedding_dim': st.session_state.ecapa_embedding_dim,
                'enable_speaker_backtracking': st.session_state.enable_speaker_backtracking,
                'backtracking_segments': st.session_state.backtracking_segments,
                'speaker_consistency_threshold': st.session_state.speaker_consistency_threshold
            }
            if hasattr(ensemble_manager.speaker_mapper, 'update_config'):
                ensemble_manager.speaker_mapper.update_config(detailed_speaker_config)
            else:
                # Apply settings to speaker_mapping_config
                ensemble_manager.speaker_mapping_config.update(detailed_speaker_config)
        
        # Configure detailed dialect handling settings
        if hasattr(ensemble_manager, 'dialect_handling_engine') and ensemble_manager.dialect_handling_engine:
            dialect_detailed_config = {
                'primary_dialect': st.session_state.primary_dialect,
                'cmudict_g2p_enabled': st.session_state.cmudict_g2p_enabled,
                'phonetic_agreement_threshold': st.session_state.phonetic_agreement_threshold
            }
            if hasattr(ensemble_manager.dialect_handling_engine, 'configure'):
                ensemble_manager.dialect_handling_engine.configure(dialect_detailed_config)
        
        # Configure post-fusion punctuation settings
        if hasattr(ensemble_manager, 'enable_post_fusion_punctuation'):
            ensemble_manager.enable_post_fusion_punctuation = st.session_state.punctuation_enabled
            if st.session_state.punctuation_enabled:
                ensemble_manager.punctuation_preset = st.session_state.punctuation_preset
                
                # Apply custom configuration if needed
                if ensemble_manager.punctuation_engine:
                    custom_config = {
                        'disfluency_level': st.session_state.disfluency_level,
                        'enable_meeting_vocabulary': st.session_state.meeting_vocabulary_enabled,
                        'enable_capitalization': st.session_state.capitalization_enabled
                    }
                    ensemble_manager.punctuation_engine.update_configuration(custom_config)
        
        # U7 Upgrade: Configure U7 settings from session state
        ensemble_manager.enable_caching = st.session_state.u7_enable_caching
        ensemble_manager.enable_selective_reprocessing = st.session_state.u7_enable_selective_reprocessing
        ensemble_manager.confidence_threshold_for_flagging = st.session_state.u7_confidence_threshold
        ensemble_manager.max_segments_for_selective_reprocessing = st.session_state.u7_max_segments_for_reprocessing
        
        # Process through ensemble pipeline
        def update_progress(step, progress, message):
            # Scale progress to remaining portion (depends on start_progress)
            remaining_progress = 100 - start_progress - 10  # Leave 10% for finalization
            scaled_progress = start_progress + 10 + int((progress / 100) * remaining_progress)
            progress_bar.progress(min(scaled_progress, 90))
            status_text.text(f"Step {step}: {message}")
        
        status_text.text("🚀 Starting ensemble processing...")
        progress_bar.progress(start_progress + 10)
        
        results = ensemble_manager.process_video(video_file_path, update_progress)
        
        
        progress_bar.progress(95)
        status_text.text("🧹 Cleaning up temporary files...")
        
        # Clean up temporary file
        
        # Store results
        st.session_state.results = results
        st.session_state.processing = False
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.success(f"🎉 Ensemble processing completed successfully!")
        st.success(f"🏆 Winner transcript selected with confidence: {results.get('winner_confidence', 0):.2f}")
        
        
        st.rerun()
        
    except Exception as e:
        st.session_state.processing = False
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Processing failed: {str(e)}")
        st.error("📋 Error details:")
        st.code(traceback.format_exc())

def process_video_from_drive(drive_file_info, expected_speakers, noise_level, scoring_weights):
    """Process video file from Google Drive by downloading it first"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.results = None
    
    tmp_path = None
    try:
        # Step 1: Download file from Google Drive (0-20%)
        status_text.text("📥 Downloading file from Google Drive...")
        progress_bar.progress(5)
        
        # Create temporary file for download
        file_extension = os.path.splitext(drive_file_info['filename'])[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_path = tmp_file.name
        
        # Download with progress tracking
        def download_progress_callback(downloaded_bytes: int, total_bytes: int):
            if total_bytes > 0:
                download_progress = int((downloaded_bytes / total_bytes) * 15)  # 0-15%
                progress_bar.progress(5 + download_progress)
                status_text.text(
                    f"📥 Downloading: {downloaded_bytes:,} / {total_bytes:,} bytes "
                    f"({downloaded_bytes/total_bytes:.1%})"
                )
        
        # Download the file
        success = download_file_from_drive(
            drive_file_info['file_id'],
            tmp_path,
            show_progress=False  # We handle progress manually
        )
        
        if not success:
            raise Exception("Failed to download file from Google Drive")
        
        progress_bar.progress(20)
        status_text.text("✅ Download completed, starting audio processing...")
        
        # Step 2: Extract pristine audio and create ASR-ready WAV (20-40%)
        status_text.text("🎵 Creating pristine audio copy...")
        progress_bar.progress(25)
        
        audio_processor = AudioProcessor()
        
        try:
            # Create pristine audio copy using stream copy
            pristine_audio_path = audio_processor.copy_audio_stream(tmp_path)
            st.session_state.pristine_audio_path = pristine_audio_path
            
            progress_bar.progress(30)
            status_text.text("🎵 Creating ASR-ready WAV...")
            
            # Create ASR-ready WAV from pristine copy
            asr_wav_path = audio_processor.make_asr_wav_from_audio(pristine_audio_path)
            st.session_state.asr_wav_path = asr_wav_path
            st.session_state.audio_processing_complete = True
            
            progress_bar.progress(40)
            status_text.text("✅ Audio processing completed, starting ensemble processing...")
            
            # Log successful audio processing
            st.success(f"🎵 Audio processing completed successfully!")
            st.info(f"📁 Pristine audio: {os.path.basename(pristine_audio_path)}")
            st.info(f"🎯 ASR-ready WAV: {os.path.basename(asr_wav_path)}")
            
        except Exception as audio_error:
            st.session_state.processing = False
            raise Exception(f"Audio processing failed: {str(audio_error)}")
        
        # Step 3: Process through ensemble pipeline using ASR-ready WAV (40-100%)
        try:
            process_video_from_local_path(
                asr_wav_path,  # Use ASR-ready WAV instead of original video
                expected_speakers, 
                noise_level, 
                scoring_weights,
                progress_bar,
                status_text,
                start_progress=40
            )
        finally:
            # Clean up downloaded temporary file (but keep audio files for session)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as cleanup_error:
                    st.warning(f"Could not clean up temporary file: {cleanup_error}")
            
            # Note: Audio files (pristine_audio_path and asr_wav_path) are kept
            # in session state for potential reuse and will be cleaned up when
            # session ends or explicitly cleared
        
    except Exception as e:
        st.session_state.processing = False
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Processing failed: {str(e)}")
        st.error("📋 Error details:")
        st.code(traceback.format_exc())
        
        # Clean up temporary file if it exists
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        # Clear audio processing state on error
        st.session_state.pristine_audio_path = None
        st.session_state.asr_wav_path = None
        st.session_state.audio_processing_complete = False

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

def display_cost_performance_metrics(results):
    """Display real-time cost and performance metrics from observability system"""
    if not results or 'cost_summary' not in results:
        return
    
    st.header("💰 Cost & Performance Analytics")
    
    cost_summary = results.get('cost_summary', {})
    system_metrics = results.get('system_metrics', {})
    obs_metadata = results.get('observability_metadata', {})
    
    # Cost metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cost = cost_summary.get('total_cost_usd', 0.0)
        st.metric(
            "💰 Total Cost",
            f"${total_cost:.4f}",
            help="Total API costs for this processing session"
        )
    
    with col2:
        api_calls = cost_summary.get('total_api_calls', 0)
        st.metric(
            "📞 API Calls",
            str(api_calls),
            help="Total number of API calls made"
        )
    
    with col3:
        memory_peak = system_metrics.get('memory_rss_mb', 0)
        st.metric(
            "🧠 Peak Memory",
            f"{memory_peak:.1f} MB",
            help="Peak memory usage during processing"
        )
    
    with col4:
        session_duration = system_metrics.get('session_duration', 0)
        st.metric(
            "⏱️ Session Duration",
            f"{session_duration:.1f}s",
            help="Total session duration including overhead"
        )
    
    # Cost breakdown
    cost_breakdown = cost_summary.get('cost_breakdown', {})
    if cost_breakdown:
        st.subheader("📊 Cost Breakdown by Service")
        
        breakdown_data = []
        for service, data in cost_breakdown.items():
            breakdown_data.append({
                'Service': service.replace('_', ' ').title(),
                'Cost (USD)': f"${data['cost']:.4f}",
                'Calls': data['calls'],
                'Avg Cost/Call': f"${data['avg_cost_per_call']:.4f}",
                'Duration (s)': f"{data['total_duration']:.1f}"
            })
        
        if breakdown_data:
            import pandas as pd
            df = pd.DataFrame(breakdown_data)
            st.dataframe(df, use_container_width=True)
    
    # Performance insights
    with st.expander("🔍 Performance Insights", expanded=False):
        st.markdown("**Observability Metadata:**")
        if obs_metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"• **Run ID:** {obs_metadata.get('run_id', 'N/A')}")
                st.write(f"• **Session ID:** {obs_metadata.get('session_id', 'N/A')}")
                st.write(f"• **Instrumentation:** {'✅ Enabled' if obs_metadata.get('instrumentation_enabled', False) else '❌ Disabled'}")
                st.write(f"• **Cost Tracking:** {'✅ Enabled' if obs_metadata.get('cost_tracking_enabled', False) else '❌ Disabled'}")
            
            with col2:
                st.write(f"• **Profiling:** {'✅ Enabled' if obs_metadata.get('profiling_enabled', False) else '❌ Disabled'}")
                st.write(f"• **Pipeline Stages:** {len(obs_metadata.get('pipeline_stages', []))}")
                stages = obs_metadata.get('pipeline_stages', [])
                if stages:
                    st.write(f"• **Stages:** {', '.join(stages)}")
        
        # System performance summary
        st.markdown("**System Performance:**")
        if system_metrics:
            st.write(f"• **CPU Usage:** {system_metrics.get('cpu_percent', 0):.1f}%")
            st.write(f"• **Memory VMS:** {system_metrics.get('memory_vms_mb', 0):.1f} MB")
            st.write(f"• **Timestamp:** {system_metrics.get('timestamp', 'N/A')}")

def display_results():
    """Display processing results and download options"""
    results = st.session_state.results
    
    st.header("🏆 Ensemble Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Winner Score",
            f"{results['winner_score']:.2f}",
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
            st.write(f"- **{component}**: {score:.2f}")
    
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
            
            st.write(f"{rank}. **Score: {score:.2f}** - {variant}")

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
        st.session_state.drive_file_info = None
        st.session_state.processing = False
        st.rerun()

def render_u7_system_management():
    """Render U7 System Management page with cache management, worklist review, and manual flagging"""
    st.title("⚡ U7 System Management")
    st.markdown("Advanced system controls for caching, worklist management, and targeted reprocessing")
    
    # Initialize U7 managers
    cache_manager = get_cache_manager()
    worklist_manager = get_worklist_manager()
    selective_asr_processor = get_selective_asr_processor()
    
    # Create tabs for different U7 features
    tab1, tab2, tab3, tab4 = st.tabs(["🗄️ Cache Management", "📋 Worklist Review", "🎯 Manual Flagging", "📊 System Status"])
    
    with tab1:
        st.header("🗄️ Intelligent Cache Management")
        
        # Cache statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cache Statistics")
            try:
                stats = cache_manager.get_statistics()
                
                st.metric("Memory Cache Hits", stats.get('memory_hits', 0))
                st.metric("Disk Cache Hits", stats.get('disk_hits', 0))
                st.metric("Cache Misses", stats.get('misses', 0))
                st.metric("Cache Sets", stats.get('cache_sets', 0))
                
                # Calculate hit rate
                total_requests = stats.get('memory_hits', 0) + stats.get('disk_hits', 0) + stats.get('misses', 0)
                if total_requests > 0:
                    hit_rate = (stats.get('memory_hits', 0) + stats.get('disk_hits', 0)) / total_requests * 100
                    st.metric("Hit Rate", f"{hit_rate:.1f}%")
                
            except Exception as e:
                st.error(f"Error retrieving cache statistics: {e}")
        
        with col2:
            st.subheader("Cache Management")
            
            if st.button("🧹 Clear All Cache", type="secondary"):
                try:
                    cache_manager.clear_all()
                    st.success("All cache cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")
            
            if st.button("📊 Refresh Cache Stats", type="primary"):
                st.rerun()
            
            # Cache size information
            try:
                cache_info = cache_manager.get_cache_info()
                st.info(f"**Cache Directory:** {cache_info.get('cache_dir', 'N/A')}")
                st.info(f"**Max Memory Cache:** {cache_info.get('max_memory_mb', 'N/A')} MB")
                st.info(f"**Max Disk Cache:** {cache_info.get('max_disk_gb', 'N/A')} GB")
            except Exception as e:
                st.warning(f"Could not retrieve cache info: {e}")
    
    with tab2:
        st.header("📋 Segment Worklist Review")
        
        # Display existing worklists
        try:
            available_worklists = worklist_manager.list_available_worklists()
            
            if available_worklists:
                st.subheader("Available Worklists")
                
                selected_worklist = st.selectbox(
                    "Select a worklist to review:",
                    options=available_worklists,
                    format_func=lambda x: f"{x['file_name']} - {x['total_segments_flagged']} flagged segments"
                )
                
                if selected_worklist:
                    # Load and display worklist details
                    worklist_data = worklist_manager.load_worklist(selected_worklist['file_path'], selected_worklist.get('run_id', 'default_run'))
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Segments", len(worklist_data.flagged_segments) if worklist_data else 0)
                    with col2:
                        st.metric("Flagged Segments", worklist_data.total_segments_flagged if worklist_data else 0)
                    with col3:
                        st.metric("Average Confidence", f"{getattr(worklist_data, 'average_confidence', 0):.2f}" if worklist_data else "0.00")
                    
                    # Display flagged segments
                    flagged_segments = worklist_data.flagged_segments if worklist_data else []
                    
                    if flagged_segments:
                        st.subheader("Flagged Segments")
                        
                        for i, segment in enumerate(flagged_segments):
                            with st.expander(f"Segment {i+1}: {segment.start_time:.1f}s - {segment.end_time:.1f}s (Confidence: {segment.current_confidence:.2f})"):
                                st.text(f"Text: {getattr(segment, 'original_transcript', 'N/A')}")
                                st.text(f"Reason: {segment.flag_reason}")
                                st.text(f"Priority: {segment.processing_priority}")
                                
                                # Option to unflag segment
                                if st.button(f"🔓 Unflag Segment {i+1}", key=f"unflag_{i}"):
                                    try:
                                        worklist_manager.unflag_segment(selected_worklist['file_path'], selected_worklist.get('run_id', 'default_run'), segment.segment_id)
                                        st.success("Segment unflagged successfully!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error unflagging segment: {e}")
                    else:
                        st.info("No flagged segments in this worklist")
            else:
                st.info("No worklists available. Process a video first to generate worklists.")
                
        except Exception as e:
            st.error(f"Error loading worklists: {e}")
    
    with tab3:
        st.header("🎯 Manual Segment Flagging")
        
        # Check if we have results to work with
        if 'results' not in st.session_state or not st.session_state.results:
            st.warning("⚠️ No transcript results available. Please process a video first from the main page.")
        else:
            results = st.session_state.results
            master_transcript = results['winner_transcript']
            segments = master_transcript.get('segments', [])
            
            st.subheader("Select Segments to Flag for Reprocessing")
            
            # Filter segments by confidence
            confidence_filter = st.slider(
                "Show segments with confidence below:",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Display segments below this confidence threshold"
            )
            
            filtered_segments = [seg for seg in segments if seg.get('confidence', 1.0) <= confidence_filter]
            
            if filtered_segments:
                st.info(f"Found {len(filtered_segments)} segments below confidence threshold")
                
                # Display segments for manual flagging
                for i, segment in enumerate(filtered_segments):
                    with st.expander(f"Segment {i+1}: {segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s (Confidence: {segment.get('confidence', 0):.2f})"):
                        st.text(f"Speaker: {segment.get('speaker', 'Unknown')}")
                        st.text_area(f"Text:", value=segment.get('text', ''), key=f"text_{i}", disabled=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            flag_reason = st.selectbox(
                                "Flagging Reason:",
                                options=["Low Confidence", "Transcription Error", "Speaker Mismatch", "Audio Quality", "Custom"],
                                key=f"reason_{i}"
                            )
                        
                        with col2:
                            if st.button(f"🚩 Flag for Reprocessing", key=f"flag_{i}", type="secondary"):
                                try:
                                    # Add segment to worklist manually
                                    worklist_manager.flag_segment_manually(
                                        file_path=st.session_state.get('uploaded_file_path', 'unknown'),
                                        run_id=results.get('run_id', 'default_run'),
                                        start_time=segment.get('start', 0),
                                        end_time=segment.get('end', 0),
                                        reason=flag_reason
                                    )
                                    st.success(f"Segment {i+1} flagged successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error flagging segment: {e}")
            else:
                st.info("No segments found below the confidence threshold")
    
    with tab4:
        st.header("📊 U7 System Status")
        
        # Display U7 system configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Configuration")
            
            config_data = {
                "Intelligent Caching": "✅ Enabled" if st.session_state.u7_enable_caching else "❌ Disabled",
                "Deterministic Processing": "✅ Enabled" if st.session_state.u7_enable_deterministic_processing else "❌ Disabled", 
                "Selective Reprocessing": "✅ Enabled" if st.session_state.u7_enable_selective_reprocessing else "❌ Disabled",
                "Confidence Threshold": f"{st.session_state.u7_confidence_threshold:.2f}",
                "Max Segments for Reprocessing": str(st.session_state.u7_max_segments_for_reprocessing)
            }
            
            for key, value in config_data.items():
                st.text(f"**{key}:** {value}")
        
        with col2:
            st.subheader("System Performance")
            
            # Try to get performance metrics
            try:
                # Display worklist statistics
                worklist_stats = worklist_manager.get_statistics()
                
                st.metric("Total Files Processed", worklist_stats.get('total_files_processed', 0))
                st.metric("Total Segments Flagged", worklist_stats.get('total_segments_flagged', 0))
                st.metric("Total Segments Reprocessed", worklist_stats.get('total_segments_reprocessed', 0))
                st.metric("Total Segments Improved", worklist_stats.get('total_segments_improved', 0))
                
                avg_improvement = worklist_stats.get('average_improvement_score', 0)
                if avg_improvement > 0:
                    st.metric("Average Improvement Score", f"{avg_improvement:.3f}")
                
            except Exception as e:
                st.warning(f"Could not retrieve performance metrics: {e}")
        
        # System health check
        st.subheader("System Health Check")
        
        health_checks = []
        
        # Check cache manager
        try:
            cache_manager.get_statistics()
            health_checks.append(("Cache Manager", "✅ Healthy"))
        except Exception as e:
            health_checks.append(("Cache Manager", f"❌ Error: {e}"))
        
        # Check worklist manager
        try:
            worklist_manager.get_statistics()
            health_checks.append(("Worklist Manager", "✅ Healthy"))
        except Exception as e:
            health_checks.append(("Worklist Manager", f"❌ Error: {e}"))
        
        # Check selective ASR processor
        try:
            selective_asr_processor.get_status()
            health_checks.append(("Selective ASR Processor", "✅ Healthy"))
        except Exception as e:
            health_checks.append(("Selective ASR Processor", f"❌ Error: {e}"))
        
        for component, status in health_checks:
            st.text(f"**{component}:** {status}")

if __name__ == "__main__":
    main()
