# Advanced Ensemble Transcription System - Codebase Structure

## Project Overview
This codebase implements a sophisticated audio-video processing system that processes long MP4 recordings (90-112 minutes) and produces synchronized outputs including speaker-diarized transcripts, caption files, and subtitled videos. The system uses an advanced ensemble approach with 4 major enhancements for human-level transcription accuracy.

## Core Architecture

### Main Application
- `app.py` - Streamlit web interface and main application entry point
- `replit.md` - Project documentation and architecture overview

### Core Processing Components (`core/`)
**Primary Engines:**
- `ensemble_manager.py` - Main orchestrator coordinating all processing stages
- `asr_engine.py` - Automatic Speech Recognition engine with multiple provider support
- `diarization_engine.py` - Speaker diarization with multiple provider support
- `audio_processor.py` - Audio extraction and preprocessing

**Advanced Enhancement Modules:**
- `overlap_diarizer.py` - Overlap-aware diarization for multi-speaker scenarios
- `overlap_fusion.py` - Timeline fusion engine for overlapped speech recovery
- `source_separation_engine.py` - Audio source separation using advanced algorithms
- `term_miner.py` - Auto-glossary extraction from speech content
- `term_store.py` - Session-based terminology storage and management
- `term_bias.py` - Adaptive biasing engine for domain-specific terms
- `text_normalizer.py` - Robust text normalization with style profiles
- `guardrail_verifier.py` - Zero-hallucination verification system
- `global_speaker_linker.py` - Long-horizon speaker consistency tracking
- `speaker_relabeler.py` - Speaker identity management and swap correction

**Supporting Components:**
- `confidence_scorer.py` - Multi-dimensional confidence scoring system
- `consensus_module.py` - Advanced consensus strategies for transcript fusion
- `calibration_module.py` - Score calibration and normalization
- `speaker_mapper.py` - ECAPA-TDNN-based speaker embedding and mapping
- `fusion_engine.py` - Advanced transcript fusion algorithms

### Provider Interfaces
**ASR Providers (`core/asr_providers/`):**
- `openai_provider.py` - OpenAI Whisper integration
- `faster_whisper_provider.py` - Faster-Whisper local processing
- `deepgram_provider.py` - Deepgram cloud ASR service

**Diarization Providers (`core/diarization_providers/`):**
- `assemblyai.py` - AssemblyAI speaker diarization service

### Utilities (`utils/`)
**Core Utilities:**
- `file_handler.py` - File I/O operations and session management
- `transcript_formatter.py` - Multi-format output generation (TXT, VTT, SRT, ASS)
- `capability_manager.py` - Dependency management and graceful fallbacks
- `audio_format_validator.py` - Audio format validation and conversion

**Advanced Features:**
- `google_drive_handler.py` - Large file upload via Google Drive integration
- `embedding_cache.py` - Speaker embedding caching system
- `stem_manifest.py` - Audio stem tracking and management
- `enhanced_structured_logger.py` - Comprehensive logging infrastructure
- `metrics_registry.py` - Performance metrics collection and analysis
- `observability.py` - System observability and monitoring

### Configuration (`config/`)
**Service Configurations:**
- `config.yaml` - Main system configuration
- `asr/whisper_variants.yaml` - ASR model configurations
- `diarization/external.yaml` - Diarization service settings
- `normalization_profiles.yaml` - Text normalization style profiles

**Enhancement Configurations:**
- `auto_glossary/glossary_config.yaml` - Glossary extraction settings
- `consensus/strategies.yaml` - Consensus algorithm configurations
- `scoring/multi_dimensional.yaml` - Confidence scoring parameters

### Testing (`tests/`)
- `conftest.py` - Pytest configuration and fixtures
- `test_acceptance_metrics.py` - Acceptance test framework
- `gold_test_set/` - Curated test datasets with ground truth

### Data & Artifacts
- `data/` - Test audio files and samples  
- `artifacts/` - Processing outputs and session artifacts
- `calibration_models/` - Pre-trained calibration models
- `metrics_registry/` - Historical performance metrics

## Key Features

### 4 Major Enhancement Systems
1. **Overlap-aware Diarization & Source Separation** - Recovers overlapped speech through per-stem processing
2. **Auto-glossary Extraction & Adaptive Biasing** - Learns domain-specific terminology automatically  
3. **Robust Text Normalization with Guardrails** - Improves readability without content hallucination
4. **Long-horizon Speaker Tracking & Relabeling** - Maintains consistent speaker identities across sessions

### Performance Targets
- **Overlapped Speech WER**: 15-35% relative reduction
- **Overall WER**: 1.0-4.5% absolute improvement  
- **Entity Accuracy**: 5-15% improvement for domain terms
- **Speaker Attribution (DER)**: 10-40% relative reduction

### Output Formats
- JSON transcript with full metadata and confidence scores
- Plain text transcript with speaker labels and timestamps
- Video caption files (VTT, SRT, ASS) with speaker styling
- Comprehensive quality audit and processing reports

## Production Features
- Graceful degradation when optional ML dependencies unavailable
- Comprehensive error handling with circuit breakers
- Auto-format conversion for audio compatibility
- Enhanced observability with structured logging
- Session-based processing with automatic cleanup
- Google Drive integration for large file handling

## Dependencies
- **Core ML**: OpenAI API, AssemblyAI, HuggingFace models
- **Audio Processing**: ffmpeg, librosa, soundfile
- **Web Framework**: Streamlit for user interface
- **ML Libraries**: PyTorch, scikit-learn, numpy
- **Storage**: Google Drive API for large file handling

This codebase represents a production-ready, sophisticated transcription system capable of human-level accuracy through advanced AI techniques for overlap recovery, adaptive learning, text enhancement, and speaker consistency.