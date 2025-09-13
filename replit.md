# replit.md

## Overview

The Advanced Ensemble Transcription System is a sophisticated video processing application that generates highly accurate transcripts from long-form MP4 recordings (up to 90 minutes) captured in noisy, multi-speaker environments. The system uses an ensemble approach to create 15 candidate transcripts by combining 3 speaker diarization variants with 5 ASR (Automatic Speech Recognition) variants each, then selects the best result using multi-dimensional confidence scoring.

The application is designed to handle recordings from meetings or discussions with approximately 10 participants captured with a single mono microphone in challenging acoustic conditions. It produces synchronized outputs including clean audio, speaker-attributed transcripts, and various caption formats.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Single-page application built with Streamlit for file upload and processing visualization
- **Real-time Progress Tracking**: Session state management for tracking processing stages and displaying results
- **File Upload Handling**: Support for MP4 video files up to 90 minutes with size validation and metadata display

### Backend Architecture
- **Modular Processing Pipeline**: Organized into distinct processing stages through the EnsembleManager orchestrator
- **Component-Based Design**: Separate engines for audio processing, diarization, ASR, and confidence scoring
- **Ensemble Processing Strategy**: 
  - 3 diarization variants with different clustering and VAD parameters
  - 5 ASR variants per diarization (15 total candidates)
  - Multi-dimensional confidence scoring across 5 metrics (D, A, L, R, O scores)
- **Concurrent Processing**: ThreadPoolExecutor for parallel ASR variant execution
- **Temporary File Management**: Session-based working directories with automatic cleanup

### Data Processing Components
- **Audio Processor**: Extracts mono 16kHz PCM audio from MP4, applies light denoising and normalization
- **Diarization Engine**: Uses pyannote.audio pipeline with variant configurations for speaker segmentation
- **ASR Engine**: OpenAI Whisper integration with configurable parameters for speech-to-text conversion
- **Confidence Scorer**: Multi-dimensional scoring system weighing diarization consistency, ASR quality, linguistic coherence, cross-run agreement, and overlap handling

### Output Generation
- **Transcript Formatter**: Generates multiple output formats (TXT, WebVTT, SRT, ASS)
- **Master JSON Structure**: Unified data format combining diarization, ASR results, speaker mapping, and confidence metrics
- **Quality Assurance**: Confidence-based winner selection and optional low-confidence span patching

## External Dependencies

### AI/ML Services
- **OpenAI API**: Whisper model for speech recognition ("whisper-1" model)
- **Hugging Face**: pyannote.audio speaker diarization pipeline ("pyannote/speaker-diarization-3.1")

### Audio/Video Processing
- **ffmpeg**: Video demuxing and audio extraction
- **librosa**: Audio loading, processing, and analysis
- **numpy**: Numerical computations for audio signal processing

### Machine Learning Libraries
- **torch**: PyTorch for neural network inference
- **scikit-learn**: TF-IDF vectorization and similarity metrics for linguistic scoring
- **pyannote.audio**: Speaker diarization and voice activity detection

### Web Framework
- **Streamlit**: Web application framework for user interface
- **concurrent.futures**: Threading for parallel processing

### Utility Libraries
- **pathlib**: File path management
- **tempfile**: Temporary file and directory creation
- **json**: Data serialization and storage

### Authentication Requirements
- **OPENAI_API_KEY**: Required environment variable for OpenAI Whisper access
- **HUGGINGFACE_TOKEN**: Required for accessing pyannote.audio models