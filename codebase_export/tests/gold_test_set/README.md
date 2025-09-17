# Gold Test Set for Ensemble Transcription System

This directory contains a curated collection of test cases designed to validate the transcription system across diverse scenarios and edge cases.

## Directory Structure

```
tests/gold_test_set/
├── audio_clips/          # Test audio/video files  
├── ground_truth/         # Ground truth annotations (JSON format)
├── metadata/             # Test case metadata and scenarios
├── test_definitions.json # Complete test suite definitions
└── README.md            # This file
```

## Test Case Coverage

### Core Scenarios (8-12 test cases)

1. **Clear Speech, Multiple Speakers**
   - Clean audio with 2-4 distinct speakers
   - Minimal overlap, good audio quality
   - Tests baseline accuracy

2. **Speaker Overlap and Crosstalk**
   - Significant speaker overlap periods
   - Interrupt patterns and simultaneous speech
   - Tests diarization under challenging conditions

3. **Accent and Language Variations**
   - Different English accents (US, UK, Australian, etc.)
   - Non-native speaker patterns
   - Tests ASR robustness

4. **Noise and Acoustic Challenges**
   - Background noise (traffic, crowds, machinery)
   - Echo and reverberation effects
   - Tests audio preprocessing effectiveness

5. **Low Volume and Whispered Speech**
   - Quiet speakers, whispers
   - Dynamic volume changes
   - Tests sensitivity and normalization

6. **Silent Regions and Pauses**
   - Extended silence periods
   - Long pauses between speakers
   - Tests voice activity detection

7. **Technical Jargon and Specialized Content**
   - Domain-specific terminology
   - Technical discussions
   - Tests linguistic model performance

8. **Short Duration Edge Cases**
   - Very brief utterances (< 1 second)
   - Single word responses
   - Tests minimum detection thresholds

## Ground Truth Format

Each test case includes:

### Speaker Segments (JSON)
```json
{
  "test_id": "clear_speech_4speakers",
  "audio_duration": 120.5,
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "speaker": "Speaker_A",
      "text": "Welcome everyone to today's meeting."
    }
  ],
  "metadata": {
    "speaker_count": 4,
    "noise_level": "low",
    "accent_types": ["US_Midwest", "UK_London"],
    "audio_quality": "high"
  }
}
```

### Quality Expectations
Each test defines expected performance ranges:
- **DER (Diarization Error Rate)**: Expected range for speaker identification accuracy
- **WER (Word Error Rate)**: Expected transcription accuracy range  
- **Processing Time**: Expected processing duration limits
- **Confidence Calibration**: Expected confidence score accuracy

## Usage in Testing

1. **Acceptance Tests**: Validate system meets quality thresholds
2. **Regression Tests**: Detect performance degradation over time
3. **Benchmark Tests**: Compare different configuration settings
4. **CI/CD Validation**: Automated quality gates in deployment pipeline

## Test Case Naming Convention

- `{scenario}_{speakers}_{duration}_{quality}` 
- Example: `overlap_heavy_3speakers_45s_medium`

## Maintenance

- Test cases should be reviewed and updated quarterly
- New edge cases should be added as they are discovered
- Ground truth should be verified by multiple annotators
- Performance expectations should be calibrated based on system improvements