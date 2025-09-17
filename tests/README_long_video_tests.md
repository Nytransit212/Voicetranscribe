# Long Video Processing Integration Tests

This directory contains comprehensive integration tests specifically designed to validate that the ensemble transcription system can handle long-form video content (2+ hours) without truncation or data loss.

## Overview

The test suite ensures that:
- ✅ Videos longer than 2 hours are processed completely without truncation
- ✅ Memory usage remains manageable during long processing
- ✅ Complete transcripts are generated for full video duration  
- ✅ System handles edge cases and errors gracefully
- ✅ Configuration properly supports long-form content

## Test Files

### 1. `test_long_video_integration.py`
**Comprehensive integration tests for long video processing**

- **2-hour video simulation**: Full end-to-end processing test
- **4-hour extreme duration test**: Validates system limits
- **Memory monitoring**: Tracks memory usage during processing
- **Transcript completeness**: Validates full content coverage
- **Performance validation**: Ensures reasonable processing times

Key test classes:
- `TestLongVideoDurationValidation`: Core duration and completeness tests
- `TestAudioProcessorDurationHandling`: Audio processor specific tests
- `TestTranscriptCompletenessValidation`: Transcript quality validation
- `TestResourceManagementLongVideos`: Resource usage validation

### 2. `test_duration_validation.py`
**Focused tests for duration handling and truncation prevention**

- **FFmpeg command validation**: Ensures no truncation flags are used
- **Configuration validation**: Verifies duration limits support long content
- **Audio processor verification**: Tests core audio handling components
- **End-to-end validation**: Mock processing pipeline tests

Key test classes:
- `TestDurationValidation`: Core validation tests
- `TestMemoryConfiguration`: Memory and chunking configuration tests
- `TestErrorHandlingConfiguration`: Error handling setup validation
- `TestEndToEndDurationValidation`: Complete pipeline tests

### 3. `test_memory_monitoring.py`
**Memory usage and resource management tests**

- **Memory profiling**: Advanced memory usage tracking
- **Leak detection**: Identifies potential memory leaks
- **Resource efficiency**: Validates memory usage patterns
- **Garbage collection**: Tests cleanup effectiveness

Key test classes:
- `TestMemoryManagement`: Core memory usage tests
- `TestMemoryLeakDetection`: Memory leak identification
- `MemoryProfiler`: Advanced profiling utility

### 4. `test_error_handling_long_videos.py`
**Error handling and edge case tests**

- **Extreme duration handling**: Tests with 8+ hour videos
- **Resource exhaustion**: Memory and disk space limits
- **Network timeouts**: Handling of service timeouts
- **Graceful degradation**: Fallback mechanisms
- **Recovery mechanisms**: Retry and recovery logic

Key test classes:
- `TestLongVideoErrorHandling`: Core error scenarios
- `TestConfigurationErrorHandling`: Configuration error handling
- `TestRecoveryMechanisms`: Recovery and retry logic

## Running Tests

### Quick Test (< 2 minutes)
```bash
python tests/run_long_video_tests.py --level quick
```

### Comprehensive Test (< 10 minutes)
```bash
python tests/run_long_video_tests.py --level comprehensive
```

### Full Test Suite (< 30 minutes)
```bash
python tests/run_long_video_tests.py --level all --coverage
```

### Individual Test Files
```bash
# Duration validation only
pytest tests/test_duration_validation.py -v

# Memory monitoring only  
pytest tests/test_memory_monitoring.py -v

# Error handling only
pytest tests/test_error_handling_long_videos.py -v

# Full integration tests (slow)
pytest tests/test_long_video_integration.py -v -m "not slow"
```

## Test Configuration

### Environment Setup
The tests require:
- Python 3.7+
- pytest with timeout plugin
- psutil for memory monitoring
- Mock/patch capabilities
- Access to core modules (audio_processor, ensemble_manager)

### Test Markers
- `@pytest.mark.slow`: Tests that take longer than 30 seconds
- `@pytest.mark.integration`: Integration tests requiring multiple components
- `@pytest.mark.memory_intensive`: Tests that use significant memory

### Configuration Validation
Tests validate these configuration aspects:
- `audio.max_duration_seconds`: Should be ≥10800 (3 hours)
- `chunking.enabled`: Should be true for long content
- `reliability.timeouts`: Should support long processing
- Chunking thresholds and parameters

## Expected Results

### Passing Tests Indicate:
1. **No Truncation**: Videos are processed completely regardless of length
2. **Memory Efficiency**: Memory usage is controlled and predictable
3. **Error Resilience**: System handles failures gracefully
4. **Configuration Correctness**: Settings support long-form content

### Failing Tests May Indicate:
1. **Truncation Issues**: Content being cut off during processing
2. **Memory Leaks**: Uncontrolled memory growth during long processing
3. **Timeout Problems**: Processing timing out for long content
4. **Configuration Issues**: Settings that limit long video support

## Integration with CI/CD

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Run Long Video Processing Tests
  run: |
    python tests/run_long_video_tests.py --level comprehensive
    
# For comprehensive coverage
- name: Full Long Video Test Suite
  run: |
    python tests/run_long_video_tests.py --level all --coverage
```

## Troubleshooting

### Common Issues

1. **Tests timing out**:
   - Increase timeout values in test configuration
   - Use `--level quick` for faster validation

2. **Memory errors during tests**:
   - Check available system memory
   - Run memory monitoring tests individually

3. **Mock processing failures**:
   - Verify mock data generation is working
   - Check test environment setup

### Debug Mode
```bash
# Run with maximum verbosity
pytest tests/test_duration_validation.py -v -s --tb=long

# Run single test with debugging
pytest tests/test_long_video_integration.py::TestLongVideoDurationValidation::test_2_hour_video_complete_processing -v -s
```

## Validation Criteria

Tests validate that the system meets these requirements:

### Duration Handling
- ✅ No FFmpeg truncation flags (`-t`, `-to`) in commands
- ✅ Full duration processing for 2+ hour content
- ✅ Proper duration detection and validation

### Memory Management  
- ✅ Peak memory usage < 8GB for long videos
- ✅ No memory leaks during repeated processing
- ✅ Effective garbage collection

### Transcript Quality
- ✅ ≥95% segment coverage of total duration
- ✅ Consistent speaker labeling throughout
- ✅ Reasonable content density (≥10 words/minute)

### Performance
- ✅ Processing time ≤0.5x real-time for long content
- ✅ Chunked processing for videos >30 minutes
- ✅ Graceful handling of extreme durations

## Contributing

When adding new long video tests:

1. **Follow naming convention**: `test_*_long_video*.py`
2. **Add appropriate markers**: `@pytest.mark.slow`, `@pytest.mark.integration`
3. **Include memory monitoring**: Use `MemoryProfiler` for resource tests
4. **Validate cleanup**: Ensure tests clean up temporary resources
5. **Document expectations**: Clear assertions with descriptive messages

## Support

For issues with long video processing tests:
1. Check test environment validation: `python tests/run_long_video_tests.py --validate-only`
2. Run quick validation first: `python tests/run_long_video_tests.py --level quick`
3. Review memory usage patterns in failed tests
4. Check configuration file settings for duration limits