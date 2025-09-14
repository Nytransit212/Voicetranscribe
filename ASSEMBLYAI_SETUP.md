# AssemblyAI Integration Setup Guide

This document provides instructions for setting up AssemblyAI's speaker diarization service as the production diarization provider for the ensemble transcription system.

## Prerequisites

1. **AssemblyAI Account**: Sign up for an AssemblyAI account at https://www.assemblyai.com/
2. **API Key**: Obtain your API key from the AssemblyAI dashboard
3. **Credits**: Ensure your account has sufficient credits for processing

## API Key Setup

### Method 1: Environment Variable (Recommended)

Set the `ASSEMBLYAI_API_KEY` environment variable:

```bash
export ASSEMBLYAI_API_KEY="your_api_key_here"
```

For persistent setup, add to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
echo 'export ASSEMBLYAI_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### Method 2: Replit Secrets (For Replit Environment)

1. Go to the Secrets tab in your Replit environment
2. Add a new secret:
   - Key: `ASSEMBLYAI_API_KEY`
   - Value: `your_api_key_here`
3. The system will automatically detect and use this secret

## API Key Format

AssemblyAI API keys follow this format:
- Start with `sk-`
- Are at least 20 characters long
- Example: `sk-1234567890abcdef1234567890abcdef`

## Configuration

The system is already configured to use AssemblyAI as the primary provider. Default settings are in `config/diarization/external.yaml`:

```yaml
diarization:
  providers:
    enabled: ["assemblyai"]
    primary: "assemblyai"
    
  assemblyai:
    max_file_size_mb: 512        # 512MB limit
    max_duration_seconds: 7200   # 2 hours max
    poll_interval: 5             # Status check interval
    max_wait_time: 1800         # 30 minutes timeout
```

## Testing the Integration

### 1. Test Provider Initialization

```python
from core.diarization_engine import DiarizationEngine

# Initialize with external provider
engine = DiarizationEngine(
    expected_speakers=5,
    noise_level='medium'
)

# Check if AssemblyAI provider is active
if engine.active_provider:
    print(f"✅ Active provider: {engine.active_provider.provider_name}")
else:
    print("⚠️ No external provider active, using fallback")
```

### 2. Test Health Check

```python
# Check provider health
if engine.active_provider:
    status = engine.active_provider.health_check()
    print(f"Provider status: {status}")
```

### 3. Test Full Pipeline

The system will automatically use AssemblyAI when processing videos:

```python
from core.ensemble_manager import EnsembleManager

manager = EnsembleManager(expected_speakers=5)
results = manager.process_video("path/to/your/video.mp4")
```

## Fallback Behavior

If AssemblyAI is unavailable, the system automatically falls back:

1. **Primary**: AssemblyAI (external API)
2. **Fallback 1**: pyannote.audio (if available and HUGGINGFACE_TOKEN set)
3. **Fallback 2**: Mock diarization (for development)

## Cost Management

AssemblyAI charges per audio processed. The system includes cost tracking:

- Default rate: ~$0.00025 per second of audio
- 90-minute video: ~$1.35 cost estimate
- Costs are logged in the processing metadata

## Troubleshooting

### Common Issues

1. **"ASSEMBLYAI_API_KEY environment variable not set"**
   - Solution: Set the API key as described above

2. **"Invalid ASSEMBLYAI_API_KEY format"**
   - Solution: Ensure your API key starts with `sk-` and is correct

3. **"AssemblyAI provider failed validation"**
   - Solution: Check your API key and internet connection
   - Verify your AssemblyAI account has sufficient credits

4. **"Provider failed, falling back to pipeline"**
   - This is normal behavior when AssemblyAI is temporarily unavailable
   - The system will retry AssemblyAI on the next request

### Debugging

Enable debug logging to see detailed provider information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check provider status:

```python
from core.diarization_engine import DiarizationEngine

engine = DiarizationEngine()
for name, provider in engine.providers.items():
    print(f"{name}: {provider.get_status()}")
```

## Production Deployment

For production deployment:

1. **Set API Key**: Ensure `ASSEMBLYAI_API_KEY` is set in production environment
2. **Monitor Costs**: Track usage and costs through AssemblyAI dashboard
3. **Rate Limiting**: The system includes built-in rate limiting and retry logic
4. **Health Monitoring**: Provider health is automatically monitored
5. **Fallback**: Ensure fallback providers are properly configured

## Performance Characteristics

- **Initialization**: ~2-3 seconds for provider setup
- **Upload**: ~30 seconds for 90-minute video file
- **Processing**: ~5-15 minutes depending on audio complexity
- **Total Time**: ~15-20 minutes for 90-minute video with 5+ speakers

## Security Considerations

- API keys are never logged or exposed
- HTTPS/TLS encryption for all API communications
- Automatic API key validation
- Circuit breaker protection against API abuse

## Support

For issues with:
- **AssemblyAI API**: Contact AssemblyAI support
- **Integration Issues**: Check logs and fallback behavior
- **Configuration**: Review `config/diarization/external.yaml`

The system is designed to be resilient and will gracefully handle API failures by falling back to alternative providers.