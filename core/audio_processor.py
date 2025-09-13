import librosa
import numpy as np
import ffmpeg
import tempfile
import os
import subprocess
from typing import Tuple, Optional

class AudioProcessor:
    """Handles audio extraction and preprocessing from video files"""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    @staticmethod
    def check_ffmpeg_availability() -> Tuple[bool, str]:
        """
        Check if FFmpeg is available and get version info.
        
        Returns:
            Tuple of (is_available, version_or_error_message)
        """
        try:
            # Try to run ffmpeg -version
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Extract version from first line
                version_line = result.stdout.split('\n')[0]
                return True, version_line
            else:
                return False, f"FFmpeg returned error code {result.returncode}"
                
        except FileNotFoundError:
            return False, "FFmpeg not found in system PATH"
        except subprocess.TimeoutExpired:
            return False, "FFmpeg command timed out"
        except Exception as e:
            return False, f"Error checking FFmpeg: {str(e)}"
    
    @staticmethod
    def get_ffmpeg_install_instructions() -> str:
        """
        Get installation instructions for FFmpeg based on the environment.
        
        Returns:
            Installation instructions string
        """
        return """
**FFmpeg Installation Instructions:**

**For Replit:**
1. Add `ffmpeg` to your system dependencies in `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["poetry-core"]
   build-backend = "poetry.core.masonry.api"
   
   [tool.poetry.dependencies]
   python = "^3.11"
   # ... other dependencies ...
   
   [tool.poetry.group.dev.dependencies]
   # Add this line:
   ffmpeg = "*"
   ```

2. Or use the Replit package manager:
   - Open the "Packages" tab in Replit
   - Search for "ffmpeg"
   - Click "Install"

**For local development:**
- **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html

**Verification:**
Run `ffmpeg -version` in your terminal to verify installation.
        """
    
    def extract_audio_from_video(self, video_path: str) -> Tuple[str, str]:
        """
        Extract audio from MP4 video file and create both raw and cleaned versions.
        
        Args:
            video_path: Path to input MP4 file
            
        Returns:
            Tuple of (raw_audio_path, clean_audio_path)
        """
        # Create temporary files for audio outputs
        raw_audio_fd, raw_audio_path = tempfile.mkstemp(suffix='_raw.wav')
        clean_audio_fd, clean_audio_path = tempfile.mkstemp(suffix='_clean.wav')
        
        # Close file descriptors immediately (we just need the paths)
        os.close(raw_audio_fd)
        os.close(clean_audio_fd)
        
        success = False
        try:
            # Extract raw audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(
                    raw_audio_path,
                    acodec='pcm_s16le',
                    ar=self.target_sr,
                    ac=1,  # mono
                    t=5400  # 90 minutes max
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Load and preprocess audio
            audio_data, sr = librosa.load(raw_audio_path, sr=self.target_sr, mono=True)
            
            # Apply preprocessing
            cleaned_audio = self._preprocess_audio(audio_data, int(sr))
            
            # Save cleaned audio
            import soundfile as sf
            sf.write(clean_audio_path, cleaned_audio, sr)
            
            success = True
            return raw_audio_path, clean_audio_path
            
        except Exception as e:
            raise Exception(f"Audio extraction failed: {str(e)}")
            
        finally:
            # Clean up temp files only on failure
            if not success:
                for path in [raw_audio_path, clean_audio_path]:
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                        except OSError:
                            pass  # Best effort cleanup
    
    def _preprocess_audio(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply light preprocessing to improve audio quality.
        
        Args:
            audio_data: Raw audio samples
            sr: Sample rate
            
        Returns:
            Preprocessed audio samples
        """
        # Trim silence from beginning and end (keep 0.5s buffer)
        trimmed_audio, _ = librosa.effects.trim(
            audio_data, 
            top_db=20,
            frame_length=2048,
            hop_length=512
        )
        
        # Add small buffer back
        buffer_samples = int(0.5 * sr)
        start_idx = max(0, len(audio_data) - len(trimmed_audio) - buffer_samples)
        end_idx = min(len(audio_data), start_idx + len(trimmed_audio) + 2 * buffer_samples)
        audio_data = audio_data[start_idx:end_idx]
        
        # Light noise reduction using spectral subtraction
        audio_data = self._spectral_subtraction(audio_data, sr)
        
        # Peak normalization (conservative)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            # Normalize to 85% of max to avoid clipping
            audio_data = audio_data * (0.85 / max_val)
        
        # Automatic gain control (very conservative)
        audio_data = self._apply_agc(audio_data, sr)
        
        return audio_data
    
    def _spectral_subtraction(self, audio_data: np.ndarray, sr: int, alpha: float = 2.0) -> np.ndarray:
        """
        Apply light spectral subtraction for noise reduction.
        
        Args:
            audio_data: Input audio samples
            sr: Sample rate
            alpha: Spectral subtraction factor
            
        Returns:
            Denoised audio samples
        """
        # STFT parameters
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Compute STFT
        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 1 second (assume initial noise)
        noise_frames = int(1.0 * sr / hop_length)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        clean_magnitude = magnitude - alpha * noise_spectrum
        
        # Ensure we don't go below 10% of original magnitude
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
        
        # Reconstruct audio
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=hop_length)
        
        return clean_audio
    
    def _apply_agc(self, audio_data: np.ndarray, sr: int, 
                   target_rms: float = 0.15, window_size: float = 1.0) -> np.ndarray:
        """
        Apply automatic gain control to maintain consistent levels.
        
        Args:
            audio_data: Input audio samples
            sr: Sample rate  
            target_rms: Target RMS level
            window_size: AGC window size in seconds
            
        Returns:
            AGC-processed audio samples
        """
        window_samples = int(window_size * sr)
        hop_samples = window_samples // 4
        
        output_audio = audio_data.copy()
        
        for i in range(0, len(audio_data) - window_samples, hop_samples):
            window = audio_data[i:i + window_samples]
            current_rms = np.sqrt(np.mean(window ** 2))
            
            if current_rms > 0:
                gain = target_rms / current_rms
                # Limit gain to prevent artifacts
                gain = np.clip(gain, 0.5, 2.0)
                
                # Apply gain with smooth transition
                output_audio[i:i + window_samples] *= gain
        
        return output_audio
    
    
    def cleanup_temp_files(self, *file_paths: str) -> None:
        """
        Clean up temporary files safely.
        
        Args:
            *file_paths: Paths to temporary files to clean up
        """
        for path in file_paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    print(f"Cleaned up temp file: {path}")
                except OSError as e:
                    print(f"Warning: Could not clean up temp file {path}: {e}")
    
    def estimate_noise_level(self, audio_path: str) -> str:
        """
        Estimate the noise level in the audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Noise level category: 'low', 'medium', or 'high'
        """
        try:
            # Load first 30 seconds for analysis
            audio_data, sr = librosa.load(audio_path, sr=self.target_sr, duration=30.0)
            
            # Compute spectral centroid and bandwidth
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            
            # Compute RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            
            # Simple heuristic for noise level estimation
            avg_centroid = np.mean(spectral_centroids)
            avg_bandwidth = np.mean(spectral_bandwidth)
            avg_rms = np.mean(rms)
            
            # Thresholds based on typical speech characteristics
            if avg_centroid > 3000 and avg_bandwidth > 2500:
                return 'high'
            elif avg_centroid > 2000 and avg_bandwidth > 2000:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            print(f"Warning: Could not estimate noise level: {e}")
            return 'medium'  # Default fallback
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            audio_data, sr = librosa.load(audio_path, sr=None)
            return len(audio_data) / sr
        except Exception as e:
            raise Exception(f"Could not determine audio duration: {str(e)}")
