"""Audio utility functions for video pipeline."""

import subprocess
from pathlib import Path
from typing import Optional, Tuple
import warnings

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa and soundfile not available. Some audio functions will not work.")


def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from video file using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path for output audio file.
                    If None, creates file with same name and .wav extension
        
    Returns:
        Path to extracted audio file
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If ffmpeg extraction fails
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}.wav"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed with return code {result.returncode}.\n"
                f"Error: {result.stderr}"
            )
        
        if not output_path.exists():
            raise RuntimeError("Audio extraction completed but output file not found")
        
        return str(output_path)
    
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to use audio extraction."
        ) from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg command failed: {e}") from e


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If librosa is not available or extraction fails
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not LIBROSA_AVAILABLE:
        # Fallback to ffprobe
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            raise RuntimeError(
                "Could not determine audio duration. Install librosa or ensure ffprobe is available."
            ) from None
    
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        duration = len(y) / sr
        return duration
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {e}") from e


def resample_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 22050
) -> str:
    """
    Resample audio file to target sample rate.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save resampled audio
        target_sr: Target sample rate (default: 22050)
        
    Returns:
        Path to resampled audio file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If librosa is not available or resampling fails
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    if not LIBROSA_AVAILABLE:
        raise RuntimeError(
            "librosa is required for audio resampling. Install with: pip install librosa soundfile"
        )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load audio
        y, sr = librosa.load(str(input_path), sr=None)
        
        # Resample if needed
        if sr != target_sr:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        else:
            y_resampled = y
        
        # Save resampled audio
        sf.write(str(output_path), y_resampled, target_sr)
        
        return str(output_path)
    
    except Exception as e:
        raise RuntimeError(f"Failed to resample audio: {e}") from e


def normalize_audio(
    input_path: str,
    output_path: str,
    target_lufs: float = -23.0
) -> str:
    """
    Normalize audio to target loudness (LUFS).
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save normalized audio
        target_lufs: Target loudness in LUFS (default: -23.0, broadcast standard)
        
    Returns:
        Path to normalized audio file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If normalization fails
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use ffmpeg with loudnorm filter for normalization
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-af', f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11',
            '-y',
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg normalization failed with return code {result.returncode}.\n"
                f"Error: {result.stderr}"
            )
        
        return str(output_path)
    
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to use audio normalization."
        ) from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg command failed: {e}") from e


def get_audio_info(audio_path: str) -> dict:
    """
    Get information about audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio information (duration, sample_rate, channels, etc.)
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If info extraction fails
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    info = {
        'path': str(audio_path),
        'duration': None,
        'sample_rate': None,
        'channels': None,
        'format': None
    }
    
    if LIBROSA_AVAILABLE:
        try:
            y, sr = librosa.load(str(audio_path), sr=None, mono=False)
            info['sample_rate'] = sr
            info['duration'] = len(y[0] if y.ndim > 1 else y) / sr
            info['channels'] = 1 if y.ndim == 1 else y.shape[0]
        except Exception:
            pass
    
    # Try ffprobe for additional info
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration:stream=sample_rate,channels,codec_name',
            '-of', 'json',
            str(audio_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        import json
        probe_data = json.loads(result.stdout)
        
        if 'format' in probe_data and 'duration' in probe_data['format']:
            info['duration'] = float(probe_data['format']['duration'])
        
        if 'streams' in probe_data and len(probe_data['streams']) > 0:
            stream = probe_data['streams'][0]
            if 'sample_rate' in stream:
                info['sample_rate'] = int(stream['sample_rate'])
            if 'channels' in stream:
                info['channels'] = int(stream['channels'])
            if 'codec_name' in stream:
                info['format'] = stream['codec_name']
    
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, FileNotFoundError):
        pass
    
    return info
