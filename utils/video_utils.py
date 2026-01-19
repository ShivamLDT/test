"""Video utility functions for reading, writing, and processing video files."""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    """
    Read all frames from a video file.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Tuple containing:
            - frames: List of numpy arrays representing video frames
            - fps: Frames per second of the video
            
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video cannot be opened or read
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Invalid FPS value: {fps}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if len(frames) == 0:
            raise ValueError(f"No frames could be read from video: {video_path}")
        
        return frames, float(fps)
    
    finally:
        cap.release()


def write_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 25.0,
    audio_path: Optional[str] = None
) -> str:
    """
    Write frames to a video file.
    
    Args:
        frames: List of numpy arrays representing video frames
        output_path: Path where the output video will be saved
        fps: Frames per second for the output video (default: 25.0)
        audio_path: Optional path to audio file to merge with video
        
    Returns:
        Path to the output video file
        
    Raises:
        ValueError: If frames list is empty or invalid parameters provided
        RuntimeError: If video writing fails
    """
    if not frames:
        raise ValueError("Frames list cannot be empty")
    
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got: {fps}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get frame dimensions from first frame
    height, width = frames[0].shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for: {output_path}")
    
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame dimensions mismatch. Expected {(height, width)}, "
                    f"got {frame.shape[:2]}"
                )
            out.write(frame)
        
        out.release()
        
        # Merge audio if provided
        if audio_path:
            return add_audio_to_video(str(output_path), audio_path)
        
        return str(output_path)
    
    except Exception as e:
        out.release()
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Failed to write video: {e}") from e


def extract_first_frame(video_path: str) -> np.ndarray:
    """
    Extract the first frame from a video file.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Numpy array representing the first frame
        
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video cannot be opened or no frames can be read
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError(f"Failed to read first frame from video: {video_path}")
        
        return frame
    
    finally:
        cap.release()


def add_audio_to_video(video_path: str, audio_path: str) -> str:
    """
    Merge audio file with video file using ffmpeg.
    
    Args:
        video_path: Path to the input video file (will be replaced)
        audio_path: Path to the audio file to merge
        
    Returns:
        Path to the output video file (same as input)
        
    Raises:
        FileNotFoundError: If video or audio file doesn't exist
        RuntimeError: If ffmpeg command fails
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Create temporary output path
    temp_output = video_path.parent / f"{video_path.stem}_temp{video_path.suffix}"
    
    try:
        # Use ffmpeg to merge audio and video
        # -i video: input video
        # -i audio: input audio
        # -c:v copy: copy video codec (no re-encoding)
        # -c:a aac: encode audio as AAC
        # -map 0:v:0: use video from first input
        # -map 1:a:0: use audio from second input
        # -shortest: finish encoding when shortest input ends
        # -y: overwrite output file
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            str(temp_output)
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
        
        # Replace original video with merged version
        temp_output.replace(video_path)
        
        return str(video_path)
    
    except subprocess.CalledProcessError as e:
        if temp_output.exists():
            temp_output.unlink()
        raise RuntimeError(f"ffmpeg command failed: {e}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to use audio merging functionality."
        ) from None
