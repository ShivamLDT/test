"""FILM frame interpolation wrapper."""

import sys
from pathlib import Path
from typing import List, Optional
import warnings

import cv2
import numpy as np

# Add frame-interpolation to path
film_path = Path(__file__).parent.parent / 'frame-interpolation'
if str(film_path) not in sys.path:
    sys.path.insert(0, str(film_path))

try:
    import tensorflow as tf
    from eval import interpolator as interpolator_lib
    from eval import util as film_util
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(e)


class FILMInterpolator:
    """FILM frame interpolator for video frame interpolation."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        times_to_interpolate: int = 1
    ):
        """
        Initialize FILM interpolator.
        
        Args:
            model_path: Path to FILM saved model.
                       Defaults to '../frame-interpolation/pretrained_models/film_net/Style/saved_model'
            device: Device to run inference on ('cuda' or 'cpu')
            times_to_interpolate: Number of times to do recursive midpoint interpolation.
                                 Higher values create more frames (2^times_to_interpolate + 1 per pair)
        """
        if not TF_AVAILABLE:
            raise ImportError(
                f"TensorFlow is required for FILM interpolation. "
                f"Install with: pip install tensorflow. Error: {TF_IMPORT_ERROR}"
            )
        
        # Set model path
        if model_path is None:
            model_path = film_path / 'pretrained_models' / 'film_net' / 'Style' / 'saved_model'
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"FILM model not found at: {model_path}\n"
                f"Please download the pretrained model from: "
                f"https://github.com/google-research/frame-interpolation"
            )
        
        # Configure TensorFlow device
        if device == 'cpu':
            tf.config.set_visible_devices([], 'GPU')
        elif device == 'cuda':
            # Try to enable GPU if available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"Warning: Could not configure GPU: {e}")
        
        # Initialize interpolator
        try:
            self.interpolator = interpolator_lib.Interpolator(str(model_path), align=None)
            self.times_to_interpolate = times_to_interpolate
            print(f"FILM interpolator initialized with model: {model_path}")
            print(f"Times to interpolate: {times_to_interpolate} (will generate {2**times_to_interpolate + 1} frames per pair)")
        except Exception as e:
            raise RuntimeError(f"Failed to load FILM model: {e}") from e
    
    def _bgr_to_rgb_normalized(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR uint8 frame to RGB float32 normalized [0, 1].
        
        Args:
            frame: BGR frame as uint8 array (H, W, 3)
            
        Returns:
            RGB frame as float32 array (H, W, 3) in range [0, 1]
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        return frame_normalized
    
    def _rgb_normalized_to_bgr(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert RGB float32 normalized [0, 1] to BGR uint8.
        
        Args:
            frame: RGB frame as float32 array (H, W, 3) in range [0, 1]
            
        Returns:
            BGR frame as uint8 array (H, W, 3)
        """
        # Clip to valid range
        frame_clipped = np.clip(frame, 0.0, 1.0)
        # Convert to uint8
        frame_uint8 = (frame_clipped * 255.0 + 0.5).astype(np.uint8)
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        return frame_bgr
    
    def interpolate_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Interpolate frames using FILM.
        
        Args:
            frames: List of input frames as numpy arrays (BGR format, uint8)
            
        Returns:
            List of interpolated frames as numpy arrays (BGR format, uint8)
        """
        if len(frames) < 2:
            print("Warning: Need at least 2 frames for interpolation. Returning original frames.")
            return frames
        
        print(f"Interpolating {len(frames)} frames with FILM (times_to_interpolate={self.times_to_interpolate})...")
        
        # Convert frames to RGB normalized format
        frames_rgb_norm = [self._bgr_to_rgb_normalized(frame) for frame in frames]
        
        # Interpolate using FILM
        try:
            if self.times_to_interpolate == 0:
                # No interpolation, just return original frames
                interpolated_frames_rgb = frames_rgb_norm
            else:
                # Use recursive interpolation
                interpolated_frames_rgb = list(
                    film_util.interpolate_recursively_from_memory(
                        frames_rgb_norm,
                        self.times_to_interpolate,
                        self.interpolator
                    )
                )
            
            # Convert back to BGR uint8
            interpolated_frames = [
                self._rgb_normalized_to_bgr(frame) for frame in interpolated_frames_rgb
            ]
            
            print(f"✓ Interpolation complete: {len(frames)} -> {len(interpolated_frames)} frames")
            return interpolated_frames
        
        except Exception as e:
            print(f"Error during FILM interpolation: {e}")
            print("Falling back to original frames.")
            return frames
