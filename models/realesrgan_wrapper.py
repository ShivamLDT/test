"""RealESRGAN wrapper for image and video upscaling."""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add Real-ESRGAN to path
realesrgan_path = Path(__file__).parent.parent / 'Real-ESRGAN'
if str(realesrgan_path) not in sys.path:
    sys.path.insert(0, str(realesrgan_path))

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class RealESRGANUpscaler:
    """RealESRGAN upscaler for images and videos."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scale: int = 2,
        tile: int = 400,
        device: Optional[str] = None,
        gpu_id: Optional[int] = None
    ):
        """
        Initialize RealESRGAN upscaler.
        
        Args:
            model_path: Path to RealESRGAN model weights.
                       Defaults to '../Real-ESRGAN/weights/RealESRGAN_x2plus.pth'
            scale: Upscaling scale factor (default: 2)
            tile: Tile size for memory efficiency (default: 400)
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
            gpu_id: GPU device ID to use (default: None, uses default GPU)
        """
        self.scale = scale
        self.tile = tile
        
        # Set model path
        if model_path is None:
            model_path = realesrgan_path / 'weights' / 'RealESRGAN_x2plus.pth'
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"RealESRGAN model not found at: {model_path}")
        
        # Initialize RRDBNet model
        # For RealESRGAN_x2plus: scale=2, num_block=23
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )
        
        # Initialize RealESRGANer
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=True,  # Use half precision for memory efficiency
            device=device,
            gpu_id=gpu_id
        )
        
        print(f"RealESRGAN initialized with scale={scale}, tile={tile}")
    
    def upscale_frame(self, img: np.ndarray) -> np.ndarray:
        """
        Upscale a single frame using RealESRGAN.
        
        Args:
            img: Input image as numpy array (BGR format, uint8)
        
        Returns:
            Upscaled image as numpy array (BGR format, uint8)
        """
        try:
            output, _ = self.upsampler.enhance(img, outscale=self.scale)
            return output
        except RuntimeError as error:
            print(f"Error during upscaling: {error}")
            print("If you encounter CUDA out of memory, try reducing the tile size.")
            raise
    
    def upscale_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Upscale a list of video frames using RealESRGAN.
        
        Args:
            frames: List of input frames as numpy arrays (BGR format, uint8)
        
        Returns:
            List of upscaled frames as numpy arrays (BGR format, uint8)
        """
        upscaled_frames = []
        total_frames = len(frames)
        
        for i, frame in enumerate(frames):
            print(f"Upscaling frame {i+1}/{total_frames}")
            upscaled_frame = self.upscale_frame(frame)
            upscaled_frames.append(upscaled_frame)
        
        return upscaled_frames
