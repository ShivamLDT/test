"""CodeFormer wrapper for face restoration in video pipeline."""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

# Add CodeFormer to path
codeformer_path = Path(__file__).parent.parent / 'CodeFormer'
if str(codeformer_path) not in sys.path:
    sys.path.insert(0, str(codeformer_path))

from basicsr.archs.codeformer_arch import CodeFormer
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.misc import get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper


class CodeFormerEnhancer:
    """CodeFormer face restoration enhancer for images and videos."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        upscale: int = 1,
        detection_model: str = 'retinaface_resnet50'
    ):
        """
        Initialize CodeFormer enhancer.
        
        Args:
            model_path: Path to CodeFormer model weights. 
                       Defaults to '../CodeFormer/weights/CodeFormer/codeformer.pth'
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
            upscale: Upscaling factor for face restoration (default: 1)
            detection_model: Face detection model to use (default: 'retinaface_resnet50')
        """
        # Set device
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        # Set model path
        if model_path is None:
            model_path = codeformer_path / 'weights' / 'CodeFormer' / 'codeformer.pth'
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"CodeFormer model not found at: {model_path}")
        
        # Initialize CodeFormer model
        self.net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=['32', '64', '128', '256']
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(str(model_path), map_location=self.device)
        if 'params_ema' in checkpoint:
            self.net.load_state_dict(checkpoint['params_ema'])
        else:
            self.net.load_state_dict(checkpoint)
        
        self.net.eval()
        
        # Initialize FaceRestoreHelper
        self.face_helper = FaceRestoreHelper(
            upscale=upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext='png',
            use_parse=True,
            device=self.device
        )
        
        self.upscale = upscale
        print(f"CodeFormer initialized on device: {self.device}")
    
    @torch.no_grad()
    def enhance_frame(
        self,
        img: np.ndarray,
        fidelity_weight: float = 0.5
    ) -> np.ndarray:
        """
        Enhance a single frame using CodeFormer.
        
        Args:
            img: Input image as numpy array (BGR format, uint8)
            fidelity_weight: Balance between quality (lower) and fidelity (higher).
                            Range: 0.0 to 1.0 (default: 0.5)
        
        Returns:
            Enhanced image as numpy array (BGR format, uint8)
        """
        # Clean previous results
        self.face_helper.clean_all()
        
        # Read image
        self.face_helper.read_image(img)
        
        # Detect faces and get landmarks
        num_det_faces = self.face_helper.get_face_landmarks_5(
            only_center_face=False,
            resize=640,
            eye_dist_threshold=5
        )
        
        if num_det_faces == 0:
            print("No faces detected, returning original image")
            return img
        
        # Align and warp faces
        self.face_helper.align_warp_face()
        
        # Restore each detected face
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            # Prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0,
                bgr2rgb=True,
                float32=True
            )
            normalize(
                cropped_face_t,
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
                inplace=True
            )
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            
            try:
                # Run inference
                output = self.net(
                    cropped_face_t,
                    w=fidelity_weight,
                    adain=True
                )[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as error:
                print(f"\tFailed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t,
                    rgb2bgr=True,
                    min_max=(-1, 1)
                )
            
            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)
        
        # Paste restored faces back to image
        restored_img = self.face_helper.paste_faces_to_input_image(
            upsample_img=None,
            draw_box=False
        )
        
        return restored_img
    
    @torch.no_grad()
    def enhance_video(
        self,
        frames: List[np.ndarray],
        fidelity_weight: float = 0.5
    ) -> List[np.ndarray]:
        """
        Enhance a list of video frames using CodeFormer.
        
        Args:
            frames: List of input frames as numpy arrays (BGR format, uint8)
            fidelity_weight: Balance between quality (lower) and fidelity (higher).
                            Range: 0.0 to 1.0 (default: 0.5)
        
        Returns:
            List of enhanced frames as numpy arrays (BGR format, uint8)
        """
        enhanced_frames = []
        total_frames = len(frames)
        
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{total_frames}")
            enhanced_frame = self.enhance_frame(frame, fidelity_weight)
            enhanced_frames.append(enhanced_frame)
        
        return enhanced_frames
