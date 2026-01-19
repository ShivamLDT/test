"""LivePortrait wrapper for audio-driven portrait animation."""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

import cv2
import numpy as np
import torch
import librosa

# Add LivePortrait to path
liveportrait_path = Path(__file__).parent.parent / 'LivePortrait'
if str(liveportrait_path) not in sys.path:
    sys.path.insert(0, str(liveportrait_path))

from src.live_portrait_pipeline import LivePortraitPipeline
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.camera import get_rotation_matrix
from src.utils.helper import dct2device


class LivePortraitAnimator:
    """LivePortrait animator for audio-driven portrait animation."""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize LivePortrait animator.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Determine device_id from device string
        if device == 'cuda':
            device_id = 0
        elif device.startswith('cuda:'):
            device_id = int(device.split(':')[1])
        else:
            device_id = 0
        
        # Load InferenceConfig
        self.inference_cfg = InferenceConfig()
        self.inference_cfg.device_id = device_id
        self.inference_cfg.flag_force_cpu = (device == 'cpu')
        
        # Update model paths to use pretrained_weights directory
        weights_dir = liveportrait_path / 'pretrained_weights' / 'liveportrait'
        base_models_dir = weights_dir / 'base_models'
        retargeting_dir = weights_dir / 'retargeting_models'
        
        self.inference_cfg.checkpoint_F = str(base_models_dir / 'appearance_feature_extractor.pth')
        self.inference_cfg.checkpoint_M = str(base_models_dir / 'motion_extractor.pth')
        self.inference_cfg.checkpoint_W = str(base_models_dir / 'warping_module.pth')
        self.inference_cfg.checkpoint_G = str(base_models_dir / 'spade_generator.pth')
        self.inference_cfg.checkpoint_S = str(retargeting_dir / 'stitching_retargeting_module.pth')
        
        # Verify all model files exist
        for attr_name in ['checkpoint_F', 'checkpoint_M', 'checkpoint_W', 'checkpoint_G', 'checkpoint_S']:
            checkpoint_path = getattr(self.inference_cfg, attr_name)
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Load CropConfig
        self.crop_cfg = CropConfig()
        self.crop_cfg.device_id = device_id
        self.crop_cfg.flag_force_cpu = (device == 'cpu')
        
        # Initialize LivePortraitPipeline
        self.pipeline = LivePortraitPipeline(
            inference_cfg=self.inference_cfg,
            crop_cfg=self.crop_cfg
        )
        
        self.device = self.pipeline.live_portrait_wrapper.device
        print(f"LivePortrait initialized on device: {self.device}")
    
    def extract_audio_features(self, audio_path: str, fps: float = 25.0) -> np.ndarray:
        """
        Extract audio features from audio file.
        
        Args:
            audio_path: Path to audio file
            fps: Target frames per second for feature extraction
            
        Returns:
            Audio feature array with shape (n_frames, feature_dim)
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Calculate frame duration
            frame_duration = 1.0 / fps
            
            # Extract features per frame
            features_list = []
            hop_length = int(sr * frame_duration)
            n_fft = 2048
            
            # Extract MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
            
            # Extract energy (RMS)
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            # Normalize features
            n_frames = mfccs.shape[1]
            for i in range(n_frames):
                frame_features = np.concatenate([
                    mfccs[:, i],
                    [spectral_centroids[i] / sr if i < len(spectral_centroids) else 0],
                    [spectral_rolloff[i] / sr if i < len(spectral_rolloff) else 0],
                    [zero_crossing_rate[i] if i < len(zero_crossing_rate) else 0],
                    [rms[i] if i < len(rms) else 0]
                ])
                features_list.append(frame_features)
            
            return np.array(features_list)
        
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio features: {e}") from e
    
    def prepare_source_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Prepare source frame for animation.
        
        Args:
            frame: Source frame as numpy array (BGR format, uint8)
            
        Returns:
            Dictionary containing prepared source information
        """
        # Resize to 256x256 if needed
        if frame.shape[:2] != (256, 256):
            frame_resized = cv2.resize(frame, (256, 256))
        else:
            frame_resized = frame.copy()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Prepare source using LivePortrait wrapper
        I_s = self.pipeline.live_portrait_wrapper.prepare_source(frame_rgb)
        x_s_info = self.pipeline.live_portrait_wrapper.get_kp_info(I_s)
        f_s = self.pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        
        return {
            'I_s': I_s,
            'x_s_info': x_s_info,
            'f_s': f_s,
            'x_s': x_s,
            'R_s': R_s,
            'x_c_s': x_s_info['kp']
        }
    
    def generate_motion_sequence(
        self,
        audio_features: np.ndarray,
        source_info: Dict[str, Any],
        control_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate motion sequence from audio features.
        
        Args:
            audio_features: Audio feature array
            source_info: Source frame information
            control_params: Optional control parameters for prompt-based edits
            
        Returns:
            List of motion parameter dictionaries
        """
        n_frames = len(audio_features)
        motion_sequence = []
        
        x_s_info = source_info['x_s_info']
        R_s = source_info['R_s']
        
        # Base expression (neutral)
        base_exp = x_s_info['exp'].clone()
        
        # Get lip array for neutral lip state
        lip_array = torch.from_numpy(self.inference_cfg.lip_array).to(
            dtype=torch.float32,
            device=self.device
        )
        
        for i in range(n_frames):
            # Extract audio features for this frame
            audio_feat = audio_features[i]
            
            # Map audio features to expression parameters
            # Use RMS energy for lip movement (indices 6, 12, 14, 17, 19, 20)
            # Use MFCC for general expression
            rms_energy = audio_feat[-1]  # Last feature is RMS
            mfcc_0 = audio_feat[0]  # First MFCC coefficient
            
            # Create expression delta based on audio
            delta_new = base_exp.clone()
            
            # Lip movement based on RMS energy (normalized to 0-1 range)
            lip_intensity = np.clip(rms_energy * 10, 0, 1)  # Scale RMS for lip movement
            
            # Apply lip movement to lip keypoints
            lip_indices = [6, 12, 14, 17, 19, 20]
            for lip_idx in lip_indices:
                # Modulate lip opening/closing based on audio energy
                delta_new[:, lip_idx, 1] += lip_intensity * 0.1  # Vertical movement
                delta_new[:, lip_idx, 0] += (mfcc_0 / 10) * 0.05  # Horizontal movement
            
            # General expression modulation
            exp_modulation = (mfcc_0 / 20) * 0.1
            delta_new[:, 1:6, :] += exp_modulation  # Affect mouth area
            
            # Apply control parameters if provided
            if control_params is not None:
                delta_new = self.apply_control_edits(delta_new, control_params, i)
            
            # Create motion parameters
            motion_params = {
                'R': R_s.clone(),
                'exp': delta_new,
                't': x_s_info['t'].clone(),
                'scale': x_s_info['scale'].clone(),
                'kp': x_s_info['kp'].clone(),
                'x_s': source_info['x_s'].clone()
            }
            
            motion_sequence.append(motion_params)
        
        return motion_sequence
    
    def apply_control_edits(
        self,
        motion_params: torch.Tensor,
        control_params: Dict[str, Any],
        frame_idx: int
    ) -> torch.Tensor:
        """
        Apply control parameter edits to motion parameters.
        
        Args:
            motion_params: Motion parameters tensor
            control_params: Control parameters dictionary
            frame_idx: Current frame index
            
        Returns:
            Modified motion parameters
        """
        modified_params = motion_params.clone()
        
        # Apply expression edits
        if 'expression_scale' in control_params:
            scale = control_params['expression_scale']
            if isinstance(scale, (list, np.ndarray)) and frame_idx < len(scale):
                modified_params *= scale[frame_idx]
            else:
                modified_params *= scale
        
        # Apply lip edits
        if 'lip_intensity' in control_params:
            intensity = control_params['lip_intensity']
            if isinstance(intensity, (list, np.ndarray)) and frame_idx < len(intensity):
                lip_scale = intensity[frame_idx]
            else:
                lip_scale = intensity
            
            lip_indices = [6, 12, 14, 17, 19, 20]
            for lip_idx in lip_indices:
                modified_params[:, lip_idx, :] *= lip_scale
        
        # Apply pose edits
        if 'head_pose' in control_params:
            pose = control_params['head_pose']
            if isinstance(pose, (list, np.ndarray)) and frame_idx < len(pose):
                # This would require modifying R matrix, simplified here
                pass
        
        return modified_params
    
    def animate_video(
        self,
        source_frame: np.ndarray,
        audio_path: str,
        control_params: Optional[Dict[str, Any]] = None,
        fps: float = 25.0
    ) -> List[np.ndarray]:
        """
        Animate video from source frame and audio.
        
        Args:
            source_frame: Source frame as numpy array (BGR format, uint8)
            audio_path: Path to audio file
            control_params: Optional control parameters for prompt-based edits
            fps: Target frames per second
            
        Returns:
            List of animated frames as numpy arrays (BGR format, uint8)
        """
        print("Extracting audio features...")
        audio_features = self.extract_audio_features(audio_path, fps)
        n_frames = len(audio_features)
        print(f"Extracted features for {n_frames} frames")
        
        print("Preparing source frame...")
        source_info = self.prepare_source_frame(source_frame)
        
        print("Generating motion sequence...")
        motion_sequence = self.generate_motion_sequence(
            audio_features,
            source_info,
            control_params
        )
        
        print(f"Animating {n_frames} frames...")
        animated_frames = []
        
        f_s = source_info['f_s']
        x_s = source_info['x_s']
        x_c_s = source_info['x_c_s']
        
        for i in range(n_frames):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processing frame {i+1}/{n_frames}")
            
            motion_params = motion_sequence[i]
            motion_params = dct2device(motion_params, self.device)
            
            # Get motion parameters
            R_new = motion_params['R']
            delta_new = motion_params['exp']
            t_new = motion_params['t']
            scale_new = motion_params['scale']
            
            # Calculate target keypoints
            x_d_i = scale_new * (x_c_s @ R_new + delta_new) + t_new
            
            # Apply stitching if enabled
            if self.inference_cfg.flag_stitching:
                x_d_i = self.pipeline.live_portrait_wrapper.stitching(x_s, x_d_i)
            
            # Apply driving multiplier
            x_d_i = x_s + (x_d_i - x_s) * self.inference_cfg.driving_multiplier
            
            # Generate frame
            with torch.no_grad():
                out = self.pipeline.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i)
                I_p_i = self.pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
            
            # Convert RGB to BGR
            frame_bgr = cv2.cvtColor(I_p_i[0], cv2.COLOR_RGB2BGR)
            animated_frames.append(frame_bgr)
        
        print("Animation complete!")
        return animated_frames
