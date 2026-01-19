"""Main video processing pipeline."""

from pathlib import Path
from typing import Optional, List
import sys

import numpy as np

# Import wrappers
from .codeformer_wrapper import CodeFormerEnhancer
from .liveportrait_wrapper import LivePortraitAnimator
from .realesrgan_wrapper import RealESRGANUpscaler
from .film_wrapper import FILMInterpolator

# Import video utilities
from utils.video_utils import read_video, write_video, extract_first_frame


class VideoPipeline:
    """Main video processing pipeline."""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize video pipeline with all models.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
        """
        print("=" * 60)
        print("Initializing Video Pipeline")
        print("=" * 60)
        
        print("\n[1/5] Initializing CodeFormer (pre-processing)...")
        self.codeformer_pre = CodeFormerEnhancer(device=device)
        
        print("\n[2/5] Initializing LivePortrait...")
        self.liveportrait = LivePortraitAnimator(device=device)
        
        print("\n[3/5] Initializing CodeFormer (post-processing)...")
        self.codeformer_post = CodeFormerEnhancer(device=device)
        
        print("\n[4/5] Initializing RealESRGAN...")
        self.realesrgan = RealESRGANUpscaler(device=device)
        
        print("\n[5/5] Initializing FILM (placeholder)...")
        self.film = FILMInterpolator(device=device)
        
        print("\n" + "=" * 60)
        print("Pipeline initialization complete!")
        print("=" * 60)
    
    def process_video(
        self,
        input_video_path: str,
        audio_path: str,
        output_path: str,
        edit_instructions: Optional[dict] = None
    ) -> str:
        """
        Process video through the complete pipeline.
        
        Args:
            input_video_path: Path to input video file
            audio_path: Path to audio file for animation
            output_path: Path to save output video
            edit_instructions: Optional dictionary with edit instructions
                              (e.g., {'expression_scale': 1.2, 'lip_intensity': 1.5})
        
        Returns:
            Path to output video file
        """
        print("\n" + "=" * 60)
        print("Starting Video Processing Pipeline")
        print("=" * 60)
        print(f"Input video: {input_video_path}")
        print(f"Audio file: {audio_path}")
        print(f"Output path: {output_path}")
        print("=" * 60)
        
        # Read input video
        print("\n📹 Reading input video...")
        frames, fps = read_video(input_video_path)
        print(f"✓ Loaded {len(frames)} frames at {fps:.2f} FPS")
        
        # Stage 1: Pre-process with CodeFormer (fidelity 0.7)
        print("\n" + "=" * 60)
        print("STAGE 1: Pre-processing with CodeFormer (fidelity=0.7)")
        print("=" * 60)
        print(f"Processing {len(frames)} frames...")
        frames_stage1 = self.codeformer_pre.enhance_video(frames, fidelity_weight=0.7)
        print(f"✓ Stage 1 complete: {len(frames_stage1)} frames processed")
        
        # Stage 2: Animate with LivePortrait
        print("\n" + "=" * 60)
        print("STAGE 2: Animation with LivePortrait")
        print("=" * 60)
        print("Extracting first frame for animation...")
        source_frame = frames_stage1[0]
        print(f"Animating with audio: {audio_path}")
        
        # Prepare control params from edit_instructions
        control_params = None
        if edit_instructions:
            control_params = edit_instructions
            print(f"Using edit instructions: {edit_instructions}")
        
        frames_stage2 = self.liveportrait.animate_video(
            source_frame=source_frame,
            audio_path=audio_path,
            control_params=control_params,
            fps=fps
        )
        print(f"✓ Stage 2 complete: {len(frames_stage2)} frames generated")
        
        # Stage 3: Post-process with CodeFormer (fidelity 0.5)
        print("\n" + "=" * 60)
        print("STAGE 3: Post-processing with CodeFormer (fidelity=0.5)")
        print("=" * 60)
        print(f"Processing {len(frames_stage2)} frames...")
        frames_stage3 = self.codeformer_post.enhance_video(frames_stage2, fidelity_weight=0.5)
        print(f"✓ Stage 3 complete: {len(frames_stage3)} frames processed")
        
        # Stage 4: Upscale with RealESRGAN
        print("\n" + "=" * 60)
        print("STAGE 4: Upscaling with RealESRGAN")
        print("=" * 60)
        print(f"Upscaling {len(frames_stage3)} frames...")
        frames_stage4 = self.realesrgan.upscale_video(frames_stage3)
        print(f"✓ Stage 4 complete: {len(frames_stage4)} frames upscaled")
        
        # Stage 5: Interpolate with FILM (placeholder)
        print("\n" + "=" * 60)
        print("STAGE 5: Frame Interpolation with FILM (placeholder)")
        print("=" * 60)
        print(f"Interpolating {len(frames_stage4)} frames...")
        frames_stage5 = self.film.interpolate_frames(frames_stage4)
        print(f"✓ Stage 5 complete: {len(frames_stage5)} frames (placeholder - no interpolation)")
        
        # Save video with audio
        print("\n" + "=" * 60)
        print("Saving output video with audio...")
        print("=" * 60)
        output_path_final = write_video(
            frames=frames_stage5,
            output_path=output_path,
            fps=fps,
            audio_path=audio_path
        )
        print(f"✓ Video saved: {output_path_final}")
        
        print("\n" + "=" * 60)
        print("Pipeline Processing Complete!")
        print("=" * 60)
        print(f"Output video: {output_path_final}")
        print(f"Total frames: {len(frames_stage5)}")
        print(f"FPS: {fps:.2f}")
        print("=" * 60)
        
        return output_path_final
