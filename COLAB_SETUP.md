# Google Colab Setup Guide for Video Processing Pipeline

## Quick Start

1. **Open Google Colab**: https://colab.research.google.com/
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
3. **Upload the notebook**: Upload `COLAB_SETUP.ipynb` or copy the cells below

## Step-by-Step Setup

### Cell 1: Install Dependencies

```python
# Install system dependencies
!apt-get update
!apt-get install -y ffmpeg

# Install Python packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install tensorflow==2.8.0
!pip install numpy==1.26.4
!pip install protobuf==3.20.3
!pip install opencv-python pillow scipy scikit-image imageio
!pip install fastapi uvicorn
!pip install basicsr facexlib gfpgan
!pip install tensorflow-addons==0.18.0
!pip install tensorflow-datasets==4.4.0
!pip install "apache-beam>=2.43.0,<2.63.0"
!pip install mediapy absl-py gin-config parameterized natsort gdown tqdm
!pip install transformers onnxruntime-gpu
!pip install gradio tyro pyyaml

print("✓ Dependencies installed!")
```

### Cell 2: Clone Repositories

```python
import os
os.chdir('/content')

# Clone repositories
!git clone https://github.com/sczhou/CodeFormer.git
!git clone https://github.com/KlingTeam/LivePortrait.git
!git clone https://github.com/xinntao/Real-ESRGAN.git
!git clone https://github.com/google-research/frame-interpolation.git

print("✓ Repositories cloned!")
```

### Cell 3: Apply Compatibility Fixes

```python
import re

# Fix basicsr torchvision import (Real-ESRGAN)
degradations_path = '/content/Real-ESRGAN/realesrgan/data/degradations.py'
if os.path.exists(degradations_path):
    with open(degradations_path, 'r') as f:
        content = f.read()
    content = content.replace(
        'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
        'from torchvision.transforms.functional import rgb_to_grayscale'
    )
    with open(degradations_path, 'w') as f:
        f.write(content)
    print("✓ Fixed Real-ESRGAN torchvision import")

# Create version.py files
version_files = [
    ('/content/Real-ESRGAN/realesrgan/version.py', '0.3.0'),
    ('/content/CodeFormer/basicsr/version.py', '1.3.2')
]

for filepath, version in version_files:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f"""# GENERATED VERSION FILE
# TIME: Generated manually
__version__ = '{version}'
__gitsha__ = 'unknown'
version_info = {tuple(map(int, version.split('.')))}
""")
    print(f"✓ Created {filepath}")

# Fix frame-interpolation tf.data.AUTOTUNE
data_lib_path = '/content/frame-interpolation/training/data_lib.py'
if os.path.exists(data_lib_path):
    with open(data_lib_path, 'r') as f:
        content = f.read()
    content = content.replace('tf.data.experimental.AUTOTUNE', 'tf.data.AUTOTUNE')
    with open(data_lib_path, 'w') as f:
        f.write(content)
    print("✓ Fixed frame-interpolation tf.data.AUTOTUNE")

# Fix Real-ESRGAN video inference
video_inference_path = '/content/Real-ESRGAN/inference_realesrgan_video.py'
if os.path.exists(video_inference_path):
    with open(video_inference_path, 'r') as f:
        content = f.read()
    
    # Fix num_process for CPU
    content = re.sub(
        r'num_process = num_gpus \* args\.num_process_per_gpu',
        r'num_process = max(1, num_gpus * args.num_process_per_gpu)  # Ensure at least 1 process for CPU mode',
        content
    )
    
    # Fix CUDA synchronization
    old_sync = '        torch.cuda.synchronize(device)'
    new_sync = '''        # Only synchronize if using CUDA device
        if device is not None and device.type == 'cuda':
            torch.cuda.synchronize(device)'''
    content = content.replace(old_sync, new_sync)
    
    with open(video_inference_path, 'w') as f:
        f.write(content)
    print("✓ Fixed Real-ESRGAN video inference")

print("\n✓ All compatibility fixes applied!")
```

### Cell 4: Download Pretrained Weights

```python
import gdown
from pathlib import Path

# Create directories
os.makedirs('/content/pretrained_models', exist_ok=True)
os.makedirs('/content/CodeFormer/weights/CodeFormer', exist_ok=True)
os.makedirs('/content/Real-ESRGAN/weights', exist_ok=True)

print("Downloading model weights...")

# CodeFormer weights
print("\n[1/4] Downloading CodeFormer weights...")
!wget -q https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth -O /content/CodeFormer/weights/CodeFormer/codeformer.pth
print("✓ CodeFormer weights downloaded")

# Real-ESRGAN weights
print("\n[2/4] Downloading Real-ESRGAN weights...")
!wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /content/Real-ESRGAN/weights/RealESRGAN_x4plus.pth
print("✓ Real-ESRGAN weights downloaded")

# LivePortrait weights (using HuggingFace)
print("\n[3/4] Downloading LivePortrait weights from HuggingFace...")
!pip install -q huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KlingTeam/LivePortrait",
    local_dir="/content/LivePortrait/pretrained_weights",
    ignore_patterns=["*.git*", "README.md", "docs"]
)
print("✓ LivePortrait weights downloaded")

# FILM weights (frame-interpolation) - Manual download needed
print("\n[4/4] FILM weights need manual download from Google Drive:")
print("https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy?usp=sharing")
print("Download film_net and vgg folders to /content/pretrained_models/")

print("\n✓ Weight download complete!")
```

### Cell 5: Test Models

```python
# Test Real-ESRGAN
print("Testing Real-ESRGAN...")
os.makedirs('/content/Real-ESRGAN/inputs', exist_ok=True)
os.makedirs('/content/Real-ESRGAN/results', exist_ok=True)
# Upload a test image to inputs/ then run:
# !cd /content/Real-ESRGAN && python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test.jpg -o results

# Test CodeFormer
print("Testing CodeFormer...")
os.makedirs('/content/CodeFormer/inputs', exist_ok=True)
# Upload test images then run:
# !cd /content/CodeFormer && python inference_codeformer.py -w 0.7 --input_path inputs/

# Test FILM
print("Testing FILM...")
os.makedirs('/content/frame-interpolation/photos', exist_ok=True)
# Upload two images (one.png, two.png) then run:
# !cd /content/frame-interpolation && python3 -m eval.interpolator_test --frame1 photos/one.png --frame2 photos/two.png --model_path /content/pretrained_models/film_net/Style/saved_model --output_frame photos/output.png

# Test LivePortrait
print("Testing LivePortrait...")
# Use example files or upload your own:
# !cd /content/LivePortrait && python inference.py -s assets/examples/source/s6.jpg -d assets/examples/driving/d0.mp4

print("Ready for testing!")
```

## Advantages of Colab

✅ **Free GPU Access**: T4 GPU (16GB VRAM) - much faster than CPU  
✅ **Pre-installed Libraries**: Many packages already available  
✅ **Easy File Management**: Upload/download via UI  
✅ **No Local Setup**: Everything runs in the cloud  
✅ **Shareable**: Easy to share with others  

## Important Notes

1. **Session Limits**: Free Colab sessions timeout after ~12 hours
2. **Storage**: ~80GB available, but model weights take ~5-10GB
3. **Download Results**: Use Colab's file browser or `files.download()`
4. **GPU Availability**: May need to wait for GPU allocation during peak times

## Quick Test Commands

After setup, you can test each model:

```bash
# Real-ESRGAN - Image upscaling
cd /content/Real-ESRGAN
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test.jpg -o results

# CodeFormer - Face restoration
cd /content/CodeFormer
python inference_codeformer.py -w 0.7 --input_path inputs/

# FILM - Frame interpolation
cd /content/frame-interpolation
python3 -m eval.interpolator_test \
  --frame1 photos/one.png \
  --frame2 photos/two.png \
  --model_path /content/pretrained_models/film_net/Style/saved_model \
  --output_frame photos/output.png

# LivePortrait - Portrait animation
cd /content/LivePortrait
python inference.py -s source.jpg -d driving_video.mp4

# Real-ESRGAN - Video upscaling
cd /content/Real-ESRGAN
python inference_realesrgan_video.py -i inputs/video/test.mp4 -o results
```

## Download Results from Colab

```python
from google.colab import files

# Download a file
files.download('/content/Real-ESRGAN/results/output.jpg')

# Or use the file browser: Files → right-click → Download
```
