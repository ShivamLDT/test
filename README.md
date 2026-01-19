# AI Video Processing Pipeline

A comprehensive video processing pipeline that combines multiple AI models for face restoration, animation, upscaling, and frame interpolation.

## Features

- **Face Restoration**: Pre and post-processing with CodeFormer for high-quality face restoration
- **Portrait Animation**: Audio-driven animation using LivePortrait
- **Video Upscaling**: 2x upscaling with RealESRGAN
- **Frame Interpolation**: Smooth motion with FILM (Frame Interpolation for Large Motion)
- **REST API**: FastAPI-based API for easy integration

## Pipeline Stages

1. **Pre-processing (CodeFormer)**: Face restoration with fidelity 0.7
2. **Animation (LivePortrait)**: Audio-driven portrait animation
3. **Post-processing (CodeFormer)**: Final face restoration with fidelity 0.5
4. **Upscaling (RealESRGAN)**: 2x resolution enhancement
5. **Interpolation (FILM)**: Frame interpolation for smoother motion

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- FFmpeg (for video/audio processing)

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install fastapi uvicorn
pip install opencv-python numpy
pip install librosa soundfile
pip install tensorflow  # For FILM interpolation
```

### Model Setup

The pipeline expects the following model directories:

```
.
├── CodeFormer/
│   └── weights/
│       └── CodeFormer/
│           └── codeformer.pth
├── LivePortrait/
│   └── pretrained_weights/
│       └── liveportrait/
│           ├── base_models/
│           │   ├── appearance_feature_extractor.pth
│           │   ├── motion_extractor.pth
│           │   ├── warping_module.pth
│           │   └── spade_generator.pth
│           └── retargeting_models/
│               └── stitching_retargeting_module.pth
├── Real-ESRGAN/
│   └── weights/
│       └── RealESRGAN_x2plus.pth
└── frame-interpolation/
    └── pretrained_models/
        └── film_net/
            └── Style/
                └── saved_model/
```

Download the models from their respective repositories:
- [CodeFormer](https://github.com/sczhou/CodeFormer)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [FILM](https://github.com/google-research/frame-interpolation)

## Usage

### Python API

```python
from models.pipeline import VideoPipeline

# Initialize pipeline
pipeline = VideoPipeline(device='cuda')

# Process video
output_path = pipeline.process_video(
    input_video_path='input.mp4',
    audio_path='audio.wav',
    output_path='output.mp4',
    edit_instructions={
        'expression_scale': 1.2,
        'lip_intensity': 1.5
    }
)
```

### REST API

Start the API server:

```bash
python -m api.main
# or
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### Endpoints

**POST `/generate-video/`**
- Upload video and audio files
- Optional: `edit_instructions` as JSON string
- Returns processed video file

Example with curl:
```bash
curl -X POST "http://localhost:8000/generate-video/" \
  -F "video=@input.mp4" \
  -F "audio=@audio.wav" \
  -F 'edit_instructions={"expression_scale": 1.2, "lip_intensity": 1.5}'
```

**GET `/health`**
- Check API and GPU status

**GET `/`**
- API information and available endpoints

**GET `/docs`**
- Interactive API documentation (Swagger UI)

### Natural Language Prompts

Use the prompt parser to convert natural language to edit instructions:

```python
from api.prompt_parser import parse_and_validate

instructions = parse_and_validate("make the expression more intense, increase lip movement")
# Returns: {'expression_scale': 1.3, 'lip_intensity': 1.5}
```

Supported prompt patterns:
- "more/intense expression" → increases expression scale
- "less/subtle expression" → decreases expression scale
- "increase lip movement" → increases lip intensity
- "reduce lip movement" → decreases lip intensity
- Numeric values: "expression scale 1.5" → sets exact value

## Configuration

### Environment Variables

Create a `.env` file (optional):

```env
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0
TEMP_DIR=./temp
OUTPUT_DIR=./outputs
```

### Edit Instructions Format

```json
{
  "expression_scale": 1.2,    // 0.1 to 3.0, default: 1.0
  "lip_intensity": 1.5,        // 0.1 to 3.0, default: 1.0
  "head_pose": "horizontal"    // Optional: pose adjustments
}
```

## Project Structure

```
.
├── api/
│   ├── main.py              # FastAPI application
│   └── prompt_parser.py     # Natural language prompt parsing
├── models/
│   ├── pipeline.py          # Main pipeline
│   ├── codeformer_wrapper.py
│   ├── liveportrait_wrapper.py
│   ├── realesrgan_wrapper.py
│   └── film_wrapper.py
├── utils/
│   ├── video_utils.py       # Video I/O utilities
│   └── audio_utils.py        # Audio processing utilities
├── inputs/                   # Input files directory
├── outputs/                  # Output files directory
├── temp/                     # Temporary files (auto-cleaned)
└── README.md
```

## Performance

- **GPU Recommended**: Processing is significantly faster on GPU
- **Memory**: Requires ~8GB+ VRAM for full pipeline
- **Processing Time**: ~1-5 minutes per minute of video (depends on resolution and hardware)

## Troubleshooting

### CUDA Out of Memory
- Reduce video resolution
- Use smaller tile sizes in RealESRGAN
- Process shorter video segments

### Model Not Found Errors
- Verify all model files are in correct directories
- Check file paths in wrapper classes

### FFmpeg Errors
- Ensure FFmpeg is installed and in PATH
- Check audio/video codec compatibility

## License

This project integrates multiple open-source models. Please refer to individual model licenses:
- CodeFormer: MIT License
- LivePortrait: Apache 2.0
- Real-ESRGAN: BSD 3-Clause
- FILM: Apache 2.0

## Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Add error handling for edge cases
- Update documentation for new features

## Acknowledgments

- [CodeFormer](https://github.com/sczhou/CodeFormer) - Face restoration
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) - Portrait animation
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Image/video upscaling
- [FILM](https://github.com/google-research/frame-interpolation) - Frame interpolation
