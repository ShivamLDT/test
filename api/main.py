"""FastAPI application for video processing pipeline."""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional
import traceback

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch

# Import pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.pipeline import VideoPipeline

# Initialize FastAPI app
app = FastAPI(
    title="AI Video Processing Pipeline API",
    description="API for processing videos through AI pipeline (CodeFormer, LivePortrait, RealESRGAN, FILM)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline at module level
print("Initializing VideoPipeline...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = VideoPipeline(device=device)
    PIPELINE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Failed to initialize pipeline: {e}")
    pipeline = None
    PIPELINE_AVAILABLE = False

# Setup temp directory
TEMP_DIR = Path(__file__).parent.parent / 'temp'
TEMP_DIR.mkdir(exist_ok=True)


def cleanup_temp_files(file_paths: list):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if isinstance(file_path, (str, Path)):
                path = Path(file_path)
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: Failed to cleanup {file_path}: {e}")


@app.get("/")
async def root():
    """API information and available endpoints."""
    return {
        "name": "AI Video Processing Pipeline API",
        "version": "1.0.0",
        "description": "Process videos through AI pipeline with face restoration, animation, upscaling, and interpolation",
        "endpoints": {
            "POST /generate-video/": {
                "description": "Process video through the complete AI pipeline",
                "parameters": {
                    "video": "Video file (UploadFile)",
                    "audio": "Audio file (UploadFile)",
                    "edit_instructions": "Optional JSON string with edit instructions (Form)"
                },
                "returns": "Processed video file"
            },
            "GET /health": {
                "description": "Health check endpoint",
                "returns": "Status information including GPU availability"
            },
            "GET /": {
                "description": "API information",
                "returns": "This information"
            }
        },
        "pipeline_status": "available" if PIPELINE_AVAILABLE else "unavailable"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    return {
        "status": "healthy" if PIPELINE_AVAILABLE else "degraded",
        "pipeline_available": PIPELINE_AVAILABLE,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "device": "cuda" if gpu_available else "cpu"
    }


@app.post("/generate-video/")
async def generate_video(
    video: UploadFile = File(..., description="Input video file"),
    audio: UploadFile = File(..., description="Audio file for animation"),
    edit_instructions: Optional[str] = Form(None, description="Optional JSON string with edit instructions")
):
    """
    Process video through the complete AI pipeline.
    
    Pipeline stages:
    1. Pre-process with CodeFormer (fidelity 0.7)
    2. Animate with LivePortrait
    3. Post-process with CodeFormer (fidelity 0.5)
    4. Upscale with RealESRGAN
    5. Interpolate with FILM
    
    Returns the processed video file.
    """
    if not PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Pipeline is not available. Check server logs for initialization errors."
        )
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Initialize file paths
    video_path = None
    audio_path = None
    output_path = None
    
    try:
        # Save uploaded files
        video_path = session_dir / f"input_video_{session_id}{Path(video.filename).suffix if video.filename else '.mp4'}"
        audio_path = session_dir / f"input_audio_{session_id}{Path(audio.filename).suffix if audio.filename else '.wav'}"
        output_path = session_dir / f"output_{session_id}.mp4"
        
        # Save video file
        print(f"Saving uploaded video to {video_path}")
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        # Save audio file
        print(f"Saving uploaded audio to {audio_path}")
        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Parse edit instructions if provided
        edit_params = None
        if edit_instructions:
            try:
                # Try parsing as JSON first
                edit_params = json.loads(edit_instructions)
                print(f"Edit instructions (JSON): {edit_params}")
            except json.JSONDecodeError:
                # If not valid JSON, try parsing as natural language prompt
                try:
                    from api.prompt_parser import parse_and_validate
                    edit_params = parse_and_validate(edit_instructions)
                    print(f"Edit instructions (parsed from prompt): {edit_params}")
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid edit_instructions format. Provide valid JSON or natural language prompt. Error: {e}"
                    )
        
        # Process video through pipeline
        print(f"Starting video processing for session {session_id}")
        result_path = pipeline.process_video(
            input_video_path=str(video_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            edit_instructions=edit_params
        )
        
        if not Path(result_path).exists():
            raise HTTPException(
                status_code=500,
                detail="Pipeline completed but output file was not created"
            )
        
        # Return the generated video
        return FileResponse(
            path=result_path,
            media_type="video/mp4",
            filename=f"processed_{session_id}.mp4",
            headers={
                "X-Session-ID": session_id,
                "Content-Disposition": f"attachment; filename=processed_{session_id}.mp4"
            }
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        error_trace = traceback.format_exc()
        print(f"Error: {error_msg}\n{error_trace}")
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )
    
    finally:
        # Note: Files are kept for download. Cleanup happens after delay.
        # In production, consider using a proper task queue (Celery, etc.)
        # or a cleanup service that runs periodically
        try:
            import threading
            import time
            
            def delayed_cleanup():
                # Wait 5 minutes before cleanup to allow file download
                time.sleep(300)
                cleanup_temp_files([session_dir])
                print(f"Cleaned up temp files for session {session_id}")
            
            cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
            cleanup_thread.start()
        except Exception as e:
            print(f"Warning: Failed to schedule cleanup: {e}")
            # Fallback: try immediate cleanup of input files only (keep output for now)
            try:
                files_to_clean = [f for f in [video_path, audio_path] if f is not None]
                if files_to_clean:
                    cleanup_temp_files(files_to_clean)
            except:
                pass


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
