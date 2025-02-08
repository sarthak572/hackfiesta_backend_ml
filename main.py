from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import shutil
import tempfile
from pathlib import Path

app = FastAPI()

OUTPUT_DIR = Path("processed_videos")  # Ensure the processed directory exists
OUTPUT_DIR.mkdir(exist_ok=True)        # Create it if not exists

def process_video(input_path, output_path):
    """Simulate video processing (replace with YOLO processing)"""
    import time
    time.sleep(10)  # Simulate processing time
    shutil.copy(input_path, output_path)  # Mock processing by copying file
    print(f"✅ Processed video saved at: {output_path}")

@app.post("/predict/")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload video and process it asynchronously."""
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_video.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = OUTPUT_DIR / file.filename  # Save processed video with same name
    background_tasks.add_task(process_video, temp_video.name, output_path)

    return {"message": f"Video {file.filename} is being processed.", "download_url": f"/download/{file.filename}"}

@app.get("/download/{video_name}")
async def download(video_name: str):
    """Download the processed video."""
    video_path = OUTPUT_DIR / video_name
    if not video_path.exists():
        return {"error": "File not found"}
    return FileResponse(video_path, media_type="video/mp4", filename=video_name)
