from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import shutil
import tempfile
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

app = FastAPI()


OUTPUT_DIR = Path("processed_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = "model.pt"  
model = YOLO(MODEL_PATH)

def process_video(input_path: str, output_path: str):
    """Process video using trained YOLO model and save output."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video")
        return

    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        results = model(frame)


        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                label = f"{model.names[class_id]}: {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)  

    cap.release()
    out.release()
    print(f"✅ Processed video saved at: {output_path}")

@app.post("/predict/")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload video and process it asynchronously using the trained model."""
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_video.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = OUTPUT_DIR / file.filename  
    background_tasks.add_task(process_video, temp_video.name, str(output_path))

    return {"message": f"Video {file.filename} is being processed.", "download_url": f"/download/{file.filename}"}

@app.get("/download/{video_name}")
async def download(video_name: str):
    """Download the processed video."""
    video_path = OUTPUT_DIR / video_name
    if not video_path.exists():
        return {"error": "File not found"}
    return FileResponse(video_path, media_type="video/mp4", filename=video_name)
