from fastapi import FastAPI, Request
import numpy as np
import cv2
from ultralytics import YOLO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io

app = FastAPI()
model = YOLO("./runs/detect/bird_detector_v1/weights/best.pt")
executor = ThreadPoolExecutor(max_workers=2)  # Process multiple requests

def process_image(data):
    """Process image in thread pool"""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # FAST inference with smaller size
    results = model(img, imgsz=320, verbose=False)  # Smaller image size
    conf = 0.0
    if len(results[0].boxes) > 0:
        conf = float(results[0].boxes.conf.max().cpu().numpy())
    return conf

@app.post("/detect")
async def detect(request: Request):
    data = await request.body()
    
    # Process in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    confidence = await loop.run_in_executor(executor, process_image, data)
    
    return str(confidence)