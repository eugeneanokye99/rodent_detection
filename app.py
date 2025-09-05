from fastapi import FastAPI, Request
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

# Load your trained model
model = YOLO("./runs/detect/bird_detector_v17/weights/best.pt")   # Make sure best.pt is in the repo

@app.post("/detect")
async def detect(request: Request):
    # Read raw bytes from ESP32 POST
    data = await request.body()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    conf = 0.0
    if len(results[0].boxes) > 0:
        conf = float(results[0].boxes.conf.max().cpu().numpy())

    return str(conf)
