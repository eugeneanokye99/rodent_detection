from fastapi import FastAPI, Request
import uvicorn
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
model = YOLO("./runs/detect/bird_detector_v1/weights/best.pt")

@app.post("/detect")
async def detect(request: Request):
    data = await request.body()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    conf = 0.0
    if len(results[0].boxes) > 0:
        conf = float(results[0].boxes.conf.max().cpu().numpy())

    return str(conf)

if name == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)