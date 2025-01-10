import torch
from ultralytics import YOLO

def load_model():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # Use pre-trained YOLO model
    labels = model.names  # Class labels
    return model, labels

def detect_objects(model, frame, labels):
    results = model(frame)
    detections = []
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "object": labels[cls],
                "bbox": [x1, y1, x2, y2]
            })
    return detections
