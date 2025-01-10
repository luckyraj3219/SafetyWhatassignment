import cv2
import os

def create_json_output(detections):
    json_data = []
    for i, det in enumerate(detections):
        json_data.append({
            "object": det["object"],
            "id": i + 1,
            "bbox": det["bbox"],
            "subobject": {}
        })
    return json_data

def crop_and_save(detections, frame, frame_count):
    os.makedirs("cropped_images", exist_ok=True)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cropped_img = frame[y1:y2, x1:x2]
        cv2.imwrite(f"cropped_images/frame{frame_count}_obj{i+1}.png", cropped_img)

def benchmark():
    import time
    start_time = time.time()
    # Simulated task
    time.sleep(1)  
    end_time = time.time()
    print(f"Inference speed: {1 / (end_time - start_time):.2f} FPS")
