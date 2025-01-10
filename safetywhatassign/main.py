import cv2
import json
import time
from model import load_model, detect_objects
from utils import create_json_output, crop_and_save, benchmark

def main(video_path):
    # Load model
    model, labels = load_model()
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    output_json = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Detect objects and sub-objects
        detections = detect_objects(model, frame, labels)
        
        # Generate JSON output
        json_data = create_json_output(detections)
        output_json.append(json_data)
        
        # Save sub-object images
        crop_and_save(detections, frame, frame_count)
        
        # Show frame (Optional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Benchmark FPS
        end_time = time.time()
        print(f"Frame {frame_count}: {1 / (end_time - start_time):.2f} FPS")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save JSON output
    with open('outputs/detections.json', 'w') as f:
        json.dump(output_json, f, indent=4)
    
    print("Processing complete. Outputs saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help="Path to the input video file")
    args = parser.parse_args()
    
    main(args.video)
