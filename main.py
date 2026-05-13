import cv2
import os
from detector import Detector
from spatial import SpatialAnalyzer
from risk_engine import RiskEngine
from guidance import GuidanceSystem

def run_soundvision(video_path, output_name):
    # 1. Check if input video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Input video '{video_path}' not found!")
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # 2. Use a more universal codec (XVID) and .avi extension for reliability
    output_path = f"/content/{output_name}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ Error: VideoWriter failed to open. Check your output path/codec.")
        return

    # Initialize components
    detector = Detector()
    spatial = SpatialAnalyzer()
    engine = RiskEngine()
    guidance = GuidanceSystem()

    print(f"🚀 Starting processing: {video_path} -> {output_path}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        # Pipeline
        detections = detector.detect(frame)
        analyzed = spatial.analyze(detections, width)
        top_risk = engine.evaluate(analyzed, width)
        message = guidance.generate(top_risk)

        # Visual feedback
        if top_risk:
            x1, y1, x2, y2 = top_risk["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    
    if os.path.exists(output_path):
        print(f"✅ SUCCESS! File created at: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    else:
        print("❌ Unknown Error: Loop finished but file still does not exist.")
