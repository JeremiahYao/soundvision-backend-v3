import cv2
import os
from detector import Detector
from spatial import SpatialAnalyzer
from risk_engine import RiskEngine
from guidance import GuidanceSystem
from tracker import ObjectTracker # Import the new stabilizer

def run_soundvision(video_path, output_name):
    # ... setup code ...
    detector = Detector()
    tracker = ObjectTracker() # Initialize stabilizer
    spatial = SpatialAnalyzer()
    engine = RiskEngine()
    guidance = GuidanceSystem()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Raw Detections
        raw_detections = detector.detect(frame)
        
        # 2. STABILIZE: This stops the '1000 bounces per second'
        stable_detections = tracker.smooth_and_track(raw_detections)
        
        # 3. Analyze 3D Space
        analyzed = spatial.analyze(stable_detections, width, height, frame_id)
        
        # 4. Evaluate Top Threat
        top_risk = engine.evaluate(analyzed)
        
        # ... rest of your code ...
def run_soundvision(video_path, output_name):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    
    out = cv2.VideoWriter(f"/content/{output_name}.avi", 
                         cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # Initialize components
    detector = Detector()
    spatial = SpatialAnalyzer()
    engine = RiskEngine()
    guidance = GuidanceSystem()

    frame_id = 0
    # LAG FIX: Only run AI every 4 frames (Adjust this: 1 = slow/precise, 10 = fast/coarse)
    ai_step = 4 
    
    current_top_risk = None
    current_msg = "Path clear."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_id % ai_step == 0:
            # 1. Detection (The heavy part)
            detections = detector.detect(frame)
            # 2. Spatial Filtering (The perspective part)
            analyzed = spatial.analyze(detections, width, height, frame_id)
            # 3. Risk Engine (The decision part)
            current_top_risk = engine.evaluate(analyzed)
            # 4. Guidance (The human part)
            current_msg = guidance.generate(current_top_risk) if current_top_risk else "Path clear."

        # UI Overlay (Runs every frame for smoothness)
        if current_top_risk:
            x1, y1, x2, y2 = current_top_risk["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, current_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Display risk score for debugging
            cv2.putText(frame, f"Score: {int(current_top_risk['risk_score'])}", (50, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(frame)
        frame_id += 1
        if frame_id % 100 == 0: print(f"Processed {frame_id} frames...")

    cap.release()
    out.release()
