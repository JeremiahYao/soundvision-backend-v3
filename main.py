import cv2
import os
from detector import Detector
from spatial import SpatialAnalyzer
from risk_engine import RiskEngine
from guidance import GuidanceSystem

def run_soundvision(video_path, output_name):
    if not os.path.exists(video_path):
        print(f"❌ Error: Input video '{video_path}' not found!")
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    output_path = f"/content/{output_name}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize components
    detector = Detector()
    spatial = SpatialAnalyzer()
    engine = RiskEngine()
    guidance = GuidanceSystem()

    print(f"🚀 Starting Sophisticated Analysis: {video_path}")

    frame_id = 0 # Added for speed tracking
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        # 1. Detect objects
        detections = detector.detect(frame)
        
        # 2. Spatial Analysis (Now passing height and frame_id)
        analyzed = spatial.analyze(detections, width, height, frame_id)
        
        # 3. Evaluate Risk
        top_risk = engine.evaluate(analyzed)
        
        # 4. Generate Guidance
        message = guidance.generate(top_risk) if top_risk else "Path clear."

        # Visual Overlay for testing
        if top_risk:
            x1, y1, x2, y2 = top_risk["bbox"]
            # Draw a box that gets thicker if risk is high
            thickness = 2 if top_risk["risk_score"] < 50 else 5
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
            cv2.putText(frame, f"{message} ({int(top_risk['risk_score'])})", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)
        frame_id += 1 # Increment frame counter
        
        if frame_id % 50 == 0:
            print(f"Processed {frame_id} frames...")

    cap.release()
    out.release()
    print(f"✅ Processing complete. Saved to {output_path}")
