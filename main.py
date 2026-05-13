import cv2
from detector import Detector
from spatial import SpatialAnalyzer
from risk_engine import RiskEngine
from guidance import GuidanceSystem

def run_soundvision(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Initialize components
    detector = Detector()
    spatial = SpatialAnalyzer()
    engine = RiskEngine()
    guidance = GuidanceSystem()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Pipeline: Detect -> Analyze -> Evaluate -> Guide
        detections = detector.detect(frame)
        analyzed = spatial.analyze(detections, width)
        top_risk = engine.evaluate(analyzed, width)
        message = guidance.generate(top_risk)

        # Visual feedback for testing
        if top_risk:
            x1, y1, x2, y2 = top_risk["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processing complete. Saved to {output_path}")
