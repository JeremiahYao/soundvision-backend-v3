from ultralytics import YOLO

class Detector:
    def __init__(self):
        # Load the pre-trained YOLOv8 nano model
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = float(box.conf[0])

            detections.append({
                "object": label,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": confidence
            })
        return detections
