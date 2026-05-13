from ultralytics import YOLO

class Detector:
    def __init__(self):
        # Load the most optimized model
        self.model = YOLO('yolov8n.pt') 
        # Only care about these specific COCO classes (0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)
        self.valid_classes = [0, 2, 3, 5, 7] 

    def detect(self, frame):
        # conf=0.5: If the AI isn't at least 50% sure, ignore it.
        # iou=0.45: Helps merge overlapping boxes.
        results = self.model(frame, conf=0.5, iou=0.45, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.valid_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "object": results.names[cls_id],
                    "conf": float(box.conf[0])
                })
        return detections
