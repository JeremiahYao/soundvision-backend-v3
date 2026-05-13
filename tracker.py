class ObjectTracker:
    def __init__(self):
        self.active_objects = {} # ID -> {bbox, frames_seen}
        self.alpha = 0.3 # Smoothing factor (0.1 = very smooth/laggy, 0.9 = jerky/fast)

    def smooth_and_track(self, detections):
        # This simplifies the world by 'locking' onto the most consistent boxes
        stable_detections = []
        
        for det in detections:
            obj_name = det["object"]
            curr_box = det["bbox"]
            
            # Simple ID based on object type and general location
            obj_id = f"{obj_name}_{int(curr_box[0]/50)}_{int(curr_box[1]/50)}"
            
            if obj_id in self.active_objects:
                prev_box = self.active_objects[obj_id]["bbox"]
                # 1. Coordinate Smoothing (Exponential Moving Average)
                # Formula: $x_{smooth} = \alpha \cdot x_{new} + (1 - \alpha) \cdot x_{old}$
                smooth_box = [
                    int(self.alpha * curr_box[i] + (1 - self.alpha) * prev_box[i])
                    for i in range(4)
                ]
                self.active_objects[obj_id]["bbox"] = smooth_box
                self.active_objects[obj_id]["frames_seen"] += 1
            else:
                self.active_objects[obj_id] = {"bbox": curr_box, "frames_seen": 1}

            # 2. Persistence Check: Only show if seen for at least 3 frames
            if self.active_objects[obj_id]["frames_seen"] >= 3:
                det["bbox"] = self.active_objects[obj_id]["bbox"]
                stable_detections.append(det)

        # Cleanup old objects (simple version)
        if len(self.active_objects) > 20: self.active_objects.clear()
        
        return stable_detections
