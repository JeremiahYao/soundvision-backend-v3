import numpy as np

class SpatialAnalyzer:
    def __init__(self):
        self.history = {} # To track object movement over frames

    def analyze(self, detections, frame_width, frame_height, frame_id):
        analyzed_data = []
        current_frame_objects = {}

        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            obj_id = f"{obj['object']}_{x1}_{y1}" # Simple ID tracking
            
            # 1. 3D Distance Proxy (using 'y2' - bottom of the box)
            # Higher y2 = closer to the user's feet
            proximity = y2 / frame_height 
            
            # 2. Centeredness (Alignment)
            center_x = (x1 + x2) / 2
            # 0.0 at edges, 1.0 at dead center
            alignment = 1.0 - (abs(center_x - (frame_width / 2)) / (frame_width / 2))

            # 3. Velocity Estimation (Sf)
            # We compare position to the last frame to see if it's getting closer
            speed_factor = 1.0
            if obj_id in self.history:
                prev_y2 = self.history[obj_id]
                delta_y = y2 - prev_y2
                if delta_y > 0: # Object is moving "down" the screen (toward user)
                    speed_factor = 1.0 + (delta_y / 10) 

            current_frame_objects[obj_id] = y2
            
            analyzed_data.append({
                **obj,
                "proximity": proximity,
                "alignment": alignment,
                "speed_factor": speed_factor,
                "center_x": center_x
            })
        
        self.history = current_frame_objects
        return analyzed_data
