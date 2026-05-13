import numpy as np

class SpatialAnalyzer:
    def __init__(self):
        self.history = {} # Stores previous positions to calculate velocity

    def analyze(self, detections, frame_width, frame_height, frame_id):
        analyzed_data = []
        current_frame_objects = {}
        
        # Define the 'Ground' - things at the very bottom are at your feet
        horizon_line = frame_height * 0.4 

        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            obj_label = obj["object"]
            
            # 1. Perspective-Corrected Distance
            # Objects near the horizon are far; objects at the bottom are critical.
            # We use a quadratic scale because distance isn't linear in a 2D image.
            normalized_y = (y2 - horizon_line) / (frame_height - horizon_line)
            proximity = max(0, normalized_y) ** 2 

            # 2. Tracking Velocity (Change in Y over time)
            speed_multiplier = 1.0
            obj_key = f"{obj_label}_{int(x1/50)}" # Grouping objects by rough location
            
            if obj_key in self.history:
                prev_y2 = self.history[obj_key]
                delta_y = y2 - prev_y2
                if delta_y > 2: # Moving toward user
                    speed_multiplier = 1.0 + (delta_y / 5.0)
                elif delta_y < -2: # Moving away from user
                    speed_multiplier = 0.5

            current_frame_objects[obj_key] = y2
            
            # 3. Path Alignment (The 'Danger Zone')
            center_x = (x1 + x2) / 2
            # High value (1.0) if in the center 30% of the screen
            lane_width = frame_width * 0.3
            dist_from_center = abs(center_x - (frame_width / 2))
            alignment = max(0, 1.0 - (dist_from_center / lane_width))

            analyzed_data.append({
                **obj,
                "proximity": proximity,
                "speed_multiplier": speed_multiplier,
                "alignment": alignment,
                "center_x": center_x
            })
            
        self.history = current_frame_objects
        return analyzed_data
