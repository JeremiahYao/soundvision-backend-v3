import numpy as np

class SpatialAnalyzer:
    def __init__(self):
        self.history = {} # Used for temporal smoothing

    def analyze(self, detections, frame_width, frame_height, frame_id):
        analyzed_data = []
        # 1. Define the Vanishing Point (Horizon)
        vp_y = frame_height * 0.45 
        
        # 2. Define the 'Walking Corridor' (The Center Lane)
        # We only focus on the middle 35% of the camera's wide-angle view
        corridor_left = frame_width * 0.32
        corridor_right = frame_width * 0.68

        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            cx = (x1 + x2) / 2
            
            # 3. Perspective-Corrected Distance (Quadratic)
            # Higher values mean the object's feet (y2) are closer to the user
            norm_y = (y2 - vp_y) / (frame_height - vp_y)
            proximity = max(0, norm_y) ** 2

            # 4. Lane Alignment (The 'Smart' Filter)
            # If the object is outside the corridor, this value drops to near 0
            if cx < corridor_left:
                alignment = max(0, 1.0 - (abs(cx - corridor_left) / (frame_width * 0.2)))
            elif cx > corridor_right:
                alignment = max(0, 1.0 - (abs(cx - corridor_right) / (frame_width * 0.2)))
            else:
                alignment = 1.0 # Dead center in user's walking path

            # 5. Motion Trend (Smoothing)
            # We track if the box is growing (approaching) or shrinking (retreating)
            obj_key = f"{obj['object']}_{int(cx/100)}"
            speed_factor = 1.0
            if obj_key in self.history:
                prev_y2 = self.history[obj_key]
                if y2 > prev_y2 + 2: speed_factor = 1.3 # Approaching
                elif y2 < prev_y2 - 2: speed_factor = 0.6 # Retreating
            
            self.history[obj_key] = y2

            analyzed_data.append({
                **obj,
                "proximity": proximity,
                "alignment": alignment,
                "speed_factor": speed_factor,
                "center_x": cx
            })
            
        return analyzed_data
