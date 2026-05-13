import numpy as np

class SpatialAnalyzer:
    def __init__(self):
        self.history = {} # Object tracking for velocity

    def analyze(self, detections, width, height, frame_id):
        analyzed = []
        # Define Horizon (40% down the screen)
        horizon = height * 0.4
        # Define the Walking Corridor (Trapezoid)
        # Base (at user feet) is 60% of width, Top (at horizon) is 20% of width
        
        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            cx = (x1 + x2) / 2
            
            # 1. Ground Distance (The 'Feet' logic)
            # 0.0 at horizon (far), 1.0 at bottom (instant danger)
            ground_proximity = (y2 - horizon) / (height - horizon)
            ground_proximity = max(0, min(1, ground_proximity))

            # 2. Corridor Check (Is it in our 3D path?)
            # As ground_proximity increases (closer), the allowed 'width' increases
            dynamic_lane_width = (width * 0.2) + (ground_proximity * (width * 0.4))
            dist_from_center = abs(cx - (width / 2))
            
            # 1.0 if inside the trapezoid, drops to 0.0 outside
            in_path_score = max(0, 1.0 - (dist_from_center / (dynamic_lane_width / 2)))

            # 3. Intent/Velocity (Is it getting closer?)
            obj_key = f"{obj['object']}_{int(cx/50)}"
            velocity_multiplier = 1.0
            area = (x2 - x1) * (y2 - y1)
            
            if obj_key in self.history:
                prev_area, prev_y2 = self.history[obj_key]
                # If area is growing OR feet are moving down the screen
                if area > prev_area * 1.05 or y2 > prev_y2 + 5:
                    velocity_multiplier = 1.5 # Approaching
                elif area < prev_area * 0.95:
                    velocity_multiplier = 0.5 # Retreating

            self.history[obj_key] = (area, y2)

            analyzed.append({
                **obj,
                "proximity": ground_proximity,
                "in_path": in_path_score,
                "velocity": velocity_multiplier,
                "center_x": cx
            })
        return analyzed
