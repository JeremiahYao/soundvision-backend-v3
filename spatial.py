import numpy as np

class SpatialAnalyzer:
    def __init__(self):
        self.tracking_history = {} # Keeps track of object motion vectors

    def analyze(self, detections, width, height, frame_id):
        # 1. Vanishing Point (VP) - The 'Perfection' Anchor
        # Most cameras place the horizon at 40-50% height.
        vp_y = height * 0.42 
        
        analyzed_data = []
        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 2. Ground-Plane Distance (Non-Linear)
            # Objects near the bottom (height) are infinitely closer than the VP.
            proximity = (y2 - vp_y) / (height - vp_y)
            proximity = max(0, min(1.0, proximity)) ** 2 # Quadratic for depth
            
            # 3. Dynamic Walking Corridor (The 3D Trapezoid)
            # The 'Danger Zone' narrows as it gets further away.
            center_line = width / 2
            # Allowable lane width: 25% at horizon, 70% at feet.
            lane_width = (width * 0.25) + (proximity * (width * 0.45))
            dist_from_center = abs(cx - center_line)
            
            # 1.0 = dead center, 0.0 = completely outside your lane.
            alignment = max(0, 1.0 - (dist_from_center / (lane_width / 2)))

            # 4. Momentum (Velocity Tracking)
            obj_id = f"{obj['object']}_{int(x1 / 30)}" # Simple temporal ID
            velocity_boost = 1.0
            if obj_id in self.tracking_history:
                prev_y2 = self.tracking_history[obj_id]
                # If it's moving DOWN the screen, it's getting CLOSER.
                if y2 > prev_y2 + 3: velocity_boost = 1.6 
                elif y2 < prev_y2 - 3: velocity_boost = 0.4 # Moving away
            
            self.tracking_history[obj_id] = y2
            
            analyzed_data.append({
                **obj,
                "proximity": proximity,
                "alignment": alignment,
                "velocity_boost": velocity_boost,
                "center_x": cx
            })
        return analyzed_data
