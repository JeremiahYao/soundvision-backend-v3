import numpy as np

class SpatialAnalyzer:
    def __init__(self):
        self.history = {} # ID -> {last_y2, last_area, timestamp}

    def analyze(self, detections, width, height, frame_id):
        analyzed = []
        horizon = height * 0.42
        
        # Assume average human walking speed 'pushes' the ground down by 2% per frame
        ego_motion_bias = 0.02 

        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            cx = (x1 + x2) / 2
            
            # 1. 3D Distance Proxy
            proximity = (y2 - horizon) / (height - horizon)
            proximity = max(0, min(1.0, proximity))

            # 2. Vector Prediction (The "Future" Part)
            obj_id = f"{obj['object']}_{int(cx/40)}"
            ttc_factor = 1.0 # 1.0 is neutral
            
            if obj_id in self.history:
                prev_prox = self.history[obj_id]['prox']
                # Delta is the change in proximity
                # We subtract ego_motion_bias because the world naturally 'comes at us'
                relative_closure = (proximity - prev_prox) - ego_motion_bias
                
                if relative_closure > 0.01:
                    # Object is closing distance FASTER than we are walking
                    # This is a high-speed approach or a collision course
                    ttc_factor = 1.0 + (relative_closure * 10)
                elif relative_closure < -0.01:
                    # Object is moving away (Target is safe)
                    ttc_factor = 0.3

            self.history[obj_id] = {'prox': proximity}

            # 3. Path Projection (Trapezoidal Danger Zone)
            # This factors in the user moving forward into a narrowing tunnel
            lane_width = (width * 0.2) + (proximity * (width * 0.5))
            alignment = max(0, 1.0 - (abs(cx - width/2) / (lane_width/2)))

            analyzed.append({
                **obj,
                "proximity": proximity,
                "alignment": alignment,
                "ttc_factor": ttc_factor,
                "center_x": cx
            })
        return analyzed
