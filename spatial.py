class SpatialAnalyzer:
    def __init__(self):
        self.history = {}

    def analyze(self, detections, frame_width, frame_height, frame_id):
        analyzed_data = []
        # The Vanishing Point (usually center-horizon)
        vp_x, vp_y = frame_width / 2, frame_height * 0.4

        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 3D Depth Calculation: Distance from the Vanishing Point
            # Objects far from the VP and close to the bottom are the highest priority
            depth_factor = (y2 - vp_y) / (frame_height - vp_y)
            depth_factor = max(0, depth_factor) ** 2 # Quadratic curve for 3D depth

            # Horizontal 'Lane' Logic
            # Objects outside the 40% center lane are heavily discounted
            lane_offset = abs(cx - vp_x) / (frame_width / 2)
            in_lane_score = max(0, 1.0 - (lane_offset / 0.4)) 

            analyzed_data.append({
                **obj,
                "depth_factor": depth_factor,
                "in_lane_score": in_lane_score,
                "cy": cy
            })
        return analyzed_data
