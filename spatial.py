import numpy as np

class SpatialAnalyzer:
    def analyze(self, detections, frame_width):
        analyzed_data = []
        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            
            # Distance Factor (Df): Estimated by the height of the box
            # A larger box height means the object is closer
            height = y2 - y1
            df = height / 100.0  

            # Direction: Based on the horizontal center of the box
            center_x = (x1 + x2) / 2
            if center_x < frame_width / 3:
                direction = "left"
            elif center_x > (2 * frame_width) / 3:
                direction = "right"
            else:
                direction = "ahead"

            analyzed_data.append({
                **obj,
                "df": df,
                "direction": direction,
                "center_x": center_x
            })
        return analyzed_data
