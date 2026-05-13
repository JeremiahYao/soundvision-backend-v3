class SpatialAnalyzer:
    def analyze(self, detections, frame_width, frame_height):
        analyzed_data = []
        img_area = frame_width * frame_height

        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            obj_area = (x2 - x1) * (y2 - y1)
            
            # Distance Factor (Df): What % of the screen does the object take?
            # 0.0 (tiny/far) to 1.0 (huge/crashing into you)
            df = min(obj_area / (img_area * 0.5), 1.0) 

            # Direction: 0.0 (dead center) to 1.0 (far edge)
            center_x = (x1 + x2) / 2
            offset = abs(center_x - (frame_width / 2)) / (frame_width / 2)
            
            direction = "ahead"
            if center_x < frame_width * 0.35: direction = "left"
            elif center_x > frame_width * 0.65: direction = "right"

            analyzed_data.append({
                **obj,
                "df": df,
                "alignment": 1.0 - offset, # 1.0 is directly in front
                "direction": direction
            })
        return analyzed_data
