import math

class RiskEngine:
    def __init__(self):
        # Professional Hazard Weights
        self.weights = {
            "car": 8.0, "truck": 10.0, "bus": 10.0, 
            "motorcycle": 7.0, "person": 4.0, "bicycle": 3.0
        }
        self.beta = 1.5 # Boosts center-path threats significantly

    def evaluate(self, spatial_data):
        if not spatial_data: return None

        for obj in spatial_data:
            wo = self.weights.get(obj["object"], 1.0)
            
            # The Sophisticated Math:
            # 1. Exponential proximity: Risk doubles as it gets closer to the bottom
            dist_score = math.exp(obj["proximity"] * 2) 
            
            # 2. Square Alignment: Only focus heavily on what is DIRECTLY in front
            align_score = 1 + (self.beta * (obj["alignment"] ** 2))
            
            # 3. Combine with Speed Factor
            obj["risk_score"] = wo * dist_score * align_score * obj["speed_factor"]

        spatial_data.sort(key=lambda x: x["risk_score"], reverse=True)
        return spatial_data[0]
