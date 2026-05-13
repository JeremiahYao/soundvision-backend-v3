import math

class RiskEngine:
    def __init__(self):
        self.weights = {"car": 10, "truck": 15, "bus": 15, "person": 5}

    def evaluate(self, spatial_data):
        if not spatial_data: return None

        for obj in spatial_data:
            wo = self.weights.get(obj["object"], 1)
            
            # Sophisticated Formula: 
            # Risk = (Weight * Depth^2) * (Lane_Alignment^3)
            # This ensures distant objects (low depth) or side objects (low lane) 
            # have their risk scores annihilated toward zero.
            
            proximity_score = obj["depth_factor"] * 10
            alignment_multiplier = obj["in_lane_score"] ** 3
            
            obj["risk_score"] = wo * proximity_score * alignment_multiplier

        # Filter: If risk is below a 'noise floor', ignore it completely
        valid_threats = [o for o in spatial_data if o["risk_score"] > 2.0]
        valid_threats.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return valid_threats[0] if valid_threats else None
