import math

class RiskEngine:
    def __init__(self):
        # Weighted by Kinetic Energy (Lethality)
        self.weights = {
            "bus": 20, "truck": 20, "car": 12, 
            "motorcycle": 8, "bicycle": 5, "person": 3
        }

    def evaluate(self, spatial_data):
        if not spatial_data: return None

        for obj in spatial_data:
            wo = self.weights.get(obj["object"], 1.0)
            
            # Distance scales exponentially (the closer it is, the riskier it gets fast)
            dist_score = math.exp(obj["proximity"] * 3) 
            
            # Alignment is cubic: this is the "Secret Sauce" 
            # It kills the risk for anything not directly in front of the user
            path_multiplier = obj["alignment"] ** 3
            
            obj["risk_score"] = wo * dist_score * path_multiplier * obj["speed_factor"]

        # Only consider objects that are actually a threat (Noise Floor)
        valid_threats = [o for o in spatial_data if o["risk_score"] > 5.0]
        valid_threats.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return valid_threats[0] if valid_threats else None
