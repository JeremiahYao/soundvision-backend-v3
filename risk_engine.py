import math

class RiskEngine:
    def __init__(self):
        self.weights = {"car": 12, "truck": 15, "bus": 15, "person": 4, "bicycle": 6}

    def evaluate(self, spatial_data):
        if not spatial_data: return None

        scored_objects = []
        for obj in spatial_data:
            wo = self.weights.get(obj["object"], 2)
            
            # The "Perfect" Formula:
            # Risk = (Base Weight * e^Proximity) * PathWeight^3 * Velocity
            # Cubing PathWeight ensures objects slightly to the side (0.5) 
            # become negligible (0.125).
            
            raw_risk = wo * math.exp(obj["proximity"] * 2.5)
            path_weight = obj["in_path"] ** 3
            
            final_score = raw_risk * path_weight * obj["velocity"]
            
            # Noise Floor: Ignore everything below a certain score to keep it 'smart'
            if final_score > 8.0:
                scored_objects.append({**obj, "risk_score": final_score})

        if not scored_objects: return None
        
        # Sort and return the single most relevant threat
        scored_objects.sort(key=lambda x: x["risk_score"], reverse=True)
        return scored_objects[0]
