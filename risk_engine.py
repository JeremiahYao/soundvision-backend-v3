import math

class RiskEngine:
    def __init__(self):
        # Specific weights based on kinetic energy/lethality
        self.weights = {
            "bus": 15.0, "truck": 15.0, "car": 10.0, 
            "motorcycle": 8.0, "bicycle": 5.0, "person": 4.0
        }
        self.beta = 2.5 # Extremely high penalty for being in the center path

    def evaluate(self, spatial_data):
        if not spatial_data: return None

        for obj in spatial_data:
            wo = self.weights.get(obj["object"], 1.0)
            
            # EXPONENTIAL distance score: 10^prox
            # This makes a "near" object significantly more weighted than "medium"
            dist_score = 10 ** obj["proximity"]
            
            # CUBIC alignment: If it's not in your path, ignore it.
            # If it IS in your path, the risk triples.
            path_score = 1 + (self.beta * (obj["alignment"] ** 3))
            
            obj["risk_score"] = wo * dist_score * path_score * obj["speed_multiplier"]

        spatial_data.sort(key=lambda x: x["risk_score"], reverse=True)
        return spatial_data[0]
