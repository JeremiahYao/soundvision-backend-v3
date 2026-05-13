import math

class RiskEngine:
    def __init__(self):
        self.weights = {"car": 15, "truck": 20, "bus": 20, "person": 5}

    def evaluate(self, spatial_data):
        if not spatial_data: return None

        scored = []
        for obj in spatial_data:
            w = self.weights.get(obj["object"], 2)
            
            # Distance + Alignment + Future Prediction
            # Note: Alignment^4 kills 'side noise' almost entirely
            score = (w * math.exp(obj["proximity"] * 3)) * (obj["alignment"] ** 4) * obj["ttc_factor"]
            
            if score > 15: # Critical threshold to avoid "chatty" AI
                scored.append({**obj, "risk_score": score})

        if not scored: return None
        return sorted(scored, key=lambda x: x["risk_score"], reverse=True)[0]
