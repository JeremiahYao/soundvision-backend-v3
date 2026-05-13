class RiskEngine:
    def __init__(self):
        self.weights = {
            "car": 7.0, "truck": 10.0, "bus": 10.0, 
            "person": 4.0, "bicycle": 3.0, "dog": 2.0
        }
        self.beta = 0.8  # High alignment penalty

    def evaluate(self, spatial_data):
        if not spatial_data: return None

        results = []
        for obj in spatial_data:
            wo = self.weights.get(obj["object"], 1.0)
            df = obj["df"]
            sf = 1.2 # Constant for now, can be improved with tracking
            
            # Raw Threat: Wo * Df * Sf
            raw_threat = wo * df * sf
            
            # Final Risk: Raw * (1 + Beta * Alignment)
            # This makes objects directly in front much scarier than those on the side
            final_risk = raw_threat * (1 + self.beta * obj["alignment"])

            results.append({**obj, "risk_score": final_risk})

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results[0] if results else None
