class RiskEngine:
    def __init__(self):
        # Object Hazard Weights (Wo) from Slide 26
        self.weights = {
            "car": 5.0, "truck": 7.0, "bus": 7.0, 
            "motorcycle": 4.0, "person": 3.0, "bicycle": 2.0
        }
        self.alpha = 1.2 # Speed sensitivity constant
        self.beta = 0.5  # Alignment tuning parameter

    def evaluate(self, spatial_data, frame_width):
        if not spatial_data:
            return None

        results = []
        for obj in spatial_data:
            # 1. Wo (Object Weight)
            wo = self.weights.get(obj["object"], 1.0)

            # 2. Df (Distance Factor) - already calculated in spatial.py
            df = obj["df"]

            # 3. Sf (Speed Factor): Simulating Alpha * v / Vmax
            sf = self.alpha * 1.0 

            # Stage 1: Raw Threat = Wo * Df * Sf
            raw_threat = wo * df * sf

            # Stage 2: Final Risk (Alignment with User Path)
            # Center of frame (width/2) is the user's path
            alignment = 1.0 - (abs(obj["center_x"] - (frame_width/2)) / (frame_width/2))
            final_risk = raw_threat * (1 + self.beta * alignment)

            results.append({**obj, "risk_score": final_risk})

        # Sort to find the highest risk
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results[0]
