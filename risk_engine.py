import math

class RiskEngine:
    def __init__(self):
        # Lethality Weights (Car vs. Person)
        self.weights = {
            "bus": 25, "truck": 25, "car": 15, 
            "motorcycle": 10, "bicycle": 7, "person": 4
        }

    def evaluate(self, spatial_data):
        if not spatial_data: return None

        scored_list = []
        for obj in spatial_data:
            base_w = self.weights.get(obj["object"], 1.0)
            
            # EXPONENTIAL distance: Risk doubles every 10% it gets closer.
            dist_score = math.exp(obj["proximity"] * 3.5)
            
            # CUBIC alignment: This is the "Perfection" filter.
            # If a car is slightly to the side (0.5), risk drops to 0.125 (0.5^3).
            # This SILENCES parked cars and cars in other lanes.
            path_multiplier = obj["alignment"] ** 3
            
            final_score = base_w * dist_score * path_multiplier * obj["velocity_boost"]

            # Noise Filter: Only 'Smart' alerts
            if final_score > 12.0:
                scored_list.append({**obj, "risk_score": final_score})

        if not scored_list: return None
        scored_list.sort(key=lambda x: x["risk_score"], reverse=True)
        return scored_list[0]
