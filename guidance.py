class GuidanceSystem:
    def generate(self, top_risk):
        score = top_risk["risk_score"]
        obj = top_risk["object"]
        
        # Spatial orientation
        if top_risk["alignment"] > 0.7:
            loc = "directly ahead"
        elif top_risk["center_x"] < 300:
            loc = "on your left"
        else:
            loc = "on your right"

        # Judgement logic
        if score > 150:
            return f"CRITICAL! {obj.upper()} {loc.upper()}! STOP NOW!"
        elif score > 70:
            return f"Warning: {obj} approaching {loc}."
        elif score > 30:
            return f"Note: {obj} {loc}."
        else:
            return "Path clear."
