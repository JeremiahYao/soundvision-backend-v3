class GuidanceSystem:
    def generate(self, top_risk):
        if not top_risk: return "Path clear."
        
        score = top_risk["risk_score"]
        obj = top_risk["object"]
        
        # Actionable Direction
        side = "directly ahead"
        if top_risk["in_path"] < 0.8:
            side = "to your left" if top_risk["center_x"] < 320 else "to your right"

        # Judgement based on refined scores
        if score > 100:
            return f"STOP! {obj.upper()} {side.upper()}!"
        elif score > 50:
            return f"Caution: {obj} {side}."
        elif score > 20:
            return f"Note: {obj} moving {side}."
        
        return "Path clear."
