class GuidanceSystem:
    def generate(self, top_risk):
        score = top_risk["risk_score"]
        obj = top_risk["object"]
        
        # Determine side
        side = "left" if top_risk["center_x"] < 300 else "right"
        if top_risk["alignment"] > 0.8: side = "center"

        # Sophisticated Judgement Levels
        if score > 80:
            return f"EMERGENCY: {obj} CRITICAL {side.upper()}! STOP!"
        elif score > 40:
            return f"High Risk: {obj} moving toward {side}."
        elif score > 20:
            return f"Caution: {obj} on {side}."
        else:
            return "Path clear."
