class GuidanceSystem:
    def generate(self, top_risk):
        score = top_risk["risk_score"]
        obj = top_risk["object"]
        
        # Decide Side
        side = "Directly Ahead"
        if top_risk["alignment"] < 0.7:
            side = "Left" if top_risk["center_x"] < 320 else "Right"

        # Perfection Thresholds
        if score > 200:
            return f"CRITICAL: {obj.upper()} {side.upper()}! STOP NOW!"
        elif score > 80:
            return f"Warning: {obj} approaching from {side}."
        elif score > 30:
            return f"Note: {obj} on your {side}."
        
        return "Path clear."
