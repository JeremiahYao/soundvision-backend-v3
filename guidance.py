class GuidanceSystem:
    def generate(self, top_risk):
        if not top_risk or top_risk["risk_score"] < 5.0:
            return "Path is clear."

        obj = top_risk["object"]
        direction = top_risk["direction"]
        
        # Priority warning logic based on risk score
        if top_risk["risk_score"] > 15:
            return f"DANGER! {obj} very close {direction}. STOP!"
        else:
            return f"Caution, {obj} {direction}."
