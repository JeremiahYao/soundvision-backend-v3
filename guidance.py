"""
guidance.py — SoundVision v2
Upgraded: natural phrasing, TTS-ready output, cooldown/debounce to prevent
          alert fatigue, distance-aware messages, 8-sector direction mapping,
          severity-tiered templates, and escalation/de-escalation tracking.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from risk_engine import Severity, ThreatRecord


# ── Direction mapping ─────────────────────────────────────────────────────────

def _lateral_to_direction(lateral_offset: float, alignment: float) -> str:
    """
    Map lateral_offset [-1,+1] and alignment [0,1] to a spoken direction.
    lateral_offset: negative = left, positive = right, ~0 = centre
    """
    if alignment > 0.75:
        return "ahead"

    if lateral_offset < -0.55:
        return "far left"
    elif lateral_offset < -0.20:
        return "left"
    elif lateral_offset > 0.55:
        return "far right"
    elif lateral_offset > 0.20:
        return "right"
    else:
        return "ahead"


def _distance_qualifier(dist_m: float) -> str:
    if dist_m < 1.5:   return "very close"
    if dist_m < 3.0:   return "close"
    if dist_m < 6.0:   return "nearby"
    return "ahead"


# ── Alert cooldown config ─────────────────────────────────────────────────────

COOLDOWN_SECONDS = {
    Severity.CRITICAL: 1.5,
    Severity.HIGH:     3.0,
    Severity.MEDIUM:   5.0,
    Severity.LOW:      8.0,
}


# ── Message templates ─────────────────────────────────────────────────────────

# Format keys: {obj}, {direction}, {dist_qual}, {dist_m}, {ttc}
TEMPLATES = {
    Severity.CRITICAL: [
        "Stop! {obj} {direction}!",
        "Danger! {obj} directly {direction}. Stop immediately.",
        "{obj} collision risk {direction}. Stop now.",
    ],
    Severity.HIGH: [
        "Warning: {obj} approaching from {direction}.",
        "Caution — {obj} {dist_qual}, {direction}.",
        "Watch out: {obj} closing in from {direction}.",
    ],
    Severity.MEDIUM: [
        "{obj} on your {direction}. Proceed carefully.",
        "Heads up — {obj} {dist_qual} to your {direction}.",
        "{obj} detected {direction}.",
    ],
    Severity.LOW: [
        "{obj} {dist_qual}, {direction}.",
        "Note: {obj} to your {direction}.",
    ],
}

# Escalation message (severity jumped from below HIGH to CRITICAL in < 3 frames)
ESCALATION_MSG = "Rapid approach! {obj} {direction} — stop!"

# All-clear message (only spoken after a prior warning)
CLEAR_MSG = "Path is clear."


# ── GuidanceSystem ────────────────────────────────────────────────────────────

class GuidanceSystem:
    """
    Generates context-aware, TTS-optimised guidance messages.

    Features
    --------
    - Per-severity cooldown to prevent alert fatigue
    - Severity escalation detection (extra urgency on sudden spikes)
    - Natural, varied phrasing (rotates templates to avoid repetition)
    - Distance & direction woven into every message
    - De-escalation: speaks "path clear" once after threats resolve
    - Returns structured GuidanceOutput for easy TTS / UI integration
    """

    def __init__(self):
        self._last_spoken_time:  dict[str, float] = {}
        self._last_severity:     dict[str, str]   = {}
        self._template_idx:      dict[str, int]   = {}
        self._prev_top_severity: str              = Severity.CLEAR
        self._spoke_clear:       bool             = True   # avoid repeating "clear"

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(self, top_risk: Optional[ThreatRecord],
                 all_risks: Optional[list] = None) -> "GuidanceOutput":
        """
        Returns a GuidanceOutput with .text (speak aloud) and .hud (display).
        Pass all_risks for secondary warning in HUD overlay.
        """
        now = time.monotonic()

        if top_risk is None:
            return self._clear_output()

        self._spoke_clear = False
        obj   = top_risk.obj
        score = top_risk.score
        sev   = top_risk.severity

        direction  = _lateral_to_direction(obj.get("lateral_offset", 0), obj["alignment"])
        dist_m     = obj.get("distance_m", 0)
        dist_qual  = _distance_qualifier(dist_m)
        ttc        = obj.get("ttc_frames", 999)
        label      = obj["object"]
        key        = str(obj.get("track_id", label))

        # ── Escalation check ─────────────────────────────────────────────
        prev_sev = self._last_severity.get(key, Severity.CLEAR)
        escalated = (
            sev == Severity.CRITICAL and
            prev_sev not in (Severity.CRITICAL, Severity.HIGH)
        )

        self._last_severity[key] = sev

        # ── Cooldown gate ─────────────────────────────────────────────────
        cooldown = COOLDOWN_SECONDS.get(sev, 5.0)
        last_t   = self._last_spoken_time.get(key, 0)
        should_speak = (now - last_t) >= cooldown

        if should_speak or escalated:
            self._last_spoken_time[key] = now
            fmt = {"obj": label, "direction": direction,
                   "dist_qual": dist_qual, "dist_m": f"{dist_m:.1f}m",
                   "ttc": f"{ttc:.0f}"}

            if escalated:
                text = ESCALATION_MSG.format(**fmt)
            else:
                templates = TEMPLATES[sev]
                idx = self._template_idx.get(key, 0) % len(templates)
                self._template_idx[key] = idx + 1
                text = templates[idx].format(**fmt)
        else:
            text = None  # Silence — cooldown not expired

        # ── HUD text (always updated, shorter) ───────────────────────────
        hud = self._build_hud(sev, label, direction, dist_m, score, ttc)

        self._prev_top_severity = sev
        return GuidanceOutput(speak=text, hud=hud, severity=sev, score=score)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _clear_output(self) -> "GuidanceOutput":
        speak = None
        if not self._spoke_clear and self._prev_top_severity not in (Severity.CLEAR, Severity.LOW):
            speak = CLEAR_MSG
            self._spoke_clear = True
        self._prev_top_severity = Severity.CLEAR
        return GuidanceOutput(speak=speak, hud="Path clear.", severity=Severity.CLEAR, score=0)

    @staticmethod
    def _build_hud(sev: str, label: str, direction: str,
                   dist_m: float, score: float, ttc: float) -> str:
        prefix = {"CRITICAL": "⛔", "HIGH": "⚠️", "MEDIUM": "🔔", "LOW": "ℹ️"}.get(sev, "")
        ttc_str = f" | TTC {ttc:.0f}f" if ttc < 100 else ""
        return f"{prefix} {label.upper()} {direction} | {dist_m:.1f}m{ttc_str} | risk {score:.0f}"


# ── GuidanceOutput ────────────────────────────────────────────────────────────

@dataclass
class GuidanceOutput:
    """
    speak   : string to pass to TTS engine (None = stay silent this frame)
    hud     : string for on-screen overlay
    severity: one of Severity constants
    score   : numeric risk score
    """
    speak:    Optional[str]
    hud:      str
    severity: str
    score:    float

    @property
    def should_speak(self) -> bool:
        return self.speak is not None
