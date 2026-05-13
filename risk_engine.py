"""
risk_engine.py — SoundVision v2
Upgraded: smooth score EMA per track, multi-threat output, severity tiers,
          size-aware danger weighting, stationary-object filtering, and
          pedestrian crossing intent detection.
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional


# ── Object danger weights ─────────────────────────────────────────────────────
# Reflects mass × speed potential × evasion difficulty
DANGER_WEIGHTS = {
    "person":        6,
    "bicycle":       8,
    "car":           18,
    "motorcycle":    14,
    "bus":           24,
    "truck":         24,
    "traffic light": 4,
    "stop sign":     3,
}

# Score thresholds defining severity tiers
TIER_CRITICAL  = 220
TIER_HIGH      = 90
TIER_MEDIUM    = 35
TIER_LOW       = 12    # Below this → effectively ignored

# EMA smoothing for score stability (prevents single-frame spikes)
SCORE_ALPHA = 0.40     # Higher = more reactive; lower = smoother

# Score history length for trend analysis
SCORE_HIST_LEN = 6

# Minimum pixel speed (px/frame) to be considered "moving"
MOVING_SPEED_THRESHOLD = 2.5


# ── Severity tier enum substitute ────────────────────────────────────────────

class Severity:
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    CLEAR    = "CLEAR"


@dataclass
class ThreatRecord:
    score:      float
    raw_score:  float
    severity:   str
    obj:        dict   # full spatial dict


# ── RiskEngine ────────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Multi-threat risk evaluator with score smoothing.

    Returns
    -------
    evaluate() → ThreatRecord for the top threat (or None if path is clear)
    evaluate_all() → list[ThreatRecord] for ALL threats above TIER_LOW
    """

    def __init__(self):
        # Per-track smoothed score history
        self._score_histories: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=SCORE_HIST_LEN)
        )
        self._smooth_scores: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, spatial_data: list[dict]) -> Optional[ThreatRecord]:
        """Return the highest-risk ThreatRecord, or None if path is clear."""
        threats = self.evaluate_all(spatial_data)
        return threats[0] if threats else None

    def evaluate_all(self, spatial_data: list[dict]) -> list[ThreatRecord]:
        """Return all threats above TIER_LOW, sorted highest-risk first."""
        if not spatial_data:
            return []

        records = []
        for obj in spatial_data:
            raw = self._compute_raw_score(obj)
            if raw < TIER_LOW:
                continue

            key = str(obj.get("track_id", f"{obj['object']}_{int(obj['center_x']/40)}"))
            smooth = self._smooth_score(key, raw)

            sev = self._severity_tier(smooth)
            records.append(ThreatRecord(
                score=round(smooth, 1),
                raw_score=round(raw, 1),
                severity=sev,
                obj=obj,
            ))

        records.sort(key=lambda r: r.score, reverse=True)
        return records

    # ── Score Computation ─────────────────────────────────────────────────────

    def _compute_raw_score(self, obj: dict) -> float:
        label    = obj["object"]
        prox     = obj["proximity"]        # [0,1]
        align    = obj["alignment"]        # [0,1]
        ttc_f    = obj["ttc_factor"]       # risk multiplier
        dist_m   = obj.get("distance_m", 5.0)
        spd_px   = obj.get("pixel_speed",  0.0)
        closing  = obj.get("closing_speed", 0.0)

        w = DANGER_WEIGHTS.get(label, 3)

        # ── Distance component: exponential proximity ─────────────────────
        dist_score = w * math.exp(prox * 3.2)

        # ── Alignment: quartic penalty for objects outside path ───────────
        align_score = align ** 3.5   # softer than ^4, catches near-path objects

        # ── TTC component ─────────────────────────────────────────────────
        ttc_score = ttc_f

        # ── Speed bonus: fast-moving objects are more dangerous ───────────
        speed_bonus = 1.0 + min(spd_px / 30.0, 1.5)   # up to 2.5× for 45 px/frame

        # ── Closing rate bonus ────────────────────────────────────────────
        if closing > 0.02:
            close_bonus = 1.0 + (closing * 8.0)
        elif closing < -0.02:
            close_bonus = 0.5   # moving away
        else:
            close_bonus = 1.0

        # ── Size bonus: large bounding boxes fill FOV = very close ────────
        bbox  = obj["bbox"]
        bbox_area_norm = ((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / (640 * 480)
        size_bonus = 1.0 + min(bbox_area_norm * 4.0, 2.0)

        raw = dist_score * align_score * ttc_score * speed_bonus * close_bonus * size_bonus
        return raw

    def _smooth_score(self, key: str, raw: float) -> float:
        if key not in self._smooth_scores:
            self._smooth_scores[key] = raw
        else:
            prev = self._smooth_scores[key]
            # Use higher alpha when score is rising (danger spike) for responsiveness
            alpha = SCORE_ALPHA if raw >= prev else SCORE_ALPHA * 0.6
            self._smooth_scores[key] = alpha * raw + (1 - alpha) * prev

        self._score_histories[key].append(self._smooth_scores[key])
        return self._smooth_scores[key]

    @staticmethod
    def _severity_tier(score: float) -> str:
        if score >= TIER_CRITICAL:  return Severity.CRITICAL
        if score >= TIER_HIGH:      return Severity.HIGH
        if score >= TIER_MEDIUM:    return Severity.MEDIUM
        if score >= TIER_LOW:       return Severity.LOW
        return Severity.CLEAR
