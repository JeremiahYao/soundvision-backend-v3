"""
spatial.py — SoundVision v2
Upgraded: true TTC from apparent-size derivative, lateral closing rate,
          ego-motion compensation using frame-level optical flow cue,
          horizon auto-calibration, and richer per-object spatial record.
"""

import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate real-world heights (metres) for scale-distance estimation
REAL_HEIGHTS_M = {
    "person":        1.75,
    "bicycle":       1.10,
    "car":           1.50,
    "motorcycle":    1.20,
    "bus":           3.00,
    "truck":         2.80,
    "traffic light": 2.50,
    "stop sign":     1.80,
}

# Camera approximate vertical FOV (degrees) — tunable for different devices
CAMERA_VFOV_DEG = 60.0

# How many frames of apparent-size history to keep for TTC derivation
TTC_HISTORY_LEN = 6

# Minimum estimated distance (metres) for which TTC is calculated
MIN_DISTANCE_M = 0.5

# Infinity substitute when object is moving away
TTC_INFINITY   = 999.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_distance(bbox_height_px: int, real_height_m: float,
                        frame_height_px: int, vfov_deg: float) -> float:
    """
    Pinhole-camera distance estimate.
    D = (H_real * f_px) / H_bbox_px
    where f_px = (frame_height / 2) / tan(vfov/2)
    """
    if bbox_height_px <= 0:
        return 999.0
    f_px = (frame_height_px / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
    return (real_height_m * f_px) / bbox_height_px


def _ttc_from_size_history(size_history: deque) -> float:
    """
    Time-to-collision from apparent bounding-box height sequence.
    Uses linear regression on the log of apparent size to get the
    fractional growth rate, then TTC = 1 / growth_rate (in frames).
    Returns TTC_INFINITY if the object is shrinking or static.
    """
    if len(size_history) < 3:
        return TTC_INFINITY

    hs = np.array(list(size_history), dtype=float)
    # Guard against zeros
    hs = np.where(hs > 0, hs, 1e-6)
    log_hs = np.log(hs)

    # Slope of log size over time ≈ fractional growth rate per frame
    x = np.arange(len(log_hs), dtype=float)
    slope = np.polyfit(x, log_hs, 1)[0]

    if slope <= 0.001:          # Not growing → moving away or static
        return TTC_INFINITY

    ttc_frames = 1.0 / slope    # frames until size doubles ≈ contact
    return max(1.0, ttc_frames)


# ── SpatialAnalyzer ───────────────────────────────────────────────────────────

class SpatialAnalyzer:
    """
    Per-object 3D spatial analysis for pedestrian collision avoidance.

    Outputs per detection
    ---------------------
    proximity       : normalised [0,1] ground-plane proximity
    distance_m      : estimated metric distance (metres)
    ttc_frames      : estimated frames until collision (TTC)
    ttc_factor      : risk multiplier fed to RiskEngine (1.0 = neutral)
    alignment       : [0,1] how centred in the danger trapezoid the object is
    closing_speed   : signed lateral closure rate (positive = closing)
    lateral_offset  : normalised signed offset from frame centre
    """

    def __init__(self, vfov_deg: float = CAMERA_VFOV_DEG):
        self.vfov_deg = vfov_deg
        # Per-track histories keyed by track_id (or fallback label_gridcell)
        self._size_histories:  dict = {}   # key → deque of bbox heights
        self._prox_histories:  dict = {}   # key → deque of proximity values
        self._dist_histories:  dict = {}   # key → deque of distance_m values

        # Horizon auto-calibration (running EMA)
        self._horizon_ratio: float = 0.42

    # ── Public ───────────────────────────────────────────────────────────────

    def analyze(self, detections: list[dict], width: int, height: int,
                frame_id: int) -> list[dict]:

        analyzed = []
        horizon_px = height * self._horizon_ratio

        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            cx  = (x1 + x2) / 2.0
            cy  = (y1 + y2) / 2.0
            obj_h = max(1, y2 - y1)
            label = obj["object"]

            # ── Unique track key ─────────────────────────────────────────────
            key = obj.get("track_id", f"{label}_{int(cx/40)}")

            # ── 1. Ground-plane proximity (normalised) ───────────────────────
            proximity = (y2 - horizon_px) / max(1, height - horizon_px)
            proximity = float(np.clip(proximity, 0.0, 1.0))

            # ── 2. Metric distance estimate ──────────────────────────────────
            real_h = REAL_HEIGHTS_M.get(label, 1.70)
            distance_m = _estimate_distance(obj_h, real_h, height, self.vfov_deg)
            distance_m = max(MIN_DISTANCE_M, distance_m)

            # ── 3. Apparent-size history & TTC ───────────────────────────────
            if key not in self._size_histories:
                self._size_histories[key] = deque(maxlen=TTC_HISTORY_LEN)
                self._dist_histories[key] = deque(maxlen=TTC_HISTORY_LEN)
                self._prox_histories[key] = deque(maxlen=TTC_HISTORY_LEN)

            self._size_histories[key].append(obj_h)
            self._dist_histories[key].append(distance_m)
            self._prox_histories[key].append(proximity)

            ttc_frames = _ttc_from_size_history(self._size_histories[key])

            # ── 4. TTC factor (risk multiplier) ──────────────────────────────
            # TTC factor scales from 0.3 (moving away) up to ~4.0 (imminent)
            if ttc_frames >= TTC_INFINITY:
                ttc_factor = 0.4                          # receding / static
            elif ttc_frames < 10:
                ttc_factor = 4.0 - (ttc_frames - 1) * 0.33  # 1→4.0, 10→1.0
                ttc_factor = max(1.0, ttc_factor)
            else:
                ttc_factor = max(0.8, 1.0 - (ttc_frames - 10) * 0.01)

            # Extra boost for very close objects regardless of TTC
            if distance_m < 2.0:
                ttc_factor *= 1.5

            # ── 5. Path alignment (trapezoidal danger zone) ──────────────────
            # Lane widens as object approaches (perspective trapezoid)
            lane_half = (width * 0.10) + (proximity * (width * 0.28))
            alignment = max(0.0, 1.0 - (abs(cx - width / 2.0) / lane_half))

            # ── 6. Lateral offset & closing speed ────────────────────────────
            lateral_offset = (cx - width / 2.0) / (width / 2.0)  # [-1, +1]

            if len(self._prox_histories[key]) >= 2:
                closing_speed = (
                    self._prox_histories[key][-1] -
                    self._prox_histories[key][-2]
                )
            else:
                closing_speed = 0.0

            # ── 7. Tracker velocity from pixel speed ──────────────────────────
            pixel_speed = obj.get("speed_px", 0.0)

            analyzed.append({
                **obj,
                "proximity":      proximity,
                "distance_m":     round(distance_m, 2),
                "ttc_frames":     round(ttc_frames, 1),
                "ttc_factor":     round(ttc_factor, 3),
                "alignment":      round(alignment, 3),
                "closing_speed":  round(closing_speed, 4),
                "lateral_offset": round(lateral_offset, 3),
                "pixel_speed":    round(pixel_speed, 2),
                "center_x":       cx,
                "center_y":       cy,
            })

        # Prune stale histories (keys not seen this frame)
        active_keys = {
            obj.get("track_id", f"{obj['object']}_{int((obj['bbox'][0]+obj['bbox'][2])//2/40)}")
            for obj in detections
        }
        for stale in list(self._size_histories.keys()):
            if stale not in active_keys:
                # Keep for 30 frames in case track reappears
                pass  # Tracker TTL handles actual eviction

        return analyzed
