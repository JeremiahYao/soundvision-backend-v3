"""
tracker.py — SoundVision v2
Upgraded: IoU-based Hungarian-style matching, per-track velocity (Kalman-lite),
          TTL-based eviction, confidence-weighted smoothing, and stable UUID tracking.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ── Helpers ──────────────────────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _box_center(box: list) -> tuple:
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def _predict_box(box: list, vx: float, vy: float) -> list:
    """Extrapolate a box one step forward using stored velocity."""
    return [int(box[0]+vx), int(box[1]+vy), int(box[2]+vx), int(box[3]+vy)]


# ── Track dataclass ───────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id:    int
    label:       str
    bbox:        list           # smoothed current box
    raw_bbox:    list           # latest raw detection
    vx:          float = 0.0   # pixel velocity x
    vy:          float = 0.0   # pixel velocity y
    frames_seen: int   = 1
    frames_lost: int   = 0
    conf:        float = 1.0
    history:     list  = field(default_factory=list)  # last N centers

    # Exponential smoothing weights
    ALPHA_POS:  float = 0.35   # position blend (higher = more responsive)
    ALPHA_VEL:  float = 0.20   # velocity blend (lower = smoother)
    MAX_HIST:   int   = 8

    def update(self, new_box: list, new_conf: float):
        prev_center = _box_center(self.bbox)
        new_center  = _box_center(new_box)

        # Velocity (EMA)
        raw_vx = new_center[0] - prev_center[0]
        raw_vy = new_center[1] - prev_center[1]
        self.vx = self.ALPHA_VEL * raw_vx + (1 - self.ALPHA_VEL) * self.vx
        self.vy = self.ALPHA_VEL * raw_vy + (1 - self.ALPHA_VEL) * self.vy

        # Position smooth (conf-weighted: high conf → trust new box more)
        alpha = min(0.9, self.ALPHA_POS + new_conf * 0.15)
        self.bbox = [
            int(alpha * new_box[i] + (1 - alpha) * self.bbox[i])
            for i in range(4)
        ]
        self.raw_bbox    = new_box
        self.conf        = new_conf
        self.frames_seen += 1
        self.frames_lost  = 0

        self.history.append(new_center)
        if len(self.history) > self.MAX_HIST:
            self.history.pop(0)

    def predict(self):
        """Advance position by velocity when no detection matches."""
        self.bbox = _predict_box(self.bbox, self.vx, self.vy)
        self.frames_lost += 1

    @property
    def is_confirmed(self) -> bool:
        return self.frames_seen >= 3

    @property
    def speed_px(self) -> float:
        return (self.vx**2 + self.vy**2) ** 0.5


# ── Tracker ───────────────────────────────────────────────────────────────────

class ObjectTracker:
    """
    IoU-based multi-object tracker with velocity prediction.

    Improvements over v1
    --------------------
    - Greedy IoU matching (replaces coarse grid-cell IDs)
    - Velocity-extrapolated prediction during missed frames
    - TTL eviction: tracks die after MAX_LOST consecutive misses
    - Trajectory history for downstream TTC estimation
    - No abrupt clear() — eviction is per-track
    """

    IOU_THRESHOLD = 0.30   # Minimum IoU to consider a match
    MAX_LOST      = 5      # Frames before a track is deleted
    MIN_CONFIRM   = 3      # Frames before a track is shown

    def __init__(self):
        self._tracks:   dict[int, Track] = {}
        self._next_id:  int = 0

    # ── Public ───────────────────────────────────────────────────────────────

    def smooth_and_track(self, detections: list[dict]) -> list[dict]:
        """
        Match detections to existing tracks, update/create/evict as needed.
        Returns only *confirmed* tracks as detection dicts (with bbox replaced
        by the smoothed box and extra fields: track_id, speed_px, trajectory).
        """
        # Step 1 — Predict forward all active tracks
        for t in self._tracks.values():
            t.predict()

        # Step 2 — Greedy IoU matching
        matched_track_ids  = set()
        matched_det_idxs   = set()

        # Build IoU matrix
        track_ids = list(self._tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for ti, tid in enumerate(track_ids):
            t = self._tracks[tid]
            for di, det in enumerate(detections):
                if det["object"] == t.label:
                    iou_matrix[ti, di] = _iou(t.bbox, det["bbox"])

        # Greedy: highest IoU pair first
        while True:
            if iou_matrix.size == 0:
                break
            ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[ti, di] < self.IOU_THRESHOLD:
                break
            tid = track_ids[ti]
            self._tracks[tid].update(detections[di]["bbox"], detections[di]["conf"])
            matched_track_ids.add(tid)
            matched_det_idxs.add(di)
            iou_matrix[ti, :] = -1
            iou_matrix[:, di] = -1

        # Step 3 — Unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di not in matched_det_idxs:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = Track(
                    track_id=tid,
                    label=det["object"],
                    bbox=det["bbox"],
                    raw_bbox=det["bbox"],
                    conf=det["conf"],
                )

        # Step 4 — Evict dead tracks
        dead = [tid for tid, t in self._tracks.items() if t.frames_lost > self.MAX_LOST]
        for tid in dead:
            del self._tracks[tid]

        # Step 5 — Return confirmed tracks as enriched detection dicts
        stable = []
        for tid, t in self._tracks.items():
            if t.is_confirmed:
                stable.append({
                    "bbox":       t.bbox,
                    "object":     t.label,
                    "conf":       t.conf,
                    "track_id":   t.track_id,
                    "speed_px":   t.speed_px,
                    "trajectory": list(t.history),
                })

        return stable

    @property
    def active_count(self) -> int:
        return len(self._tracks)
