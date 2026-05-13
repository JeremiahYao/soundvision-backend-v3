"""
main.py — SoundVision v2
Upgraded: threaded TTS (pyttsx3 / gTTS fallback), adaptive frame-skip,
          full tracker integration, rich HUD overlay, graceful shutdown,
          performance metrics, and clean single-pass pipeline.
"""

import cv2
import sys
import time
import threading
import queue
import argparse
import numpy as np
from pathlib import Path

from detector   import Detector
from tracker    import ObjectTracker
from spatial    import SpatialAnalyzer
from risk_engine import RiskEngine, Severity
from guidance   import GuidanceSystem, GuidanceOutput


# ── TTS Engine (optional) ─────────────────────────────────────────────────────

class TTSEngine:
    """
    Non-blocking text-to-speech.
    Attempts pyttsx3 (offline), falls back to printing (silent mode).
    Speech runs in a daemon thread so it never blocks the video pipeline.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=3)
        self._engine = None
        self._available = False

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 160)
            self._available = True
            print("[TTS] pyttsx3 engine ready.")
        except Exception:
            print("[TTS] pyttsx3 unavailable — silent mode.")

        worker = threading.Thread(target=self._run, daemon=True)
        worker.start()

    def speak(self, text: str):
        """Enqueue a phrase (drops if queue full to avoid build-up)."""
        if not text:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass  # Older phrases get dropped — recency > completeness

    def _run(self):
        while True:
            text = self._queue.get()
            if self._available:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    pass
            else:
                print(f"[AUDIO] {text}")


# ── HUD Rendering ─────────────────────────────────────────────────────────────

# Severity → BGR colour
SEV_COLOUR = {
    Severity.CRITICAL: (0,   0,   255),
    Severity.HIGH:     (0,   100, 255),
    Severity.MEDIUM:   (0,   200, 255),
    Severity.LOW:      (0,   220, 180),
    Severity.CLEAR:    (0,   220,  80),
}

def _draw_bbox(frame, obj: dict, severity: str):
    x1, y1, x2, y2 = obj["bbox"]
    colour = SEV_COLOUR.get(severity, (180, 180, 180))
    thick  = 3 if severity in (Severity.CRITICAL, Severity.HIGH) else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)

    # Label tag above box
    label = f"{obj['object']} {obj.get('distance_m', 0):.1f}m"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), colour, -1)
    cv2.putText(frame, label, (x1+2, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)


def _draw_hud(frame, guidance: GuidanceOutput, fps: float, frame_id: int,
              all_threats: list, width: int, height: int):
    """Render the full semi-transparent HUD overlay."""

    sev    = guidance.severity
    colour = SEV_COLOUR.get(sev, (200, 200, 200))

    # ── Top banner ────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, guidance.hud, (12, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.80, colour, 1, cv2.LINE_AA)

    # ── FPS / frame counter (top-right) ───────────────────────────────────
    fps_text = f"FPS {fps:.1f}  |  frame {frame_id}"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, fps_text, (width - fw - 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Secondary threats (bottom strip) ──────────────────────────────────
    if len(all_threats) > 1:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, height-38), (width, height), (0,0,0), -1)
        cv2.addWeighted(overlay2, 0.45, frame, 0.55, 0, frame)
        others = all_threats[1:4]   # show up to 3 secondary threats
        txt = "  |  ".join(
            f"{t.obj['object']} {t.obj.get('distance_m',0):.1f}m [{t.severity}]"
            for t in others
        )
        cv2.putText(frame, txt, (10, height-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1, cv2.LINE_AA)


# ── Adaptive Frame Skipping ───────────────────────────────────────────────────

class AdaptiveSkipper:
    """
    Dynamically adjusts AI inference frequency based on elapsed inference time.
    Keeps effective AI FPS between TARGET_LOW and TARGET_HIGH.
    """
    TARGET_HIGH = 12.0   # Hz — aim for 12 AI inferences/sec max
    TARGET_LOW  = 4.0    # Hz — minimum
    MIN_SKIP    = 1
    MAX_SKIP    = 10

    def __init__(self, video_fps: float):
        self.video_fps   = max(1.0, video_fps)
        self._skip       = max(self.MIN_SKIP, int(video_fps / self.TARGET_HIGH))
        self._last_t     = time.monotonic()

    def record_inference_time(self, elapsed_s: float):
        """Called after each AI inference block to adapt skip rate."""
        if elapsed_s <= 0:
            return
        actual_ai_fps = 1.0 / elapsed_s
        if actual_ai_fps < self.TARGET_LOW:
            self._skip = min(self._skip + 1, self.MAX_SKIP)
        elif actual_ai_fps > self.TARGET_HIGH and self._skip > self.MIN_SKIP:
            self._skip = max(self._skip - 1, self.MIN_SKIP)

    @property
    def skip(self) -> int:
        return self._skip


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_soundvision(video_path: str, output_name: str,
                    show_window: bool = False, tts_enabled: bool = True):
    """
    Full SoundVision v2 pipeline.

    Parameters
    ----------
    video_path   : Path to input video (or '0' for webcam)
    output_name  : Base name for output video file (saved to /content/)
    show_window  : Display live OpenCV window (requires display)
    tts_enabled  : Enable text-to-speech audio guidance
    """

    # ── Open source ───────────────────────────────────────────────────────
    src = 0 if video_path == "0" else video_path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    print(f"[Main] Source: {video_path}  |  {width}×{height} @ {fps_src:.1f} fps")

    # ── Output writer ─────────────────────────────────────────────────────
    out_path = f"/content/{output_name}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))

    # ── Components ────────────────────────────────────────────────────────
    detector = Detector(async_mode=True)
    tracker  = ObjectTracker()
    spatial  = SpatialAnalyzer()
    engine   = RiskEngine()
    guidance = GuidanceSystem()
    tts      = TTSEngine() if tts_enabled else None
    skipper  = AdaptiveSkipper(fps_src)

    # ── State ─────────────────────────────────────────────────────────────
    frame_id        = 0
    all_threats     = []
    current_risk    = None
    current_guide   = GuidanceOutput(speak=None, hud="Path clear.",
                                     severity=Severity.CLEAR, score=0)
    fps_display     = 0.0
    t_last_frame    = time.monotonic()

    print("[Main] Pipeline running. Press Q (if window open) to quit.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── AI inference block (adaptive skip) ────────────────────────
            if frame_id % skipper.skip == 0:
                t0 = time.monotonic()

                raw_dets   = detector.detect_async(frame)
                stable     = tracker.smooth_and_track(raw_dets)
                analyzed   = spatial.analyze(stable, width, height, frame_id)
                all_threats = engine.evaluate_all(analyzed)
                current_risk = all_threats[0] if all_threats else None
                current_guide = guidance.generate(current_risk, all_threats)

                if tts and current_guide.should_speak:
                    tts.speak(current_guide.speak)

                skipper.record_inference_time(time.monotonic() - t0)

            # ── Draw bounding boxes for ALL threats ───────────────────────
            for threat in all_threats:
                _draw_bbox(frame, threat.obj, threat.severity)

            # ── HUD overlay ───────────────────────────────────────────────
            dt = time.monotonic() - t_last_frame
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(dt, 1e-6))
            t_last_frame = time.monotonic()

            _draw_hud(frame, current_guide, fps_display, frame_id,
                      all_threats, width, height)

            # ── Write & display ───────────────────────────────────────────
            out.write(frame)

            if show_window:
                cv2.imshow("SoundVision v2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── Progress ──────────────────────────────────────────────────
            frame_id += 1
            if frame_id % 100 == 0:
                pct = f"{frame_id/total_f*100:.1f}%" if total_f > 0 else f"frame {frame_id}"
                print(f"  [{pct}] fps={fps_display:.1f}  skip={skipper.skip}"
                      f"  tracks={tracker.active_count}"
                      f"  threats={len(all_threats)}"
                      f"  top={current_guide.severity}")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        detector.stop()
        cap.release()
        out.release()
        if show_window:
            cv2.destroyAllWindows()
        print(f"\n[Main] Done. Output saved to: {out_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoundVision v2 — Pedestrian Collision Avoidance")
    parser.add_argument("video",        type=str,           help="Path to input video or '0' for webcam")
    parser.add_argument("output",       type=str,           help="Output file base name")
    parser.add_argument("--show",       action="store_true",help="Display live window")
    parser.add_argument("--no-tts",     action="store_true",help="Disable text-to-speech")
    args = parser.parse_args()

    run_soundvision(
        video_path=args.video,
        output_name=args.output,
        show_window=args.show,
        tts_enabled=not args.no_tts,
    )
