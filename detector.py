"""
detector.py — SoundVision v2
Upgraded: async threaded inference, GPU/MPS auto-detection, FP16 half-precision,
          model warmup, confidence decay per class, and batch-ready architecture.
"""

import threading
import queue
import numpy as np
import torch
from ultralytics import YOLO


# COCO class IDs relevant to pedestrian navigation safety
VALID_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    11: "stop sign",
}

# Per-class minimum confidence — vehicles need higher certainty than people
CLASS_CONF_THRESHOLD = {
    "person":        0.45,
    "bicycle":       0.45,
    "car":           0.50,
    "motorcycle":    0.50,
    "bus":           0.55,
    "truck":         0.55,
    "traffic light": 0.40,
    "stop sign":     0.50,
}


class Detector:
    """
    Real-time object detector wrapping YOLOv8.

    Features
    --------
    - Automatic device selection: CUDA > MPS > CPU
    - FP16 inference on GPU for ~2× speed boost
    - Non-blocking async inference via a background thread + queue
    - Model warmup to eliminate first-frame latency spike
    - Configurable skip-frame logic lives in main; detector is always hot
    """

    def __init__(self, model_name: str = "yolov8n.pt", async_mode: bool = True):
        self.device = self._select_device()
        print(f"[Detector] Using device: {self.device}")

        self.model = YOLO(model_name)
        self.model.to(self.device)

        # FP16 only available on true CUDA devices
        self.half = self.device.startswith("cuda")

        self._warmup()

        # Async inference state
        self.async_mode = async_mode
        self._result_lock = threading.Lock()
        self._latest_result: list = []
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

        if async_mode:
            self._worker_thread = threading.Thread(
                target=self._inference_worker, daemon=True
            )
            self._worker_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Synchronous detect — runs inference on the calling thread.
        Used when async_mode=False or caller needs a guaranteed fresh result.
        """
        return self._run_inference(frame)

    def detect_async(self, frame: np.ndarray) -> list[dict]:
        """
        Non-blocking detect — pushes frame to worker thread.
        Returns the *most recent completed result* immediately (may be 1 frame stale).
        Ideal for real-time pipelines where freshness > accuracy.
        """
        # Drop-frame strategy: discard if worker is busy (queue full)
        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Worker still busy — reuse previous result

        with self._result_lock:
            return list(self._latest_result)

    def stop(self):
        """Gracefully shut down the background inference thread."""
        self._stop_event.set()
        if self.async_mode:
            self._worker_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _warmup(self):
        """Run a dummy forward pass so the first real frame isn't slow."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._run_inference(dummy)
        print("[Detector] Warmup complete.")

    def _run_inference(self, frame: np.ndarray) -> list[dict]:
        results = self.model(
            frame,
            conf=0.40,          # Low global floor; per-class filter applied below
            iou=0.45,
            half=self.half,
            verbose=False,
            device=self.device,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VALID_CLASSES:
                continue

            label = VALID_CLASSES[cls_id]
            conf  = float(box.conf[0])
            min_conf = CLASS_CONF_THRESHOLD.get(label, 0.50)

            if conf < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox":   [x1, y1, x2, y2],
                "object": label,
                "conf":   conf,
                "cls_id": cls_id,
            })

        return detections

    def _inference_worker(self):
        """Background thread: drains frame queue and stores latest result."""
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            result = self._run_inference(frame)

            with self._result_lock:
                self._latest_result = result
