"""
utils/video_stream.py – Async-friendly webcam capture using OpenCV.

Provides:
    • ``VideoStream``  – threaded capture wrapper for low-latency reads
    • ``async_frame_generator`` – async generator yielding (timestamp, frame)
"""

import asyncio
import threading
import time
from typing import AsyncGenerator, Optional, Tuple

import cv2
import numpy as np

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class VideoStream:
    """Threaded OpenCV video capture.

    A background thread continuously grabs frames so the main / async loop
    never blocks on ``cap.read()``.
    """

    def __init__(
        self,
        src: int = config.WEBCAM_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ) -> None:
        self.src = src
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────
    def start(self) -> "VideoStream":
        """Open camera and begin background capture."""
        if self._running:
            return self

        self._cap = cv2.VideoCapture(self.src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {self.src}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("VideoStream started (src=%s, %dx%d)", self.src, self.width, self.height)
        return self

    def stop(self) -> None:
        """Signal the background thread to stop and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("VideoStream stopped")

    # ── internal ─────────────────────────────────────────────
    def _capture_loop(self) -> None:
        while self._running:
            if self._cap is None:
                break
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("Frame grab failed; retrying…")
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._timestamp = time.time()

    # ── public read ──────────────────────────────────────────
    def read(self) -> Tuple[float, Optional[np.ndarray]]:
        """Return (timestamp, frame) without blocking."""
        with self._lock:
            return self._timestamp, self._frame.copy() if self._frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running


# ── Async wrapper ────────────────────────────────────────────
async def async_frame_generator(
    stream: VideoStream,
    target_fps: int = config.FPS_TARGET,
) -> AsyncGenerator[Tuple[float, np.ndarray], None]:
    """Yield (timestamp, frame) at roughly *target_fps* via ``asyncio.sleep``."""
    interval = 1.0 / target_fps
    while stream.is_running:
        ts, frame = stream.read()
        if frame is not None:
            yield ts, frame
        await asyncio.sleep(interval)


# ── Mock stream for testing without a webcam ─────────────────
class MockVideoStream(VideoStream):
    """Generates synthetic noise frames.  Useful for CI / mock mode."""

    def start(self) -> "MockVideoStream":
        self._running = True
        logger.info("MockVideoStream started (no camera required)")
        return self

    def read(self) -> Tuple[float, Optional[np.ndarray]]:  # type: ignore[override]
        frame = np.random.randint(
            0, 255, (self.height, self.width, 3), dtype=np.uint8
        )
        return time.time(), frame

    def stop(self) -> None:
        self._running = False
        logger.info("MockVideoStream stopped")
