"""Background camera capture thread (OpenCV -> Qt signal)."""
from __future__ import annotations

import sys

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


def _backend() -> int:
    # DirectShow opens much faster than MSMF on Windows.
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def list_cameras(max_index: int = 5) -> list[int]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _backend())
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, index: int, width: int = 1920, height: int = 1080, parent=None):
        super().__init__(parent)
        self.index = index
        self.width = width
        self.height = height
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self.index}")
            return
        # Ask for a high resolution: small text needs pixels.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self._running = True
        failures = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 30:
                    self.error.emit("Camera stopped delivering frames")
                    break
                self.msleep(30)
                continue
            failures = 0
            self.frame_ready.emit(frame)
            self.msleep(15)
        cap.release()

    def stop(self):
        self._running = False
        self.wait(2000)
