"""PySide6 main window: camera scan / image file -> PP-OCRv6 -> text."""
from __future__ import annotations

import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QGuiApplication, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from .camera import CameraThread, list_cameras
from .engine import ENGINES, LANGUAGES, TIERS, OcrEngine, available_devices, device_name
from .scanner import draw_boxes, find_document, warp_document

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
AUTO_CAPTURE_FRAMES = 20     # page must be steady this many frames
AUTO_CAPTURE_TOLERANCE = 12  # max corner drift in pixels between frames


class Worker(QThread):
    """Runs a callable off the GUI thread."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, parent=None):
        super().__init__(parent)
        self.fn, self.args = fn, args

    def run(self):
        try:
            self.done.emit(self.fn(*self.args))
        except Exception:  # surface any engine error in the UI
            self.failed.emit(traceback.format_exc())


def bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def read_image(path: str) -> np.ndarray | None:
    # np.fromfile + imdecode handles non-ASCII (e.g. Korean) paths on Windows.
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class ImageView(QLabel):
    def __init__(self):
        super().__init__("Start the camera, open an image, or drop / paste one here")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#202124; color:#bbb; border-radius:6px;")
        self._pixmap: QPixmap | None = None

    def show_image(self, image: np.ndarray):
        self._pixmap = bgr_to_pixmap(image)
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PP-OCRv6 Scanner")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        self.engine: OcrEngine | None = None
        self.engine_key: tuple | None = None
        self.camera: CameraThread | None = None
        self.latest_frame: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.pending_image: np.ndarray | None = None
        self.worker: Worker | None = None
        self._frame_count = 0
        self._last_quad: np.ndarray | None = None
        self._steady = 0

        self._build_ui()
        self._refresh_cameras()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.view = ImageView()

        # --- camera controls
        self.camera_combo = QComboBox()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Re-scan for cameras")
        refresh_btn.clicked.connect(self._refresh_cameras)
        self.camera_btn = QPushButton("Start camera")
        self.camera_btn.clicked.connect(self._toggle_camera)
        self.scan_btn = QPushButton("Scan (Space)")
        self.scan_btn.setEnabled(False)
        self.scan_btn.clicked.connect(self._capture)
        self.scan_btn.setShortcut(Qt.Key_Space)
        open_btn = QPushButton("Open image...")
        open_btn.clicked.connect(self._open_image)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        cam_row.addWidget(self.camera_combo, 1)
        cam_row.addWidget(refresh_btn)
        cam_row.addWidget(self.camera_btn)
        cam_row.addWidget(self.scan_btn)
        cam_row.addWidget(open_btn)

        self.doc_check = QCheckBox("Detect page && flatten (document scan)")
        self.doc_check.setChecked(True)
        self.auto_check = QCheckBox("Auto-capture when page is steady")
        opt_row = QHBoxLayout()
        opt_row.addWidget(self.doc_check)
        opt_row.addWidget(self.auto_check)
        opt_row.addStretch()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.view, 1)
        left_layout.addLayout(cam_row)
        left_layout.addLayout(opt_row)

        # --- OCR settings
        self.lang_combo = QComboBox()
        for key, profile in LANGUAGES.items():
            self.lang_combo.addItem(profile.label, key)
        self.tier_combo = QComboBox()
        self.tier_combo.addItems(TIERS)
        self.tier_combo.setToolTip("PP-OCRv6 model size: medium = most accurate, tiny = fastest")
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(ENGINES)
        self.device_combo = QComboBox()
        for dev in available_devices():  # GPUs first, so CUDA is the default
            self.device_combo.addItem(f"{dev}  ({device_name(dev)})", dev)
        self.device_combo.setToolTip(
            "GPU entries appear when paddlepaddle-gpu (CUDA) is installed")
        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setSingleStep(0.05)
        self.min_score.setValue(0.5)

        settings = QGroupBox("OCR settings")
        form = QFormLayout(settings)
        form.addRow("Language:", self.lang_combo)
        form.addRow("Model tier:", self.tier_combo)
        form.addRow("Backend:", self.engine_combo)
        form.addRow("Device:", self.device_combo)
        form.addRow("Min confidence:", self.min_score)

        self.rerun_btn = QPushButton("Run OCR again")
        self.rerun_btn.setEnabled(False)
        self.rerun_btn.clicked.connect(lambda: self._run_ocr(self.current_image))

        # --- result
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Recognized text appears here...")
        font = self.text_edit.font()
        font.setPointSize(11)
        self.text_edit.setFont(font)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.text_edit.toPlainText()))
        save_btn = QPushButton("Save .txt...")
        save_btn.clicked.connect(self._save_text)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.rerun_btn)
        btn_row.addStretch()
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(save_btn)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(settings)
        right_layout.addWidget(self.text_edit, 1)
        right_layout.addLayout(btn_row)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        paste = QAction(self)
        paste.setShortcut(QKeySequence.Paste)
        paste.triggered.connect(self._paste_image)
        self.addAction(paste)

        self.statusBar().showMessage("Ready. OCR models download on first use.")

    # ------------------------------------------------------------ camera
    def _refresh_cameras(self):
        self.camera_combo.clear()
        cams = list_cameras()
        for idx in cams:
            self.camera_combo.addItem(f"Camera {idx}", idx)
        if not cams:
            self.camera_combo.addItem("No camera found", None)
        self.camera_btn.setEnabled(bool(cams))

    def _toggle_camera(self):
        if self.camera:
            self._stop_camera()
            return
        index = self.camera_combo.currentData()
        if index is None:
            return
        self.camera = CameraThread(index)
        self.camera.frame_ready.connect(self._on_frame)
        self.camera.error.connect(self._on_camera_error)
        self.camera.start()
        self.camera_btn.setText("Stop camera")
        self.scan_btn.setEnabled(True)
        self._steady = 0
        self.statusBar().showMessage("Camera running. Press Space to scan.")

    def _stop_camera(self):
        if self.camera:
            self.camera.stop()
            self.camera = None
        self.camera_btn.setText("Start camera")
        self.scan_btn.setEnabled(False)

    def _on_camera_error(self, msg: str):
        self._stop_camera()
        QMessageBox.warning(self, "Camera", msg)

    def _on_frame(self, frame: np.ndarray):
        self.latest_frame = frame
        self._frame_count += 1
        preview = frame
        if self.doc_check.isChecked():
            # Edge detection every 3rd frame keeps the preview smooth.
            if self._frame_count % 3 == 0:
                quad = find_document(frame)
                self._update_steadiness(quad)
                self._last_quad = quad
            if self._last_quad is not None:
                color = (0, 200, 0) if self._steady < AUTO_CAPTURE_FRAMES // 2 else (0, 220, 255)
                preview = draw_boxes(frame, [self._last_quad], color, 3)
        self.view.show_image(preview)

        if (self.auto_check.isChecked() and self.doc_check.isChecked()
                and self._steady >= AUTO_CAPTURE_FRAMES and not self._busy()):
            self._steady = 0
            self._capture()

    def _update_steadiness(self, quad):
        if quad is None or self._last_quad is None:
            self._steady = 0
            return
        drift = np.abs(np.sort(quad, axis=0) - np.sort(self._last_quad, axis=0)).max()
        self._steady = self._steady + 3 if drift < AUTO_CAPTURE_TOLERANCE else 0

    def _capture(self):
        if self.latest_frame is None:
            return
        frame = self.latest_frame.copy()
        if self.doc_check.isChecked():
            quad = find_document(frame)
            if quad is not None:
                frame = warp_document(frame, quad) ?: frame
        self._stop_camera()  # freeze on the captured shot
        self._run_ocr(frame)

    # ------------------------------------------------------------- input
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", IMAGE_FILTER)
        if path:
            self._load_path(path)

    def _load_path(self, path: str):
        image = read_image(path)
        if image is None:
            QMessageBox.warning(self, "Open image", f"Cannot read image:\n{path}")
            return
        self._stop_camera()
        self._run_ocr(image)

    def _paste_image(self):
        if self.text_edit.hasFocus():
            self.text_edit.paste()
            return
        qimg = QGuiApplication.clipboard().image()
        if qimg.isNull():
            return
        qimg = qimg.convertToFormat(QImage.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        arr = np.frombuffer(qimg.constBits(), np.uint8).reshape(h, qimg.bytesPerLine())
        rgb = arr[:, : w * 3].reshape(h, w, 3)
        self._stop_camera()
        self._run_ocr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self._load_path(urls[0].toLocalFile())

    # --------------------------------------------------------------- OCR
    def _busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _settings_key(self) -> tuple:
        return (self.lang_combo.currentData(), self.tier_combo.currentText(),
                self.engine_combo.currentText(), self.device_combo.currentData())

    def _run_ocr(self, image: np.ndarray | None):
        if image is None:
            return
        if self._busy():
            self.statusBar().showMessage("Still working on the previous image...")
            return
        self.current_image = image
        self.view.show_image(image)
        self.rerun_btn.setEnabled(False)

        key = self._settings_key()
        if self.engine is None or self.engine_key != key:
            self.statusBar().showMessage(
                "Loading OCR models (first run downloads them, please wait)...")
            self.pending_image = image
            self._start_worker(lambda: (key, OcrEngine(*key)), self._on_engine_ready)
        else:
            self._recognize(image)

    def _on_engine_ready(self, payload):
        self.engine_key, self.engine = payload
        image, self.pending_image = self.pending_image, None
        self._recognize(image)

    def _recognize(self, image: np.ndarray):
        self.statusBar().showMessage("Recognizing text...")
        self._start_worker(self.engine.recognize, self._on_result,
                           image, self.min_score.value())

    def _on_result(self, result):
        self.text_edit.setPlainText(result.text)
        self.view.show_image(draw_boxes(self.current_image, [l.box for l in result.lines]))
        avg = (sum(l.score for l in result.lines) / len(result.lines)) if result.lines else 0
        self.statusBar().showMessage(
            f"{len(result.lines)} lines recognized (avg confidence {avg:.2f}) "
            f"in {result.seconds:.2f}s on {self.engine.device} ({self.engine.engine})")
        self.rerun_btn.setEnabled(True)

    def _start_worker(self, fn, on_done, *args):
        self.worker = Worker(fn, *args, parent=self)
        self.worker.done.connect(on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_failed(self, tb: str):
        self.rerun_btn.setEnabled(self.current_image is not None)
        self.statusBar().showMessage("OCR failed")
        QMessageBox.critical(self, "OCR error", tb[-3000:])

    # -------------------------------------------------------------- misc
    def _save_text(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save text", "ocr.txt", "Text (*.txt)")
        if path:
            Path(path).write_text(self.text_edit.toPlainText(), encoding="utf-8")
            self.statusBar().showMessage(f"Saved {path}")

    def closeEvent(self, event):
        self._stop_camera()
        if self.worker:
            self.worker.wait(5000)
        super().closeEvent(event)
