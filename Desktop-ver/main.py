import logging
import os
import sys

# Download models from Hugging Face (where PP-OCRv6 is published) and skip the
# slow multi-host connectivity probe. Override by setting these env vars.
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Silence OpenCV warnings while probing for cameras that don't exist.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from PySide6.QtWidgets import QApplication

from ocr_app.ui import MainWindow


def main():
    logging.getLogger("httpx").setLevel(logging.WARNING)  # per-file download logs
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
