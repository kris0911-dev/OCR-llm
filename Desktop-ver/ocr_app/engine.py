"""OCR engine wrapper around PaddleOCR (PP-OCRv6).

This module has no GUI dependencies so it can be reused by other front-ends
(e.g. the planned Android version).

Language notes
--------------
PP-OCRv6 (medium / small) recognizes Simplified & Traditional Chinese, English,
Japanese and 46 Latin-script languages, but NOT Korean. For Korean we keep the
PP-OCRv6 text *detector* (language-agnostic) and pair it with PaddleOCR's
dedicated Korean recognizer, which also reads English letters and digits.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

TIERS = ("medium", "small", "tiny")
ENGINES = ("auto", "paddle", "onnxruntime", "transformers")

KOREAN_REC_MODEL = "korean_PP-OCRv5_mobile_rec"

# Bundled models (see download_models.py). Override with OCR_MODELS_DIR.
MODELS_DIR = Path(os.environ.get("OCR_MODELS_DIR",
                                 Path(__file__).resolve().parent.parent / "models"))
_WEIGHTS_FILE = {"paddle": "inference.pdiparams", "onnxruntime": "inference.onnx"}


def local_model_dir(name: str, engine: str) -> str | None:
    """Bundled folder for ``name`` if it holds weights usable by ``engine``."""
    folder = MODELS_DIR / name
    weights = _WEIGHTS_FILE.get(engine)
    if weights and (folder / weights).is_file() and (folder / "inference.yml").is_file():
        return str(folder)
    return None


@dataclass(frozen=True)
class LanguageProfile:
    label: str
    # None -> use the PP-OCRv6 recognizer of the selected tier
    rec_model: str | None


LANGUAGES: dict[str, LanguageProfile] = {
    "korean": LanguageProfile("Korean (+English)", KOREAN_REC_MODEL),
    "multi": LanguageProfile("PP-OCRv6 multilingual (CN/EN/JP/Latin)", None),
}


def _paddle_cuda_count() -> int:
    """GPUs visible to a CUDA build of Paddle (paddlepaddle-gpu), else 0."""
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            return paddle.device.cuda.device_count()
    except Exception:  # broken CUDA install -> treat as CPU-only
        pass
    return 0


def _ort_cuda_available() -> bool:
    """True when onnxruntime-gpu is installed and its CUDA libraries load."""
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            # Loads CUDA/cuDNN from the nvidia-* pip wheels installed by
            # onnxruntime-gpu[cuda,cudnn]; without it ORT silently uses the CPU.
            ort.preload_dlls()
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _gpu_names() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


@lru_cache(maxsize=1)
def gpu_backends() -> tuple[str, ...]:
    """Backends that can run on CUDA in this environment, preferred first."""
    backends = []
    if _paddle_cuda_count():
        backends.append("paddle")
    if _ort_cuda_available():
        backends.append("onnxruntime")
    return tuple(backends)


@lru_cache(maxsize=1)
def available_devices() -> tuple[str, ...]:
    """e.g. ("gpu:0", "cpu"). GPUs come first so CUDA is the default."""
    if not gpu_backends():
        return ("cpu",)
    count = _paddle_cuda_count() or max(1, len(_gpu_names()))
    return tuple(f"gpu:{i}" for i in range(count)) + ("cpu",)


def default_device() -> str:
    return available_devices()[0]


def device_name(device: str) -> str:
    if device.startswith("gpu"):
        names = _gpu_names()
        index = int(device.split(":")[1])
        if index < len(names):
            return names[index]
    return device.upper()


def resolve_engine(engine: str, device: str) -> str:
    """Pick a concrete backend for ``engine="auto"``."""
    if engine != "auto":
        return engine
    if device.startswith("gpu") and gpu_backends():
        return gpu_backends()[0]
    return "paddle"


@dataclass
class OcrLine:
    text: str
    score: float
    box: np.ndarray  # (N, 2) int polygon


@dataclass
class OcrResult:
    lines: list[OcrLine] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class OcrEngine:
    def __init__(self, language: str = "korean", tier: str = "medium",
                 engine: str = "auto", device: str = "auto"):
        if language not in LANGUAGES:
            raise ValueError(f"Unknown language: {language}")
        if tier not in TIERS:
            raise ValueError(f"Unknown tier: {tier}")
        self.language = language
        self.tier = tier
        self.device = default_device() if device == "auto" else device
        self.engine = resolve_engine(engine, self.device)
        if self.device.startswith("gpu") and self.engine not in gpu_backends():
            raise RuntimeError(
                f"Backend '{self.engine}' has no CUDA support in this environment "
                f"(CUDA-capable backends: {', '.join(gpu_backends()) or 'none'}). "
                "Install requirements-gpu.txt, or choose the CPU device.")
        self._ocr = self._build()

    def _build(self):
        from paddleocr import PaddleOCR  # heavy import, keep lazy

        self.det_model = f"PP-OCRv6_{self.tier}_det"
        self.rec_model = LANGUAGES[self.language].rec_model or f"PP-OCRv6_{self.tier}_rec"
        kwargs = dict(
            text_detection_model_name=self.det_model,
            text_recognition_model_name=self.rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self.device,
        )
        # Prefer the bundled models (offline); otherwise PaddleOCR downloads them.
        for key, name in (("text_detection_model_dir", self.det_model),
                          ("text_recognition_model_dir", self.rec_model)):
            local = local_model_dir(name, self.engine)
            if local:
                kwargs[key] = local
        if self.engine == "paddle":
            # Paddle 3.x's oneDNN (MKL-DNN) CPU path crashes on PP-OCRv6 models
            # ("ConvertPirAttribute2RuntimeAttribute not support ...").
            kwargs["enable_mkldnn"] = False
        else:
            kwargs["engine"] = self.engine
        if self.engine == "onnxruntime":
            import onnxruntime as ort

            # Hide harmless warnings (unused initializers in the converted models).
            ort.set_default_logger_severity(3)
        return PaddleOCR(**kwargs)

    def recognize(self, image_bgr: np.ndarray, min_score: float = 0.5) -> OcrResult:
        result = OcrResult()
        start = time.perf_counter()
        for res in self._ocr.predict(image_bgr):
            texts = res["rec_texts"]
            scores = res["rec_scores"]
            polys = res["rec_polys"]
            for text, score, poly in zip(texts, scores, polys):
                if score >= min_score and text.strip():
                    result.lines.append(
                        OcrLine(text, float(score), np.asarray(poly, dtype=np.int32))
                    )
        result.lines = _reading_order(result.lines)
        result.seconds = time.perf_counter() - start
        return result


def _reading_order(lines: list[OcrLine]) -> list[OcrLine]:
    """Sort lines top-to-bottom, then left-to-right within the same row."""
    if not lines:
        return lines
    heights = [max(1, np.ptp(l.box[:, 1])) for l in lines]
    row_tol = float(np.median(heights)) * 0.5
    ordered = sorted(lines, key=lambda l: (l.box[:, 1].mean(), l.box[:, 0].min()))
    rows: list[list[OcrLine]] = []
    for line in ordered:
        cy = line.box[:, 1].mean()
        if rows and abs(rows[-1][0].box[:, 1].mean() - cy) <= row_tol:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [l for row in rows for l in sorted(row, key=lambda l: l.box[:, 0].min())]
