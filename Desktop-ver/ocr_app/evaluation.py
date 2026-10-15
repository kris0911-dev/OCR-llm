"""Accuracy / latency metrics and a synthetic Korean+English test set.

No GUI dependencies - reusable by the Android version.
"""
from __future__ import annotations

import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- metrics

def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or word lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TextScore:
    cer: float           # character error rate, whitespace collapsed
    cer_nospace: float   # CER ignoring all spaces (Korean spacing errors are common)
    wer: float           # word error rate
    exact: bool          # page text matches exactly (after normalization)


def score_text(prediction: str, truth: str) -> TextScore:
    pred, ref = normalize(prediction), normalize(truth)
    pred_ns, ref_ns = pred.replace(" ", ""), ref.replace(" ", "")
    ref_words = ref.split()
    return TextScore(
        cer=edit_distance(pred, ref) / max(1, len(ref)),
        cer_nospace=edit_distance(pred_ns, ref_ns) / max(1, len(ref_ns)),
        wer=edit_distance(pred.split(), ref_words) / max(1, len(ref_words)),
        exact=pred == ref,
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, q))


@dataclass
class RunStats:
    """Aggregated results for one (language, tier, backend, device) config."""
    config: dict
    load_s: float = 0.0
    warmup_s: float = 0.0
    latencies: list[float] = field(default_factory=list)
    scores: list[TextScore] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lat = [s * 1000 for s in self.latencies]
        has_truth = bool(self.scores)
        mean = lambda xs: statistics.fmean(xs) if xs else float("nan")
        return {
            **self.config,
            "images": len(self.latencies),
            "load_s": round(self.load_s, 2),
            "warmup_ms": round(self.warmup_s * 1000, 1),
            "mean_ms": round(mean(lat), 1),
            "p50_ms": round(percentile(lat, 50), 1),
            "p95_ms": round(percentile(lat, 95), 1),
            "img_per_s": round(1000 / mean(lat), 2) if lat else float("nan"),
            "cer_%": round(100 * mean([s.cer for s in self.scores]), 2) if has_truth else None,
            "cer_nospace_%": round(100 * mean([s.cer_nospace for s in self.scores]), 2) if has_truth else None,
            "wer_%": round(100 * mean([s.wer for s in self.scores]), 2) if has_truth else None,
            "exact_%": round(100 * mean([s.exact for s in self.scores]), 1) if has_truth else None,
            "mean_conf": round(mean(self.confidences), 3),
        }


# ---------------------------------------------------------------- datasets

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    name: str
    image: np.ndarray
    truth: str | None   # None -> latency only
    subset: str = "files"


def load_folder(folder: str | Path) -> list[Sample]:
    """Images in ``folder``; ground truth from a same-named UTF-8 ``.txt``."""
    samples = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        gt = path.with_suffix(".txt")
        truth = gt.read_text(encoding="utf-8") if gt.exists() else None
        samples.append(Sample(path.name, image, truth))
    return samples


def _common_hangul() -> list[str]:
    """The 2,350 common Hangul syllables (those in the KS X 1001 / EUC-KR set).

    Built from code points so the source stays ASCII-only. Rare syllables from
    the full U+AC00..U+D7A3 block are excluded: recognizers aren't trained on
    them and they would inflate the error rate unfairly.
    """
    # Python's codec writes the other syllables as 8-byte jamo sequences, so
    # "common" == encodes to exactly 2 bytes.
    return [chr(code) for code in range(0xAC00, 0xD7A4)
            if len(chr(code).encode("euc-kr")) == 2]


HANGUL = _common_hangul()
EN_WORDS = ["Invoice", "Total", "Order", "Model", "Scan", "Report", "PP-OCRv6", "GPU", "OCR"]


def korean_line(rng: random.Random) -> str:
    """A line of random Korean 'words' with occasional numbers / English words."""
    words = []
    for _ in range(rng.randint(3, 7)):
        roll = rng.random()
        if roll < 0.1:
            words.append(str(rng.randint(1, 99999)))
        elif roll < 0.18:
            words.append(rng.choice(EN_WORDS))
        else:
            words.append("".join(rng.choice(HANGUL) for _ in range(rng.randint(1, 4))))
    return " ".join(words)


EN_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Invoice No. INV-2026-0042 Total: $1,284.50",
    "PP-OCRv6 medium, small and tiny models",
    "Please scan the document and save as text.",
    "Meeting room B, 3rd floor, 10:30 AM",
]


def _find_font(font_path: str | None) -> str:
    candidates = [font_path] if font_path else []
    candidates += [
        r"C:\Windows\Fonts\malgun.ttf",                              # Windows
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",                # macOS
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",           # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("No Korean font found; pass --font path/to/font.ttf")


def _camera_like(page: np.ndarray, rng: random.Random) -> np.ndarray:
    """Place the page on a background with perspective, blur, noise, shading."""
    h, w = page.shape[:2]
    bw, bh = int(w * 1.35), int(h * 1.35)
    bg = np.full((bh, bw, 3), rng.randint(30, 90), np.uint8)
    jitter = lambda: rng.uniform(-0.05, 0.05)
    ox, oy = (bw - w) / 2, (bh - h) / 2
    dst = np.float32([
        [ox + w * jitter(), oy + h * jitter()],
        [ox + w + w * jitter(), oy + h * jitter()],
        [ox + w + w * jitter(), oy + h + h * jitter()],
        [ox + w * jitter(), oy + h + h * jitter()],
    ])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    out = cv2.warpPerspective(page, cv2.getPerspectiveTransform(src, dst), (bw, bh),
                              dst=bg, borderMode=cv2.BORDER_TRANSPARENT)
    shade = np.linspace(rng.uniform(0.75, 0.95), 1.0, bw, dtype=np.float32)
    out = (out.astype(np.float32) * shade[None, :, None])
    out += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 6, out.shape)
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0)
    return out


def synthetic_set(n: int, seed: int = 0, font_path: str | None = None,
                  camera: bool = True) -> list[Sample]:
    """``n`` rendered pages with known text (+ camera-like copies if ``camera``)."""
    from PIL import Image, ImageDraw, ImageFont

    font_file = _find_font(font_path)
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        size = rng.choice([16, 20, 24, 30, 36, 44])  # small sizes separate the tiers
        font = ImageFont.truetype(font_file, size)
        lines = [korean_line(rng) if rng.random() < 0.75 else rng.choice(EN_SENTENCES)
                 for _ in range(rng.randint(4, 10))]
        width = max(int(font.getlength(t)) for t in lines) + 120
        height = int(len(lines) * size * 1.9) + 100
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        for k, text in enumerate(lines):
            draw.text((60, 50 + k * size * 1.9), text, fill="black", font=font)
        page = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        truth = "\n".join(lines)
        samples.append(Sample(f"synth_{i:03d}_clean", page, truth, "clean"))
        if camera:
            samples.append(Sample(f"synth_{i:03d}_camera", _camera_like(page, rng),
                                  truth, "camera"))
    return samples
