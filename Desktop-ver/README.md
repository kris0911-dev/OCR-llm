# PP-OCRv6 Scanner — Desktop version

Desktop OCR app (Python + PySide6) built on [PaddleOCR PP-OCRv6](https://huggingface.co/blog/paddlepaddle/pp-ocrv6).
Scan documents with a camera, or open, drop, or paste an image, and get editable text.

## Features
- **Camera scan**: live preview with page-edge detection, perspective flattening, and optional
  auto-capture when the page holds still. Press **Space** to scan.
- **Image input**: Open file, drag and drop, or **Ctrl+V** from the clipboard. Non-ASCII (Korean) file paths work.
- **Korean support** (see below), plus the PP-OCRv6 multilingual model for Chinese, English, Japanese and 46 Latin-script languages.
- Selectable model tier (medium / small / tiny), backend (Paddle / ONNX Runtime / Transformers) and device (**CUDA GPU** / CPU).
- Text boxes are drawn over the image. Text comes out in reading order and can be copied or saved as UTF-8 `.txt`.

## Korean
PP-OCRv6's recognizers do not cover Korean. In **Korean** mode the app uses the PP-OCRv6
*detector*, which works for any language, together with PaddleOCR's dedicated Korean recognizer
`korean_PP-OCRv5_mobile_rec`. That recognizer also reads English letters and digits, so mixed
Korean/English documents work.

## Setup
```powershell
cd Desktop-ver
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
### Models (offline)
All models live in `models/` inside this repo, and the app loads them from there with no network access:
```
models/
  PP-OCRv6_{medium,small,tiny}_det/   text detection (all languages)
  PP-OCRv6_{medium,small,tiny}_rec/   recognition: CN/EN/JP/Latin
  korean_PP-OCRv5_mobile_rec/         recognition: Korean + English
```
Each folder holds the Paddle weights (`inference.pdiparams`, used on CPU) and the ONNX weights
(`inference.onnx`, used on GPU), about 360 MB in total. They're stored with **Git LFS**
(see `.gitattributes`), so run `git lfs install` before cloning or committing.

To (re)populate the folder, run this on a machine with internet:
```powershell
python download_models.py                    # all tiers, Paddle + ONNX
python download_models.py --formats paddle   # CPU only, about half the size
```
If a model is missing from `models/`, the app falls back to downloading it from Hugging Face.
Set `OCR_MODELS_DIR` to use a different folder.

### NVIDIA GPU (CUDA)
```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\activate
pip install -r requirements-gpu.txt
python main.py
```
GPU inference uses ONNX Runtime's CUDA provider. The CUDA 12 and cuDNN 9 runtimes come in as pip
wheels, so you only need an NVIDIA driver, not the CUDA Toolkit. With **Backend: auto**, the app
uses the GPU when one is available. The status bar shows the device and backend for each scan.
Paddle's own CUDA build (`paddlepaddle-gpu`) also works if installed. It is not on PyPI for Windows; see `requirements-gpu.txt`.

## Model tiers
PP-OCRv6 comes in three sizes: **medium** (34.5M params, the default and most accurate),
**small** (7.7M) and **tiny** (1.5M, fastest). PaddlePaddle released no large or xlarge model.
Medium is the top tier, and it outperforms the older and larger PP-OCRv5_server.

In Korean mode, the tier changes only the text detector. Recognition always uses the Korean model.

## Evaluation (latency and accuracy)
```powershell
python evaluate.py                                   # synthetic KO/EN pages, all tiers, all devices
python evaluate.py --data my_scans --scan            # your images + same-name .txt ground truth
python evaluate.py --tiers medium tiny --devices gpu:0 cpu --synthetic 20 --out results.csv
```
Reports per tier, device and subset (clean pages vs camera-like photos):

| metric | meaning |
|---|---|
| `load_s`, `warmup_ms` | model load time and first inference (excluded from latency stats) |
| `mean_ms`, `p50_ms`, `p95_ms`, `img_per_s` | OCR latency per image (detection + recognition) |
| `cer_%` | character error rate vs ground truth (whitespace collapsed) |
| `cer_nospace_%` | CER ignoring spaces. Korean word-spacing errors are common and often harmless |
| `wer_%`, `exact_%` | word error rate, share of pages read exactly |
| `mean_conf` | mean recognizer confidence |

Measured on a laptop with an RTX 4070 Laptop GPU (10 Korean+English pages, Korean mode):

| tier | GPU (ONNX Runtime CUDA) | CPU (Paddle) | GPU speedup |
|---|---|---|---|
| medium | 262 ms / page (p95 319) | 13.9 s / page | ~53x |
| small | 192 ms / page (p95 225) | 4.8 s / page | ~25x |
| tiny | 203 ms / page (p95 240) | 4.1 s / page | ~20x |

On the GPU every tier takes about 0.2 s per page, so **medium is the right choice there**: you get the
best accuracy at almost no cost. On CPU, small and tiny are 3x faster than medium. Accuracy is the same on GPU and CPU.

The built-in synthetic set generates Korean lines from random common Hangul syllables (built
from Unicode code points, so the source has no Korean text). Random syllables have no language
context, and the recognizer tends to "correct" rare syllables to similar common ones. So its
Korean CER (about 25%) is a **pessimistic stress test** for comparing tiers and devices against each other.
Real sentences score far better (0% CER ignoring spaces in our tests). For real-world accuracy,
use `--data` with your own labeled camera photos.

## Layout
```
main.py              entry point
evaluate.py          latency / accuracy benchmark CLI
download_models.py   fetches all models into models/ (for offline use)
models/              bundled PP-OCRv6 + Korean models (Git LFS)
ocr_app/engine.py    OCR wrapper + device/backend selection (no GUI deps)
ocr_app/evaluation.py  CER/WER metrics, latency stats, synthetic test set
ocr_app/scanner.py   page detection / perspective warp (OpenCV only)
ocr_app/camera.py    camera capture thread
ocr_app/ui.py        PySide6 main window
```
The Android version lives in [`../Android-ver`](../Android-ver). It bundles the ONNX models from
`models/` and reimplements this pipeline in Kotlin with the same results.
