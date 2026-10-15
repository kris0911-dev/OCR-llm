"""Download all OCR models into ./models so the app runs fully offline.

Run once on a machine with internet; afterwards copy/commit the models folder.

  python download_models.py                  # everything (Paddle + ONNX, ~360 MB)
  python download_models.py --formats paddle # CPU only (~180 MB)
  python download_models.py --tiers medium   # only the medium tier (+ Korean)
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from ocr_app.engine import KOREAN_REC_MODEL, MODELS_DIR, TIERS

# Files to take from each repo. inference.yml/json come from the Paddle repo
# (the ONNX repos only add inference.onnx to the same folder).
PADDLE_FILES = ["inference.json", "inference.pdiparams", "inference.yml", "config.json"]
ONNX_FILES = ["inference.onnx"]
OPTIONAL = {"config.json"}  # only some repos have it


def model_names(tiers) -> list[str]:
    names = [f"PP-OCRv6_{t}_{kind}" for t in tiers for kind in ("det", "rec")]
    return names + [KOREAN_REC_MODEL]


def fetch(repo: str, filename: str, dest: Path) -> str:
    target = dest / filename
    if target.exists() and target.stat().st_size > 0:
        return "cached"
    try:
        path = hf_hub_download(repo, filename)  # resumable, uses the HF cache
    except Exception as exc:
        if filename in OPTIONAL:
            return "n/a"
        raise RuntimeError(f"{repo}/{filename}: {exc}") from exc
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    return f"{target.stat().st_size / 2**20:.1f} MB"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    p.add_argument("--formats", nargs="+", default=["paddle", "onnx"],
                   choices=["paddle", "onnx"])
    args = p.parse_args()

    for name in model_names(args.tiers):
        dest = MODELS_DIR / name
        jobs = []
        if "paddle" in args.formats:
            jobs += [(f"PaddlePaddle/{name}", f) for f in PADDLE_FILES]
        if "onnx" in args.formats:
            jobs += [(f"PaddlePaddle/{name}_onnx", f) for f in ONNX_FILES]
            if "paddle" not in args.formats:  # runner still needs the yml
                jobs.append((f"PaddlePaddle/{name}_onnx", "inference.yml"))
        for repo, filename in jobs:
            print(f"{name:30} {filename:22} {fetch(repo, filename, dest)}", flush=True)

    total = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    print(f"\nModels in {MODELS_DIR} ({total / 2**20:.0f} MB). The app now runs offline.")


if __name__ == "__main__":
    main()
