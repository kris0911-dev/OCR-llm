"""Benchmark OCR latency and accuracy across model tiers / devices / backends.

Examples
--------
  # synthetic Korean+English pages (clean + camera-like), all tiers, all devices
  python evaluate.py

  # your own images: page1.jpg + page1.txt (UTF-8 ground truth), ...
  python evaluate.py --data path/to/folder --scan

  python evaluate.py --tiers medium --devices gpu:0 cpu --synthetic 20 --out results.csv
"""
import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from ocr_app.engine import ENGINES, LANGUAGES, TIERS, OcrEngine, available_devices
from ocr_app.evaluation import RunStats, load_folder, score_text, synthetic_set
from ocr_app.scanner import find_document, warp_document

COLUMNS = ["tier", "backend", "device", "subset", "images", "load_s", "warmup_ms",
           "mean_ms", "p50_ms", "p95_ms", "img_per_s", "cer_%", "cer_nospace_%",
           "wer_%", "exact_%", "mean_conf"]
CSV_COLUMNS = ["lang", *COLUMNS, "det_model", "rec_model"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="folder of images (+ same-name .txt ground truth)")
    p.add_argument("--synthetic", type=int, default=10,
                   help="number of synthetic pages when --data is not given (default 10)")
    p.add_argument("--no-camera", action="store_true",
                   help="synthetic: skip the camera-like (perspective/blur/noise) copies")
    p.add_argument("--scan", action="store_true",
                   help="--data: apply page detection + flattening before OCR, like the app")
    p.add_argument("--lang", default="korean", choices=LANGUAGES)
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    p.add_argument("--devices", nargs="+", default=None,
                   help=f"default: all available {list(available_devices())}")
    p.add_argument("--engine", default="auto", choices=ENGINES)
    p.add_argument("--runs", type=int, default=1, help="repeat each image N times for latency")
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--font", help="Korean TTF for synthetic pages")
    p.add_argument("--out", help="write results to .csv")
    return p.parse_args()


def prepare(sample, scan: bool):
    if scan or sample.subset == "camera":
        quad = find_document(sample.image)
        if quad is not None:
            warped = warp_document(sample.image, quad)
            if warped is not None:
                return warped
    return sample.image


def evaluate(config: dict, samples, args) -> dict[str, RunStats]:
    t = time.perf_counter()
    engine = OcrEngine(args.lang, config["tier"], args.engine, config["device"])
    load_s = time.perf_counter() - t
    config.update(backend=engine.engine, det_model=engine.det_model,
                  rec_model=engine.rec_model)
    print(f"  models: {engine.det_model} + {engine.rec_model}", flush=True)

    images = [prepare(s, args.scan) for s in samples]
    warmup = engine.recognize(images[0], args.min_score).seconds  # excluded from stats

    stats: dict[str, RunStats] = defaultdict(lambda: RunStats(dict(config)))
    for sample, image in zip(samples, images):
        for run in range(args.runs):
            result = engine.recognize(image, args.min_score)
            for key in (sample.subset, "ALL"):
                st = stats[key]
                st.latencies.append(result.seconds)
                if run == 0:
                    st.confidences += [l.score for l in result.lines]
                    if sample.truth is not None:
                        st.scores.append(score_text(result.text, sample.truth))
    for key, st in stats.items():
        st.config["subset"] = key
        st.load_s, st.warmup_s = load_s, warmup
    return stats


def print_table(rows):
    cells = [[("-" if r.get(c) is None else str(r.get(c))) for c in COLUMNS] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(COLUMNS)]
    line = "  ".join(c.rjust(w) for c, w in zip(COLUMNS, widths))
    print("\n" + line + "\n" + "-" * len(line))
    for row in cells:
        print("  ".join(v.rjust(w) for v, w in zip(row, widths)))


def main():
    args = parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # per-file download logs
    if args.data:
        samples = load_folder(args.data)
        if not samples:
            sys.exit(f"No images found in {args.data}")
    else:
        samples = synthetic_set(args.synthetic, font_path=args.font,
                                camera=not args.no_camera)
    truth_count = sum(s.truth is not None for s in samples)
    print(f"{len(samples)} images ({truth_count} with ground truth)")

    devices = args.devices or list(available_devices())
    rows = []
    for device in devices:
        for tier in args.tiers:
            config = {"lang": args.lang, "tier": tier, "device": device}
            print(f"> {tier} on {device} ...", flush=True)
            try:
                stats = evaluate(config, samples, args)
            except Exception as exc:  # keep benchmarking other configs
                print(f"  failed: {exc}")
                continue
            rows += [st.summary() for _, st in sorted(stats.items(),
                                                      key=lambda kv: kv[0] == "ALL")]
    if not rows:
        sys.exit("No configuration ran successfully.")
    print_table(rows)
    print("\nLatency = OCR only (detection + recognition), warm-up run excluded. "
          "CER/WER: lower is better.")
    if not args.data and args.lang == "korean":
        print("Note: synthetic Korean lines are random syllables with no language context, "
              "so CER is a pessimistic stress test\n(compare configs against each other). "
              "For real-world accuracy, use --data with your own labeled photos.")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
