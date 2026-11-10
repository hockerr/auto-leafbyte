"""Generate the scale-detection report for the 30 benchmark paper images.

Writes ``out/scale/scale_report.csv`` with columns:
``file,found,method,quality,side_min_px,side_max_px,flags``

Scratch tool, not part of the package.
"""
import csv
import time
from pathlib import Path

from leafbyte import imageops, scale
from leafbyte.config import Config

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "paper" / "images"
OUT = Path(__file__).resolve().parent / "scale_report.csv"


def main():
    cfg = Config()
    files = sorted(p.name for p in IMG_DIR.glob("*.png") if p.name != "leafWithScale.png")
    rows = []
    timings = []
    for name in files:
        raw = imageops.load_image(IMG_DIR / name)
        normalized = imageops.resize_within(raw, cfg.max_dim)
        start = time.perf_counter()
        detection = scale.detect(normalized, cfg)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        details = detection.details or {}
        side_min = details.get("side_min_px", 0.0) if detection.found else 0.0
        side_max = details.get("side_max_px", 0.0) if detection.found else 0.0
        rows.append({
            "file": name,
            "found": bool(detection.found),
            "method": detection.method,
            "quality": round(float(detection.quality), 4),
            "side_min_px": round(float(side_min), 1),
            "side_max_px": round(float(side_max), 1),
            "flags": "; ".join(detection.flags),
        })
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "found", "method", "quality",
                                                    "side_min_px", "side_max_px", "flags"])
        writer.writeheader()
        writer.writerows(rows)
    found = sum(1 for row in rows if row["found"])
    print(f"{found}/{len(rows)} found; wrote {OUT}")
    print(f"runtime detect: mean {sum(timings)/len(timings)*1000:.1f} ms, "
          f"max {max(timings)*1000:.1f} ms")
    for row in rows:
        print(f"{row['file']:10s} found={row['found']!s:5s} {row['method']:13s} "
              f"q={row['quality']:.3f} sides={row['side_min_px']}-{row['side_max_px']} "
              f"{row['flags']}")


if __name__ == "__main__":
    main()
