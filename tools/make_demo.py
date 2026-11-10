"""Generate the small demo set used to illustrate the human-review workflow.

The paper photos in ``data/paper/images`` are clean lab scans: almost all of them
are measured automatically.  To show what "low precision -> human review" looks
like in practice we degrade a few of them the way a real field photo degrades:

* ``blur``       - out-of-focus: the 4 scale dots smear into the paper
* ``glare``      - a specular highlight wipes out one scale dot
* ``cut_leaf``   - the leaf crosses the edge of the 17 cm square / frame
* ``dark``       - underexposed, leaf and paper both dark, weak contrast

Usage::

    PYTHONPATH=src python tools/make_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from leafbyte import ccl, imageops  # noqa: E402

SOURCE_DIR = ROOT / "data" / "paper" / "images"
TARGET_DIR = ROOT / "data" / "demo"


def _save(image: Image.Image, name: str) -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    path = TARGET_DIR / name
    image.save(path)
    return path


def main() -> int:
    co1 = imageops.load_image(SOURCE_DIR / "CO1.png")
    gb4 = imageops.load_image(SOURCE_DIR / "GB4.png")
    sh2 = imageops.load_image(SOURCE_DIR / "SH2.png")

    # 2 clean photos that should be measured automatically
    _save(Image.fromarray(co1), "demo01_CO1_clean.png")
    _save(Image.fromarray(gb4), "demo02_GB4_clean.png")

    # out of focus: dots lose contrast
    blurred = Image.fromarray(co1).filter(ImageFilter.GaussianBlur(radius=14))
    _save(blurred, "demo03_CO1_blurred.png")

    # specular glare on the bottom-left scale dot
    glare = Image.fromarray(sh2).convert("RGB")
    draw = ImageDraw.Draw(glare, "RGBA")
    radians = np.linspace(0, 2 * np.pi, 200)
    centre = (60, 800)
    radius_x, radius_y = 90, 70
    polygon = [(centre[0] + radius_x * np.cos(t), centre[1] + radius_y * np.sin(t)) for t in radians]
    draw.polygon(polygon, fill=(255, 255, 255, 235))
    _save(glare, "demo04_SH2_glare.png")

    # a torn / fragmented leaf: a paper-coloured band cuts the lamina in two,
    # so "which blob is the leaf" becomes ambiguous (competing_leaf flag).
    # The band is placed from the leaf's own bounding box so it is guaranteed to
    # split it regardless of where the leaf sits in the photo.
    threshold = imageops.suggested_threshold(co1)
    info = ccl.label_components(imageops.binarize(co1, threshold))
    leaf = info.leaf_label()
    ys, xs = np.nonzero(info.label_map == leaf)
    x0, x1 = int(xs.min()) - 10, int(xs.max()) + 10
    y_mid = int((ys.min() + ys.max()) / 2)
    half_height = max(6, int(0.06 * (ys.max() - ys.min())))
    paper = tuple(int(channel) for channel in np.median(co1[0:20, 0:20], axis=(0, 1)))
    fractured = Image.fromarray(co1).convert("RGB")
    ImageDraw.Draw(fractured).rectangle([x0, y_mid - half_height, x1, y_mid + half_height],
                                        fill=paper)
    _save(fractured, "demo05_CO1_fractured.png")

    # underexposed
    dark = np.clip(np.asarray(co1, dtype=np.float32) * 0.55, 0, 255).astype(np.uint8)
    _save(Image.fromarray(dark), "demo06_CO1_dark.png")

    print(f"wrote demo images to {TARGET_DIR}")
    for path in sorted(TARGET_DIR.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
