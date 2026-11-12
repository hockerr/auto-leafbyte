"""Pipeline configuration (plain dataclass, JSON serialisable).

Owned by the Team Lead.  ``run.json`` written next to the results records the
exact configuration used, so every automated measurement is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- image normalisation -------------------------------------------------
    #: Longest side after the initial resize; upstream uses 1200 and never
    #: scales up (``resizeImage(_:within:)``).
    max_dim: int = 1200
    #: ``"white"`` => occupied pixels are *darker* than the threshold (a leaf
    #: on white paper, the upstream default).  ``"black"`` => brighter.
    background: str = "white"

    #: How the leaf is separated from the paper.
    #: ``"luma"``   = upstream behaviour: global Otsu on luma (needs white paper)
    #: ``"color"``  = excess-green ``2G-R-B`` (works on grey / uneven / bluish
    #:                backgrounds and under-exposed photos)
    #: ``"auto"``   = try luma, and fall back to colour when the luma result is
    #:                not leaf-like (background leaking into the "leaf")
    leaf_mode: str = "auto"
    #: ExG cut in ``2G-R-B`` units.  ``None`` = Otsu on the ExG histogram of the
    #: rectified image (adaptive, recommended); a number forces a fixed cut.
    color_threshold: float | None = None
    #: In ``auto`` mode the colour segmentation is only computed when the luma
    #: result shows a competing blob bigger than this fraction of the leaf (the
    #: signature of a grey / vignetted background).  Keeps white-paper photos
    #: as fast as before.
    color_probe_competing_ratio: float = 0.05
    #: Sanity clamp for the adaptive ExG cut (in ExG units).  The cut is Otsu on
    #: the ExG histogram; the clamp only guards against pathological images.
    color_threshold_range: tuple[float, float] = (6.0, 40.0)
    #: Treat a hole as eaten area only when its interior looks like the paper
    #: background; darker lesions / fold shadows are filled back into the leaf.
    hole_validate_background: bool = True
    #: Interior luma below ``hole_dark_ratio * paper luma`` is "tissue, not hole".
    hole_dark_ratio: float = 0.55
    #: An interior patch whose mean RGB is farther than this (Euclidean, 0-255
    #: units) from the paper colour is tissue too (yellow/brown necrosis).
    hole_paper_colour_distance: float = 45.0
    #: Binary closing radius applied to the colour mask (fills veins/speckle).
    mask_closing: int = 2
    #: ExG perturbation used for the stability metric in colour mode.
    color_stability_delta: float = 4.0
    #: Relative area change that still counts as "stable" in colour mode.  The
    #: leaf/background boundary is a gradient of a few ExG units, so the edge is
    #: inherently a bit softer than the white-paper luma path.
    color_stability_tolerance: float = 0.035
    #: Extra luma thresholds tried for scale-dot detection when the Otsu pass
    #: fails (black dots are simply very dark, independent of the background).
    dot_thresholds: tuple[float, ...] = (0.20, 0.15, 0.25, 0.30)

    # --- segmentation --------------------------------------------------------
    #: ``None`` => Otsu on the luma histogram.  A float in [0, 1] forces it.
    threshold: float | None = None
    #: Threshold offsets (in luma units, 0..1) that are also evaluated for the
    #: stability metric and offered as candidates in the review UI.
    candidate_deltas: tuple[float, ...] = (-0.04, -0.02, 0.0, 0.02, 0.04)
    #: Threshold offsets used for the stability ("precision") metric.
    stability_deltas: tuple[float, ...] = (-0.02, 0.02)

    # --- scale ---------------------------------------------------------------
    #: Physical length in cm of the scale mark (upstream default is 10.0; the
    #: paper protocol used a 17 cm square of four dots).
    scale_cm: float = 17.0
    #: How the scale is expressed: ``"quad"`` = four corner marks forming the
    #: square, ``"quad_relaxed"`` = four marks but free quadrilateral geometry.
    scale_mode: str = "quad"
    #: Accepted size of one scale mark in the *pre-rectification* image, px.
    scale_mark_min_px: int = 19
    scale_mark_max_px: int = 50_000
    #: A scale mark is a blob of roughly this aspect ratio (1.0 = round).
    scale_mark_max_aspect: float = 2.5

    # --- measurement ---------------------------------------------------------
    #: Holes smaller than this many pixels are still counted but flagged.
    tiny_hole_px: int = 25
    #: Ignore empty components this close to the leaf border?  (0 = upstream
    #: behaviour: every enclosed connected hole counts).
    hole_min_px: int = 1

    # --- quality / routing ---------------------------------------------------
    #: ``confidence >= auto_accept_min`` => accepted without review.
    #: 0.92 is the value calibrated on the 29 paper samples with published
    #: areas: it auto-accepts 24/30 with no sample outside the 1.5 % area
    #: tolerance.  Lower it to trade review effort for coverage.
    auto_accept_min: float = 0.92
    #: ``confidence < review_min`` => treated as failed (needs a human).
    review_min: float = 0.45
    #: Relative total-area change under +-``stability_deltas`` that already
    #: counts as "unstable" (0.02 = 2 %).
    stability_tolerance: float = 0.02
    #: Absolute area tolerance in cm^2 used to score accuracy against a
    #: reference value, when one is known.
    reference_tolerance_cm2: float = 0.05

    # --- review UI -----------------------------------------------------------
    #: Max candidate overlays rendered per flagged image.
    review_candidates: int = 5

    # --- io ------------------------------------------------------------------
    overlay_max_px: int = 900
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io ---
    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("candidate_deltas", "stability_deltas", "dot_thresholds",
                    "color_threshold_range"):
            d[key] = list(d[key])
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    #: Keys introduced together with the colour segmentation.  A config file that
    #: has none of them predates that feature, so replaying it must keep the old
    #: (luma-only) behaviour rather than silently picking up the new defaults.
    NEW_GENERATION_KEYS = ("leaf_mode", "color_threshold", "color_threshold_range",
                           "mask_closing", "color_stability_delta")
    LEGACY_DEFAULTS = {"leaf_mode": "luma", "hole_validate_background": False}

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in fields}
        if not any(key in data for key in cls.NEW_GENERATION_KEYS):
            for key, legacy in cls.LEGACY_DEFAULTS.items():
                kwargs.setdefault(key, legacy)
        for key in ("candidate_deltas", "stability_deltas", "dot_thresholds",
                    "color_threshold_range"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


DEFAULT_CONFIG = Config()
