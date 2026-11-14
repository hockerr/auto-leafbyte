"""Ecological leaf traits from the segmented leaf (numpy only).

Everything here works on the *rectified* image produced by :mod:`leafbyte.scale`
so that pixels are square and ``units_per_px`` (``scale_cm / scale_px_length``)
converts lengths to cm.  When no scale was found (relative-only mode) every
length/area trait is ``None`` and only the dimensionless traits are reported.

Traits
------
Size
    ``leaf_area_cm2``      remaining leaf tissue (leaf pixels, holes excluded)
    ``intact_area_cm2``    leaf outline area (tissue + consumed holes)
    ``perimeter_cm``       outer contour length (Vossepoel–Smeulders chain-code
                           estimator, unbiased for digitised curves)
    ``length_cm``          long side of the minimum-area bounding rectangle
                           (follows the leaf's main axis)
    ``width_cm``           short side of that rectangle (maximum width)
    ``max_feret_cm``       maximum Feret diameter (longest caliper distance)
    ``min_feret_cm``       minimum caliper width
    ``convex_area_cm2``    convex hull area
    ``equiv_diameter_cm``  diameter of the circle with the same area
Shape (dimensionless)
    ``aspect_ratio``       length / width
    ``circularity``        4πA / P²          (1 = circle; low = lobed/serrated)
    ``roundness``          4A / (π · max_feret²)
    ``solidity``           A / convex hull area (low = lobed, toothed, eaten margins)
    ``rectangularity``     A / (length · width)  (bounding-rectangle fill)
    ``dissection_index``   P / (2√(πA))      (1 = circle)
Colour (leaf tissue pixels only)
    ``mean_r/g/b``         mean 0–255 channel values
    ``gcc``                green chromatic coordinate G / (R + G + B)
    ``exg``                excess green 2g − r − b on chromatic coordinates
Herbivory
    ``hole_count``, ``mean_hole_cm2``

Specific leaf area (SLA, cm²/g) and leaf mass per area (LMA, g/m²) need the
leaf dry mass, which a photo cannot give; :func:`attach_dry_mass` adds them
once the user supplies masses.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

#: Directions for 8-connected chain codes, clockwise on screen (y grows down):
#: E, SE, S, SW, W, NW, N, NE.
_DIRS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

#: Column order used by the CSV writers and the GUI.
TRAIT_COLUMNS = [
    "leaf_area_cm2", "intact_area_cm2", "perimeter_cm", "length_cm", "width_cm",
    "max_feret_cm", "min_feret_cm", "aspect_ratio", "circularity", "roundness", "solidity",
    "rectangularity", "convex_area_cm2", "equiv_diameter_cm", "dissection_index",
    "mean_r", "mean_g", "mean_b", "gcc", "exg", "hole_count", "mean_hole_cm2",
]

#: Columns added by :func:`attach_dry_mass`.
MASS_COLUMNS = ["dry_mass_mg", "sla_cm2_g", "lma_g_m2"]

#: Human readable labels (Chinese) + units, shared with the GUI.
TRAIT_LABELS = {
    "leaf_area_cm2": ("剩余叶面积", "cm²"),
    "intact_area_cm2": ("完整叶面积", "cm²"),
    "perimeter_cm": ("叶周长", "cm"),
    "length_cm": ("叶长", "cm"),
    "width_cm": ("叶宽", "cm"),
    "max_feret_cm": ("最大卡径", "cm"),
    "min_feret_cm": ("最小卡径", "cm"),
    "aspect_ratio": ("长宽比", ""),
    "circularity": ("圆度", ""),
    "roundness": ("圆整度", ""),
    "solidity": ("凸度（实心度）", ""),
    "rectangularity": ("矩形度", ""),
    "convex_area_cm2": ("凸包面积", "cm²"),
    "equiv_diameter_cm": ("等效直径", "cm"),
    "dissection_index": ("叶缘分裂指数", ""),
    "mean_r": ("平均R", ""),
    "mean_g": ("平均G", ""),
    "mean_b": ("平均B", ""),
    "gcc": ("绿色色度坐标GCC", ""),
    "exg": ("过绿指数ExG", ""),
    "hole_count": ("孔洞数", "个"),
    "mean_hole_cm2": ("平均孔洞面积", "cm²"),
    "dry_mass_mg": ("干重", "mg"),
    "sla_cm2_g": ("比叶面积SLA", "cm²/g"),
    "lma_g_m2": ("比叶重LMA", "g/m²"),
}


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def trace_outer_contour(mask: np.ndarray) -> list[int]:
    """8-connected chain code of the outer boundary of the top-left object.

    Returns the list of direction codes (indices into ``_DIRS``).  An empty list
    means a single pixel or an empty mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.pad(mask, 1)
    ys, xs = np.nonzero(padded)
    first = np.lexsort((xs, ys))[0]
    start = (int(xs[first]), int(ys[first]))
    x, y = start
    direction = 0
    first_move: int | None = None
    chain: list[int] = []
    limit = 4 * int(mask.sum()) + 8
    for _ in range(limit):
        # Moore-neighbour tracing: after arriving via ``direction``, start the
        # clockwise scan two steps counter-clockwise from it.  This must be the
        # same for axial and diagonal moves; using d+7 for diagonals skipped a
        # neighbour and inflated the perimeter of jagged / noisy leaf margins.
        search = (direction + 6) % 8
        for k in range(8):
            nd = (search + k) % 8
            dx, dy = _DIRS[nd]
            if padded[y + dy, x + dx]:
                break
        else:
            return []  # isolated pixel
        if (x, y) == start and first_move is not None and nd == first_move:
            break
        if first_move is None:
            first_move = nd
        chain.append(nd)
        dx, dy = _DIRS[nd]
        x, y = x + dx, y + dy
        direction = nd
    return chain


def chain_perimeter(chain: Iterable[int]) -> float:
    """Vossepoel & Smeulders (1982) length estimate of a chain code, in px."""
    codes = list(chain)
    if not codes:
        return 0.0
    even = sum(1 for c in codes if c % 2 == 0)
    odd = len(codes) - even
    corners = sum(1 for a, b in zip(codes, codes[1:] + codes[:1]) if a != b)
    return 0.980 * even + 1.406 * odd - 0.091 * corners


def _boundary_corner_points(mask: np.ndarray) -> np.ndarray:
    """Pixel-corner coordinates of every boundary pixel, ``(n, 2)`` as (x, y)."""
    padded = np.pad(mask, 1)
    inner = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
             & padded[1:-1, :-2] & padded[1:-1, 2:])
    edge = mask & ~inner
    ys, xs = np.nonzero(edge)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    corners = np.concatenate([
        np.stack([xs, ys], 1), np.stack([xs + 1, ys], 1),
        np.stack([xs, ys + 1], 1), np.stack([xs + 1, ys + 1], 1),
    ])
    return np.unique(corners, axis=0)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; returns hull vertices counter-clockwise."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def feret(hull: np.ndarray) -> dict[str, float]:
    """Caliper measurements of a convex hull, in px.

    ``max_feret`` longest caliper distance, ``min_feret`` narrowest caliper
    width, and ``length``/``width`` = long/short side of the minimum-area
    bounding rectangle (rotating calipers over the hull edges).
    """
    zero = {"max_feret": 0.0, "min_feret": 0.0, "length": 0.0, "width": 0.0}
    if len(hull) < 2:
        return zero
    diff = hull[:, None, :] - hull[None, :, :]
    max_feret = math.sqrt(float((diff ** 2).sum(-1).max()))
    if len(hull) < 3:
        return {"max_feret": max_feret, "min_feret": 0.0, "length": max_feret, "width": 0.0}
    min_feret = math.inf
    best = (math.inf, max_feret, 0.0)
    for k in range(len(hull)):
        a, b = hull[k], hull[(k + 1) % len(hull)]
        edge = b - a
        norm = math.hypot(edge[0], edge[1])
        if norm == 0:
            continue
        along = edge / norm
        normal = np.array([-along[1], along[0]])
        p = hull @ along
        q = hull @ normal
        ext_a = float(p.max() - p.min())
        ext_n = float(q.max() - q.min())
        min_feret = min(min_feret, ext_n)
        if ext_a * ext_n < best[0]:
            best = (ext_a * ext_n, max(ext_a, ext_n), min(ext_a, ext_n))
    return {"max_feret": max_feret, "min_feret": 0.0 if min_feret is math.inf else min_feret,
            "length": best[1], "width": best[2]}


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def compute_traits(leaf_mask, hole_mask=None, image=None,
                   units_per_px: float | None = None) -> dict[str, Any]:
    """All traits for one leaf.

    ``leaf_mask``: bool mask of leaf tissue; ``hole_mask``: consumed holes
    (inside the outline); ``image``: RGB uint8 array of the same shape (for
    colour); ``units_per_px``: cm per pixel, ``None`` when no scale.
    """
    out: dict[str, Any] = {key: None for key in TRAIT_COLUMNS}
    leaf = np.asarray(leaf_mask, dtype=bool)
    if leaf.ndim != 2 or not leaf.any():
        return out
    holes = np.zeros_like(leaf) if hole_mask is None else np.asarray(hole_mask, dtype=bool)
    outline = leaf | holes

    leaf_px = float(leaf.sum())
    area_px = float(outline.sum())
    perimeter_px = chain_perimeter(trace_outer_contour(outline))
    hull = convex_hull(_boundary_corner_points(outline))
    hull_area_px = polygon_area(hull)
    cal = feret(hull)
    length_px, width_px = cal["length"], cal["width"]

    u = float(units_per_px) if units_per_px else None

    def cm(px):
        return None if u is None else round(px * u, 4)

    def cm2(px):
        return None if u is None else round(px * u * u, 4)

    out.update(
        leaf_area_cm2=cm2(leaf_px),
        intact_area_cm2=cm2(area_px),
        perimeter_cm=cm(perimeter_px),
        length_cm=cm(length_px),
        width_cm=cm(width_px),
        max_feret_cm=cm(cal["max_feret"]),
        min_feret_cm=cm(cal["min_feret"]),
        convex_area_cm2=cm2(hull_area_px),
        equiv_diameter_cm=cm(math.sqrt(4.0 * area_px / math.pi)),
        aspect_ratio=round(length_px / width_px, 4) if width_px > 0 else None,
        circularity=(round(min(1.0, 4.0 * math.pi * area_px / perimeter_px ** 2), 4)
                     if perimeter_px > 0 else None),
        roundness=(round(4.0 * area_px / (math.pi * cal["max_feret"] ** 2), 4)
                   if cal["max_feret"] > 0 else None),
        solidity=round(min(1.0, area_px / hull_area_px), 4) if hull_area_px > 0 else None,
        rectangularity=(round(area_px / (length_px * width_px), 4)
                        if length_px > 0 and width_px > 0 else None),
        dissection_index=(round(perimeter_px / (2.0 * math.sqrt(math.pi * area_px)), 4)
                          if area_px > 0 else None),
    )

    # herbivory holes ------------------------------------------------------
    if hole_mask is not None:
        from leafbyte import ccl  # local import keeps this module numpy-only at import time
        info = ccl.label_components(holes) if holes.any() else None
        count = len(info.occupied_labels()) if info is not None else 0
        out["hole_count"] = count
        out["mean_hole_cm2"] = cm2(float(holes.sum()) / count) if count else (0.0 if u else None)

    # colour ----------------------------------------------------------------
    if image is not None:
        img = np.asarray(image)
        if img.ndim == 3 and img.shape[:2] == leaf.shape:
            pixels = img[leaf][:, :3].astype(np.float64)
            r, g, b = pixels.mean(axis=0)
            total = pixels.sum(axis=1)
            total[total == 0] = 1.0
            chroma = pixels / total[:, None]
            gcc = float(chroma[:, 1].mean())
            exg = float((2 * chroma[:, 1] - chroma[:, 0] - chroma[:, 2]).mean())
            out.update(mean_r=round(float(r), 2), mean_g=round(float(g), 2),
                       mean_b=round(float(b), 2), gcc=round(gcc, 4), exg=round(exg, 4))
    return out


def attach_dry_mass(record: dict, dry_mass_mg) -> dict:
    """Add ``dry_mass_mg``, ``sla_cm2_g`` and ``lma_g_m2`` to one record.

    SLA uses the remaining leaf tissue area (the tissue that was weighed);
    falls back to ``total_cm2`` minus ``consumed_cm2`` when traits are missing.
    """
    try:
        mass = float(dry_mass_mg)
    except (TypeError, ValueError):
        mass = None
    record["dry_mass_mg"] = mass
    record["sla_cm2_g"] = None
    record["lma_g_m2"] = None
    traits = record.get("traits") or {}
    area = traits.get("leaf_area_cm2")
    if area is None and record.get("total_cm2") is not None:
        area = float(record["total_cm2"]) - float(record.get("consumed_cm2") or 0.0)
    if mass and mass > 0 and area and area > 0:
        grams = mass / 1000.0
        record["sla_cm2_g"] = round(area / grams, 3)
        record["lma_g_m2"] = round(grams / (area / 1e4), 3)
    return record


def trait_value(record: dict, column: str):
    """Look a trait/mass column up on a record (CSV writer helper)."""
    if column in MASS_COLUMNS:
        return record.get(column)
    return (record.get("traits") or {}).get(column)
