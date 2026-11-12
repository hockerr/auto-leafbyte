"""Scale-mark detection and perspective rectification.

Contract: ``docs/ALGORITHM.md`` §3.4 (finding the marks), §3.5 (rectification) and
§6.3 (API).  Owner: ``scale``.

The paper protocol used by the benchmark photographs is a square printed on
white paper whose four corners are marked with black round dots.  The dots are
found as *compact occupied connected components* and the four that form the
best near-square quadrilateral are kept.  This is deliberately different from
upstream LeafByte's "2nd..5th biggest dark component" heuristic: the paper
photos contain dark photo-frame edges, tape strips, glare and a leaf, so "the
five biggest blobs" is not a reliable set (it fails on 9 of the 30 benchmark
images).  Here every component is filtered on size, aspect ratio, bbox fill
ratio and border contact first, and the remaining candidates are ranked by a
geometric squareness score, so spurious blobs are rejected instead of shifted
into the output.

Coordinate convention (identical to :mod:`leafbyte.types`): ``(x, y)`` with
``x`` = column, ``y`` = row, origin at the top-left pixel, y increasing
downwards.  Quadrilaterals are always returned ordered ``TL, TR, BR, BL``.

Public API
----------
``detect(img, cfg)``
    Find the four scale marks in a *normalised* RGB image (already resized by
    :func:`leafbyte.imageops.resize_within`).  Returns a
    :class:`~leafbyte.types.ScaleDetection`.
``rectify(img, detection, cfg)``
    Perspective-correct the detected quadrilateral into an ``S x S`` RGB image.
    ``S = min(cfg.max_dim, floor(min(bbox_w, bbox_h)))`` and
    ``scale_px_length = S`` (upstream ``sizeToAdjustTo``).  Returns
    :class:`~leafbyte.types.RectifiedImage` or ``None``.
``detect_and_rectify(img, cfg)``
    Convenience wrapper returning ``(detection, rectified_or_None)``.
``order_quad(points)`` / ``quad_metrics(points)`` / ``solve_homography(src, dst)``
    Pure-geometry helpers, also used by the unit tests.
"""

from __future__ import annotations

import itertools

import numpy as np

from leafbyte import ccl, imageops
from leafbyte.config import Config
from leafbyte.types import RectifiedImage, ScaleDetection

__all__ = [
    "MIN_QUAD_QUALITY",
    "GOOD_QUAD_QUALITY",
    "detect",
    "rectify",
    "detect_and_rectify",
    "order_quad",
    "quad_metrics",
    "solve_homography",
    "apply_homography",
]

# ---------------------------------------------------------------------------
# tuning constants (kept as module constants so callers/tests can see them)
# ---------------------------------------------------------------------------
#: A quadrilateral below this quality is not reported as ``found``.
MIN_QUAD_QUALITY = 0.30
#: Above this quality the primary attempt is accepted immediately (no extra
#: threshold/relaxed attempts are run, which keeps the common path fast).
GOOD_QUAD_QUALITY = 0.55
#: Hard cap on the number of candidate blobs fed to the C(25,4) search.
CANDIDATE_LIMIT = 25
#: Hard cap on the triple search for the 3-dot fallback.
THREE_DOT_LIMIT = 12
#: Components whose geometry is measured at all (safety cap for noisy masks).
MAX_BLOBS_MEASURED = 80
#: Minimum bbox fill ratio of a scale-mark blob (``size / (bbox_w*bbox_h)``).
MIN_MARK_FILL = 0.40
MIN_MARK_FILL_RELAXED = 0.35
#: A scale mark may occupy at most this fraction of the image (leaves do not).
MAX_MARK_AREA_FRACTION = 0.02
#: Relaxed pass: smaller/faster blobs are allowed (faded or fragmented dots).
RELAXED_MIN_SIZE_FACTOR = 0.5
RELAXED_ASPECT_FACTOR = 1.4
#: Quality multiplier of a relaxed-pass detection.
RELAXED_QUALITY_FACTOR = 0.85
#: Quality multiplier when the fourth corner had to be inferred (3 dots only).
THREE_DOT_QUALITY_FACTOR = 0.60
#: Below this ratio between the smallest and biggest mark the marks are called
#: "uneven" (a faded dot) in the flags.
UNEVEN_MARK_RATIO = 0.60
#: Weight of the "mark prominence" term in the quad quality: the four chosen
#: blobs should be the biggest compact blobs available, otherwise four tiny
#: dust specks that happen to form a square would win over the real marks.
PROMINENCE_WEIGHT = 0.25
#: Each mark must cover at least this fraction of the image.  On the paper
#: protocol the dots cover ~2.5e-4..5e-4 of the photo (17 cm square filling the
#: frame), while paper dust specs cover ~2e-5; the floor keeps a quartet of
#: specks from being accepted as a quad while leaving 5x headroom below the
#: faintest benchmark dot.
MIN_MARK_AREA_FRACTION = 5e-5
#: Additional threshold offsets tried (in luma units) when the primary Otsu
#: threshold finds no good quad; positive = more pixels become occupied, which
#: recovers faded marks.
THRESHOLD_DELTAS = (0.05, 0.10)

_QUAD_INDEX_CACHE: dict[int, np.ndarray] = {}
_TRIPLE_INDEX_CACHE: dict[int, np.ndarray] = {}


# ---------------------------------------------------------------------------
# pure geometry
# ---------------------------------------------------------------------------
def _order_quads_vec(points: np.ndarray) -> np.ndarray:
    """Order ``(M, 4, 2)`` point sets as ``TL, TR, BR, BL`` (vectorised).

    Points are sorted by their angle around the centroid (clockwise in image
    coordinates, where y grows downwards), then the cyclic sequence is rotated
    so that it starts at the corner most likely to be top-left (smallest
    ``x + y``).  This is the standard 4-point ordering and is stable for the
    mild rotations / perspective of the leaf protocol.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[1] != 4 or pts.shape[2] != 2:
        raise ValueError(f"expected (M, 4, 2) points, got {pts.shape}")
    centre = pts.mean(axis=1, keepdims=True)
    angles = np.arctan2(pts[:, :, 1] - centre[:, :, 1], pts[:, :, 0] - centre[:, :, 0])
    order = np.argsort(angles, axis=1, kind="stable")
    ordered = np.take_along_axis(pts, order[:, :, None], axis=1)
    start = np.argmin(ordered[:, :, 0] + ordered[:, :, 1], axis=1)
    offsets = (start[:, None] + np.arange(4, dtype=np.intp)[None, :]) % 4
    return np.take_along_axis(ordered, offsets[:, :, None], axis=1)


def order_quad(points) -> np.ndarray:
    """Return the four points ordered ``TL, TR, BR, BL`` as a ``(4, 2)`` array.

    Works on any corner order.  Degenerate inputs (duplicate or collinear
    points) still return an array; use :func:`quad_metrics` to inspect it.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"a quadrilateral needs exactly 4 points, got {pts.shape[0]}")
    return _order_quads_vec(pts[None, ...])[0]


def _quad_metrics_vec(ordered: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised metrics for already ordered ``(M, 4, 2)`` quadrilaterals."""
    p = np.asarray(ordered, dtype=np.float64)
    nxt = np.roll(p, -1, axis=1)
    prv = np.roll(p, 1, axis=1)
    edges = nxt - p
    edge_next = np.roll(edges, -1, axis=1)
    cross = edges[:, :, 0] * edge_next[:, :, 1] - edges[:, :, 1] * edge_next[:, :, 0]
    sides = np.linalg.norm(edges, axis=2)
    diag0 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    diag1 = np.linalg.norm(p[:, 3] - p[:, 1], axis=1)
    area = 0.5 * np.abs((p[:, :, 0] * nxt[:, :, 1] - p[:, :, 1] * nxt[:, :, 0]).sum(axis=1))
    bbox_w = p[:, :, 0].max(axis=1) - p[:, :, 0].min(axis=1)
    bbox_h = p[:, :, 1].max(axis=1) - p[:, :, 1].min(axis=1)
    e1 = prv - p
    e2 = nxt - p
    denom = np.linalg.norm(e1, axis=2) * np.linalg.norm(e2, axis=2)
    cosang = np.where(denom > 0.0, (e1 * e2).sum(axis=2) / np.maximum(denom, 1e-12), 1.0)
    angles = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return {
        "points": p,
        "cross": cross,
        "sides": sides,
        "diag0": diag0,
        "diag1": diag1,
        "area": area,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "cosang": cosang,
        "angles": angles,
    }


def _quad_metrics_at(metrics: dict[str, np.ndarray], index: int) -> dict:
    """Turn the vectorised metrics of one quad into a JSON-friendly dict."""
    sides = metrics["sides"][index]
    diag0 = float(metrics["diag0"][index])
    diag1 = float(metrics["diag1"][index])
    diag_min, diag_max = sorted((diag0, diag1))
    s_max = float(sides.max())
    s_min = float(sides.min())
    cos_abs = np.abs(metrics["cosang"][index])
    angles = metrics["angles"][index]
    side_ratio = s_min / s_max if s_max > 0 else 0.0
    diagonal_ratio = diag_min / diag_max if diag_max > 0 else 0.0
    angle_score = float(np.clip(1.0 - float(cos_abs.mean()), 0.0, 1.0))
    squareness = float((max(side_ratio, 0.0) * max(diagonal_ratio, 0.0) * angle_score) ** (1.0 / 3.0))
    return {
        "sides": [float(v) for v in sides],
        "side_min": s_min,
        "side_max": s_max,
        "side_ratio": float(side_ratio),
        "diagonals": [diag0, diag1],
        "diagonal_ratio": float(diagonal_ratio),
        "width": float(metrics["bbox_w"][index]),
        "height": float(metrics["bbox_h"][index]),
        "area": float(metrics["area"][index]),
        "angles": [float(v) for v in angles],
        "convex": bool(np.all(metrics["cross"][index] > 0) or np.all(metrics["cross"][index] < 0)),
        "squareness": squareness,
    }


def quad_metrics(points) -> dict:
    """Geometry summary of one quadrilateral in ``{...}`` form.

    Keys include ``points`` (re-ordered ``TL,TR,BR,BL``), ``sides``,
    ``side_min``/``side_max``/``side_ratio``, ``diagonals``/``diagonal_ratio``,
    ``width``/``height``/``area``, ``angles``, ``convex`` and ``squareness``
    (0..1, 1 = perfect square).
    """
    ordered = order_quad(points)
    metrics = _quad_metrics_vec(ordered[None, ...])
    out = _quad_metrics_at(metrics, 0)
    out["points"] = [[float(x), float(y)] for x, y in ordered]
    return out


def apply_homography(matrix: np.ndarray, points) -> np.ndarray:
    """Apply a 3x3 homography to an ``(N, 2)`` array of points."""
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([pts, np.ones(pts.shape[0], dtype=np.float64)])
    mapped = homogeneous @ mat.T
    w = mapped[:, 2].copy()
    w[w == 0.0] = 1e-12
    return mapped[:, :2] / w[:, None]


def solve_homography(src_points, dst_points) -> np.ndarray:
    """Solve the 3x3 projective transform mapping ``src_points`` to ``dst_points``.

    Straight 8-parameter DLT: ``(x, y, 1) ~ H (u, v, 1)``, i.e. ``H`` maps source
    image coordinates to destination coordinates.  Needs at least four
    correspondences.  Raises :class:`ValueError` for a degenerate configuration
    (collinear points, duplicates, non-finite solution).
    """
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape or src.shape[0] < 4:
        raise ValueError("solve_homography needs >= 4 paired points of equal shape")
    number = src.shape[0]
    lhs = np.zeros((2 * number, 8), dtype=np.float64)
    rhs = np.zeros(2 * number, dtype=np.float64)
    for index, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        lhs[2 * index, 0:3] = (x, y, 1.0)
        lhs[2 * index, 6] = -u * x
        lhs[2 * index, 7] = -u * y
        rhs[2 * index] = u
        lhs[2 * index + 1, 3:6] = (x, y, 1.0)
        lhs[2 * index + 1, 6] = -v * x
        lhs[2 * index + 1, 7] = -v * y
        rhs[2 * index + 1] = v
    try:
        if number == 4:
            params = np.linalg.solve(lhs, rhs)
        else:
            params = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    except np.linalg.LinAlgError as exc:  # pragma: no cover - numpy wording varies
        raise ValueError("degenerate point configuration for homography") from exc
    if not np.all(np.isfinite(params)) or float(np.max(np.abs(params))) > 1e12:
        raise ValueError("degenerate point configuration for homography")
    matrix = np.append(params, 1.0).reshape(3, 3)
    # Sanity check: four correspondences must be reproduced exactly enough.
    reprojected = apply_homography(matrix, src)
    scale = max(1.0, float(np.max(np.abs(dst))))
    if float(np.max(np.abs(reprojected - dst))) > 1e-4 * scale:
        raise ValueError("degenerate point configuration for homography")
    return matrix


# ---------------------------------------------------------------------------
# candidate blobs
# ---------------------------------------------------------------------------
def _label_stats(label_map: np.ndarray, label: int) -> dict | None:
    """Bbox / centroid / fill of one connected component."""
    ys, xs = np.nonzero(label_map == label)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    size = int(xs.size)
    return {
        "size": size,
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "bbox": (x0, y0, x1, y1),
        "width": width,
        "height": height,
        "aspect": max(width, height) / max(1, min(width, height)),
        "fill": size / float(width * height),
        "border_touch": bool(x0 == 0 or y0 == 0 or x1 == label_map.shape[1] - 1
                             or y1 == label_map.shape[0] - 1),
    }


def _candidate_shape(fill: float, aspect: float) -> float:
    """0..1 "looks like a round dot" score from bbox fill and aspect ratio."""
    roundness = min(1.0, 1.15 / max(aspect, 1e-9))
    return float(min(1.0, fill / 0.75) * roundness)


def _candidates_from_components(components, cfg: Config, relaxed: bool) -> list[dict]:
    """Filter occupied components down to plausible scale-mark blobs.

    Contract filters (§3.4): size in ``[scale_mark_min_px, scale_mark_max_px]``,
    bbox aspect ``<= scale_mark_max_aspect``, area ``< 2%`` of the image, no
    border contact.  Additionally a bbox fill-ratio gate removes thin strips /
    tape edges (a round dot fills 0.7-0.8 of its bbox, a diagonal strip ~0.4 or
    less).  The qualifying blobs are ranked by a "dot-likeness" score so that
    the bounded C(n,4) search sees the most plausible candidates first.
    """
    width, height = int(components.width), int(components.height)
    if width <= 0 or height <= 0:
        return []
    min_px = max(1, int(round(cfg.scale_mark_min_px * (RELAXED_MIN_SIZE_FACTOR if relaxed else 1.0))))
    max_px = max(min_px, int(cfg.scale_mark_max_px))
    max_aspect = float(cfg.scale_mark_max_aspect) * (RELAXED_ASPECT_FACTOR if relaxed else 1.0)
    min_fill = MIN_MARK_FILL_RELAXED if relaxed else MIN_MARK_FILL
    area = float(width) * float(height)

    labels: list[int] = []
    for label in components.occupied_labels():
        size = int(components.label_to_size[label])
        if size < min_px:  # sorted biggest-first: everything smaller follows
            break
        if size > max_px or size / area >= MAX_MARK_AREA_FRACTION:
            continue
        labels.append(label)
        if len(labels) >= MAX_BLOBS_MEASURED:
            break

    candidates: list[dict] = []
    label_map = components.label_map
    for label in labels:
        stats = _label_stats(label_map, label)
        if stats is None:
            continue
        if stats["aspect"] > max_aspect or stats["fill"] < min_fill:
            continue
        if stats["border_touch"]:
            continue  # a mark glued to the frame edge is not trustworthy
        stats["label"] = int(label)
        stats["area_fraction"] = stats["size"] / area
        stats["shape"] = _candidate_shape(stats["fill"], stats["aspect"])
        candidates.append(stats)

    if not candidates:
        return []
    sizes = np.array([c["size"] for c in candidates], dtype=np.float64)
    median = float(np.median(sizes))
    for cand in candidates:
        proximity = median / (median + abs(cand["size"] - median)) if median > 0 else 0.0
        cand["dot_score"] = float(cand["shape"] * proximity)
    candidates.sort(key=lambda c: (-c["dot_score"], -c["size"], c["label"]))
    return candidates


def _quad_combos(n: int) -> np.ndarray:
    """All 4-subsets of ``range(n)`` as an ``(M, 4)`` integer array (cached)."""
    combos = _QUAD_INDEX_CACHE.get(n)
    if combos is None:
        combos = np.array(list(itertools.combinations(range(n), 4)), dtype=np.intp)
        _QUAD_INDEX_CACHE[n] = combos
    return combos


def _triple_combos(n: int) -> np.ndarray:
    """All 3-subsets of ``range(n)`` as an ``(M, 3)`` integer array (cached)."""
    combos = _TRIPLE_INDEX_CACHE.get(n)
    if combos is None:
        combos = np.array(list(itertools.combinations(range(n), 3)), dtype=np.intp)
        _TRIPLE_INDEX_CACHE[n] = combos
    return combos


# ---------------------------------------------------------------------------
# quadrilateral scoring
# ---------------------------------------------------------------------------
def _evaluate_quads(ordered: np.ndarray, sizes: np.ndarray, shapes: np.ndarray,
                    width: int, height: int, all_sizes: np.ndarray | None = None,
                    all_shapes: np.ndarray | None = None,
                    combos: np.ndarray | None = None) -> dict:
    """Score already-ordered quadrilaterals and mark the geometrically valid ones.

    Validity gates (documented in ``docs/ALGORITHM.md`` §3.4): convex, side
    ratio ``<= 2.5`` (``min/max >= 0.40``), diagonal ratio ``<= 2.0``,
    quad/image area fraction in ``[0.10, 0.98]``, bbox aspect ``<= 2.5``,
    corner angles in ``[40, 140]`` degrees, the four mark blobs within a
    factor 4 of each other in size, and each mark covering at least
    :data:`MIN_MARK_AREA_FRACTION` of the image.  The score is a weighted
    geometric mean of squareness, area coverage, mark-size similarity, blob
    roundness and mark prominence, so a perfect square made of prominent dots
    scores ~1.0 and junk scores near 0.
    """
    metrics = _quad_metrics_vec(ordered)
    sides = metrics["sides"]
    s_min = sides.min(axis=1)
    s_max = sides.max(axis=1)
    diag_min = np.minimum(metrics["diag0"], metrics["diag1"])
    diag_max = np.maximum(metrics["diag0"], metrics["diag1"])
    bbox_min = np.minimum(metrics["bbox_w"], metrics["bbox_h"])
    bbox_max = np.maximum(metrics["bbox_w"], metrics["bbox_h"])
    size_min = np.asarray(sizes, dtype=np.float64).min(axis=1)
    size_max = np.asarray(sizes, dtype=np.float64).max(axis=1)
    total_pixels = float(width) * float(height)

    with np.errstate(divide="ignore", invalid="ignore"):
        side_ratio = s_min / np.maximum(s_max, 1e-12)
        diagonal_ratio = diag_min / np.maximum(diag_max, 1e-12)
        area_fraction = metrics["area"] / total_pixels
        bbox_ratio = bbox_max / np.maximum(bbox_min, 1e-12)
        size_ratio = size_min / np.maximum(size_max, 1e-12)
        opp1 = np.minimum(sides[:, 0], sides[:, 2]) / np.maximum(
            np.maximum(sides[:, 0], sides[:, 2]), 1e-12)
        opp2 = np.minimum(sides[:, 1], sides[:, 3]) / np.maximum(
            np.maximum(sides[:, 1], sides[:, 3]), 1e-12)
        square_opposite = np.sqrt(np.clip(opp1 * opp2, 0.0, 1.0))
        angle_score = np.clip(1.0 - np.abs(metrics["cosang"]).mean(axis=1), 0.0, 1.0)

    cross = metrics["cross"]
    convex = np.all(cross > 0.0, axis=1) | np.all(cross < 0.0, axis=1)
    finite = np.all(np.isfinite(ordered), axis=(1, 2))
    valid = (
        finite
        & (s_min > 1.0)
        & convex
        & (side_ratio >= 0.40)
        & (diagonal_ratio >= 0.50)
        & (area_fraction >= 0.10)
        & (area_fraction <= 0.98)
        & (bbox_ratio <= 2.5)
        & (size_ratio >= 0.25)
        & (size_min / total_pixels >= MIN_MARK_AREA_FRACTION)
        & (metrics["angles"].min(axis=1) >= 40.0)
        & (metrics["angles"].max(axis=1) <= 140.0)
    )

    area_score = np.clip(area_fraction / 0.5, 0.0, 1.0)
    shape_score = np.clip(np.asarray(shapes, dtype=np.float64).mean(axis=1), 1e-6, 1.0)
    size_score = np.clip(size_ratio, 1e-6, 1.0)

    # Mark prominence: the chosen blobs should be the biggest *dot-like* blobs
    # in the image apart from themselves.  A quad of tiny dust specks gets a low
    # score because real marks were left out; an unrelated huge blob (leaf
    # fragment, tape) does not count as a reference because it is not dot-like.
    if all_sizes is not None and combos is not None:
        all_sizes = np.asarray(all_sizes, dtype=np.float64)
        if all_shapes is not None:
            shape_arr = np.asarray(all_shapes, dtype=np.float64)
            eligible = np.flatnonzero(shape_arr >= 0.70)
        else:
            eligible = np.arange(all_sizes.size)
        if eligible.size == 0:
            eligible = np.arange(all_sizes.size)
        eligible = eligible[np.argsort(-all_sizes[eligible], kind="stable")]
        count = all_sizes.size
        chosen_mask = np.zeros((combos.shape[0], count), dtype=bool)
        chosen_mask[np.arange(combos.shape[0])[:, None], combos] = True
        remaining = ~chosen_mask[:, eligible]
        has_other = remaining.any(axis=1)
        first_other = np.argmax(remaining, axis=1)
        reference = all_sizes[eligible][first_other]
        prominence = np.where(has_other, size_min / np.maximum(reference, 1e-9), 1.0)
    else:
        prominence = np.ones(size_min.shape[0], dtype=np.float64)
    prominence = np.clip(prominence, 1e-6, 1.0)

    quality = (
        np.power(np.clip(side_ratio, 1e-6, 1.0), 0.20)
        * np.power(np.clip(diagonal_ratio, 1e-6, 1.0), 0.15)
        * np.power(np.clip(square_opposite, 1e-6, 1.0), 0.15)
        * np.power(np.clip(angle_score, 1e-6, 1.0), 0.20)
        * np.power(area_score, 0.10)
        * np.power(size_score, 0.12)
        * np.power(shape_score, 0.08)
        * np.power(prominence, PROMINENCE_WEIGHT)
    )
    quality = np.where(valid, np.clip(quality, 0.0, 1.0), 0.0)
    return {
        "valid": valid,
        "quality": quality,
        "metrics": metrics,
        "side_ratio": side_ratio,
        "diagonal_ratio": diagonal_ratio,
        "area_fraction": area_fraction,
        "bbox_ratio": bbox_ratio,
        "size_ratio": size_ratio,
        "square_opposite": square_opposite,
        "angle_score": angle_score,
        "area_score": area_score,
        "shape_score": shape_score,
        "prominence": prominence,
    }


def _search_quads(candidates: list[dict], width: int, height: int):
    """Best 4-dot quadrilateral among ``candidates`` (or ``None`` + stats)."""
    stats = {"n_candidates": len(candidates), "n_quads": 0, "n_geometry_ok": 0,
             "best_quality": 0.0}
    if len(candidates) < 4:
        return None, stats
    if len(candidates) > CANDIDATE_LIMIT:
        candidates = candidates[:CANDIDATE_LIMIT]
        stats["n_candidates"] = len(candidates)
    points = np.array([[c["cx"], c["cy"]] for c in candidates], dtype=np.float64)
    sizes = np.array([c["size"] for c in candidates], dtype=np.float64)
    shapes = np.array([c["shape"] for c in candidates], dtype=np.float64)
    combos = _quad_combos(len(candidates))
    stats["n_quads"] = int(combos.shape[0])
    ordered = _order_quads_vec(points[combos])
    scored = _evaluate_quads(ordered, sizes[combos], shapes[combos], width, height,
                             all_sizes=sizes, all_shapes=shapes, combos=combos)
    valid = scored["valid"]
    stats["n_geometry_ok"] = int(valid.sum())
    if not valid.any():
        return None, stats
    stats["best_quality"] = float(scored["quality"][valid].max())
    quality = np.where(valid, scored["quality"], -np.inf)
    # Deterministic tie-break: highest quality, then largest area, then the
    # lexicographically smallest candidate-index combination.
    ordering = np.lexsort((np.arange(quality.size), -scored["metrics"]["area"], -quality))
    best = int(ordering[0])
    chosen = combos[best]
    metrics = _quad_metrics_at(scored["metrics"], best)
    metrics["mark_size_ratio"] = float(scored["size_ratio"][best])
    result = {
        "points": ordered[best],
        "quality": float(scored["quality"][best]),
        "candidate_indices": tuple(int(i) for i in chosen),
        "metrics": metrics,
    }
    return result, stats


def _search_quads_three_dot(candidates: list[dict], width: int, height: int):
    """Infer the missing fourth corner from three dots (parallelogram completion).

    For three corners of a square every possible fourth corner is one of three
    parallelogram completions; the one that closes the best near-square
    quadrilateral is kept.  The returned quality is the geometric quality of
    that quad (the caller multiplies by :data:`THREE_DOT_QUALITY_FACTOR`).
    """
    stats = {"n_candidates": len(candidates), "n_triples": 0, "n_geometry_ok": 0,
             "best_quality": 0.0}
    if len(candidates) < 3:
        return None, stats
    used = candidates[:min(len(candidates), THREE_DOT_LIMIT)]
    points = np.array([[c["cx"], c["cy"]] for c in used], dtype=np.float64)
    sizes = np.array([c["size"] for c in used], dtype=np.float64)
    shapes = np.array([c["shape"] for c in used], dtype=np.float64)
    triples = _triple_combos(len(used))
    stats["n_triples"] = int(triples.shape[0])
    a = points[triples[:, 0]]
    b = points[triples[:, 1]]
    c = points[triples[:, 2]]
    completions = np.stack([a + b - c, a + c - b, b + c - a], axis=1)  # (T,3,2)
    known = np.repeat(points[triples][:, None, :, :], 3, axis=1)       # (T,3,3,2)
    quads = np.concatenate([known, completions[:, :, None, :]], axis=2)  # (T,3,4,2)
    quads = quads.reshape(-1, 4, 2)

    known_sizes = np.repeat(sizes[triples][:, None, :], 3, axis=1)     # (T,3,3)
    proxy_size = np.median(sizes[triples], axis=1)[:, None]            # (T,1)
    quad_sizes = np.concatenate(
        [known_sizes, np.repeat(proxy_size[:, :, None], 3, axis=1)], axis=2).reshape(-1, 4)
    known_shapes = np.repeat(shapes[triples][:, None, :], 3, axis=1)
    proxy_shape = np.mean(shapes[triples], axis=1)[:, None]
    quad_shapes = np.concatenate(
        [known_shapes, np.repeat(proxy_shape[:, :, None], 3, axis=1)], axis=2).reshape(-1, 4)

    ordered = _order_quads_vec(quads)
    scored = _evaluate_quads(ordered, quad_sizes, quad_shapes, width, height)
    inside = (
        (completions.reshape(-1, 2)[:, 0] >= 0.0)
        & (completions.reshape(-1, 2)[:, 0] <= width - 1.0)
        & (completions.reshape(-1, 2)[:, 1] >= 0.0)
        & (completions.reshape(-1, 2)[:, 1] <= height - 1.0)
    )
    valid = scored["valid"] & inside
    stats["n_geometry_ok"] = int(valid.sum())
    if not valid.any():
        return None, stats
    stats["best_quality"] = float(scored["quality"][valid].max())
    quality = np.where(valid, scored["quality"], -np.inf)
    ordering = np.lexsort((np.arange(quality.size), -scored["metrics"]["area"], -quality))
    best = int(ordering[0])
    triple_index, completion_index = divmod(best, 3)
    result = {
        "points": ordered[best],
        "quality": float(scored["quality"][best]),
        "candidate_indices": tuple(int(i) for i in triples[triple_index]),
        "inferred_point": [float(completions[triple_index, completion_index, 0]),
                           float(completions[triple_index, completion_index, 1])],
        "metrics": _quad_metrics_at(scored["metrics"], best),
    }
    return result, stats


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def _candidate_payload(candidates: list[dict], selected: set[int]) -> list[dict]:
    """JSON-friendly description of the candidate blobs."""
    payload = []
    for index, cand in enumerate(candidates):
        payload.append({
            "label": int(cand["label"]),
            "size": int(cand["size"]),
            "centroid": [round(float(cand["cx"]), 3), round(float(cand["cy"]), 3)],
            "bbox": [int(v) for v in cand["bbox"]],
            "width": int(cand["width"]),
            "height": int(cand["height"]),
            "aspect": round(float(cand["aspect"]), 4),
            "fill": round(float(cand["fill"]), 4),
            "area_fraction": round(float(cand["area_fraction"]), 6),
            "dot_score": round(float(cand["dot_score"]), 4),
            "border_touch": bool(cand["border_touch"]),
            "selected": index in selected,
        })
    return payload


def _detection_details(result: dict, threshold: float, candidates: list[dict],
                       attempt: int, width: int, height: int) -> dict:
    metrics = result["metrics"]
    side_max = max(metrics["side_max"], 1e-9)
    details = {
        "threshold": round(float(threshold), 6),
        "attempt": int(attempt),
        "n_candidates": len(candidates),
        "selected_labels": [int(candidates[i]["label"]) for i in result["candidate_indices"]],
        "selected_sizes": [int(candidates[i]["size"]) for i in result["candidate_indices"]],
        "side_lengths_px": [round(float(v), 3) for v in metrics["sides"]],
        "side_min_px": round(float(metrics["side_min"]), 3),
        "side_max_px": round(float(metrics["side_max"]), 3),
        "side_mean_px": round(float(np.mean(metrics["sides"])), 3),
        "side_ratios": [round(float(v) / side_max, 6) for v in metrics["sides"]],
        "diagonal_ratio": round(float(metrics["diagonal_ratio"]), 6),
        "side_ratio": round(float(metrics["side_ratio"]), 6),
        "area_fraction": round(float(metrics["area"]) / float(width * height), 6),
        "bbox_aspect": round(float(max(metrics["width"], metrics["height"]) / max(
            min(metrics["width"], metrics["height"]), 1e-9)), 6),
        "squareness": round(float(metrics["squareness"]), 6),
        "mark_size_ratio": round(float(metrics.get("mark_size_ratio", 1.0)), 6),
    }
    if "inferred_point" in result:
        details["inferred_point"] = [round(float(v), 3) for v in result["inferred_point"]]
    return details


def _success_detection(result: dict, candidates: list[dict], method: str, factor: float,
                       threshold: float, attempt: int, width: int, height: int,
                       flags: list[str]) -> ScaleDetection:
    metrics = result["metrics"]
    base_quality = float(np.clip(result["quality"], 0.0, 1.0))
    quality = float(np.clip(base_quality * factor, 0.0, 1.0))
    flags = list(flags)
    if metrics.get("mark_size_ratio", 1.0) < UNEVEN_MARK_RATIO:
        flags.append("uneven scale-mark sizes")
    if quality < 0.5:
        flags.append("weak scale detection")
    selected = set(result["candidate_indices"])
    details = _detection_details(result, threshold, candidates, attempt, width, height)
    details["base_quality"] = round(base_quality, 6)
    details["image_area"] = int(width) * int(height)
    return ScaleDetection(
        found=True,
        points=[[float(x), float(y)] for x, y in result["points"]],
        quality=quality,
        method=method,
        candidates=_candidate_payload(candidates, selected),
        flags=flags,
        details=details,
    )


def _failure_detection(candidates: list[dict], threshold: float, flags: list[str],
                       stats: dict | None = None) -> ScaleDetection:
    details = {
        "threshold": round(float(threshold), 6),
        "n_candidates": len(candidates),
    }
    if stats:
        details.update({
            "n_quads": int(stats.get("n_quads", 0)),
            "n_geometry_ok": int(stats.get("n_geometry_ok", 0)),
            "best_quality": round(float(stats.get("best_quality", 0.0)), 6),
        })
    return ScaleDetection(found=False, points=[], quality=0.0, method="none",
                          candidates=_candidate_payload(candidates, set()),
                          flags=list(flags), details=details)


def _good_mark_count(candidates: list[dict]) -> int:
    """How many *prominent, dot-like* marks are among the candidates?

    Used to decide whether the 3-dot fallback is justified: it is only tried
    when exactly three genuine-looking marks remain and a fourth one is
    plausibly missing (dust specks and leaf fragments must not count).
    """
    if not candidates:
        return 0
    dot_like = [c for c in candidates if c["shape"] >= 0.70]
    if not dot_like:
        dot_like = candidates
    reference = max(c["size"] for c in dot_like)
    return sum(1 for c in candidates
               if c["shape"] >= 0.60 and c["size"] >= 0.35 * reference)


def _failure_flags(candidates: list[dict], stats: dict | None) -> list[str]:
    count = len(candidates)
    flags: list[str] = []
    if count < 4:
        flags.append(f"only {count} scale-mark candidate(s) found (need 4)")
    else:
        flags.append(f"no near-square quad among {count} scale-mark candidate(s)")
    best = float(stats.get("best_quality", 0.0)) if stats else 0.0
    if best > 0.0:
        flags.append(f"best quad quality {best:.2f} < {MIN_QUAD_QUALITY:.2f}")
    return flags


def _measure_candidates(image: np.ndarray, threshold: float, cfg: Config,
                        relaxed: bool) -> list[dict]:
    mask = imageops.binarize(image, threshold, cfg.background)
    components = ccl.label_components(mask)
    return _candidates_from_components(components, cfg, relaxed)


def detect(img, cfg: Config | None = None) -> ScaleDetection:
    """Find the four scale marks (paper protocol: black dots at square corners).

    Parameters
    ----------
    img
        Normalised RGB image (``(H, W, 3)`` uint8, longest side <= 1200), e.g.
        the output of :func:`leafbyte.imageops.resize_within`.
    cfg
        Pipeline configuration; ``cfg.threshold`` forces the binarisation
        threshold, otherwise Otsu is used.

    Returns
    -------
    ScaleDetection
        ``found=True`` with ``points`` ordered ``TL, TR, BR, BL`` and
        ``method`` one of ``"quad"`` (primary threshold, strict candidates),
        ``"quad_relaxed"`` (alternate threshold or relaxed candidate filters) or
        ``"quad_3dot"`` (fourth corner inferred; low confidence, flagged).
        ``found=False`` with ``method="none"``, ``quality=0.0`` and precise
        ``flags`` when no trustworthy quadrilateral exists.

    The routine is deterministic: no randomness, bounded candidate sets and
    bounded enumeration (``C(25, 4)`` at most), so the worst case stays well
    below the 3 s/image budget even for noisy inputs.
    """
    cfg = cfg or Config()
    image = np.asarray(img)
    if image.ndim not in (2, 3) or image.shape[0] < 8 or image.shape[1] < 8:
        return _failure_detection([], 0.0, ["image too small for scale detection"])
    height, width = int(image.shape[0]), int(image.shape[1])

    if cfg.threshold is not None:
        base_threshold = float(cfg.threshold)
    else:
        base_threshold = float(imageops.suggested_threshold(image))
    base_threshold = float(np.clip(base_threshold, 0.01, 0.99))
    direction = -1.0 if cfg.background == "black" else 1.0
    thresholds: list[tuple[int, float]] = [(0, base_threshold)]
    for step, delta in enumerate(THRESHOLD_DELTAS, start=1):
        candidate_threshold = float(np.clip(base_threshold + direction * delta, 0.01, 0.99))
        if abs(candidate_threshold - base_threshold) > 1e-9:
            thresholds.append((step, candidate_threshold))

    cache: dict[tuple[float, bool], tuple[list[dict], dict | None, dict]] = {}

    def attempt(threshold: float, relaxed: bool):
        key = (round(threshold, 6), relaxed)
        if key not in cache:
            candidates = _measure_candidates(image, threshold, cfg, relaxed)
            result, stats = _search_quads(candidates, width, height)
            cache[key] = (candidates, result, stats)
        return cache[key]

    primary_candidates, _, primary_stats = attempt(base_threshold, False)

    best: dict | None = None

    def consider(threshold: float, attempt_index: int, result: dict, candidates: list[dict],
                 method: str, factor: float) -> None:
        nonlocal best
        final = float(result["quality"]) * factor
        if final < MIN_QUAD_QUALITY:
            return
        if best is None or final > best["final_quality"] + 1e-12:
            best = {
                "threshold": threshold,
                "attempt": attempt_index,
                "result": result,
                "candidates": candidates,
                "method": method,
                "factor": factor,
                "final_quality": final,
            }

    # --- pass A: strict candidates, primary threshold then alternates -------
    for attempt_index, threshold in thresholds:
        candidates, result, _stats = attempt(threshold, False)
        if result is not None:
            before = best
            method = "quad" if attempt_index == 0 else "quad_relaxed"
            consider(threshold, attempt_index, result, candidates, method, 1.0)
            if best is not before and best is not None and method == "quad_relaxed":
                best["extra_flags"] = [f"alternate threshold {threshold:.3f}"]
        if best is not None and best["final_quality"] >= GOOD_QUAD_QUALITY:
            break

    # --- pass B: relaxed candidates (faded / fragmented dots) ---------------
    if best is None or best["final_quality"] < GOOD_QUAD_QUALITY:
        for attempt_index, threshold in thresholds[:2]:
            candidates, result, _stats = attempt(threshold, True)
            if result is None:
                continue
            before = best
            consider(threshold, attempt_index, result, candidates, "quad_relaxed",
                     RELAXED_QUALITY_FACTOR)
            if best is not before and best is not None:
                best["extra_flags"] = ["relaxed scale-mark detection"]
                if attempt_index:
                    best["extra_flags"].append(f"alternate threshold {threshold:.3f}")

    # --- pass C: three genuine dots only -> infer the fourth corner ----------
    weak_best = best is None or best["final_quality"] < GOOD_QUAD_QUALITY
    if weak_best and _good_mark_count(primary_candidates) == 3:
        candidates = primary_candidates
        if len(candidates) < 3:
            candidates = _measure_candidates(image, base_threshold, cfg, relaxed=True)
        result, _stats = _search_quads_three_dot(candidates, width, height)
        if result is not None:
            before = best
            consider(base_threshold, 0, result, candidates, "quad_3dot",
                     THREE_DOT_QUALITY_FACTOR)
            if best is not before and best is not None:
                best["extra_flags"] = ["4th scale mark inferred"]

    # --- pass D: absolute dark thresholds ----------------------------------
    # The scale dots are simply very dark.  When the paper is grey or the light
    # is uneven, a global Otsu cut can land *inside* the background intensity
    # range and swallow the dots, so retry with absolute low luma cuts.
    if best is None:
        for dark in getattr(cfg, "dot_thresholds", ()) or ():
            threshold = float(np.clip(float(dark), 0.01, 0.99))
            if abs(threshold - base_threshold) < 1e-9:
                continue
            for relaxed in (False, True):
                candidates, result, _stats = attempt(threshold, relaxed)
                if result is None:
                    continue
                before = best
                factor = RELAXED_QUALITY_FACTOR if relaxed else 1.0
                consider(threshold, 1, result, candidates, "quad_dark", factor)
                if best is not before and best is not None:
                    best["extra_flags"] = [f"dark-dot threshold {threshold:.2f}"]
            if best is not None and best["final_quality"] >= GOOD_QUAD_QUALITY:
                break

    if best is None:
        flags = _failure_flags(primary_candidates, primary_stats)
        return _failure_detection(primary_candidates, base_threshold, flags, primary_stats)

    flags = list(best.get("extra_flags", []))
    return _success_detection(best["result"], best["candidates"], best["method"],
                              best["factor"], best["threshold"], best["attempt"],
                              width, height, flags)


# ---------------------------------------------------------------------------
# rectification
# ---------------------------------------------------------------------------
def _quad_is_usable(points: np.ndarray) -> bool:
    """True if the quad has non-zero area and is convex."""
    metrics = _quad_metrics_vec(np.asarray(points, dtype=np.float64)[None, ...])
    sides = metrics["sides"][0]
    cross = metrics["cross"][0]
    if not np.all(np.isfinite(points)):
        return False
    if sides.min() <= 1.0:
        return False
    return bool(np.all(cross > 0.0) or np.all(cross < 0.0))


def rectify(img, detection: ScaleDetection | None, cfg: Config | None = None
            ) -> RectifiedImage | None:
    """Perspective-correct a detected quadrilateral into an ``S x S`` RGB image.

    ``S = min(cfg.max_dim, floor(min(bbox_w, bbox_h)))`` where the bbox is the
    bounding box of the quadrilateral (upstream ``sizeToAdjustTo``), and
    ``scale_px_length = S`` because the four dots are the corners of the known
    physical square.  The warp is a true projective transform (inverse mapping
    with bilinear sampling via :func:`leafbyte.imageops.warp_quad`).  Returns
    ``None`` for a missing detection or a degenerate quadrilateral.
    """
    cfg = cfg or Config()
    if detection is None or not detection.found:
        return None
    if len(detection.points) != 4:
        return None
    quad = order_quad(detection.points)
    if not _quad_is_usable(quad):
        return None
    size = int(imageops.rectified_size_for_quad(quad, cfg.max_dim))
    if size < 2:
        return None
    dst = [(0.0, 0.0), (float(size), 0.0), (float(size), float(size)), (0.0, float(size))]
    try:
        homography = solve_homography(quad, dst)
    except ValueError:
        return None
    warped = imageops.warp_quad(np.asarray(img), quad, size)
    return RectifiedImage(image=warped, scale_px_length=float(size),
                          homography=homography, size=size)


def detect_and_rectify(img, cfg: Config | None = None
                       ) -> tuple[ScaleDetection, RectifiedImage | None]:
    """Run :func:`detect` then :func:`rectify`; degrade cleanly on failure."""
    cfg = cfg or Config()
    detection = detect(img, cfg)
    rectified = rectify(img, detection, cfg)
    if detection.found and rectified is None:
        detection.found = False
        detection.method = "none"
        detection.quality = 0.0
        detection.flags.append("rectification failed")
    return detection, rectified
