"""Area statistics on the rectified leaf image (upstream LeafByte port).

Ported from ``ResultsViewController.useConnectedComponentsResults``
(``docs/upstream/ResultsViewController.swift``); see ``docs/ALGORITHM.md``
§3.6 and §6.4.

Semantics (must match upstream)
-------------------------------
1. The **leaf** is the largest *occupied* component (``label > 0``); in the
   absence of a user-drawing layer ``leaf_px = size(leaf)``.
2. **Consumed** pixels are the empty components (``label < 0``) that are not
   the outside-of-image background (:data:`~leafbyte.types.BACKGROUND_LABEL`)
   and whose ``empty_to_occupied`` set contains the leaf label.  Upstream:

       emptyLabelToNeighboringOccupiedLabels[key].isDisjoint(with: leafLabels)

   i.e. a hole does *not* have to be fully enclosed, touching the leaf is
   enough.  Holes smaller than ``cfg.hole_min_px`` are ignored (the default 1
   keeps upstream behaviour: every hole counts).
3. ``total_px = leaf_px + consumed_px``.
4. Components marked by ``excluded_mask`` are not counted as consumed
   (upstream's ``labelsToExclude``).  Upstream also never excludes the leaf
   itself, which is reproduced here (the leaf is never shrunk by the mask).
5. ``total_cm2 = total_px * (cfg.scale_cm / scale_px_length) ** 2`` and
   ``percent_consumed = consumed_cm2 / total_cm2 * 100``.

:class:`~leafbyte.types.ComponentsInfo` is already canonicalised to one
representative label per equivalence class, so the leaf's equivalence class is
just ``{leaf_label}`` (confirmed by the Lead: ``empty_to_occupied`` keys and
values use canonical labels).  A hole that touches the image border is labelled
:data:`~leafbyte.types.BACKGROUND_LABEL` by :mod:`leafbyte.ccl` and therefore
never consumed.

QC inputs produced here
-----------------------
``compute_results`` fills the extra ``MeasureResult`` fields the Lead added for
routing: ``leaf_touches_border``, ``competing_ratio`` (second largest occupied
/ leaf), ``hole_count`` and ``tiny_holes`` (< ``cfg.tiny_hole_px``).  They are
plain numbers; :mod:`leafbyte.quality` turns them into a confidence.

Public API (docs/ALGORITHM.md §6.4)
-----------------------------------
``analyze(rect_img, cfg, threshold=None, excluded_mask=None) -> MeasureAnalysis``
``compute_results(ci, cfg, excluded_mask=None) -> MeasureResult``
``to_cm2(result, cfg, scale_px_length) -> (total_cm2, consumed_cm2, percent)``
plus the additive helpers :func:`measure_dict`, :func:`summarize` and
:func:`otsu_separation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from leafbyte.types import BACKGROUND_LABEL, ComponentsInfo, MeasureResult

#: Second-largest occupied component / largest occupied component at or above
#: which the "which blob is the leaf" choice is considered ambiguous.  Kept in
#: sync with :mod:`leafbyte.quality`; ``calibrate.py`` may sweep both.
COMPETING_LEAF_RATIO = 0.25


@dataclass
class MeasureAnalysis:
    """Everything :func:`analyze` computed for one rectified image.

    The four documented fields are the contract; the remaining ones are
    convenience evidence for the quality stage / debugging so that batch does
    not have to recompute histograms.
    """

    threshold: float
    mask: np.ndarray
    components: ComponentsInfo
    result: MeasureResult
    #: Otsu threshold of the same image (``== threshold`` unless forced).
    otsu_threshold: float | None = None
    #: Otsu separability ``eta`` in [0, 1] at ``threshold`` (see
    #: :func:`otsu_separation`).
    otsu_separation: float | None = None
    #: 256-bin luma histogram (numpy int64).
    histogram: np.ndarray | None = None
    #: Extra numeric evidence, see :func:`summarize`.
    stats: dict[str, Any] = field(default_factory=dict)
    #: Which segmentation produced the numbers: ``"luma"`` (upstream behaviour)
    #: or ``"color"`` (excess-green, for grey/uneven backgrounds).
    mode: str = "luma"
    #: The ExG cut actually used when ``mode == "color"``.
    color_threshold: float | None = None
    #: Continuous segmentation field (``>0`` inside the leaf) and its level, used
    #: to trace sub-pixel vector contours for figures.
    contour_field: np.ndarray | None = None
    contour_level: float = 0.0


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def analyze(rect_img, cfg, threshold: float | None = None,
            excluded_mask=None, color_threshold: float | None = None) -> MeasureAnalysis:
    """Segment a rectified image and compute the upstream leaf-area statistics.

    Parameters
    ----------
    rect_img : numpy.ndarray
        ``(S, S, 3)`` uint8 RGB image (``RectifiedImage.image``).  Grayscale
        ``(S, S)`` and float arrays (``0..1`` or ``0..255``) are tolerated.
    cfg : leafbyte.config.Config
    threshold : float, optional
        Force the luma threshold in ``[0, 1]``.  ``None`` falls back to
        ``cfg.threshold`` when set (a fixed threshold) and otherwise to Otsu
        on this image.
    excluded_mask : numpy.ndarray, optional
        Boolean ``(S, S)`` bitmap of reviewer-excluded pixels; those
        components are not counted as consumed.
    color_threshold : float, optional
        ExG cut for the colour segmentation (``cfg.color_threshold`` by
        default).  Only used when the colour path is selected; the stability
        metric perturbs it.

    Returns
    -------
    MeasureAnalysis
        ``threshold`` / ``mask`` / ``components`` / ``result`` plus
        ``otsu_threshold``, ``otsu_separation``, ``histogram`` and ``stats``.

    Raises
    ------
    ImportError
        If :mod:`leafbyte.imageops` / :mod:`leafbyte.ccl` (Lead-owned) are not
        installed yet.
    ValueError
        If ``rect_img`` cannot be interpreted as an RGB image.
    """
    img = _to_uint8_rgb(rect_img)

    # Lazy import: the modules are Lead-owned and this file must stay testable
    # with hand-built ComponentsInfo fixtures before they exist.
    from leafbyte import imageops as _imageops
    from leafbyte import ccl as _ccl

    hist = np.asarray(_imageops.luma_histogram(_imageops.luma(img)),
                      dtype=np.int64)
    otsu = float(_imageops.otsu_threshold(hist))
    fixed = getattr(cfg, "threshold", None)
    if threshold is not None:
        thr = float(threshold)
    elif fixed is not None:
        thr = float(fixed)
    else:
        thr = otsu

    mask = np.asarray(
        _imageops.binarize(img, thr, getattr(cfg, "background", "white")),
        dtype=bool,
    )
    components = _ccl.label_components(mask)
    result = compute_results(components, cfg, excluded_mask=excluded_mask)
    mode = "luma"
    used_color: float | None = None
    filled_mask = None

    leaf_mode = str(getattr(cfg, "leaf_mode", "auto") or "auto").lower()
    if leaf_mode in {"color", "auto"}:
        probe_threshold = float(getattr(cfg, "color_probe_competing_ratio", 0.05) or 0.05)
        worth_probing = (
            leaf_mode == "color"
            or not _is_leaf_like(result, mask)
            or result.competing_ratio > probe_threshold
            or _speckle_only(result, cfg)
        )
        colour = _analyse_colour(img, cfg, excluded_mask, color_threshold) if worth_probing else None
        if colour is not None:
            take = leaf_mode == "color" or _should_use_colour(result, mask, colour[2], colour[0], cfg)
            if take:
                mask, components, result, used_color, filled_mask = colour
                mode = "color"

    # continuous field for sub-pixel contour tracing (>0 inside the leaf)
    if mode == "color" and used_color is not None:
        contour_field = _imageops.exg(img) - float(used_color)
        if filled_mask is not None and np.any(filled_mask):
            contour_field = np.where(filled_mask, float(np.abs(contour_field).max()) + 1e3,
                                     contour_field)
    else:
        contour_field = float(thr) - _imageops.luma(img)

    stats = summarize(components, result, cfg)
    stats["segmentation_mode"] = mode
    if used_color is not None:
        stats["color_threshold"] = used_color
    return MeasureAnalysis(
        threshold=thr,
        mask=mask,
        components=components,
        result=result,
        otsu_threshold=otsu,
        otsu_separation=otsu_separation(hist, thr),
        histogram=hist,
        stats=stats,
        mode=mode,
        color_threshold=used_color,
        contour_field=contour_field,
        contour_level=0.0,
    )


def _needs_colour_fallback(result: MeasureResult, mask: np.ndarray) -> bool:
    """True when the luma segmentation does not look like a leaf on paper."""
    if result.leaf_label is None:
        return True
    if result.leaf_touches_border:
        return True
    if mask.size == 0:
        return True
    fraction = float(result.leaf_px) / float(mask.size)
    return fraction > 0.15 or fraction < 0.002


def _speckle_only(result: MeasureResult, cfg) -> bool:
    """Consumed area made exclusively of sub-``tiny_hole_px`` specks."""
    if result.consumed_px <= 0 or result.hole_count <= 0:
        return False
    if result.tiny_holes != result.hole_count:
        return False
    if result.total_px <= 0:
        return False
    return result.consumed_px / float(result.total_px) < 0.02


def _should_use_colour(luma_result, luma_mask, colour_result, colour_mask, cfg) -> bool:
    """Decide whether the excess-green segmentation explains the photo better.

    The luma/white-paper segmentation stays the default (it is what upstream
    does and what the published reference values were produced with).  Colour is
    only substituted when it is demonstrably cleaner:

    * the luma result is not leaf-like at all (background leaking in, leaf cut
      off by the crop), or
    * a second luma component competes with the leaf (>10 % of its size) - the
      classic symptom of a grey / vignetted background, or
    * the luma "consumed" area consists only of 1-2 px speckles and the colour
      segmentation reports a clean leaf with none.
    """
    if not _is_leaf_like(colour_result, colour_mask):
        return False
    if colour_result.competing_ratio > 0.05:
        return False
    if not _is_leaf_like(luma_result, luma_mask):
        return True
    if luma_result.competing_ratio > 0.10:
        return True
    return _speckle_only(luma_result, cfg) and not _speckle_only(colour_result, cfg) \
        and colour_result.consumed_px == 0


def _is_leaf_like(result: MeasureResult, mask: np.ndarray) -> bool:
    if result.leaf_label is None or mask.size == 0:
        return False
    if result.leaf_touches_border:
        return False
    fraction = float(result.leaf_px) / float(mask.size)
    return 5e-4 <= fraction <= 0.35


def _analyse_colour(img, cfg, excluded_mask=None, color_threshold=None):
    """Segment the leaf by excess-green; returns ``(mask, components, result, cut)``."""
    from leafbyte import imageops as _imageops
    from leafbyte import ccl as _ccl

    cut = color_threshold if color_threshold is not None else getattr(cfg, "color_threshold", None)
    if cut is None:
        # adaptive: Otsu on the ExG channel of this very image
        cut = _imageops.exg_threshold(img)
        bounds = getattr(cfg, "color_threshold_range", None) or (6.0, 40.0)
        cut = float(np.clip(cut, float(bounds[0]), float(bounds[1])))
    cut = float(cut)
    mask = np.asarray(_imageops.exg(img) > cut, dtype=bool)
    radius = int(getattr(cfg, "mask_closing", 2) or 0)
    if radius > 0:
        mask = _imageops.close_mask(mask, radius)
    if not mask.any():
        return None
    components = _ccl.label_components(mask)
    result = compute_results(components, cfg, excluded_mask=excluded_mask)
    filled_mask = None

    # --- dark interior patches are tissue, not eaten holes -----------------
    # The ExG criterion calls anything non-green "empty".  A necrotic lesion, a
    # fold shadow or the dark side of a crease is non-green but the lamina is
    # still there, and only background (paper) showing through is a real hole.
    # So holes whose interior is clearly darker than the paper are filled back
    # into the leaf; only paper-coloured holes count as consumed area.
    if getattr(cfg, "hole_validate_background", True):
        paper_pixels = rect_pixels = None
        outside = components.label_map == BACKGROUND_LABEL
        if not outside.any():
            outside = ~mask
        paper = img[outside] if outside.any() else None
        if paper is not None and paper.size:
            paper_luma = float(np.mean(_luma_of(paper)))
        else:
            paper_luma = 0.75
        dark_limit = paper_luma * float(getattr(cfg, "hole_dark_ratio", 0.55))
        filled_labels: list[int] = []
        filled_px = 0
        kept: list[dict[str, Any]] = []
        for hole in result.holes:
            label = int(hole["label"])
            pix = img[components.label_map == label]
            if pix.size == 0:
                kept.append(hole)
                continue
            if float(np.mean(_luma_of(pix))) < dark_limit:
                filled_labels.append(label)
                filled_px += int(hole["area_px"])
            else:
                kept.append(hole)
        if filled_labels:
            filled_mask = np.isin(components.label_map, filled_labels)
            mask = mask | filled_mask
            components = _ccl.label_components(mask)
            previous_total = result.total_px
            result = compute_results(components, cfg, excluded_mask=excluded_mask)
            result.flags.append("dark_patches_filled")
            # total area (leaf + eaten) must not change: tissue was moved from
            # "consumed" into "leaf"
            if result.total_px != previous_total:
                result.flags.append("filled_area_mismatch")
            result.stats_filled_px = filled_px
    return mask, components, result, cut, filled_mask


def compute_results(ci: ComponentsInfo, cfg,
                    excluded_mask=None) -> MeasureResult:
    """Apply the upstream area rules to a :class:`ComponentsInfo`.

    ``leaf_px`` = size of the largest occupied component;
    ``consumed_px`` = summed size of the non-background empty components that
    neighbour the leaf (minus excluded components);
    ``total_px = leaf_px + consumed_px``.

    Never raises on a degenerate/empty component map: it returns a zero result
    with a flag (``no_leaf`` / ``empty_components``) instead.
    """
    result = MeasureResult()
    if ci is None or getattr(ci, "label_map", None) is None:
        result.flags.append("no_components")
        return result
    label_map = np.asarray(ci.label_map)
    if label_map.ndim != 2 or label_map.size == 0:
        result.flags.append("empty_components")
        return result

    occupied = ci.occupied_labels()
    if not occupied:
        result.flags.append("no_leaf")
        return result

    leaf = int(occupied[0])
    leaf_px = int(ci.size(leaf))
    result.leaf_label = leaf
    result.leaf_px = leaf_px
    leaf_class = {leaf}

    # --- excluded components (upstream labelsToExclude) ------------------
    excluded_labels: set[int] = set()
    excluded_px = 0
    aligned = _align_bool_mask(excluded_mask, label_map.shape)
    if aligned is not None:
        excluded_arr, mismatch = aligned
        if mismatch:
            result.flags.append("excluded_mask_shape_mismatch")
        excluded_px = int(np.count_nonzero(excluded_arr))
        if excluded_px:
            for raw in np.unique(label_map[excluded_arr]):
                label = int(raw)
                if label != 0:
                    excluded_labels.add(label)
            # Upstream filters `$0 != leafLabel`, so the leaf itself is never
            # dropped by an exclusion mark.
            if leaf in excluded_labels:
                result.flags.append("excluded_leaf_region")

    # --- consumed holes ---------------------------------------------------
    if ci.empty_labels() and not ci.empty_to_occupied:
        result.flags.append("missing_adjacency")
    min_hole_px = max(1, int(getattr(cfg, "hole_min_px", 1) or 1))
    consumed = 0
    holes: list[dict[str, Any]] = []
    for empty in ci.empty_labels():
        size = int(ci.size(empty))
        if size < min_hole_px:
            continue
        if empty in excluded_labels:
            continue
        neighbours = ci.empty_to_occupied.get(empty) or ()
        if not leaf_class.intersection(neighbours):
            continue
        consumed += size
        holes.append({
            "label": int(empty),
            "area_px": size,
            "centroid": _component_centroid(label_map, empty),
        })

    result.consumed_px = int(consumed)
    result.total_px = int(leaf_px + consumed)
    result.holes = holes
    result.excluded_px = int(excluded_px)

    # --- QC inputs (see MeasureResult docstring) --------------------------
    result.leaf_touches_border = _touches_border(label_map, leaf)
    if result.leaf_touches_border:
        result.flags.append("leaf_touches_border")
    if len(occupied) >= 2 and leaf_px > 0:
        result.competing_ratio = float(ci.size(occupied[1])) / float(leaf_px)
        if result.competing_ratio >= COMPETING_LEAF_RATIO:
            result.flags.append("competing_leaf")
    result.hole_count = len(holes)
    tiny_px = int(getattr(cfg, "tiny_hole_px", 25) or 0)
    result.tiny_holes = sum(1 for h in holes if h["area_px"] < tiny_px)
    if result.tiny_holes:
        result.flags.append("tiny_holes")
    if not ci.empty_labels() and BACKGROUND_LABEL not in ci.label_to_size:
        result.flags.append("no_empty_components")
    return result


def to_cm2(result: MeasureResult, cfg,
           scale_px_length: float) -> tuple[float, float, float]:
    """Convert pixel areas to cm².

    ``units_per_pixel = cfg.scale_cm / scale_px_length`` and
    ``area_cm2 = units_per_pixel ** 2 * pixels`` (upstream
    ``convertPixelsToUnits2``), i.e. ``scale_px_length`` is the pixel length of
    a mark that is ``cfg.scale_cm`` long after rectification.

    Returns ``(total_cm2, consumed_cm2, percent_consumed)``.

    Raises
    ------
    ValueError
        If ``scale_px_length`` is missing or not positive: absolute area is
        undefined without a scale (upstream ``fatalError``s here).  Callers
        must guard on ``scale.found`` / ``scale_px_length``.
    """
    try:
        length = float(scale_px_length)
    except (TypeError, ValueError):
        length = 0.0
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(
            "to_cm2 needs a positive scale_px_length "
            f"(got {scale_px_length!r}); check scale.found before converting")
    units_per_pixel = float(getattr(cfg, "scale_cm", 17.0)) / length
    area_per_pixel = units_per_pixel * units_per_pixel
    total = area_per_pixel * float(int(result.total_px))
    consumed = area_per_pixel * float(int(result.consumed_px))
    percent = 100.0 * consumed / total if total else 0.0
    return total, consumed, percent


# ---------------------------------------------------------------------------
# helpers used by batch / quality / tests
# ---------------------------------------------------------------------------
def summarize(ci: ComponentsInfo, result: MeasureResult, cfg=None) -> dict:
    """Numeric evidence about the component map, for quality and debugging.

    Returns a flat dict (all JSON-serialisable):
    ``occupied_sizes`` (descending), ``leaf_touches_border``,
    ``competing_ratio``, ``interior_competing_ratio``, ``n_holes``,
    ``hole_px``, ``tiny_holes``, ``tiny_hole_px``, ``background_px``.
    """
    stats: dict[str, Any] = {
        "occupied_sizes": [],
        "leaf_touches_border": bool(getattr(result, "leaf_touches_border", False)),
        "competing_ratio": float(getattr(result, "competing_ratio", 0.0) or 0.0),
        "interior_competing_ratio": 0.0,
        "n_holes": int(getattr(result, "hole_count", len(result.holes)) or 0),
        "hole_px": int(result.consumed_px),
        "tiny_holes": int(getattr(result, "tiny_holes", 0) or 0),
        "tiny_hole_px": 0,
        "background_px": int(ci.size(BACKGROUND_LABEL)) if ci is not None else 0,
    }
    if ci is None:
        return stats
    label_map = np.asarray(ci.label_map)
    occupied = ci.occupied_labels()
    stats["occupied_sizes"] = [int(ci.size(l)) for l in occupied]
    tiny_px = int(getattr(cfg, "tiny_hole_px", 25) or 0)
    tiny = [int(h["area_px"]) for h in result.holes
            if int(h["area_px"]) < tiny_px]
    stats["tiny_holes"] = len(tiny)
    stats["tiny_hole_px"] = int(sum(tiny))
    leaf = result.leaf_label
    if leaf is not None:
        stats["leaf_touches_border"] = _touches_border(label_map, leaf)
        leaf_px = max(1, int(ci.size(leaf)))
        if len(occupied) >= 2:
            stats["competing_ratio"] = float(ci.size(occupied[1])) / leaf_px
            for candidate in occupied[1:]:
                if not _touches_border(label_map, candidate):
                    stats["interior_competing_ratio"] = (
                        float(ci.size(candidate)) / leaf_px)
                    break
    return stats


def measure_dict(analysis: MeasureAnalysis, include_stats: bool = True) -> dict:
    """Build the ``"measure"`` sub-dict of an analysis record (§7 schema).

    Contains every documented ``MeasureResult`` field (``total_px``,
    ``consumed_px``, ``leaf_px``, ``holes``, ``flags``, ``excluded_px``,
    ``leaf_touches_border``, ``competing_ratio``, ``hole_count``,
    ``tiny_holes``) plus, unless disabled, the :func:`summarize` extras.  Batch
    may use this or build the dict itself; :func:`leafbyte.quality.evaluate`
    understands either.
    """
    result = analysis.result
    out = {
        "total_px": int(result.total_px),
        "consumed_px": int(result.consumed_px),
        "leaf_px": int(result.leaf_px),
        "leaf_label": result.leaf_label,
        "holes": [
            {
                "label": int(h["label"]),
                "area_px": int(h["area_px"]),
                "centroid": ([float(h["centroid"][0]), float(h["centroid"][1])]
                             if h.get("centroid") is not None else None),
            }
            for h in result.holes
        ],
        "excluded_px": int(result.excluded_px),
        "leaf_touches_border": bool(result.leaf_touches_border),
        "competing_ratio": float(result.competing_ratio),
        "hole_count": int(result.hole_count),
        "tiny_holes": int(result.tiny_holes),
        "flags": list(result.flags),
    }
    out["mode"] = getattr(analysis, "mode", "luma")
    filled = getattr(result, "stats_filled_px", None)
    if filled:
        out["filled_dark_px"] = int(filled)
    if getattr(analysis, "color_threshold", None) is not None:
        out["color_threshold"] = float(analysis.color_threshold)
    if include_stats:
        stats = dict(analysis.stats or {})
        for key, value in stats.items():
            out.setdefault(key, value)
    return out


def otsu_separation(hist, threshold) -> float:
    """Otsu's separability ``eta`` in ``[0, 1]`` at ``threshold``.

    ``eta = between-class variance / total variance`` of a 256-bin luma
    histogram.  1.0 = perfectly bimodal split, 0.0 = the threshold does not
    separate the histogram at all.  Bucket ``i`` represents luma
    ``i / (n_buckets - 1)``; ``threshold`` is in ``[0, 1]`` and pixels with
    luma ``< threshold`` are class 0 (white-background convention; for a black
    background the two classes swap and ``eta`` is unchanged).
    """
    hist_arr = np.asarray(hist, dtype=np.float64).ravel()
    if hist_arr.size < 2:
        return 0.0
    total = float(hist_arr.sum())
    if total <= 0.0:
        return 0.0
    probabilities = hist_arr / total
    buckets = np.arange(hist_arr.size, dtype=np.float64)
    mean = float((probabilities * buckets).sum())
    variance = float((probabilities * (buckets - mean) ** 2).sum())
    if variance <= 0.0:
        return 0.0
    try:
        cut_value = float(threshold)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(cut_value):
        return 0.0
    cut = int(np.floor(np.clip(cut_value, 0.0, 1.0) * (hist_arr.size - 1)))
    cut = min(max(cut, 0), hist_arr.size - 2)
    weight0 = float(probabilities[: cut + 1].sum())
    weight1 = 1.0 - weight0
    if weight0 <= 0.0 or weight1 <= 0.0:
        return 0.0
    mean0 = float((probabilities[: cut + 1] * buckets[: cut + 1]).sum()) / weight0
    mean1 = float((probabilities[cut + 1:] * buckets[cut + 1:]).sum()) / weight1
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    return float(np.clip(between / variance, 0.0, 1.0))


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------
def _luma_of(pixels: np.ndarray) -> np.ndarray:
    """Luma (0..1) of an (N, 3) uint8 array of pixel colours."""
    array = np.asarray(pixels, dtype=np.float64)
    return (0.299 * array[:, 0] + 0.587 * array[:, 1] + 0.114 * array[:, 2]) / 255.0


def _to_uint8_rgb(img) -> np.ndarray:
    """Coerce common image representations to ``(H, W, 3)`` uint8 RGB."""
    array = np.asarray(img)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(
            f"rect_img must be (H, W, 3) RGB, got shape {array.shape!r}")
    if array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
        array = np.clip(np.nan_to_num(array), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _align_bool_mask(mask, shape: tuple[int, int]):
    """Return ``(bool_array, shape_mismatch)`` or ``None`` for ``mask=None``."""
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.ndim == 3:
        array = np.any(array, axis=-1)
    if array.ndim != 2:
        return np.zeros(shape, dtype=bool), True
    array = array.astype(bool, copy=False)
    if array.shape == shape:
        return array, False
    height, width = shape
    out = np.zeros(shape, dtype=bool)
    copy_h = min(height, array.shape[0])
    copy_w = min(width, array.shape[1])
    out[:copy_h, :copy_w] = array[:copy_h, :copy_w]
    return out, True


def _touches_border(label_map: np.ndarray, label: int) -> bool:
    """True if any pixel of ``label`` lies on the edge of ``label_map``."""
    if label_map.size == 0:
        return False
    return bool(
        np.any(label_map[0, :] == label)
        or np.any(label_map[-1, :] == label)
        or np.any(label_map[:, 0] == label)
        or np.any(label_map[:, -1] == label)
    )


def _component_centroid(label_map: np.ndarray,
                        label: int) -> list[float] | None:
    """``[x, y]`` centroid of every pixel equal to ``label`` (None if absent)."""
    rows, cols = np.nonzero(label_map == label)
    if rows.size == 0:
        return None
    return [float(cols.mean()), float(rows.mean())]
