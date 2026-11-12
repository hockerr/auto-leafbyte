"""Quality metrics, confidence and human-review routing.

Receives one analysis record (dict, schema ``docs/ALGORITHM.md`` §7, as filled
by ``batch.py``) and augments it in place with:

``metrics``
    ``{name: score in [0, 1]}`` for the eight metrics described below.
``confidence``
    Weighted mean of ``metrics`` in ``[0, 1]``, capped when a blocking problem
    is present.  Always returned, even for failed/partial records.
``status``
    ``"auto"`` (accepted without a human), ``"review"`` (goes to the human
    review queue) or ``"failed"`` (pipeline could not measure it at all).
``flags``
    Machine-readable problem tokens (``"leaf_touches_border"``, ...).
``reason``
    One human-readable sentence explaining the decision.
``quality_details``
    Raw evidence per metric (debugging / ``calibrate.py``).

Design rules
------------
* **Honest:** missing evidence lowers confidence, never raises it.
* **Conservative:** a record with no usable scale, a leaf on the crop border,
  an area unstable under ±threshold, an ambiguous leaf choice, etc. is NEVER
  ``auto`` -- status is forced to ``review`` even if the numbers look nice.
  ``scale.found == False`` and ``measure.leaf_touches_border == True`` are the
  user-mandated hard cases.
* **Never crashes:** every metric is computed defensively; a partial record
  (e.g. the scale was not found) yields ``failed``/``review``, not an
  exception.
* **Calibratable:** every weight and cut-point is a module-level constant and
  :func:`evaluate` accepts a ``weights`` override.  ``calibrate.py`` can sweep
  ``WEIGHTS`` (or use :func:`set_params`/``DEFAULTS``) against
  ``data/paper/ground_truth.csv``; :func:`combine` is exported so candidate
  weight sets can be scored cheaply from a cached ``metrics`` dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from leafbyte.measure import otsu_separation

# ---------------------------------------------------------------------------
# weights (sum to 1.0); calibrate.py may replace or perturb these
# ---------------------------------------------------------------------------
WEIGHTS: dict[str, float] = {
    "scale_quality": 0.22,
    "threshold_quality": 0.15,
    "stability": 0.18,
    "leaf_border_touch": 0.12,
    "competing_leaf": 0.08,
    "hole_noise": 0.08,
    "percent_extremes": 0.05,
    "leaf_fraction": 0.12,
}

# ---------------------------------------------------------------------------
# tunable constants (calibrate.py may sweep; see DEFAULTS/set_params)
# ---------------------------------------------------------------------------
#: Score used when a metric cannot be computed from the record at all.
MISSING_EVIDENCE = 0.50
#: Score for a scale detection that reports ``found=True`` but no quality.
SCALE_FOUND_NO_QUALITY = 0.60
#: Neutral-ish scores when the border/competing evidence is absent: absence is
#: not proof of a problem, but it is also not proof of correctness.
UNKNOWN_BORDER = 0.55
UNKNOWN_COMPETING = 0.55
#: Weight of ``separation`` vs ``bimodality`` inside the threshold metric.
THRESHOLD_SEPARATION_WEIGHT = 0.72
#: Relative total-area change at which the stability score is ~0.5 (multiple
#: of ``cfg.stability_tolerance``); higher = more forgiving.
STABILITY_DECAY = 1.0
#: rel_range above ``STABILITY_REVIEW_FACTOR * cfg.stability_tolerance`` is a
#: hard review blocker.
STABILITY_REVIEW_FACTOR = 1.0
#: Second-largest-component / leaf ratio at which the leaf choice is ambiguous.
COMPETING_LEAF_RATIO = 0.25
#: Ratio range over which the competing-leaf score decays from 1 to 0.
COMPETING_LEAF_SOFT = 0.50
#: Fraction of ``near_threshold_fraction`` that halves the threshold score.
NEAR_THRESHOLD_SOFT = 0.10
#: Threshold quality below this blocks auto-accept.
THRESHOLD_MIN_FOR_AUTO = 0.35
#: Bimodality coefficient at/above which a histogram counts as bimodal
#: (classic 5/9 cut).
BIMODAL_GOOD = 0.555
#: Percent consumed above this is suspicious (score starts decaying).
PERCENT_MAX_PLAUSIBLE = 85.0
#: Percent consumed above this is a hard review blocker.
PERCENT_REVIEW_MAX = 90.0
#: Plausible leaf area as a fraction of the rectified scale square
#: (total_cm2 / scale_cm**2; paper data is 4.8-60 cm2 -> 1.7-21 %).
LEAF_FRACTION_MIN = 0.01
LEAF_FRACTION_MAX = 0.35
#: Width of the score fade-out outside the plausible leaf-fraction window.
LEAF_FRACTION_FADE = 0.20
#: Leaf-fraction metric below this is a hard review blocker.
LEAF_FRACTION_MIN_SCORE = 0.15
#: Hole-noise metric below this is a hard review blocker.
HOLE_NOISE_MIN_FOR_AUTO = 0.20
#: Confidence ceiling for records with a hard blocker (forces review).
HARD_BLOCK_CONFIDENCE = 0.55
#: Confidence ceiling for fatal records, as a fraction of ``cfg.review_min``.
FATAL_CONFIDENCE_FRACTION = 0.50

_METRICS: list[tuple[str, Any]] = []

_HUMAN: dict[str, str] = {
    "analysis_failed": "analysis stage failed",
    "analysis_error": "analysis raised an error",
    "no_measurement": "no measurement in record",
    "no_leaf": "no leaf/occupied component found",
    "scale_not_found": "scale not found; absolute area unavailable",
    "relative_only": "relative measurement only (no scale); needs a human",
    "scale_invalid_length": "scale length is missing/non-positive",
    "leaf_touches_border": "leaf touches the crop border (area is truncated)",
    "area_unstable": "area unstable under +-threshold",
    "weak_threshold_separation": "weak leaf/background separation",
    "competing_leaf": "another large component competes with the leaf",
    "tiny_hole_noise": "consumed area dominated by tiny speckle holes",
    "percent_implausible": "implausible percent-consumed",
    "leaf_fraction_implausible": "leaf area implausible for the scale square",
}

# lowercase name -> module constant, for set_params()/DEFAULTS
_PARAMS: dict[str, str] = {
    "weights": "WEIGHTS",
    "missing_evidence": "MISSING_EVIDENCE",
    "scale_found_no_quality": "SCALE_FOUND_NO_QUALITY",
    "unknown_border": "UNKNOWN_BORDER",
    "unknown_competing": "UNKNOWN_COMPETING",
    "threshold_separation_weight": "THRESHOLD_SEPARATION_WEIGHT",
    "stability_decay": "STABILITY_DECAY",
    "stability_review_factor": "STABILITY_REVIEW_FACTOR",
    "competing_leaf_ratio": "COMPETING_LEAF_RATIO",
    "competing_leaf_soft": "COMPETING_LEAF_SOFT",
    "near_threshold_soft": "NEAR_THRESHOLD_SOFT",
    "threshold_min_for_auto": "THRESHOLD_MIN_FOR_AUTO",
    "bimodal_good": "BIMODAL_GOOD",
    "percent_max_plausible": "PERCENT_MAX_PLAUSIBLE",
    "percent_review_max": "PERCENT_REVIEW_MAX",
    "leaf_fraction_min": "LEAF_FRACTION_MIN",
    "leaf_fraction_max": "LEAF_FRACTION_MAX",
    "leaf_fraction_fade": "LEAF_FRACTION_FADE",
    "leaf_fraction_min_score": "LEAF_FRACTION_MIN_SCORE",
    "hole_noise_min_for_auto": "HOLE_NOISE_MIN_FOR_AUTO",
    "hard_block_confidence": "HARD_BLOCK_CONFIDENCE",
    "fatal_confidence_fraction": "FATAL_CONFIDENCE_FRACTION",
}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def evaluate(record: dict, cfg, weights: dict | None = None) -> dict:
    """Score one analysis record and decide ``auto`` / ``review`` / ``failed``.

    ``record`` is mutated in place when it is a dict (and returned); otherwise
    a new dict is returned.  The added keys are documented in the module
    docstring.

    ``weights`` (optional) is a *partial* override of :data:`WEIGHTS` -- e.g.
    ``{"stability": 0.3}`` -- merged over the defaults, so ``calibrate.py`` can
    sweep one weight at a time.  It only affects the confidence value, not the
    hard blockers.

    The function never raises: any internal error results in
    ``status="failed"`` with a ``quality_error`` flag.
    """
    out: dict = record if isinstance(record, dict) else dict(record or {})
    flags: list[str] = []
    _extend_flags(flags, out.get("flags"))
    details: dict[str, Any] = {}

    try:
        metrics: dict[str, float] = {}
        for name, func in _metric_table():
            try:
                score, metric_flags, evidence = func(out, cfg)
            except Exception as exc:  # never crash on a partial record
                score, metric_flags, evidence = (
                    MISSING_EVIDENCE, [f"{name}_error"], {"error": repr(exc)})
            metrics[name] = _clip(score)
            _extend_flags(flags, metric_flags)
            details[name] = evidence if isinstance(evidence, dict) else {}

        confidence = combine(metrics, weights)
        fatal, blockers = _gate(out, cfg, metrics, details)
        _extend_flags(flags, fatal)
        _extend_flags(flags, blockers)

        auto_min, review_min = _status_thresholds(cfg)
        if fatal:
            status = "failed"
            confidence = min(confidence, review_min * FATAL_CONFIDENCE_FRACTION)
        elif blockers:
            # Never auto: a known-problematic record always goes to a human.
            status = "review"
            confidence = min(confidence, HARD_BLOCK_CONFIDENCE)
        elif confidence >= auto_min:
            status = "auto"
        elif confidence >= review_min:
            status = "review"
        else:
            status = "failed"

        out["metrics"] = {name: round(float(value), 6)
                          for name, value in metrics.items()}
        out["confidence"] = round(float(_clip(confidence)), 6)
        out["status"] = status
        out["flags"] = flags
        out["reason"] = _reason(status, float(out["confidence"]),
                                fatal, blockers)
        out["quality_details"] = details
    except Exception as exc:  # last-resort safety net: still return a record
        out["metrics"] = {}
        out["confidence"] = 0.0
        out["status"] = "failed"
        _extend_flags(flags, ["quality_error"])
        out["flags"] = flags
        out["reason"] = f"failed (quality evaluation error): {exc!r}"
        out["quality_details"] = {"error": repr(exc)}
    return out


def combine(metrics: dict, weights: dict | None = None) -> float:
    """Weighted mean of the metric scores, clamped to ``[0, 1]``.

    ``weights`` is a partial override merged over :data:`WEIGHTS` (unknown
    metric names are ignored).  Returns :data:`MISSING_EVIDENCE` when nothing
    can be combined.
    """
    merged = dict(WEIGHTS)
    if weights:
        merged.update({k: v for k, v in weights.items() if k in WEIGHTS})
    numerator = 0.0
    denominator = 0.0
    for name, weight in merged.items():
        value = metrics.get(name)
        if value is None:
            continue
        try:
            weight_f = float(weight)
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(weight_f) or weight_f <= 0.0:
            continue
        numerator += weight_f * value_f
        denominator += weight_f
    if denominator <= 0.0:
        return MISSING_EVIDENCE
    return _clip(numerator / denominator)


def bimodality_coefficient(hist) -> float:
    """Sarle's bimodality coefficient of a histogram (0..1).

    ``BC = (skew**2 + 1) / kurtosis`` using non-excess moments; bimodal
    histograms exceed the classic 5/9 threshold, unimodal/normal ones are
    around 0.17-0.35.  Used only as a fallback when ``threshold_metrics`` is
    missing from the record.
    """
    array = np.asarray(hist, dtype=np.float64).ravel()
    total = float(array.sum())
    if array.size < 2 or total <= 0.0:
        return 0.0
    probabilities = array / total
    index = np.arange(array.size, dtype=np.float64)
    mean = float((probabilities * index).sum())
    variance = float((probabilities * (index - mean) ** 2).sum())
    if variance <= 0.0:
        return 0.0
    std = np.sqrt(variance)
    skew = float((probabilities * ((index - mean) / std) ** 3).sum())
    kurtosis = float((probabilities * ((index - mean) / std) ** 4).sum())
    if kurtosis <= 0.0:
        return 0.0
    return float((skew * skew + 1.0) / kurtosis)


def get_params() -> dict:
    """Current values of every tunable (weights included)."""
    return {name: globals()[constant] for name, constant in _PARAMS.items()}


def set_params(**kwargs) -> dict:
    """Set tunables by lower-case name (``weights=``, ``stability_decay=``...).

    Returns the updated :func:`get_params`.  Unknown names raise ``KeyError``
    so a calibrate sweep cannot silently misspell a knob.
    """
    globals_ = globals()
    unknown = [key for key in kwargs if key not in _PARAMS]
    if unknown:
        raise KeyError(f"unknown quality parameter(s): {unknown}")
    for key, value in kwargs.items():
        globals_[_PARAMS[key]] = value
    return get_params()


#: Snapshot of the default tunables (kept in sync by :func:`get_params`).
DEFAULTS = get_params()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _metric_scale(record: dict, cfg) -> tuple[float, list[str], dict]:
    scale = record.get("scale")
    if not isinstance(scale, dict):
        return MISSING_EVIDENCE, [], {"found": None}
    found = scale.get("found")
    if found is None:  # tolerate a record that only has a valid length
        found = _num(scale.get("scale_px_length")) is not None
    if not found:
        return 0.0, ["scale_not_found"], {"found": False}
    length = _num(scale.get("scale_px_length"))
    if length is None or length <= 0.0:
        return 0.0, ["scale_invalid_length"], {"found": True,
                                               "scale_px_length": length}
    method = str(scale.get("method") or "")
    quality = _num(scale.get("quality"))
    detail = {"found": True, "quality": quality, "method": method,
              "scale_px_length": length}
    if method == "manual":
        return 1.0, [], detail
    if quality is None:
        return SCALE_FOUND_NO_QUALITY, [], detail
    score = _clip(quality)
    if method == "quad_relaxed":
        score *= 0.9
    elif method in {"quad_dark", "quad_3dot"}:
        # found with a fallback path (absolute darkness / inferred corner)
        score *= 0.90 if method == "quad_dark" else 0.75
    return _clip(score), [], detail


def _metric_threshold(record: dict, cfg) -> tuple[float, list[str], dict]:
    separation = bimodality = near = None
    threshold_metrics = record.get("threshold_metrics")
    if isinstance(threshold_metrics, dict):
        separation = _num(threshold_metrics.get("separation"))
        bimodality = _num(threshold_metrics.get("bimodality"))
        near = _num(threshold_metrics.get("near_threshold_fraction"))
    if separation is None:
        separation = _num(record.get("otsu_separation"))
    threshold = _num(record.get("threshold"))
    histogram = record.get("histogram")
    if histogram is None:
        histogram = record.get("luma_histogram")
    if bimodality is None and histogram is not None:
        try:
            bimodality = bimodality_coefficient(histogram)
        except Exception:
            bimodality = None
    if separation is None and histogram is not None and threshold is not None:
        try:
            separation = otsu_separation(histogram, threshold)
        except Exception:
            separation = None

    parts: list[tuple[float, float]] = []
    if separation is not None:
        parts.append((_clip(separation), THRESHOLD_SEPARATION_WEIGHT))
    if bimodality is not None:
        normalized = _clip((bimodality - 0.20) / (BIMODAL_GOOD - 0.20))
        parts.append((normalized, 1.0 - THRESHOLD_SEPARATION_WEIGHT))
    detail = {"separation": separation, "bimodality": bimodality,
              "near_threshold_fraction": near}
    if parts:
        total_weight = sum(weight for _, weight in parts)
        score = sum(value * weight for value, weight in parts) / total_weight
        if near is not None:
            # Lots of pixels sitting on the fence => the exact cut matters.
            score *= 1.0 - 0.5 * _clip(near / NEAR_THRESHOLD_SOFT)
        return _clip(score), [], detail

    otsu = _num(record.get("otsu_threshold"))
    if threshold is not None and otsu is not None:
        # Weak prior only: equality proves the pipeline used Otsu, not that
        # the image is bimodal.
        difference = abs(threshold - otsu)
        score = 0.55 + 0.20 * _clip(1.0 - difference / 0.10)
        detail["threshold_minus_otsu"] = threshold - otsu
        return _clip(score), [], detail
    return MISSING_EVIDENCE, [], detail


def _stability_tolerance(record: dict, cfg) -> float:
    """Area stability budget: the colour path has a softer edge than white paper."""
    segmentation = record.get("segmentation")
    mode = str((segmentation or {}).get("mode") or "luma") if isinstance(segmentation, dict) else "luma"
    if mode == "color":
        return float(_cfg(cfg, "color_stability_tolerance", 0.035) or 0.035)
    return float(_cfg(cfg, "stability_tolerance", 0.02) or 0.02)


def _metric_stability(record: dict, cfg) -> tuple[float, list[str], dict]:
    tolerance = _stability_tolerance(record, cfg)
    rel_range = None
    stability = record.get("stability")
    if isinstance(stability, dict):
        rel_range = _num(stability.get("rel_range"))
        if rel_range is None:
            totals = _num_list(stability.get("total_cm2"))
            if len(totals) >= 2:
                rel_range = _relative_range(totals)
    if rel_range is None:
        rel_range = _relative_range_from_candidates(record, cfg)
    detail = {"rel_range": rel_range, "tolerance": tolerance}
    if rel_range is None:
        return MISSING_EVIDENCE, [], detail
    scale = max(tolerance * STABILITY_DECAY, 1e-9)
    score = 1.0 / (1.0 + (float(rel_range) / scale) ** 2)
    flags = (["area_unstable"]
             if float(rel_range) > tolerance * STABILITY_REVIEW_FACTOR
             else [])
    return _clip(score), flags, detail


def _metric_border(record: dict, cfg) -> tuple[float, list[str], dict]:
    measure = _measure_dict(record)
    flags = _all_flags(record)
    touches = measure.get("leaf_touches_border")
    if touches is None or isinstance(touches, str):
        touches = _flag_match(flags, "leaf_touches_border", "touches_border",
                              "border_touch")
    if touches is None:
        return UNKNOWN_BORDER, [], {"touches": None}
    if touches:
        return 0.0, ["leaf_touches_border"], {"touches": True}
    return 1.0, [], {"touches": False}


def _metric_competing(record: dict, cfg) -> tuple[float, list[str], dict]:
    measure = _measure_dict(record)
    ratio = _num(measure.get("competing_ratio"))
    if ratio is None:
        ratio = _num(measure.get("competing_leaf_interior_ratio"))
    if ratio is None:
        ratio = _num(measure.get("competing_leaf_ratio"))
    if ratio is None and _flag_match(_all_flags(record), "competing_leaf"):
        ratio = COMPETING_LEAF_RATIO
    detail = {"ratio": ratio}
    if ratio is None:
        return UNKNOWN_COMPETING, [], detail
    score = _clip(1.0 - max(0.0, ratio - 0.02) / COMPETING_LEAF_SOFT)
    flags = ["competing_leaf"] if ratio >= COMPETING_LEAF_RATIO else []
    return score, flags, detail


def _metric_hole_noise(record: dict, cfg) -> tuple[float, list[str], dict]:
    """Fraction of consumed area that is threshold speckle ("tiny" holes).

    A tiny hole costs almost nothing in total area, but when *all* consumed
    area is speckle the reported consumed/percent number is not trustworthy, so
    this metric gates auto-accept.  When the consumed area is dwarfed by real
    holes the score stays high.
    """
    measure = _measure_dict(record)
    if not isinstance(measure, dict) or not measure:
        return MISSING_EVIDENCE, [], {"holes": None}
    tiny_px = float(_cfg(cfg, "tiny_hole_px", 25) or 25)
    holes = measure.get("holes")
    if isinstance(holes, list) and holes:
        sizes = []
        for hole in holes:
            if not isinstance(hole, dict):
                continue
            area = _num(hole.get("area_px"))
            if area is not None:
                sizes.append(max(0.0, area))
        if sizes:
            tiny = [size for size in sizes if size < tiny_px]
            hole_area = sum(sizes)
            fraction_area = sum(tiny) / hole_area if hole_area > 0 else 0.0
            fraction_count = len(tiny) / len(sizes)
            noise = max(fraction_area, 0.5 * fraction_count)
            detail = {
                "n_holes": len(sizes), "tiny_holes": len(tiny),
                "tiny_area_fraction": fraction_area,
                "tiny_count_fraction": fraction_count,
                "noise": noise, "tiny_hole_px": tiny_px,
            }
            return _clip(1.0 - noise), [], detail
    hole_count = _num(measure.get("hole_count"))
    if hole_count is None:
        hole_count = _num(measure.get("n_holes"))
    tiny_holes = _num(measure.get("tiny_holes"))
    consumed = _num(measure.get("consumed_px"))
    if hole_count is not None and tiny_holes is not None and hole_count > 0:
        fraction_count = tiny_holes / hole_count
        noise = 0.5 * fraction_count
        return _clip(1.0 - noise), [], {"n_holes": hole_count,
                                        "tiny_holes": tiny_holes,
                                        "tiny_count_fraction": fraction_count,
                                        "noise": noise}
    if consumed is not None:
        return (1.0 if consumed <= 0 else 0.60), [], {"holes": None,
                                                      "consumed_px": consumed}
    return MISSING_EVIDENCE, [], {"holes": None}


def _metric_percent(record: dict, cfg) -> tuple[float, list[str], dict]:
    percent = _num(record.get("percent_consumed"))
    if percent is None:
        measure = _measure_dict(record)
        consumed = _num(measure.get("consumed_px"))
        total = _num(measure.get("total_px"))
        if consumed is not None and total is not None and total > 0:
            percent = 100.0 * consumed / total
    if percent is None:
        return MISSING_EVIDENCE, [], {"percent": None}
    detail = {"percent": percent}
    if percent <= PERCENT_MAX_PLAUSIBLE:
        return _clip(1.0 if percent >= 0.0 else 0.0), [], detail
    score = _clip(1.0 - (percent - PERCENT_MAX_PLAUSIBLE)
                  / (100.0 - PERCENT_MAX_PLAUSIBLE))
    return score, [], detail


def _metric_leaf_fraction(record: dict, cfg) -> tuple[float, list[str], dict]:
    measure = _measure_dict(record)
    total_px = _num(measure.get("total_px"))
    scale = record.get("scale")
    length = _num(scale.get("scale_px_length")) if isinstance(scale, dict) else None
    detail = {"fraction": None, "total_px": total_px,
              "scale_px_length": length}
    if total_px is None or length is None or length <= 0.0:
        return MISSING_EVIDENCE, [], detail
    fraction = total_px / (length * length)
    detail["fraction"] = fraction
    if fraction < LEAF_FRACTION_MIN:
        return _clip(fraction / LEAF_FRACTION_MIN), [], detail
    if fraction > LEAF_FRACTION_MAX:
        return _clip(1.0 - (fraction - LEAF_FRACTION_MAX)
                     / LEAF_FRACTION_FADE), [], detail
    return 1.0, [], detail


_METRICS = [
    ("scale_quality", _metric_scale),
    ("threshold_quality", _metric_threshold),
    ("stability", _metric_stability),
    ("leaf_border_touch", _metric_border),
    ("competing_leaf", _metric_competing),
    ("hole_noise", _metric_hole_noise),
    ("percent_extremes", _metric_percent),
    ("leaf_fraction", _metric_leaf_fraction),
]


def _metric_table():
    """The ordered ``(name, function)`` metric table (indirection for tests)."""
    return _METRICS


# ---------------------------------------------------------------------------
# gates / status
# ---------------------------------------------------------------------------
def _gate(record: dict, cfg, metrics: dict,
          details: dict) -> tuple[list[str], list[str]]:
    """Return ``(fatal_problems, review_blockers)`` for the record."""
    fatal: list[str] = []
    reviewers: list[str] = []

    if record.get("ok") is False:
        fatal.append("analysis_failed")
    if record.get("error") not in (None, "", False):
        fatal.append("analysis_error")

    measure = _measure_dict(record)
    if not measure:
        fatal.append("no_measurement")
    else:
        total = _num(measure.get("total_px"))
        leaf = _num(measure.get("leaf_px"))
        if total is None and leaf is None:
            fatal.append("no_measurement")
        elif (total is not None and total <= 0) or (leaf is not None and leaf <= 0):
            fatal.append("no_leaf")

    if fatal:
        return fatal, reviewers

    # No scale = no absolute area.  Upstream cannot express it either, so the
    # record is still reviewable (a human can measure/enter the scale) but it
    # must NEVER be auto-accepted: for the user this is exactly the
    # "low precision -> human review" case.
    relative_only = bool(record.get("relative_only"))
    scale = record.get("scale")
    scale_found = bool(isinstance(scale, dict) and scale.get("found"))
    if relative_only:
        reviewers.append("relative_only")
    if not scale_found:
        reviewers.append("scale_not_found")
    else:
        length = _num(scale.get("scale_px_length"))
        if length is None or length <= 0.0:
            reviewers.append("scale_invalid_length")

    if _bool(details.get("leaf_border_touch", {}).get("touches")):
        reviewers.append("leaf_touches_border")
    tolerance = _stability_tolerance(record, cfg)
    rel_range = _num(details.get("stability", {}).get("rel_range"))
    if rel_range is not None and rel_range > tolerance * STABILITY_REVIEW_FACTOR:
        reviewers.append("area_unstable")
    if metrics.get("threshold_quality", 1.0) < THRESHOLD_MIN_FOR_AUTO:
        reviewers.append("weak_threshold_separation")
    percent = _num(details.get("percent_extremes", {}).get("percent"))
    if percent is not None and percent > PERCENT_REVIEW_MAX:
        reviewers.append("percent_implausible")
    ratio = _num(details.get("competing_leaf", {}).get("ratio"))
    if ratio is not None and ratio >= COMPETING_LEAF_RATIO:
        reviewers.append("competing_leaf")
    if metrics.get("hole_noise", 1.0) < HOLE_NOISE_MIN_FOR_AUTO:
        reviewers.append("tiny_hole_noise")
    if metrics.get("leaf_fraction", 1.0) < LEAF_FRACTION_MIN_SCORE:
        reviewers.append("leaf_fraction_implausible")
    return fatal, _unique(reviewers)


def _status_thresholds(cfg) -> tuple[float, float]:
    auto_min = float(_cfg(cfg, "auto_accept_min", 0.75))
    review_min = float(_cfg(cfg, "review_min", 0.45))
    if not np.isfinite(auto_min):
        auto_min = 0.75
    if not np.isfinite(review_min):
        review_min = 0.45
    auto_min = _clip(auto_min)
    review_min = min(_clip(review_min), auto_min)
    return auto_min, review_min


def _reason(status: str, confidence: float, fatal: list[str],
            blockers: list[str]) -> str:
    problems = [token for token in _unique(fatal + blockers) if token in _HUMAN]
    if status == "auto":
        return (f"auto-accepted (confidence {confidence:.2f}); "
                "no blocking quality issues")
    label = "needs human review" if status == "review" else "failed"
    if problems:
        human = "; ".join(_HUMAN[token] for token in problems)
        return f"{label} (confidence {confidence:.2f}): {human}"
    if status == "review":
        return (f"{label} (confidence {confidence:.2f}): confidence below "
                "auto-accept threshold")
    return (f"{label} (confidence {confidence:.2f}): confidence below "
            "review threshold")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _cfg(cfg, name: str, default):
    if cfg is None:
        return default
    value = getattr(cfg, name, default)
    return default if value is None else value


def _clip(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING_EVIDENCE
    if not np.isfinite(number):
        return MISSING_EVIDENCE
    return float(min(1.0, max(0.0, number)))


def _num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    else:
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None
    return number if np.isfinite(number) else None


def _num_list(values) -> list[float]:
    out = []
    if isinstance(values, (list, tuple, np.ndarray)):
        for value in values:
            number = _num(value)
            if number is not None:
                out.append(number)
    return out


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _measure_dict(record: dict) -> dict:
    measure = record.get("measure") if isinstance(record, dict) else None
    return measure if isinstance(measure, dict) else {}


def _extend_flags(flags: list[str], values) -> None:
    if values is None:
        return
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return
    for value in values:
        token = str(value)
        if token and token not in flags:
            flags.append(token)


def _all_flags(record: dict) -> list[str]:
    flags: list[str] = []
    _extend_flags(flags, record.get("flags"))
    _extend_flags(flags, _measure_dict(record).get("flags"))
    return flags


def _flag_match(flags: list[str], *tokens: str) -> bool | None:
    if not flags:
        return None
    lowered = [flag.lower() for flag in flags]
    return any(token in flag for flag in lowered for token in tokens)


def _unique(values) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _relative_range(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    base = float(np.mean(values))
    if base <= 0.0:
        return None
    return (max(values) - min(values)) / base


def _relative_range_from_candidates(record: dict, cfg) -> float | None:
    """Fallback stability estimate from the ``candidates`` list (§7)."""
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None
    base = _num(record.get("total_cm2"))
    if base is None or base <= 0.0:
        base = _num(_measure_dict(record).get("total_px"))
    if base is None or base <= 0.0:
        return None
    deltas = getattr(cfg, "stability_deltas", None) if cfg is not None else None
    try:
        max_delta = max(abs(float(delta)) for delta in deltas) if deltas else 0.02
    except (TypeError, ValueError):
        max_delta = 0.02
    if max_delta <= 0.0:
        max_delta = 0.02
    center = _num(record.get("threshold"))
    totals: list[float] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        total = _num(candidate.get("total_cm2"))
        if total is None:
            continue
        candidate_threshold = _num(candidate.get("threshold"))
        if center is not None and candidate_threshold is not None:
            if abs(candidate_threshold - center) > max_delta + 1e-9:
                continue
        totals.append(total)
    if len(totals) < 2:
        return None
    return _relative_range(totals)
