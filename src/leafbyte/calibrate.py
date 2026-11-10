"""Calibrate the confidence model against samples with known areas.

Owned by the Team Lead.  ``data/paper/ground_truth.csv`` gives the paper's
measured area for 29 of the 30 paper photos, including 9 that the upstream app's
automatic scale detection gets wrong.  That is a perfectly good labelled set for
answering the only question the user actually cares about:

    "if I trust the automatic result, how often is it wrong?"

So the calibration searches the routing thresholds (and optionally the metric
weights) for the setting that maximises the number of automatically accepted
samples subject to *zero* auto-accepted samples outside the area tolerance, and
reports the trade-off curve.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from . import batch, quality
from .config import Config


def _reference_catalogue(records: list[dict], ground_truth: dict) -> dict[str, dict]:
    catalogue: dict[str, dict] = {}
    for record in records:
        reference = (ground_truth.get(record["file"])
                     or ground_truth.get(Path(record["file"]).stem))
        if reference and reference.get("total_cm2") is not None:
            catalogue[record["file"]] = reference
    return catalogue


def _status_for(record: dict, cfg: Config, weights: dict | None) -> tuple[str, float]:
    probe = copy.deepcopy(record)
    evaluated = quality.evaluate(probe, cfg, weights=weights) if weights is not None \
        else quality.evaluate(probe, cfg)
    return str(evaluated.get("status", "failed")), float(evaluated.get("confidence") or 0.0)


def evaluate_scenario(
    records: list[dict],
    catalogue: dict[str, dict],
    cfg: Config,
    weights: dict | None = None,
    tolerance_cm2: float | None = None,
    rel_tolerance: float = 0.015,
) -> dict:
    """Route every record with the given settings and score the routing."""
    auto_count = 0
    auto_with_reference = 0
    auto_within = 0
    upstream_accurate = 0
    false_auto: list[dict] = []
    missed_reference: list[str] = []
    disabled_still_auto: list[str] = []
    errors: list[float] = []

    for record in records:
        status, confidence = _status_for(record, cfg, weights)
        reference = catalogue.get(record["file"])
        if status != "auto":
            if reference is not None:
                missed_reference.append(record["file"])
                if reference.get("reference_status") == "app_test_disabled":
                    pass
            continue
        auto_count += 1
        if reference is not None:
            if reference.get("reference_status") == "app_test_disabled":
                disabled_still_auto.append(record["file"])
            auto_with_reference += 1
            measured = record.get("total_cm2")
            if measured is None:
                false_auto.append({"file": record["file"], "error": None,
                                   "confidence": round(confidence, 3), "reason": "auto with no area"})
                continue
            error = float(measured) - float(reference["total_cm2"])
            tolerance = max(
                tolerance_cm2 if tolerance_cm2 is not None else cfg.reference_tolerance_cm2,
                rel_tolerance * abs(reference["total_cm2"]))
            errors.append(abs(error))
            if abs(error) <= tolerance:
                auto_within += 1
                if reference.get("reference_status") == "app_test_disabled":
                    upstream_accurate += 1
            else:
                false_auto.append({"file": record["file"], "error": round(error, 4),
                                   "tolerance": round(tolerance, 4),
                                   "confidence": round(confidence, 3)})

    return {
        "auto_accept_min": round(cfg.auto_accept_min, 3),
        "review_min": round(cfg.review_min, 3),
        "auto": auto_count,
        "auto_with_reference": auto_with_reference,
        "auto_within_tolerance": auto_within,
        "false_auto_accept": len(false_auto),
        "false_auto_detail": false_auto,
        # Upstream LeafByte cannot measure these at all (its scale detection
        # fails); if we auto-accept them we must be accurate, so track both.
        "upstream_failures_auto_accepted": disabled_still_auto,
        "upstream_failures_auto_accurate": upstream_accurate,
        "missed_reference": missed_reference,
        "mean_abs_error": round(float(np.mean(errors)), 4) if errors else None,
        "max_abs_error": round(float(np.max(errors)), 4) if errors else None,
        "precision": round(auto_within / auto_with_reference, 4) if auto_with_reference else None,
    }


def weight_variants(base: dict) -> list[tuple[str, dict]]:
    """A handful of perturbations of the quality weights (generically derived)."""
    variants: list[tuple[str, dict]] = [("default", dict(base))]
    keys = list(base)
    for key in keys:
        if not isinstance(base[key], (int, float)) or base[key] == 0:
            continue
        for factor, label in ((2.0, "x2"), (0.5, "x0.5")):
            variant = dict(base)
            variant[key] = base[key] * factor
            total = sum(float(v) for v in variant.values() if isinstance(v, (int, float)))
            if total > 0:
                variant = {k: (float(v) / total if isinstance(v, (int, float)) else v)
                           for k, v in variant.items()}
            variants.append((f"{key}_{label}", variant))
    return variants


def sweep(
    records: list[dict],
    catalogue: dict[str, dict],
    cfg: Config,
    weights: dict | None = None,
    auto_grid: np.ndarray | None = None,
    review_grid: np.ndarray | None = None,
    rel_tolerance: float = 0.015,
) -> list[dict]:
    auto_grid = auto_grid if auto_grid is not None else np.round(np.arange(0.40, 0.99, 0.02), 2)
    review_grid = review_grid if review_grid is not None else np.round(np.arange(0.10, 0.61, 0.05), 2)
    rows: list[dict] = []
    for auto_min in auto_grid:
        for review_min in review_grid:
            if review_min >= auto_min:
                continue
            probe = copy.deepcopy(cfg)
            probe.auto_accept_min = float(auto_min)
            probe.review_min = float(review_min)
            rows.append(evaluate_scenario(records, catalogue, probe, weights=weights,
                                          rel_tolerance=rel_tolerance))
    return rows


def _score(row: dict) -> tuple:
    """Rank: no false auto-accepts first, then precision, coverage, fewer misses."""
    return (
        1 if row["false_auto_accept"] == 0 else 0,
        row["precision"] if row["precision"] is not None else 0.0,
        row["auto"],
        -len(row["missed_reference"]),
    )


def main(args) -> int:
    out_dir = Path(args.out)
    records = batch.load_records(out_dir)
    ground_truth = batch.load_ground_truth(args.ground_truth)
    catalogue = _reference_catalogue(records, ground_truth)
    if not catalogue:
        print("no records matched the ground truth file")
        return 1

    cfg = Config.load(out_dir / "config.json") if (out_dir / "config.json").exists() else Config()
    if getattr(args, "tol_abs", None) is not None:
        cfg.reference_tolerance_cm2 = float(args.tol_abs)
    results: dict = {
        "reference_samples": len(catalogue),
        "variants": {},
        "chosen": None,
    }

    best_overall: tuple[tuple, str, dict] | None = None
    base_weights = getattr(quality, "WEIGHTS", None)
    variants = weight_variants(base_weights) if isinstance(base_weights, dict) and base_weights else [("default", None)]

    for label, weights in variants:
        rows = sweep(records, catalogue, cfg, weights=weights,
                     rel_tolerance=getattr(args, "tol_rel", 0.015))
        if not rows:
            continue
        best = max(rows, key=_score)
        results["variants"][label] = {
            "best": best,
            "curve": [
                {key: row[key] for key in ("auto_accept_min", "review_min", "auto",
                                           "auto_within_tolerance", "false_auto_accept")}
                for row in rows
            ],
            "weights": weights,
        }
        score = _score(best)
        if best_overall is None or score > best_overall[0]:
            best_overall = (score, label, best)

    if best_overall is None:
        print("calibration failed: no scenario evaluated")
        return 1

    _, best_label, best = best_overall
    results["chosen"] = {"variant": best_label, **best}
    cfg.auto_accept_min = float(best["auto_accept_min"])
    cfg.review_min = float(best["review_min"])
    results["chosen_config"] = {"auto_accept_min": cfg.auto_accept_min,
                                "review_min": cfg.review_min}

    (out_dir / "calibration.json").write_text(
        json.dumps(batch._jsonable(results), indent=2, ensure_ascii=False), encoding="utf-8")

    if getattr(args, "write_config", False):
        cfg.save(out_dir / "config.json")

    print(f"reference samples : {len(catalogue)}")
    print(f"best variant      : {best_label}")
    print(f"auto_accept_min   : {best['auto_accept_min']}   review_min: {best['review_min']}")
    print(f"auto-accepted     : {best['auto']}")
    print(f"  with reference  : {best['auto_with_reference']}")
    print(f"  within tolerance: {best['auto_within_tolerance']}")
    print(f"  false accepts   : {best['false_auto_accept']}  "
          f"(tolerance: max({cfg.reference_tolerance_cm2} cm2, "
          f"{getattr(args, 'tol_rel', 0.015):.1%} of reference))")
    if best["false_auto_detail"]:
        for item in best["false_auto_detail"]:
            print(f"    ! {item['file']}: error {item['error']} (tol {item.get('tolerance')}) "
                  f"confidence {item['confidence']}")
    failures = best["upstream_failures_auto_accepted"]
    print(f"upstream-incompatible samples auto-accepted: {len(failures)} "
          f"(accurate: {best['upstream_failures_auto_accurate']}) {failures}")
    print(f"missed (correct but queued): {len(best['missed_reference'])}")
    print(f"written: {out_dir / 'calibration.json'}")
    return 0
