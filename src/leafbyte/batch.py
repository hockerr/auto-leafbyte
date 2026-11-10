"""Batch orchestration: a folder of leaf photos -> analysis records -> routing.

Owned by the Team Lead.  This is the glue that turns the algorithm modules into
a usable automation pipeline::

    photos -> normalise -> Otsu threshold -> find the 4 scale marks -> rectify
           -> segment leaf + enclosed holes -> cm^2 -> stability -> QC score
           -> auto-accepted results OR a human review queue

Nothing here is image-algorithm specific; the maths lives in ``imageops``,
``ccl``, ``scale``, ``measure`` and ``quality``.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import imageops, measure, quality, review, scale
from .config import Config
from .types import BACKGROUND_LABEL, MeasureResult

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

RESULTS_JSONL = "results.jsonl"
RESULTS_CSV = "results.csv"
FINAL_CSV = "final_results.csv"
FINAL_JSONL = "final_results.jsonl"
RUN_JSON = "run.json"

CSV_COLUMNS = [
    "file", "status", "confidence", "total_cm2", "consumed_cm2", "percent_consumed",
    "threshold", "scale_method", "scale_px_length", "flags", "reason",
]

#: Final table = results + accuracy vs the published value (when known) + audit trail.
FINAL_COLUMNS = CSV_COLUMNS + [
    "reference_total_cm2", "err_total_cm2", "rel_err_pct", "within_tolerance",
    "source", "reviewed_by", "review_note",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def natural_key(path: Path | str) -> list:
    """Sort ``leaf2.png`` before ``leaf10.png``."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", Path(path).name)]


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    images = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]
    return sorted(images, key=natural_key)


def _stem(path: Path) -> str:
    return path.stem


def _r(x: float | None, digits: int = 6) -> float | None:
    if x is None:
        return None
    value = float(x)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _masks(analysis) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Leaf mask and consumed-hole mask out of a ``measure.MeasureAnalysis``."""
    components = getattr(analysis, "components", None)
    result: MeasureResult = analysis.result
    if components is None or result.leaf_label is None:
        return None, None
    label_map = components.label_map
    leaf_mask = label_map == result.leaf_label
    hole_mask = np.zeros_like(leaf_mask)
    for hole in result.holes:
        hole_mask |= label_map == hole["label"]
    return leaf_mask, hole_mask


def _write_jpeg(image, path: Path, quality_level: int = 85) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(imageops.save_png_bytes(image, quality_level))
    return path


# --------------------------------------------------------------------------
# one image
# --------------------------------------------------------------------------
def analyze_image(path: str | Path, cfg: Config) -> dict[str, Any]:
    """Run the full measurement for one photo.  Never raises."""
    path = Path(path)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "ok": False,
        "stage": "load",
        "error": None,
        "image_size": None,
        "normalized_size": None,
        "threshold": None,
        "otsu_threshold": None,
        "threshold_metrics": {},
        "histogram": [],
        "scale": {"found": False, "method": "none", "quality": 0.0, "points": [],
                  "scale_px_length": None, "flags": ["not attempted"], "details": {}},
        "measure": {},
        "stability": {},
        "total_cm2": None,
        "consumed_cm2": None,
        "percent_consumed": None,
        "relative_only": False,
        "candidates": [],
        "metrics": {},
        "confidence": 0.0,
        "status": "failed",
        "flags": [],
        "reason": "",
        "timing_ms": None,
    }

    try:
        # 1. normalise ----------------------------------------------------
        image = imageops.load_image(path)
        record["image_size"] = [int(image.shape[1]), int(image.shape[0])]
        normalized = imageops.resize_within(image, cfg.max_dim)
        record["normalized_size"] = [int(normalized.shape[1]), int(normalized.shape[0])]
        record["_normalized"] = normalized

        # 2. threshold ----------------------------------------------------
        otsu = imageops.suggested_threshold(normalized)
        threshold = cfg.threshold if cfg.threshold is not None else otsu
        record["otsu_threshold"] = _r(otsu)
        record["threshold"] = _r(threshold)
        record["threshold_metrics"] = {k: _r(v) for k, v in
                                       imageops.threshold_diagnostics(normalized, threshold).items()}
        histogram = imageops.luma_histogram(imageops.luma(normalized))
        cut_index = int(np.clip(round(threshold * (imageops.HISTOGRAM_BUCKETS - 1)), 0,
                                imageops.HISTOGRAM_BUCKETS - 1))
        record["histogram"] = [int(value) for value in histogram]
        record["otsu_separation"] = _r(imageops.histogram_separation(histogram, cut_index))

        # 3. scale + rectification ---------------------------------------
        record["stage"] = "scale"
        detection = scale.detect(normalized, cfg)
        record["scale"] = {
            "found": bool(detection.found),
            "method": detection.method,
            "quality": _r(detection.quality),
            "points": [[_r(float(x), 3), _r(float(y), 3)] for x, y in detection.points],
            "scale_px_length": None,
            "flags": list(detection.flags),
            "details": {k: (v if not isinstance(v, float) else _r(v))
                        for k, v in (detection.details or {}).items()},
        }
        rectified = scale.rectify(normalized, detection, cfg) if detection.found else None

        if rectified is not None:
            record["scale"]["scale_px_length"] = _r(rectified.scale_px_length)
            record["stage"] = "measure"
            analysis = measure.analyze(rectified.image, cfg, threshold=threshold)
            result = analysis.result
            total_cm2, consumed_cm2, percent = measure.to_cm2(result, cfg, rectified.scale_px_length)
            record["measure"] = _measure_payload(analysis)
            record["total_cm2"] = _r(total_cm2)
            record["consumed_cm2"] = _r(consumed_cm2)
            record["percent_consumed"] = _r(percent)

            # 4. stability ("precision") under threshold perturbation ------
            record["stage"] = "stability"
            totals: list[float] = []
            for delta in cfg.stability_deltas:
                perturbed = float(np.clip(threshold + delta, 0.01, 0.99))
                perturbed_analysis = measure.analyze(rectified.image, cfg, threshold=perturbed)
                perturbed_total, _, _ = measure.to_cm2(
                    perturbed_analysis.result, cfg, rectified.scale_px_length)
                totals.append(float(perturbed_total))
            mean_total = float(np.mean(totals)) if totals else 0.0
            rel_range = 0.0
            if mean_total > 0 and totals:
                rel_range = (max(totals) - min(totals)) / mean_total
            record["stability"] = {
                "deltas": [float(d) for d in cfg.stability_deltas],
                "total_cm2": [_r(t) for t in totals],
                "rel_range": _r(rel_range),
            }
            record["_rectified"] = rectified
            record["_analysis"] = analysis
        else:
            # No absolute scale: fall back to the upstream "percent only" mode so
            # the sample is still reviewable, then let the QC model route it.
            record["relative_only"] = True
            record["stage"] = "measure_relative"
            analysis = measure.analyze(normalized, cfg, threshold=threshold)
            result = analysis.result
            record["measure"] = _measure_payload(analysis)
            record["percent_consumed"] = _r(result.percent_consumed)
            record["_analysis"] = analysis

        record["stage"] = "done"
        record["ok"] = True

    except Exception as exc:  # noqa: BLE001 - a bad photo must not kill the batch
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["ok"] = False
        record["flags"] = list(record.get("flags") or [])
        record["flags"].append(f"exception: {type(exc).__name__}")
        record["_traceback"] = traceback.format_exc(limit=3)

    record["timing_ms"] = int((time.perf_counter() - started) * 1000)
    return record


def _measure_payload(analysis) -> dict[str, Any]:
    """The ``measure`` sub-record: whatever ``measure.measure_dict`` says it is.

    ``measure.py`` owns this dict (it knows which evidence the QC model needs);
    the local fallback only exists so batch.py keeps working if that helper is
    missing.
    """
    builder = getattr(measure, "measure_dict", None)
    if callable(builder):
        payload = builder(analysis)
        return json.loads(json.dumps(_jsonable(payload), ensure_ascii=False))
    result: MeasureResult = analysis.result
    return {
        "total_px": int(result.total_px),
        "consumed_px": int(result.consumed_px),
        "leaf_px": int(result.leaf_px),
        "leaf_label": result.leaf_label,
        "holes": [
            {"label": int(h["label"]), "area_px": int(h["area_px"]),
             "centroid": [float(h.get("centroid", [0, 0])[0]), float(h.get("centroid", [0, 0])[1])]}
            for h in result.holes
        ],
        "leaf_touches_border": bool(result.leaf_touches_border),
        "competing_ratio": _r(result.competing_ratio),
        "hole_count": int(result.hole_count),
        "tiny_holes": int(result.tiny_holes),
        "excluded_px": int(result.excluded_px),
        "flags": list(result.flags),
    }


# --------------------------------------------------------------------------
# review candidates / overlays
# --------------------------------------------------------------------------
def build_candidates(record: dict[str, Any], out_dir: str | Path, cfg: Config) -> list[dict[str, Any]]:
    """Threshold-offset alternatives + overlays for one flagged sample."""
    out_dir = Path(out_dir)
    rectified = record.get("_rectified")
    candidates: list[dict[str, Any]] = []
    base_threshold = record.get("threshold")
    if base_threshold is None:
        return candidates

    overlay_dir = out_dir / "overlays"
    candidate_dir = overlay_dir / "candidates"
    stem = Path(record["file"]).stem

    for index, delta in enumerate(cfg.candidate_deltas):
        threshold = float(np.clip(base_threshold + float(delta), 0.01, 0.99))
        candidate: dict[str, Any] = {
            "candidate_id": f"t{index}",
            "threshold": _r(threshold),
            "total_cm2": None,
            "consumed_cm2": None,
            "percent_consumed": None,
            "selected": False,
            "overlay": None,
        }
        try:
            if rectified is not None:
                analysis = measure.analyze(rectified.image, cfg, threshold=threshold)
                total_cm2, consumed_cm2, percent = measure.to_cm2(
                    analysis.result, cfg, rectified.scale_px_length)
                candidate.update(total_cm2=_r(total_cm2), consumed_cm2=_r(consumed_cm2),
                                 percent_consumed=_r(percent))
                leaf_mask, hole_mask = _masks(analysis)
                overlay = imageops.render_overlay(rectified.image, mask=leaf_mask,
                                                  holes_mask=hole_mask, max_size=cfg.overlay_max_px)
            else:
                normalized = record.get("_normalized")
                if normalized is None:
                    raise ValueError("record carries no image for candidate rendering")
                analysis = measure.analyze(normalized, cfg, threshold=threshold)
                candidate["percent_consumed"] = _r(analysis.result.percent_consumed)
                leaf_mask, hole_mask = _masks(analysis)
                overlay = imageops.render_overlay(normalized, mask=leaf_mask,
                                                  holes_mask=hole_mask, max_size=cfg.overlay_max_px)
            target = candidate_dir / f"{stem}__t{index}.jpg"
            _write_jpeg(overlay, target)
            candidate["overlay"] = _relative(target, out_dir)
        except Exception as exc:  # noqa: BLE001
            candidate["error"] = f"{type(exc).__name__}: {exc}"
        candidates.append(candidate)

    # mark the candidate the automation actually used
    for candidate in candidates:
        if candidate["threshold"] is not None and record.get("threshold") is not None:
            if abs(candidate["threshold"] - record["threshold"]) < 1e-9:
                candidate["selected"] = True
                break

    if len(candidates) > cfg.review_candidates:
        keep = [c for c in candidates if c["selected"]][:1]
        for candidate in candidates:
            if len(keep) >= cfg.review_candidates:
                break
            if candidate not in keep:
                keep.append(candidate)
        candidates = keep
    return candidates


def build_review_overlays(record: dict[str, Any], out_dir: str | Path, cfg: Config) -> None:
    """Original-with-quad and rectified-with-mask overlays for the review page."""
    out_dir = Path(out_dir)
    overlay_dir = out_dir / "overlays"
    stem = Path(record["file"]).stem
    scale_info = record.get("scale") or {}
    points = scale_info.get("points") or []

    source = record.get("_normalized")
    if source is None:
        record["overlay_original"] = None
    else:
        quad = points if len(points) == 4 else None
        overlay = imageops.render_overlay(source, quad=quad, scale_points=points,
                                          max_size=cfg.overlay_max_px)
        record["overlay_original"] = _relative(_write_jpeg(overlay, overlay_dir / f"{stem}_original.jpg"), out_dir)

    analysis = record.get("_analysis")
    if analysis is not None and record.get("_rectified") is not None:
        leaf_mask, hole_mask = _masks(analysis)
        overlay = imageops.render_overlay(record["_rectified"].image, mask=leaf_mask,
                                          holes_mask=hole_mask, max_size=cfg.overlay_max_px)
        record["overlay_rectified"] = _relative(_write_jpeg(overlay, overlay_dir / f"{stem}_rectified.jpg"), out_dir)
    else:
        record["overlay_rectified"] = None


# --------------------------------------------------------------------------
# ground truth comparison
# --------------------------------------------------------------------------
def load_ground_truth(path: str | Path) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            entry: dict[str, Any] = {"reference_status": (row.get("reference_status") or "").strip()}
            for key, column in (("total_cm2", "expected_total_cm2"),
                                ("consumed_cm2", "expected_consumed_cm2")):
                raw = (row.get(column) or "").strip()
                entry[key] = float(raw) if raw else None
            table[filename] = entry
            # records carry "CO1.png", the reference table usually "CO1"
            stem = Path(filename).stem
            if stem != filename:
                table.setdefault(stem, entry)
            else:
                table.setdefault(filename + ".png", entry)
    return table


def attach_reference(records: Iterable[dict[str, Any]], ground_truth: dict[str, dict[str, Any]],
                     cfg: Config) -> dict[str, Any]:
    """Attach per-record error vs the reference and return accuracy statistics."""
    stats: dict[str, Any] = {
        "compared": 0,
        "auto_compared": 0,
        "auto_within_tolerance": 0,
        "false_auto_accept": 0,
        "max_abs_error_total_cm2": 0.0,
        "mean_abs_error_total_cm2": 0.0,
        "per_file": {},
    }
    errors: list[float] = []
    auto_errors: list[float] = []
    for record in records:
        reference = ground_truth.get(record["file"]) or ground_truth.get(Path(record["file"]).stem)
        if not reference or reference.get("total_cm2") is None:
            record["reference"] = None
            continue
        stats["compared"] += 1
        entry: dict[str, Any] = {
            "total_cm2": reference["total_cm2"],
            "consumed_cm2": reference.get("consumed_cm2"),
            "reference_status": reference.get("reference_status"),
            "error_total_cm2": None,
            "error_consumed_cm2": None,
            "rel_error_total": None,
            "within_tolerance": None,
        }
        if record.get("total_cm2") is not None:
            error = float(record["total_cm2"]) - reference["total_cm2"]
            entry["error_total_cm2"] = _r(error)
            entry["rel_error_total"] = _r(error / reference["total_cm2"] if reference["total_cm2"] else None)
            tolerance = max(cfg.reference_tolerance_cm2, 0.015 * abs(reference["total_cm2"]))
            entry["within_tolerance"] = abs(error) <= tolerance
            errors.append(abs(error))
            if record.get("status") == "auto":
                stats["auto_compared"] += 1
                auto_errors.append(abs(error))
                if entry["within_tolerance"]:
                    stats["auto_within_tolerance"] += 1
                else:
                    stats["false_auto_accept"] += 1
        if record.get("consumed_cm2") is not None and reference.get("consumed_cm2") is not None:
            entry["error_consumed_cm2"] = _r(float(record["consumed_cm2"]) - reference["consumed_cm2"])
        record["reference"] = entry
        stats["per_file"][record["file"]] = entry

    if errors:
        stats["mean_abs_error_total_cm2"] = _r(float(np.mean(errors)))
        stats["max_abs_error_total_cm2"] = _r(float(np.max(errors)))
    if auto_errors:
        stats["auto_mean_abs_error_total_cm2"] = _r(float(np.mean(auto_errors)))
        stats["auto_max_abs_error_total_cm2"] = _r(float(np.max(auto_errors)))
    return stats


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_results(records: list[dict[str, Any]], out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / RESULTS_JSONL
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    csv_path = out_dir / RESULTS_CSV
    write_csv(records, csv_path)
    return {"jsonl": jsonl_path, "csv": csv_path}


def write_csv(records: list[dict[str, Any]], path: str | Path,
              columns: list[str] | None = None) -> Path:
    columns = columns or CSV_COLUMNS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {}
            for column in columns:
                if column == "flags":
                    row[column] = "; ".join(record.get("flags") or [])
                elif column == "scale_method":
                    row[column] = (record.get("scale") or {}).get("method")
                elif column == "scale_px_length":
                    row[column] = (record.get("scale") or {}).get("scale_px_length")
                elif column == "reference_total_cm2":
                    row[column] = (record.get("reference") or {}).get("total_cm2")
                elif column == "err_total_cm2":
                    row[column] = (record.get("reference") or {}).get("error_total_cm2")
                elif column == "rel_err_pct":
                    rel = (record.get("reference") or {}).get("rel_error_total")
                    row[column] = round(100 * rel, 3) if rel is not None else None
                elif column == "within_tolerance":
                    row[column] = (record.get("reference") or {}).get("within_tolerance")
                elif column in {"confidence", "percent_consumed"}:
                    value = record.get(column)
                    row[column] = round(float(value), 4) if value is not None else None
                else:
                    value = record.get(column)
                    if isinstance(value, float):
                        value = round(value, 3)
                    row[column] = value
            writer.writerow(row)
    return path


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"auto": 0, "review": 0, "failed": 0, "done": 0, "rejected": 0}
    confidences: list[float] = []
    for record in records:
        status = record.get("status", "failed")
        counts[status] = counts.get(status, 0) + 1
        if record.get("confidence") is not None:
            confidences.append(float(record["confidence"]))
    percentages = [float(r["percent_consumed"]) for r in records if r.get("percent_consumed") is not None]
    return {
        "images": len(records),
        "auto": counts.get("auto", 0),
        "review": counts.get("review", 0),
        "failed": counts.get("failed", 0),
        "done": counts.get("done", 0),
        "rejected": counts.get("rejected", 0),
        "auto_rate": round(counts.get("auto", 0) / len(records), 4) if records else 0.0,
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
        "mean_percent_consumed": round(float(np.mean(percentages)), 4) if percentages else None,
        "total_time_s": round(sum(float(r.get("timing_ms") or 0) for r in records) / 1000.0, 2),
    }


# --------------------------------------------------------------------------
# the batch entry point
# --------------------------------------------------------------------------
def run_batch(
    image_dir: str | Path,
    out_dir: str | Path,
    cfg: Config | None = None,
    ground_truth_path: str | Path | None = None,
    make_review: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Measure every image in ``image_dir`` and route low-precision ones to review."""
    cfg = cfg or Config()
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"no images found in {image_dir}")

    log = progress or (lambda message: None)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []

    for index, path in enumerate(images, start=1):
        record = analyze_image(path, cfg)
        record = quality.evaluate(record, cfg)
        if record.get("status") != "auto":
            try:
                build_review_overlays(record, out_dir, cfg)
                record["candidates"] = build_candidates(record, out_dir, cfg)
            except Exception as exc:  # noqa: BLE001
                record.setdefault("flags", []).append(f"overlay failed: {type(exc).__name__}")
        else:
            record["candidates"] = []
            record["overlay_original"] = None
            record["overlay_rectified"] = None
        records.append(record)
        log(f"[{index}/{len(images)}] {path.name}: {record['status']} "
            f"(confidence {float(record.get('confidence') or 0):.3f})")

    accuracy = None
    if ground_truth_path is None:
        candidate_path = image_dir.parent / "ground_truth.csv"
        if candidate_path.exists():
            ground_truth_path = candidate_path
    if ground_truth_path is not None and Path(ground_truth_path).exists():
        accuracy = attach_reference(records, load_ground_truth(ground_truth_path), cfg)

    written = write_results(records, out_dir)
    config_path = out_dir / "config.json"
    cfg.save(config_path)

    summary = summarize(records)
    run_info = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image_dir": str(image_dir),
        "out_dir": str(out_dir),
        "config": cfg.to_dict(),
        "ground_truth": str(ground_truth_path) if ground_truth_path is not None else None,
        "summary": summary,
        "accuracy": accuracy,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "files": {key: str(value) for key, value in written.items()},
    }
    (out_dir / RUN_JSON).write_text(json.dumps(_jsonable(run_info), indent=2, ensure_ascii=False),
                                    encoding="utf-8")

    review_path = None
    if make_review:
        queued = [r for r in records if r.get("status") in {"review", "failed"}]
        review_path = review.build(out_dir, queued, cfg)
        run_info["review_page"] = str(review_path)

    log(f"done: {summary['auto']} auto / {summary['review']} review / {summary['failed']} failed")
    return {"records": records, "summary": summary, "accuracy": accuracy,
            "out_dir": out_dir, "review_page": review_path}


def load_records(out_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(out_dir) / RESULTS_JSONL
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
