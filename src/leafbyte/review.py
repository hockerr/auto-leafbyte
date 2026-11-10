"""Human review queue: offline HTML review page + decisions merge.

Contract: ``docs/ALGORITHM.md`` §6.6 and §7.

Public API
----------
``build(out_dir, records, cfg=None) -> Path``
    Filter ``records`` down to the samples quality control sent to a human
    (``status in {"review", "failed"}``, plus ``ok == False`` records), write
    ``<out_dir>/review/queue.json`` and a **fully offline** ``index.html`` and
    return the ``Path`` to ``index.html``.  The page needs no server, no
    network and no CDN: every image is a relative file next to the page and
    the queue data is embedded as a JavaScript object (no ``fetch``/``JSON``
    loading).  Originals are copied downscaled into ``review/originals/`` so
    that the ``out/`` tree is a portable bundle.

``apply(records, decisions, cfg=None, reviewer=None, summary_out=None) -> list[dict]``
    Merge the ``decisions`` mapping (``{filename: decision}``, exactly the
    ``decisions`` object inside ``decisions.json``) into a *copy* of
    ``records`` and return the new list.  ``reviewer`` (the CLI
    ``--reviewer`` value) wins over an embedded ``payload["reviewer"]``,
    then ``"unknown"``.  Never mutates the input, never crashes on unknown
    files/actions (it records warnings instead), and is idempotent: applying
    the same decisions twice yields the same records.  Records without a
    decision are returned untouched (status ``review``/``failed``).

``export_payload(reviewer, decisions, reviewed_at=None) -> dict``
    Build the exact ``decisions.json`` document (the browser page implements
    the same conversion in JavaScript; keeping the Python twin makes the
    schema unit-testable).

Record fields used by :func:`build` (all optional except ``file``/``path``;
see §7)::

    file                    # "CO1.png"     (decision keys are file names)
    path                    # original photo path (relative to out_dir or cwd)
    status                  # "auto" | "review" | "failed" (or ok=False)
    threshold, total_cm2, consumed_cm2, percent_consumed, confidence
    flags, reason, error, scale
    overlay_original        # relative to out_dir: photo + yellow scale quad
    overlay_rectified       # relative to out_dir: rectified crop, leaf red /
                            #   consumed holes blue (the main overlay)
    candidates: [{"candidate_id": "t2", "threshold", "total_cm2",
                  "consumed_cm2", "percent_consumed", "selected",
                  "overlay", "error"}]         # overlay relative to out_dir

Overlay colour convention (see :func:`leafbyte.imageops.render_overlay`):
leaf = red, consumed holes = blue, scale marks / rectification quad = yellow.

Output ``queue.json``::

    {"schema": "leafbyte.review.queue/1", "generated_at": ..., "queue_id": ...,
     "out_dir": ..., "scale_cm": 17.0, "count": N, "total_records": M,
     "items": [{"index", "file", "status", "image", "overlay", "threshold",
                "total_cm2", "consumed_cm2", "percent_consumed", "confidence",
                "flags", "reason", "error", "candidates": [...]}, ...]}

Decisions schema (§6.6), one entry per reviewed file::

    {"reviewer": "zs", "reviewed_at": "2026-11-10T12:00:00",
     "decisions": {
       "CO1.png": {"action": "accept", "threshold": 0.53,
                    "total_cm2": 23.95, "consumed_cm2": 1.0, "note": ""},
       "AI2.png": {"action": "override", "total_cm2": 22.565,
                   "consumed_cm2": 2.279, "note": "ImageJ 手测"},
       "CA1.png": {"action": "reject", "note": "反光，需重拍"}}}

``apply`` returns records that additionally carry the audit fields
``source`` (``"auto"`` or ``"review:<reviewer>"``), ``original_confidence``,
``original_threshold``, ``reviewed_by``, ``review_note``, ``review_action``,
``reviewed_at`` and ``status`` (``"done"`` after accept/override,
``"rejected"`` after reject; untouched records keep their previous status).
A rejected record keeps no final measurement (``total_cm2`` etc. are
``None``).  Problems are appended to ``record["review_warnings"]``.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

from leafbyte.config import Config

try:  # Pillow is a hard project dependency, but never fail the module on it.
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - only when Pillow is missing
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

#: Records with these statuses go to the review queue.
REVIEW_STATUSES = ("review", "failed")

#: Version tag written into queue.json.
QUEUE_SCHEMA = "leafbyte.review.queue/1"

#: Image suffixes we are willing to copy without Pillow (fallback path).
_COPYABLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

_THRESHOLD_MATCH_TOL = 1e-6


# ---------------------------------------------------------------------------
# small value helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """Local wall-clock time as ``YYYY-MM-DDTHH:MM:SS`` (matches §6.6)."""
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def _text(value) -> str:
    return "" if value is None else str(value)


def _num(value) -> float | None:
    """Best-effort finite float conversion; ``None`` when not a real number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _round6(value) -> float | None:
    """Round to 6 significant digits (ALGORITHM.md §7: JSON keeps 6 sig digits)."""
    number = _num(value)
    if number is None:
        return None
    if number == 0:
        return 0.0
    return float(f"{number:.6g}")


def _percent(consumed, total) -> float | None:
    """``consumed / total * 100``; ``None`` when total is not positive."""
    c = _num(consumed)
    t = _num(total)
    if c is None or t is None or t <= 0:
        return None
    return 100.0 * c / t


def _jsonable(value):
    """Recursively make a value JSON-serialisable (numpy scalars, Path, NaN...)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value, key=lambda item: str(item))]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalar
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


# ---------------------------------------------------------------------------
# path / asset helpers
# ---------------------------------------------------------------------------
def _search_bases(out_dir: Path) -> list[Path]:
    """Directories used to resolve relative paths found in records."""
    bases: list[Path] = []
    for base in (Path.cwd(), out_dir, out_dir / "review", out_dir.parent):
        if base not in bases:
            bases.append(base)
    return bases


def _find_local(raw, out_dir: Path) -> Path | None:
    """Resolve a possibly-relative path from a record to an existing file."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("path") or raw.get("file") or raw.get("overlay")
    text = str(raw).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for base in _search_bases(out_dir):
        try:
            resolved = (base / candidate).resolve()
        except (OSError, RuntimeError):  # pragma: no cover - odd Windows paths
            resolved = base / candidate
        if resolved.is_file():
            return resolved
    return None


def _rel_posix(target: Path, base: Path) -> str | None:
    """Relative POSIX path from ``base`` to ``target`` (for file:// URLs)."""
    try:
        rel = os.path.relpath(str(target), str(base))
    except ValueError:  # different drive on Windows
        return None
    return rel.replace(os.sep, "/")


def _safe_stem(stem: str) -> str:
    """ASCII-only asset name stem (keeps file:// URLs free of escaping issues)."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned[:40] or "img"


def _asset_max_px(cfg) -> int:
    value = int(getattr(cfg, "overlay_max_px", 900) or 900)
    return max(160, value)


def _prepare_original(record: dict, out_dir: Path, review_dir: Path, index: int, cfg) -> str | None:
    """Copy a downscaled original into ``review/originals/``; return relative path.

    A missing/unreadable original is not an error: the caller writes a
    placeholder into the page instead.
    """
    raw = (
        record.get("path")
        or record.get("image_path")
        or record.get("original")
        or record.get("image_path")
    )
    source = _find_local(raw, out_dir)
    if source is None:
        return None

    originals = review_dir / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(source.stem)
    max_px = _asset_max_px(cfg)

    if Image is not None:
        try:
            with Image.open(source) as image:
                if ImageOps is not None:
                    image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")
                image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
                name = f"{index:03d}_{stem}.jpg"
                image.save(originals / name, format="JPEG", quality=82, optimize=True)
                return f"originals/{name}"
        except Exception:
            pass  # fall back to a plain copy below

    suffix = source.suffix.lower()
    if suffix in _COPYABLE_SUFFIXES:
        try:
            name = f"{index:03d}_{stem}{suffix}"
            shutil.copyfile(source, originals / name)
            return f"originals/{name}"
        except Exception:
            return None
    return None


def _resolve_asset(raw, out_dir: Path, review_dir: Path) -> str | None:
    """Resolve an overlay path and express it relative to ``index.html``."""
    target = _find_local(raw, out_dir)
    if target is None:
        return None
    return _rel_posix(target, review_dir)


def _build_candidates(record: dict, out_dir: Path, review_dir: Path) -> list[dict]:
    """Normalise ``record["candidates"]`` for embedding in the page."""
    candidates: list[dict] = []
    raw_list = record.get("candidates")
    if not isinstance(raw_list, list):
        return candidates
    for position, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            continue
        total = _num(raw.get("total_cm2"))
        consumed = _num(raw.get("consumed_cm2"))
        percent = _num(raw.get("percent_consumed"))
        if percent is None:
            percent = _percent(consumed, total)
        candidate_id = raw.get("candidate_id")
        candidates.append(
            {
                "index": position + 1,
                "candidate_id": None if candidate_id is None else str(candidate_id),
                "threshold": _round6(raw.get("threshold")),
                "total_cm2": _round6(total),
                "consumed_cm2": _round6(consumed),
                "percent_consumed": _round6(percent),
                "selected": bool(raw.get("selected", False)),
                "overlay": _resolve_asset(
                    raw.get("overlay") or raw.get("overlay_path") or raw.get("png"),
                    out_dir,
                    review_dir,
                ),
                "error": _text(raw.get("error")) or None,
            }
        )
    return candidates


def _pick_main_overlay(record: dict, candidates: list[dict], out_dir: Path, review_dir: Path) -> str | None:
    """Choose the overlay for the *current* measurement.

    Priority: ``overlay_rectified`` (the leaf/holes measurement view), then
    ``overlay_original`` (photo with the scale quad), then a generic overlay
    field, then the candidate marked ``selected``, then the candidate whose
    threshold is closest to the record threshold.
    """
    for key in ("overlay_rectified", "overlay_original", "overlay", "review_overlay", "overlay_path"):
        overlay = _resolve_asset(record.get(key), out_dir, review_dir)
        if overlay:
            return overlay

    for candidate in candidates:
        if candidate.get("selected") and candidate.get("overlay"):
            return candidate.get("overlay")

    threshold = _num(record.get("threshold"))
    best: dict | None = None
    best_delta: float | None = None
    for candidate in candidates:
        if not candidate.get("overlay"):
            continue
        candidate_threshold = candidate.get("threshold")
        if threshold is not None and candidate_threshold is not None:
            delta = abs(float(candidate_threshold) - threshold)
        else:
            delta = 0.0
        if best is None or best_delta is None or delta < best_delta:
            best = candidate
            best_delta = delta
    return best.get("overlay") if best else None


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _status_of(record: dict) -> str:
    """Routing status of a record (``auto`` / ``review`` / ``failed``)."""
    status = record.get("status")
    ok = record.get("ok", True)
    stage = record.get("stage")
    if status in REVIEW_STATUSES:
        return str(status)
    if ok is False or stage in ("failed", "error"):
        return "failed"
    if isinstance(status, str) and status:
        return status
    return "auto" if ok is not False else "failed"


def _file_of(record: dict, position: int) -> str:
    """Stable decision key of a record: its file name."""
    name = record.get("file")
    if name:
        return str(name)
    path = record.get("path") or record.get("image_path") or record.get("original")
    if path:
        return Path(str(path)).name
    return f"sample_{position + 1}"


def _queue_item(record: dict, position: int, out_dir: Path, review_dir: Path, cfg) -> dict:
    candidates = _build_candidates(record, out_dir, review_dir)
    total = _num(record.get("total_cm2"))
    consumed = _num(record.get("consumed_cm2"))
    percent = _num(record.get("percent_consumed"))
    if percent is None:
        percent = _percent(consumed, total)
    overlay_original = _resolve_asset(record.get("overlay_original"), out_dir, review_dir)
    overlay_rectified = _resolve_asset(record.get("overlay_rectified"), out_dir, review_dir)
    main_overlay = _pick_main_overlay(record, candidates, out_dir, review_dir)
    image = _prepare_original(record, out_dir, review_dir, position, cfg)
    if image is None and overlay_original:
        # Graceful degradation: show the annotated photo when the plain
        # original could not be located.
        image = overlay_original
    default_selected = None
    for candidate in candidates:
        if candidate.get("selected"):
            default_selected = candidate["index"]
            break
    scale = record.get("scale") if isinstance(record.get("scale"), dict) else {}
    item = {
        "index": position + 1,
        "file": _file_of(record, position),
        "status": _status_of(record),
        "image": image,
        "overlay": main_overlay,
        "overlay_original": overlay_original,
        "overlay_rectified": overlay_rectified,
        "threshold": _round6(record.get("threshold")),
        "total_cm2": _round6(total),
        "consumed_cm2": _round6(consumed),
        "percent_consumed": _round6(percent),
        "confidence": _round6(record.get("confidence")),
        "scale_found": bool(scale.get("found")) if scale else None,
        "scale_method": _text(scale.get("method")) or None,
        "scale_quality": _round6(scale.get("quality")),
        "flags": [_text(flag) for flag in (record.get("flags") or [])],
        "reason": _text(record.get("reason")),
        "error": _text(record.get("error")) or None,
        "candidates": candidates,
        "default_selected": default_selected,
    }
    return _jsonable(item)


def build(out_dir, records, cfg=None) -> Path:
    """Write ``<out_dir>/review/queue.json`` + ``index.html``; return the page.

    Parameters
    ----------
    out_dir:
        Batch output directory (``out/``).  The page is written to
        ``out_dir/review/``; the original photos are copied downscaled into
        ``out_dir/review/originals/`` so the folder is self-contained.
    records:
        Result records as produced by the batch step (§7 schema).  Only
        ``status in {"review", "failed"}`` records become review cards.
    cfg:
        :class:`leafbyte.config.Config` (optional, defaults are fine).
    """
    cfg = cfg or Config()
    out_dir = Path(out_dir)
    review_dir = out_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    all_records = [record for record in (records or []) if isinstance(record, dict)]
    flagged = [record for record in all_records if _status_of(record) in REVIEW_STATUSES]

    items = [
        _queue_item(record, position, out_dir, review_dir, cfg)
        for position, record in enumerate(flagged)
    ]
    files = [str(item.get("file")) for item in items]
    queue_id = hashlib.sha1("\n".join(files).encode("utf-8")).hexdigest()[:12] if files else "empty"

    queue = {
        "schema": QUEUE_SCHEMA,
        "generated_at": _now_iso(),
        "queue_id": queue_id,
        "out_dir": str(out_dir),
        "scale_cm": _round6(getattr(cfg, "scale_cm", None)) or 0.0,
        "count": len(items),
        "total_records": len(all_records),
        "items": items,
    }
    queue = _jsonable(queue)

    (review_dir / "queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    index = review_dir / "index.html"
    index.write_text(_render_html(queue), encoding="utf-8")
    return index


# ---------------------------------------------------------------------------
# decisions merge
# ---------------------------------------------------------------------------
def _split_decisions(decisions):
    """Accept either the full decisions document or a bare ``{file: decision}``."""
    if not isinstance(decisions, dict):
        return "", "", {}
    inner = decisions.get("decisions")
    if isinstance(inner, dict):
        return _text(decisions.get("reviewer")), _text(decisions.get("reviewed_at")), inner
    return "", "", {
        str(key): value for key, value in decisions.items() if isinstance(value, dict)
    }


def _match_candidate(record: dict, threshold=None, candidate_id=None) -> dict | None:
    """Find the candidate referenced by ``candidate_id`` and/or ``threshold``.

    ``candidate_id`` (e.g. ``"t2"``, written by the page and honoured by the
    Lead's CLI) wins; otherwise the candidate whose threshold matches to
    6-significant-digit tolerance is used.
    """
    candidates = [c for c in (record.get("candidates") or []) if isinstance(c, dict)]
    if not candidates:
        return None
    if candidate_id is not None and str(candidate_id).strip() != "":
        wanted = str(candidate_id).strip()
        for candidate in candidates:
            if str(candidate.get("candidate_id") or "") == wanted:
                return candidate
    target = _num(threshold)
    if target is None:
        return None
    best: dict | None = None
    best_delta: float | None = None
    for candidate in candidates:
        value = _num(candidate.get("threshold"))
        if value is None:
            continue
        delta = abs(value - target)
        if best_delta is None or delta < best_delta:
            best, best_delta = candidate, delta
    if best is None or best_delta is None or best_delta > _THRESHOLD_MATCH_TOL:
        return None
    return best


def _prepare_audit(record: dict) -> None:
    """Give every record the audit fields without overwriting earlier runs."""
    record.setdefault("source", "auto")
    record.setdefault("original_confidence", _round6(record.get("confidence")))
    record.setdefault("original_threshold", _round6(record.get("threshold")))
    record.setdefault("reviewed_by", None)
    record.setdefault("review_note", "")
    record.setdefault("review_action", None)
    record.setdefault("reviewed_at", None)


def _mark_reviewed(record: dict, action: str, source: str, reviewer: str, reviewed_at: str, note: str) -> None:
    record["source"] = source
    record["status"] = "rejected" if action == "reject" else "done"
    record["reviewed_by"] = reviewer
    record["review_note"] = note
    record["review_action"] = action
    record["reviewed_at"] = reviewed_at


def _warn(record: dict, summary: dict, file: str, message: str) -> None:
    warnings = record.setdefault("review_warnings", [])
    if not isinstance(warnings, list):
        warnings = []
        record["review_warnings"] = warnings
    if message not in warnings:
        warnings.append(message)
    summary["warnings"].append({"file": file, "message": message})
    summary["warning_count"] += 1


def _set_measurement(record: dict, total, consumed) -> None:
    total_rounded = _round6(total)
    consumed_rounded = _round6(consumed)
    record["total_cm2"] = total_rounded
    record["consumed_cm2"] = consumed_rounded
    record["percent_consumed"] = _round6(_percent(consumed_rounded, total_rounded))


def _validated_numbers(total, consumed, record: dict, summary: dict, file: str, action: str):
    """Validate human numbers, warning instead of raising; return ``(t, c)`` or None."""
    total_number = _num(total)
    consumed_number = _num(consumed)
    if total_number is None or total_number <= 0:
        _warn(record, summary, file, f"{action}: total_cm2 must be a positive number; ignored")
        return None
    if consumed_number is None:
        consumed_number = 0.0
        _warn(record, summary, file, f"{action}: missing consumed_cm2, assumed 0")
    if consumed_number < 0:
        consumed_number = 0.0
        _warn(record, summary, file, f"{action}: negative consumed_cm2 clamped to 0")
    if consumed_number > total_number:
        consumed_number = total_number
        _warn(record, summary, file, f"{action}: consumed_cm2 > total_cm2, clamped to total")
    return total_number, consumed_number


def _set_percent_only(record: dict, percent) -> bool:
    """Accept a relative result (no scale found): only ``percent_consumed``."""
    value = _num(percent)
    if value is None:
        return False
    record["percent_consumed"] = _round6(min(100.0, max(0.0, value)))
    return True


def _do_accept(record: dict, decision: dict, summary: dict, file: str) -> bool | None:
    candidate = _match_candidate(
        record, decision.get("threshold"), decision.get("candidate_id")
    )
    if candidate is not None:
        total = _num(candidate.get("total_cm2"))
        consumed = _num(candidate.get("consumed_cm2"))
        threshold = _num(candidate.get("threshold"))
        percent = _num(candidate.get("percent_consumed"))
    else:
        total = _num(decision.get("total_cm2"))
        consumed = _num(decision.get("consumed_cm2"))
        threshold = _num(decision.get("threshold"))
        percent = _num(decision.get("percent_consumed"))
        if total is None and percent is None:
            _warn(
                record,
                summary,
                file,
                f"accept: no candidate matching candidate_id/threshold "
                f"{decision.get('candidate_id')!r}/{decision.get('threshold')!r} "
                "and no usable numbers; ignored",
            )
            return None
        _warn(
            record,
            summary,
            file,
            "accept: no matching candidate; used the numbers in the decision",
        )
    if total is None:
        # Relative-only measurement (scale not found): accept the percentage.
        if not _set_percent_only(record, percent):
            _warn(record, summary, file, "accept: candidate has no usable numbers; ignored")
            return None
        if threshold is not None:
            record["threshold"] = _round6(threshold)
        return True
    numbers = _validated_numbers(total, consumed, record, summary, file, "accept")
    if numbers is None:
        return None
    total, consumed = numbers
    _set_measurement(record, total, consumed)
    if threshold is not None:
        record["threshold"] = _round6(threshold)
    return True


def _do_override(record: dict, decision: dict, summary: dict, file: str) -> bool | None:
    total = _num(decision.get("total_cm2"))
    percent = _num(decision.get("percent_consumed"))
    if total is None and percent is not None:
        if not _set_percent_only(record, percent):
            return None
        threshold = _num(decision.get("threshold"))
        if threshold is not None:
            record["threshold"] = _round6(threshold)
        return True
    numbers = _validated_numbers(
        total, decision.get("consumed_cm2"), record, summary, file, "override"
    )
    if numbers is None:
        return None
    total, consumed = numbers
    _set_measurement(record, total, consumed)
    threshold = _num(decision.get("threshold"))
    if threshold is not None:
        record["threshold"] = _round6(threshold)
    return True


def _do_reject(record: dict) -> bool:
    """Reject-as-unusable: keep no final measurement value."""
    record["total_cm2"] = None
    record["consumed_cm2"] = None
    record["percent_consumed"] = None
    return True


def apply(records, decisions, cfg=None, reviewer=None, summary_out=None) -> list[dict]:
    """Merge human ``decisions`` into ``records`` and return *new* records.

    ``decisions`` is the ``decisions`` object from ``decisions.json``
    (``{file: decision}``) or the full document.  ``reviewer`` is the CLI
    ``--reviewer`` value; when omitted the payload reviewer is used, then
    ``"unknown"``.  The input records are never mutated and the output is
    JSON-serialisable.

    Unknown decision files / unknown actions / invalid numbers do not raise:
    each affected record gets a ``review_warnings`` list entry and the same
    message is collected in the optional ``summary_out`` JSON file
    (``{"reviewer", "reviewed_at", "counts", "changed", "unknown_files",
    "warnings", "warning_count"}``).
    """
    cfg = cfg or Config()  # noqa: F841 - kept for signature compatibility
    payload_reviewer, reviewed_at_raw, decision_map = _split_decisions(decisions)
    reviewer = (
        _text(reviewer).strip()
        or payload_reviewer.strip()
        or "unknown"
    )
    source = f"review:{reviewer}"
    reviewed_at = reviewed_at_raw.strip() or _now_iso()

    summary = {
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "counts": {"accept": 0, "override": 0, "reject": 0, "unreviewed": 0},
        "changed": [],
        "unknown_files": [],
        "warnings": [],
        "warning_count": 0,
    }

    output: list[dict] = []
    seen: set[str] = set()
    record_list = records if isinstance(records, (list, tuple)) else []
    for position, raw_record in enumerate(record_list):
        if not isinstance(raw_record, dict):
            output.append(copy.deepcopy(raw_record))
            continue
        record = copy.deepcopy(raw_record)
        file = _file_of(record, position)
        seen.add(file)
        _prepare_audit(record)

        decision = decision_map.get(file)
        if not isinstance(decision, dict):
            summary["counts"]["unreviewed"] += 1
            output.append(_jsonable(record))
            continue

        action = _text(decision.get("action")).strip().lower()
        note = _text(decision.get("note"))

        if action == "accept":
            applied = _do_accept(record, decision, summary, file)
        elif action == "override":
            applied = _do_override(record, decision, summary, file)
        elif action == "reject":
            applied = _do_reject(record)
        else:
            _warn(record, summary, file, f"unknown decision action {action!r}; ignored")
            applied = None

        if applied:
            _mark_reviewed(record, action, source, reviewer, reviewed_at, note)
            summary["counts"][action] += 1
            summary["changed"].append(
                {
                    "file": file,
                    "action": action,
                    "status": record.get("status"),
                    "total_cm2": record.get("total_cm2"),
                    "consumed_cm2": record.get("consumed_cm2"),
                    "percent_consumed": record.get("percent_consumed"),
                    "source": source,
                }
            )
        else:
            summary["counts"]["unreviewed"] += 1
        output.append(_jsonable(record))

    for key in decision_map:
        name = str(key)
        if name not in seen:
            summary["unknown_files"].append(name)
            summary["warnings"].append(
                {"file": name, "message": f"decision for unknown file {name!r} ignored"}
            )
            summary["warning_count"] += 1

    if summary_out is not None:
        path = Path(summary_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def export_payload(reviewer, decisions: dict, reviewed_at: str | None = None) -> dict:
    """Build the exact ``decisions.json`` document from an internal map.

    ``decisions`` maps file name -> ``{"action", "candidate_id", "threshold",
    "total_cm2", "consumed_cm2", "percent_consumed", "note"}``.  Mirrors
    ``decisionToSchema`` in the generated page so the download schema can be
    tested without a browser.
    """
    payload: dict = {
        "reviewer": _text(reviewer).strip() or "unknown",
        "reviewed_at": _text(reviewed_at).strip() or _now_iso(),
        "decisions": {},
    }
    if not isinstance(decisions, dict):
        return payload
    for file, decision in decisions.items():
        if not isinstance(decision, dict):
            continue
        action = _text(decision.get("action")).strip().lower()
        note = _text(decision.get("note"))
        if action == "accept":
            entry = {
                "action": "accept",
                "threshold": _round6(decision.get("threshold")),
                "total_cm2": _round6(decision.get("total_cm2")),
                "consumed_cm2": _round6(decision.get("consumed_cm2")),
                "note": note,
            }
            candidate_id = _text(decision.get("candidate_id")).strip()
            if candidate_id:
                entry["candidate_id"] = candidate_id
            percent = _round6(decision.get("percent_consumed"))
            if percent is not None and entry["total_cm2"] is None:
                # relative-only result: no scale, so the percentage is the value
                entry["percent_consumed"] = percent
            payload["decisions"][str(file)] = entry
        elif action == "override":
            entry = {
                "action": "override",
                "total_cm2": _round6(decision.get("total_cm2")),
                "consumed_cm2": _round6(decision.get("consumed_cm2")),
                "note": note,
            }
            percent = _round6(decision.get("percent_consumed"))
            if percent is not None and entry["total_cm2"] is None:
                entry["percent_consumed"] = percent
            payload["decisions"][str(file)] = entry
        elif action == "reject":
            payload["decisions"][str(file)] = {"action": "reject", "note": note}
        # unknown actions are simply omitted from the exported document
    return payload


# ---------------------------------------------------------------------------
# HTML page rendering
# ---------------------------------------------------------------------------
def _json_for_script(value) -> str:
    """JSON safe to embed inside a ``<script>`` element."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _render_html(queue: dict) -> str:
    """Fill the offline HTML template with the embedded queue data."""
    # Replace the queue id before embedding the data, so user-supplied text can
    # never be mistaken for a template token.
    return _HTML_TEMPLATE.replace(
        "__QUEUE_ID__", _text(queue.get("queue_id")) or "empty"
    ).replace("__QUEUE_JSON__", _json_for_script(queue))


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeafByte 人工审核</title>
<style>
:root{--bg:#f3f5f9;--card:#fff;--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--accent:#2563eb;--ok:#059669;--warn:#b45309;--bad:#dc2626}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;z-index:30;display:flex;flex-wrap:wrap;gap:12px;align-items:center;background:#fff;border-bottom:1px solid var(--line);padding:8px 14px;box-shadow:0 1px 4px rgba(0,0,0,.04)}
.brand{font-weight:700;font-size:15px;white-space:nowrap}
.meta{color:var(--muted);font-size:12px;flex:1;min-width:170px}
.progress{display:flex;align-items:center;gap:8px;min-width:200px}
.progress .bar{width:130px;height:8px;background:#e8edf5;border-radius:5px;overflow:hidden}
.progress .bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#22c55e,#2563eb);transition:width .2s}
#progressText{font-size:12px;white-space:nowrap}
.reviewer{font-size:12px;color:var(--muted);white-space:nowrap}
.reviewer input{width:118px;padding:5px 8px;border:1px solid var(--line);border-radius:6px;font-size:13px}
.btn{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:7px;padding:6px 12px;font-size:13px;cursor:pointer;white-space:nowrap;font-family:inherit}
.btn:hover{background:#f8fafc}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.primary:hover{filter:brightness(1.06)}
.btn.danger{color:#991b1b;border-color:#fecaca;background:#fef2f2}
.btn.ghost{color:var(--muted)}
.btn:disabled{opacity:.45;cursor:not-allowed}
.layout{display:flex;align-items:flex-start;gap:14px;padding:14px;max-width:1500px;margin:0 auto}
aside{position:sticky;top:70px;width:252px;max-height:calc(100vh - 92px);overflow:auto;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px;flex:none}
.side-head{font-weight:600;font-size:12px;color:var(--muted);padding:4px 6px 8px;border-bottom:1px solid var(--line);margin-bottom:6px}
.side-item{display:flex;align-items:center;gap:6px;padding:6px 7px;border-radius:7px;cursor:pointer;font-size:12.5px}
.side-item:hover{background:#f1f5f9}
.side-item.active{background:#e0e7ff;font-weight:600}
.side-item .mark{width:16px;text-align:center;font-weight:700;flex:none}
.mark.ok{color:var(--ok)}.mark.bad{color:var(--bad)}.mark.override{color:#7c3aed}
.side-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.side-conf{color:var(--muted);font-size:11px;flex:none}
.side-foot{margin-top:8px;padding:8px 6px 2px;border-top:1px solid var(--line);color:var(--muted);font-size:11px;line-height:1.8}
main{flex:1;min-width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.card-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;flex-wrap:wrap}
.fname{font-size:16px}
.idx{color:var(--muted);font-size:12px}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-size:11.5px;background:#f1f5f9;color:#475569;border-radius:999px;padding:2px 9px;white-space:nowrap}
.chip.warn{background:#fef3c7;color:#92400e}
.chip.ok{background:#dcfce7;color:#166534}
.chip.bad{background:#fee2e2;color:#991b1b}
.chip.override{background:#ede9fe;color:#5b21b6}
.nums{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:12px 0 8px}
.num{background:#f8fafc;border:1px solid #eef2f7;border-radius:8px;padding:7px 10px}
.num .k{font-size:11px;color:var(--muted)}
.num .v{font-size:17px;font-weight:650;letter-spacing:.2px}
.notes-row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:4px 0 10px}
.reason{background:#f8fafc;border-left:3px solid #94a3b8;padding:5px 9px;border-radius:4px;font-size:12.5px;color:#334155}
.reason.err{border-left-color:var(--bad);color:#991b1b;background:#fef2f2}
.flag{font-size:11px;background:#fef3c7;color:#92400e;border-radius:5px;padding:1px 7px}
.shots{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin:6px 0 14px}
.shots figure{margin:0}
.shots figcaption{font-size:12px;color:var(--muted);margin-bottom:4px}
.shot{position:relative;background:#0f172a;border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center;min-height:180px}
.shot img{max-width:100%;max-height:58vh;display:block}
.ph{color:#cbd5e1;font-size:12.5px;padding:36px 14px;text-align:center;line-height:1.8}
.sec-title{font-size:13px;font-weight:650;margin:4px 0 8px}
.hint{color:var(--muted);font-weight:400;font-size:12px}
.cand-row{display:flex;gap:9px;overflow-x:auto;padding:2px 2px 8px}
.cand{flex:none;width:152px;border:2px solid var(--line);border-radius:9px;background:#fff;padding:5px;text-align:left;cursor:pointer;display:flex;flex-direction:column;gap:2px;font-size:11.5px;color:#334155;font-family:inherit}
.cand:hover{border-color:#c7d2fe}
.cand.selected{border-color:var(--accent);box-shadow:0 0 0 2px #dbeafe}
.cand-thumb{display:flex;align-items:center;justify-content:center;height:92px;background:#0f172a;border-radius:5px;overflow:hidden;margin-bottom:3px}
.cand-thumb img{max-width:100%;max-height:100%}
.cand-thumb .ph{padding:8px;font-size:11px}
.cand-name{font-weight:700;color:var(--ink)}
.cand-line{display:block;white-space:nowrap}
.cand-line.err{color:var(--bad);white-space:normal}
.empty{color:var(--muted);font-size:13px;padding:14px;background:#f8fafc;border-radius:8px}
.manual{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin:12px 0;padding:10px;background:#f8fafc;border:1px solid #eef2f7;border-radius:9px}
.manual label{display:flex;flex-direction:column;gap:4px;font-size:11.5px;color:var(--muted)}
.manual label.grow{flex:1;min-width:220px}
.manual input{padding:6px 8px;border:1px solid #d7dde8;border-radius:6px;font-size:13px;background:#fff;font-family:inherit}
.decided{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46;border-radius:8px;padding:8px 10px;font-size:12.5px;margin-bottom:10px}
.actions{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
.actions .spacer{flex:1}
.legend{margin:12px 0 30px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:9px 12px;font-size:12px;color:#475569;display:flex;gap:8px;flex-wrap:wrap;align-items:center;line-height:1.9}
.sw{display:inline-block;width:13px;height:13px;border-radius:3px;vertical-align:-2px;margin-left:10px}
.sw.red{background:rgba(255,40,40,.75)}.sw.blue{background:rgba(40,120,255,.75)}.sw.yellow{background:rgba(255,220,0,.95);border:1px solid #d4a900}
.toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(20px);background:#111827;color:#fff;padding:10px 16px;border-radius:9px;font-size:13px;opacity:0;pointer-events:none;transition:.2s;z-index:50;max-width:80vw;text-align:center}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.toast.warn{background:#92400e}.toast.ok{background:#065f46}
@media(max-width:900px){aside{display:none}.shots{grid-template-columns:1fr}}
</style>
</head>
<body>
<header id="topbar">
  <div class="brand">🍃 LeafByte 人工审核</div>
  <div class="meta" id="queueMeta"></div>
  <div class="progress"><div class="bar"><i id="progressBar"></i></div><span id="progressText">0 / 0 reviewed</span></div>
  <label class="reviewer">审核人 <input id="reviewer" type="text" maxlength="40" placeholder="姓名/工号" autocomplete="off"></label>
  <label class="btn ghost" for="importFile" title="导入之前的 decisions.json 继续审核">导入 decisions.json<input id="importFile" type="file" accept=".json,application/json" hidden></label>
  <button type="button" class="btn primary" id="exportBtn" title="导出 decisions.json（快捷键 E）">⬇ 导出 decisions.json</button>
</header>
<div class="layout">
  <aside>
    <div class="side-head">文件列表</div>
    <div id="fileList"></div>
    <div class="side-foot">← → 上一张 / 下一张<br>1-9 选择候选 · A 采纳<br>O 采用手填数值 · R 弃用<br>U / Ctrl+Z 撤销 · E 导出</div>
  </aside>
  <main>
    <div id="cardHost"></div>
    <div class="legend">
      <b>叠加图图例（与批处理绘制一致）：</b><span class="sw red"></span>红色 = 检测到的叶片
      <span class="sw blue"></span>蓝色 = 被啃食的洞（计入 consumed_cm2）
      <span class="sw yellow"></span>黄色 = 4 个比例尺标记 / 矫正四边框
    </div>
  </main>
</div>
<div id="toast" class="toast"></div>
<script>
"use strict";
const QUEUE = __QUEUE_JSON__;
const QUEUE_ID = "__QUEUE_ID__";
const LS_KEY = "leafbyte_review_" + QUEUE_ID;
const ITEMS = Array.isArray(QUEUE.items) ? QUEUE.items : [];
const STATE = { idx: 0, decisions: {}, notes: {}, manual: {}, selected: {}, undo: [] };

/* ---- pure helpers: the Python twin lives in tests/test_review.py ---- */
function num(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function round6(v) {
  const n = num(v);
  if (n === null) return null;
  if (n === 0) return 0;
  return Number(n.toPrecision(6));
}
function fmt(v, digits) {
  const n = num(v);
  if (n === null) return "—";
  return n.toFixed(digits === undefined ? 3 : digits);
}
function nowIso() {
  const d = new Date();
  const p = function (n) { return String(n).padStart(2, "0"); };
  return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) + "T" + p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
}
function esc(s) {
  return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}
function decisionToSchema(d) {
  if (!d || typeof d !== "object") return null;
  const note = d.note ? String(d.note) : "";
  if (d.action === "accept") {
    const entry = { action: "accept", threshold: round6(d.threshold), total_cm2: round6(d.total_cm2), consumed_cm2: round6(d.consumed_cm2), note: note };
    if (d.candidate_id) entry.candidate_id = String(d.candidate_id);
    if (num(entry.total_cm2) === null && num(d.percent_consumed) !== null) entry.percent_consumed = round6(d.percent_consumed);
    return entry;
  }
  if (d.action === "override") {
    const entry = { action: "override", total_cm2: round6(d.total_cm2), consumed_cm2: round6(d.consumed_cm2), note: note };
    if (num(entry.total_cm2) === null && num(d.percent_consumed) !== null) entry.percent_consumed = round6(d.percent_consumed);
    return entry;
  }
  if (d.action === "reject") return { action: "reject", note: note };
  return null;
}
function buildExportPayload(reviewer, decisions, reviewedAt) {
  const payload = {
    reviewer: String(reviewer || "").trim(),
    reviewed_at: reviewedAt || nowIso(),
    decisions: {}
  };
  ITEMS.forEach(function (item) {
    const d = decisions ? decisions[item.file] : null;
    const schema = d ? decisionToSchema(d) : null;
    if (schema) payload.decisions[item.file] = schema;
  });
  return payload;
}

/* ---- DOM helpers ---- */
function $(selector) { return document.querySelector(selector); }
let toastTimer = null;
function toast(message, kind) {
  const el = $("#toast");
  if (!el) return;
  el.textContent = String(message);
  el.className = "toast show" + (kind ? " " + kind : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function () { el.className = "toast"; }, 3200);
}
function imgFail(img) {
  img.style.display = "none";
  const host = img.parentNode;
  if (!host) return;
  const ph = document.createElement("div");
  ph.className = "ph";
  ph.textContent = img.getAttribute("data-ph") || "图片缺失";
  host.appendChild(ph);
  host.classList.add("missing");
}
function imgBlock(src, placeholder, cls) {
  if (!src) return '<div class="ph">' + esc(placeholder) + "</div>";
  return '<img class="' + esc(cls || "") + '" src="' + esc(src) + '" alt="" loading="lazy" data-ph="' + esc(placeholder) + '" onerror="imgFail(this)">';
}
function currentItem() { return ITEMS.length ? ITEMS[STATE.idx] : null; }
function decisionOf(item) { return item ? STATE.decisions[item.file] : null; }
function noteOf(item) {
  const typed = STATE.notes[item.file];
  if (typeof typed === "string") return typed;
  const d = STATE.decisions[item.file];
  return d && d.note ? String(d.note) : "";
}
function selectedCandidateIndex(item) {
  if (!item) return null;
  if (Object.prototype.hasOwnProperty.call(STATE.selected, item.file)) return STATE.selected[item.file];
  const fallback = item.default_selected;
  return fallback === null || fallback === undefined ? null : fallback;
}
function defaultManual(item) {
  const d = STATE.decisions[item.file];
  if (d && d.action !== "reject") return { total: d.total_cm2, consumed: d.consumed_cm2 };
  const selected = selectedCandidateIndex(item);
  let candidate = null;
  if (selected !== null && selected !== undefined) {
    candidate = (item.candidates || []).filter(function (c) { return c.index === selected; })[0] || null;
  }
  if (candidate) return { total: candidate.total_cm2, consumed: candidate.consumed_cm2 };
  return { total: item.total_cm2, consumed: item.consumed_cm2 };
}
function manualValues(item) {
  const def = defaultManual(item);
  const m = STATE.manual[item.file] || {};
  return {
    total: m.total !== undefined ? m.total : def.total,
    consumed: m.consumed !== undefined ? m.consumed : def.consumed
  };
}
function decidedText(d) {
  if (!d) return "";
  if (d.action === "accept") {
    if (num(d.total_cm2) === null && num(d.percent_consumed) !== null) {
      return "采纳相对结果（啃食 " + fmt(d.percent_consumed, 2) + "%，无比例尺）";
    }
    return "采纳候选（阈值 " + fmt(d.threshold, 4) + "，总 " + fmt(d.total_cm2) + " cm²，啃 " + fmt(d.consumed_cm2) + " cm²）";
  }
  if (d.action === "override") return "手填数值（总 " + fmt(d.total_cm2) + " cm²，啃 " + fmt(d.consumed_cm2) + " cm²）";
  if (d.action === "reject") return "弃用 / 需重拍" + (d.note ? "：" + d.note : "");
  return String(d.action || "");
}
function statusChip(item) {
  const d = decisionOf(item);
  if (d && d.action === "reject") return '<span class="chip bad">已弃用</span>';
  if (d) return '<span class="chip ok">已审：' + (d.action === "accept" ? "采纳候选" : d.action === "override" ? "手填数值" : esc(d.action)) + "</span>";
  return item.status === "failed" ? '<span class="chip warn">自动失败 · 待审</span>' : '<span class="chip warn">需复核</span>';
}

/* ---- rendering ---- */
function renderHeader() {
  const total = ITEMS.length;
  const done = ITEMS.filter(function (it) { return !!STATE.decisions[it.file]; }).length;
  const percent = total ? Math.round(100 * done / total) : 0;
  const textEl = $("#progressText");
  if (textEl) textEl.textContent = done + " / " + total + " reviewed (" + percent + "%)";
  const barEl = $("#progressBar");
  if (barEl) barEl.style.width = percent + "%";
  const metaEl = $("#queueMeta");
  if (metaEl) {
    const failed = ITEMS.filter(function (it) { return it.status === "failed"; }).length;
    metaEl.textContent = "共 " + total + " 张待审（其中自动失败 " + failed + " 张）· 生成于 " + (QUEUE.generated_at || "");
  }
}
function renderSidebar() {
  const host = $("#fileList");
  if (!host) return;
  host.innerHTML = ITEMS.map(function (item, i) {
    const d = STATE.decisions[item.file];
    const cls = "side-item" + (i === STATE.idx ? " active" : "") + (d ? " done" : "");
    let mark = "·", markClass = "";
    if (d) {
      mark = d.action === "reject" ? "✕" : d.action === "override" ? "✎" : "✓";
      markClass = d.action === "reject" ? "bad" : d.action === "override" ? "override" : "ok";
    } else if (item.status === "failed") {
      mark = "!";
      markClass = "bad";
    }
    return '<div class="' + cls + '" data-jump="' + i + '" title="' + esc(item.file) + '">'
      + '<span class="mark ' + markClass + '">' + mark + "</span>"
      + '<span class="side-name">' + esc(item.file) + "</span>"
      + '<span class="side-conf">' + fmt(item.confidence, 2) + "</span></div>";
  }).join("");
}
function cardHtml(item) {
  const d = decisionOf(item);
  const selected = selectedCandidateIndex(item);
  const man = manualValues(item);
  const note = noteOf(item);
  const candidates = item.candidates || [];
  const candHtml = candidates.length ? candidates.map(function (c) {
    const isSelected = selected === c.index;
    const label = c.candidate_id ? esc(c.candidate_id) : "#" + c.index;
    return '<button type="button" class="cand' + (isSelected ? " selected" : "") + '" data-act="cand" data-index="' + c.index + '" title="选择候选 ' + label + '">'
      + '<span class="cand-thumb">' + imgBlock(c.overlay, "无叠加图", "thumb") + "</span>"
      + '<span class="cand-name">' + (isSelected ? "✓ " : "") + label + "</span>"
      + '<span class="cand-line">阈值 ' + fmt(c.threshold, 4) + "</span>"
      + '<span class="cand-line">总 ' + fmt(c.total_cm2) + " · 啃 " + fmt(c.consumed_cm2) + " cm²</span>"
      + '<span class="cand-line">啃食 ' + fmt(c.percent_consumed, 2) + "%</span>"
      + (c.error ? '<span class="cand-line err">' + esc(c.error) + "</span>" : "")
      + "</button>";
  }).join("") : '<div class="empty">批处理没有给出候选：可以直接手填数值（O）或弃用（R）</div>';
  const flags = (item.flags || []).map(function (f) { return '<span class="flag">' + esc(f) + "</span>"; }).join("");
  const reason = item.reason ? '<div class="reason">' + esc(item.reason) + "</div>" : "";
  const error = item.error ? '<div class="reason err">' + esc(item.error) + "</div>" : "";
  const scaleChip = item.scale_found === false
    ? '<span class="chip bad">比例尺未找到</span>'
    : item.scale_found === true ? '<span class="chip">比例尺已找到</span>' : "";
  const numbers = '<div class="nums">'
    + '<div class="num"><div class="k">总面积 total_cm2</div><div class="v">' + fmt(item.total_cm2) + ' <small>cm²</small></div></div>'
    + '<div class="num"><div class="k">被啃食 consumed_cm2</div><div class="v">' + fmt(item.consumed_cm2) + ' <small>cm²</small></div></div>'
    + '<div class="num"><div class="k">啃食比例 percent_consumed</div><div class="v">' + fmt(item.percent_consumed, 2) + ' <small>%</small></div></div>'
    + '<div class="num"><div class="k">置信度 confidence</div><div class="v">' + fmt(item.confidence, 3) + '</div></div>'
    + "</div>";
  const decided = d
    ? '<div class="decided">已记录：' + esc(decidedText(d)) + '<button type="button" class="btn ghost" data-act="undo">撤销 (U)</button></div>'
    : "";
  const scaleNote = QUEUE.scale_cm ? " · 比例尺 " + fmt(QUEUE.scale_cm, 2) + " cm" : "";
  return '<article class="card">'
    + '<div class="card-head"><div><span class="idx">#' + item.index + '</span> <b class="fname">' + esc(item.file) + "</b></div>"
    + '<div class="chips">' + statusChip(item) + scaleChip
    + '<span class="chip">阈值 ' + fmt(item.threshold, 4) + "</span>"
    + '<span class="chip">置信度 ' + fmt(item.confidence, 3) + "</span>"
    + '<span class="chip">' + esc(item.status) + scaleNote + "</span></div></div>"
    + numbers
    + (reason || flags || error ? '<div class="notes-row">' + reason + flags + error + "</div>" : "")
    + '<div class="shots">'
    + '<figure><figcaption>原图 · ' + esc(item.file) + '</figcaption><div class="shot">' + imgBlock(item.image, "原图缺失（记录里的 path 找不到）", "") + "</div></figure>"
    + '<figure><figcaption>原图 + 比例尺标注（黄色框 / 叉）</figcaption><div class="shot">' + imgBlock(item.overlay_original, "比例尺叠加图缺失", "") + "</div></figure>"
    + '<figure><figcaption>矫正后 · 叶片(红) / 啃食(蓝)（阈值 ' + fmt(item.threshold, 4) + '）</figcaption><div class="shot">' + imgBlock(item.overlay_rectified || item.overlay, "叠加图缺失（看候选项缩略图）", "") + "</div></figure>"
    + "</div>"
    + '<div class="sec-title">候选阈值 <span class="hint">点击选择，或按 1-9；选好后按 A 采纳</span></div>'
    + '<div class="cand-row">' + candHtml + "</div>"
    + '<div class="manual">'
    + '<label>手填总面积 (cm²)<input id="mTotal" data-field="total" type="number" step="0.001" min="0" value="' + esc(man.total === null || man.total === undefined ? "" : man.total) + '"></label>'
    + '<label>手填被啃食 (cm²)<input id="mConsumed" data-field="consumed" type="number" step="0.001" min="0" value="' + esc(man.consumed === null || man.consumed === undefined ? "" : man.consumed) + '"></label>'
    + '<label class="grow">审核备注<input id="mNote" data-field="note" type="text" maxlength="200" value="' + esc(note) + '" placeholder="例如：反光，需重拍 / ImageJ 手测"></label>'
    + "</div>"
    + decided
    + '<div class="actions">'
    + '<button type="button" class="btn primary" data-act="accept">采纳候选 (A)</button>'
    + '<button type="button" class="btn" data-act="override">采用手填数值 (O)</button>'
    + '<button type="button" class="btn danger" data-act="reject">弃用 / 需重拍 (R)</button>'
    + '<span class="spacer"></span>'
    + '<button type="button" class="btn ghost" data-act="prev"' + (STATE.idx <= 0 ? " disabled" : "") + ">← 上一张</button>"
    + '<button type="button" class="btn ghost" data-act="next"' + (STATE.idx >= ITEMS.length - 1 ? " disabled" : "") + ">下一张 →</button>"
    + "</div>"
    + "</article>";
}
function render() {
  renderHeader();
  renderSidebar();
  const host = $("#cardHost");
  if (!host) return;
  if (!ITEMS.length) {
    host.innerHTML = '<div class="card"><h2>没有待审核样本</h2><p class="hint">批处理没有把任何图片路由到人工审核，或还没有生成结果。</p></div>';
    return;
  }
  host.innerHTML = cardHtml(ITEMS[STATE.idx]);
}

/* ---- actions ---- */
function persist() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({
      reviewer: ($("#reviewer") || { value: "" }).value,
      decisions: STATE.decisions
    }));
  } catch (e) { /* file:// may not provide localStorage */ }
}
function restore() {
  let saved = null;
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) saved = JSON.parse(raw);
  } catch (e) { return; }
  if (!saved || typeof saved !== "object") return;
  if (saved.reviewer) {
    const reviewer = $("#reviewer");
    if (reviewer) reviewer.value = String(saved.reviewer);
  }
  if (saved.decisions && typeof saved.decisions === "object") {
    const known = {};
    ITEMS.forEach(function (item) { known[item.file] = true; });
    Object.keys(saved.decisions).forEach(function (name) {
      if (known[name] && saved.decisions[name] && typeof saved.decisions[name] === "object") {
        STATE.decisions[name] = saved.decisions[name];
      }
    });
  }
}
function nextUnreviewed() {
  const n = ITEMS.length;
  if (!n) return null;
  for (let k = 1; k <= n; k += 1) {
    const idx = (STATE.idx + k) % n;
    if (!STATE.decisions[ITEMS[idx].file]) return idx;
  }
  return null;
}
function setDecision(item, decision) {
  const file = item.file;
  const before = STATE.decisions[file] ? JSON.parse(JSON.stringify(STATE.decisions[file])) : null;
  STATE.decisions[file] = decision;
  STATE.undo.push({ file: file, before: before, selected: STATE.selected[file], manual: STATE.manual[file] });
  if (STATE.undo.length > 200) STATE.undo.shift();
  delete STATE.manual[file];
  persist();
  const next = nextUnreviewed();
  if (next !== null) STATE.idx = next;
  render();
  toast("已记录 " + file + "：" + decidedText(decision), "ok");
}
function undo() {
  const snapshot = STATE.undo.pop();
  if (!snapshot) { toast("没有可撤销的操作", "warn"); return; }
  if (snapshot.before) STATE.decisions[snapshot.file] = snapshot.before;
  else delete STATE.decisions[snapshot.file];
  if (snapshot.selected !== undefined) STATE.selected[snapshot.file] = snapshot.selected;
  if (snapshot.manual !== undefined) STATE.manual[snapshot.file] = snapshot.manual;
  for (let i = 0; i < ITEMS.length; i += 1) {
    if (ITEMS[i].file === snapshot.file) { STATE.idx = i; break; }
  }
  persist();
  render();
  toast("已撤销 " + snapshot.file, "ok");
}
function gotoItem(idx) {
  if (!ITEMS.length) return;
  STATE.idx = Math.min(ITEMS.length - 1, Math.max(0, idx | 0));
  render();
  window.scrollTo(0, 0);
}
function step(delta) {
  if (!ITEMS.length) return;
  const idx = Math.min(ITEMS.length - 1, Math.max(0, STATE.idx + delta));
  if (idx !== STATE.idx) gotoItem(idx);
}
function selectCandidate(index) {
  const item = currentItem();
  if (!item) return;
  const exists = (item.candidates || []).some(function (c) { return c.index === index; });
  if (!exists) { toast("没有候选 #" + index, "warn"); return; }
  if (selectedCandidateIndex(item) === index) STATE.selected[item.file] = null;
  else STATE.selected[item.file] = index;
  delete STATE.manual[item.file];
  render();
}
function accept() {
  const item = currentItem();
  if (!item) return;
  const selected = selectedCandidateIndex(item);
  let candidate = null;
  if (selected !== null && selected !== undefined) {
    candidate = (item.candidates || []).filter(function (c) { return c.index === selected; })[0] || null;
  }
  const source = candidate || {
    threshold: item.threshold,
    total_cm2: item.total_cm2,
    consumed_cm2: item.consumed_cm2,
    percent_consumed: item.percent_consumed
  };
  const candidateId = candidate && candidate.candidate_id ? String(candidate.candidate_id) : null;
  const total = num(source.total_cm2);
  if (total === null || total <= 0) {
    const percent = num(source.percent_consumed);
    if (percent !== null) {
      // relative-only result: no scale was found, so only the % is available
      setDecision(item, {
        action: "accept",
        candidate_id: candidateId,
        threshold: round6(source.threshold),
        total_cm2: null,
        consumed_cm2: null,
        percent_consumed: round6(percent),
        note: noteOf(item)
      });
      return;
    }
    toast("这张图没有可用的测量数值，请手填（O）或弃用（R）", "warn");
    return;
  }
  let consumed = num(source.consumed_cm2);
  if (consumed === null) consumed = 0;
  setDecision(item, {
    action: "accept",
    candidate_id: candidateId,
    threshold: round6(source.threshold),
    total_cm2: round6(total),
    consumed_cm2: round6(consumed),
    note: noteOf(item)
  });
}
function overrideManual() {
  const item = currentItem();
  if (!item) return;
  const totalInput = $("#mTotal");
  const consumedInput = $("#mConsumed");
  const total = num(totalInput ? totalInput.value : null);
  let consumed = num(consumedInput ? consumedInput.value : null);
  if (total === null || total <= 0) { toast("总面积必须是大于 0 的数字", "warn"); if (totalInput) totalInput.focus(); return; }
  if (consumed === null) consumed = 0;
  if (consumed < 0) { toast("被啃食面积不能为负数", "warn"); return; }
  if (consumed > total) { toast("被啃食面积不能大于总面积", "warn"); return; }
  setDecision(item, {
    action: "override",
    total_cm2: round6(total),
    consumed_cm2: round6(consumed),
    note: noteOf(item)
  });
}
function reject() {
  const item = currentItem();
  if (!item) return;
  setDecision(item, { action: "reject", note: noteOf(item) });
}
function exportDecisions() {
  const reviewerEl = $("#reviewer");
  const reviewer = reviewerEl ? String(reviewerEl.value || "").trim() : "";
  if (!reviewer) { toast("请先在右上角填写审核人姓名，再导出", "warn"); if (reviewerEl) reviewerEl.focus(); return; }
  const payload = buildExportPayload(reviewer, STATE.decisions);
  const count = Object.keys(payload.decisions).length;
  if (!count) { toast("还没有任何审核结果", "warn"); return; }
  const text = JSON.stringify(payload, null, 2);
  const blob = new Blob([text], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "decisions.json";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  toast("已导出 decisions.json（" + count + " 条），请把它放回 out/review/", "ok");
}
function importDecisions(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function () {
    let data = null;
    try { data = JSON.parse(String(reader.result)); } catch (e) { toast("不是有效的 decisions.json 文件", "warn"); return; }
    if (!data || typeof data !== "object" || !data.decisions || typeof data.decisions !== "object") {
      toast("decisions.json 里没有 decisions 字段", "warn");
      return;
    }
    const known = {};
    ITEMS.forEach(function (item) { known[item.file] = item; });
    let imported = 0;
    let skipped = 0;
    Object.keys(data.decisions).forEach(function (name) {
      const d = data.decisions[name];
      if (!known[name] || !d || typeof d !== "object") { skipped += 1; return; }
      const action = String(d.action || "");
      if (action === "reject") {
        STATE.decisions[name] = { action: "reject", note: String(d.note || "") };
        imported += 1;
        return;
      }
      if (action === "accept" || action === "override") {
        const total = num(d.total_cm2);
        let consumed = num(d.consumed_cm2);
        const percent = num(d.percent_consumed);
        if (total === null && percent === null) { skipped += 1; return; }
        if (total !== null && total <= 0) { skipped += 1; return; }
        if (consumed === null) consumed = 0;
        STATE.decisions[name] = {
          action: action,
          candidate_id: d.candidate_id ? String(d.candidate_id) : null,
          threshold: round6(d.threshold),
          total_cm2: total === null ? null : round6(total),
          consumed_cm2: total === null ? null : round6(consumed),
          percent_consumed: percent === null ? null : round6(percent),
          note: String(d.note || "")
        };
        imported += 1;
        return;
      }
      skipped += 1;
    });
    if (data.reviewer) {
      const reviewerEl = $("#reviewer");
      if (reviewerEl) reviewerEl.value = String(data.reviewer);
    }
    STATE.undo = [];
    persist();
    render();
    toast("导入 " + imported + " 条" + (skipped ? "，忽略 " + skipped + " 条" : ""), "ok");
  };
  reader.readAsText(file, "utf-8");
  event.target.value = "";
}
function onKey(event) {
  const target = event.target;
  const typing = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT");
  if (typing) {
    if (event.key === "Escape" && target.blur) target.blur();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && String(event.key).toLowerCase() === "z") {
    event.preventDefault();
    undo();
    return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key;
  let handled = true;
  if (key === "ArrowLeft" || key === "PageUp") step(-1);
  else if (key === "ArrowRight" || key === "PageDown") step(1);
  else if (/^[1-9]$/.test(key)) selectCandidate(Number(key));
  else if (key === "a" || key === "A") accept();
  else if (key === "o" || key === "O") overrideManual();
  else if (key === "r" || key === "R") reject();
  else if (key === "u" || key === "U") undo();
  else if (key === "e" || key === "E") exportDecisions();
  else handled = false;
  if (handled) event.preventDefault();
}
function bind() {
  const cardHost = $("#cardHost");
  if (cardHost) {
    cardHost.addEventListener("click", function (event) {
      const target = event.target && event.target.closest ? event.target.closest("[data-act]") : null;
      if (!target) return;
      const act = target.getAttribute("data-act");
      if (act === "cand") selectCandidate(Number(target.getAttribute("data-index")));
      else if (act === "accept") accept();
      else if (act === "override") overrideManual();
      else if (act === "reject") reject();
      else if (act === "undo") undo();
      else if (act === "prev") step(-1);
      else if (act === "next") step(1);
    });
    cardHost.addEventListener("input", function (event) {
      const item = currentItem();
      if (!item) return;
      const el = event.target;
      if (el.id === "mTotal" || el.id === "mConsumed") {
        const manual = STATE.manual[item.file] || {};
        if (el.id === "mTotal") manual.total = el.value;
        else manual.consumed = el.value;
        STATE.manual[item.file] = manual;
      } else if (el.id === "mNote") {
        STATE.notes[item.file] = el.value;
      }
    });
  }
  const fileList = $("#fileList");
  if (fileList) {
    fileList.addEventListener("click", function (event) {
      const target = event.target && event.target.closest ? event.target.closest("[data-jump]") : null;
      if (target) gotoItem(Number(target.getAttribute("data-jump")));
    });
  }
  const exportBtn = $("#exportBtn");
  if (exportBtn) exportBtn.addEventListener("click", exportDecisions);
  const importFile = $("#importFile");
  if (importFile) importFile.addEventListener("change", importDecisions);
  const reviewerEl = $("#reviewer");
  if (reviewerEl) reviewerEl.addEventListener("input", persist);
  document.addEventListener("keydown", onKey);
}
function boot() {
  restore();
  if (ITEMS.length) {
    let first = 0;
    for (let i = 0; i < ITEMS.length; i += 1) {
      if (!STATE.decisions[ITEMS[i].file]) { first = i; break; }
    }
    STATE.idx = first;
  }
  render();
  bind();
  const restored = Object.keys(STATE.decisions).length;
  if (restored) toast("已恢复上次进度：" + restored + " 条（浏览器本地保存）", "ok");
}
if (typeof window !== "undefined" && typeof document !== "undefined") {
  boot();
}
</script>
</body>
</html>
"""
