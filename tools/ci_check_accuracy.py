"""CI accuracy gate: fail the build when automatic measurements go wrong.

The point of the project is "automate, but never answer confidently when the
measurement is wrong".  This script turns that promise into an exit code:

* no sample may ``failed``,
* **no automatically accepted sample may be outside the area tolerance**
  (``--tolerance`` relative, or ``--abs-tolerance`` cm², whichever is larger),
* automation must keep working (``--min-auto`` samples accepted),
* the review queue must stay small (``--max-review``),
* the mean error of auto-accepted samples must stay below ``--max-mean-error``.

It reads ``<run>/run.json`` (accuracy block) and ``<run>/results.jsonl`` (the
per-sample values), so it works on any run folder.

    python tools/ci_check_accuracy.py --run out/paper
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load(run_dir: Path) -> tuple[dict, list[dict]]:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise SystemExit(f"no run.json in {run_dir}")
    info = json.loads(run_json.read_text(encoding="utf-8"))
    results = run_dir / "results.jsonl"
    records = []
    if results.exists():
        records = [json.loads(line) for line in
                   results.read_text(encoding="utf-8").splitlines() if line.strip()]
    return info, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run folder (out/paper)")
    parser.add_argument("--tolerance", type=float, default=0.015,
                        help="relative area tolerance used to judge auto-accepts")
    parser.add_argument("--abs-tolerance", type=float, default=0.05,
                        help="absolute area tolerance in cm^2")
    parser.add_argument("--min-auto", type=int, default=20,
                        help="fail if fewer samples than this are auto-accepted")
    parser.add_argument("--max-review", type=int, default=10,
                        help="fail if more samples than this need review")
    parser.add_argument("--max-mean-error", type=float, default=0.20,
                        help="max mean |error| (cm^2) of auto-accepted samples")
    parser.add_argument("--max-abs-error", type=float, default=0.45,
                        help="max single |error| (cm^2) of auto-accepted samples")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    info, records = load(run_dir)
    summary = info.get("summary") or {}
    accuracy = info.get("accuracy") or {}

    failed = int(summary.get("failed", 0))
    auto = int(summary.get("auto", 0))
    review = int(summary.get("review", 0))
    compared = int(accuracy.get("auto_compared", 0))
    within = int(accuracy.get("auto_within_tolerance", 0))
    false_accepts = int(accuracy.get("false_auto_accept", 0))
    mean_error = accuracy.get("auto_mean_abs_error_total_cm2")
    max_error = accuracy.get("auto_max_abs_error_total_cm2")

    # recompute the tolerance check locally so a stale run.json cannot hide it
    bad: list[str] = []
    for record in records:
        if record.get("status") != "auto":
            continue
        reference = record.get("reference") or {}
        expected = reference.get("total_cm2")
        measured = record.get("total_cm2")
        if expected is None or measured is None:
            continue
        allowed = max(args.abs_tolerance, args.tolerance * abs(float(expected)))
        error = float(measured) - float(expected)
        if abs(error) > allowed:
            bad.append(f"{record['file']}: measured {measured:.3f} vs {expected:.3f} "
                       f"(error {error:+.3f} cm^2 > {allowed:.3f})")

    problems: list[str] = []
    if failed:
        problems.append(f"{failed} sample(s) failed")
    if false_accepts:
        problems.append(f"{false_accepts} auto-accepted sample(s) outside tolerance "
                        f"(run.json)")
    if bad:
        problems.append("auto-accepted errors exceed the tolerance:\n  " +
                        "\n  ".join(bad))
    if compared and within < compared:
        problems.append(f"only {within}/{compared} auto-accepted samples are within tolerance")
    if auto < args.min_auto:
        problems.append(f"only {auto} auto-accepted (expected >= {args.min_auto}); "
                        "automation regressed")
    if review > args.max_review:
        problems.append(f"{review} samples need review (expected <= {args.max_review})")
    if mean_error is not None and mean_error > args.max_mean_error:
        problems.append(f"mean error {mean_error} cm^2 > {args.max_mean_error}")
    if max_error is not None and max_error > args.max_abs_error:
        problems.append(f"max error {max_error} cm^2 > {args.max_abs_error}")

    print(f"run            : {run_dir}")
    print(f"samples        : {summary.get('images', len(records))}")
    print(f"auto / review  : {auto} / {review}  (failed {failed})")
    print(f"auto accuracy  : {within}/{compared} within tolerance "
          f"({args.tolerance:.1%} or {args.abs_tolerance} cm^2)")
    print(f"mean / max err : {mean_error} / {max_error} cm^2")

    if problems:
        print("\nACCURACY GATE FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nACCURACY GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
