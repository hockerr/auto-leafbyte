"""Command line interface: ``python -m leafbyte <command>``.

Commands
--------
run      measure a folder of photos (automatic; low-precision ones are queued)
review   (re)render the offline human-review page for the queued samples
apply    merge the reviewer's decisions.json and write the final result table
report   print a summary of an existing run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import batch, review
from .config import Config


def _print(message: str = "") -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config) if args.config else Config()
    if args.scale_cm is not None:
        cfg.scale_cm = args.scale_cm
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.max_dim is not None:
        cfg.max_dim = args.max_dim
    if args.background is not None:
        cfg.background = args.background
    if getattr(args, "leaf_mode", None) is not None:
        cfg.leaf_mode = args.leaf_mode
    if getattr(args, "color_threshold", None) is not None:
        cfg.color_threshold = args.color_threshold
    if args.auto_accept_min is not None:
        cfg.auto_accept_min = args.auto_accept_min
    if args.review_min is not None:
        cfg.review_min = args.review_min

    result = batch.run_batch(
        args.image_dir,
        args.out,
        cfg=cfg,
        ground_truth_path=args.ground_truth,
        make_review=not args.no_review,
        progress=_print if not args.quiet else None,
        overlay_mode=getattr(args, "overlays", "review"),
        save_figures=getattr(args, "save_figures", False),
        export_pdf=getattr(args, "pdf", False),
    )
    _print("")
    _print(f"results : {Path(args.out) / batch.RESULTS_CSV}")
    _print(f"details : {Path(args.out) / batch.RESULTS_JSONL}")
    if result.get("review_page"):
        _print(f"review  : {result['review_page']}")
    accuracy = result.get("accuracy")
    if accuracy and accuracy.get("auto_compared"):
        _print(f"auto-accepted samples inside tolerance: "
               f"{accuracy['auto_within_tolerance']}/{accuracy['auto_compared']}")
        _print(f"mean |error| of auto-accepted total area: "
               f"{accuracy.get('auto_mean_abs_error_total_cm2')} cm2")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    records = batch.load_records(out_dir)
    cfg = Config.load(out_dir / "config.json") if (out_dir / "config.json").exists() else Config()
    queued = [record for record in records if record.get("status") in {"review", "failed"}]
    page = review.build(out_dir, queued, cfg)
    _print(f"{len(queued)} sample(s) queued for review")
    _print(f"open this page in a browser: {page}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    records = batch.load_records(out_dir)
    cfg = Config.load(out_dir / "config.json") if (out_dir / "config.json").exists() else Config()
    decisions_path = Path(args.decisions)
    payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions", payload)
    reviewer = args.reviewer or payload.get("reviewer") or "unknown"

    # keep the pre-review table for auditing before it is overwritten
    if (out_dir / batch.RESULTS_CSV).exists() and not (out_dir / "results.pre_review.csv").exists():
        (out_dir / "results.pre_review.csv").write_bytes((out_dir / batch.RESULTS_CSV).read_bytes())
    merged = review.apply(records, decisions, cfg, reviewer=reviewer)

    # Recompute the "vs published value" comparison, otherwise a human-corrected
    # value would keep the error of the automatic measurement it replaced.
    run_json = out_dir / batch.RUN_JSON
    run_info = json.loads(run_json.read_text(encoding="utf-8")) if run_json.exists() else {}
    ground_truth = run_info.get("ground_truth")
    if not ground_truth and run_info.get("image_dir"):
        candidate = Path(run_info["image_dir"]).parent / "ground_truth.csv"
        ground_truth = str(candidate) if candidate.exists() else None
    accuracy = None
    if ground_truth and Path(ground_truth).exists():
        accuracy = batch.attach_reference(merged, batch.load_ground_truth(ground_truth), cfg)

    batch.write_results(merged, out_dir)
    batch.write_csv(merged, out_dir / batch.FINAL_CSV, columns=batch.FINAL_COLUMNS)
    traits_path = batch.write_traits_csv(merged, out_dir)
    final_jsonl = out_dir / batch.FINAL_JSONL
    with open(final_jsonl, "w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(batch._jsonable(record), ensure_ascii=False) + "\n")

    summary = batch.summarize(merged)
    done = sum(1 for record in merged if record.get("source") and str(record["source"]).startswith("review:"))
    _print(f"applied {len(decisions)} decision(s) from {decisions_path.name} (reviewer: {reviewer})")
    _print(f"human-reviewed samples now in results: {done}")
    _print(f"final table: {out_dir / batch.FINAL_CSV}")
    _print(f"traits table refreshed: {traits_path}")
    _print(f"status: {summary['auto']} auto / {summary['done']} human-accepted / "
           f"{summary['rejected']} rejected / {summary['review']} still awaiting review / "
           f"{summary['failed']} failed")
    if accuracy and accuracy.get("compared"):
        _print(f"accuracy vs reference ({accuracy['compared']} samples): "
               f"mean |error| {accuracy.get('mean_abs_error_total_cm2')} cm2, "
               f"max {accuracy.get('max_abs_error_total_cm2')} cm2, "
               f"within tolerance {sum(1 for r in merged if (r.get('reference') or {}).get('within_tolerance'))}/"
               f"{accuracy['compared']}")
    if run_info:
        run_info["accuracy"] = accuracy
        run_info["review"] = {
            "reviewed_at": decisions_path.name,
            "reviewer": reviewer,
            "decisions": len(decisions),
            "human_accepted": summary["done"],
            "rejected": summary["rejected"],
        }
        run_json.write_text(json.dumps(batch._jsonable(run_info), indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    records = batch.load_records(out_dir)
    summary = batch.summarize(records)
    _print(f"images           : {summary['images']}")
    _print(f"auto-accepted    : {summary['auto']}")
    _print(f"needs review     : {summary['review']}")
    _print(f"failed           : {summary['failed']}")
    if summary.get("done") or summary.get("rejected"):
        _print(f"human-accepted   : {summary['done']}")
        _print(f"rejected by human: {summary['rejected']}")
    _print(f"mean confidence  : {summary['mean_confidence']}")
    if summary.get("mean_percent_consumed") is not None:
        _print(f"mean herbivory   : {summary['mean_percent_consumed']:.3f} %")

    run_json = out_dir / batch.RUN_JSON
    if run_json.exists():
        info = json.loads(run_json.read_text(encoding="utf-8"))
        accuracy = info.get("accuracy")
        if accuracy and accuracy.get("compared"):
            _print("")
            _print(f"reference samples compared : {accuracy['compared']}")
            _print(f"mean |error| total area    : {accuracy.get('mean_abs_error_total_cm2')} cm2")
            _print(f"max  |error| total area    : {accuracy.get('max_abs_error_total_cm2')} cm2")
            if accuracy.get("auto_compared"):
                _print(f"auto-accepted within tol   : "
                       f"{accuracy['auto_within_tolerance']}/{accuracy['auto_compared']}")

    if args.verbose:
        _print("")
        _print(f"{'file':<28}{'status':<10}{'conf':>6}{'total':>10}{'cons':>9}{'%':>8}  flags")
        for record in records:
            flags = "; ".join(record.get("flags") or [])
            _print(f"{record['file']:<28}{record.get('status', ''):<10}"
                   f"{float(record.get('confidence') or 0):>6.3f}"
                   f"{_fmt(record.get('total_cm2')):>10}{_fmt(record.get('consumed_cm2')):>9}"
                   f"{_fmt(record.get('percent_consumed')):>8}  {flags}")
    return 0


def _fmt(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def cmd_calibrate(args: argparse.Namespace) -> int:
    from . import calibrate

    return calibrate.main(args)


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leafbyte",
        description="Automatic LeafByte measurement with a human-review queue for low-precision samples.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure a folder of photos")
    run.add_argument("image_dir", help="folder with .png/.jpg leaf photos")
    run.add_argument("--out", required=True, help="output folder (results + review queue)")
    run.add_argument("--scale-cm", type=float, default=None,
                     help="physical length of the scale mark in cm (default 17 for the 4-dot square)")
    run.add_argument("--threshold", type=float, default=None,
                     help="force the luma threshold instead of Otsu (0-1)")
    run.add_argument("--max-dim", type=int, default=None, help="normalise longest side (default 1200)")
    run.add_argument("--background", choices=["white", "black"], default=None)
    run.add_argument("--leaf-mode", choices=["auto", "luma", "color"], default=None,
                     help="leaf segmentation: auto (default), luma (white paper) or color (excess green)")
    run.add_argument("--color-threshold", type=float, default=None,
                     help="ExG cut for --leaf-mode color (default 5.0)")
    run.add_argument("--config", default=None, help="JSON config produced by a previous run")
    run.add_argument("--ground-truth", default=None, help="CSV with expected areas for accuracy stats")
    run.add_argument("--auto-accept-min", type=float, default=None,
                     help="confidence needed to accept without review (0-1)")
    run.add_argument("--review-min", type=float, default=None,
                     help="below this confidence a sample counts as failed")
    run.add_argument("--save-figures", dest="save_figures", action="store_true", default=True,
                     help="save per-sample leaf mask + histogram/threshold figure into <out>/figures/ (default: on)")
    run.add_argument("--no-save-figures", dest="save_figures", action="store_false",
                     help="skip the per-sample threshold figures")
    run.add_argument("--pdf", action="store_true",
                     help="also export a vector PDF (off by default; SVG + PNG plate are always written)")
    run.add_argument("--overlays", choices=["review", "all"], default="review",
                     help="write overlay images for review items only (default) or for every sample")
    run.add_argument("--no-review", action="store_true", help="do not render the review page")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)

    review_cmd = sub.add_parser("review", help="(re)render the review page")
    review_cmd.add_argument("--out", required=True)
    review_cmd.set_defaults(func=cmd_review)

    apply_cmd = sub.add_parser("apply", help="merge reviewer decisions into the final table")
    apply_cmd.add_argument("--out", required=True)
    apply_cmd.add_argument("--decisions", required=True, help="decisions.json exported by the review page")
    apply_cmd.add_argument("--reviewer", default=None)
    apply_cmd.set_defaults(func=cmd_apply)

    report = sub.add_parser("report", help="summarise a run")
    report.add_argument("--out", required=True)
    report.add_argument("-v", "--verbose", action="store_true")
    report.set_defaults(func=cmd_report)

    calibrate_cmd = sub.add_parser("calibrate", help="tune the confidence model against known areas")
    calibrate_cmd.add_argument("--out", required=True, help="run folder with results.jsonl")
    calibrate_cmd.add_argument("--ground-truth", required=True)
    calibrate_cmd.add_argument("--tol-rel", type=float, default=0.015,
                               help="relative area tolerance used to label an auto-accept as correct")
    calibrate_cmd.add_argument("--tol-abs", type=float, default=0.05,
                               help="absolute area tolerance in cm^2")
    calibrate_cmd.add_argument("--write-config", action="store_true",
                               help="persist the calibrated thresholds into <out>/config.json")
    calibrate_cmd.set_defaults(func=cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
