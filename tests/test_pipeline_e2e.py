"""End-to-end regression tests for the whole pipeline (Lead-owned).

These are the tests that guard the promise made to the user:

* a clean paper photo is measured within a fraction of a percent and accepted
  automatically;
* a photo the automation cannot trust is routed to the review queue instead of
  being silently answered;
* the documented artifacts (results.csv, results.jsonl, run.json, review page,
  final_results.csv) really come out of a batch run.

Run with::

    PYTHONPATH=src python -m unittest tests.test_pipeline_e2e -v
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from leafbyte import batch, cli, imageops, quality, review  # noqa: E402
from leafbyte.config import Config  # noqa: E402

PAPER = ROOT / "data" / "paper" / "images"

if not PAPER.exists():  # the upstream photos are fetched, not committed
    raise unittest.SkipTest("paper images missing - run tools/fetch_paper_data.py")
#: Scratch space.  Deliberately inside the workspace: the platform sandbox does
#: not always allow removing directories under the OS temp folder, which would
#: turn a passing test into a tearDown error.
SCRATCH = ROOT / "out" / "tests"
SCRATCH.mkdir(parents=True, exist_ok=True)


def scratch_dir(tag: str) -> Path:
    path = SCRATCH / f"{tag}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path
TRUTH = {  # from data/paper/ground_truth.csv (upstream published values, cm^2)
    "CO1.png": 23.951,
    "GB4.png": 14.953,
    "AI2.png": 22.565,   # upstream LeafByte fails on this one; we still measure it
}


class SingleImageTest(unittest.TestCase):
    """analyse_image + quality.evaluate, no filesystem side effects."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = Config()
        cls.cfg.scale_cm = 17.0

    def _evaluate(self, name: str) -> dict:
        record = batch.analyze_image(PAPER / name, self.cfg)
        return quality.evaluate(record, self.cfg)

    def test_clean_paper_photos_are_accurate_and_auto(self) -> None:
        for name, expected in TRUTH.items():
            with self.subTest(image=name):
                record = self._evaluate(name)
                self.assertTrue(record["ok"], record.get("error"))
                self.assertTrue(record["scale"]["found"], "scale marks must be found")
                self.assertIsNotNone(record["total_cm2"])
                self.assertLessEqual(abs(record["total_cm2"] - expected),
                                     max(0.35, 0.02 * expected),
                                     f"{name}: measured {record['total_cm2']} vs {expected}")
                self.assertEqual("auto", record["status"],
                                 f"{name} should be auto-accepted, flags={record['flags']}")
                self.assertGreaterEqual(record["confidence"], self.cfg.auto_accept_min)

    def test_unusable_photo_is_routed_to_review(self) -> None:
        """A photo with no usable scale must never be answered confidently."""
        source = imageops.load_image(PAPER / "CO1.png")
        blurred = np.asarray(
            Image.fromarray(source).filter(ImageFilter.GaussianBlur(radius=16))
        ).copy()
        path = scratch_dir("blurred") / "blurred.png"
        Image.fromarray(blurred).save(path)
        record = quality.evaluate(batch.analyze_image(path, self.cfg), self.cfg)
        self.assertNotEqual("auto", record["status"])
        self.assertIn(record["status"], {"review", "failed"})
        self.assertIsNone(record["total_cm2"])
        self.assertLess(record["confidence"], self.cfg.auto_accept_min)
        self.assertTrue(any("scale" in flag for flag in record["flags"]), record["flags"])

    def test_confirming_threshold_never_raises(self) -> None:
        """A forced bad threshold degrades gracefully instead of crashing."""
        cfg = Config()
        cfg.threshold = 0.01  # everything is "foreground"
        record = quality.evaluate(batch.analyze_image(PAPER / "CO1.png", cfg), cfg)
        self.assertIn(record["status"], {"review", "failed"})


class BatchRunTest(unittest.TestCase):
    """The full two-step workflow on a tiny folder."""

    def test_run_then_apply(self) -> None:
        work = scratch_dir("batch")
        image_dir = work / "images"
        image_dir.mkdir()
        out_dir = work / "out"
        for name in ("CO1.png", "GB4.png"):
            (image_dir / name).write_bytes((PAPER / name).read_bytes())

        if True:
            result = batch.run_batch(image_dir, out_dir, cfg=Config(), progress=None)
            self.assertEqual(2, result["summary"]["images"])

            # documented artifacts
            for relative in (batch.RESULTS_CSV, batch.RESULTS_JSONL, batch.RUN_JSON,
                             "config.json"):
                self.assertTrue((out_dir / relative).exists(), relative)
            records = batch.load_records(out_dir)
            self.assertEqual(2, len(records))
            self.assertTrue(all(record["status"] in {"auto", "review", "failed"} for record in records))
            run_info = json.loads((out_dir / batch.RUN_JSON).read_text(encoding="utf-8"))
            self.assertIn("summary", run_info)
            self.assertEqual(17.0, run_info["config"]["scale_cm"])

            # a review page is always rendered (even when nothing is queued)
            page = out_dir / "review" / "index.html"
            self.assertTrue(page.exists())
            html = page.read_text(encoding="utf-8")
            self.assertNotIn("https://", html.replace("https://zoegp.science", ""))
            self.assertIn("decisions.json", html)

            # apply a hand-written decision and confirm the final table
            decisions = {"CO1.png": {"action": "override", "total_cm2": 24.0,
                                     "consumed_cm2": 1.0, "note": "manual check"}}
            merged = review.apply(records, decisions, Config())
            batch.write_results(merged, out_dir)
            batch.write_csv(merged, out_dir / batch.FINAL_CSV,
                            columns=batch.CSV_COLUMNS + ["source", "reviewed_by", "review_note"])
            final = (out_dir / batch.FINAL_CSV).read_text(encoding="utf-8-sig")
            self.assertIn("CO1.png", final)
            self.assertIn("24.0", final)
            overridden = next(record for record in merged if record["file"] == "CO1.png")
            self.assertAlmostEqual(24.0, overridden["total_cm2"])
            self.assertTrue(str(overridden.get("source", "")).startswith("review"))

    def test_cli_reports(self) -> None:
        work = scratch_dir("cli")
        image_dir = work / "images"
        image_dir.mkdir()
        (image_dir / "GB4.png").write_bytes((PAPER / "GB4.png").read_bytes())
        out_dir = work / "out"
        self.assertEqual(0, cli.main(["run", str(image_dir), "--out", str(out_dir),
                                      "--scale-cm", "17", "--quiet"]))
        self.assertEqual(0, cli.main(["report", "--out", str(out_dir)]))


if __name__ == "__main__":
    unittest.main()
