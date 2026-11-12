"""End-to-end test that needs no downloaded data.

A clean clone has no paper photographs (``data/paper/images`` is gitignored), so
``tests/test_pipeline_e2e.py`` skips itself there.  This module builds a
synthetic grey-background photo and runs the whole pipeline on it, so a fresh
clone still exercises measure + scale + quality + figures + traits.
"""

from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from leafbyte import batch  # noqa: E402
from leafbyte.config import Config  # noqa: E402
from tests.test_colour_segmentation import grey_background_photo  # noqa: E402

SCRATCH = ROOT / "out" / "tests"
SCRATCH.mkdir(parents=True, exist_ok=True)


class SyntheticPipelineTest(unittest.TestCase):
    def test_full_pipeline_on_a_synthetic_photo(self) -> None:
        work = SCRATCH / f"synthetic-{uuid.uuid4().hex[:8]}"
        image_dir = work / "images"
        image_dir.mkdir(parents=True)
        out_dir = work / "out"
        photo = grey_background_photo()
        Image.fromarray(photo).save(image_dir / "synthetic-1.png")

        result = batch.run_batch(image_dir, out_dir, cfg=Config(), progress=None, save_figures=True)
        records = batch.load_records(out_dir)
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertTrue(record["ok"], record.get("error"))
        # scale marks found by the dark-threshold fallback, area computed
        self.assertTrue(record["scale"]["found"])
        self.assertIsNotNone(record["total_cm2"])
        self.assertEqual("color", (record["measure"] or {}).get("mode"))
        # traits + figures + tables
        self.assertIsNotNone((record.get("traits") or {}).get("perimeter_cm"))
        self.assertTrue((out_dir / "results.csv").exists())
        self.assertTrue((out_dir / "leaf_traits.csv").exists())
        self.assertTrue((out_dir / "run.json").exists())
        self.assertTrue(any((out_dir / "figures").glob("*_threshold.png")))
        manifests = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(1, manifests["summary"]["images"])

    def test_results_csv_has_tissue_and_intact_area(self) -> None:
        import csv

        work = SCRATCH / f"synthetic2-{uuid.uuid4().hex[:8]}"
        image_dir = work / "images"
        image_dir.mkdir(parents=True)
        Image.fromarray(grey_background_photo(hole_radius=20)).save(image_dir / "synthetic-hole.png")
        out_dir = work / "out"
        batch.run_batch(image_dir, out_dir, cfg=Config(), progress=None, save_figures=False)
        with open(out_dir / "results.csv", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        row = rows[0]
        self.assertIn("leaf_area_cm2", row)
        self.assertIn("intact_cm2", row)
        # remaining tissue < intact area when a hole was eaten
        self.assertLess(float(row["leaf_area_cm2"]), float(row["intact_cm2"]))


if __name__ == "__main__":
    unittest.main()
