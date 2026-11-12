"""Regression tests for issues raised in the code review of commit 8d3dbfc."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from leafbyte import measure, review  # noqa: E402
from leafbyte.config import Config  # noqa: E402

SCRATCH = ROOT / "out" / "tests"
SCRATCH.mkdir(parents=True, exist_ok=True)


class LegacyConfigTest(unittest.TestCase):
    def test_missing_leaf_mode_keeps_luma(self) -> None:
        """An old config.json has no leaf_mode: replaying it must not change behaviour."""
        cfg = Config.from_dict({"scale_cm": 17.0, "threshold": 0.5})
        self.assertEqual("luma", cfg.leaf_mode)
        self.assertFalse(cfg.hole_validate_background)

    def test_explicit_leaf_mode_is_respected(self) -> None:
        cfg = Config.from_dict({"leaf_mode": "color"})
        self.assertEqual("color", cfg.leaf_mode)
        self.assertTrue(cfg.hole_validate_background)


class PublisherDefaultTest(unittest.TestCase):
    def test_repository_is_private_by_default(self) -> None:
        import publish_github

        parser = publish_github.build_parser()
        self.assertTrue(parser.parse_args(["--token", "x"]).private)
        self.assertFalse(parser.parse_args(["--token", "x", "--public"]).private)


class DryMassMatchingTest(unittest.TestCase):
    def _lookup(self):
        import attach_dry_mass

        return attach_dry_mass._lookup

    def test_short_code_does_not_match_longer_sample(self) -> None:
        lookup = self._lookup()
        masses = {"140盐肤木-1": 100.0}
        # "-10" must NOT inherit the mass of "-1"
        self.assertIsNone(lookup(masses, "Herbivory Data 140盐肤木-10 (2026.07.08).png"))
        self.assertIsNone(lookup(masses, "Herbivory Data 140盐肤木-11 (2026.07.08).png"))

    def test_exact_and_normalised_forms_match(self) -> None:
        lookup = self._lookup()
        masses = {"140盐肤木-1": 100.0}
        self.assertEqual(100.0, lookup(masses, "Herbivory Data 140盐肤木-1 (2026.07.08).png"))
        self.assertEqual(100.0, lookup(masses, "140盐肤木-1"))
        self.assertIsNone(lookup(masses, "140盐肤木-2"))


class ReviewApplyColourTest(unittest.TestCase):
    def test_accept_writes_colour_threshold_back(self) -> None:
        record = {
            "file": "x.png", "status": "review", "confidence": 0.55,
            "threshold": 0.51,
            "segmentation": {"mode": "color", "color_threshold": 16.0},
            "candidates": [
                {"candidate_id": "t2", "threshold": 0.53, "color_threshold": 21.5,
                 "total_cm2": 20.0, "consumed_cm2": 1.0, "percent_consumed": 5.0},
            ],
            "traits": {"leaf_area_cm2": 19.0, "intact_area_cm2": 20.0, "perimeter_cm": 20.0},
        }
        merged = review.apply([record], {"x.png": {"action": "accept", "candidate_id": "t2"}},
                              Config(), reviewer="tester")
        out = merged[0]
        self.assertEqual(21.5, out["segmentation"]["color_threshold"])
        self.assertEqual(20.0, out["total_cm2"])
        # area traits follow the accepted numbers
        self.assertEqual(19.0, out["traits"]["leaf_area_cm2"])
        self.assertEqual(20.0, out["traits"]["intact_area_cm2"])


class SummaryAreaColumnTest(unittest.TestCase):
    def test_leaf_area_is_tissue_not_intact(self) -> None:
        import make_summary_table as summary

        record = {
            "file": "sample.png", "status": "auto", "confidence": 0.9,
            "total_cm2": 22.351, "consumed_cm2": 2.336, "percent_consumed": 10.4,
            "measure": {"mode": "luma"}, "scale": {"scale_px_length": 750},
            "flags": [], "reason": "",
            "traits": {"leaf_area_cm2": 20.015, "intact_area_cm2": 22.351},
        }
        row = summary.sample_row(record, 1, "test")
        self.assertAlmostEqual(20.015, row["leaf_area_cm2"])
        self.assertAlmostEqual(22.351, row["intact_cm2"])
        labels = dict(summary.SAMPLE_COLUMNS)
        self.assertEqual("叶面积_cm2", labels["leaf_area_cm2"])
        self.assertEqual("完整叶面积_cm2", labels["intact_cm2"])


if __name__ == "__main__":
    unittest.main()
