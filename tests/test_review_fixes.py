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


class OverrideRefreshesAreasTest(unittest.TestCase):
    """Round-2 review #1/#2: a manual override must not leave stale areas."""

    def _record(self):
        return {
            "file": "x.png", "status": "review", "confidence": 0.55, "threshold": 0.51,
            "segmentation": {"mode": "color", "color_threshold": 16.0},
            "total_cm2": 20.0, "consumed_cm2": 1.0, "percent_consumed": 5.0,
            "leaf_area_cm2": 19.0, "intact_cm2": 20.0,
            "traits": {"leaf_area_cm2": 19.0, "intact_area_cm2": 20.0,
                       "perimeter_cm": 25.0, "length_cm": 11.0},
        }

    def test_override_updates_traits_and_top_level_columns(self) -> None:
        merged = review.apply([self._record()],
                              {"x.png": {"action": "override", "total_cm2": 25.0,
                                         "consumed_cm2": 2.0}},
                              Config(), reviewer="t")
        out = merged[0]
        self.assertAlmostEqual(25.0, out["intact_cm2"])
        self.assertAlmostEqual(23.0, out["leaf_area_cm2"])
        self.assertAlmostEqual(23.0, out["traits"]["leaf_area_cm2"])
        self.assertAlmostEqual(25.0, out["traits"]["intact_area_cm2"])
        # geometry traits are from the original run -> flagged as stale
        self.assertTrue(out["traits_stale"])
        self.assertAlmostEqual(25.0, out["traits"]["perimeter_cm"])

    def test_final_csv_exposes_stale_flag(self) -> None:
        from leafbyte import batch

        merged = review.apply([self._record()],
                              {"x.png": {"action": "override", "total_cm2": 25.0,
                                         "consumed_cm2": 2.0}},
                              Config(), reviewer="t")
        self.assertIn("traits_stale", batch.CSV_COLUMNS)
        self.assertIn("traits_stale", batch.FINAL_COLUMNS)


class CandidateModeTest(unittest.TestCase):
    """Round-2: a luma candidate must not inject a colour threshold."""

    def test_luma_candidate_keeps_colour_threshold(self) -> None:
        record = {
            "file": "x.png", "status": "review", "confidence": 0.55, "threshold": 0.51,
            "segmentation": {"mode": "color", "color_threshold": 16.0},
            "candidates": [{"candidate_id": "t2", "mode": "luma", "threshold": 0.53,
                            "color_threshold": 99.0, "total_cm2": 20.0,
                            "consumed_cm2": 1.0, "percent_consumed": 5.0}],
            "traits": {"leaf_area_cm2": 19.0, "intact_area_cm2": 20.0},
        }
        merged = review.apply([record], {"x.png": {"action": "accept", "candidate_id": "t2"}},
                              Config(), reviewer="t")
        self.assertEqual(16.0, merged[0]["segmentation"]["color_threshold"])

    def test_colour_candidate_updates_colour_threshold(self) -> None:
        record = {
            "file": "x.png", "status": "review", "confidence": 0.55, "threshold": 0.51,
            "segmentation": {"mode": "color", "color_threshold": 16.0},
            "candidates": [{"candidate_id": "t2", "mode": "color", "threshold": 0.53,
                            "color_threshold": 21.5, "total_cm2": 20.0,
                            "consumed_cm2": 1.0, "percent_consumed": 5.0}],
            "traits": {"leaf_area_cm2": 19.0, "intact_area_cm2": 20.0},
        }
        merged = review.apply([record], {"x.png": {"action": "accept", "candidate_id": "t2"}},
                              Config(), reviewer="t")
        self.assertEqual(21.5, merged[0]["segmentation"]["color_threshold"])


class ConfigWiringTest(unittest.TestCase):
    def test_saturation_knobs_are_real_fields(self) -> None:
        cfg = Config()
        self.assertAlmostEqual(0.22, cfg.hole_tissue_saturation)
        self.assertEqual(25, cfg.hole_saturation_min_px)
        # the dead key from the previous round is gone
        self.assertFalse(hasattr(cfg, "hole_paper_colour_distance"))
        restored = Config.from_dict(cfg.to_dict())
        self.assertAlmostEqual(cfg.hole_tissue_saturation, restored.hole_tissue_saturation)

    def test_saturation_rule_only_applies_above_the_size_floor(self) -> None:
        """Small patches are left alone (their edges tint the mean saturation)."""
        import numpy as np
        from tests.test_colour_segmentation import grey_background_photo
        from leafbyte import measure

        base = grey_background_photo()
        h, w = base.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w * 0.5, h * 0.5
        half_axis = 0.5 * w * 0.28
        leaf = ((xx - cx) / half_axis) ** 2 + ((yy - cy) / (30000 / (np.pi * half_axis))) ** 2 <= 1
        patch = (xx - (cx - half_axis * 0.35)) ** 2 + (yy - cy) ** 2 <= 4 ** 2
        img = base.copy()
        img[patch & leaf] = (150, 120, 90)          # saturated, ~50 px (survives closing)

        # default floor 25 px -> treated as necrotic tissue and filled back
        filled = measure.analyze(img, Config()).result
        self.assertEqual(0, filled.consumed_px)
        self.assertIn("pale_patches_filled", filled.flags)

        # raise the floor above the patch size -> the saturation rule is skipped
        cfg = Config()
        cfg.hole_saturation_min_px = 500
        kept = measure.analyze(img, cfg).result
        self.assertGreater(kept.consumed_px, 0)


class PerimeterTraceTest(unittest.TestCase):
    """Round-2 #3: the diagonal search start skipped a neighbour."""

    def test_reviewer_mask_matches_textbook_trace(self) -> None:
        import numpy as np
        from leafbyte import traits

        mask = np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)
        chain = traits.trace_outer_contour(mask)
        self.assertEqual([1, 7, 2, 4, 4, 6], chain)
        self.assertAlmostEqual(6.28, traits.chain_perimeter(chain), delta=0.05)

    def test_circle_perimeter_is_accurate(self) -> None:
        import math
        import numpy as np
        from leafbyte import traits

        for radius in (30, 60, 120):
            yy, xx = np.mgrid[0:2 * radius + 30, 0:2 * radius + 30]
            disc = (xx - (radius + 15)) ** 2 + (yy - (radius + 15)) ** 2 <= radius ** 2
            perimeter = traits.chain_perimeter(traits.trace_outer_contour(disc))
            expected = 2 * math.pi * radius
            self.assertLess(abs(perimeter - expected) / expected, 0.01)


class RelativeOnlyTraitsTest(unittest.TestCase):
    """Round-2: unit-free colour traits must be reported without a scale."""

    def test_colour_traits_without_scale(self) -> None:
        from tests.test_colour_segmentation import grey_background_photo
        from leafbyte import batch, measure

        image = grey_background_photo()
        analysis = measure.analyze(image, Config())
        traits = batch._traits_payload(analysis, None, Config(), image=image)
        self.assertIsNotNone(traits.get("gcc"))
        self.assertIsNotNone(traits.get("exg"))
        self.assertIsNone(traits.get("intact_area_cm2"))   # needs a scale


if __name__ == "__main__":
    unittest.main()
