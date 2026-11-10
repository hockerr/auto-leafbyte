"""Unit tests for :mod:`leafbyte.measure` (upstream area semantics).

Two kinds of tests:

* ``compute_results`` is exercised with **hand-built** ``ComponentsInfo`` label
  maps, built exactly the way ``ccl.label_components`` describes: positive =
  occupied, negative = empty, ``-1`` = outside-of-image background (empties
  that touch the border are merged into it), ``empty_to_occupied`` keyed and
  valued by canonical labels.
* ``analyze`` is exercised end-to-end on small synthetic RGB images through the
  real ``imageops``/``ccl`` modules (they are Lead-owned; this is the
  integration check).
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from leafbyte.config import Config
from leafbyte.measure import (
    MeasureAnalysis,
    analyze,
    compute_results,
    measure_dict,
    otsu_separation,
    summarize,
    to_cm2,
)
from leafbyte.types import BACKGROUND_LABEL, ComponentsInfo, MeasureResult


def make_ci(label_map, empty_to_occupied=None) -> ComponentsInfo:
    """Build a ``ComponentsInfo`` the way ``ccl.label_components`` would.

    ``member_point`` is the top-most (then left-most) pixel of the component;
    ``empty_to_occupied`` must be supplied by the caller (adjacency knowledge
    is not recoverable from the label map alone without re-running the CCL).
    """
    array = np.asarray(label_map, dtype=np.int32)
    height, width = array.shape
    label_to_size: dict[int, int] = {}
    label_to_member_point: dict[int, tuple[int, int]] = {}
    for raw in np.unique(array):
        label = int(raw)
        mask = array == label
        label_to_size[label] = int(mask.sum())
        rows, cols = np.nonzero(mask)
        order = np.lexsort((cols, rows))  # min y, then min x
        label_to_member_point[label] = (int(cols[order[0]]),
                                        int(rows[order[0]]))
    return ComponentsInfo(
        width=width,
        height=height,
        label_map=array,
        label_to_size=label_to_size,
        label_to_member_point=label_to_member_point,
        empty_to_occupied=dict(empty_to_occupied or {}),
    )


def _solid_leaf_map(size: int = 12) -> np.ndarray:
    label_map = np.full((size, size), BACKGROUND_LABEL, dtype=np.int32)
    label_map[1:-1, 1:-1] = 1
    return label_map


def _two_holes_map() -> tuple[np.ndarray, dict]:
    label_map = np.full((16, 16), BACKGROUND_LABEL, dtype=np.int32)
    label_map[1:15, 1:15] = 1         # leaf: 14 * 14 = 196 px
    label_map[3:7, 3:7] = -2          # hole: 4 * 4 = 16 px
    label_map[9:12, 10:13] = -3       # hole: 3 * 3 = 9 px
    return label_map, {-2: {1}, -3: {1}}


# ---------------------------------------------------------------------------
# compute_results: upstream semantics on hand-built component maps
# ---------------------------------------------------------------------------
class ComputeResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()

    def test_solid_leaf_no_consumed(self) -> None:
        label_map = _solid_leaf_map()
        result = compute_results(make_ci(label_map), self.cfg)
        self.assertEqual(result.leaf_px, 100)
        self.assertEqual(result.leaf_label, 1)
        self.assertEqual(result.consumed_px, 0)
        self.assertEqual(result.total_px, 100)
        self.assertEqual(result.holes, [])
        self.assertEqual(result.hole_count, 0)
        self.assertFalse(result.leaf_touches_border)
        self.assertEqual(result.percent_consumed, 0.0)
        self.assertNotIn("leaf_touches_border", result.flags)

    def test_two_enclosed_holes_are_consumed(self) -> None:
        label_map, adjacency = _two_holes_map()
        result = compute_results(make_ci(label_map, adjacency), self.cfg)
        # leaf_px is the *occupied* pixel count (upstream standardPart):
        # 14*14 leaf minus the 16+9 hole pixels.
        self.assertEqual(result.leaf_px, 196 - 25)
        self.assertEqual(result.consumed_px, 16 + 9)
        self.assertEqual(result.total_px, 196)
        self.assertEqual([h["label"] for h in result.holes], [-2, -3])
        self.assertEqual([h["area_px"] for h in result.holes], [16, 9])
        # centroid is (x, y); hole -2 occupies rows 3..6 / cols 3..6.
        self.assertEqual(result.holes[0]["centroid"], [4.5, 4.5])
        self.assertEqual(result.hole_count, 2)
        self.assertAlmostEqual(result.percent_consumed, 100.0 * 25 / 196)

    def test_hole_touching_border_is_background_and_not_consumed(self) -> None:
        # ccl merges every empty region connected to the image border into
        # BACKGROUND_LABEL (-1).  This fixture has the border background (-1,
        # adjacent to the leaf) plus one truly enclosed hole (-2): only the
        # hole may be consumed, exactly like upstream's `!= backgroundLabel`.
        label_map = np.full((16, 16), BACKGROUND_LABEL, dtype=np.int32)
        label_map[1:15, 1:15] = 1
        label_map[9:12, 10:13] = -2                  # true enclosed hole
        ci = make_ci(label_map, {BACKGROUND_LABEL: {1}, -2: {1}})
        result = compute_results(ci, self.cfg)
        self.assertEqual(result.leaf_px, 196 - 9)
        self.assertEqual(result.consumed_px, 9)
        self.assertEqual(result.total_px, 196)
        self.assertEqual([h["label"] for h in result.holes], [-2])
        self.assertIn(BACKGROUND_LABEL, ci.label_to_size)
        self.assertNotIn(BACKGROUND_LABEL, ci.empty_labels())
        self.assertNotIn(BACKGROUND_LABEL, [h["label"] for h in result.holes])

    def test_split_leaf_uses_largest_component_only(self) -> None:
        label_map = np.full((18, 18), BACKGROUND_LABEL, dtype=np.int32)
        label_map[1:9, 1:9] = 1       # 8 * 8 = 64 px, largest -> leaf
        label_map[11:16, 1:6] = 2     # 5 * 5 = 25 px, second leaf/chunk
        label_map[3:6, 3:6] = -3      # hole inside component 1 -> consumed
        label_map[12:15, 2:5] = -2    # hole inside component 2 -> NOT consumed
        ci = make_ci(label_map, {-3: {1}, -2: {2}})
        result = compute_results(ci, self.cfg)
        # occupied sizes: component 1 = 8*8-9 = 55, component 2 = 5*5-9 = 16
        self.assertEqual(result.leaf_label, 1)
        self.assertEqual(result.leaf_px, 55)
        self.assertEqual(result.consumed_px, 9)         # only the -3 hole
        self.assertEqual(result.total_px, 64)
        self.assertEqual([h["label"] for h in result.holes], [-3])
        self.assertAlmostEqual(result.competing_ratio, 16 / 55)
        self.assertIn("competing_leaf", result.flags)

    def test_excluded_mask_drops_consumed_hole(self) -> None:
        label_map, adjacency = _two_holes_map()
        ci = make_ci(label_map, adjacency)
        excluded = np.zeros(label_map.shape, dtype=bool)
        excluded[3:7, 3:7] = True  # cover hole -2 entirely
        result = compute_results(ci, self.cfg, excluded_mask=excluded)
        self.assertEqual(result.excluded_px, 16)
        self.assertEqual(result.consumed_px, 9)
        self.assertEqual([h["label"] for h in result.holes], [-3])
        self.assertNotIn("excluded_leaf_region", result.flags)

    def test_excluded_mask_over_leaf_never_shrinks_the_leaf(self) -> None:
        label_map, adjacency = _two_holes_map()
        ci = make_ci(label_map, adjacency)
        excluded = np.zeros(label_map.shape, dtype=bool)
        excluded[1:3, 1:3] = True  # leaf pixels only
        result = compute_results(ci, self.cfg, excluded_mask=excluded)
        self.assertEqual(result.leaf_px, 196 - 25)  # upstream never shrinks it
        self.assertEqual(result.consumed_px, 25)
        self.assertEqual(result.total_px, 196)
        self.assertIn("excluded_leaf_region", result.flags)

    def test_excluded_mask_shape_mismatch_is_tolerated(self) -> None:
        label_map, adjacency = _two_holes_map()
        ci = make_ci(label_map, adjacency)
        result = compute_results(ci, self.cfg,
                                 excluded_mask=np.zeros((4, 4), dtype=bool))
        self.assertEqual(result.total_px, 196)
        self.assertIn("excluded_mask_shape_mismatch", result.flags)

    def test_hole_min_px_skips_small_holes(self) -> None:
        label_map, adjacency = _two_holes_map()
        ci = make_ci(label_map, adjacency)
        cfg = Config(hole_min_px=10)
        result = compute_results(ci, cfg)
        self.assertEqual(result.consumed_px, 16)
        self.assertEqual(result.total_px, 187)

    def test_tiny_holes_are_counted_but_flagged(self) -> None:
        label_map = np.full((16, 16), BACKGROUND_LABEL, dtype=np.int32)
        label_map[1:15, 1:15] = 1
        label_map[3:5, 3:5] = -2  # 4 px < cfg.tiny_hole_px (25)
        result = compute_results(make_ci(label_map, {-2: {1}}), self.cfg)
        self.assertEqual(result.consumed_px, 4)  # still counted, upstream-like
        self.assertEqual(result.total_px, 196)
        self.assertEqual(result.hole_count, 1)
        self.assertEqual(result.tiny_holes, 1)
        self.assertIn("tiny_holes", result.flags)

    def test_degenerate_maps_do_not_crash(self) -> None:
        all_background = compute_results(
            make_ci(np.full((4, 4), BACKGROUND_LABEL, dtype=np.int32)),
            self.cfg)
        self.assertEqual(all_background.total_px, 0)
        self.assertIsNone(all_background.leaf_label)
        self.assertIn("no_leaf", all_background.flags)

        all_occupied = compute_results(
            make_ci(np.ones((4, 4), dtype=np.int32)), self.cfg)
        self.assertEqual(all_occupied.total_px, 16)
        self.assertTrue(all_occupied.leaf_touches_border)
        self.assertIn("no_empty_components", all_occupied.flags)

        self.assertIn("no_components", compute_results(None, self.cfg).flags)
        empty = make_ci(np.zeros((0, 0), dtype=np.int32))
        self.assertIn("empty_components", compute_results(empty, self.cfg).flags)

    def test_missing_adjacency_is_not_counted(self) -> None:
        label_map, _ = _two_holes_map()
        result = compute_results(make_ci(label_map, {}), self.cfg)
        self.assertEqual(result.consumed_px, 0)
        self.assertIn("missing_adjacency", result.flags)


# ---------------------------------------------------------------------------
# analyze: synthetic images through the real imageops + ccl modules
# ---------------------------------------------------------------------------
class AnalyzeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        # 60x60 white "paper", dark 40x40 leaf, one 4x4 white hole inside it.
        self.image = np.full((60, 60, 3), 230, dtype=np.uint8)
        self.image[10:50, 10:50] = 40
        self.image[20:24, 20:24] = 230

    def test_analyze_reproduces_upstream_semantics(self) -> None:
        analysis = analyze(self.image, self.cfg)
        self.assertAlmostEqual(analysis.threshold, analysis.otsu_threshold)
        self.assertTrue(0.15 < analysis.threshold < 0.90)
        self.assertEqual(analysis.mask.shape, (60, 60))
        self.assertEqual(int(analysis.mask.sum()), 40 * 40 - 16)
        self.assertEqual(analysis.result.leaf_px, 40 * 40 - 16)
        self.assertEqual(analysis.result.consumed_px, 16)
        self.assertEqual(analysis.result.total_px, 1600)
        self.assertEqual(analysis.result.hole_count, 1)
        self.assertFalse(analysis.result.leaf_touches_border)
        self.assertAlmostEqual(analysis.result.percent_consumed, 1.0)
        self.assertGreater(analysis.otsu_separation, 0.90)
        self.assertEqual(np.asarray(analysis.histogram).shape, (256,))

    def test_analyze_forced_threshold(self) -> None:
        analysis = analyze(self.image, self.cfg, threshold=0.5)
        self.assertAlmostEqual(analysis.threshold, 0.5)
        self.assertEqual(analysis.result.total_px, 1600)

    def test_analyze_uses_cfg_threshold_when_set(self) -> None:
        cfg = Config(threshold=0.33)
        analysis = analyze(self.image, cfg)
        self.assertAlmostEqual(analysis.threshold, 0.33)

    def test_analyze_excluded_mask(self) -> None:
        excluded = np.zeros((60, 60), dtype=bool)
        excluded[20:24, 20:24] = True
        analysis = analyze(self.image, self.cfg, excluded_mask=excluded)
        self.assertEqual(analysis.result.consumed_px, 0)
        self.assertEqual(analysis.result.excluded_px, 16)
        self.assertEqual(analysis.result.total_px, 40 * 40 - 16)

    def test_analyze_leaf_touches_border(self) -> None:
        image = np.full((60, 60, 3), 230, dtype=np.uint8)
        image[:, :40] = 40  # leaf cut by the left edge of the crop
        analysis = analyze(image, self.cfg)
        self.assertTrue(analysis.result.leaf_touches_border)
        self.assertIn("leaf_touches_border", analysis.result.flags)

    def test_analyze_black_background_mode(self) -> None:
        image = np.full((60, 60, 3), 20, dtype=np.uint8)
        image[10:50, 10:50] = 200
        image[20:24, 20:24] = 20
        analysis = analyze(image, Config(background="black"))
        self.assertEqual(analysis.result.leaf_px, 40 * 40 - 16)
        self.assertEqual(analysis.result.consumed_px, 16)
        self.assertEqual(analysis.result.total_px, 1600)

    def test_analyze_accepts_grayscale_and_float_images(self) -> None:
        gray = np.full((60, 60), 230, dtype=np.uint8)
        gray[10:50, 10:50] = 40
        analysis = analyze(gray, self.cfg)
        self.assertEqual(analysis.result.total_px, 1600)

        float_image = self.image.astype(np.float64) / 255.0
        float_analysis = analyze(float_image, self.cfg)
        self.assertEqual(float_analysis.result.total_px, 1600)

    def test_analyze_output_is_json_ready(self) -> None:
        analysis = analyze(self.image, self.cfg)
        payload = measure_dict(analysis)
        text = json.dumps(payload)  # must not raise (no numpy scalars)
        self.assertIn('"total_px": 1600', text)
        self.assertEqual(payload["total_px"], 1600)
        self.assertEqual(payload["hole_count"], 1)
        self.assertIsInstance(payload["holes"][0]["area_px"], int)
        self.assertIsInstance(payload["holes"][0]["centroid"][0], float)


# ---------------------------------------------------------------------------
# to_cm2 / otsu_separation / summarize
# ---------------------------------------------------------------------------
class ConversionTests(unittest.TestCase):
    def test_to_cm2_uses_upstream_formula(self) -> None:
        cfg = Config(scale_cm=17.0)
        result = MeasureResult(total_px=1000, consumed_px=100)
        total, consumed, percent = to_cm2(result, cfg, 170.0)
        # units_per_pixel = 17 / 170 = 0.1 cm/px -> 0.01 cm^2/px
        self.assertAlmostEqual(total, 10.0)
        self.assertAlmostEqual(consumed, 1.0)
        self.assertAlmostEqual(percent, 10.0)

    def test_to_cm2_matches_result_percent(self) -> None:
        result = MeasureResult(total_px=53400, consumed_px=2210)
        total, consumed, percent = to_cm2(result, Config(), 806.0)
        self.assertAlmostEqual(percent, result.percent_consumed)
        self.assertAlmostEqual(consumed / total * 100.0, percent)

    def test_to_cm2_rejects_missing_scale(self) -> None:
        result = MeasureResult(total_px=10, consumed_px=1)
        for bad in (None, 0, -3, float("nan"), "oops"):
            with self.assertRaises(ValueError):
                to_cm2(result, Config(), bad)

    def test_zero_total_gives_zero_percent(self) -> None:
        total, consumed, percent = to_cm2(MeasureResult(), Config(), 100.0)
        self.assertEqual((total, consumed, percent), (0.0, 0.0, 0.0))


class OtsuSeparationTests(unittest.TestCase):
    def test_two_peaks_give_high_separation(self) -> None:
        hist = np.zeros(256, dtype=np.int64)
        hist[40] = 500
        hist[220] = 500
        self.assertGreater(otsu_separation(hist, 0.5), 0.99)

    def test_single_peak_gives_zero(self) -> None:
        hist = np.zeros(256, dtype=np.int64)
        hist[128] = 1000
        self.assertEqual(otsu_separation(hist, 0.5), 0.0)

    def test_degenerate_histograms(self) -> None:
        self.assertEqual(otsu_separation(np.zeros(256), 0.5), 0.0)
        self.assertEqual(otsu_separation([], 0.5), 0.0)
        # threshold 1.0 puts every bucket in the first class -> degenerate.
        hist = np.zeros(256, dtype=np.int64)
        hist[40] = 5
        hist[220] = 5
        self.assertEqual(otsu_separation(hist, 1.0), 0.0)

    def test_separation_is_bounded(self) -> None:
        hist = np.random.default_rng(7).integers(0, 50, size=256)
        value = otsu_separation(hist, 0.42)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


class SummarizeTests(unittest.TestCase):
    def test_summarize_reports_qc_evidence(self) -> None:
        label_map, adjacency = _two_holes_map()
        ci = make_ci(label_map, adjacency)
        result = compute_results(ci, Config())
        stats = summarize(ci, result, Config())
        self.assertEqual(stats["occupied_sizes"], [196 - 25])
        self.assertEqual(stats["n_holes"], 2)
        self.assertEqual(stats["hole_px"], 25)
        self.assertFalse(stats["leaf_touches_border"])
        self.assertEqual(stats["competing_ratio"], 0.0)
        json.dumps(stats)  # JSON-friendly

    def test_measure_dict_without_analysis_stats(self) -> None:
        label_map, adjacency = _two_holes_map()
        ci = make_ci(label_map, adjacency)
        result = compute_results(ci, Config())
        analysis = MeasureAnalysis(
            threshold=0.5,
            mask=label_map > 0,
            components=ci,
            result=result,
        )
        payload = measure_dict(analysis)
        self.assertEqual(payload["total_px"], 196)
        self.assertEqual(payload["leaf_label"], 1)
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
