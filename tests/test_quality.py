"""Unit tests for :mod:`leafbyte.quality` (confidence + review routing).

The records are hand-built from the documented schema (``docs/ALGORITHM.md``
§7) so the tests are deterministic and cover partial / failed records too.
One end-to-end test feeds the output of the real :func:`leafbyte.measure.analyze`
on a synthetic photo through the quality model.
"""

from __future__ import annotations

import copy
import unittest

import numpy as np

from leafbyte import imageops, quality
from leafbyte.config import Config
from leafbyte.measure import analyze, measure_dict, to_cm2


def make_good_record(**overrides) -> dict:
    """A record that looks like a clean, absolute-area measurement."""
    record = {
        "file": "CO1.png",
        "path": "data/paper/images/CO1.png",
        "ok": True,
        "stage": "done",
        "error": None,
        "image_size": [900, 1200],
        "normalized_size": [900, 1200],
        "threshold": 0.53,
        "otsu_threshold": 0.55,
        "threshold_metrics": {
            "separation": 0.93,
            "bimodality": 0.80,
            "near_threshold_fraction": 0.01,
        },
        "histogram": [0] * 256,
        "otsu_separation": 0.93,
        "scale": {
            "found": True,
            "method": "quad",
            "quality": 0.92,
            "points": [],
            "scale_px_length": 806.0,
            "flags": [],
            "details": {},
        },
        "measure": {
            "total_px": 53400,
            "consumed_px": 2210,
            "leaf_px": 51190,
            "leaf_label": 1,
            "holes": [{"label": -3, "area_px": 2210,
                       "centroid": [400.0, 400.0]}],
            "excluded_px": 0,
            "leaf_touches_border": False,
            "competing_ratio": 0.0,
            "hole_count": 1,
            "tiny_holes": 0,
            "tiny_hole_px": 0,
            "flags": [],
        },
        "stability": {"deltas": [-0.02, 0.02],
                      "total_cm2": [23.9, 24.0], "rel_range": 0.003},
        "total_cm2": 23.951,
        "consumed_cm2": 0.992,
        "percent_consumed": 4.141,
        "relative_only": False,
        "candidates": [],
        "flags": [],
    }
    record.update(overrides)
    return record


METRIC_NAMES = (
    "scale_quality",
    "threshold_quality",
    "stability",
    "leaf_border_touch",
    "competing_leaf",
    "hole_noise",
    "percent_extremes",
    "leaf_fraction",
)


class GoodRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.record = make_good_record()

    def test_good_record_is_auto_accepted(self) -> None:
        out = quality.evaluate(self.record, self.cfg)
        self.assertIs(out, self.record)  # mutates in place
        self.assertEqual(out["status"], "auto")
        self.assertGreaterEqual(out["confidence"], self.cfg.auto_accept_min)
        self.assertLessEqual(out["confidence"], 1.0)
        self.assertTrue(out["reason"].startswith("auto-accepted"))
        for name in METRIC_NAMES:
            self.assertIn(name, out["metrics"])
            self.assertGreaterEqual(out["metrics"][name], 0.0)
            self.assertLessEqual(out["metrics"][name], 1.0)
        self.assertEqual(out["flags"], [])
        self.assertIn("quality_details", out)

    def test_metrics_capture_real_evidence(self) -> None:
        quality.evaluate(self.record, self.cfg)
        metrics = self.record["metrics"]
        self.assertGreater(metrics["scale_quality"], 0.85)
        self.assertGreater(metrics["threshold_quality"], 0.80)
        self.assertGreater(metrics["stability"], 0.95)
        self.assertEqual(metrics["leaf_border_touch"], 1.0)
        self.assertEqual(metrics["hole_noise"], 1.0)
        self.assertEqual(metrics["percent_extremes"], 1.0)

    def test_weights_override_changes_confidence(self) -> None:
        base = quality.evaluate(copy.deepcopy(make_good_record()),
                                self.cfg)["confidence"]
        heavier = quality.evaluate(
            copy.deepcopy(make_good_record()), self.cfg,
            weights={"scale_quality": 1.0})["confidence"]
        self.assertNotAlmostEqual(base, heavier, places=6)
        # A partial override merges over WEIGHTS instead of replacing it.
        merged = quality.combine(dict.fromkeys(METRIC_NAMES, 1.0),
                                 weights={"stability": 0.5})
        self.assertAlmostEqual(merged, 1.0)

    def test_combine_ignores_unknown_and_zero_weights(self) -> None:
        metrics = dict.fromkeys(METRIC_NAMES, 0.5)
        self.assertAlmostEqual(quality.combine(metrics), 0.5)
        self.assertAlmostEqual(
            quality.combine(metrics, weights={"not_a_metric": 99.0,
                                              "stability": 0.0}), 0.5)

    def test_set_params_roundtrip(self) -> None:
        original = quality.get_params()
        try:
            quality.set_params(stability_decay=2.0)
            self.assertEqual(quality.get_params()["stability_decay"], 2.0)
        finally:
            quality.set_params(**original)
        self.assertEqual(quality.DEFAULTS["weights"], quality.WEIGHTS)
        with self.assertRaises(KeyError):
            quality.set_params(not_a_knob=1)


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()

    def _assert_not_auto(self, record: dict) -> dict:
        out = quality.evaluate(record, self.cfg)
        self.assertNotEqual(out["status"], "auto")
        self.assertTrue(out["reason"])
        self.assertLessEqual(out["confidence"], quality.HARD_BLOCK_CONFIDENCE)
        return out

    def test_scale_not_found_is_never_auto(self) -> None:
        record = make_good_record(
            scale={"found": False, "method": "none", "quality": 0.0,
                   "points": [], "scale_px_length": None,
                   "flags": ["only 2 scale mark candidates"], "details": {}},
            total_cm2=None, consumed_cm2=None,
            stability={},
        )
        out = self._assert_not_auto(record)
        self.assertEqual(out["status"], "review")
        self.assertIn("scale_not_found", out["flags"])

    def test_relative_only_record_goes_to_review(self) -> None:
        record = make_good_record(
            relative_only=True,
            scale={"found": False, "method": "none", "quality": 0.0,
                   "points": [], "scale_px_length": None, "flags": [],
                   "details": {}},
            total_cm2=None,
            consumed_cm2=None,
            percent_consumed=4.141,
        )
        out = self._assert_not_auto(record)
        self.assertEqual(out["status"], "review")
        self.assertIn("relative_only", out["flags"])

    def test_unstable_area_is_never_auto(self) -> None:
        record = make_good_record(
            stability={"deltas": [-0.02, 0.02], "total_cm2": [20.0, 25.0],
                       "rel_range": 0.25})
        out = self._assert_not_auto(record)
        self.assertEqual(out["status"], "review")
        self.assertIn("area_unstable", out["flags"])
        self.assertLess(out["metrics"]["stability"], 0.20)

    def test_leaf_touching_border_is_never_auto(self) -> None:
        record = make_good_record(
            measure={**make_good_record()["measure"],
                     "leaf_touches_border": True,
                     "flags": ["leaf_touches_border"]})
        out = self._assert_not_auto(record)
        self.assertEqual(out["metrics"]["leaf_border_touch"], 0.0)
        self.assertIn("leaf_touches_border", out["flags"])

    def test_competing_component_is_never_auto(self) -> None:
        record = make_good_record(
            measure={**make_good_record()["measure"], "competing_ratio": 0.60})
        out = self._assert_not_auto(record)
        self.assertIn("competing_leaf", out["flags"])
        self.assertLess(out["metrics"]["competing_leaf"], 0.2)

    def test_tiny_hole_noise_is_never_auto(self) -> None:
        # All consumed area is sub-threshold speckle: the reported consumed /
        # percent-consumed number is not trustworthy -> human review.
        holes = [{"label": -2 - i, "area_px": 2, "centroid": [1.0, 1.0]}
                 for i in range(20)]
        record = make_good_record(
            measure={**make_good_record()["measure"],
                     "holes": holes, "hole_count": 20, "tiny_holes": 20,
                     "consumed_px": 40, "total_px": 51190 + 40})
        out = self._assert_not_auto(record)
        self.assertIn("tiny_hole_noise", out["flags"])
        self.assertLess(out["metrics"]["hole_noise"], 0.2)

    def test_real_holes_dominating_speckle_do_not_block_auto(self) -> None:
        # One normal hole (2000 px) plus two 4 px speckles: the consumed
        # number is dominated by real area, so it stays trustworthy.
        holes = [{"label": -2, "area_px": 2000, "centroid": [100.0, 100.0]},
                 {"label": -3, "area_px": 4, "centroid": [10.0, 10.0]},
                 {"label": -4, "area_px": 4, "centroid": [12.0, 12.0]}]
        record = make_good_record(
            measure={**make_good_record()["measure"],
                     "holes": holes, "hole_count": 3, "tiny_holes": 2,
                     "consumed_px": 2008, "total_px": 51190 + 2008})
        out = quality.evaluate(record, self.cfg)
        self.assertEqual(out["status"], "auto")
        self.assertGreater(out["metrics"]["hole_noise"], 0.5)

    def test_implausible_percent_consumed_is_never_auto(self) -> None:
        record = make_good_record(percent_consumed=97.0)
        out = self._assert_not_auto(record)
        self.assertIn("percent_implausible", out["flags"])
        self.assertLess(out["metrics"]["percent_extremes"], 0.5)

    def test_weak_threshold_separation_is_never_auto(self) -> None:
        record = make_good_record(threshold_metrics={
            "separation": 0.10, "bimodality": 0.10,
            "near_threshold_fraction": 0.30})
        out = self._assert_not_auto(record)
        self.assertIn("weak_threshold_separation", out["flags"])

    def test_missing_evidence_lowers_confidence_but_does_not_crash(self) -> None:
        full = quality.evaluate(copy.deepcopy(make_good_record()),
                                self.cfg)["confidence"]
        record = make_good_record(
            threshold_metrics={}, histogram=[], otsu_separation=None,
            stability={},
            measure={k: v for k, v in make_good_record()["measure"].items()
                     if k not in ("leaf_touches_border", "competing_ratio")})
        out = quality.evaluate(record, self.cfg)
        self.assertLess(out["confidence"], full)
        self.assertNotEqual(out["status"], "auto")
        self.assertEqual(set(out["metrics"]), set(METRIC_NAMES))

    def test_partial_record_fails_without_raising(self) -> None:
        out = quality.evaluate({"file": "broken.png"}, self.cfg)
        self.assertEqual(out["status"], "failed")
        self.assertLess(out["confidence"], self.cfg.review_min)
        self.assertIn("no_measurement", out["flags"])
        self.assertIn("failed", out["reason"])

    def test_failed_analysis_is_failed(self) -> None:
        out = quality.evaluate(make_good_record(ok=False), self.cfg)
        self.assertEqual(out["status"], "failed")
        self.assertIn("analysis_failed", out["flags"])

    def test_error_field_is_failed(self) -> None:
        out = quality.evaluate(make_good_record(error="RuntimeError: boom"),
                               self.cfg)
        self.assertEqual(out["status"], "failed")
        self.assertIn("analysis_error", out["flags"])

    def test_no_leaf_is_failed(self) -> None:
        record = make_good_record(
            measure={**make_good_record()["measure"],
                     "total_px": 0, "leaf_px": 0, "consumed_px": 0})
        out = quality.evaluate(record, self.cfg)
        self.assertEqual(out["status"], "failed")
        self.assertIn("no_leaf", out["flags"])

    def test_existing_flags_are_preserved(self) -> None:
        out = quality.evaluate(make_good_record(flags=["scale: relaxed quad"]),
                               self.cfg)
        self.assertIn("scale: relaxed quad", out["flags"])

    def test_confidence_is_returned_even_for_failures(self) -> None:
        out = quality.evaluate({}, Config())
        self.assertIsInstance(out["confidence"], float)
        self.assertGreaterEqual(out["confidence"], 0.0)
        self.assertLessEqual(out["confidence"], 1.0)

    def test_tunables_are_exposed(self) -> None:
        self.assertIn("scale_quality", quality.WEIGHTS)
        self.assertAlmostEqual(sum(quality.WEIGHTS.values()), 1.0)
        self.assertIn("stability_decay", quality.DEFAULTS)


class EndToEndTests(unittest.TestCase):
    """Real measure.analyze output -> quality.evaluate."""

    def setUp(self) -> None:
        # Deliberately pin the routing cuts: the shipped default
        # ``auto_accept_min`` is calibrated by the Lead on the paper data, so
        # these tests must not depend on its current value.
        self.cfg = Config()
        self.cfg.auto_accept_min = 0.75
        self.cfg.review_min = 0.45
        # 60x60 white background, dark 20x20 leaf: 400 occupied pixels.
        image = np.full((60, 60, 3), 230, dtype=np.uint8)
        image[20:40, 20:40] = 40
        self.analysis = analyze(image, self.cfg)
        self.image = image

    def _record(self, **overrides) -> dict:
        result = self.analysis.result
        total_cm2, consumed_cm2, percent = to_cm2(result, self.cfg, 60.0)
        diagnostics = imageops.threshold_diagnostics(
            self.image, self.analysis.threshold)
        record = {
            "file": "synthetic.png",
            "ok": True,
            "stage": "done",
            "threshold": self.analysis.threshold,
            "otsu_threshold": self.analysis.otsu_threshold,
            "threshold_metrics": {
                "separation": diagnostics["separation"],
                "bimodality": diagnostics["bimodality"],
                "near_threshold_fraction": diagnostics["near_threshold_fraction"],
            },
            "histogram": np.asarray(self.analysis.histogram).tolist(),
            "otsu_separation": self.analysis.otsu_separation,
            "scale": {"found": True, "method": "quad", "quality": 0.95,
                      "points": [], "scale_px_length": 60.0, "flags": [],
                      "details": {}},
            "measure": measure_dict(self.analysis),
            "stability": {"deltas": [-0.02, 0.02],
                          "total_cm2": [total_cm2, total_cm2],
                          "rel_range": 0.0},
            "total_cm2": total_cm2,
            "consumed_cm2": consumed_cm2,
            "percent_consumed": percent,
            "relative_only": False,
            "flags": [],
        }
        record.update(overrides)
        return record

    def test_real_measurement_is_auto(self) -> None:
        out = quality.evaluate(self._record(), self.cfg)
        self.assertEqual(out["status"], "auto")
        self.assertGreaterEqual(out["confidence"], self.cfg.auto_accept_min)

    def test_real_measurement_routes_to_review_under_stricter_cut(self) -> None:
        record = self._record()
        quality.evaluate(record, self.cfg)
        strict = Config()
        strict.auto_accept_min = 0.96
        strict.review_min = 0.45
        out = quality.evaluate(copy.deepcopy(record), strict)
        self.assertEqual(out["status"], "review")

    def test_real_measurement_without_scale_is_review(self) -> None:
        out = quality.evaluate(self._record(
            scale={"found": False, "method": "none", "quality": 0.0,
                   "points": [], "scale_px_length": None, "flags": [],
                   "details": {}},
            total_cm2=None, consumed_cm2=None, relative_only=True,
        ), self.cfg)
        self.assertEqual(out["status"], "review")
        self.assertIn("relative_only", out["flags"])


if __name__ == "__main__":
    unittest.main()
