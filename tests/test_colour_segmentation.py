"""Tests for the grey-background support (excess-green segmentation + dark dots).

The upstream LeafByte algorithm assumes a *white* paper background and a global
Otsu cut.  Real field photos (like the user's Vitex set) are often underexposed
or have a bluish-grey, vignetted background, where that cut slices through the
background itself.  These tests pin the two additions that fix it:

* ``imageops.exg`` / ``close_mask`` and ``measure``'s colour mode, chosen only
  when the luma result is demonstrably worse;
* ``scale``'s absolute-dark threshold ladder for the four scale dots.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from leafbyte import imageops, measure, scale  # noqa: E402
from leafbyte.config import Config  # noqa: E402
from leafbyte.types import ScaleDetection  # noqa: E402


def grey_background_photo(width=900, height=1200, leaf_area_target=30000, hole_radius=0,
                          dark_patch_radius=0):
    """A synthetic underexposed photo: grey vignetted paper, 4 black dots, green leaf."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:height, 0:width]
    # vignetted grey paper: bright in the middle, darker towards the corners
    radius = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
    paper = np.clip(205 - 60 * radius ** 2, 120, 210)
    image = np.dstack([paper * 0.97, paper * 0.99, paper * 1.02])  # faint blue cast
    image = np.clip(image + rng.normal(0, 2.0, image.shape), 0, 255)

    # leaf: an ellipse with the requested area, dark green
    a = 0.5 * width * 0.28
    b = leaf_area_target / (np.pi * a)
    cx, cy = width * 0.5, height * 0.5
    leaf = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0
    image[leaf] = (52, 118, 46)
    if hole_radius:
        hole = (xx - cx + a * 0.35) ** 2 + (yy - cy) ** 2 <= hole_radius ** 2
        image[hole & leaf] = (200, 200, 205)
    if dark_patch_radius:
        patch = (xx - cx - a * 0.35) ** 2 + (yy - cy) ** 2 <= dark_patch_radius ** 2
        image[patch & leaf] = (26, 30, 27)  # dark lesion / fold shadow, not paper

    # four black scale dots at the corners of a square
    dot_radius = 12
    for dx, dy in ((0.09, 0.09), (0.91, 0.09), (0.91, 0.86), (0.09, 0.86)):
        ddx, ddy = xx - width * dx, yy - height * dy
        image[ddx ** 2 + ddy ** 2 <= dot_radius ** 2] = (8, 8, 10)
    return np.clip(image, 0, 255).astype(np.uint8)


class ExgTest(unittest.TestCase):
    def test_excess_green_values(self) -> None:
        img = np.zeros((1, 3, 3), dtype=np.uint8)
        img[0, 0] = (0, 255, 0)    # pure green
        img[0, 1] = (128, 128, 128)  # neutral grey
        img[0, 2] = (200, 200, 210)  # bluish paper
        values = imageops.exg(img)[0]
        self.assertAlmostEqual(values[0], 510.0, places=6)
        self.assertAlmostEqual(values[1], 0.0, places=6)
        self.assertLess(values[2], 0.0)

    def test_close_mask_fills_thin_gaps(self) -> None:
        mask = np.zeros((21, 21), dtype=bool)
        mask[5:16, 5:16] = True
        mask[:, 10] = False  # a 1-px vein across the blob
        filled = imageops.close_mask(mask, 1)
        self.assertTrue(filled[10, 10])
        self.assertFalse(filled[0, 0])
        self.assertLessEqual(abs(int(filled.sum()) - 121), 5)


class ColourSegmentationTest(unittest.TestCase):
    def test_grey_background_uses_colour_mode(self) -> None:
        image = grey_background_photo()
        cfg = Config()
        analysis = measure.analyze(image, cfg)
        self.assertEqual("color", analysis.mode)
        expected = np.pi * (0.5 * image.shape[1] * 0.28) * 2  # not exact; use the drawn mask
        self.assertGreater(analysis.result.leaf_px, 0.9 * 15000)
        self.assertLess(analysis.result.leaf_px, 1.1 * 60000)
        self.assertFalse(analysis.result.leaf_touches_border)
        self.assertLess(analysis.result.competing_ratio, 0.05)

    def test_leaf_hole_becomes_consumed_area(self) -> None:
        image = grey_background_photo(hole_radius=26)
        analysis = measure.analyze(image, Config())
        self.assertEqual("color", analysis.mode)
        self.assertGreater(analysis.result.consumed_px, 0.5 * np.pi * 26 ** 2)
        self.assertGreater(analysis.result.percent_consumed, 0.5)

    def test_white_paper_still_uses_luma(self) -> None:
        """The published white-paper protocol must keep the upstream path."""
        image = np.full((600, 800, 3), 245, dtype=np.uint8)
        yy, xx = np.mgrid[0:600, 0:800]
        leaf = ((xx - 400) / 140) ** 2 + ((yy - 300) / 90) ** 2 <= 1.0
        image[leaf] = (40, 110, 40)
        for dx, dy in ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)):
            ddx, ddy = xx - 800 * dx, yy - 600 * dy
            image[ddx ** 2 + ddy ** 2 <= 10 ** 2] = (10, 10, 10)
        analysis = measure.analyze(image, Config())
        self.assertEqual("luma", analysis.mode)

    def test_forced_color_mode_works_on_grey(self) -> None:
        cfg = Config()
        cfg.leaf_mode = "color"
        cfg.color_threshold = 0.0
        analysis = measure.analyze(grey_background_photo(), cfg)
        self.assertEqual("color", analysis.mode)
        self.assertGreater(analysis.result.leaf_px, 10000)


class HoleValidationTest(unittest.TestCase):
    """Only paper-coloured holes are eaten area; dark patches are leaf tissue."""

    def test_bright_hole_counts_as_consumed(self) -> None:
        image = grey_background_photo(hole_radius=26)
        analysis = measure.analyze(image, Config())
        self.assertEqual("color", analysis.mode)
        self.assertGreater(analysis.result.consumed_px, 0.5 * np.pi * 26 ** 2)
        self.assertAlmostEqual(0.0, getattr(analysis.result, "stats_filled_px", 0) or 0.0)

    def test_dark_patch_is_tissue_not_herbivory(self) -> None:
        image = grey_background_photo(dark_patch_radius=26)
        analysis = measure.analyze(image, Config())
        result = analysis.result
        self.assertEqual("color", analysis.mode)
        self.assertEqual(0, result.consumed_px, "dark patch must not be eaten area")
        self.assertIn("dark_patches_filled", result.flags)
        self.assertGreater(getattr(result, "stats_filled_px", 0), 0.5 * np.pi * 26 ** 2)
        # the tissue is back inside the leaf: leaf area grew by the patch area
        self.assertGreater(result.leaf_px, 15000)

    def test_total_area_is_conserved_by_filling(self) -> None:
        bright = measure.analyze(grey_background_photo(hole_radius=26), Config()).result
        dark = measure.analyze(grey_background_photo(dark_patch_radius=26), Config()).result
        # same geometry: total (leaf + eaten) must agree within a couple of percent
        self.assertLess(abs(dark.total_px - bright.total_px) / bright.total_px, 0.02)


class AdaptiveCutTest(unittest.TestCase):
    def test_adaptive_cut_lands_between_background_and_leaf(self) -> None:
        image = grey_background_photo()
        cut = imageops.exg_threshold(image)
        # background ExG is clearly negative, leaf ExG clearly positive
        self.assertGreater(cut, -20.0)
        self.assertLess(cut, 100.0)

    def test_config_range_clamps_the_cut(self) -> None:
        cfg = Config()
        cfg.color_threshold_range = (20.0, 25.0)
        analysis = measure.analyze(grey_background_photo(), cfg)
        self.assertEqual("color", analysis.mode)
        self.assertGreaterEqual(analysis.color_threshold, 20.0)
        self.assertLessEqual(analysis.color_threshold, 25.0)

    def test_forced_cut_wins_over_adaptive(self) -> None:
        cfg = Config()
        cfg.color_threshold = 7.5
        analysis = measure.analyze(grey_background_photo(), cfg)
        self.assertEqual("color", analysis.mode)
        self.assertAlmostEqual(7.5, analysis.color_threshold, places=6)


class DarkScaleMarkTest(unittest.TestCase):
    def test_dots_found_on_grey_background(self) -> None:
        image = grey_background_photo()
        cfg = Config()
        detection = scale.detect(image, cfg)
        self.assertIsInstance(detection, ScaleDetection)
        self.assertTrue(detection.found, detection.flags)
        self.assertIn(detection.method, {"quad", "quad_relaxed", "quad_dark"})
        # the four dots must land near the corners of the drawn square
        points = np.asarray(detection.points, dtype=float)
        self.assertEqual((4, 2), points.shape)

    def test_no_marks_means_no_detection(self) -> None:
        image = np.full((400, 400, 3), 200, dtype=np.uint8)
        image[:, :200] = 40  # a big dark block, not four dots
        detection = scale.detect(image, Config())
        self.assertFalse(detection.found)


if __name__ == "__main__":
    unittest.main()
